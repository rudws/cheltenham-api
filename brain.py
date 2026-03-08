from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import logging
import os
import re
import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 1. Initialize the API
app = FastAPI()

# Health check for Render: HEAD / and GET /health
@app.head("/")
def head_root():
    return {"status": "ok"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Hello message (GET /)
@app.get("/")
def read_root():
    return {"status": "The Cheltenham AI Brain is Live and Running!"}

# CORS: allow all origins for Lovable pre-flight (OPTIONS) and requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. The Data – fetched live from The Racing API
# Pro endpoint: GET /v1/racecards/pro/{race_id} returns { "runners": [...] } or { "racecards": [ { "runners": [...] } ] }
THERACINGAPI_BASE_URL = os.getenv("THERACINGAPI_BASE_URL", "https://api.theracingapi.com")
THERACINGAPI_API_KEY = os.getenv("THERACINGAPI_API_KEY")
GOLD_CUP_RACE_ID = os.getenv("GOLD_CUP_RACE_ID")  # e.g. "rac_11894545" – default when no ?race_id given


def _map_runners(runners_raw: list) -> list:
    """Map raw API runner objects to our standard runner dict (horse, ofr, form, lbs, etc.)."""
    runners = []
    for r in runners_raw:
        name = r.get("horse") or r.get("runner_name") or r.get("name")
        if not name:
            continue

        raw_rating = r.get("ofr") or r.get("official_rating") or r.get("rating") or r.get("hra_rating")
        try:
            rating = float(raw_rating) if raw_rating not in (None, "", "-") else 0.0
        except (TypeError, ValueError):
            rating = 0.0

        recent_form = r.get("form") or r.get("recent_form") or r.get("form_string") or ""
        form_score = r.get("form_score")
        if form_score is None:
            score = 0
            for ch in str(recent_form)[:5]:
                if ch == "1":
                    score += 3
                elif ch == "2":
                    score += 2
                elif ch == "3":
                    score += 1
            form_score = score

        course_wins = r.get("course_wins") or r.get("wins_at_course") or r.get("cd_wins") or 0

        odds = r.get("best_decimal_odds") or r.get("decimal_odds") or r.get("sp_decimal")
        try:
            odds = float(odds) if odds not in (None, "", "-") else 0.0
        except (TypeError, ValueError):
            odds = 0.0

        age = r.get("age") or r.get("horse_age") or 0
        try:
            age = int(age) if age not in (None, "", "-") else 0
        except (TypeError, ValueError):
            age = 0

        weight_lbs = r.get("lbs") or r.get("weight_lbs") or r.get("weight") or 0
        try:
            weight_lbs = float(weight_lbs) if weight_lbs not in (None, "", "-") else 0.0
        except (TypeError, ValueError):
            weight_lbs = 0.0

        jockey = r.get("jockey") or r.get("jockey_name") or ""
        trainer = r.get("trainer") or r.get("trainer_name") or ""

        last_run = r.get("last_run") or r.get("days_since_last_run") or r.get("days_last_run") or r.get("dsr") or 0
        try:
            days_since_last_run = int(last_run) if last_run not in (None, "", "-") else 0
        except (TypeError, ValueError):
            days_since_last_run = 0

        history = r.get("history") or []

        runners.append(
            {
                "name": name,
                "rating": float(rating),
                "form_score": float(form_score),
                "course_wins": int(course_wins) if course_wins is not None else 0,
                "odds": float(odds),
                "recent_form": str(recent_form),
                "age": int(age),
                "weight_lbs": float(weight_lbs),
                "jockey": jockey or "UNKNOWN",
                "trainer": trainer or "UNKNOWN",
                "days_since_last_run": int(days_since_last_run),
                "history": history,
            }
        )
    return runners


def fetch_runners_for_race(race_id: Optional[str]) -> tuple[list, Optional[str]]:
    """
    Fetch runners from The Racing API pro endpoint for the given race_id.
    Returns (runners_list, error_message). error_message is None on success.
    """
    if not THERACINGAPI_API_KEY:
        return [], "THERACINGAPI_API_KEY environment variable is not set."
    if not race_id or not str(race_id).strip():
        return [], "No race ID provided. Set ?race_id=rac_xxx or GOLD_CUP_RACE_ID env var."

    url = f"{THERACINGAPI_BASE_URL}/v1/racecards/pro/{race_id.strip()}"
    headers = {"x-api-key": THERACINGAPI_API_KEY, "Accept": "application/json"}

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        logger.warning("Racing API request failed for race_id=%s: %s", race_id, e)
        return [], "Race data not available yet. Try a live race ID from today."

    # Aggressive data nesting: try data.runners, then data.racecards[0].runners
    runners_raw = data.get("runners")
    if not runners_raw and data.get("racecards"):
        racecards = data.get("racecards") or []
        first_race = racecards[0] if racecards else {}
        runners_raw = first_race.get("runners") if isinstance(first_race, dict) else []
    runners_raw = runners_raw or []

    if not runners_raw:
        logger.info("No runners in API response for race_id=%s", race_id)
        return [], "No runners found for this race ID yet."

    runners = _map_runners(runners_raw)
    if not runners:
        logger.info("Mapped runners list empty for race_id=%s", race_id)
        return [], "No runners found for this race ID yet."

    return runners, None


def _compute_form_momentum(form_string: str) -> float:
    """
    Convert a form string like '1-2-F-1' into a weighted score.
    More recent runs (right-most) are weighted more heavily.
    Falls/PUs are penalised.
    """
    if not form_string:
        return 0.0

    tokens = re.split(r"[-\s/]+", str(form_string).upper().strip())
    tokens = [t for t in tokens if t]  # remove empty strings

    if not tokens:
        return 0.0

    # Process from oldest to newest, then apply increasing weights to newer runs
    score = 0.0
    max_runs = 6
    recent_tokens = tokens[-max_runs:]
    n = len(recent_tokens)
    for idx, token in enumerate(recent_tokens):
        # Weight grows towards the most recent run
        weight = 0.5 + (idx + 1) * (0.5 / max(n, 1))

        base = 0.0
        if token == "1":
            base = 4.0
        elif token == "2":
            base = 2.5
        elif token == "3":
            base = 1.5
        elif token == "4":
            base = 0.5
        elif token in {"F", "UR", "PU", "BD", "RR"}:
            base = -4.0
        # else leave base at 0 for other finishes

        score += base * weight

    return float(score)


def _compute_age_curve(age: float) -> float:
    """
    Gold Cup age curve:
    - Prime: 7-9 get a positive boost
    - Young (<=6): modest positive (unproven but improving)
    - Older (10+): penalised increasingly with age
    """
    if age is None or age <= 0:
        return 0.0

    if 7 <= age <= 9:
        return 1.5
    if age <= 6:
        return 0.5

    # 10+ penalised, scaled by years over 9
    return -1.0 - 0.2 * (age - 9)


def calculate_going_score(history, current_going: Optional[str]) -> float:
    """
    Calculate a going score based on the horse's history.

    For each past race where the going matches the current_going (substring match,
    e.g. 'SOFT' matches 'GOOD TO SOFT') and the position is 1st, 2nd or 3rd,
    we add:
    - 1st: +3 points
    - 2nd: +2 points
    - 3rd: +1 point
    """
    if not history or not current_going:
        return 0.0

    target = str(current_going).upper().strip()
    score = 0.0

    for race in history:
        if not isinstance(race, dict):
            continue

        going_val = race.get("going")
        if not going_val:
            continue

        going_str = str(going_val).upper()
        if target not in going_str:
            continue

        pos_val = race.get("position")
        try:
            pos = int(pos_val)
        except (TypeError, ValueError):
            continue

        if pos == 1:
            score += 3.0
        elif pos == 2:
            score += 2.0
        elif pos == 3:
            score += 1.0

    return score


def calculate_course_affinity(history) -> float:
    """
    Massive mathematical boost for proven Cheltenham winners.

    If the horse has any past race at 'Cheltenham' where position == 1,
    we add a large fixed bonus. Multiple Cheltenham wins stack.
    """
    if not history:
        return 0.0

    boost = 0.0
    for race in history:
        if not isinstance(race, dict):
            continue

        course = str(race.get("course") or "").upper()
        pos_val = race.get("position")
        try:
            pos = int(pos_val)
        except (TypeError, ValueError):
            continue

        if course == "CHELTENHAM" and pos == 1:
            boost += 10.0  # "Massive" boost per Cheltenham win

    return boost


# 3. The API Endpoint Lovable will call
@app.get("/predict/gold-cup")
def predict_race(
    race_id: Optional[str] = Query(
        None,
        description="Optional race ID (e.g. rac_11894545). If omitted, uses GOLD_CUP_RACE_ID env var.",
    ),
    going: Optional[str] = Query(
        None,
        description="Optional going description, e.g. 'Soft', 'Good to Soft'. If omitted, going-based features are neutral.",
    ),
):
    # Use the provided race_id OR fall back to the one in your environment variables
    target_id = (race_id and race_id.strip()) or GOLD_CUP_RACE_ID

    runners, fetch_error = fetch_runners_for_race(target_id)
    if fetch_error:
        return {"error": fetch_error}
    if not runners:
        return {"error": "No runners found for this race ID yet."}

    logger.info("predict_race target_id=%s runners_found=%d", target_id, len(runners))

    try:
        df = pd.DataFrame(runners)
        df = df.fillna(0)

        # Bulletproof implied_prob: avoid division by zero
        df["implied_prob"] = df["odds"].apply(lambda x: 1.0 / x if x and x > 0 else 0.0)
        df["form_momentum"] = df["recent_form"].apply(_compute_form_momentum)
        df["age_curve"] = df["age"].apply(_compute_age_curve)
        df["days_since_last_run"] = df["days_since_last_run"].fillna(0).astype(float)
        df["going_score"] = df["history"].apply(
            lambda hist: calculate_going_score(hist, going)
        )
        df["course_affinity"] = df["history"].apply(calculate_course_affinity)

        df["jockey"] = df["jockey"].fillna("UNKNOWN").astype(str)
        df["trainer"] = df["trainer"].fillna("UNKNOWN").astype(str)

        jockey_encoder = LabelEncoder()
        trainer_encoder = LabelEncoder()
        df["jockey_encoded"] = jockey_encoder.fit_transform(df["jockey"])
        df["trainer_encoded"] = trainer_encoder.fit_transform(df["trainer"])

        features = [
            "rating",
            "form_score",
            "course_wins",
            "implied_prob",
            "form_momentum",
            "age_curve",
            "going_score",
            "course_affinity",
            "weight_lbs",
            "days_since_last_run",
            "jockey_encoded",
            "trainer_encoded",
        ]
        X = df[features]

        jockey_rank = df["jockey_encoded"].rank(pct=True)
        trainer_rank = df["trainer_encoded"].rank(pct=True)
        recency_factor = 1.0 / (1.0 + (df["days_since_last_run"] / 30.0))

        y = (
            (df["rating"] * 0.30)
            + (df["form_score"] * 1.5)
            + (df["course_wins"] * 3.0)
            + (df["form_momentum"] * 2.5)
            + (df["age_curve"] * 6.0)
            + (df["implied_prob"] * 50.0)
            + (df["going_score"] * 6.0)
            + (df["course_affinity"] * 8.0)
            + (recency_factor * 5.0)
            + (jockey_rank * 4.0)
            + (trainer_rank * 4.0)
        )

        model = HistGradientBoostingRegressor(
            max_depth=4,
            learning_rate=0.05,
            max_iter=100,
            early_stopping=True,
            n_iter_no_change=10,
            validation_fraction=0.1,
            l2_regularization=0.0,
            random_state=42,
        )
        model.fit(X, y)

        df["ai_raw_score"] = model.predict(X)
        min_score = df["ai_raw_score"].min()
        max_score = df["ai_raw_score"].max()

        if max_score == min_score:
            df["confidence"] = 50.0
        else:
            df["confidence"] = ((df["ai_raw_score"] - min_score) / (max_score - min_score)) * 100

        df["confidence"] = df["confidence"].clip(lower=0, upper=100).round(1)
        df_sorted = df.sort_values(by="confidence", ascending=False)

        return df_sorted[["name", "rating", "odds", "confidence"]].to_dict(orient="records")

    except Exception as e:
        print(f"Error: {e}")
        return {"error": "Race data not available yet. Try a live race ID from today."}