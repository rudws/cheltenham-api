from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import os
import re
import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

# 1. Initialize the API
app = FastAPI()

# Hello message
@app.get("/")
def read_root():
    return {"status": "The Cheltenham AI Brain is Live and Running!"}

# Allow Lovable to talk to this API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your Lovable URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. The Data – fetched live from The Racing API
# Structure (from your 2026-03-08_racecards.json): response is { "racecards": [ { "race_id": "rac_xxxx", "course", "date", "race_name", "runners": [ { "horse", "ofr", "form", "lbs", "last_run", "age", "jockey", "trainer", ... } ] }, ... ] }
# GOLD_CUP_RACE_ID = the "race_id" of the Gold Cup race (e.g. rac_12345678). Get it from a racecards response for Cheltenham Friday 13 March 2026.
THERACINGAPI_BASE_URL = os.getenv("THERACINGAPI_BASE_URL", "https://api.theracingapi.com")
THERACINGAPI_API_KEY = os.getenv("THERACINGAPI_API_KEY")
GOLD_CUP_RACE_ID = os.getenv("GOLD_CUP_RACE_ID")  # e.g. "rac_11894545" – use the race_id for the Gold Cup from your API


def get_gold_cup_runners():
    """
    Fetch live runners for the Cheltenham Gold Cup from The Racing API.

    Expects API response: { "racecards": [ { "race_id", "runners": [ { "horse", "ofr", "form", "lbs", "last_run", "age", "jockey", "trainer", ... } ] } ] }
    """
    if not THERACINGAPI_API_KEY:
        raise HTTPException(status_code=500, detail="THERACINGAPI_API_KEY environment variable is not set.")
    if not GOLD_CUP_RACE_ID:
        raise HTTPException(status_code=500, detail="GOLD_CUP_RACE_ID environment variable is not set.")

    endpoint = f"{THERACINGAPI_BASE_URL}/v1/racecards/{GOLD_CUP_RACE_ID}"

    headers = {
        "x-api-key": THERACINGAPI_API_KEY,
        "Accept": "application/json",
    }

    params = {}
    if os.getenv("THERACINGAPI_INCLUDE_ODDS", "").lower() in ("1", "true", "yes"):
        params["include_odds"] = "true"
    if os.getenv("THERACINGAPI_INCLUDE_RATINGS", "").lower() in ("1", "true", "yes"):
        params["include_ratings"] = "true"

    try:
        response = requests.get(endpoint, headers=headers, params=params or None, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Error calling The Racing API: {exc}")

    data = response.json()

    # Response shape: { "racecards": [ { "race_id", "runners": [...] } ] } or single race
    racecards = data.get("racecards") or []
    if racecards:
        race = next((r for r in racecards if str(r.get("race_id")) == str(GOLD_CUP_RACE_ID)), racecards[0])
        runners_raw = race.get("runners") or []
    else:
        runners_raw = data.get("runners") or data.get("race_runners") or []

    if not runners_raw:
        raise HTTPException(
            status_code=500,
            detail="No runners in response. Check GOLD_CUP_RACE_ID and that the API returns racecards with a 'runners' array.",
        )

    runners = []
    for r in runners_raw:
        name = r.get("horse") or r.get("runner_name") or r.get("name")
        if not name:
            continue

        # Official rating (API uses "ofr"; can be "-" or missing)
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

        course_wins = (
            r.get("course_wins")
            or r.get("wins_at_course")
            or r.get("cd_wins")
            or 0
        )

        odds = (
            r.get("best_decimal_odds")
            or r.get("decimal_odds")
            or r.get("sp_decimal")
        )
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

    if not runners:
        raise HTTPException(status_code=500, detail="Mapped runners list is empty after processing The Racing API response.")

    return runners


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
    going: Optional[str] = Query(
        None,
        description="Optional going description for this race, e.g. 'Soft', 'Good to Soft'. If omitted, going-based features are neutral.",
    )
):
    runners = get_gold_cup_runners()
    df = pd.DataFrame(runners)

    # --- Advanced feature engineering ---

    # Implied probability from decimal odds
    df["implied_prob"] = df["odds"].apply(lambda x: 1.0 / x if x and x > 0 else 0.0)

    # Form momentum from recent form string
    df["form_momentum"] = df["recent_form"].apply(_compute_form_momentum)

    # Age curve feature
    df["age_curve"] = df["age"].apply(_compute_age_curve)

    # Days since last run (convert missing/zero to a neutral value)
    df["days_since_last_run"] = df["days_since_last_run"].fillna(0).astype(float)

    # Going score: performance on the user-selected going (if provided)
    df["going_score"] = df["history"].apply(
        lambda hist: calculate_going_score(hist, going)
    )

    # Course affinity: Cheltenham-winning history
    df["course_affinity"] = df["history"].apply(calculate_course_affinity)

    # Encode jockey and trainer categorically
    df["jockey"] = df["jockey"].fillna("UNKNOWN").astype(str)
    df["trainer"] = df["trainer"].fillna("UNKNOWN").astype(str)

    jockey_encoder = LabelEncoder()
    trainer_encoder = LabelEncoder()

    df["jockey_encoded"] = jockey_encoder.fit_transform(df["jockey"])
    df["trainer_encoded"] = trainer_encoder.fit_transform(df["trainer"])

    # --- Model features ---
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

    # --- Synthetic target: blend traditional factors with market and connections ---
    # Rating + form + course form as backbone, blended with:
    # - Form momentum
    # - Age curve
    # - Market view (implied probability)
    # - Recency (shorter layoff preferred)
    # - Jockey/trainer ranks

    # Normalised ranks for jockey/trainer (0-1, higher = stronger historically within this race)
    jockey_rank = df["jockey_encoded"].rank(pct=True)
    trainer_rank = df["trainer_encoded"].rank(pct=True)

    # Recency factor: 1 when running today after <=7 days, decays with longer layoffs
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
        max_iter=400,
        l2_regularization=0.0,
        random_state=42,
    )
    model.fit(X, y)

    df["ai_raw_score"] = model.predict(X)

    # Normalize to 0-100%
    min_score = df["ai_raw_score"].min()
    max_score = df["ai_raw_score"].max()

    if max_score == min_score:
        df["confidence"] = 50.0
    else:
        df["confidence"] = ((df["ai_raw_score"] - min_score) / (max_score - min_score)) * 100

    df["confidence"] = df["confidence"].clip(lower=0, upper=100).round(1)

    # Show full engineered feature matrix in the terminal for inspection
    print(df.head())

    df_sorted = df.sort_values(by="confidence", ascending=False)

    # Return the exact JSON Lovable needs
    return df_sorted[["name", "rating", "odds", "confidence"]].to_dict(orient="records")