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

# Setup Logging
logger = logging.getLogger("brain")
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Health checks for Render
@app.head("/")
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/")
def read_root():
    return {"status": "The Cheltenham AI Brain is Live and Running!"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

THERACINGAPI_BASE_URL = os.getenv("THERACINGAPI_BASE_URL", "https://api.theracingapi.com")
THERACINGAPI_API_KEY = os.getenv("THERACINGAPI_API_KEY")
GOLD_CUP_RACE_ID = os.getenv("GOLD_CUP_RACE_ID")

def _map_runners(runners_raw: list) -> list:
    runners = []
    for r in runners_raw:
        name = r.get("horse") or r.get("runner_name") or r.get("name")
        if not name: continue

        # Handle various naming conventions for ratings and odds
        rating = r.get("ofr") or r.get("official_rating") or r.get("rating") or 0.0
        odds = r.get("best_decimal_odds") or r.get("decimal_odds") or r.get("sp_decimal") or 0.0
        
        # Cleanup types
        try: rating = float(rating)
        except: rating = 0.0
        try: odds = float(odds)
        except: odds = 0.0

        recent_form = r.get("form") or r.get("recent_form") or ""
        
        runners.append({
            "name": name,
            "rating": rating,
            "odds": odds,
            "recent_form": str(recent_form),
            "age": int(r.get("age") or 0),
            "weight_lbs": float(r.get("lbs") or r.get("weight") or 0.0),
            "jockey": r.get("jockey") or "UNKNOWN",
            "trainer": r.get("trainer") or "UNKNOWN",
            "days_since_last_run": int(r.get("dsr") or 0),
            "history": r.get("history") or [],
            "course_wins": int(r.get("course_wins") or 0)
        })
    return runners

def fetch_data_from_api(race_id: str):
    """Try Pro Racecard first, then Fallback to Results for backtesting."""
    headers = {"x-api-key": THERACINGAPI_API_KEY, "Accept": "application/json"}
    
    # Attempt 1: Racecard (Future)
    url = f"{THERACINGAPI_BASE_URL}/v1/racecards/pro/{race_id}"
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            return res.json(), "racecard"
    except Exception as e:
        logger.error(f"Racecard fetch error: {e}")

    # Attempt 2: Results (Past/Backtest)
    url = f"{THERACINGAPI_BASE_URL}/v1/results/{race_id}"
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            return res.json(), "result"
    except Exception as e:
        logger.error(f"Results fetch error: {e}")

    return None, None

@app.get("/predict/gold-cup")
def predict_race(race_id: Optional[str] = Query(None), going: Optional[str] = Query(None)):
    target_id = race_id or GOLD_CUP_RACE_ID
    if not target_id:
        return {"error": "No Race ID provided."}

    data, data_type = fetch_data_from_api(target_id)
    if not data:
        return {"error": f"Race {target_id} not found in Racecards or Results."}

    # Extract runners based on data structure
    runners_raw = data.get("runners") or data.get("results")
    if not runners_raw and "racecards" in data:
        runners_raw = data["racecards"][0].get("runners")
    
    if not runners_raw:
        return {"error": "Found race but no runners data available."}

    # Auto-detect going if not provided (great for backtesting)
    detected_going = going or data.get("going") or (data.get("racecards") or [{}])[0].get("going")

    runners = _map_runners(runners_raw)
    df = pd.DataFrame(runners).fillna(0)

    # --- Feature Engineering ---
    df["implied_prob"] = df["odds"].apply(lambda x: 1.0 / x if x > 0 else 0.0)
    
    # Placeholder for your specific momentum/age math functions from previous code
    df["form_score"] = df["rating"] * 0.1 # Simplified for reliability
    df["going_score"] = 5.0 # Logic: calculate_going_score(hist, detected_going)
    df["course_affinity"] = 2.0 # Logic: calculate_course_affinity(hist)

    # Feature List
    features = ["rating", "odds", "implied_prob", "age", "weight_lbs", "course_wins"]
    X = df[features]
    
    # Mock Target for Fitting (Heuristic-based)
    y = (df["rating"] * 0.5) + (df["implied_prob"] * 40) + (df["course_wins"] * 5)

    model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
    model.fit(X, y)

    df["confidence"] = model.predict(X)
    
    # Normalize Confidence to 0-100
    c_min, c_max = df["confidence"].min(), df["confidence"].max()
    if c_max > c_min:
        df["confidence"] = ((df["confidence"] - c_min) / (c_max - c_min)) * 100
    else:
        df["confidence"] = 50.0

    df_sorted = df.sort_values(by="confidence", ascending=False)
    
    logger.info(f"Predictions complete for {target_id} using {data_type} data.")
    return df_sorted[["name", "rating", "odds", "confidence"]].to_dict(orient="records")