from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import logging
import os
import re
import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
# Setup Professional Logging
logger = logging.getLogger("brain")
logging.basicConfig(level=logging.INFO)

# 1. Initialize the FastAPI Application
app = FastAPI(title="Cheltenham Gold Cup AI Brain", version="2.0.0")

# Render-specific Health Checks & Root Endpoints
@app.head("/")
def head_root():
    return {"status": "ok"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "The Cheltenham AI Brain is Live",
        "endpoints": ["/predict/gold-cup", "/health"]
    }

# Enable CORS for Lovable and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from Environment Variables
THERACINGAPI_BASE_URL = os.getenv("THERACINGAPI_BASE_URL", "https://api.theracingapi.com")
THERACINGAPI_API_KEY = os.getenv("THERACINGAPI_API_KEY")
GOLD_CUP_RACE_ID = os.getenv("GOLD_CUP_RACE_ID")

# --- ADVANCED MATHEMATICAL HEURISTICS ---

def _compute_form_momentum(form_string: str) -> float:
    """
    Analyzes the 'recent_form' string (e.g., '1-2-F-1').
    Applies an exponential decay weight where most recent runs matter more.
    Penalizes non-finishes (F, PU, UR).
    """
    if not form_string: return 0.0
    # Split by common separators and clean
    tokens = re.split(r"[-\s/]+", str(form_string).upper().strip())
    tokens = [t for t in tokens if t]
    if not tokens: return 0.0
    
    score = 0.0
    max_runs = 6
    recent_tokens = tokens[-max_runs:]
    n = len(recent_tokens)
    
    for idx, token in enumerate(recent_tokens):
        # Weighting: 0.5 for oldest run, up to 1.0 for the most recent
        weight = 0.5 + (idx + 1) * (0.5 / max(n, 1))
        
        base = 0.0
        if token == "1": base = 4.5
        elif token == "2": base = 3.0
        elif token == "3": base = 2.0
        elif token == "4": base = 1.0
        elif token in {"F", "UR", "PU", "BD", "RR"}: base = -5.0
        
        score += (base * weight)
    return float(score)

def _compute_age_curve(age: float) -> float:
    """
    Age optimization for 3m+ Steeplechases like the Gold Cup.
    Prime years: 7, 8, 9.
    """
    if not age or age <= 0: return 0.0
    if 7 <= age <= 9: return 2.0
    if age <= 6: return 0.8
    # Penalize horses 10+ based on intensity of regression
    return -1.5 - (0.5 * (age - 9))

def calculate_going_score(history: List[Dict[str, Any]], current_going: Optional[str]) -> float:
    """
    Cross-references horse history against the current track condition.
    """
    if not history or not current_going: return 0.0
    target = str(current_going).upper().strip()
    score = 0.0
    for race in history:
        if not isinstance(race, dict): continue
        h_going = str(race.get("going") or "").upper()
        if target in h_going:
            pos = str(race.get("position") or "")
            if pos == "1": score += 4.0
            elif pos == "2": score += 2.5
            elif pos == "3": score += 1.5
    return score

def calculate_course_affinity(history: List[Dict[str, Any]]) -> float:
    """
    The 'Cheltenham Factor'. Looks for past success at the course.
    """
    if not history: return 0.0
    boost = 0.0
    for race in history:
        if not isinstance(race, dict): continue
        course = str(race.get("course") or "").upper()
        if "CHELTENHAM" in course:
            pos = str(race.get("position") or "")
            if pos == "1": boost += 12.0
            elif pos in ["2", "3"]: boost += 5.0
    return boost

# --- DATA PROCESSING & API INTERACTION ---

def _map_runners(runners_raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Comprehensive runner mapping for the AI Model."""
    mapped = []
    for r in runners_raw:
        # Resolve horse name across API variations
        name = r.get("horse") or r.get("runner_name") or r.get("name") or r.get("horse_name")
        if not name: continue

        # Official Rating (OFR) Resolution
        ofr = r.get("ofr") or r.get("official_rating") or r.get("rating") or r.get("hra_rating")
        try: ofr = float(ofr) if ofr not in (None, "", "-") else 140.0
        except: ofr = 140.0

        # Form Score (Fallback to manual string analysis if missing)
        recent_form = r.get("form") or r.get("recent_form") or r.get("form_string") or ""
        form_score = r.get("form_score")
        if form_score is None:
            # Manual basic form calc
            s = 0
            for ch in str(recent_form)[:5]:
                if ch == "1": s += 4
                elif ch == "2": s += 2
                elif ch in ["3", "4"]: s += 1
            form_score = s

        # Price/Odds Resolution
        odds = r.get("best_decimal_odds") or r.get("decimal_odds") or r.get("sp_decimal")
        try: odds = float(odds) if odds not in (None, "", "-") else 12.0
        except: odds = 12.0

        mapped.append({
            "name": name,
            "rating": float(ofr),
            "form_score": float(form_score),
            "course_wins": int(r.get("course_wins") or r.get("cd_wins") or 0),
            "odds": float(odds),
            "recent_form": str(recent_form),
            "age": int(r.get("age") or r.get("horse_age") or 7),
            "weight_lbs": float(r.get("lbs") or r.get("weight") or r.get("weight_lbs") or 160.0),
            "jockey": str(r.get("jockey") or r.get("jockey_name") or "UNKNOWN"),
            "trainer": str(r.get("trainer") or r.get("trainer_name") or "UNKNOWN"),
            "days_since_last_run": int(r.get("dsr") or r.get("last_run") or r.get("days_since_last_run") or 30),
            "history": r.get("history") or [],
        })
    return mapped

def fetch_runners_for_race(race_id: Optional[str]) -> tuple[list, Optional[str]]:
    # Use the Key from your Render Environment
    if not THERACINGAPI_API_KEY:
        return [], "ERROR: API Key is missing from Render Environment Variables."

    # This is how the 'Username/Password' is bypassed
    url = f"{THERACINGAPI_BASE_URL}/v1/racecards/pro/{race_id}"
    headers = {"x-api-key": THERACINGAPI_API_KEY, "Accept": "application/json"}

    try:
        response = requests.get(url, headers=headers, timeout=15)
        # If this says 401 or 403, your API key is wrong
        # If this says 404, the Race ID hasn't been generated yet
        response.raise_for_status() 
        data = response.json()
        
        # Drill down into the data
        runners_raw = data.get("runners") or data.get("racecards", [{}])[0].get("runners", [])
        return _map_runners(runners_raw), None
        
    except requests.exceptions.HTTPError as e:
        return [], f"API Error: {e.response.status_code} - Check your Key and ID."

# --- MAIN AI PREDICTION ENGINE ---

@app.get("/predict/gold-cup")
def predict_gold_cup(
    race_id: Optional[str] = Query(None),
    going: Optional[str] = Query(None)
):
    """The High-Precision Analysis Route."""
    target_id = race_id or GOLD_CUP_RACE_ID
    if not target_id:
        return {"error": "Missing Race ID. Set in Render or use ?race_id="}

    runners, err = fetch_live_data(target_id)
    if err: return {"error": err}

    try:
        # Load into Pandas for feature engineering
        df = pd.DataFrame(runners).fillna(0)
        
        # 1. Advanced Feature Calculations
        df["implied_prob"] = df["odds"].apply(lambda x: 1.0 / x if x > 0 else 0.0)
        df["momentum_score"] = df["recent_form"].apply(_compute_form_momentum)
        df["age_score"] = df["age"].apply(_compute_age_curve)
        df["going_suitability"] = df["history"].apply(lambda h: calculate_going_score(h, going))
        df["cheltenham_affinity"] = df["history"].apply(calculate_course_affinity)

        # 2. Categorical Encoding (Jockey & Trainer)
        le_j = LabelEncoder()
        le_t = LabelEncoder()
        df["j_encoded"] = le_j.fit_transform(df["jockey"].astype(str))
        df["t_encoded"] = le_t.fit_transform(df["trainer"].astype(str))

        # 3. Model Feature Selection
        features = [
            "rating", "form_score", "course_wins", "implied_prob",
            "momentum_score", "age_score", "going_suitability",
            "cheltenham_affinity", "weight_lbs", "days_since_last_run",
            "j_encoded", "t_encoded"
        ]
        X = df[features]
        
        # 4. Target Heuristic (Wait for Training)
        # In this AI brain, we weight 'cheltenham_affinity' and 'momentum' heavily
        y = (
            (df["rating"] * 0.35) + 
            (df["form_score"] * 1.5) + 
            (df["momentum_score"] * 3.0) + 
            (df["implied_prob"] * 55.0) + 
            (df["cheltenham_affinity"] * 10.0) +
            (df["going_suitability"] * 7.0) +
            (df["age_score"] * 5.0)
        )

        # 5. Execute Histogram Gradient Boosting
        model = HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.07,
            max_depth=5,
            random_state=42
        )
        model.fit(X, y)

        # 6