from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from pydantic import BaseModel
from dotenv import load_dotenv

from llm_logic import classify_glucose_risk, generate_llm_response

# =========================
# Env
# =========================
load_dotenv()

# =========================
# Paths (match your structure)
# =========================
BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / "Data" / "cgm_591.csv"

MODEL_DIR = BASE_DIR / "Model"
WEIGHTS_PATH = MODEL_DIR / "lstm_model.keras" / "model.weights.h5"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
CONFIG_PATH = MODEL_DIR / "config.json"

# Sampling interval
STEP_MINUTES = 5
HOUR_STEPS = 12  # 60 minutes / 5 minutes

# =========================
# App
# =========================
app = FastAPI(title="Blood Glucose Forecasting", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for local dev
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static + Templates
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# =========================
# Load Data
# =========================
if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

DF = pd.read_csv(CSV_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

required_cols = {"timestamp", "sgv"}
missing = required_cols - set(DF.columns)
if missing:
    raise ValueError(f"CSV missing columns {missing}. It must contain {required_cols}")

# =========================
# Load Scaler + Config
# =========================
if not SCALER_PATH.exists():
    raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")
scaler = joblib.load(SCALER_PATH)

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
cfg = pd.read_json(CONFIG_PATH, typ="series")

if "lookback" not in cfg:
    raise ValueError("config.json must contain 'lookback' (e.g., {\"lookback\": 12})")

LOOKBACK = int(cfg["lookback"])

# Optional hyperparams (if not found, defaults are used)
LSTM_UNITS = int(cfg.get("lstm_units", 32))
DENSE_UNITS = int(cfg.get("dense_units", 16))

# =========================
# Build Model (Architecture) then load weights
# =========================
def build_lstm_model(lookback: int, units: int = 32, dense_units: int = 16) -> tf.keras.Model:
    inp = tf.keras.layers.Input(shape=(lookback, 1), name="input_window")
    x = tf.keras.layers.LSTM(units, name="lstm")(inp)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="dense_relu")(x)
    out = tf.keras.layers.Dense(1, name="output")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="LSTM_Forecaster")
    model.compile(optimizer="adam", loss="mse", metrics=[tf.keras.metrics.MAE])
    return model

if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")

model = build_lstm_model(LOOKBACK, units=LSTM_UNITS, dense_units=DENSE_UNITS)

# IMPORTANT: Must build the model once before loading weights
_ = model.predict(np.zeros((1, LOOKBACK, 1), dtype=np.float32), verbose=0)

try:
    model.load_weights(str(WEIGHTS_PATH))
except Exception as e:
    raise RuntimeError(
        "Weights failed to load. This means the model architecture in app.py "
        "does not match the saved weights.\n"
        "Fix by updating lstm_units / dense_units in config.json to match training.\n\n"
        f"Original error:\n{e}"
    )

# =========================
# Helpers
# =========================
def make_future_times(last_ts: pd.Timestamp, steps: int = HOUR_STEPS, step_minutes: int = STEP_MINUTES):
    return pd.date_range(
        start=last_ts + pd.Timedelta(minutes=step_minutes),
        periods=steps,
        freq=f"{step_minutes}min",
    )

def recursive_forecast_1hour(last_values: np.ndarray, steps: int = HOUR_STEPS) -> List[float]:
    """
    last_values: (LOOKBACK, 1) real scale
    returns: list of length steps
    """
    window = last_values.astype(float).reshape(-1, 1)
    preds: List[float] = []

    for _ in range(steps):
        x_scaled = scaler.transform(window).reshape(1, LOOKBACK, 1)
        y_scaled = model.predict(x_scaled, verbose=0)

        # one-step output
        y_real = float(scaler.inverse_transform(y_scaled)[0][0])
        preds.append(y_real)

        window = np.vstack([window[1:], [[y_real]]])

    return preds

def compute_trend(history_values: List[float]) -> str:
    if len(history_values) < 2:
        return "stable"
    delta = history_values[-1] - history_values[0]
    if delta > 10:
        return "rising"
    if delta < -10:
        return "falling"
    return "stable"

# =========================
# Schemas
# =========================
class RecommendRequest(BaseModel):
    predicted_glucose: float
    trend: str = "stable"

# =========================
# Pages
# =========================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# =========================
# API
# =========================
@app.get("/api/v1/health")
def health():
    return {
        "status": "ok",
        "lookback": LOOKBACK,
        "rows": int(len(DF)),
        "model": {"lstm_units": LSTM_UNITS, "dense_units": DENSE_UNITS},
    }

@app.get("/api/v1/forecast_hour")
def forecast_hour():
    if len(DF) < LOOKBACK:
        raise HTTPException(400, f"Need {LOOKBACK} points, only have {len(DF)}")

    # last 1 hour REAL (12 points)
    hist = DF.tail(HOUR_STEPS).copy()
    history_values = hist["sgv"].astype(float).tolist()
    history_times = hist["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist()

    # next 1 hour PREDICTED (12 points)
    last_window = DF.tail(LOOKBACK)["sgv"].astype(float).values.reshape(-1, 1)
    forecast_values = recursive_forecast_1hour(last_window, steps=HOUR_STEPS)

    last_ts = DF["timestamp"].iloc[-1]
    future_times = make_future_times(last_ts, steps=HOUR_STEPS, step_minutes=STEP_MINUTES)
    forecast_times = future_times.strftime("%Y-%m-%dT%H:%M:%S").tolist()

    after_1h = float(forecast_values[-1])
    summary = {
        "after_1h": after_1h,
        "max_1h": float(max(forecast_values)),
        "min_1h": float(min(forecast_values)),
        "avg_1h": float(sum(forecast_values) / len(forecast_values)),
        "trend": compute_trend(history_values),
    }

    return {
        "step_minutes": STEP_MINUTES,
        "lookback": LOOKBACK,
        "history": {"times": history_times, "values": history_values},
        "forecast": {"times": forecast_times, "values": forecast_values},
        "summary": summary,
    }

@app.post("/api/v1/recommendation")
def recommendation(req: RecommendRequest):
    risk = classify_glucose_risk(req.predicted_glucose)

    try:
        advice = generate_llm_response(req.predicted_glucose, risk, trend=req.trend)
    except Exception as e:
        # لا نطيّح السيرفر — رجّعي رسالة واضحة لواجهة المستخدم
        advice = f"LLM error: {type(e).__name__}: {str(e)[:200]}"

    return {"risk": risk, "advice": advice}

