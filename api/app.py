from __future__ import annotations

import os
import asyncio
from typing import Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "data" / "cgm_591.csv"
MODEL_PATH = BASE_DIR / "model" / "lstm_model.h5"
SCALER_PATH = BASE_DIR / "model" / "scaler.pkl"
CONFIG_PATH = BASE_DIR / "model" / "config.json"

STREAM_STEP_SECONDS = 1 # second

# =========================
# App
# =========================
app = FastAPI(title="Fake Nightscout API + Prediction", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Load data
# =========================
DF = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
DF = DF.sort_values("timestamp").reset_index(drop=True)

REPLAY_INDEX = 0

# =========================
# Load ML
# =========================
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

with open(CONFIG_PATH, "r") as f:
    LOOKBACK = int(pd.read_json(f, typ="series")["lookback"])

# =========================
# Helpers
# =========================
def current_df():
    return DF.iloc[:REPLAY_INDEX].copy()

def nightscout_entry(row):
    return {
        "_id": f"fake_{int(row['date'])}",
        "type": "sgv",
        "date": int(row["date"]),
        "dateString": str(row["dateString"]),
        "sgv": int(row["sgv"]),
        "direction": "Flat",
        "device": "FakeCGM",
    }

# =========================
# Replay
# =========================
async def replay_task():
    global REPLAY_INDEX
    while True:
        await asyncio.sleep(STREAM_STEP_SECONDS)
        if REPLAY_INDEX < len(DF):
            REPLAY_INDEX += 1

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(replay_task())


# =========================
# Endpoints
# =========================
@app.get("/")
def root():
    return {"status": "running"}

@app.get("/api/v1/predict")
def predict():
    df = current_df()

    if len(df) < LOOKBACK:
        raise HTTPException(400, f"Need {LOOKBACK}, only have {len(df)}")

    values = df.tail(LOOKBACK)["sgv"].astype(float).values.reshape(-1, 1)
    values_scaled = scaler.transform(values).reshape(1, LOOKBACK, 1)

    y_scaled = model.predict(values_scaled, verbose=0)
    y = scaler.inverse_transform(y_scaled)

    return {"prediction": float(y[0][0])}
@app.get("/api/v1/compare")
def compare():
    df = current_df()

    if len(df) < LOOKBACK:
        return {"detail": f"Need {LOOKBACK}, only have {len(df)}"}

    # آخر قراءة حقيقية
    last_real = float(df["sgv"].iloc[-1])

    # التوقع
    values = df.tail(LOOKBACK)["sgv"].astype(float).values.reshape(-1, 1)
    values_scaled = scaler.transform(values).reshape(1, LOOKBACK, 1)
    y_scaled = model.predict(values_scaled, verbose=0)
    prediction = float(scaler.inverse_transform(y_scaled)[0][0])

    return {
        "real": last_real,
        "predicted": prediction,
        "error": abs(last_real - prediction)
    }

