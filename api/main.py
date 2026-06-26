# Fast API for prediction service for the WindGuard anomaly detection system
import os
import csv
import logging
from datetime import datetime
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Validation of the input
class SensorReading(BaseModel):
    type: str                = Field(default="L", pattern="^[LMHlmh]$")
    air_temperature: float   = Field(..., gt=280, lt=320)
    process_temperature: float = Field(..., gt=290, lt=330)
    rotational_speed: float  = Field(..., ge=0, lt=3000)
    torque: float            = Field(..., ge=0, lt=100)
    tool_wear: float         = Field(..., ge=0, lt=300)


class BatchRequest(BaseModel):
    readings: List[SensorReading]

# Load the trained model

app = FastAPI(title="WindGuard Anomaly Detection API", version="1.0")

model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_names = joblib.load("model/features.pkl")

THRESHOLD = 0.5  # Threshold for classifying a reading as a failure
LOG_PATH = "monitoring/logs/prediction.csv"

logging.basicConfig(level=logging.INFO)

# Ensure monitoring/logs directory exists
log_dir = os.path.join("monitoring", "logs")
if os.path.isfile(log_dir):
    os.remove(log_dir)  # Remove if it's a file instead of directory
os.makedirs(log_dir, exist_ok=True)


if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, mode='w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp","air_temp", "proc_temp", "rpm", "torque", "tool_wear", "failure_prob", "label"])

# Helper functions
def prepare_features(reading: SensorReading) -> list:
    type_map = {"l": 0, "m": 1, "h": 2}
    air_temp = reading.air_temperature
    proc_temp = reading.process_temperature
    rpm = reading.rotational_speed
    torque = reading.torque
    tool_wear = reading.tool_wear
   
    temp_diff = proc_temp - air_temp
    power_w = torque * (rpm * 2 * 3.14159 / 60)
    wear_torque = tool_wear * torque


    return [type_map.get(reading.type.lower(), 0), air_temp, proc_temp, rpm, torque, tool_wear, temp_diff, power_w, wear_torque]

def save_prediction(features: list, probability: float, label: str):
    with open(LOG_PATH, mode='a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),
            features[1],  # air_temp, 
            features[2],  # proc_temp
            features[3],  # rpm
            features[4],  # torque
            features[5],  # tool_wear
            round(probability, 4),  # failure_prob
            label
        ])


# Endpoints

@app.get("/health")
def health_check():
    return {"status": "ok", "model loaded": True}

@app.post("/predict")
def predict(reading: SensorReading):
    features = prepare_features(reading)
    scaled = scaler.transform([features])
    probability = model.predict_proba(scaled)[0][1]
    label = "Failure" if probability >= THRESHOLD else "No Failure"

    save_prediction(features, probability, label)

    return {
        "failure_probability": round(probability, 4),
        "label": label,
        "threshold": THRESHOLD
    }


@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    results = []
    for reading in request.readings:
        features = prepare_features(reading)
        scaled = scaler.transform([features])
        probability = float(model.predict_proba(scaled)[0][1])
        label = "Failure" if probability >= THRESHOLD else "No Failure"

        save_prediction(features, probability, label)

        results.append({
            "failure_probability": round(probability, 4),
            "label": label            
        })

    return {"predictions": results}