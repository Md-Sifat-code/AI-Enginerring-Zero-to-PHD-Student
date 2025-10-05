from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# ----------------------------
# Load model
# ----------------------------
app = FastAPI(title="Cashback Return-Rate Predictor")
model = joblib.load("cashback_lr_pipeline.pkl")

# ----------------------------
# Define payload
# ----------------------------
class Payload(BaseModel):
    cashback_bdt: float
    bottle_size_ml: int
    plastic_grade: int
    market_recycle_price_bdt_per_kg: float
    user_density_per_km2: int
    awareness_index: float
    distance_to_collection_km: float
    rain_mm: float
    campaign: int
    region: str
    bottles_sold: int = 10000  # optional, for ROI calculation

# ----------------------------
# Prediction endpoint
# ----------------------------
@app.post("/predict")
def predict(p: Payload):
    try:
        # Prepare input dictionary
        X_dict = {
            "cashback_bdt": [float(p.cashback_bdt)],
            "bottle_size_ml": [int(p.bottle_size_ml)],
            "plastic_grade": [int(p.plastic_grade)],
            "market_recycle_price_bdt_per_kg": [float(p.market_recycle_price_bdt_per_kg)],
            "user_density_per_km2": [int(p.user_density_per_km2)],
            "awareness_index": [float(p.awareness_index)],
            "distance_to_collection_km": [float(p.distance_to_collection_km)],
            "rain_mm": [float(p.rain_mm)],
            "campaign": [int(p.campaign)],
            "region": [str(p.region)]
        }

        # Convert to DataFrame
        X_df = pd.DataFrame(X_dict)

        # Make prediction
        y = float(model.predict(X_df)[0])
        y = max(0.0, min(100.0, y))  # clip between 0-100%

        # Calculate expected returned bottles
        returned = int(round(p.bottles_sold * y / 100.0))

        return {
            "return_rate_percent": y,
            "expected_returns": returned
        }

    except Exception as e:
        # Catch all errors and return as JSON
        return {"error": str(e)}

# ----------------------------
# Optional root endpoint
# ----------------------------
@app.get("/")
def root():
    return {"message": "Cashback Return-Rate Predictor API is running!"}
