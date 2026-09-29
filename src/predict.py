from __future__ import annotations
import argparse
import joblib
import pandas as pd
from config import MODELS_DIR

EXAMPLE = {
    "Area_sqft": 1800, "Bedrooms": 3, "Bathrooms": 2, "Floors": 2,
    "HouseAge_years": 8, "DistanceToCity_km": 7.5, "SchoolRating": 8.2,
    "CrimeRate": 2.1, "ParkingSpaces": 2, "LotSize_sqft": 2600,
    "PropertyType": "Independent House", "LocationQuality": "Good",
    "Furnishing": "Semi-Furnished", "HasGarage": 1, "HasGarden": 1,
}


def predict_one(values: dict = EXAMPLE) -> float:
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    row = pd.DataFrame([values])
    return float(model.predict(row)[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one sample house-price prediction.")
    parser.parse_args()
    price = predict_one()
    print(f"Predicted price: ${price:,.2f}")
