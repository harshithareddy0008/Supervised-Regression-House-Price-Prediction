from pathlib import Path
import sys
import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from config import MODELS_DIR

st.set_page_config(page_title="House Price Regression", page_icon="🏠", layout="centered")
st.title("🏠 House Price Prediction")
st.caption("End-to-end supervised regression demo. Values are for the bundled synthetic benchmarking dataset.")

model_path = MODELS_DIR / "best_model.joblib"
if not model_path.exists():
    st.error("Model file not found. Run: python src/train_models.py")
    st.stop()
model = joblib.load(model_path)

area = st.number_input("Area (sqft)", 450.0, 6500.0, 1800.0, 50.0)
bedrooms = st.slider("Bedrooms", 1, 7, 3)
bathrooms = st.slider("Bathrooms", 1, 5, 2)
floors = st.slider("Floors / apartment floor", 1, 15, 2)
age = st.number_input("House age (years)", 0.0, 60.0, 8.0, 1.0)
distance = st.number_input("Distance to city (km)", 0.3, 45.0, 7.5, 0.5)
school = st.slider("School rating", 2.0, 10.0, 8.2, 0.1)
crime = st.slider("Crime rate", 0.0, 10.0, 2.1, 0.1)
parking = st.slider("Parking spaces", 0, 4, 2)
lot = st.number_input("Lot size (sqft)", 150.0, 14000.0, 2600.0, 50.0)
ptype = st.selectbox("Property type", ["Apartment", "Townhouse", "Independent House", "Villa"], index=2)
location = st.selectbox("Location quality", ["Poor", "Average", "Good", "Premium"], index=2)
furnishing = st.selectbox("Furnishing", ["Unfurnished", "Semi-Furnished", "Furnished"], index=1)
garage = st.checkbox("Has garage", True)
garden = st.checkbox("Has garden", True)

if st.button("Predict price", type="primary"):
    row = pd.DataFrame([{
        "Area_sqft": area, "Bedrooms": bedrooms, "Bathrooms": bathrooms,
        "Floors": floors, "HouseAge_years": age, "DistanceToCity_km": distance,
        "SchoolRating": school, "CrimeRate": crime, "ParkingSpaces": parking,
        "LotSize_sqft": lot, "PropertyType": ptype, "LocationQuality": location,
        "Furnishing": furnishing, "HasGarage": int(garage), "HasGarden": int(garden),
    }])
    pred = float(model.predict(row)[0])
    st.metric("Predicted House Price", f"${pred:,.0f}")
