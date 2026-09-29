from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "housing_regression.csv"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed" / "housing_cleaned.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RANDOM_STATE = 42
TARGET = "Price_USD"

NUMERIC_FEATURES = [
    "Area_sqft", "Bedrooms", "Bathrooms", "Floors", "HouseAge_years",
    "DistanceToCity_km", "SchoolRating", "CrimeRate", "ParkingSpaces",
    "LotSize_sqft", "HasGarage", "HasGarden"
]
CATEGORICAL_FEATURES = ["PropertyType", "LocationQuality", "Furnishing"]
