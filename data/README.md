# Dataset

`raw/housing_regression.csv` contains 6,000 synthetic residential-property records generated with a fixed random seed (`42`). It is intentionally designed for regression learning and benchmarking.

## Target
- `Price_USD` — continuous house-price target.

## Features
- `Area_sqft`: built-up area.
- `Bedrooms`, `Bathrooms`, `Floors`.
- `HouseAge_years`.
- `DistanceToCity_km`.
- `SchoolRating` (2–10).
- `CrimeRate` (0–10).
- `ParkingSpaces`.
- `LotSize_sqft`.
- `PropertyType`: Apartment, Townhouse, Independent House, Villa.
- `LocationQuality`: Poor, Average, Good, Premium.
- `Furnishing`: Unfurnished, Semi-Furnished, Furnished.
- `HasGarage`, `HasGarden`: binary 0/1.

A small amount of missing predictor data is included deliberately so the project demonstrates imputation and pipelines. The target has no missing values.

**Portfolio note:** this is a generated educational dataset, not real market data. Say that clearly in interviews. The generation logic is available in `src/generate_dataset.py`.
