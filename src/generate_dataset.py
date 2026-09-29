"""Recreate the bundled regression dataset with the same random seed."""
from pathlib import Path
import numpy as np
import pandas as pd


def generate_dataset(output_path: str | Path, n: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    property_types = np.array(['Apartment', 'Townhouse', 'Independent House', 'Villa'])
    locations = np.array(['Poor', 'Average', 'Good', 'Premium'])
    furnishings = np.array(['Unfurnished', 'Semi-Furnished', 'Furnished'])

    property_type = rng.choice(property_types, n, p=[0.42, 0.18, 0.30, 0.10])
    location_quality = rng.choice(locations, n, p=[0.12, 0.43, 0.32, 0.13])
    furnishing = rng.choice(furnishings, n, p=[0.35, 0.42, 0.23])

    area = np.clip(rng.lognormal(mean=np.log(1650), sigma=0.42, size=n), 450, 5200)
    area *= np.select(
        [property_type=='Apartment', property_type=='Townhouse', property_type=='Independent House', property_type=='Villa'],
        [0.82, 1.00, 1.12, 1.35], default=1.0
    )
    area = np.clip(area, 450, 6500)
    bedrooms = np.clip(np.round(area/650 + rng.normal(0.2, 0.7, n)), 1, 7).astype(int)
    bathrooms = np.clip(np.round(bedrooms*0.62 + rng.normal(0.5, 0.6, n)), 1, 5).astype(int)
    floors = np.where(property_type=='Apartment', rng.integers(1, 16, n), rng.integers(1, 4, n))
    house_age = np.clip(rng.gamma(2.4, 7.0, n), 0, 60)
    distance = np.clip(rng.gamma(2.0, 5.2, n), 0.3, 45)
    school = np.clip(rng.normal(6.7, 1.45, n), 2, 10)
    crime = np.clip(rng.beta(2.0, 5.0, n)*10, 0, 10)
    parking = np.clip((bedrooms/2.3 + rng.normal(0, 0.8, n)).round(), 0, 4).astype(int)
    has_garage = rng.binomial(1, np.where(property_type=='Apartment', 0.35, 0.72), n)
    has_garden = rng.binomial(1, np.where(property_type=='Apartment', 0.08, 0.62), n)
    lot_multiplier = np.where(property_type=='Apartment', rng.uniform(0.25, 0.60, n), rng.uniform(1.15, 2.60, n))
    lot_size = np.clip(area * lot_multiplier + rng.normal(0, 120, n), 150, 14000)

    location_effect = pd.Series(location_quality).map({'Poor':-70000, 'Average':0, 'Good':95000, 'Premium':240000}).to_numpy()
    property_effect = pd.Series(property_type).map({'Apartment':-25000, 'Townhouse':35000, 'Independent House':85000, 'Villa':210000}).to_numpy()
    furnishing_effect = pd.Series(furnishing).map({'Unfurnished':0, 'Semi-Furnished':24000, 'Furnished':52000}).to_numpy()

    price = (
        55000 + area*175 + bedrooms*8500 + bathrooms*15500 + school*10500
        - house_age*2200 - distance*3100 - crime*10500 + parking*7500
        + lot_size*12 + has_garage*22000 + has_garden*30000
        + location_effect + property_effect + furnishing_effect
        + 0.012*np.square(np.maximum(area-1800, 0))
        + np.where(location_quality=='Premium', area*28, 0)
        + rng.normal(0, 42000, n)
    )
    price = np.clip(price, 70000, None)

    df = pd.DataFrame({
        'Area_sqft': np.round(area, 1), 'Bedrooms': bedrooms, 'Bathrooms': bathrooms,
        'Floors': floors, 'HouseAge_years': np.round(house_age, 1),
        'DistanceToCity_km': np.round(distance, 2), 'SchoolRating': np.round(school, 2),
        'CrimeRate': np.round(crime, 2), 'ParkingSpaces': parking,
        'LotSize_sqft': np.round(lot_size, 1), 'PropertyType': property_type,
        'LocationQuality': location_quality, 'Furnishing': furnishing,
        'HasGarage': has_garage, 'HasGarden': has_garden, 'Price_USD': np.round(price, 2)
    })
    for col, frac in {'HouseAge_years':0.018, 'SchoolRating':0.022, 'CrimeRate':0.016, 'LotSize_sqft':0.012, 'Furnishing':0.010}.items():
        idx = rng.choice(n, size=int(n*frac), replace=False)
        df.loc[idx, col] = np.nan

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    from config import RAW_DATA
    data = generate_dataset(RAW_DATA)
    print(f"Saved {len(data):,} rows to {RAW_DATA}")
