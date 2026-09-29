from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET


def load_data(path) -> pd.DataFrame:
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(num_steps)
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ], remainder="drop", verbose_feature_names_out=False)
