from __future__ import annotations
import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from config import RAW_DATA, MODELS_DIR, REPORTS_DIR, RANDOM_STATE
from data_preprocessing import load_data, split_features_target, build_preprocessor
from evaluate import regression_metrics


def tune(model_name: str = "random_forest"):
    df = load_data(RAW_DATA)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)

    if model_name == "random_forest":
        model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        params = {
            "model__n_estimators": [150, 250, 400],
            "model__max_depth": [None, 12, 20, 30],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 0.8, 1.0],
        }
    elif model_name == "gradient_boosting":
        model = GradientBoostingRegressor(random_state=RANDOM_STATE)
        params = {
            "model__n_estimators": [100, 180, 260, 350],
            "model__learning_rate": [0.02, 0.05, 0.08, 0.12],
            "model__max_depth": [2, 3, 4, 5],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__subsample": [0.8, 0.9, 1.0],
        }
    else:
        raise ValueError("Use 'random_forest' or 'gradient_boosting'.")

    pipe = Pipeline([
        ("preprocess", build_preprocessor(scale_numeric=False)),
        ("model", model),
    ])
    search = RandomizedSearchCV(
        pipe, params, n_iter=18, cv=5, scoring="neg_root_mean_squared_error",
        n_jobs=-1, random_state=RANDOM_STATE, verbose=1
    )
    search.fit(X_train, y_train)
    preds = search.predict(X_test)
    n_features = len(search.best_estimator_.named_steps["preprocess"].get_feature_names_out())
    metrics = regression_metrics(y_test, preds, n_features)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, MODELS_DIR / f"tuned_{model_name}.joblib")
    pd.DataFrame([{**metrics, **{f"param_{k}":v for k,v in search.best_params_.items()}}]).to_csv(
        REPORTS_DIR / f"tuning_{model_name}.csv", index=False
    )
    print("Best parameters:", search.best_params_)
    print("Test metrics:", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="random_forest", choices=["random_forest", "gradient_boosting"])
    args = parser.parse_args()
    tune(args.model)
