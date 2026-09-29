from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor,
    GradientBoostingRegressor, HistGradientBoostingRegressor
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

from config import RAW_DATA, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, RANDOM_STATE
from data_preprocessing import load_data, split_features_target, build_preprocessor
from evaluate import regression_metrics


class CatBoostSklearnAdapter(RegressorMixin, BaseEstimator):
    """Compatibility adapter for CatBoost with newer scikit-learn tag checks."""
    def __init__(self, iterations=260, depth=6, learning_rate=0.05, random_seed=42):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.random_seed = random_seed

    def fit(self, X, y):
        from catboost import CatBoostRegressor
        self.model_ = CatBoostRegressor(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            loss_function="RMSE",
            random_seed=self.random_seed,
            verbose=False,
            allow_writing_files=False,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def get_models():
    models = {
        "Dummy Baseline": (DummyRegressor(strategy="median"), False),
        "Multiple Linear Regression": (LinearRegression(), True),
        "Ridge Regression": (Ridge(alpha=1.0), True),
        "Lasso Regression": (Lasso(alpha=250.0, max_iter=20000), True),
        "ElasticNet Regression": (ElasticNet(alpha=100.0, l1_ratio=0.5, max_iter=20000), True),
        "KNN Regressor": (KNeighborsRegressor(n_neighbors=7, weights="distance"), True),
        "Support Vector Regression": (SVR(kernel="rbf", C=1000, epsilon=0.1), True),
        "Decision Tree Regressor": (DecisionTreeRegressor(max_depth=12, min_samples_leaf=4, random_state=RANDOM_STATE), False),
        "Random Forest Regressor": (RandomForestRegressor(n_estimators=160, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE), False),
        "Extra Trees Regressor": (ExtraTreesRegressor(n_estimators=160, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE), False),
        "AdaBoost Regressor": (AdaBoostRegressor(n_estimators=160, learning_rate=0.05, random_state=RANDOM_STATE), False),
        "Gradient Boosting Regressor": (GradientBoostingRegressor(n_estimators=180, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE), False),
        "HistGradientBoosting Regressor": (HistGradientBoostingRegressor(max_iter=220, learning_rate=0.08, random_state=RANDOM_STATE), False),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost Regressor"] = (XGBRegressor(
            n_estimators=260, learning_rate=0.05, max_depth=5,
            subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror",
            random_state=RANDOM_STATE, n_jobs=-1
        ), False)
    except Exception:
        pass

    try:
        import catboost  # noqa: F401
        models["CatBoost Regressor"] = (CatBoostSklearnAdapter(
            iterations=260, depth=6, learning_rate=0.05, random_seed=RANDOM_STATE
        ), False)
    except Exception:
        pass
    return models


def build_pipeline(estimator, scale_numeric: bool):
    return Pipeline([
        ("preprocess", build_preprocessor(scale_numeric=scale_numeric)),
        ("model", estimator),
    ])


def train_all_models(run_cv: bool = False):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(RAW_DATA)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    results = []
    fitted = {}
    for name, (estimator, scale_numeric) in get_models().items():
        print(f"Training: {name}")
        pipe = build_pipeline(clone(estimator), scale_numeric)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        n_features = len(pipe.named_steps["preprocess"].get_feature_names_out())
        row = {"Model": name, **regression_metrics(y_test, preds, n_features)}
        if run_cv:
            cv_scores = -cross_val_score(pipe, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
            row["CV_RMSE_mean"] = cv_scores.mean()
            row["CV_RMSE_std"] = cv_scores.std()
        results.append(row)
        fitted[name] = pipe

    results_df = pd.DataFrame(results).sort_values("RMSE", ascending=True).reset_index(drop=True)
    results_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    best_name = results_df.iloc[0]["Model"]
    best_pipe = fitted[best_name]
    joblib.dump(best_pipe, MODELS_DIR / "best_model.joblib")
    (MODELS_DIR / "best_model_name.txt").write_text(str(best_name), encoding="utf-8")

    # Save test predictions from best model.
    best_preds = best_pipe.predict(X_test)
    pred_df = X_test.copy()
    pred_df["Actual_Price_USD"] = y_test.values
    pred_df["Predicted_Price_USD"] = best_preds
    pred_df["Absolute_Error_USD"] = np.abs(pred_df["Actual_Price_USD"] - pred_df["Predicted_Price_USD"])
    pred_df.to_csv(REPORTS_DIR / "best_model_test_predictions.csv", index=False)

    # Model comparison chart.
    top = results_df.sort_values("RMSE", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["Model"], top["RMSE"])
    ax.set_xlabel("RMSE (USD)")
    ax.set_title("Regression Model Comparison - Lower RMSE is Better")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_rmse_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Actual vs predicted.
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, best_preds, alpha=0.35)
    low = min(y_test.min(), best_preds.min())
    high = max(y_test.max(), best_preds.max())
    ax.plot([low, high], [low, high])
    ax.set_xlabel("Actual Price (USD)")
    ax.set_ylabel("Predicted Price (USD)")
    ax.set_title(f"Actual vs Predicted - {best_name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "actual_vs_predicted.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\nTop models:")
    print(results_df[["Model", "MAE", "RMSE", "R2", "MAPE_percent"]].head(10).to_string(index=False))
    print(f"\nBest model saved: {best_name}")
    return results_df, best_pipe


if __name__ == "__main__":
    train_all_models(run_cv=False)
