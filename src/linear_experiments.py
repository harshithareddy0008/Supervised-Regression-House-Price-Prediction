"""Simple and polynomial regression experiments kept separate for learning clarity."""
from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from config import RAW_DATA, REPORTS_DIR, FIGURES_DIR, RANDOM_STATE, TARGET


def run():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_DATA)
    X = df[["Area_sqft"]]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)

    rows = []

    # Simple Linear Regression: y = b0 + b1*x
    simple = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LinearRegression()),
    ])
    simple.fit(X_train, y_train)
    pred = simple.predict(X_test)
    rows.append({"Experiment":"Simple Linear Regression", "Degree":1,
                 "RMSE":np.sqrt(mean_squared_error(y_test, pred)), "R2":r2_score(y_test, pred)})

    # Polynomial Regression on the same single feature.
    for degree in [2, 3, 4]:
        poly = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("model", LinearRegression()),
        ])
        poly.fit(X_train, y_train)
        pred = poly.predict(X_test)
        rows.append({"Experiment":f"Polynomial Regression degree {degree}", "Degree":degree,
                     "RMSE":np.sqrt(mean_squared_error(y_test, pred)), "R2":r2_score(y_test, pred)})

    out = pd.DataFrame(rows).sort_values("RMSE")
    out.to_csv(REPORTS_DIR / "simple_polynomial_results.csv", index=False)

    # Plot fitted curves for interpretation.
    grid = pd.DataFrame({"Area_sqft":np.linspace(X["Area_sqft"].min(), X["Area_sqft"].max(), 250)})
    fig, ax = plt.subplots(figsize=(9, 6))
    sample = df.sample(1000, random_state=RANDOM_STATE)
    ax.scatter(sample["Area_sqft"], sample[TARGET], alpha=0.2)
    ax.plot(grid["Area_sqft"], simple.predict(grid), label="Simple linear")
    best_degree = int(out.iloc[0]["Degree"])
    best_poly = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("poly", PolynomialFeatures(degree=best_degree, include_bias=False)),
        ("model", LinearRegression()),
    ]).fit(X_train, y_train)
    ax.plot(grid["Area_sqft"], best_poly.predict(grid), label=f"Polynomial degree {best_degree}")
    ax.set_xlabel("Area (sqft)")
    ax.set_ylabel("Price (USD)")
    ax.set_title("Simple Linear vs Polynomial Regression")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "linear_vs_polynomial.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    run()
