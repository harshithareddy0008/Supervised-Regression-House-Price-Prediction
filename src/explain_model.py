from __future__ import annotations
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from config import RAW_DATA, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, RANDOM_STATE
from data_preprocessing import load_data, split_features_target


def run():
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    df = load_data(RAW_DATA)
    X, y = split_features_target(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)

    # Permuting original columns gives easy-to-explain feature importance.
    result = permutation_importance(
        model, X_test, y_test, scoring="neg_root_mean_squared_error",
        n_repeats=8, random_state=RANDOM_STATE, n_jobs=-1
    )
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance_Mean": result.importances_mean,
        "Importance_Std": result.importances_std,
    }).sort_values("Importance_Mean", ascending=False)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    importance.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)

    top = importance.head(12).sort_values("Importance_Mean")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["Feature"], top["Importance_Mean"])
    ax.set_xlabel("Increase in RMSE after permutation")
    ax.set_title("Permutation Feature Importance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "permutation_feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(importance.to_string(index=False))


if __name__ == "__main__":
    run()
