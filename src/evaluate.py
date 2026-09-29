from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error


def regression_metrics(y_true, y_pred, n_features: int | None = None) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    adjusted_r2 = np.nan
    if n_features is not None and len(y_true) > n_features + 1:
        adjusted_r2 = 1 - (1-r2) * (len(y_true)-1) / (len(y_true)-n_features-1)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted_R2": adjusted_r2,
        "MAPE_percent": mape,
    }
