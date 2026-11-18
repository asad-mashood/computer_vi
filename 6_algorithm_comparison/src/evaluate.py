"""Regression metrics used to compare the six models."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    """
    MAE  = (1/n) * sum |y_i - y_hat_i|
    MSE  = (1/n) * sum (y_i - y_hat_i)^2
    RMSE = sqrt(MSE)
    R2   = 1 - SS_res / SS_tot = 1 - sum (y_i - y_hat_i)^2 / sum (y_i - y_bar)^2
    """
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }
