"""Regression metrics (all computed in Pakistani Rupees unless stated otherwise)."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    """
    MAE   = (1/n) * sum |y - y_hat|                       (PKR)
    MSE   = (1/n) * sum (y - y_hat)^2                     (PKR^2)
    RMSE  = sqrt(MSE)                                     (PKR)
    R2    = 1 - sum (y - y_hat)^2 / sum (y - y_bar)^2
    R2(log) = R2 computed on log(price) - how well the model ranks cheap vs expensive homes
    MAPE  = (100/n) * sum |y - y_hat| / y                 (% error, scale-free)
    Within 10% / 20% = share of properties whose prediction is within +-10% / +-20% of the price
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mse = mean_squared_error(y_true, y_pred)
    ape = np.abs(y_true - y_pred) / y_true
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
        "R2 (log)": r2_score(np.log(y_true), np.log(np.clip(y_pred, 1, None))),
        "MAPE (%)": 100 * ape.mean(),
        "Within 10%": 100 * (ape <= 0.10).mean(),
        "Within 20%": 100 * (ape <= 0.20).mean(),
    }
