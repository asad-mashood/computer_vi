"""
The six regression algorithms.

Every model is:   log/exp target wrapper  ->  Pipeline( preprocessor -> regressor )

* Property prices are extremely skewed (a few crore-rupee kanal houses next to many
  50-lakh flats). All six models therefore learn  log(price)  and their predictions are
  converted back with  exp(.)  so every metric is reported in Pakistani Rupees (PKR).
* Preprocessing (scaling, one-hot, location target-encoding) is fitted on the
  training data only and saved inside the model file.
"""
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from .data_preprocessing import RANDOM_STATE, build_preprocessor

MODEL_ORDER = ["Linear Regression", "Decision Tree", "Random Forest",
               "Neural Network", "XGBoost", "KNN"]


def get_regressors(knn_k: int = 10) -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=14, min_samples_leaf=8,
                                               random_state=RANDOM_STATE),
        # max_leaf_nodes keeps each tree (and the saved model file) compact; chosen on a validation split
        "Random Forest": RandomForestRegressor(n_estimators=80, max_features=0.4, min_samples_leaf=2,
                                               max_leaf_nodes=1200, n_jobs=-1,
                                               random_state=RANDOM_STATE),
        "Neural Network": TransformedTargetRegressor(
            regressor=MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam",
                                   alpha=1e-4, learning_rate_init=1e-3, batch_size=128, max_iter=300,
                                   early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
                                   random_state=RANDOM_STATE),
            transformer=StandardScaler()),
        "XGBoost": XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=8, min_child_weight=3,
                                subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, reg_alpha=0.0,
                                tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE),
        "KNN": KNeighborsRegressor(n_neighbors=knn_k, weights="distance"),
    }


def build_model(regressor):
    """log-target wrapper around (preprocessing -> regressor)."""
    return TransformedTargetRegressor(
        regressor=Pipeline([("preprocess", build_preprocessor()), ("model", regressor)]),
        func=np.log, inverse_func=np.exp, check_inverse=False)


def get_models(knn_k: int = 10) -> dict:
    return {name: build_model(reg) for name, reg in get_regressors(knn_k).items()}


def inner(model):
    """Return (fitted preprocessor, fitted estimator) of a wrapped model."""
    pipe = model.regressor_
    return pipe.named_steps["preprocess"], pipe.named_steps["model"]
