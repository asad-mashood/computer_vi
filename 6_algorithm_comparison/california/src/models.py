"""
The six regression algorithms compared in this project.

Each model is wrapped in a Pipeline(preprocessor -> regressor) so that
imputation, scaling and encoding are learned from the training data only
and are saved together with the model.
"""
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

# Fixed display order / colour per model (used in every figure)
MODEL_ORDER = [
    "Linear Regression",
    "Decision Tree",
    "Random Forest",
    "Neural Network",
    "XGBoost",
    "KNN",
]


def get_regressors(knn_k: int = 7) -> dict:
    """Return {name: unfitted estimator}. Hyper-parameters are documented in README."""
    return {
        # y = b0 + b1 x1 + ... + bp xp, fitted by Ordinary Least Squares
        "Linear Regression": LinearRegression(),

        # single CART tree; depth & leaf size limited to reduce over-fitting
        "Decision Tree": DecisionTreeRegressor(
            max_depth=12, min_samples_leaf=10, random_state=RANDOM_STATE),

        # bagging of 150 decorrelated trees (bootstrap rows + random feature subsets)
        "Random Forest": RandomForestRegressor(
            n_estimators=150, min_samples_leaf=3, max_features=0.5,
            n_jobs=-1, random_state=RANDOM_STATE),

        # Multi-layer perceptron 128-64-32 (ReLU, Adam). The target is also
        # standardised during training (and inverse-transformed for predictions)
        # because gradient descent is unstable on raw targets in the 100,000s.
        "Neural Network": TransformedTargetRegressor(
            regressor=MLPRegressor(
                hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam",
                alpha=1e-3, learning_rate_init=1e-3, batch_size=64, max_iter=300,
                early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
                random_state=RANDOM_STATE),
            transformer=StandardScaler()),

        # gradient boosted trees with L2 regularisation (lambda) and shrinkage (eta)
        "XGBoost": XGBRegressor(
            n_estimators=800, learning_rate=0.05, max_depth=7, min_child_weight=3,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE),

        # average of the k nearest training points (Euclidean, distance-weighted)
        "KNN": KNeighborsRegressor(n_neighbors=knn_k, weights="distance"),
    }


def build_pipeline(regressor) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", regressor)])


def get_pipelines(knn_k: int = 7) -> dict:
    return {name: build_pipeline(reg) for name, reg in get_regressors(knn_k).items()}
