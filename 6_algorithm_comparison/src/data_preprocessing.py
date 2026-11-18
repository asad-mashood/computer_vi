"""
Data loading, feature engineering, train/test split and preprocessing.

Every model in this project uses EXACTLY the same:
  * dataset            -> data/housing.csv (California Housing, 20,640 districts)
  * input features     -> 9 raw features + 3 engineered ratios (see FEATURES)
  * target variable    -> median_house_value (USD)
  * train/test split   -> 80:20, random_state=42
  * preprocessing      -> median imputation + StandardScaler (numeric),
                          one-hot encoding (categorical)
so the comparison between algorithms is fair.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "housing.csv"

TARGET = "median_house_value"
RANDOM_STATE = 42
TEST_SIZE = 0.20

NUMERIC_FEATURES = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
    # engineered features
    "rooms_per_household", "bedrooms_per_room", "population_per_household",
]
CATEGORICAL_FEATURES = ["ocean_proximity"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV."""
    return pd.read_csv(path)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-household ratios, which are far more informative than raw block totals."""
    df = df.copy()
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]  # NaN where bedrooms missing -> imputed
    df["population_per_household"] = df["population"] / df["households"]
    return df


def get_train_test(path: Path = DATA_PATH):
    """Return X_train, X_test, y_train, y_test using the shared 80:20 split."""
    df = add_features(load_data(path))
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def build_preprocessor() -> ColumnTransformer:
    """
    Numeric : fill missing values with the TRAINING median, then z-score scale
              z = (x - mean_train) / std_train
    Category: one-hot encode ocean_proximity.
    The transformer is fitted on the training fold only (inside each Pipeline),
    so no information from the test set leaks into training.
    """
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def describe_dataset(path: Path = DATA_PATH) -> dict:
    df = load_data(path)
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_total_bedrooms": int(df["total_bedrooms"].isna().sum()),
        "target_mean": float(df[TARGET].mean()),
        "target_capped_at_500001": int((df[TARGET] >= 500001).sum()),
    }
