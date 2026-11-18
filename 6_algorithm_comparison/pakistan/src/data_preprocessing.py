"""
Data cleaning, feature engineering, train/test split and preprocessing
for the Zameen.com (Pakistan) house-price dataset.

Every model uses EXACTLY the same:
  * dataset          -> data/zameen_clean.csv (cleaned from data/raw/zameen-updated.csv.xz)
  * input features   -> FEATURES (16 inputs)
  * target           -> price in PKR (models learn log(price), metrics are reported in PKR)
  * split            -> 80:20, random_state=42
  * preprocessing    -> scaling, one-hot encoding, location target-encoding (fitted on train only)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "zameen-updated.csv.xz"
CLEAN_PATH = ROOT / "data" / "zameen_clean.csv"

TARGET = "price"
RANDOM_STATE = 42
TEST_SIZE = 0.20
SQFT_PER_MARLA = 225          # Zameen.com convention (1 kanal = 20 marla)

PROPERTY_TYPES = ["House", "Flat", "Upper Portion", "Lower Portion", "Penthouse", "Farm House"]

# Approximate centre (well-known landmark) of each city, used for distance_to_centre_km
CITY_CENTRES = {
    "Karachi": (24.8607, 67.0011),      # Saddar
    "Lahore": (31.5497, 74.3436),       # Mall Road
    "Islamabad": (33.7077, 73.0498),    # Blue Area
    "Rawalpindi": (33.5973, 73.0479),   # Saddar
    "Faisalabad": (31.4187, 73.0791),   # Clock Tower
}

NUMERIC_FEATURES = [
    "area_marla", "bedrooms", "baths", "latitude", "longitude",
    # engineered
    "log_area_marla", "marla_per_bedroom", "baths_per_bedroom", "total_rooms",
    "distance_to_centre_km", "listed_by_agency",
]
ONEHOT_FEATURES = ["property_type", "city"]
TARGET_ENCODED_FEATURES = ["city_location"]            # ~1,400 neighbourhoods
FEATURES = NUMERIC_FEATURES + ONEHOT_FEATURES + TARGET_ENCODED_FEATURES


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(raw: pd.DataFrame, return_log: bool = False):
    """Turn the raw Zameen listings into a clean 'houses for sale' table.
    Each step is logged so the report can show how many rows it removed."""
    log = [("Raw listings (sale + rent)", len(raw))]
    df = raw[raw["purpose"] == "For Sale"].copy()
    log.append(("Keep 'For Sale' only (rent prices are monthly, a different target)", len(df)))

    df = df[df["property_type"].isin(PROPERTY_TYPES)]
    log.append(("Keep residential types (drop single 'Room' listings)", len(df)))

    df["area_marla"] = np.where(df["Area Type"] == "Kanal", df["Area Size"] * 20, df["Area Size"])
    dup_cols = ["property_type", "price", "location", "city", "latitude", "longitude",
                "baths", "bedrooms", "area_marla"]
    df = df.drop_duplicates(subset=dup_cols)
    log.append(("Remove duplicate listings (same property posted again)", len(df)))

    df = df[df["bedrooms"].between(1, 12) & df["baths"].between(1, 12)]
    log.append(("Bedrooms and baths between 1 and 12 (0 = not filled in)", len(df)))

    df = df[df["area_marla"].between(1, 200) & (df["price"] > 0)]
    log.append(("Area between 1 marla and 10 kanal, price > 0", len(df)))

    # coordinates must lie within ~0.6 degrees (~65 km) of the city centre (removes swapped/garbage GPS)
    c = df["city"].map(CITY_CENTRES)
    clat, clon = c.str[0], c.str[1]
    df = df[((df["latitude"] - clat).abs() < 0.6) & ((df["longitude"] - clon).abs() < 0.6)]
    log.append(("GPS coordinates inside the listed city", len(df)))

    # price per marla outliers, judged within each city (typos such as an extra zero)
    ppm = df["price"] / df["area_marla"]
    lo = ppm.groupby(df["city"]).transform(lambda s: s.quantile(0.01))
    hi = ppm.groupby(df["city"]).transform(lambda s: s.quantile(0.99))
    df = df[(ppm >= lo) & (ppm <= hi)]
    log.append(("Remove extreme price-per-marla (1st / 99th percentile per city)", len(df)))

    keep = ["property_id", "property_type", "city", "province_name", "location", "latitude", "longitude",
            "area_marla", "bedrooms", "baths", "agency", "date_added", "price"]
    df = df[keep].reset_index(drop=True)
    df["listed_by_agency"] = df.pop("agency").notna().astype(int)
    return (df, log) if return_log else df


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered features (computed row by row, so there is no information leak)."""
    df = df.copy()
    df["log_area_marla"] = np.log(df["area_marla"])
    df["marla_per_bedroom"] = df["area_marla"] / df["bedrooms"]
    df["baths_per_bedroom"] = df["baths"] / df["bedrooms"]
    df["total_rooms"] = df["bedrooms"] + df["baths"]
    c = df["city"].map(CITY_CENTRES)
    df["distance_to_centre_km"] = haversine_km(df["latitude"], df["longitude"], c.str[0], c.str[1])
    df["city_location"] = df["city"] + " | " + df["location"]
    if "listed_by_agency" not in df:
        df["listed_by_agency"] = 0
    return df


def build_clean_dataset(force: bool = False) -> pd.DataFrame:
    if CLEAN_PATH.exists() and not force:
        return pd.read_csv(CLEAN_PATH)
    df = clean(load_raw())
    df.to_csv(CLEAN_PATH, index=False)
    return df


def get_train_test():
    """X_train, X_test, y_train, y_test (y = price in PKR) using the shared 80:20 split."""
    df = add_features(build_clean_dataset())
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def build_preprocessor() -> ColumnTransformer:
    """
    numeric       -> StandardScaler   z = (x - mean_train) / std_train
    property_type -> one-hot          (5 columns, reference = House)
    city          -> one-hot          (4 columns, reference = Karachi)
    city_location -> target encoding  (average log-price of that neighbourhood in the TRAINING data,
                                       shrunk towards the overall mean for rare neighbourhoods;
                                       computed with internal 5-fold cross-fitting to avoid leakage)
    """
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        # drop one reference category per feature (House, Karachi) so Linear Regression is not collinear;
        # unseen categories at prediction time are encoded as all-zeros
        ("cat", OneHotEncoder(drop=["House", "Karachi"], handle_unknown="ignore", sparse_output=False),
         ONEHOT_FEATURES),
        ("loc", Pipeline_TE(), TARGET_ENCODED_FEATURES),
    ])


def Pipeline_TE():
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    return Pipeline([("te", TargetEncoder(target_type="continuous",
                                          cv=KFold(5, shuffle=True, random_state=RANDOM_STATE))),
                     ("scale", StandardScaler())])


def format_pkr(x: float) -> str:
    """Pakistani style: 2.35 Crore / 85.0 Lakh."""
    if abs(x) >= 1e7:
        return f"{x/1e7:,.2f} Crore"
    if abs(x) >= 1e5:
        return f"{x/1e5:,.1f} Lakh"
    return f"{x:,.0f}"
