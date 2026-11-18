"""
Load a trained model from models/ and predict the price of new districts.

    python predict.py                       # predict 3 example districts with all six models
    python predict.py --model xgboost       # one model only
"""
import argparse
import sys
from pathlib import Path

# Ensure local src/ package is discoverable from any CWD
DIR = Path(__file__).resolve().parent
if str(DIR) not in sys.path:
    sys.path.insert(0, str(DIR))

import joblib
import pandas as pd

from src.data_preprocessing import FEATURES, add_features

MODELS_DIR = DIR / "models"

EXAMPLES = pd.DataFrame([
    # coastal Bay Area, high income
    dict(longitude=-122.25, latitude=37.85, housing_median_age=30, total_rooms=2500,
         total_bedrooms=450, population=1100, households=420, median_income=7.5,
         ocean_proximity="NEAR BAY"),
    # inland, modest income
    dict(longitude=-119.80, latitude=36.75, housing_median_age=25, total_rooms=2200,
         total_bedrooms=430, population=1300, households=400, median_income=2.8,
         ocean_proximity="INLAND"),
    # LA area, average income
    dict(longitude=-118.30, latitude=34.05, housing_median_age=35, total_rooms=1800,
         total_bedrooms=420, population=1200, households=390, median_income=3.9,
         ocean_proximity="<1H OCEAN"),
])


def main(model_name=None):
    X = add_features(EXAMPLES)[FEATURES]
    paths = [MODELS_DIR / f"{model_name}.joblib"] if model_name else sorted(MODELS_DIR.glob("*.joblib"))
    out = EXAMPLES[["median_income", "ocean_proximity"]].copy()
    for p in paths:
        try:
            out[p.stem] = joblib.load(p).predict(X).round(0)
        except Exception as e:
            out[p.stem] = f"Error: {e}"
    pd.set_option("display.width", 200)
    print(out.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["linear_regression", "decision_tree", "random_forest",
                                        "neural_network", "xgboost", "knn"])
    main(ap.parse_args().model)
