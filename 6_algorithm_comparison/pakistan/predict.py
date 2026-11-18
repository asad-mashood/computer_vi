"""
Predict the price of example properties with every saved model.

    python predict.py
    python predict.py --model xgboost
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.data_preprocessing import FEATURES, add_features, format_pkr

MODELS_DIR = Path(__file__).resolve().parent / "models"

EXAMPLES = pd.DataFrame([
    dict(property_type="House", city="Lahore", location="DHA Defence", latitude=31.4740, longitude=74.4040,
         area_marla=20, bedrooms=5, baths=6, listed_by_agency=1),                      # 1 kanal house, DHA Lahore
    dict(property_type="House", city="Islamabad", location="G-13", latitude=33.6510, longitude=72.9640,
         area_marla=7, bedrooms=4, baths=4, listed_by_agency=1),                       # 7 marla, G-13
    dict(property_type="Flat", city="Karachi", location="Gulistan-e-Jauhar", latitude=24.9160, longitude=67.1270,
         area_marla=6, bedrooms=3, baths=3, listed_by_agency=1),                       # 3-bed flat, Karachi
])


def main(model=None):
    X = add_features(EXAMPLES)[FEATURES]
    paths = [MODELS_DIR / f"{model}.joblib"] if model else sorted(MODELS_DIR.glob("*.joblib"))
    out = EXAMPLES[["city", "location", "property_type", "area_marla", "bedrooms"]].copy()
    for p in paths:
        out[p.stem] = [format_pkr(v) for v in joblib.load(p).predict(X)]
    pd.set_option("display.width", 250)
    print(out.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["linear_regression", "decision_tree", "random_forest",
                                        "neural_network", "xgboost", "knn"])
    main(ap.parse_args().model)
