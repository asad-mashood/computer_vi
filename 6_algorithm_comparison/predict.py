"""
Master CLI Prediction Script for Both Regression Datasets:
  - Pakistan Real Estate (Zameen.com) in PKR
  - California Housing Benchmark in USD

Usage:
    python predict.py                    # runs inference on both datasets
    python predict.py --dataset pakistan  # runs only on Pakistan dataset
    python predict.py --dataset california # runs only on California dataset
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_pakistan(model_name=None):
    print("=" * 80)
    print(" 🇵🇰 PAKISTAN REAL ESTATE (ZAMEEN.COM) - 6 ALGORITHM PREDICTIONS (PKR)")
    print("=" * 80)
    cmd = [sys.executable, str(ROOT / "pakistan" / "predict.py")]
    if model_name:
        cmd.extend(["--model", model_name])
    subprocess.run(cmd, check=True, cwd=str(ROOT / "pakistan"))


def run_california(model_name=None):
    print("\n" + "=" * 80)
    print(" 🇺🇸 CALIFORNIA HOUSING BENCHMARK - 6 ALGORITHM PREDICTIONS (USD)")
    print("=" * 80)
    cmd = [sys.executable, str(ROOT / "california" / "predict.py")]
    if model_name:
        cmd.extend(["--model", model_name])
    subprocess.run(cmd, check=True, cwd=str(ROOT / "california"))


def main():
    parser = argparse.ArgumentParser(description="Multi-dataset 6-algorithm regression predictor")
    parser.add_argument("--dataset", choices=["both", "pakistan", "california"], default="both",
                        help="Which dataset to run predictions on")
    parser.add_argument("--model", type=str, default=None,
                        help="Optional specific model filter (e.g. xgboost, random_forest)")
    args = parser.parse_args()

    if args.dataset in ["both", "pakistan"]:
        run_pakistan(args.model)
    if args.dataset in ["both", "california"]:
        run_california(args.model)


if __name__ == "__main__":
    main()
