"""
Train, evaluate and compare six regression algorithms on Pakistani property prices (Zameen.com).

    python train.py            # full run (with 5-fold cross-validation, ~10 min on 2 CPU cores)
    python train.py --no-cv    # faster (~3 min)

Outputs
  data/zameen_clean.csv, data/train.csv, data/test.csv
  models/*.joblib                  trained models (cleaning-independent: preprocessing included)
  results/*.csv / *.md / *.json    metrics, predictions, error breakdowns
  results/figures/*.png            all figures used in the report
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor

from src import visualize as viz
from src.data_preprocessing import (FEATURES, format_pkr, NUMERIC_FEATURES, RANDOM_STATE, TARGET, add_features,
                                    clean, get_train_test, load_raw, CLEAN_PATH)
from src.evaluate import regression_metrics
from src.models import MODEL_ORDER, build_model, get_models, inner

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parent
MODELS_DIR, RESULTS_DIR = ROOT / "models", ROOT / "results"
FIG = RESULTS_DIR / "figures"
FILE_NAMES = {"Linear Regression": "linear_regression", "Decision Tree": "decision_tree",
              "Random Forest": "random_forest", "Neural Network": "neural_network",
              "XGBoost": "xgboost", "KNN": "knn"}


def log_rmse(y, p):
    return float(np.sqrt(np.mean((np.log(y) - np.log(np.clip(p, 1, None))) ** 2)))


def select_knn_k(X, y, cv):
    ks = list(range(1, 31, 2))
    scorer = make_scorer(log_rmse, greater_is_better=False)
    scores = [-cross_val_score(build_model(KNeighborsRegressor(n_neighbors=k, weights="distance")),
                               X, y, cv=cv, scoring=scorer, n_jobs=-1).mean() for k in ks]
    return ks, scores, ks[int(np.argmin(scores))]


def describe_tree_split(pre):
    """Build a function that turns a (scaled) tree split back into original units."""
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    num = pre.named_transformers_["num"]
    loc = pre.named_transformers_["loc"].named_steps["scale"]
    n_num = len(NUMERIC_FEATURES)

    def f(i, thr):
        name = names[i]
        if i < n_num:
            v = thr * num.scale_[i] + num.mean_[i]
            unit = {"area_marla": " marla", "distance_to_centre_km": " km"}.get(name, "")
            return f"{name} ≤ {v:,.2f}{unit}"
        if name == "city_location":
            v = np.exp(thr * loc.scale_[0] + loc.mean_[0])
            return f"neighbourhood typical\nprice ≤ {viz.pkr(v)}"
        return name.replace("property_type_", "type ≠ ").replace("city_", "city ≠ ")   # left branch = value 0
    return f


def main(run_cv=True):
    for d in (MODELS_DIR, FIG):
        d.mkdir(parents=True, exist_ok=True)

    # ---------------- data ----------------
    raw = load_raw()
    clean_df, clean_log = clean(raw, return_log=True)
    clean_df.to_csv(CLEAN_PATH, index=False)
    df = add_features(clean_df)
    pd.DataFrame(clean_log, columns=["step", "rows"]).to_csv(RESULTS_DIR / "cleaning_log.csv", index=False)
    print(f"Clean dataset: {len(df):,} rows")

    X_train, X_test, y_train, y_test = get_train_test()
    X_train.assign(**{TARGET: y_train}).to_csv(ROOT / "data" / "train.csv", index_label="row_id", float_format="%.6g")
    X_test.assign(**{TARGET: y_test}).to_csv(ROOT / "data" / "test.csv", index_label="row_id", float_format="%.6g")
    print(f"Train {X_train.shape}  Test {X_test.shape}  (80:20, random_state={RANDOM_STATE})")

    viz.plot_cleaning_funnel(clean_log, FIG / "cleaning_funnel.png")
    viz.plot_eda_target(df, FIG / "eda_price_distribution.png")
    viz.plot_eda_city_type(df, FIG / "eda_city_and_type.png")
    viz.plot_eda_maps(df, FIG / "eda_city_maps.png")
    viz.plot_top_locations(df, FIG / "eda_top_neighbourhoods.png")
    viz.plot_eda_correlation(df, FIG / "eda_correlation.png")
    viz.plot_data_split(y_train, y_test, FIG / "data_split.png")

    cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    print("Choosing k for KNN ...")
    ks, k_scores, best_k = select_knn_k(X_train, y_train, cv)
    viz.plot_knn_k(ks, k_scores, best_k, FIG / "knn_k_selection.png")
    print(f"  best k = {best_k}")

    # ---------------- training ----------------
    rows, preds, train_preds, fitted = [], {}, {}, {}
    for name, model in get_models(knn_k=best_k).items():
        print(f"Training {name} ...")
        t0 = time.perf_counter(); model.fit(X_train, y_train); t = time.perf_counter() - t0
        preds[name] = model.predict(X_test); train_preds[name] = model.predict(X_train)
        m = regression_metrics(y_test, preds[name])
        row = {"Model": name, **m, "Train R2": regression_metrics(y_train, train_preds[name])["R2"],
               "Train time (s)": t}
        if run_cv:
            s = cross_val_score(clone(model), X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1)
            row["CV RMSE (mean)"], row["CV RMSE (std)"] = -s.mean(), s.std()
        rows.append(row); fitted[name] = model
        joblib.dump(model, MODELS_DIR / f"{FILE_NAMES[name]}.joblib", compress=("xz", 9))
        print(f"  R2={m['R2']:.4f}  RMSE={viz.pkr(m['RMSE'])}  MAPE={m['MAPE (%)']:.1f}%  ({t:.1f}s)")

    metrics = pd.DataFrame(rows).set_index("Model").loc[MODEL_ORDER]
    metrics["Rank (RMSE)"] = metrics["RMSE"].rank().astype(int)
    metrics.to_csv(RESULTS_DIR / "metrics.csv", float_format="%.6f")
    ranked = metrics.sort_values("RMSE")

    md = ["| Rank | Model | MAE (PKR) | MSE (PKR²) | RMSE (PKR) | R² | R² (log) | MAPE | Within ±10% | Train R² | Train time (s) |",
          "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n, r in ranked.iterrows():
        md.append(f"| {int(r['Rank (RMSE)'])} | {n} | {format_pkr(r.MAE)} | {r.MSE:.3e} | {format_pkr(r.RMSE)} | {r.R2:.4f} | "
                  f"{r['R2 (log)']:.4f} | {r['MAPE (%)']:.1f}% | {r['Within 10%']:.1f}% | {r['Train R2']:.4f} | {r['Train time (s)']:.1f} |")
    (RESULTS_DIR / "metrics.md").write_text("\n".join(md) + "\n")

    # predictions rounded to the nearest 1,000 PKR (far below any model's error) to keep the files small
    pd.DataFrame({"actual": y_test.values, **{n: np.round(p, -3) for n, p in preds.items()}},
                 index=y_test.index).to_csv(RESULTS_DIR / "test_predictions.csv", float_format="%.0f", index_label="row_id")
    pd.DataFrame({"actual": y_train.values, **{n: np.round(p, -3) for n, p in train_preds.items()}},
                 index=y_train.index).to_csv(RESULTS_DIR / "train_predictions.csv", float_format="%.0f", index_label="row_id")

    # ---------------- error breakdowns ----------------
    band = pd.cut(y_test.values, viz.PRICE_BANDS, labels=viz.BAND_LABELS, right=False)
    t_band = viz.error_by_group(y_test, preds, band, viz.BAND_LABELS)
    t_city = viz.error_by_group(y_test, preds, X_test["city"].values, viz.CITY_ORDER)
    types = X_test["property_type"].value_counts().index.tolist()
    t_type = viz.error_by_group(y_test, preds, X_test["property_type"].values, types)
    t_band.to_csv(RESULTS_DIR / "mape_by_price_range.csv", float_format="%.2f")
    t_city.to_csv(RESULTS_DIR / "mape_by_city.csv", float_format="%.2f")
    t_type.to_csv(RESULTS_DIR / "mape_by_property_type.csv", float_format="%.2f")

    # ---------------- model insight ----------------
    pre, xgb = inner(fitted["XGBoost"]); names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    imps = {"Random Forest": pd.Series(inner(fitted["Random Forest"])[1].feature_importances_, index=names),
            "XGBoost": pd.Series(xgb.feature_importances_, index=names)}
    pd.DataFrame(imps).to_csv(RESULTS_DIR / "feature_importance.csv", float_format="%.5f")
    lin = pd.Series(inner(fitted["Linear Regression"])[1].coef_, index=names)
    lin.to_csv(RESULTS_DIR / "linear_regression_coefficients.csv", header=["coefficient_on_log_price"])

    # ---------------- figures ----------------
    viz.plot_actual_vs_predicted(y_test, preds, metrics, FIG / "actual_vs_predicted.png")
    viz.plot_sample_predictions(y_test, preds, FIG / "sample_predictions.png")
    viz.plot_metric_bars(metrics, FIG / "metrics_comparison.png")
    viz.plot_residuals(y_test, preds, FIG / "residuals.png")
    viz.plot_abs_error_box(y_test, preds, FIG / "abs_error_boxplot.png")
    viz.plot_train_vs_test(metrics, FIG / "train_vs_test_r2.png")
    if run_cv:
        viz.plot_cv(metrics, FIG / "cross_validation_rmse.png")
    viz.plot_error_groups(t_band, "Mean absolute percentage error by price range (test set)", FIG / "mape_by_price_range.png")
    viz.plot_error_groups(t_city, "Mean absolute percentage error by city (test set)", FIG / "mape_by_city.png")
    viz.plot_error_groups(t_type, "Mean absolute percentage error by property type (test set)", FIG / "mape_by_property_type.png")
    viz.plot_feature_importance(imps, FIG / "feature_importance.png")
    viz.plot_linear_coefficients(lin, FIG / "linear_coefficients.png")
    viz.plot_nn_loss(inner(fitted["Neural Network"])[1].regressor_, FIG / "nn_training_curve.png")
    pre_dt, dt = inner(fitted["Decision Tree"])
    viz.plot_tree_top(dt, describe_tree_split(pre_dt), FIG / "decision_tree_top_levels.png")

    best = ranked.index[0]
    summary = {"raw_rows": int(len(raw)), "clean_rows": int(len(df)),
               "split": {"train": int(len(X_train)), "test": int(len(X_test))},
               "features": FEATURES, "knn_best_k": best_k, "best_model_by_rmse": best,
               "metrics": metrics.round(6).reset_index().to_dict(orient="records")}
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "\n".join(md) + f"\n\nBest model (lowest test RMSE): {best}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cv", action="store_true")
    main(run_cv=not ap.parse_args().no_cv)
