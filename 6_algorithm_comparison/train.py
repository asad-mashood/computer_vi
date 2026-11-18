"""
Train, evaluate and compare six regression algorithms on California Housing.

    python train.py            # full run (includes 5-fold cross-validation)
    python train.py --no-cv    # faster: skip cross-validation

Outputs
  models/*.joblib                 trained pipelines (preprocessing + model)
  results/metrics.csv / .md       comparison table
  results/test_predictions.csv    actual vs predicted for every test row
  results/figures/*.png           all figures used in the report
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score

from src.data_preprocessing import (RANDOM_STATE, TARGET, add_features, describe_dataset,
                                    get_train_test, load_data)
from src.evaluate import regression_metrics
from src.models import MODEL_ORDER, build_pipeline, get_pipelines
from sklearn.neighbors import KNeighborsRegressor
from src import visualize as viz

ROOT = Path(__file__).resolve().parent
MODELS_DIR, RESULTS_DIR = ROOT / "models", ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
FILE_NAMES = {
    "Linear Regression": "linear_regression", "Decision Tree": "decision_tree",
    "Random Forest": "random_forest", "Neural Network": "neural_network",
    "XGBoost": "xgboost", "KNN": "knn",
}


def select_knn_k(X_train, y_train, cv):
    ks = list(range(1, 31, 2))
    rmse = []
    for k in ks:
        pipe = build_pipeline(KNeighborsRegressor(n_neighbors=k, weights="distance"))
        s = cross_val_score(pipe, X_train, y_train, cv=cv,
                            scoring="neg_root_mean_squared_error", n_jobs=-1)
        rmse.append(-s.mean())
    best = ks[int(np.argmin(rmse))]
    return ks, rmse, best


def main(run_cv: bool = True):
    for d in (MODELS_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    info = describe_dataset()
    print("Dataset:", info)
    X_train, X_test, y_train, y_test = get_train_test()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}  (80:20 split, random_state={RANDOM_STATE})")

    # Save the exact training and test sets used by every model
    X_train.assign(**{TARGET: y_train}).to_csv(ROOT / "data" / "train.csv", index_label="row_id")
    X_test.assign(**{TARGET: y_test}).to_csv(ROOT / "data" / "test.csv", index_label="row_id")
    viz.plot_data_split(y_train, y_test, FIG_DIR / "data_split.png")
    viz.plot_geo(load_data(), FIG_DIR / "eda_geo_prices.png")

    viz.plot_eda(add_features(load_data()), FIG_DIR / "eda_correlation.png",
                 FIG_DIR / "eda_target.png")

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    print("Selecting k for KNN with 5-fold CV on the training set ...")
    ks, k_rmse, best_k = select_knn_k(X_train, y_train, cv)
    viz.plot_knn_k(ks, k_rmse, best_k, FIG_DIR / "knn_k_selection.png")
    print(f"  best k = {best_k}")

    rows, preds, train_preds, fitted = [], {}, {}, {}
    for name, pipe in get_pipelines(knn_k=best_k).items():
        print(f"Training {name} ...")
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        y_pred = pipe.predict(X_test)
        test_m = regression_metrics(y_test, y_pred)
        train_preds[name] = pipe.predict(X_train)
        train_r2 = regression_metrics(y_train, train_preds[name])["R2"]
        row = {"Model": name, **test_m, "Train R2": train_r2, "Train time (s)": train_time}

        if run_cv:
            s = cross_val_score(build_pipeline(pipe.named_steps["model"]), X_train, y_train,
                                cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
            row["CV RMSE (mean)"], row["CV RMSE (std)"] = -s.mean(), s.std()

        rows.append(row)
        preds[name], fitted[name] = y_pred, pipe
        joblib.dump(pipe, MODELS_DIR / f"{FILE_NAMES[name]}.joblib", compress=3)
        print(f"  R2={test_m['R2']:.4f}  RMSE={test_m['RMSE']:,.0f}  MAE={test_m['MAE']:,.0f}  ({train_time:.1f}s)")

    metrics = pd.DataFrame(rows).set_index("Model").loc[MODEL_ORDER]
    ranked = metrics.sort_values("RMSE")
    metrics["Rank (RMSE)"] = metrics["RMSE"].rank().astype(int)
    metrics.to_csv(RESULTS_DIR / "metrics.csv", float_format="%.6f")

    # Markdown table for the README / report
    md = ["| Rank | Model | MAE ($) | MSE ($²) | RMSE ($) | R² | Train R² | CV RMSE ($) | Train time (s) |",
          "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in ranked.iterrows():
        cvs = f"{r['CV RMSE (mean)']:,.0f} ± {r['CV RMSE (std)']:,.0f}" if run_cv else "—"
        md.append(f"| {int(metrics.loc[name, 'Rank (RMSE)'])} | {name} | {r.MAE:,.0f} | {r.MSE:.3e} | "
                  f"{r.RMSE:,.0f} | {r.R2:.4f} | {r['Train R2']:.4f} | {cvs} | {r['Train time (s)']:.2f} |")
    (RESULTS_DIR / "metrics.md").write_text("\n".join(md) + "\n")

    pd.DataFrame({"actual": y_test.values, **{n: preds[n] for n in MODEL_ORDER}},
                 index=y_test.index).to_csv(RESULTS_DIR / "test_predictions.csv", float_format="%.2f",
                                            index_label="row_id")
    pd.DataFrame({"actual": y_train.values, **{n: train_preds[n] for n in MODEL_ORDER}},
                 index=y_train.index).to_csv(RESULTS_DIR / "train_predictions.csv", float_format="%.2f",
                                             index_label="row_id")
    err_tbl = viz.error_by_price_range(y_test, preds)
    err_tbl.to_csv(RESULTS_DIR / "error_by_price_range.csv", float_format="%.2f")

    # Feature importances of the tree ensembles
    imps = {}
    for name in ("Random Forest", "XGBoost"):
        p = fitted[name]
        names = [n.split("__", 1)[1] for n in p.named_steps["preprocess"].get_feature_names_out()]
        imps[name] = pd.Series(p.named_steps["model"].feature_importances_, index=names)
    viz.plot_feature_importance(imps, FIG_DIR / "feature_importance.png")

    viz.plot_actual_vs_predicted(y_test, preds, metrics, FIG_DIR / "actual_vs_predicted.png")
    viz.plot_metric_bars(metrics, FIG_DIR / "metrics_comparison.png")
    viz.plot_residuals(y_test, preds, FIG_DIR / "residuals.png")
    viz.plot_sample_predictions(y_test, preds, FIG_DIR / "sample_predictions.png")
    viz.plot_train_vs_test(metrics, FIG_DIR / "train_vs_test_r2.png")
    if run_cv:
        viz.plot_cv(metrics, FIG_DIR / "cross_validation_rmse.png")
    viz.plot_error_by_price(err_tbl, FIG_DIR / "error_by_price_range.png")
    viz.plot_abs_error_box(y_test, preds, FIG_DIR / "abs_error_boxplot.png")
    viz.plot_nn_loss(fitted["Neural Network"].named_steps["model"].regressor_, FIG_DIR / "nn_training_curve.png")
    feat_names = [n.split("__", 1)[1] for n in fitted["Decision Tree"].named_steps["preprocess"].get_feature_names_out()]
    scaler = fitted["Decision Tree"].named_steps["preprocess"].named_transformers_["num"].named_steps["scaler"]
    viz.plot_tree_top(fitted["Decision Tree"].named_steps["model"], feat_names, scaler.mean_, scaler.scale_,
                      len(scaler.mean_), FIG_DIR / "decision_tree_top_levels.png")
    lin = fitted["Linear Regression"].named_steps["model"]
    pd.Series(lin.coef_, index=feat_names).to_csv(RESULTS_DIR / "linear_regression_coefficients.csv", header=["coefficient"])
    viz.plot_linear_coefficients(pd.Series(lin.coef_, index=feat_names), FIG_DIR / "linear_coefficients.png")

    best = ranked.index[0]
    viz.plot_error_map(X_test, y_test, preds[best], best, FIG_DIR / "best_model_error_map.png")
    summary = {"dataset": info, "split": {"train": len(X_train), "test": len(X_test)},
               "knn_best_k": best_k, "best_model": best,
               "metrics": metrics.round(6).reset_index().to_dict(orient="records")}
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "\n".join(md))
    print(f"\nBest model (lowest test RMSE): {best}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cv", action="store_true", help="skip 5-fold cross-validation")
    main(run_cv=not ap.parse_args().no_cv)
