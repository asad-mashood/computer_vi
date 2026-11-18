"""All figures for the report. Each model keeps the same colour in every chart."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .models import MODEL_ORDER

COLORS = dict(zip(MODEL_ORDER, [
    "#2a78d6",  # blue    - Linear Regression
    "#eb6834",  # orange  - Decision Tree
    "#1baf7a",  # aqua    - Random Forest
    "#eda100",  # yellow  - Neural Network
    "#e87ba4",  # magenta - XGBoost
    "#008300",  # green   - KNN
]))
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e6e5e0"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "text.parse_math": False,   # treat "$" as a currency sign, not LaTeX
})


def _k(x, _pos=None):
    return f"${x/1000:,.0f}k"


def plot_actual_vs_predicted(y_test, predictions: dict, metrics: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), sharex=True, sharey=True)
    lo, hi = 0, 520_000
    for ax, name in zip(axes.ravel(), MODEL_ORDER):
        ax.scatter(y_test, predictions[name], s=8, alpha=0.35, color=COLORS[name],
                   edgecolors="none")
        ax.plot([lo, hi], [lo, hi], color=INK, lw=1.2, ls="--", label="Perfect prediction")
        r = metrics.loc[name]
        ax.set_title(name)
        ax.text(0.03, 0.97, f"R² = {r.R2:.3f}\nRMSE = ${r.RMSE:,.0f}\nMAE = ${r.MAE:,.0f}",
                transform=ax.transAxes, va="top", fontsize=9, color=INK,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID))
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.xaxis.set_major_formatter(_k); ax.yaxis.set_major_formatter(_k)
    for ax in axes[1]:
        ax.set_xlabel("Actual median house value")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted median house value")
    axes[0, 0].legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle("Actual vs predicted house values on the 20% test set (n = 4,128)",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out); plt.close(fig)


def plot_metric_bars(metrics: pd.DataFrame, out: Path):
    panels = [("MAE", "MAE (USD) — lower is better", True),
              ("RMSE", "RMSE (USD) — lower is better", True),
              ("MSE", "MSE (USD²) — lower is better", True),
              ("R2", "R² score — higher is better", False)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for ax, (col, title, money) in zip(axes.ravel(), panels):
        vals = metrics.loc[MODEL_ORDER, col]
        y = np.arange(len(MODEL_ORDER))[::-1]
        ax.barh(y, vals, color=[COLORS[m] for m in MODEL_ORDER], height=0.62)
        ax.set_yticks(y, MODEL_ORDER)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        vmax = vals.max()
        for yi, v in zip(y, vals):
            label = (f"{v/1e9:.2f}B" if col == "MSE" else f"${v:,.0f}") if money else f"{v:.3f}"
            ax.text(v + vmax * 0.01, yi, label, va="center", fontsize=9, color=INK)
        ax.set_xlim(0, vmax * 1.18 if money else 1.0)
        if col == "MSE":
            ax.xaxis.set_major_formatter(lambda x, _: f"{x/1e9:.0f}B")
        elif money:
            ax.xaxis.set_major_formatter(_k)
    fig.suptitle("Test-set performance of the six regression models",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out); plt.close(fig)


def plot_residuals(y_test, predictions: dict, out: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    bins = np.linspace(-250_000, 250_000, 61)
    for ax, name in zip(axes.ravel(), MODEL_ORDER):
        res = np.asarray(y_test) - predictions[name]
        ax.hist(res, bins=bins, color=COLORS[name], alpha=0.9, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color=INK, lw=1.1, ls="--")
        ax.set_title(f"{name}  (mean {res.mean():+,.0f}, sd {res.std():,.0f})")
        ax.xaxis.set_major_formatter(_k)
    for ax in axes[1]:
        ax.set_xlabel("Residual = actual − predicted")
    for ax in axes[:, 0]:
        ax.set_ylabel("Number of test districts")
    fig.suptitle("Residual distributions on the test set (narrower and centred on 0 is better)",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out); plt.close(fig)


def plot_sample_predictions(y_test, predictions: dict, out: Path, n: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(y_test), n, replace=False))
    x = np.arange(n)
    y_true = np.asarray(y_test)[idx]
    order = np.argsort(y_true)
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.plot(x, y_true[order], color=INK, lw=2.4, marker="o", ms=5, label="Actual", zorder=5)
    for name in MODEL_ORDER:
        ax.plot(x, predictions[name][idx][order], color=COLORS[name], lw=1.4,
                marker="o", ms=3.5, alpha=0.9, label=name)
    ax.set_xlabel(f"{n} randomly chosen test districts (sorted by actual value)")
    ax.set_ylabel("Median house value")
    ax.yaxis.set_major_formatter(_k)
    ax.legend(ncol=7, loc="upper left", frameon=False, fontsize=9)
    ax.set_title("Actual vs predicted values for individual test districts", loc="left")
    fig.tight_layout()
    fig.savefig(out); plt.close(fig)


def plot_feature_importance(importances: dict, out: Path, top: int = 12):
    fig, axes = plt.subplots(1, len(importances), figsize=(14, 5.5))
    for ax, (name, s) in zip(np.atleast_1d(axes), importances.items()):
        s = s.sort_values().tail(top)
        ax.barh(s.index, s.values, color=COLORS[name], height=0.62)
        ax.set_title(f"{name} — feature importance", loc="left")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out); plt.close(fig)


def plot_knn_k(ks, scores, best_k, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(ks, scores, color=COLORS["KNN"], lw=2, marker="o", ms=5)
    ax.axvline(best_k, color=INK2, ls="--", lw=1)
    ax.annotate(f"best k = {best_k}", (best_k, max(scores)), xytext=(8, -14),
                textcoords="offset points", color=INK)
    ax.set_xlabel("Number of neighbours k")
    ax.set_ylabel("5-fold CV RMSE on training set (USD)")
    ax.yaxis.set_major_formatter(_k)
    ax.set_title("Choosing k for KNN (training data only)", loc="left")
    fig.tight_layout()
    fig.savefig(out); plt.close(fig)


def plot_eda(df: pd.DataFrame, out_corr: Path, out_target: Path):
    num = df.select_dtypes("number")
    corr = num.corr()
    cmap = LinearSegmentedColormap.from_list("div", ["#eb6834", "#efeee9", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Pearson correlation")
    ax.set_title("Correlation matrix of numeric features", loc="left")
    fig.tight_layout(); fig.savefig(out_corr); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(df["median_house_value"], bins=60, color=COLORS["Linear Regression"],
                 edgecolor="white", linewidth=0.5)
    axes[0].xaxis.set_major_formatter(_k)
    axes[0].set_title("Target: median_house_value (note the cap at $500k)", loc="left")
    axes[0].set_xlabel("Median house value"); axes[0].set_ylabel("Districts")
    axes[1].scatter(df["median_income"], df["median_house_value"], s=4, alpha=0.2,
                    color=COLORS["Linear Regression"], edgecolors="none")
    axes[1].yaxis.set_major_formatter(_k)
    axes[1].set_title("Strongest predictor: median_income", loc="left")
    axes[1].set_xlabel("Median income (tens of thousands USD)"); axes[1].set_ylabel("Median house value")
    fig.tight_layout(); fig.savefig(out_target); plt.close(fig)


# ---------------------------------------------------------------------------
# Additional analysis figures
# ---------------------------------------------------------------------------
SPLIT_COLORS = {"Train (80%)": "#2a78d6", "Test (20%)": "#eb6834"}


def plot_data_split(y_train, y_test, out: Path):
    """Target distribution of the training and test sets (shows the split is representative)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 2.4]})
    ax = axes[0]
    sizes = [len(y_train), len(y_test)]
    ax.bar(list(SPLIT_COLORS), sizes, color=list(SPLIT_COLORS.values()), width=0.55)
    for i, s in enumerate(sizes):
        ax.text(i, s + 250, f"{s:,} rows", ha="center", color=INK)
    ax.set_ylim(0, max(sizes) * 1.15)
    ax.set_title("80:20 train/test split", loc="left"); ax.grid(axis="x", visible=False)
    ax = axes[1]
    bins = np.linspace(0, 510_000, 52)
    for (lab, col), y in zip(SPLIT_COLORS.items(), (y_train, y_test)):
        ax.hist(y, bins=bins, density=True, histtype="step", lw=2, color=col,
                label=f"{lab}: mean ${np.mean(y):,.0f}, median ${np.median(y):,.0f}")
    ax.xaxis.set_major_formatter(_k); ax.set_yticks([])
    ax.set_xlabel("Median house value"); ax.set_ylabel("Density")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
    ax.legend(frameon=False, loc="upper center", ncol=1)
    ax.set_title("Target distribution: train and test sets have the same shape", loc="left")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def _seq_cmap():
    return LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])


def plot_geo(df: pd.DataFrame, out: Path):
    """House values on the map of California."""
    fig, ax = plt.subplots(figsize=(8.5, 8))
    d = df.sort_values("median_house_value")
    sc = ax.scatter(d["longitude"], d["latitude"], c=d["median_house_value"], cmap=_seq_cmap(),
                    s=6, alpha=0.8, edgecolors="none")
    cb = fig.colorbar(sc, ax=ax, fraction=0.04)
    cb.set_label("Median house value"); cb.ax.yaxis.set_major_formatter(_k)
    for name, lon, lat in [("San Francisco", -122.42, 37.77), ("Los Angeles", -118.24, 34.05),
                           ("San Diego", -117.16, 32.72), ("Sacramento", -121.49, 38.58),
                           ("Fresno", -119.79, 36.74)]:
        ax.annotate(name, (lon, lat), xytext=(10, 4), textcoords="offset points", fontsize=9,
                    color=INK, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=GRID, alpha=0.85))
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Location drives price: coastal districts are the most expensive", loc="left")
    ax.set_aspect(1.15)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_train_vs_test(metrics: pd.DataFrame, out: Path):
    """Train vs test R² — the gap shows over-fitting, low values on both show under-fitting."""
    fig, ax = plt.subplots(figsize=(12, 4.8))
    x = np.arange(len(MODEL_ORDER)); w = 0.36
    tr = metrics.loc[MODEL_ORDER, "Train R2"].values
    te = metrics.loc[MODEL_ORDER, "R2"].values
    ax.bar(x - w/2 - 0.01, tr, w, color="#9ec5f4", label="Train R²")
    ax.bar(x + w/2 + 0.01, te, w, color="#2a78d6", label="Test R²")
    for xi, a, b in zip(x, tr, te):
        ax.text(xi - w/2, a + 0.012, f"{a:.3f}", ha="center", fontsize=8.5, color=INK)
        ax.text(xi + w/2, b + 0.012, f"{b:.3f}", ha="center", fontsize=8.5, color=INK)
        ax.text(xi, max(a, b) + 0.065, f"gap {a-b:.3f}", ha="center", fontsize=9, color=INK2,
                fontweight="bold")
    ax.set_xticks(x, MODEL_ORDER); ax.set_ylim(0, 1.18); ax.grid(axis="x", visible=False)
    ax.set_ylabel("R² score")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.set_title("Training vs test R² (large gap = over-fitting, both low = under-fitting)", loc="left")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_cv(metrics: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(11, 4.4))
    m = metrics.loc[MODEL_ORDER]
    y = np.arange(len(MODEL_ORDER))[::-1]
    ax.barh(y, m["CV RMSE (mean)"], xerr=m["CV RMSE (std)"], height=0.6,
            color=[COLORS[n] for n in MODEL_ORDER], error_kw=dict(ecolor=INK, capsize=4, lw=1.2))
    ax.scatter(m["RMSE"], y, marker="D", s=40, color=INK, zorder=5, label="Test-set RMSE")
    for yi, v, s in zip(y, m["CV RMSE (mean)"], m["CV RMSE (std)"]):
        ax.text(v + s + 900, yi, f"${v:,.0f} ± {s:,.0f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, MODEL_ORDER); ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(_k); ax.set_xlim(0, m["CV RMSE (mean)"].max() * 1.3)
    ax.set_xlabel("RMSE (USD)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("5-fold cross-validation RMSE on the training set (bar = mean, line = ±1 std)", loc="left")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def error_by_price_range(y_test, predictions: dict) -> pd.DataFrame:
    edges = [0, 100_000, 200_000, 300_000, 400_000, 500_002]
    labels = ["< $100k", "$100–200k", "$200–300k", "$300–400k", "≥ $400k"]
    yb = pd.cut(np.asarray(y_test), edges, labels=labels, right=False)
    rows = {n: pd.Series(np.abs(np.asarray(y_test) - p)).groupby(yb, observed=False).mean()
            for n, p in predictions.items()}
    t = pd.DataFrame(rows)[MODEL_ORDER]
    t.insert(0, "Test rows", pd.Series(yb).value_counts().reindex(labels).values)
    return t


def plot_error_by_price(tbl: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(tbl)); w = 0.135
    for i, n in enumerate(MODEL_ORDER):
        ax.bar(x + (i - 2.5) * (w + 0.008), tbl[n], w, color=COLORS[n], label=n)
    ax.set_xticks(x, [f"{i}\n(n = {c:,})" for i, c in zip(tbl.index, tbl["Test rows"])])
    ax.yaxis.set_major_formatter(_k); ax.grid(axis="x", visible=False)
    ax.set_ylabel("Mean absolute error (USD)")
    ax.legend(ncol=6, frameon=False, loc="upper left", fontsize=9)
    ax.set_ylim(0, tbl[MODEL_ORDER].values.max() * 1.18)
    ax.set_title("Mean absolute error by actual price range (test set)", loc="left")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_abs_error_box(y_test, predictions: dict, out: Path):
    fig, ax = plt.subplots(figsize=(12, 4.8))
    data = [np.abs(np.asarray(y_test) - predictions[n]) for n in MODEL_ORDER]
    bp = ax.boxplot(data, vert=False, patch_artist=True, showfliers=False, widths=0.55,
                    medianprops=dict(color=INK, lw=1.6), whiskerprops=dict(color=INK2),
                    capprops=dict(color=INK2))
    for patch, n in zip(bp["boxes"], MODEL_ORDER):
        patch.set_facecolor(COLORS[n]); patch.set_edgecolor("white")
    for i, d in enumerate(data, start=1):
        ax.text(np.percentile(d, 75) + 3000, i + 0.33, f"median ${np.median(d):,.0f}",
                fontsize=8.5, color=INK)
    ax.set_yticks(range(1, 7), MODEL_ORDER); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(_k); ax.set_xlabel("Absolute error |actual − predicted| (outliers hidden)")
    ax.set_title("Spread of absolute errors on the test set", loc="left")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_nn_loss(mlp, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    axes[0].plot(np.arange(1, len(mlp.loss_curve_) + 1), mlp.loss_curve_, color=COLORS["Neural Network"], lw=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training loss (squared error / 2, scaled target)")
    axes[0].set_title("Neural network training loss", loc="left")
    vs = mlp.validation_scores_
    best = int(np.argmax(vs)) + 1
    axes[1].plot(np.arange(1, len(vs) + 1), vs, color=COLORS["Neural Network"], lw=2)
    axes[1].axvline(best, color=INK2, ls="--", lw=1)
    axes[1].text(0.97, 0.06, f"best epoch {best}, validation R² = {max(vs):.3f}\n"
                 f"(stopped after {len(vs)} epochs)", transform=axes[1].transAxes, ha="right",
                 color=INK, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID))
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation R² (10% hold-out of training set)")
    axes[1].set_title("Early stopping: training stops when validation R² stops improving", loc="left")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_tree_top(tree, feature_names, num_mean, num_scale, n_numeric, out: Path, depth: int = 3):
    """Draw the first levels of the fitted tree with thresholds converted back to ORIGINAL units."""
    t = tree.tree_
    fig, ax = plt.subplots(figsize=(20, 8.5))
    ax.set_axis_off()
    vmin, vmax = t.value[:, 0, 0].min(), t.value[:, 0, 0].max()
    cmap = _seq_cmap()

    def label(node):
        f, thr = t.feature[node], t.threshold[node]
        name = feature_names[f]
        if f < n_numeric:
            return f"{name} ≤ {thr * num_scale[f] + num_mean[f]:,.2f}"
        return f"{name.replace('ocean_proximity_', 'ocean = ')}?"

    def draw(node, x, y, dx, level):
        val, n = t.value[node, 0, 0], t.n_node_samples[node]
        leafish = level == depth or t.children_left[node] == -1
        txt = (f"{label(node)}\n" if not leafish else "") + f"n = {n:,}\nmean price ${val/1000:,.0f}k"
        shade = (val - vmin) / (vmax - vmin)
        ax.text(x, y, txt, ha="center", va="center", fontsize=10,
                color="white" if shade > 0.55 else INK,
                bbox=dict(boxstyle="round,pad=0.5", fc=cmap(0.15 + 0.85 * shade), ec="white", lw=1.5))
        if leafish:
            return
        for child, sign, lab in ((t.children_left[node], -1, None), (t.children_right[node], 1, None)):
            cx, cy = x + sign * dx, y - 1
            ax.plot([x, cx], [y - 0.22, cy + 0.25], color=INK2, lw=1, zorder=0)
            f = t.feature[node]
            yes_no = ("yes", "no") if f < n_numeric else ("no", "yes")
            ax.text((x + cx) / 2 + sign * 0.15, (y + cy) / 2 + 0.05, yes_no[0] if sign < 0 else yes_no[1],
                    fontsize=9, color=INK2, ha="center")
            draw(child, cx, cy, dx / 2, level + 1)

    draw(0, 0, 0, 2 ** (depth - 1) * 1.05, 0)
    ax.set_xlim(-2 ** depth * 1.08, 2 ** depth * 1.08); ax.set_ylim(-depth - 0.6, 0.6)
    ax.set_title(f"Decision tree — first {depth} levels of {tree.get_depth()} "
                 f"({tree.get_n_leaves():,} leaves in total). Darker = higher predicted price.",
                 loc="left", fontsize=13)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_linear_coefficients(coefs: pd.Series, out: Path):
    s = coefs.sort_values()
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(s.index, s.values, color=["#eb6834" if v < 0 else "#2a78d6" for v in s.values], height=0.62)
    ax.axvline(0, color=INK2, lw=1)
    ax.xaxis.set_major_formatter(lambda x, _: f"{'-' if x < 0 else ''}${abs(x)/1000:,.0f}k")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Change in predicted price for +1 standard deviation (or category = 1)")
    ax.set_title("Linear Regression coefficients (blue raises, orange lowers the price)", loc="left")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_error_map(X_test, y_test, pred, name, out: Path):
    err = np.abs(np.asarray(y_test) - pred)
    order = np.argsort(err)
    cmap = LinearSegmentedColormap.from_list("err", ["#fbe0d3", "#f2a07c", "#eb6834", "#b8421a", "#7a2a0e"])
    fig, ax = plt.subplots(figsize=(8.5, 8))
    sc = ax.scatter(X_test["longitude"].values[order], X_test["latitude"].values[order], c=err[order],
                    cmap=cmap, vmax=np.percentile(err, 98), s=9, edgecolors="none")
    cb = fig.colorbar(sc, ax=ax, fraction=0.04); cb.set_label("Absolute error"); cb.ax.yaxis.set_major_formatter(_k)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_aspect(1.15)
    ax.set_title(f"Where does {name} make its largest errors? (test set)", loc="left")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
