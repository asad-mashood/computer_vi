"""All figures for the report. Each model keeps the same colour in every chart."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FuncFormatter

from .models import MODEL_ORDER

COLORS = dict(zip(MODEL_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
CITY_ORDER = ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad"]
CITY_COLORS = dict(zip(CITY_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e6e5e0"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
WARM = LinearSegmentedColormap.from_list("warm", ["#fbe0d3", "#f2a07c", "#eb6834", "#b8421a", "#7a2a0e"])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 140, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "text.parse_math": False,
})


def pkr(x, _pos=None):
    """Axis label in Pakistani units."""
    if abs(x) >= 1e7:
        v = x / 1e7
        return f"{v:,.0f} Cr" if v >= 10 or float(v).is_integer() else f"{v:,.1f} Cr"
    if abs(x) >= 1e5:
        return f"{x/1e5:,.0f} L"
    return f"{x:,.0f}"


PKR = FuncFormatter(pkr)
LOG_TICKS = [5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8, 2e8, 5e8, 1e9]


def _log_axes(ax, lo, hi, both=True):
    ax.set_xscale("log"); ax.set_yscale("log") if both else None
    ticks = [t for t in LOG_TICKS if lo <= t <= hi]
    ax.set_xticks(ticks); ax.xaxis.set_major_formatter(PKR); ax.xaxis.set_minor_formatter(FuncFormatter(lambda *a: ""))
    if both:
        ax.set_yticks(ticks); ax.yaxis.set_major_formatter(PKR); ax.yaxis.set_minor_formatter(FuncFormatter(lambda *a: ""))


def _save(fig, out, dpi=None):
    fig.tight_layout(); fig.savefig(out, dpi=dpi or plt.rcParams["savefig.dpi"]); plt.close(fig)


# ---------------------------------------------------------------------------
# Data / EDA
# ---------------------------------------------------------------------------
def plot_cleaning_funnel(log, out):
    labels = [s for s, _ in log]; n = [v for _, v in log]
    fig, ax = plt.subplots(figsize=(12, 5))
    y = np.arange(len(n))[::-1]
    ax.barh(y, n, color="#2a78d6", height=0.6)
    for yi, v, prev in zip(y, n, [None] + n[:-1]):
        lab = f"{v:,}" + (f"   (−{prev - v:,})" if prev else "")
        ax.text(v + 1500, yi, lab, va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [f"{i}. {l}" for i, l in enumerate(labels)], fontsize=9)
    ax.set_xlim(0, max(n) * 1.22); ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:,.0f}k"))
    ax.set_xlabel("Rows remaining")
    ax.set_title("Data cleaning: from 168,446 raw Zameen.com listings to a clean dataset", loc="left")
    _save(fig, out)


def plot_eda_target(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    ax = axes[0]
    bins = np.logspace(np.log10(df.price.min()), np.log10(df.price.max()), 70)
    ax.hist(df.price, bins=bins, color="#2a78d6", edgecolor="white", linewidth=0.4)
    ax.set_xscale("log"); ax.set_xticks([t for t in LOG_TICKS if df.price.min() <= t <= df.price.max()])
    ax.xaxis.set_major_formatter(PKR); ax.xaxis.set_minor_formatter(FuncFormatter(lambda *a: ""))
    ax.axvline(df.price.median(), color=INK, ls="--", lw=1.2)
    ax.text(df.price.median() * 1.08, ax.get_ylim()[1] * 0.92, f"median {pkr(df.price.median())}", color=INK)
    ax.set_xlabel("Price (PKR, log scale)"); ax.set_ylabel("Listings")
    ax.set_title("Price distribution (very skewed → models learn log(price))", loc="left")
    ax = axes[1]
    for c in CITY_ORDER:
        d = df[df.city == c]
        ax.scatter(d.area_marla, d.price, s=3, alpha=0.25, color=CITY_COLORS[c], edgecolors="none", label=c)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([1, 2, 3, 5, 10, 20, 40, 100, 200], ["1", "2", "3", "5", "10", "20\n(1 kanal)", "40", "100", "200"])
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda *a: ""))
    ax.set_yticks([t for t in LOG_TICKS if df.price.min() <= t <= df.price.max()])
    ax.yaxis.set_major_formatter(PKR); ax.yaxis.set_minor_formatter(FuncFormatter(lambda *a: ""))
    ax.set_xlabel("Area (marla, log scale)"); ax.set_ylabel("Price (PKR, log scale)")
    leg = ax.legend(frameon=False, markerscale=4, loc="upper left")
    for h in leg.legend_handles: h.set_alpha(1)
    ax.set_title("Strongest predictor: area — but the city shifts the whole cloud", loc="left")
    _save(fig, out, dpi=115)


def plot_eda_city_type(df, out):
    d = df.assign(ppm=df.price / df.area_marla)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 1.3]})
    med = d.groupby("city").ppm.median().reindex(CITY_ORDER)
    cnt = d.city.value_counts().reindex(CITY_ORDER)
    ax = axes[0]
    ax.bar(CITY_ORDER, med, color=[CITY_COLORS[c] for c in CITY_ORDER], width=0.6)
    for i, (v, n) in enumerate(zip(med, cnt)):
        ax.text(i, v * 1.02, f"{pkr(v)}\n({n:,} listings)", ha="center", fontsize=9, color=INK)
    ax.yaxis.set_major_formatter(PKR); ax.set_ylim(0, med.max() * 1.25); ax.grid(axis="x", visible=False)
    ax.set_ylabel("Median price per marla")
    ax.set_title("Median price per marla by city", loc="left")
    ax = axes[1]
    types = d.property_type.value_counts().index
    med_t = d.groupby("property_type").price.median().reindex(types)
    ax.barh(types[::-1], med_t[::-1], color="#2a78d6", height=0.6)
    for i, (t, v) in enumerate(zip(types[::-1], med_t[::-1])):
        ax.text(v * 1.02, i, f"{pkr(v)}  (n = {d.property_type.value_counts()[t]:,})", va="center", fontsize=9, color=INK)
    ax.xaxis.set_major_formatter(PKR); ax.set_xlim(0, med_t.max() * 1.45); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Median price")
    ax.set_title("Median price by property type", loc="left")
    _save(fig, out)


def plot_eda_maps(df, out):
    """One map per city; colour = how expensive a listing's price per marla is compared with its own city."""
    d = df.assign(ppm=df.price / df.area_marla)
    d["pct"] = d.groupby("city").ppm.rank(pct=True) * 100
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.6))
    for ax, c in zip(axes, CITY_ORDER):
        s = d[d.city == c].sort_values("pct")
        lo, hi = s[["longitude", "latitude"]].quantile(0.005), s[["longitude", "latitude"]].quantile(0.995)
        sc = ax.scatter(s.longitude, s.latitude, c=s.pct, cmap=SEQ, vmin=0, vmax=100, s=7, edgecolors="none")
        ax.set_xlim(lo.longitude, hi.longitude); ax.set_ylim(lo.latitude, hi.latitude)
        ax.set_title(f"{c}  (median {pkr(s.ppm.median())}/marla)", loc="left", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal", adjustable="box"); ax.grid(False)
    cb = fig.colorbar(sc, ax=axes, fraction=0.012, pad=0.01)
    cb.set_label("Price per marla percentile within the city")
    fig.suptitle("Where is property expensive? Each dot is a listing (dark = among the most expensive per marla in that city)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.savefig(out); plt.close(fig)


def plot_top_locations(df, out, n=15):
    d = df.assign(ppm=df.price / df.area_marla)
    g = d.groupby(["city", "location"]).agg(ppm=("ppm", "median"), n=("ppm", "size")).reset_index()
    g = g[g.n >= 50]
    top, bot = g.nlargest(n, "ppm"), g.nsmallest(n, "ppm")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, t, title in ((axes[0], top, f"{n} most expensive neighbourhoods"), (axes[1], bot, f"{n} cheapest neighbourhoods")):
        t = t.sort_values("ppm")
        ax.barh(t.location + " (" + t.city + ")", t.ppm, color=[CITY_COLORS[c] for c in t.city], height=0.65)
        for i, (v, k) in enumerate(zip(t.ppm, t.n)):
            ax.text(v * 1.01, i, f"{pkr(v)}", va="center", fontsize=8.5, color=INK)
        ax.xaxis.set_major_formatter(PKR); ax.set_xlim(0, t.ppm.max() * 1.18); ax.grid(axis="y", visible=False)
        ax.set_title(title + " (median price per marla, ≥ 50 listings)", loc="left")
        ax.tick_params(axis="y", labelsize=8.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CITY_COLORS[c]) for c in CITY_ORDER]
    axes[1].legend(handles, CITY_ORDER, frameon=False, loc="lower right")
    _save(fig, out)


def plot_eda_correlation(df, out):
    cols = ["price", "area_marla", "log_area_marla", "bedrooms", "baths", "total_rooms", "marla_per_bedroom",
            "baths_per_bedroom", "distance_to_centre_km", "latitude", "longitude", "listed_by_agency"]
    d = df[cols].copy(); d["log_price"] = np.log(d.pop("price"))
    corr = d.corr()
    cmap = LinearSegmentedColormap.from_list("div", ["#eb6834", "#efeee9", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right"); ax.set_yticks(range(len(corr)), corr.columns)
    ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Pearson correlation")
    ax.set_title("Correlation matrix (numeric features vs log price)", loc="left")
    _save(fig, out)


def plot_data_split(y_train, y_test, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 2.4]})
    cols = {"Train (80%)": "#2a78d6", "Test (20%)": "#eb6834"}
    ax = axes[0]; sizes = [len(y_train), len(y_test)]
    ax.bar(list(cols), sizes, color=list(cols.values()), width=0.55)
    for i, s in enumerate(sizes):
        ax.text(i, s + 600, f"{s:,} rows", ha="center", color=INK)
    ax.set_ylim(0, max(sizes) * 1.15); ax.grid(axis="x", visible=False)
    ax.set_title("80:20 train/test split", loc="left")
    ax = axes[1]
    bins = np.logspace(np.log10(min(y_train.min(), y_test.min())), np.log10(max(y_train.max(), y_test.max())), 60)
    for (lab, col), y in zip(cols.items(), (y_train, y_test)):
        ax.hist(y, bins=bins, density=True, histtype="step", lw=2, color=col,
                label=f"{lab}: median {pkr(np.median(y))}, mean {pkr(np.mean(y))}")
    ax.set_xscale("log"); ax.set_xticks([t for t in LOG_TICKS if bins[0] <= t <= bins[-1]])
    ax.xaxis.set_major_formatter(PKR); ax.xaxis.set_minor_formatter(FuncFormatter(lambda *a: ""))
    ax.set_yticks([]); ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlabel("Price (log scale)")
    ax.set_title("Train and test sets have the same price distribution", loc="left")
    _save(fig, out)


# ---------------------------------------------------------------------------
# Model results
# ---------------------------------------------------------------------------
def plot_actual_vs_predicted(y_test, preds, metrics, out):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    lo, hi = 5e5, 1e9
    for ax, n in zip(axes.ravel(), MODEL_ORDER):
        ax.scatter(y_test, preds[n], s=4, alpha=0.25, color=COLORS[n], edgecolors="none")
        ax.plot([lo, hi], [lo, hi], color=INK, lw=1.2, ls="--", label="Perfect prediction")
        r = metrics.loc[n]
        ax.text(0.03, 0.97, f"R² = {r.R2:.3f}\nRMSE = {pkr(r.RMSE)}\nMAPE = {r['MAPE (%)']:.1f}%",
                transform=ax.transAxes, va="top", fontsize=9, color=INK,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID))
        ax.set_title(n); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        _log_axes(ax, lo, hi)
    for ax in axes[1]: ax.set_xlabel("Actual price (log scale)")
    for ax in axes[:, 0]: ax.set_ylabel("Predicted price (log scale)")
    axes[0, 0].legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle(f"Actual vs predicted prices on the 20% test set (n = {len(y_test):,})",
                 fontsize=13, fontweight="bold", color=INK)
    _save(fig, out, dpi=105)


def plot_sample_predictions(y_test, preds, out, n=40, seed=1):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(y_test), n, replace=False))
    yt = np.asarray(y_test)[idx]; order = np.argsort(yt); x = np.arange(n)
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.plot(x, yt[order], color=INK, lw=2.4, marker="o", ms=5, label="Actual", zorder=5)
    for m in MODEL_ORDER:
        ax.plot(x, preds[m][idx][order], color=COLORS[m], lw=1.3, marker="o", ms=3.2, alpha=0.9, label=m)
    ax.set_yscale("log"); ax.set_yticks([t for t in LOG_TICKS if yt.min() / 2 <= t <= yt.max() * 2])
    ax.yaxis.set_major_formatter(PKR); ax.yaxis.set_minor_formatter(FuncFormatter(lambda *a: ""))
    ax.set_xlabel(f"{n} randomly chosen test properties (sorted by actual price)"); ax.set_ylabel("Price (log scale)")
    ax.legend(ncol=7, loc="upper left", frameon=False, fontsize=9)
    ax.set_title("Actual vs predicted price for individual test properties", loc="left")
    _save(fig, out)


def plot_metric_bars(metrics, out):
    panels = [("MAE", "MAE (PKR) — lower is better", "pkr"), ("RMSE", "RMSE (PKR) — lower is better", "pkr"),
              ("MSE", "MSE (PKR²) — lower is better", "sci"), ("R2", "R² score — higher is better", "r2"),
              ("MAPE (%)", "MAPE (%) — lower is better", "pct"), ("Within 10%", "Predicted within ±10% — higher is better", "pct")]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    for ax, (col, title, kind) in zip(axes.ravel(), panels):
        vals = metrics.loc[MODEL_ORDER, col]; y = np.arange(len(MODEL_ORDER))[::-1]
        ax.barh(y, vals, color=[COLORS[m] for m in MODEL_ORDER], height=0.62)
        ax.set_yticks(y, MODEL_ORDER); ax.set_title(title, loc="left"); ax.grid(axis="y", visible=False)
        vmax = vals.max()
        for yi, v in zip(y, vals):
            lab = {"pkr": pkr(v), "sci": f"{v:.2e}", "r2": f"{v:.4f}", "pct": f"{v:.1f}%"}[kind]
            ax.text(v + vmax * 0.01, yi, lab, va="center", fontsize=9, color=INK)
        ax.set_xlim(0, 1.0 if kind == "r2" else vmax * 1.2)
        if kind == "pkr": ax.xaxis.set_major_formatter(PKR)
    fig.suptitle("Test-set performance of the six regression models", fontsize=13, fontweight="bold", color=INK)
    _save(fig, out)


def plot_residuals(y_test, preds, out):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    bins = np.linspace(-100, 100, 81)
    for ax, n in zip(axes.ravel(), MODEL_ORDER):
        pe = 100 * (preds[n] - np.asarray(y_test)) / np.asarray(y_test)
        ax.hist(np.clip(pe, -100, 100), bins=bins, color=COLORS[n], edgecolor="white", linewidth=0.4)
        ax.axvline(0, color=INK, ls="--", lw=1.1)
        ax.set_title(f"{n}  (median {np.median(pe):+.1f}%, IQR {np.subtract(*np.percentile(pe, [75, 25])):.1f} pts)")
    for ax in axes[1]: ax.set_xlabel("Percentage error = (predicted − actual) / actual")
    for ax in axes[:, 0]: ax.set_ylabel("Test properties")
    fig.suptitle("Distribution of percentage errors (narrow and centred on 0 is best)",
                 fontsize=13, fontweight="bold", color=INK)
    _save(fig, out)


def plot_abs_error_box(y_test, preds, out):
    fig, ax = plt.subplots(figsize=(12, 4.8))
    data = [100 * np.abs(preds[n] - np.asarray(y_test)) / np.asarray(y_test) for n in MODEL_ORDER]
    bp = ax.boxplot(data, vert=False, patch_artist=True, showfliers=False, widths=0.55,
                    medianprops=dict(color=INK, lw=1.6), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    for patch, n in zip(bp["boxes"], MODEL_ORDER):
        patch.set_facecolor(COLORS[n]); patch.set_edgecolor("white")
    for i, d in enumerate(data, start=1):
        ax.text(np.percentile(d, 75) + 0.8, i + 0.33, f"median {np.median(d):.1f}%", fontsize=8.5, color=INK)
    ax.set_yticks(range(1, 7), MODEL_ORDER); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Absolute percentage error (outliers hidden)")
    ax.set_title("Spread of absolute percentage errors on the test set", loc="left")
    _save(fig, out)


def plot_train_vs_test(metrics, out):
    fig, ax = plt.subplots(figsize=(12, 4.8))
    x = np.arange(len(MODEL_ORDER)); w = 0.36
    tr, te = metrics.loc[MODEL_ORDER, "Train R2"].values, metrics.loc[MODEL_ORDER, "R2"].values
    ax.bar(x - w/2 - 0.01, tr, w, color="#9ec5f4", label="Train R²")
    ax.bar(x + w/2 + 0.01, te, w, color="#2a78d6", label="Test R²")
    for xi, a, b in zip(x, tr, te):
        ax.text(xi - w/2, a + 0.012, f"{a:.3f}", ha="center", fontsize=8.5, color=INK)
        ax.text(xi + w/2, b + 0.012, f"{b:.3f}", ha="center", fontsize=8.5, color=INK)
        ax.text(xi, max(a, b) + 0.065, f"gap {a-b:+.3f}", ha="center", fontsize=9, color=INK2, fontweight="bold")
    ax.set_xticks(x, MODEL_ORDER); ax.set_ylim(0, 1.18); ax.grid(axis="x", visible=False)
    ax.set_ylabel("R² score"); ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.set_title("Training vs test R² (large gap = over-fitting, both low = under-fitting)", loc="left")
    _save(fig, out)


def plot_cv(metrics, out):
    fig, ax = plt.subplots(figsize=(11, 4.4))
    m = metrics.loc[MODEL_ORDER]; y = np.arange(len(MODEL_ORDER))[::-1]
    ax.barh(y, m["CV RMSE (mean)"], xerr=m["CV RMSE (std)"], height=0.6,
            color=[COLORS[n] for n in MODEL_ORDER], error_kw=dict(ecolor=INK, capsize=4, lw=1.2))
    ax.scatter(m["RMSE"], y, marker="D", s=40, color=INK, zorder=5, label="Test-set RMSE")
    for yi, v, s in zip(y, m["CV RMSE (mean)"], m["CV RMSE (std)"]):
        ax.text(v + s + m["CV RMSE (mean)"].max() * 0.02, yi, f"{pkr(v)} ± {pkr(s)}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, MODEL_ORDER); ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(PKR); ax.set_xlim(0, m["CV RMSE (mean)"].max() * 1.35)
    ax.set_xlim(0, m["CV RMSE (mean)"].max() * 1.6)
    ax.set_xlabel("RMSE (PKR)"); ax.legend(frameon=False, loc="lower right")
    ax.set_title("5-fold cross-validation RMSE on the training set (bar = mean, line = ±1 std)", loc="left")
    _save(fig, out)


PRICE_BANDS = [0, 5e6, 1e7, 2e7, 5e7, np.inf]
BAND_LABELS = ["< 50 Lakh", "50 L – 1 Cr", "1 – 2 Cr", "2 – 5 Cr", "≥ 5 Cr"]


def error_by_group(y_test, preds, groups, order):
    y = np.asarray(y_test)
    rows = {n: pd.Series(100 * np.abs(preds[n] - y) / y).groupby(np.asarray(groups)).mean().reindex(order)
            for n in MODEL_ORDER}
    t = pd.DataFrame(rows)
    t.insert(0, "Test rows", pd.Series(np.asarray(groups)).value_counts().reindex(order).values)
    return t


def plot_error_groups(tbl, title, out):
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(tbl)); w = 0.135
    for i, n in enumerate(MODEL_ORDER):
        ax.bar(x + (i - 2.5) * (w + 0.008), tbl[n], w, color=COLORS[n], label=n)
    ax.set_xticks(x, [f"{i}\n(n = {c:,})" for i, c in zip(tbl.index, tbl["Test rows"])])
    ax.grid(axis="x", visible=False); ax.set_ylabel("MAPE (%)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(ncol=6, frameon=False, loc="upper left", fontsize=9)
    ax.set_ylim(0, tbl[MODEL_ORDER].values.max() * 1.2)
    ax.set_title(title, loc="left")
    _save(fig, out)


def plot_feature_importance(imps, out, top=14):
    fig, axes = plt.subplots(1, len(imps), figsize=(15, 5.8))
    for ax, (name, s) in zip(np.atleast_1d(axes), imps.items()):
        s = s.sort_values().tail(top)
        ax.barh(s.index, s.values, color=COLORS[name], height=0.62)
        ax.set_title(f"{name} — feature importance", loc="left"); ax.grid(axis="y", visible=False)
    _save(fig, out)


def plot_knn_k(ks, scores, best_k, out):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(ks, scores, color=COLORS["KNN"], lw=2, marker="o", ms=5)
    ax.axvline(best_k, color=INK2, ls="--", lw=1)
    ax.annotate(f"best k = {best_k}", (best_k, min(scores)), xytext=(8, 12), textcoords="offset points", color=INK)
    ax.set_xlabel("Number of neighbours k"); ax.set_ylabel("5-fold CV RMSE of log(price)")
    ax.set_title("Choosing k for KNN (training data only)", loc="left")
    _save(fig, out)


def plot_nn_loss(mlp, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    axes[0].plot(np.arange(1, len(mlp.loss_curve_) + 1), mlp.loss_curve_, color=COLORS["Neural Network"], lw=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training loss (scaled log price)")
    axes[0].set_title("Neural network training loss", loc="left")
    vs = mlp.validation_scores_; best = int(np.argmax(vs)) + 1
    axes[1].plot(np.arange(1, len(vs) + 1), vs, color=COLORS["Neural Network"], lw=2)
    axes[1].axvline(best, color=INK2, ls="--", lw=1)
    axes[1].text(0.97, 0.06, f"best epoch {best}, validation R² = {max(vs):.3f}\n(stopped after {len(vs)} epochs)",
                 transform=axes[1].transAxes, ha="right", color=INK, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID))
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation R² (10% of training set)")
    axes[1].set_title("Early stopping keeps the best epoch", loc="left")
    _save(fig, out)


def plot_linear_coefficients(coefs, out):
    s = coefs.sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(s.index, s.values, color=["#eb6834" if v < 0 else "#2a78d6" for v in s.values], height=0.62)
    for i, v in enumerate(s.values):
        ax.text(v + (0.03 if v >= 0 else -0.03), i, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right",
                fontsize=8.5, color=INK)
    ax.axvline(0, color=INK2, lw=1); ax.grid(axis="y", visible=False)
    lim = np.abs(s.values).max() * 1.2; ax.set_xlim(-lim, lim)
    ax.set_xlabel("Coefficient on log(price): +0.10 ≈ +10.5% price per +1 std (numeric) or vs. House/Karachi (categories)")
    ax.set_title("Linear Regression coefficients (blue raises, orange lowers the price)", loc="left")
    _save(fig, out)


def plot_tree_top(tree, describe_split, out, depth=3):
    """describe_split(feature_index, threshold) -> human-readable condition in original units."""
    t = tree.tree_
    fig, ax = plt.subplots(figsize=(16, 7.5)); ax.set_axis_off()
    vals = np.exp(t.value[:, 0, 0]); vmin, vmax = np.log(vals.min()), np.log(vals.max())

    def draw(node, x, y, dx, level):
        val, n = vals[node], t.n_node_samples[node]
        leaf = level == depth or t.children_left[node] == -1
        txt = ("" if leaf else describe_split(t.feature[node], t.threshold[node]) + "\n") + f"n = {n:,}\nprice ≈ {pkr(val)}"
        shade = (np.log(val) - vmin) / (vmax - vmin)
        ax.text(x, y, txt, ha="center", va="center", fontsize=10, color="white" if shade > 0.55 else INK,
                bbox=dict(boxstyle="round,pad=0.5", fc=SEQ(0.15 + 0.85 * shade), ec="white", lw=1.5))
        if leaf:
            return
        for child, sign in ((t.children_left[node], -1), (t.children_right[node], 1)):
            cx, cy = x + sign * dx, y - 1
            ax.plot([x, cx], [y - 0.22, cy + 0.25], color=INK2, lw=1, zorder=0)
            ax.text((x + cx) / 2 + sign * 0.15, (y + cy) / 2 + 0.05, "yes" if sign < 0 else "no", fontsize=9, color=INK2, ha="center")
            draw(child, cx, cy, dx / 2, level + 1)

    draw(0, 0, 0, 2 ** (depth - 1) * 1.05, 0)
    ax.set_xlim(-2 ** depth * 1.08, 2 ** depth * 1.08); ax.set_ylim(-depth - 0.6, 0.6)
    ax.set_title(f"Decision tree — first {depth} levels of {tree.get_depth()} ({tree.get_n_leaves():,} leaves). "
                 "Darker = more expensive; 'price ≈' = typical price in that node.", loc="left", fontsize=12)
    _save(fig, out)
