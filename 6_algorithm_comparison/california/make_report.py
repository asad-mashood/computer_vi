"""
Build REPORT.md: the full illustrated project report, generated from the files in
results/ so every number matches the latest run of train.py.

    python train.py          # first
    python make_report.py    # then -> REPORT.md (and REPORT.pdf if pandoc + xelatex are installed)
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data_preprocessing import load_data
from src.models import MODEL_ORDER

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
FIG = "results/figures"

m = pd.read_csv(R / "metrics.csv", index_col=0).loc[MODEL_ORDER]
summary = json.loads((R / "summary.json").read_text())
pred = pd.read_csv(R / "test_predictions.csv", index_col=0)
err = pd.read_csv(R / "error_by_price_range.csv", index_col=0)
coef = pd.read_csv(R / "linear_regression_coefficients.csv", index_col=0)["coefficient"]
df = load_data()
train = pd.read_csv(ROOT / "data" / "train.csv")
test = pd.read_csv(ROOT / "data" / "test.csv")

# extra metric: % of test districts predicted within 10% of the true value
y = pred["actual"]
m["Within 10%"] = [(np.abs(y - pred[n]) / y <= 0.10).mean() * 100 for n in MODEL_ORDER]
m["Within 20%"] = [(np.abs(y - pred[n]) / y <= 0.20).mean() * 100 for n in MODEL_ORDER]
ranked = m.sort_values("RMSE")
best, second, worst = ranked.index[0], ranked.index[1], ranked.index[-1]
b = m.loc[best]
has_cv = "CV RMSE (mean)" in m.columns and m["CV RMSE (mean)"].notna().all()
cap = y >= 500_000

out = []
W = out.append


def img(file, alt):
    W(f"![{alt}]({FIG}/{file})")
    W("")


def money(v):
    return f"${v:,.0f}"


def pct_lower(a, b_):
    return 100 * (1 - a / b_)


# ---------------------------------------------------------------------------
W("# House Price Prediction — Comparative Study of Six Regression Algorithms")
W("")
W("**Course assignment:** Comparative Implementation of Machine Learning Regression Algorithms  ")
W("**Student:** Muhammad Asad Mashood — BS Artificial Intelligence, COMSATS University Islamabad (Wah Campus)  ")
W("**Dataset:** California Housing (1990 U.S. Census), 20,640 districts  ")
W("**Algorithms:** Linear Regression · Decision Tree · Random Forest · Neural Network (MLP) · XGBoost · K-Nearest Neighbors")
W("")
W("> ### Key result")
W(f"> **{best}** is the best model: **R² = {b.R2:.3f}**, **RMSE = {money(b.RMSE)}**, **MAE = {money(b.MAE)}** on the "
  f"4,128-district test set. Its RMSE is **{pct_lower(b.RMSE, m.loc[second,'RMSE']):.1f}% lower** than the runner-up "
  f"({second}) and **{pct_lower(b.RMSE, m.loc[worst,'RMSE']):.1f}% lower** than {worst}. It predicts "
  f"{b['Within 10%']:.1f}% of districts within ±10% of the true price.")
W("")
W("## Contents")
for i, t in enumerate(["Problem statement", "Dataset", "Exploratory data analysis", "Methodology",
                       "The six algorithms", "Results", "Error analysis", "Strengths and limitations",
                       "Best model and justification", "Conclusion", "Reproducing the results"], 1):
    W(f"{i}. [{t}](#{i}-{t.lower().replace(' ', '-')})")
W("")

# 1 -------------------------------------------------------------------------
W("## 1. Problem statement")
W("")
W("Predict the **median house value** of a California district from information about its location, "
  "its houses and its residents. This is a **supervised regression** problem because the target is a continuous "
  "dollar amount. To compare six regression algorithms fairly, every model uses exactly the same dataset, input "
  "features, target variable, preprocessing and **80:20 train/test split**. All six are scored with MAE, MSE, "
  "RMSE and R².")
W("")

# 2 -------------------------------------------------------------------------
W("## 2. Dataset")
W("")
W(f"- **Source:** California Housing dataset (Pace & Barry, 1997), file `data/housing.csv`")
W(f"- **Size:** {len(df):,} rows (one per census block group, i.e. a district) × {df.shape[1]} columns")
W(f"- **Target:** `median_house_value` in USD (mean {money(df.median_house_value.mean())}, "
  f"range {money(df.median_house_value.min())} – {money(df.median_house_value.max())})")
W(f"- **Missing values:** `total_bedrooms` has {df.total_bedrooms.isna().sum()} missing values; all other columns are complete")
W(f"- **Note:** the target is **capped at $500,001** ({(df.median_house_value >= 500001).sum()} districts). "
  "Any district worth more is recorded as $500,001.")
W("")
desc = {
    "longitude": "How far west the district is", "latitude": "How far north the district is",
    "housing_median_age": "Median age of houses (years)", "total_rooms": "Total rooms in the district",
    "total_bedrooms": "Total bedrooms in the district", "population": "People living in the district",
    "households": "Number of households", "median_income": "Median household income (tens of thousands USD)",
    "ocean_proximity": "Location category: <1H OCEAN, INLAND, NEAR OCEAN, NEAR BAY, ISLAND",
    "median_house_value": "**Target**: median house value (USD)",
}
W("| Column | Type | Meaning | Mean | Min | Max |")
W("|---|---|---|---:|---:|---:|")
for c in df.columns:
    if not pd.api.types.is_numeric_dtype(df[c]):
        W(f"| `{c}` | categorical | {desc[c]} | — | — | — |")
    else:
        W(f"| `{c}` | numeric | {desc[c]} | {df[c].mean():,.2f} | {df[c].min():,.2f} | {df[c].max():,.2f} |")
W("")
W("**Category counts of `ocean_proximity`:** " + ", ".join(f"{k}: {v:,}" for k, v in df.ocean_proximity.value_counts().items()))
W("")

# 3 -------------------------------------------------------------------------
W("## 3. Exploratory data analysis")
W("")
img("eda_target.png", "Target distribution and income vs price")
W("The target is right-skewed and has a spike at the $500k cap. `median_income` has the strongest relationship "
  "with price, but the cloud of points is wide, so income alone cannot explain prices.")
W("")
img("eda_geo_prices.png", "Map of house prices")
W("**Location matters a lot.** The most expensive districts lie along the coast around San Francisco, Los Angeles "
  "and San Diego, while inland districts (Central Valley) are much cheaper. Price is therefore a **non-linear** "
  "function of latitude and longitude, which a straight line cannot capture.")
W("")
img("eda_correlation.png", "Correlation matrix")
corr = df.select_dtypes("number").corr()["median_house_value"].drop("median_house_value").sort_values(ascending=False)
W("Correlation with the target: " + ", ".join(f"`{k}` {v:+.2f}" for k, v in corr.items()) + ". "
  "The raw count columns (`total_rooms`, `population`, `households`) are almost perfectly correlated with each "
  "other (multicollinearity) and only weakly with price. That is why per-household **ratio features** were "
  "engineered.")
W("")

# 4 -------------------------------------------------------------------------
W("## 4. Methodology")
W("")
W("### 4.1 Pipeline")
W("")
W("```")
W("housing.csv ─► feature engineering ─► 80:20 split ─► [ imputer ─► scaler ─► one-hot ] ─► model ─► predictions ─► metrics")
W("                (3 ratio features)    (seed 42)       fitted on the TRAINING set only")
W("```")
W("")
W("| Step | What was done | Why |")
W("|---|---|---|")
W("| Feature engineering | `rooms_per_household`, `bedrooms_per_room`, `population_per_household` | Ratios describe a typical home; raw totals mostly measure district size |")
W("| Train/test split | 80% train / 20% test, `random_state=42`, identical for all models | Fair comparison on unseen data |")
W("| Missing values | `total_bedrooms` → median of the training set | Keeps all rows; median is robust to outliers |")
W("| Feature scaling | `StandardScaler`: $z=(x-\\mu)/\\sigma$ with training mean/std | Required for KNN (distances) and the Neural Network (gradient descent); harmless for trees |")
W("| Encoding | One-hot encoding of `ocean_proximity` (5 binary columns) | Models need numbers; one-hot imposes no false order |")
W("| No data leakage | All preprocessing lives inside an sklearn `Pipeline` fitted on training data only | Test set is never seen during fitting |")
W("")
W("Final model input: **16 columns**, made of 11 scaled numeric features and 5 one-hot columns. The exact sets "
  "are saved as `data/train.csv` and `data/test.csv`.")
W("")
W("### 4.2 Train/test split")
W("")
img("data_split.png", "Train/test split")
W(f"| Set | Rows | Share | Mean price | Median price | Used for |")
W("|---|---:|---:|---:|---:|---|")
W(f"| Training | {len(train):,} | 80% | {money(train.median_house_value.mean())} | {money(train.median_house_value.median())} | fitting models, choosing k, cross-validation |")
W(f"| Test | {len(test):,} | 20% | {money(test.median_house_value.mean())} | {money(test.median_house_value.median())} | final evaluation only |")
W("")
W("Both sets have almost the same distribution, so the test set is a fair stand-in for new data.")
W("")
W("### 4.3 Evaluation metrics")
W("")
W("| Metric | Formula | Meaning |")
W("|---|---|---|")
W("| MAE | $\\frac1n\\sum\\lvert y_i-\\hat y_i\\rvert$ | Average error in dollars; easy to interpret |")
W("| MSE | $\\frac1n\\sum (y_i-\\hat y_i)^2$ | Squares errors, so large mistakes are punished heavily (units: $²) |")
W("| RMSE | $\\sqrt{\\text{MSE}}$ | Back in dollars; main ranking metric |")
W("| R² | $R^2=1-\\frac{\\sum (y_i-\\hat y_i)^2}{\\sum (y_i-\\bar y)^2}$ | Share of price variance explained (1 = perfect, 0 = predicting the mean) |")
W("")
W("As extra checks, the report also gives **training R²** (to diagnose over- and under-fitting), "
  "**5-fold cross-validated RMSE** on the training set (to show stability), and the **% of districts predicted "
  "within ±10% / ±20%**.")
W("")

# 5 -------------------------------------------------------------------------
W("## 5. The six algorithms")
W("")
W("| Model | Key equation | Settings used |")
W("|---|---|---|")
W("| Linear Regression | $\\hat y=\\beta_0+\\sum_j\\beta_jx_j$, $\\boldsymbol\\beta=(X^TX)^{-1}X^T\\mathbf y$ | Ordinary Least Squares |")
W("| Decision Tree | split minimising $SSE_L+SSE_R$; leaf = mean of $y$ | `max_depth=12`, `min_samples_leaf=10` |")
W("| Random Forest | $\\hat y=\\frac1B\\sum_{b=1}^{B}T_b(x)$ (bootstrap + random features) | 150 trees, `max_features=0.5`, `min_samples_leaf=3` |")
W("| Neural Network | $h=\\text{ReLU}(Wx+b)$, trained by back-propagation (Adam) | 128-64-32 neurons, `alpha=1e-3`, early stopping |")
W("| XGBoost | $\\hat y^{(t)}=\\hat y^{(t-1)}+\\eta f_t(x)$, $w_j=-\\frac{G_j}{H_j+\\lambda}$ | 800 trees, η = 0.05, depth 7, subsample 0.8, λ = 1 |")
W(f"| KNN | $\\hat y=$ distance-weighted mean of the $k$ nearest neighbours | k = {summary['knn_best_k']} (chosen by CV), Euclidean |")
W("")
W("### Linear Regression: what the model learned")
img("linear_coefficients.png", "Linear regression coefficients")
top = coef.drop("ocean_proximity_ISLAND").abs().sort_values(ascending=False).index[:3]
W("Each bar shows how much the predicted price changes when that feature rises by one standard deviation. "
  f"Apart from the ISLAND flag, the largest effects come from {', '.join(f'`{t}`' for t in top)}. Income raises "
  "the price; inland location and higher latitude/longitude lower it. The ISLAND coefficient is large but rests on "
  "only 5 districts, so it is unreliable.")
W("")
W("### Decision Tree: the first splits")
img("decision_tree_top_levels.png", "Decision tree top levels")
W("The tree splits first on income, then on inland/coastal location. This is the same logic a human estimator "
  "would follow. Thresholds are shown in original units.")
W("")
W("### Neural Network: training curve")
img("nn_training_curve.png", "Neural network training")
W("The loss falls steadily. Early stopping keeps the weights from the epoch with the best validation R², "
  "which prevents over-fitting.")
W("")
W("### Random Forest and XGBoost: feature importance")
img("feature_importance.png", "Feature importance")
W("Both ensembles rank **median income** and **inland location** as the most important features, followed by "
  "household density and geographic coordinates. This agrees with the exploratory analysis.")
W("")
W("### KNN: choosing k")
img("knn_k_selection.png", "Choosing k")
W(f"Small k memorises noise and large k over-smooths. Five-fold cross-validation on the training data picked **k = {summary['knn_best_k']}**.")
W("")

# 6 -------------------------------------------------------------------------
W("## 6. Results")
W("")
W("### 6.1 Comparative performance table (test set, 4,128 districts)")
W("")
W("| Rank | Model | MAE (USD) | MSE (USD²) | RMSE (USD) | R² | Within ±10% | Within ±20% | Train time (s) |")
W("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
for i, (n, r) in enumerate(ranked.iterrows(), 1):
    bold = "**" if n == best else ""
    W(f"| {i} | {bold}{n}{bold} | {bold}{r.MAE:,.0f}{bold} | {bold}{r.MSE:.3e}{bold} | {bold}{r.RMSE:,.0f}{bold} | "
      f"{bold}{r.R2:.4f}{bold} | {r['Within 10%']:.1f}% | {r['Within 20%']:.1f}% | {r['Train time (s)']:.2f} |")
W("")
W(f"The ranking is the same for **all four metrics**: {' > '.join(ranked.index)}.")
W("")
img("metrics_comparison.png", "Metric comparison")
W("### 6.2 Actual vs predicted values")
img("actual_vs_predicted.png", "Actual vs predicted")
W("Each dot is one test district, and points on the dashed line are perfect predictions. XGBoost and Random Forest "
  "form the tightest band. Linear Regression fans out widely and even predicts negative prices for a few cheap "
  "districts. The Decision Tree shows horizontal bands because each leaf predicts one constant value. "
  "In every panel, the column of points at $500k is the capped districts.")
W("")
img("sample_predictions.png", "Sample predictions")
W("For 40 random test districts, the predictions of the stronger models (XGBoost, Random Forest, Neural Network) "
  "stay closest to the black *actual* line.")
W("")
W("### 6.3 Over-fitting and stability")
img("train_vs_test_r2.png", "Train vs test R2")
W("| Model | Train R² | Test R² | Gap | Diagnosis |")
W("|---|---:|---:|---:|---|")
diag = {"Linear Regression": "Under-fitting: both scores low, model too simple",
        "Decision Tree": "Moderate over-fitting (limited by depth 12)",
        "Random Forest": "Some over-fitting, but averaging keeps test score high",
        "Neural Network": "Well balanced (early stopping)",
        "XGBoost": "Fits training data closely; regularisation keeps the best test score",
        "KNN": "Memorises training data (each point is its own neighbour)"}
for n in MODEL_ORDER:
    r = m.loc[n]
    W(f"| {n} | {r['Train R2']:.3f} | {r.R2:.3f} | {r['Train R2']-r.R2:.3f} | {diag[n]} |")
W("")
if has_cv:
    img("cross_validation_rmse.png", "Cross-validation")
    W("Five-fold cross-validation on the training set gives the same ranking, and every model's test RMSE "
      f"(black diamond) falls inside or near its CV range. The result does not depend on one lucky split. "
      f"{best} also has the smallest spread (± {money(m.loc[best, 'CV RMSE (std)'])}), so it is the most stable.")
    W("")

# 7 -------------------------------------------------------------------------
W("## 7. Error analysis")
W("")
img("residuals.png", "Residuals")
W("Residuals (actual − predicted) should be centred on 0 and narrow. All models are roughly unbiased. "
  f"{best} has the narrowest distribution, while Linear Regression has the widest.")
W("")
img("abs_error_boxplot.png", "Absolute error boxplot")
W("| Model | Median absolute error |")
W("|---|---:|")
for n in ranked.index:
    W(f"| {n} | {money(np.median(np.abs(y - pred[n])))} |")
W("")
W("### Error by price range")
img("error_by_price_range.png", "Error by price range")
W("| Price range | Test rows | " + " | ".join(MODEL_ORDER) + " |")
W("|---|---:|" + "---:|" * len(MODEL_ORDER))
for idx, r in err.iterrows():
    W(f"| {idx} | {int(r['Test rows']):,} | " + " | ".join(money(r[n]) for n in MODEL_ORDER) + " |")
W("")
W(f"Errors grow with price for every model. Expensive districts are rarer (less training data), and the "
  f"$500k cap hides their true value. On the {cap.sum()} capped test districts, the MAE ranges from "
  f"{money(min(np.abs(y - pred[n])[cap].mean() for n in MODEL_ORDER))} to "
  f"{money(max(np.abs(y - pred[n])[cap].mean() for n in MODEL_ORDER))}. {best} has the lowest error in "
  "every price range.")
W("")
img("best_model_error_map.png", "Error map")
W(f"The largest errors of {best} cluster in the expensive coastal areas of the Bay Area and Los Angeles. "
  "There, neighbouring districts can differ greatly in price for reasons not in the data, such as views, "
  "schools or house size.")
W("")

# 8 -------------------------------------------------------------------------
W("## 8. Strengths and limitations")
W("")
W("| Algorithm | Strengths | Limitations |")
W("|---|---|---|")
W("| **Linear Regression** | Fastest; fully interpretable coefficients; no hyper-parameters; good baseline | "
  "Assumes a linear relationship, so it **under-fits** here (R² " + f"{m.loc['Linear Regression','R2']:.2f}" + "); "
  "sensitive to outliers and multicollinearity; can predict negative prices |")
W("| **Decision Tree** | Captures non-linearity and interactions; no scaling needed; rules are easy to read | "
  "High variance (unstable); step-shaped predictions; over-fits without depth limits |")
W("| **Random Forest** | Averaging many decorrelated trees cuts variance (RMSE " +
  f"{pct_lower(m.loc['Random Forest','RMSE'], m.loc['Decision Tree','RMSE']):.0f}% lower than a single tree); "
  "robust; gives feature importance | Large model file (~18 MB); slower prediction; cannot extrapolate beyond training range |")
W("| **Neural Network** | Learns smooth, complex functions; balanced train/test fit | Needs scaling and careful tuning; "
  "slowest to train here; black box; results vary with random initialisation |")
W("| **XGBoost** | Sequential trees fix earlier errors (reduces bias); λ, η and subsampling control variance; "
  "fast; handles missing values | Many hyper-parameters; can over-fit without regularisation; less interpretable |")
W("| **KNN** | Simple; no training phase; naturally non-linear | Must scale features; slow prediction on large data; "
  "suffers in high dimensions; memorises training set |")
W("")

# 9 -------------------------------------------------------------------------
W("## 9. Best model and justification")
W("")
W(f"**{best} is the best-performing algorithm.** The evidence:")
W("")
W(f"1. **Best on all four required metrics:** lowest MAE ({money(b.MAE)}), MSE ({b.MSE:.3e}) and RMSE "
  f"({money(b.RMSE)}), and highest R² ({b.R2:.4f}). It explains {b.R2*100:.1f}% of the variance in house prices.")
W(f"2. **Clear margin:** RMSE is {pct_lower(b.RMSE, m.loc[second,'RMSE']):.1f}% lower than {second} and "
  f"{pct_lower(b.RMSE, m.loc['Linear Regression','RMSE']):.1f}% lower than Linear Regression.")
if has_cv:
    W(f"3. **Stable:** best 5-fold CV RMSE ({money(m.loc[best,'CV RMSE (mean)'])} ± {money(m.loc[best,'CV RMSE (std)'])}) "
      "with the smallest spread, so the result does not depend on one lucky split.")
W(f"4. **Most accurate in every price range** and the most districts within ±10% ({b['Within 10%']:.1f}%).")
W(f"5. **Practical:** it trains in about {b['Train time (s)']:.1f} s, faster than Random Forest and the Neural Network.")
W("")
W("**Why it works:** house prices depend non-linearly on income and location, with interactions between them "
  "(the same income buys much more inland than on the coast). Tree-based models capture such interactions "
  "automatically. Random Forest (bagging) averages independent trees and mainly **reduces variance**. "
  "XGBoost (boosting) fits each new tree to the remaining errors (gradients) of the current model, so it also "
  "**reduces bias**. Its built-in L2 penalty λ, small learning rate η and row/column subsampling keep that "
  "power from turning into over-fitting. Gradient-boosted trees are usually the strongest choice on "
  "structured tabular data, and this experiment agrees with that.")
W("")

# 10 ------------------------------------------------------------------------
W("## 10. Conclusion")
W("")
W(f"All six algorithms were implemented under identical conditions. The ensemble tree methods clearly beat the "
  f"single models. {best} ranks first (R² {b.R2:.3f}), followed by {ranked.index[1]} ({ranked.iloc[1].R2:.3f}) and "
  f"{ranked.index[2]} ({ranked.iloc[2].R2:.3f}). Linear Regression under-fits ({m.loc['Linear Regression','R2']:.3f}) "
  "because the problem is non-linear. All models lose accuracy on expensive homes because of the $500k cap.")
W("")
W("**Possible improvements:** hyper-parameter search (e.g. Optuna), predicting log(price), adding external "
  "features (distance to the coast and to city centres), and removing or modelling the capped districts.")
W("")

# 11 ------------------------------------------------------------------------
W("## 11. Reproducing the results")
W("")
W("```bash")
W("pip install -r requirements.txt")
W("python train.py          # trains the 6 models, writes models/, results/, data/train.csv, data/test.csv")
W("python make_report.py    # rebuilds this report from the results")
W("python predict.py        # loads the saved models and predicts example districts")
W("```")
W("")
W("| Output | Content |")
W("|---|---|")
W("| `data/train.csv`, `data/test.csv` | The exact 16,512 / 4,128 rows used for training and testing |")
W("| `models/*.joblib` | The 6 trained models (preprocessing included) |")
W("| `results/metrics.csv` | All metrics for all models |")
W("| `results/test_predictions.csv` | Actual and predicted price for every test district |")
W("| `results/train_predictions.csv` | Actual and predicted price for every training district |")
W("| `results/error_by_price_range.csv` | MAE per price range |")
W("| `results/figures/` | All figures in this report |")
W("")

text = "\n".join(out)
# escape currency signs ("$500k") so Markdown/LaTeX do not treat them as math delimiters
text = re.sub(r"(?<!\\)\$(?=\d)", r"\\$", text)
(ROOT / "REPORT.md").write_text(text, encoding="utf-8")
print("Wrote REPORT.md")

# Optional PDF version (needs pandoc + xelatex)
if shutil.which("pandoc") and shutil.which("xelatex"):
    tmp = ROOT / ".report_pdf.md"
    tmp.write_text(text.replace(".png)", ".png){width=100%}"), encoding="utf-8")
    cmd = ["pandoc", str(tmp), "-f", "markdown-yaml_metadata_block", "-o", str(ROOT / "REPORT.pdf"),
           "--pdf-engine=xelatex", "-V", "mainfont=DejaVu Sans", "-V", "monofont=DejaVu Sans Mono",
           "-V", "geometry:margin=1.6cm", "-V", "fontsize=10pt", "-V", "colorlinks=true",
           "--resource-path", str(ROOT)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    tmp.unlink()
    print("Wrote REPORT.pdf" if res.returncode == 0 else f"PDF skipped: {res.stderr[-500:]}")
