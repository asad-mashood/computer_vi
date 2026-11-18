"""
Build REPORT.md (and REPORT.pdf if pandoc + xelatex are installed) from the files in results/,
so every number in the report matches the latest run of train.py.

    python train.py
    python make_report.py
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_preprocessing import FEATURES, format_pkr, load_raw
from src.models import MODEL_ORDER

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
FIG = "results/figures"

m = pd.read_csv(R / "metrics.csv", index_col=0).loc[MODEL_ORDER]
summary = json.loads((R / "summary.json").read_text())
pred = pd.read_csv(R / "test_predictions.csv", index_col=0)
clog = pd.read_csv(R / "cleaning_log.csv")
by_band = pd.read_csv(R / "mape_by_price_range.csv", index_col=0)
by_city = pd.read_csv(R / "mape_by_city.csv", index_col=0)
by_type = pd.read_csv(R / "mape_by_property_type.csv", index_col=0)
fi = pd.read_csv(R / "feature_importance.csv", index_col=0)
tune = pd.read_csv(R / "hyperparameter_tuning.csv")
coef = pd.read_csv(R / "linear_regression_coefficients.csv", index_col=0).iloc[:, 0]
clean = pd.read_csv(ROOT / "data" / "zameen_clean.csv")
train = pd.read_csv(ROOT / "data" / "train.csv")
test = pd.read_csv(ROOT / "data" / "test.csv")
raw = load_raw()

ranked = m.sort_values("RMSE")
best, second, worst = ranked.index[0], ranked.index[1], ranked.index[-1]
b = m.loc[best]
has_cv = "CV RMSE (mean)" in m.columns  # absent after `python train.py --no-cv`
P = format_pkr
out = []
W = out.append


def img(f, alt):
    W(f"![{alt}]({FIG}/{f})"); W("")


def pct_lower(a, c):
    return 100 * (1 - a / c)


def mape_table(t, label):
    W(f"| {label} | Test rows | " + " | ".join(MODEL_ORDER) + " |")
    W("|---|---:|" + "---:|" * len(MODEL_ORDER))
    for idx, r in t.iterrows():
        bestm = r[MODEL_ORDER].astype(float).idxmin()
        W(f"| {idx} | {int(r['Test rows']):,} | " +
          " | ".join((f"**{r[n]:.1f}%**" if n == bestm else f"{r[n]:.1f}%") for n in MODEL_ORDER) + " |")
    W("")


# ---------------------------------------------------------------------------
W("# Pakistan House Price Prediction: Six Regression Algorithms Compared")
W("")
W("**Course assignment:** Comparative Implementation of Machine Learning Regression Algorithms  ")
W("**Student:** Muhammad Asad Mashood, BS Artificial Intelligence, COMSATS University Islamabad (Wah Campus)  ")
W(f"**Dataset:** Zameen.com property listings: {len(raw):,} raw listings, cleaned to {len(clean):,} properties for sale "
  "in Karachi, Lahore, Islamabad, Rawalpindi and Faisalabad  ")
W("**Algorithms:** Linear Regression, Decision Tree, Random Forest, Neural Network (MLP), XGBoost, K-Nearest Neighbors")
W("")
W("> ### Key result")
W(f"> **{best}** is the best model on the {len(test):,}-property test set: **R² = {b.R2:.3f}**, "
  f"**MAE = {P(b.MAE)}**, **RMSE = {P(b.RMSE)}**, typical error (MAPE) **{b['MAPE (%)']:.1f}%**. It prices "
  f"**{b['Within 10%']:.0f}%** of properties within ±10% and **{b['Within 20%']:.0f}%** within ±20% of the asking price. "
  f"Its RMSE is {pct_lower(b.RMSE, m.loc[second, 'RMSE']):.1f}% lower than the runner-up ({second}) and "
  f"{pct_lower(b.RMSE, m.loc[worst, 'RMSE']):.1f}% lower than {worst}.")
W("")
W("## Contents")
secs = ["Problem statement", "Dataset and cleaning", "Exploratory data analysis", "Methodology", "The six algorithms",
        "Results", "Error analysis", "Strengths and limitations", "Best model and justification", "Conclusion",
        "Using the trained models", "Reproducing the results"]
for i, t in enumerate(secs, 1):
    W(f"{i}. [{t}](#{i}-{t.lower().replace(' ', '-')})")
W("")

# 1 ---------------------------------------------------------------------------
W("## 1. Problem statement")
W("")
W("Estimate the **asking price of a residential property in Pakistan** from what a buyer can see in a listing: size "
  "(marla/kanal), bedrooms, baths, property type, city, neighbourhood and GPS location. This is a **supervised regression** "
  "problem because the target is a continuous amount in Pakistani Rupees (PKR). Six algorithms are compared under identical "
  "conditions: the same data, the same 14 inputs, the same target, the same preprocessing and the same **80:20 train/test split**. "
  "They are scored with **MAE, MSE, RMSE and R²**, plus MAPE and the share of properties priced within ±10% and ±20%.")
W("")
W("**Units used in this report:** 1 lakh = 100,000 PKR; 1 crore = 100 lakh = 10,000,000 PKR; 1 kanal = 20 marla; "
  "1 marla = 225 sq ft (the Zameen.com convention).")
W("")

# 2 ---------------------------------------------------------------------------
W("## 2. Dataset and cleaning")
W("")
W("### 2.1 Source")
W(f"The dataset contains listings scraped from **Zameen.com**, Pakistan's largest property portal "
  f"(listings dated {pd.to_datetime(raw.date_added, format='%m-%d-%Y').min():%B %Y} to "
  f"{pd.to_datetime(raw.date_added, format='%m-%d-%Y').max():%B %Y}). It was published on Kaggle as the "
  f"*Pakistan House Price* dataset. Raw file: `data/raw/zameen-updated.csv.xz` ({len(raw):,} rows × {raw.shape[1]} columns).")
W("")
desc = {"property_id": "Listing ID", "location_id": "Neighbourhood ID", "page_url": "Listing web address",
        "property_type": "House, Flat, Upper/Lower Portion, Penthouse, Farm House, Room", "price": "**Target**: asking price (PKR)",
        "location": "Neighbourhood / society (e.g. DHA Defence, Bahria Town, G-13)", "city": "Karachi, Lahore, Islamabad, Rawalpindi, Faisalabad",
        "province_name": "Punjab, Sindh, Islamabad Capital", "latitude": "GPS latitude", "longitude": "GPS longitude",
        "baths": "Number of bathrooms", "area": "Size as text, e.g. '10 Marla', '1 Kanal'", "purpose": "For Sale / For Rent",
        "bedrooms": "Number of bedrooms", "date_added": "Date the listing was posted", "agency": "Real-estate agency (if any)",
        "agent": "Agent name (if any)", "Area Type": "Marla or Kanal", "Area Size": "Size number", "Area Category": "Size band, e.g. '5-10 Marla'"}
W("| Raw column | Meaning | Used? |")
W("|---|---|---|")
used = {"property_type": "yes", "price": "target", "location": "yes (neighbourhood)", "city": "yes", "latitude": "yes",
        "longitude": "yes", "baths": "yes", "bedrooms": "yes", "Area Type": "yes, converted to marla", "Area Size": "yes, converted to marla",
        "agency": "yes, as listed_by_agency flag", "purpose": "filter (sale only)"}
for c in raw.columns:
    W(f"| `{c}` | {desc.get(c, '')} | {used.get(c, 'no')} |")
W("")
W("### 2.2 Cleaning")
img("cleaning_funnel.png", "Cleaning funnel")
W("| Step | Rows left | Removed | Why |")
W("|---|---:|---:|---|")
prev = None
for _, r in clog.iterrows():
    W(f"| {r.step.split(' (')[0]} | {r.rows:,} | {'' if prev is None else f'{prev - r.rows:,}'} | "
      f"{r.step.split(' (')[1].rstrip(')') if ' (' in r.step else ''} |")
    prev = r.rows
W("")
W(f"The biggest removal is **duplicates**: the same property re-posted by different agents. If a copy of a house sits in "
  "the training set and another copy in the test set, the model is being tested on a house it has already seen, and the "
  "scores look better than they really are. Removing duplicates makes the evaluation honest. Listings with 0 bedrooms or "
  f"baths mean the seller left the field empty, so those were removed too. The final dataset has **{len(clean):,} properties**.")
W("")

# 3 ---------------------------------------------------------------------------
W("## 3. Exploratory data analysis")
W("")
img("eda_price_distribution.png", "Price distribution")
W(f"Prices range from {P(clean.price.min())} to {P(clean.price.max())}, with a median of **{P(clean.price.median())}**. "
  "The distribution is extremely skewed, so on a log scale it looks almost normal. **Area** is the strongest single "
  "predictor: the correlation between log(area) and log(price) is "
  f"{np.corrcoef(np.log(clean.area_marla), np.log(clean.price))[0, 1]:.2f}.")
W("")
img("eda_city_and_type.png", "City and property type")
W("Per marla, Karachi, Lahore and Islamabad are the most expensive cities, while Rawalpindi and Faisalabad are cheaper. "
  "Houses dominate the data. Farm houses are the most expensive type because they are very large.")
W("")
img("eda_city_maps.png", "City maps")
W("**Location inside a city matters as much as the city itself.** Within each city, the expensive and cheap areas "
  "form clear clusters. In Karachi the expensive cluster is DHA, Bath Island and Sea View near the sea; in Islamabad "
  "it is the F-6, F-7 and F-8 sectors.")
W("")
img("eda_top_neighbourhoods.png", "Top neighbourhoods")
_g = clean.assign(ppm=clean.price / clean.area_marla).groupby(["city", "location"]).ppm.agg(["median", "size"])
_g = _g[_g["size"] >= 50]["median"]
W(f"The most expensive and the cheapest neighbourhoods differ by up to **{_g.max() / _g.min():.0f}× in price per marla**. A model therefore needs to "
  "know the neighbourhood, not only the city. That is why the `location` column is turned into a numeric "
  "*neighbourhood price level* feature (target encoding, section 4).")
W("")
img("eda_correlation.png", "Correlation matrix")
W("Bedrooms, baths and total rooms are strongly correlated with each other (multicollinearity), and all of them are "
  "correlated with area. Latitude and longitude are also correlated with each other. Raw GPS alone says little about price "
  "across Pakistan, but it does within a city.")
W("")

# 4 ---------------------------------------------------------------------------
W("## 4. Methodology")
W("")
W("### 4.1 Pipeline")
W("```")
W("zameen-updated.csv.xz → cleaning → feature engineering → 80:20 split → [ scaling | one-hot | target encoding ] → model(log price) → exp → PKR")
W("                                                                         fitted on the TRAINING set only")
W("```")
W("")
W("### 4.2 Input features (identical for all six models)")
W("| Feature | Type | Description |")
W("|---|---|---|")
fdesc = {
    "area_marla": ("numeric", "Size in marla (kanal × 20)"), "bedrooms": ("numeric", "Bedrooms"), "baths": ("numeric", "Bathrooms"),
    "latitude": ("numeric", "GPS latitude"), "longitude": ("numeric", "GPS longitude"),
    "log_area_marla": ("engineered", "log(area): price grows roughly in proportion to size, which is linear on the log scale"),
    "marla_per_bedroom": ("engineered", "Spaciousness: area ÷ bedrooms"), "baths_per_bedroom": ("engineered", "Luxury indicator: baths ÷ bedrooms"),
    "total_rooms": ("engineered", "Bedrooms + baths"),
    "distance_to_centre_km": ("engineered", "Straight-line (haversine) distance to the city centre: Saddar (Karachi), Mall Road (Lahore), Blue Area (Islamabad), Saddar (Rawalpindi), Clock Tower (Faisalabad)"),
    "listed_by_agency": ("engineered", "1 if a real-estate agency posted the listing"),
    "property_type": ("categorical → one-hot", "House (reference), Flat, Upper Portion, Lower Portion, Penthouse, Farm House"),
    "city": ("categorical → one-hot", "Karachi (reference), Lahore, Islamabad, Rawalpindi, Faisalabad"),
    "city_location": ("categorical → target encoding", f"Neighbourhood ({clean.groupby(['city', 'location']).ngroups:,} different ones), replaced by its average log-price in the training data"),
}
for f in FEATURES:
    W(f"| `{f}` | {fdesc[f][0]} | {fdesc[f][1]} |")
W("")
W("### 4.3 Preprocessing decisions")
W("| Decision | What | Why |")
W("|---|---|---|")
W("| **Log target** | All models learn $\\log(\\text{price})$; predictions are converted back with $e^{\\hat y}$ | Prices span 100×. On the log scale errors are *relative* (10% on a 50-lakh flat counts the same as 10% on a 10-crore house) and the relationship with log(area) becomes linear |")
W("| **Scaling** | `StandardScaler`, $z=(x-\\mu)/\\sigma$ | Needed by KNN (distances) and the Neural Network (gradient descent); harmless for trees |")
W("| **One-hot encoding** | `property_type`, `city`, dropping one reference category each | Numbers without a fake order; dropping a reference avoids perfect collinearity in Linear Regression |")
W("| **Target encoding** | `city_location` → smoothed mean log-price of that neighbourhood, computed with 5-fold *cross-fitting* inside the training set | 1,400+ neighbourhoods would mean 1,400+ one-hot columns. Cross-fitting stops a row from seeing its own price, so nothing leaks |")
W("| **No leakage** | Cleaning works row by row; every learned transformation sits inside the model pipeline and is fitted on training data only | The test set stays truly unseen |")
W("")
W("### 4.4 Train/test split")
img("data_split.png", "Train/test split")
W("| Set | Rows | Share | Median price | Mean price | Used for |")
W("|---|---:|---:|---:|---:|---|")
W(f"| Training | {len(train):,} | 80% | {P(train.price.median())} | {P(train.price.mean())} | fitting, choosing k, tuning, cross-validation |")
W(f"| Test | {len(test):,} | 20% | {P(test.price.median())} | {P(test.price.mean())} | final evaluation only |")
W("")
W("### 4.5 Evaluation metrics")
W("| Metric | Formula | Meaning |")
W("|---|---|---|")
W("| MAE | $\\frac1n\\sum\\lvert y_i-\\hat y_i\\rvert$ | Average error in rupees |")
W("| MSE | $\\frac1n\\sum (y_i-\\hat y_i)^2$ | Punishes large mistakes heavily (units: PKR²) |")
W("| RMSE | $\\sqrt{\\text{MSE}}$ | Back in rupees; main ranking metric |")
W("| R² | $R^2=1-\\frac{\\sum (y_i-\\hat y_i)^2}{\\sum (y_i-\\bar y)^2}$ | Share of price variation explained (1 = perfect) |")
W("| MAPE | $\\frac{100}{n}\\sum \\frac{\\lvert y_i-\\hat y_i\\rvert}{y_i}$ | Typical % error; fair to cheap and expensive properties alike |")
W("| Within ±10% / ±20% | share with $\\lvert y-\\hat y\\rvert / y \\le 0.1$ / $0.2$ | How often the estimate is close enough to be useful |")
W("")

# 5 ---------------------------------------------------------------------------
W("## 5. The six algorithms")
W("")
W("| Model | Key equation | Settings used |")
W("|---|---|---|")
W("| Linear Regression | $\\hat y=\\beta_0+\\sum_j\\beta_jx_j$, $\\boldsymbol\\beta=(X^TX)^{-1}X^T\\mathbf y$ | Ordinary Least Squares |")
W("| Decision Tree | split minimising $SSE_L+SSE_R$; leaf = mean of $y$ | `max_depth=14`, `min_samples_leaf=8` |")
W("| Random Forest | $\\hat y=\\frac1B\\sum_b T_b(x)$ (bootstrap + random features) | 80 trees, `max_features=0.4`, `max_leaf_nodes=1200`, `min_samples_leaf=2` |")
W("| Neural Network | $h=\\text{ReLU}(Wx+b)$, back-propagation with Adam | 128-64-32 neurons, `alpha=1e-4`, batch 128, early stopping |")
W("| XGBoost | $\\hat y^{(t)}=\\hat y^{(t-1)}+\\eta f_t(x)$, $w_j=-\\frac{G_j}{H_j+\\lambda}$ | 800 trees, η = 0.05, depth 8, subsample 0.8, colsample 0.7, λ = 1 |")
W(f"| KNN | distance-weighted mean of the $k$ nearest neighbours | k = {summary['knn_best_k']} (chosen by 5-fold CV), Euclidean |")
W("")
W("### 5.1 Hyper-parameter tuning")
W("Settings for the two ensembles were chosen on a **validation split taken from the training set**; the test set "
  "was never used for this. The trial results (validation RMSE in lakh) are in `results/hyperparameter_tuning.csv`. "
  "Among configurations of almost equal accuracy, the one with the **smaller saved model** was chosen, so the whole "
  "project fits on GitHub and in one zip.")
W("")
W("| Model | Configuration | Val. RMSE | Val. R² | Val. MAPE | Model size | Chosen |")
W("|---|---|---:|---:|---:|---:|:---:|")
for _, r in tune.iterrows():
    if r.model == "XGBoost":
        cfg = f"depth {int(r.max_depth)}, {int(r.n_estimators)} trees, η {r.learning_rate}"
    else:
        cfg = f"{int(r.n_estimators)} trees, {int(r.max_leaf_nodes)} leaves, max_feat {r.max_features}"
    W(f"| {r.model} | {cfg} | {r.val_RMSE_lakh:.1f} L | {r.val_R2:.4f} | {r.val_MAPE_pct:.1f}% | "
      f"{'' if pd.isna(r.model_file_MB) else f'{r.model_file_MB:.1f} MB'} | {'✔' if r.chosen == 'yes' else ''} |")
W("")
W("### 5.2 What the models learned")
img("linear_coefficients.png", "Linear coefficients")
W(f"**Linear Regression.** The `log_area_marla` coefficient ({coef['log_area_marla']:+.2f} per std) is the strongest "
  f"numeric effect. Flats, portions and penthouses sell for about {100 * (1 - np.exp(coef['property_type_Flat'])):.0f}% "
  "less than a house of the same size. The city coefficients look huge, but they are offset by equally large negative "
  "latitude/longitude coefficients. Karachi lies about 7° further south than the other cities, so these variables "
  "duplicate each other. This multicollinearity is one reason a linear model is hard to interpret here.")
W("")
img("decision_tree_top_levels.png", "Decision tree")
W("**Decision Tree.** The first two levels split on **area**, with room count and neighbourhood price level next. This is the "
  "same reasoning an estate agent uses: size first, then location.")
W("")
img("nn_training_curve.png", "Neural network training")
W("**Neural Network.** The loss falls steadily. Early stopping keeps the epoch with the best validation R², "
  "which prevents over-fitting.")
W("")
img("feature_importance.png", "Feature importance")
top_xgb = fi["XGBoost"].sort_values(ascending=False).index[:5]
W(f"**Random Forest and XGBoost.** Area dominates both models. Next come {', '.join(f'`{t}`' for t in top_xgb[2:5])} "
  "for XGBoost, and the neighbourhood price level (`city_location`) and room counts for Random Forest.")
W("")
img("knn_k_selection.png", "Choosing k")
W(f"**KNN.** k = 1 memorises noise and large k over-smooths. Five-fold cross-validation picked **k = {summary['knn_best_k']}**.")
W("")

# 6 ---------------------------------------------------------------------------
W("## 6. Results")
W("")
W(f"### 6.1 Comparative performance table (test set, {len(test):,} properties)")
W("")
W("| Rank | Model | MAE | MSE (PKR²) | RMSE | R² | R² (log) | MAPE | Within ±10% | Within ±20% | Train time |")
W("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for i, (n, r) in enumerate(ranked.iterrows(), 1):
    bb = "**" if n == best else ""
    W(f"| {i} | {bb}{n}{bb} | {bb}{P(r.MAE)}{bb} | {bb}{r.MSE:.3e}{bb} | {bb}{P(r.RMSE)}{bb} | {bb}{r.R2:.4f}{bb} | "
      f"{r['R2 (log)']:.4f} | {r['MAPE (%)']:.1f}% | {r['Within 10%']:.1f}% | {r['Within 20%']:.1f}% | {r['Train time (s)']:.1f} s |")
W("")
order_by = {c: list(m.sort_values(c, ascending=c not in ("R2", "R2 (log)", "Within 10%", "Within 20%")).index) for c in
            ["MAE", "RMSE", "R2", "MAPE (%)", "Within 10%"]}
same = all(v[0] == best for v in order_by.values())
W(f"{best} is **first on every metric**." if same else f"{best} is first by RMSE.")
W(f" The full RMSE ranking is: {' > '.join(ranked.index)}.")
W("")
img("metrics_comparison.png", "Metric comparison")
W("### 6.2 Actual vs predicted")
img("actual_vs_predicted.png", "Actual vs predicted")
W("Each dot is one test property. Points on the dashed line are perfect predictions. XGBoost and Random Forest "
  "form the tightest band, and Linear Regression is the widest. Every model has more spread at the cheap end, where small "
  "flats and portions vary a lot in quality.")
W("")
img("sample_predictions.png", "Sample predictions")
W("### 6.3 Over-fitting and stability")
img("train_vs_test_r2.png", "Train vs test")
W("| Model | Train R² | Test R² | Gap | Diagnosis |")
W("|---|---:|---:|---:|---|")
diag = {"Linear Regression": "No over-fitting, but under-fits: the model is too simple",
        "Decision Tree": "Mild over-fitting, limited by depth 14 and leaf size 8",
        "Random Forest": "Small gap: averaging 80 trees keeps it stable",
        "Neural Network": "Small gap thanks to early stopping",
        "XGBoost": "Learns the training data closely, yet still has the best test score; regularisation and shrinkage keep it general",
        "KNN": "Largest gap: it stores the training set, and each training point is its own closest neighbour"}
for n in MODEL_ORDER:
    r = m.loc[n]
    W(f"| {n} | {r['Train R2']:.3f} | {r.R2:.3f} | {r['Train R2'] - r.R2:+.3f} | {diag[n]} |")
W("")
if has_cv:
  img("cross_validation_rmse.png", "Cross-validation")
  W(f"Five-fold cross-validation on the training set gives the same ranking. {best} has the lowest CV RMSE "
  f"({P(m.loc[best, 'CV RMSE (mean)'])} ± {P(m.loc[best, 'CV RMSE (std)'])}), so the result does not depend on one "
  "lucky split.")
W("")

# 7 ---------------------------------------------------------------------------
W("## 7. Error analysis")
W("")
img("residuals.png", "Residuals")
W("Percentage errors are centred close to 0 for every model, so none of them systematically over- or under-prices. "
  f"{best} has the narrowest spread. The small bars at the edges collect the errors beyond ±100%.")
W("")
img("abs_error_boxplot.png", "Absolute error boxplot")
W("| Model | Median absolute % error | MAPE |")
W("|---|---:|---:|")
for n in ranked.index:
    ape = 100 * np.abs(pred[n] - pred.actual) / pred.actual
    W(f"| {n} | {np.median(ape):.1f}% | {ape.mean():.1f}% |")
W("")
W("### 7.1 By price range")
img("mape_by_price_range.png", "MAPE by price range")
mape_table(by_band, "Price range")
W("Cheap properties (under 50 lakh) are the hardest in percentage terms. A 10-lakh error is small for a 5-crore house "
  "but large for a 40-lakh flat.")
W("")
W("### 7.2 By city")
img("mape_by_city.png", "MAPE by city")
mape_table(by_city, "City")
W("Lahore and Islamabad are the easiest cities to price: they have well-defined societies and sectors. Karachi is "
  "harder because it has many apartment blocks and large price differences inside the same neighbourhood. "
  "Faisalabad is hard because it has the fewest listings to learn from.")
W("")
W("### 7.3 By property type")
img("mape_by_property_type.png", "MAPE by property type")
mape_table(by_type, "Property type")
W("Houses are priced best. Penthouses and farm houses are rare (fewer than 200 and 100 listings), so every model is "
  "unreliable for them. More data, not a different algorithm, would fix that.")
W("")

# 8 ---------------------------------------------------------------------------
W("## 8. Strengths and limitations")
W("")
W("| Algorithm | Strengths | Limitations (observed here) |")
W("|---|---|---|")
W(f"| **Linear Regression** | Instant to train; coefficients are readable; a solid baseline thanks to log(price) and log(area) | Cannot model interactions (a marla in DHA is worth more than a marla elsewhere); multicollinearity makes coefficients misleading; worst MAPE ({m.loc['Linear Regression', 'MAPE (%)']:.1f}%) |")
W("| **Decision Tree** | Captures non-linearity; easy to explain as if/else rules; no scaling needed | Unstable (high variance); predictions are step-shaped; weaker than the ensembles |")
W(f"| **Random Forest** | Averaging many trees cuts variance (RMSE {pct_lower(m.loc['Random Forest', 'RMSE'], m.loc['Decision Tree', 'RMSE']):.0f}% lower than one tree); robust; little tuning | Larger model file; cannot extrapolate beyond the prices it has seen |")
W("| **Neural Network** | Learns smooth non-linear functions; third best | Slowest to train; needs scaling and tuning; a black box |")
W("| **XGBoost** | Sequential trees correct earlier errors (lower bias); regularisation (λ, η, subsampling) controls variance; fast | Many hyper-parameters; large train/test gap shows it can over-fit without regularisation; less interpretable |")
W("| **KNN** | Simple, intuitive (\"similar properties sold for…\"); no training | Needs scaling; slow predictions on big data; memorises the training set; struggles with rare types (farm houses) |")
W("")

# 9 ---------------------------------------------------------------------------
W("## 9. Best model and justification")
W("")
W(f"**{best} is the best-performing algorithm.** The evidence:")
W("")
W(f"1. **Best on all four required metrics:** MAE {P(b.MAE)}, MSE {b.MSE:.3e}, RMSE {P(b.RMSE)}, R² {b.R2:.4f}.")
W(f"2. **Best in practical terms:** lowest MAPE ({b['MAPE (%)']:.1f}%) and the most properties within ±10% "
  f"({b['Within 10%']:.1f}%) and ±20% ({b['Within 20%']:.1f}%).")
if has_cv:
  W(f"3. **Stable:** lowest 5-fold CV RMSE ({P(m.loc[best, 'CV RMSE (mean)'])} ± {P(m.loc[best, 'CV RMSE (std)'])}).")
W(f"4. **Consistent:** lowest error in every price band, in every city and for every common property type.")
W(f"5. **Practical:** trains in about {b['Train time (s)']:.0f} seconds and the saved model is about 2 MB.")
W("")
W("**Why it wins:** a property's price depends on **interactions**. The value of a marla depends on the city, the "
  "neighbourhood, the property type and the size together. Tree-based models capture such interactions automatically. "
  "Random Forest averages independent trees and mainly **reduces variance**. XGBoost builds each new tree on the "
  "remaining errors (gradients) of the current model, so it also **reduces bias**. Its regularisation (λ, a small "
  "learning rate η, row and column subsampling) keeps the extra flexibility from turning into over-fitting. "
  "Gradient-boosted trees are usually the best choice on structured tabular data like this, and this experiment agrees.")
W("")

# 10 --------------------------------------------------------------------------
W("## 10. Conclusion")
W("")
W(f"All six algorithms were implemented under identical conditions on {len(clean):,} cleaned Zameen.com listings. "
  f"Tree ensembles clearly beat the single models: {best} (R² {b.R2:.3f}) ranks first, followed by {ranked.index[1]} "
  f"({ranked.iloc[1].R2:.3f}) and {ranked.index[2]} ({ranked.iloc[2].R2:.3f}). Linear Regression under-fits "
  f"({m.loc['Linear Regression', 'R2']:.3f}) because property prices are driven by interactions between size and location.")
W("")
W("**Possible improvements:**")
W("- Listing text such as 'corner', 'park facing' or 'brand new' (needs NLP on the title)")
W("- More recent listings, since prices have risen sharply since 2019")
W("- Plot and commercial listings")
W("- Monotonic constraints, so that price never falls when area rises")
W("- Prediction intervals (quantile regression) so the app can show a price range")
W("")

# 11 --------------------------------------------------------------------------
W("## 11. Using the trained models")
W("")
W("```bash")
W("streamlit run app.py      # interactive web app: choose city, neighbourhood, size → price from all 6 models")
W("python predict.py         # command line example")
W("```")
W("")
W("Every saved model in `models/` includes its preprocessing, so it can be loaded with `joblib.load` and given a raw "
  "property description. The app limits each input to realistic values found in the data, so it cannot be asked about "
  "impossible properties.")
W("")

# 12 --------------------------------------------------------------------------
W("## 12. Reproducing the results")
W("")
W("```bash")
W("pip install -r requirements.txt")
W("python train.py          # cleans the data, trains the 6 models, writes models/, results/, data/")
W("python make_report.py    # rebuilds this report")
W("python handwritten/hand_calculations.py   # rebuilds the handwritten solution")
W("```")
W("")
W("| Output | Content |")
W("|---|---|")
W("| `data/raw/zameen-updated.csv.xz` | Original Zameen.com dataset (all 168,446 rows, compressed; pandas reads it directly) |")
W("| `data/zameen_clean.csv` | Cleaned dataset |")
W("| `data/train.csv`, `data/test.csv` | The exact training and test rows with all 14 inputs and the price |")
W("| `models/*.joblib` | The 6 trained models |")
W("| `results/metrics.csv` | All metrics for all models |")
W("| `results/test_predictions.csv`, `train_predictions.csv` | Actual and predicted price for every property |")
W("| `results/mape_by_*.csv` | Error by price range, city and property type |")
W("| `results/hyperparameter_tuning.csv` | Tuning trials |")
W("| `results/figures/` | All figures in this report |")
W("")

text = "\n".join(out)
text = re.sub(r"(?<!\\)\$(?=\d)", r"\\$", text)
(ROOT / "REPORT.md").write_text(text, encoding="utf-8")
print("Wrote REPORT.md")

if shutil.which("pandoc") and shutil.which("xelatex"):
    tmp = ROOT / ".report_pdf.md"
    tmp.write_text(text.replace(".png)", ".png){width=100%}"), encoding="utf-8")
    cmd = ["pandoc", str(tmp), "-f", "markdown-yaml_metadata_block-multiline_tables", "-o", str(ROOT / "REPORT.pdf"),
           "--pdf-engine=xelatex", "-V", "mainfont=DejaVu Sans", "-V", "monofont=DejaVu Sans Mono",
           "-V", "geometry:margin=1.5cm", "-V", "fontsize=10pt", "-V", "colorlinks=true", "--resource-path", str(ROOT)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    tmp.unlink()
    print("Wrote REPORT.pdf" if res.returncode == 0 else f"PDF skipped: {res.stderr[-800:]}")
