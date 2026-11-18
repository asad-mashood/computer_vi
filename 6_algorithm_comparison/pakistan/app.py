"""
Interactive price estimator for Pakistani properties, using the six trained models.

    pip install streamlit
    streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.data_preprocessing import FEATURES, add_features, format_pkr

ROOT = Path(__file__).resolve().parent
MODEL_FILES = {"XGBoost": "xgboost", "Random Forest": "random_forest", "Neural Network": "neural_network",
               "Decision Tree": "decision_tree", "KNN": "knn", "Linear Regression": "linear_regression"}

st.set_page_config(page_title="Pakistan House Price Estimator", page_icon="🏠", layout="wide")


@st.cache_resource
def load_models():
    return {n: joblib.load(ROOT / "models" / f"{f}.joblib") for n, f in MODEL_FILES.items()}


@st.cache_data
def load_data():
    df = pd.read_csv(ROOT / "data" / "zameen_clean.csv")
    metrics = pd.read_csv(ROOT / "results" / "metrics.csv", index_col=0)
    return df, metrics


models = load_models()
df, metrics = load_data()

st.title("🏠 Pakistan House Price Estimator")
st.caption(f"Six machine-learning models trained on {len(df):,} Zameen.com listings "
           "(Karachi, Lahore, Islamabad, Rawalpindi, Faisalabad). Prices are 2018–19 asking prices.")

tab_predict, tab_compare, tab_help = st.tabs(["💰 Estimate a price", "📊 Model comparison", "❓ How it works"])

# ---------------------------------------------------------------------------
with tab_predict:
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        st.subheader("Describe the property")
        city = st.selectbox("City", ["Lahore", "Karachi", "Islamabad", "Rawalpindi", "Faisalabad"])
        loc_counts = df[df.city == city].location.value_counts()
        locations = loc_counts[loc_counts >= 5].index.tolist()
        location = st.selectbox(f"Neighbourhood / society ({len(locations)} with 5+ listings)", locations)
        ptype = st.selectbox("Property type", ["House", "Flat", "Upper Portion", "Lower Portion", "Penthouse", "Farm House"])
        c1, c2 = st.columns([2, 1])
        unit = c2.radio("Unit", ["Marla", "Kanal"], horizontal=True)
        size = c1.number_input(f"Size ({unit.lower()})", min_value=1.0 if unit == "Marla" else 0.5,
                               max_value=200.0 if unit == "Marla" else 10.0,
                               value=10.0 if unit == "Marla" else 1.0, step=0.5 if unit == "Kanal" else 1.0)
        area = size * (20 if unit == "Kanal" else 1)
        c3, c4 = st.columns(2)
        beds = c3.slider("Bedrooms", 1, 12, 4)
        baths = c4.slider("Bathrooms", 1, 12, 4)
        agency = st.checkbox("Listed by a real-estate agency", value=True)

    # GPS = typical coordinates of the chosen neighbourhood
    sub = df[(df.city == city) & (df.location == location)]
    lat, lon = sub.latitude.median(), sub.longitude.median()
    x = add_features(pd.DataFrame([dict(property_type=ptype, city=city, location=location, latitude=lat, longitude=lon,
                                        area_marla=area, bedrooms=beds, baths=baths, listed_by_agency=int(agency))]))[FEATURES]

    with right:
        st.subheader("Estimated price")
        warn = []
        mpb = area / beds
        lo, hi = (df.area_marla / df.bedrooms).quantile([0.01, 0.99])
        if not lo <= mpb <= hi:
            warn.append(f"{area:g} marla with {beds} bedrooms is unusual in the data ({mpb:.1f} marla per bedroom; "
                        f"typical range {lo:.1f}–{hi:.1f}). Predictions may be unreliable.")
        if baths > beds + 3:
            warn.append("Many more bathrooms than bedrooms is rare in the data.")
        for w in warn:
            st.warning(w)

        preds = {n: float(m.predict(x)[0]) for n, m in models.items()}
        best = preds["XGBoost"]; mape = metrics.loc["XGBoost", "MAPE (%)"]
        st.metric("Best model (XGBoost)", format_pkr(best),
                  help="XGBoost had the lowest error on 12,476 unseen test properties.")
        st.caption(f"Typical error of this model is ±{mape:.0f}%, so a realistic range is "
                   f"**{format_pkr(best * (1 - mape / 100))} – {format_pkr(best * (1 + mape / 100))}**. "
                   f"That is about {format_pkr(best / area)} per marla.")

        rows = [{"Model": n, "Prediction": format_pkr(p), "Test R²": f"{metrics.loc[n, 'R2']:.3f}",
                 "Typical error (MAPE)": f"{metrics.loc[n, 'MAPE (%)']:.1f}%"} for n, p in preds.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.bar_chart(pd.Series({n: p / 1e7 for n, p in preds.items()}, name="Crore PKR"), horizontal=True)

        st.subheader("Similar real listings")
        comp = sub.assign(diff=(sub.area_marla - area).abs() + (sub.bedrooms - beds).abs() * 0.5)
        comp = comp[comp.property_type == ptype].nsmallest(6, "diff")
        if len(comp):
            st.dataframe(pd.DataFrame({"Type": comp.property_type, "Area (marla)": comp.area_marla,
                                       "Beds": comp.bedrooms, "Baths": comp.baths,
                                       "Asking price": comp.price.map(format_pkr),
                                       "Per marla": (comp.price / comp.area_marla).map(format_pkr)}),
                         hide_index=True, width="stretch")
        else:
            st.info(f"No {ptype.lower()} listings in {location} to compare with.")

# ---------------------------------------------------------------------------
with tab_compare:
    st.subheader("How the six models performed on 12,476 unseen test properties")
    t = metrics.sort_values("RMSE")[["R2", "MAE", "RMSE", "MAPE (%)", "Within 10%", "Within 20%", "Train R2"]].copy()
    t.insert(0, "Rank", range(1, len(t) + 1))
    for c in ("MAE", "RMSE"):
        t[c] = t[c].map(format_pkr)
    st.dataframe(t.round(3), width="stretch")
    fig_dir = ROOT / "results" / "figures"
    for f, cap in [("actual_vs_predicted.png", "Actual vs predicted"), ("metrics_comparison.png", "All metrics"),
                   ("mape_by_city.png", "Error by city"), ("feature_importance.png", "What drives the price")]:
        st.image(str(fig_dir / f), caption=cap, width="stretch")

# ---------------------------------------------------------------------------
with tab_help:
    st.markdown("""
**How does it work?** Every time you change an input, the property is sent to all six
models and each one returns a price.

* **Inputs:** city, neighbourhood, type, size, bedrooms, bathrooms. The app fills in the neighbourhood's typical GPS
  location and computes the same engineered features used in training (distance to the city centre, marla per
  bedroom, ...).
* **Why do models disagree?** Each algorithm learns differently: a straight line (Linear Regression), if/else rules
  (Decision Tree), many averaged trees (Random Forest), layers of neurons (Neural Network), trees that correct each
  other (XGBoost), or the most similar listings (KNN). XGBoost was the most accurate on unseen data.
* **Units:** 1 lakh = 1,00,000 PKR · 1 crore = 100 lakh · 1 kanal = 20 marla.
* **Limits:** prices are 2018–19 asking prices from Zameen.com, so today's market is higher. Rare property types
  (farm houses, penthouses) have few examples and wider errors.
""")
