"""
Master Streamlit Web Application: Comparative Machine Learning Regression
Covers both:
  - 🇵🇰 Pakistan Real Estate (Zameen.com) in Pakistani Rupees (PKR)
  - 🇺🇸 California Housing Benchmark in US Dollars (USD)

Run with:
    streamlit run app.py
"""
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Cross-Market ML Regression Benchmark",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Dataset Selection
st.sidebar.image("https://img.shields.io/badge/Algorithms-6%20Regression%20Models-blue?style=for-the-badge", use_container_width=True)
st.sidebar.title("🌍 Market Selection")
dataset_choice = st.sidebar.radio(
    "Select Real Estate Market:",
    ["🇵🇰 Pakistan Real Estate (PKR)", "🇺🇸 California Benchmark (USD)"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏆 Evaluated Models")
st.sidebar.markdown("""
1. **XGBoost Regressor** (Rank #1)
2. **Random Forest Regressor** (Rank #2)
3. **Multi-Layer Perceptron (Neural Network)**
4. **Decision Tree Regressor**
5. **K-Nearest Neighbors (KNN)**
6. **Linear Regression (OLS Baseline)**
""")

st.sidebar.markdown("---")
st.sidebar.caption("Author: Muhammad Asad Mashood  \nComparative ML Regression Study")

if dataset_choice.startswith("🇵🇰"):
    sys.path.insert(0, str(ROOT / "pakistan"))
    import importlib
    # Load and execute Pakistan app logic
    pak_app = ROOT / "pakistan" / "app.py"
    with open(pak_app, "r", encoding="utf-8") as f:
        code = f.read()
    # Strip set_page_config from inner app
    code_cleaned = "\n".join([line for line in code.split("\n") if "st.set_page_config" not in line])
    # Execute in dedicated namespace
    exec(code_cleaned, {"__file__": str(pak_app), "__name__": "__main__"})

else:
    sys.path.insert(0, str(ROOT / "california"))
    cal_app = ROOT / "california" / "app.py"
    with open(cal_app, "r", encoding="utf-8") as f:
        code = f.read()
    code_cleaned = "\n".join([line for line in code.split("\n") if "st.set_page_config" not in line])
    exec(code_cleaned, {"__file__": str(cal_app), "__name__": "__main__"})
