import streamlit as st

from utils import theme

st.set_page_config(
    page_title="NIFTY Hybrid Econometric-ML Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme must be selected before CSS/plot templates are applied on this run
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "dark"

import plotly.io as pio
pio.templates.default = theme.plot_template()

theme.inject_css()

with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">
    <span style="font-size:1.6rem;">📈</span>
    <span style="font-weight:800; font-size:1.15rem; letter-spacing:-0.02em;">NIFTY Analytics</span>
    </div>
    <div class="muted" style="margin-bottom:14px;">Hybrid Econometric–ML Dissertation Dashboard</div>
    """, unsafe_allow_html=True)

    theme.theme_toggle_control()
    st.divider()

pages = {
    "Overview": [
        st.Page("views/home.py", title="Home", icon="🏠", default=True),
        st.Page("views/dataset_explorer.py", title="Dataset Explorer", icon="🗂️"),
    ],
    "Exploration & Feature Engineering": [
        st.Page("views/eda.py", title="Exploratory Data Analysis", icon="🔍"),
        st.Page("views/vif.py", title="VIF & Multicollinearity", icon="🧮"),
        st.Page("views/pca.py", title="PCA", icon="🧭"),
    ],
    "Clustering & Regression": [
        st.Page("views/clustering.py", title="Clustering", icon="🧩"),
        st.Page("views/ols_daily_return.py", title="OLS — Daily Return Drivers", icon="📉"),
        st.Page("views/regularized_regression.py", title="Regularized Regression — Volume", icon="📦"),
    ],
    "Time Series & Machine Learning": [
        st.Page("views/arima_close.py", title="ARIMA — Close Price", icon="📊"),
        st.Page("views/time_series_workflow.py", title="Time Series Workflow — Volume", icon="⏱️"),
        st.Page("views/volatility_arch_garch.py", title="Volatility — ARCH/GARCH", icon="🌊"),
        st.Page("views/var_granger.py", title="VAR & Granger Causality", icon="🔗"),
        st.Page("views/ann_forecasting.py", title="ANN Forecasting Model", icon="🧠"),
    ],
    "Results": [
        st.Page("views/model_comparison.py", title="Model Comparison", icon="🏆"),
        st.Page("views/forecasting.py", title="Interactive Forecasting", icon="🔮"),
        st.Page("views/conclusion.py", title="Conclusion", icon="📖"),
    ],
}

pg = st.navigation(pages)

with st.sidebar:
    st.divider()
    if st.session_state.get("data_ready"):
        st.success("Data loaded", icon="✅")
    else:
        st.warning("Load data on the Home page", icon="⚠️")
    st.caption("Built with Streamlit · statsmodels · scikit-learn · TensorFlow/Keras")

pg.run()
