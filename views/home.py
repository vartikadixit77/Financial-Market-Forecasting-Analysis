import streamlit as st
import pandas as pd
import numpy as np

from utils import data_pipeline as dp
from utils import theme, ui

theme.inject_css()

# ---------------------------------------------------------------------------
# Hero section
# ---------------------------------------------------------------------------
st.markdown("""
<div class="big-title">📈 Financial Market Forecasting & Analysis</div>
<div class="subtitle">Hybrid Econometric–Machine Learning Framework for Indian Equity Markets (NSE &amp; BSE)</div>
""", unsafe_allow_html=True)

c1, c2 = st.columns([2.2, 1])

with c1:
    st.markdown("""
    <div class="dash-card accent-left">
    <span class="badge">🎓 Dissertation</span><br><br>
    <b>Title:</b> Financial Market Forecasting and Analysis using Machine Learning, Time Series, and Econometric Models<br>
    <b>Student:</b> Vartika &nbsp;|&nbsp; M.Sc. Applied Statistics<br>
    <b>Institute:</b> Symbiosis Statistical Institute, Pune &nbsp;|&nbsp; Research Internship: University of Lucknow<br>
    <b>Supervisor:</b> <i>[Prof. (Dr.) Masood H. Siddiqui]</i>
    </div>
    """, unsafe_allow_html=True)

    ui.objective_box(
        "To build and rigorously compare econometric (ARIMA, SARIMA, ARCH/GARCH, VAR, OLS) and "
        "machine-learning (ANN) approaches for modelling liquidity, volatility, and price "
        "discovery in Indian equity markets — using both market microstructure data and RBI "
        "macroeconomic indicators — and to identify which framework forecasts NIFTY50 trading "
        "behaviour most reliably."
    )

with c2:
    st.markdown("""
    <div class="dash-card">
    <span class="badge gold">🗺️ How to use this dashboard</span><br><br>
    Use the sidebar to move through the pipeline in order — each stage builds on the one before it:
    <ol style="margin-top:8px; margin-bottom:0;">
    <li>Data → Dataset Explorer, EDA</li>
    <li>Feature Engineering → VIF, PCA</li>
    <li>Modelling → Clustering, Regression</li>
    <li>Time Series & ML → ARIMA family, Volatility, VAR, ANN</li>
    <li>Results → Model Comparison, Forecasting, Conclusion</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Data loading (sidebar-driven, same as before)
# ---------------------------------------------------------------------------
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_MARKET = DATA_DIR / "market_microstructure_dataset.csv"
DEFAULT_MACRO = DATA_DIR / "50_Macroeconomic_Indicators_rbi.xlsx"


def load_data():
    st.sidebar.markdown("#### 📁 Data source")
    use_default = DEFAULT_MARKET.exists() and DEFAULT_MACRO.exists()

    source = st.sidebar.radio(
        "Choose data source",
        ["Use bundled dataset", "Upload my own files"] if use_default else ["Upload my own files"],
        index=0, label_visibility="collapsed",
    )

    if source == "Use bundled dataset":
        df_market = dp.load_market(str(DEFAULT_MARKET))
        df_macro = dp.load_macro(str(DEFAULT_MACRO))
    else:
        market_file = st.sidebar.file_uploader("Market microstructure CSV", type=["csv"])
        macro_file = st.sidebar.file_uploader("RBI Macroeconomic Indicators (.xlsx)", type=["xlsx"])
        if not (market_file and macro_file):
            st.sidebar.warning("Upload both files to proceed.")
            return None, None
        df_market = dp.load_market(market_file)
        df_macro = dp.load_macro(macro_file)

    return df_market, df_macro


df_market, df_macro = load_data()

if df_market is not None and df_macro is not None:
    st.session_state["df_market"] = df_market
    st.session_state["df_macro"] = df_macro
    st.session_state["data_ready"] = True
else:
    st.session_state["data_ready"] = False

# ---------------------------------------------------------------------------
# Dataset summary + workflow
# ---------------------------------------------------------------------------
if st.session_state.get("data_ready"):
    inner = dp.build_inner_merge(df_market, df_macro)
    asof = dp.build_asof_merge(df_market, df_macro)

    st.subheader("📊 Dataset Summary")
    ui.kpi_row([
        ("Total Observations", f"{inner.shape[0]}", "Rows in the working (inner-join) dataset used for every model"),
        ("Variables", f"{inner.shape[1] - 1}", "Market + macroeconomic predictor columns, excluding Date"),
        ("Study Period", f"{inner['Date'].min().date()} → {inner['Date'].max().date()}", None),
        ("Target Variables", "Volume · Close · Returns", "NIFTY50_Volume (primary forecasting target), NIFTY50_Close, NIFTY50_Daily_Return"),
    ])

    st.markdown("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="dash-card">
        <span class="badge">🔀 Two merge strategies</span><br><br>
        <b>merge_asof</b> (backward-fill) — <b>{}</b> rows — used only for exploratory EDA.<br>
        <b>Inner join</b> on Date — <b>{}</b> rows — the working dataset for VIF, PCA,
        clustering, OLS, and every time-series / ML model in this dashboard.
        </div>
        """.format(asof.shape[0], inner.shape[0]), unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="dash-card">
        <span class="badge gold">🔑 Key Findings (preview)</span><br><br>
        • Multicollinearity is severe among raw price/volume series — VIF reduction is essential.<br>
        • Clustering surfaces distinct market regimes (calm, stressed, crisis) purely from data structure.<br>
        • Linear time-series models struggle with volume's regime-dependent noise — see Model Comparison.<br>
        • The ANN, given the same information set as ARIMA/SARIMA, tests whether non-linearity helps.
        </div>
        """, unsafe_allow_html=True)

    st.subheader("🧭 Project Workflow")
    st.markdown("""
    <div class="dash-card">
    <div style="display:flex; flex-wrap:wrap; gap:10px; align-items:center; font-weight:600;">
    <span>📥 Raw Data</span><span>→</span>
    <span>🔍 EDA</span><span>→</span>
    <span>🧮 VIF</span><span>→</span>
    <span>🧭 PCA</span><span>→</span>
    <span>🧩 Clustering</span><span>→</span>
    <span>📉 Regression (OLS / Ridge / Lasso)</span><span>→</span>
    <span>⏱️ Time Series (ARIMA/SARIMA/ARCH/GARCH/VAR)</span><span>→</span>
    <span>🧠 ANN</span><span>→</span>
    <span>🏆 Comparison &amp; Forecast</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.success("Data loaded — use the sidebar to move through the pipeline.")
else:
    st.warning("Load the two source files in the sidebar to begin (or use the bundled sample dataset).")

st.divider()
st.caption(
    "This dashboard reproduces, end to end, the full dissertation pipeline across four source "
    "notebooks — Clustering & Regression Analysis, OLS Regression Analysis, Time Series "
    "Forecasting Workflow, and the ANN Forecasting Model — on live data rather than static "
    "notebook output."
)
