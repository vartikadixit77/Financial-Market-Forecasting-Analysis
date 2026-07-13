import streamlit as st

from utils import theme, ui

theme.inject_css()
ui.page_header(
    "Conclusion",
    "Major findings, contributions, limitations, and where this work could go next.",
    "📖",
)

if not st.session_state.get("data_ready"):
    st.warning("Go to the Home page and load the data first.")
    st.stop()

st.subheader("🔑 Major Findings")
st.markdown("""
<div class="dash-card">
<ol style="margin:0;">
<li><b>Severe multicollinearity</b> exists among raw market microstructure variables (correlated index/stock price series); VIF-based reduction is a necessary first step before any regression or clustering.</li>
<li><b>Distinct market regimes are recoverable</b> from the data using unsupervised clustering alone — no labels are required, and the four algorithms tested broadly agree on regime structure.</li>
<li><b>Macro variables have limited but statistically significant explanatory power</b> over NIFTY50 daily returns — currency, commodity, and monetary-policy shocks matter, but daily equity returns remain dominated by unexplained noise, as economic theory would predict.</li>
<li><b>NIFTY50 trading volume is noisy and regime-dependent</b> — classical linear time-series models (ARIMA, SARIMA, VAR) achieve modest, sometimes negative, out-of-sample R² on this series.</li>
<li><b>Volatility clustering is present</b> in NIFTY50 returns (confirmed by the ARCH-effect test), justifying GARCH-family modelling for risk applications distinct from volume forecasting.</li>
<li><b>The ANN, given an identical information set to ARIMA/SARIMA</b>, provides a fair test of whether non-linearity adds forecasting value — see the Model Comparison page for the specific result on this dataset.</li>
</ol>
</div>
""", unsafe_allow_html=True)

st.subheader("🎓 Research Contributions")
st.markdown("""
<div class="dash-card accent-left">
<ul style="margin:0;">
<li>A reproducible, end-to-end hybrid econometric-ML pipeline combining market microstructure and RBI macroeconomic data — from raw files through VIF, PCA, clustering, regression, classical time series, volatility modelling, VAR/Granger causality, and a neural network, in one consistent codebase.</li>
<li>A methodologically fair ANN-vs-econometrics comparison: identical train/test split, identical information set, identical metric suite — avoiding the common pitfall of comparing models trained on different data or evaluated on different metrics.</li>
<li>An interactive dashboard that makes every stage of the analysis inspectable and explainable to a non-technical audience, rather than leaving the results locked inside static notebooks.</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.subheader("⚠️ Limitations")
st.markdown("""
<div class="dash-card">
<ul style="margin:0;">
<li>All model comparisons rest on a <b>single</b> chronological train/test split and a single held-out window — results could shift under rolling-window or multiple-split validation.</li>
<li>The dataset spans a fixed historical period; structural breaks or regime changes outside this window are not captured.</li>
<li>The ANN uses only lagged values of the target (for a fair comparison with ARIMA/SARIMA) and does not yet exploit the exogenous macro/market variables available to VAR.</li>
<li>GARCH models return volatility, a genuinely different target from volume, so it cannot be scored on the same metric set — this is a scope limitation, not a modelling flaw.</li>
<li>Clustering results are sensitive to the chosen number of clusters/components and the VIF threshold used upstream; both are adjustable in this dashboard and should be sensitivity-tested.</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.subheader("🚀 Future Scope")
st.markdown("""
<div class="dash-card">
<ul style="margin:0;">
<li>Extend the ANN to ingest exogenous macro/market variables (matching VAR's information set) for a more complete non-linear benchmark.</li>
<li>Repeat the full model comparison across multiple rolling windows rather than one split, to test the stability of the ANN's advantage (or lack thereof).</li>
<li>Explore regime-aware forecasting that conditions the model choice on the cluster/regime identified for the current period.</li>
<li>Incorporate higher-frequency (daily/intraday) data where available, since several macro series here are only weekly.</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.divider()
ui.conclusion_box(
    "Taken together, this dissertation demonstrates that a disciplined, transparent hybrid "
    "econometric-ML pipeline can both surface interpretable structure (regimes, significant "
    "macro drivers, volatility clustering) and rigorously test whether machine learning adds "
    "genuine forecasting value on top of classical benchmarks — rather than assuming it does."
)
