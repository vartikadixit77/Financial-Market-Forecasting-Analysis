import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.stats.diagnostic import het_arch

from utils import data_pipeline as dp
from utils import theme, ui

theme.inject_css()
ui.page_header(
    "Volatility Modelling — ARCH & GARCH",
    "Markets don't just move — they move in clusters of calm and turbulence. This models that pattern directly.",
    "🌊",
)

if not st.session_state.get("data_ready"):
    st.warning("Go to the Home page and load the data first.")
    st.stop()

ui.objective_box(
    "To test whether NIFTY50 daily returns exhibit 'volatility clustering' (large moves "
    "followed by large moves, calm followed by calm) and, if so, model that conditional "
    "volatility directly using ARCH and GARCH rather than assuming constant variance."
)
ui.methodology_box(
    "The Engle ARCH-effect test checks whether squared/absolute returns are autocorrelated — "
    "evidence that volatility itself has memory. If present, ARCH(1) models today's variance "
    "as a function of yesterday's squared shock; GARCH(1,1) extends this with a persistence "
    "term on yesterday's variance, usually fitting volatility clustering more parsimoniously."
)

if not st.session_state.get("data_ready"):
    st.warning("Go to the Home page and load the data first.")
    st.stop()

try:
    from arch import arch_model
except ImportError:
    st.error("The `arch` package isn't installed. Add `arch` to requirements.txt and reinstall.")
    st.stop()

df_market = st.session_state["df_market"]
returns = df_market.set_index("Date")["NIFTY50_Close"].pct_change().dropna() * 100

tab1, tab2, tab3 = st.tabs(["Returns & ARCH Test", "ARCH(1)", "GARCH(1,1)"])

with tab1:
    fig = px.line(x=returns.index, y=returns.values, title="Daily returns of NIFTY50 (%)", labels={"x": "Date", "y": "Return (%)"})
    st.plotly_chart(fig, use_container_width=True)

    arch_test = het_arch(returns)
    c1, c2 = st.columns(2)
    c1.metric("LM Statistic", round(arch_test[0], 4))
    c2.metric("P-value", round(arch_test[1], 4))
    st.write("✅ ARCH effect present — proceed with GARCH modelling." if arch_test[1] < 0.05 else "No ARCH effect detected.")

with tab2:
    with st.spinner("Fitting ARCH(1)..."):
        arch_fit = arch_model(returns, mean="Constant", vol="ARCH", p=1).fit(disp="off")
    st.text(arch_fit.summary().as_text())
    st.session_state["arch_aic"] = arch_fit.aic
    st.session_state["arch_bic"] = arch_fit.bic

    fig = px.line(x=arch_fit.conditional_volatility.index, y=arch_fit.conditional_volatility.values,
                  title="Estimated conditional volatility (ARCH)", labels={"x": "Date", "y": "Volatility"})
    st.plotly_chart(fig, use_container_width=True)

    horizon = st.slider("Forecast horizon (days)", 5, 60, 30, key="arch_h")
    forecast = arch_fit.forecast(horizon=horizon)
    forecast_var = forecast.variance.iloc[-1]
    fig_fc = px.line(x=[str(i) for i in range(1, len(forecast_var) + 1)], y=forecast_var.values,
                      title="Forecasted variance", labels={"x": "Days ahead", "y": "Variance"})
    st.plotly_chart(fig_fc, use_container_width=True)

with tab3:
    with st.spinner("Fitting GARCH(1,1)..."):
        garch_fit = arch_model(returns, mean="Constant", vol="GARCH", p=1, q=1).fit(disp="off")
    st.text(garch_fit.summary().as_text())

    fig = px.line(x=garch_fit.conditional_volatility.index, y=garch_fit.conditional_volatility.values,
                  title="Estimated conditional volatility (GARCH)", labels={"x": "Date", "y": "Volatility"})
    st.plotly_chart(fig, use_container_width=True)

    horizon2 = st.slider("Forecast horizon (days)", 5, 60, 30, key="garch_h")
    forecast2 = garch_fit.forecast(horizon=horizon2)
    forecast_var2 = forecast2.variance.iloc[-1]
    fig_fc2 = px.line(x=[str(i) for i in range(1, len(forecast_var2) + 1)], y=forecast_var2.values,
                       title="Forecasted variance", labels={"x": "Days ahead", "y": "Variance"})
    st.plotly_chart(fig_fc2, use_container_width=True)

    st.subheader("ARCH vs GARCH — model fit comparison")
    comp = pd.DataFrame({
        "Model": ["ARCH(1)", "GARCH(1,1)"],
        "AIC": [st.session_state.get("arch_aic", np.nan), garch_fit.aic],
        "BIC": [st.session_state.get("arch_bic", np.nan), garch_fit.bic],
        "Log-Likelihood": [np.nan, garch_fit.loglikelihood],
    })
    st.dataframe(comp, use_container_width=True)

st.divider()
ui.business_box(
    "Conditional volatility forecasts from GARCH feed directly into risk management: "
    "position sizing, Value-at-Risk, and options pricing all depend on a forward-looking "
    "volatility estimate rather than a flat historical average."
)
