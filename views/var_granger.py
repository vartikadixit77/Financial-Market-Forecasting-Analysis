import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.stats.stattools import durbin_watson

from utils import data_pipeline as dp
from utils import theme, ui

theme.inject_css()
ui.page_header(
    "VAR Model & Granger Causality",
    "Does NIFTY50 volume respond to macro shocks, and can we say one variable 'causes' another?",
    "🔗",
)

if not st.session_state.get("data_ready"):
    st.warning("Go to the Home page and load the data first.")
    st.stop()

ui.objective_box(
    "To model NIFTY50 volume jointly with currency, commodity, and monetary-policy variables "
    "in a system where each variable can depend on lagged values of all the others — and to "
    "test formally whether each macro variable has statistically significant predictive power "
    "over volume (Granger causality)."
)
ui.methodology_box(
    "Granger causality tests whether past values of X improve the prediction of Y beyond Y's "
    "own past — it is a statistical, predictive notion of causality, not a claim about true "
    "economic causation. IRF (Impulse Response Functions) show how a one-time shock to one "
    "variable propagates through the system over time; FEVD (Forecast Error Variance "
    "Decomposition) shows what share of each variable's forecast error is attributable to "
    "shocks in each variable in the system."
)

if not st.session_state.get("data_ready"):
    st.warning("Go to the Home page and load the data first.")
    st.stop()

df_market = st.session_state["df_market"]
df_macro = st.session_state["df_macro"]
merged = dp.build_inner_merge(df_market, df_macro)

VAR_COLS = ["NIFTY50_Volume", "USD_INR_Daily_Return", "GOLD_Daily_Return",
            "BRENT_CRUDE_Daily_Return", "Monetary_Policy_Shock", "Inflation_Proxy_YoY"]
available_cols = [c for c in VAR_COLS if c in merged.columns]

if len(available_cols) < 2:
    st.error("Not enough of the required VAR variables are present in the merged dataset.")
    st.stop()

var_data = merged.set_index("Date")[available_cols].dropna()

tab1, tab2, tab3, tab4 = st.tabs(["Stationarity", "VAR Model & Forecast", "IRF & FEVD", "Granger Causality"])

with tab1:
    st.write("ADF test on each variable (level):")
    rows = []
    for col in var_data.columns:
        result = adfuller(var_data[col].dropna())
        rows.append({"Variable": col, "ADF Statistic": result[0], "P-value": result[1],
                      "Stationary": "Yes" if result[1] < 0.05 else "No"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    var_diff = var_data.diff().dropna()
    st.write("First-differenced series used for VAR (all variables jointly differenced):")
    st.dataframe(var_diff.head(), use_container_width=True)
    st.session_state["var_diff"] = var_diff

with tab2:
    var_diff = st.session_state.get("var_diff")
    if var_diff is None:
        st.info("Visit the Stationarity tab first.")
        st.stop()

    max_lag = st.slider("Max lag to test", 2, 10, 10)
    model = VAR(var_diff)
    with st.spinner("Selecting optimal lag order..."):
        lag_order = model.select_order(maxlags=max_lag)
    st.text(lag_order.summary().as_text())

    best_lag = max(1, lag_order.selected_orders["bic"])
    st.write(f"**Optimal lag (BIC):** {best_lag}")

    train_size = int(len(var_diff) * 0.80)
    train_var = var_diff.iloc[:train_size]
    test_var = var_diff.iloc[train_size:]

    var_model = VAR(train_var).fit(best_lag)
    st.text(str(var_model.summary())[:3000])

    forecast = var_model.forecast(y=train_var.values[-best_lag:], steps=len(test_var))
    forecast_df = pd.DataFrame(forecast, columns=var_diff.columns, index=test_var.index)

    target_col = st.selectbox("Variable to plot", var_diff.columns.tolist())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_var.index, y=train_var[target_col], name="Training"))
    fig.add_trace(go.Scatter(x=test_var.index, y=test_var[target_col], name="Actual"))
    fig.add_trace(go.Scatter(x=test_var.index, y=forecast_df[target_col], name="VAR Forecast"))
    fig.update_layout(title=f"VAR forecast — {target_col} (differenced)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Durbin-Watson (residual autocorrelation)")
    dw = durbin_watson(var_model.resid)
    st.dataframe(pd.DataFrame({"Variable": var_diff.columns, "Durbin-Watson": np.round(dw, 3)}), use_container_width=True)

    st.session_state["var_model"] = var_model
    st.session_state["var_diff_cols"] = var_diff.columns.tolist()

with tab3:
    var_model = st.session_state.get("var_model")
    if var_model is None:
        st.info("Visit the VAR Model & Forecast tab first.")
        st.stop()

    periods = st.slider("IRF/FEVD periods", 5, 20, 10)
    irf = var_model.irf(periods)
    fevd = var_model.fevd(periods)

    st.subheader("Impulse Response Function")
    fig_irf = irf.plot()
    st.pyplot(fig_irf)

    st.subheader("Forecast Error Variance Decomposition")
    fig_fevd = fevd.plot()
    st.pyplot(fig_fevd)

with tab4:
    var_diff = st.session_state.get("var_diff")
    if var_diff is None:
        st.info("Visit the Stationarity tab first.")
        st.stop()

    st.write("Granger causality: does each variable help predict `NIFTY50_Volume`?")
    maxlag = st.slider("Max lag", 1, 8, 5, key="granger_lag")
    causal_vars = [c for c in var_diff.columns if c != "NIFTY50_Volume"]

    results = []
    for var in causal_vars:
        try:
            test_result = grangercausalitytests(var_diff[["NIFTY50_Volume", var]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[lag][0]["ssr_ftest"][1], 4) for lag in range(1, maxlag + 1)]
            best_p = min(p_values)
            results.append({"Variable": var, "Best p-value (lags 1-{})".format(maxlag): best_p,
                             "Granger-causes Volume (5%)": "Yes" if best_p < 0.05 else "No"})
        except Exception as e:
            results.append({"Variable": var, "Best p-value (lags 1-{})".format(maxlag): np.nan,
                             "Granger-causes Volume (5%)": "Error"})

    st.dataframe(pd.DataFrame(results), use_container_width=True)

    st.divider()
    ui.conclusion_box(
        "Variables flagged 'Yes' have statistically significant lead-lag predictive power over "
        "NIFTY50 volume at the 5% level. This doesn't prove economic causation, but it does "
        "justify including them as exogenous predictors in a volume-forecasting system."
    )
