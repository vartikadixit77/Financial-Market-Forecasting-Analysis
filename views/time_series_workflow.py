import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from utils import data_pipeline as dp
from utils import theme, ui

theme.inject_css()
ui.page_header(
    "Time Series Forecasting Workflow — NIFTY50 Trading Volume",
    "Building up from a naive baseline through AR, MA, ARMA, ARIMA, to seasonal SARIMA.",
    "⏱️",
)

if not st.session_state.get("data_ready"):
    st.warning("Go to the Home page and load the data first.")
    st.stop()

ui.objective_box(
    "To systematically test whether NIFTY50 trading volume is stationary, identify the best "
    "linear time-series specification by information criteria (AIC/BIC), and check whether "
    "adding a 52-week seasonal component (SARIMA) improves on plain ARIMA."
)

if not st.session_state.get("data_ready"):
    st.warning("Go to the Home page and load the data first.")
    st.stop()

df_market = st.session_state["df_market"]
df_macro = st.session_state["df_macro"]
merged = dp.build_inner_merge(df_market, df_macro)

ts_full = merged.set_index("Date")["NIFTY50_Volume"]

tab1, tab2, tab3 = st.tabs(["Exploratory & Stationarity", "AR / MA / ARMA / ARIMA", "SARIMA"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(x=ts_full.index, y=ts_full.values, title="NIFTY50 Trading Volume — full series",
                       labels={"x": "Date", "y": "Volume"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(ts_full, nbins=30, title="Distribution of NIFTY50 Trading Volume")
        st.plotly_chart(fig, use_container_width=True)

    monthly = ts_full.resample("ME").mean()
    fig2 = px.line(x=monthly.index, y=monthly.values, title="Monthly average volume", labels={"x": "Date", "y": "Avg volume"})
    st.plotly_chart(fig2, use_container_width=True)

    adf = adfuller(ts_full)
    st.write(f"**ADF (original series):** stat={adf[0]:.4f}, p={adf[1]:.4g} → "
             + ("Stationary" if adf[1] < 0.05 else "Non-stationary"))

    ts_diff = ts_full.diff().dropna()
    adf_diff = adfuller(ts_diff)
    st.write(f"**ADF (first difference):** stat={adf_diff[0]:.4f}, p={adf_diff[1]:.4g} → "
             + ("Stationary → d=1" if adf_diff[1] < 0.05 else "Still non-stationary"))

with tab2:
    train_size = int(len(ts_full) * 0.80)
    train = ts_full.iloc[:train_size]
    test = ts_full.iloc[train_size:]
    st.write(f"Training observations: **{len(train)}**  |  Testing observations: **{len(test)}**")

    baseline_forecast = np.repeat(train.iloc[-1], len(test))

    @st.cache_data(show_spinner=False)
    def fit_ts_models(_train_hash):
        rows = []
        forecasts = {}

        baseline_m = dp.evaluate_forecast(test, baseline_forecast)
        rows.append({"Model": "Naive Baseline", "AIC": np.nan, "BIC": np.nan, **baseline_m})
        forecasts["Naive Baseline"] = baseline_forecast

        specs = {"AR(3)": (3, 0, 0), "MA(1)": (0, 0, 1), "ARMA(3,1)": (3, 0, 1), "ARIMA(1,1,1)": (1, 1, 1)}
        for name, order in specs.items():
            fit = ARIMA(train, order=order).fit()
            fc = fit.forecast(steps=len(test))
            m = dp.evaluate_forecast(test, fc)
            rows.append({"Model": name, "AIC": fit.aic, "BIC": fit.bic, **m})
            forecasts[name] = fc.values

        return pd.DataFrame(rows), forecasts

    comparison_df, forecasts = fit_ts_models(hash(tuple(train.values.round(2))))

    st.dataframe(comparison_df.round(3), use_container_width=True)

    model_pick = st.selectbox("Visualize forecast for", list(forecasts.keys()), index=len(forecasts) - 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train.values, name="Training Data"))
    fig.add_trace(go.Scatter(x=test.index, y=test.values, name="Actual"))
    fig.add_trace(go.Scatter(x=test.index, y=forecasts[model_pick], name=f"{model_pick} Forecast"))
    fig.update_layout(title=f"{model_pick} — Forecast vs Actual")
    st.plotly_chart(fig, use_container_width=True)

    best_bic = comparison_df.dropna(subset=["BIC"]).sort_values("BIC").iloc[0]
    st.success(f"Best model by BIC among candidates: **{best_bic['Model']}**")

with tab3:
    train_size = int(len(ts_full) * 0.80)
    train = ts_full.iloc[:train_size]
    test = ts_full.iloc[train_size:]

    with st.spinner("Fitting SARIMA(1,1,1)x(1,0,1,52) — this can take a minute..."):
        sarima_fit = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 52),
                              enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    sarima_forecast = sarima_fit.forecast(steps=len(test))
    m = dp.evaluate_forecast(test, sarima_forecast)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAE", f"{m['MAE']:.1f}")
    c2.metric("RMSE", f"{m['RMSE']:.1f}")
    c3.metric("MAPE", f"{m['MAPE']:.2f}%")
    c4.metric("AIC", f"{sarima_fit.aic:.1f}")
    c5.metric("BIC", f"{sarima_fit.bic:.1f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train.values, name="Training Data"))
    fig.add_trace(go.Scatter(x=test.index, y=test.values, name="Actual"))
    fig.add_trace(go.Scatter(x=test.index, y=sarima_forecast.values, name="SARIMA Forecast"))
    fig.update_layout(title="SARIMA(1,1,1)(1,0,1,52) — Forecast vs Actual")
    st.plotly_chart(fig, use_container_width=True)

    residuals = sarima_fit.resid
    fig2 = px.histogram(residuals, nbins=30, title="SARIMA residual distribution")
    st.plotly_chart(fig2, use_container_width=True)

    st.session_state["sarima_metrics"] = m

st.divider()
ui.conclusion_box(
    "Trading volume is noisy and regime-driven, so linear models here typically show modest "
    "R² and can even score negative R² on a single held-out test window — that's expected for "
    "volume rather than price, and is exactly why the ANN model (see the ANN page) is tested "
    "as a non-linear alternative on this same series."
)
