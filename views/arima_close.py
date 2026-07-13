import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import scipy.stats as stats

from utils import data_pipeline as dp
from utils import theme, ui

theme.inject_css()
ui.page_header(
    "ARIMA(1,1,1) — NIFTY50 Close Price",
    "A classical econometric benchmark: can past prices alone forecast future prices?",
    "📊",
)

if not st.session_state.get("data_ready"):
    st.warning("Go to the Home page and load the data first.")
    st.stop()

ui.objective_box(
    "To establish a simple, interpretable econometric benchmark for NIFTY50 Close price "
    "forecasting before comparing against more complex models later in the dashboard."
)
ui.methodology_box(
    "ARIMA(p,d,q) models a series as a combination of its own past values (AR) and past "
    "forecast errors (MA), after differencing d times to remove trend. Here, d=1 was selected "
    "after confirming the raw series is non-stationary but its first difference is — see the "
    "Stationarity tab."
)

if not st.session_state.get("data_ready"):
    st.warning("Go to the Home page and load the data first.")
    st.stop()

df_market = st.session_state["df_market"]
df_macro = st.session_state["df_macro"]
merged = dp.build_inner_merge(df_market, df_macro)

ts_df = merged.set_index("Date")[["NIFTY50_Close"]].copy()
ts = ts_df["NIFTY50_Close"]

tab1, tab2, tab3, tab4 = st.tabs(["Stationarity", "Model Fit", "Forecast Evaluation", "Residual Diagnostics"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts, mode="lines", name="NIFTY50 Close"))
    fig.update_layout(title="NIFTY50 Close over time")
    st.plotly_chart(fig, use_container_width=True)

    adf = adfuller(ts)
    st.write(f"**ADF Statistic:** {adf[0]:.4f}  |  **P-value:** {adf[1]:.4g}")
    st.write("✅ Stationary" if adf[1] < 0.05 else "❌ Non-stationary — differencing required")

    ts_diff = ts.diff().dropna()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=ts_diff.index, y=ts_diff, mode="lines", name="First difference"))
    fig2.update_layout(title="First-order differenced series")
    st.plotly_chart(fig2, use_container_width=True)

    adf_diff = adfuller(ts_diff)
    st.write(f"**ADF Statistic (differenced):** {adf_diff[0]:.4f}  |  **P-value:** {adf_diff[1]:.4g}")
    st.write("✅ Differenced series is stationary → d = 1" if adf_diff[1] < 0.05 else "❌ Still non-stationary")

with tab2:
    train_size = int(len(ts_df) * 0.80)
    train = ts_df.iloc[:train_size]
    test = ts_df.iloc[train_size:]
    st.write(f"Training observations: **{len(train)}**  |  Testing observations: **{len(test)}**")

    with st.spinner("Fitting ARIMA(1,1,1)..."):
        arima_result = ARIMA(train["NIFTY50_Close"], order=(1, 1, 1)).fit()
    st.text(arima_result.summary().as_text())
    st.session_state["arima_close_result"] = arima_result
    st.session_state["arima_close_train_test"] = (train, test)

with tab3:
    train, test = st.session_state["arima_close_train_test"]
    arima_result = st.session_state["arima_close_result"]
    forecast = arima_result.forecast(steps=len(test))
    forecast = pd.Series(forecast.values, index=test.index)

    m = dp.evaluate_forecast(test["NIFTY50_Close"], forecast)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAE", f"{m['MAE']:.2f}")
    c2.metric("RMSE", f"{m['RMSE']:.2f}")
    c3.metric("MAPE", f"{m['MAPE']:.2f}%")
    c4.metric("R²", f"{m['R2']:.4f}")
    c5.metric("AIC", f"{arima_result.aic:.1f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train["NIFTY50_Close"], name="Training Data"))
    fig.add_trace(go.Scatter(x=test.index, y=test["NIFTY50_Close"], name="Actual"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast, name="Forecast"))
    fig.update_layout(title="ARIMA(1,1,1) Forecast vs Actual")
    st.plotly_chart(fig, use_container_width=True)

    st.download_button("Download ARIMA_Forecast.csv",
                        pd.DataFrame({"Date": forecast.index, "Actual": test["NIFTY50_Close"].values, "Forecast": forecast.values}).to_csv(index=False),
                        file_name="ARIMA_Forecast.csv")

with tab4:
    arima_result = st.session_state["arima_close_result"]
    residuals = arima_result.resid

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=residuals, mode="lines", name="Residuals"))
        fig.update_layout(title="Residuals of ARIMA model")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        import plotly.express as px
        fig2 = px.histogram(residuals, nbins=30, title="Distribution of residuals")
        st.plotly_chart(fig2, use_container_width=True)

    qq = stats.probplot(residuals, dist="norm")
    qq_df = pd.DataFrame({"Theoretical": qq[0][0], "Sample": qq[0][1]})
    import plotly.express as px
    fig3 = px.scatter(qq_df, x="Theoretical", y="Sample", title="Q-Q Plot of residuals")
    st.plotly_chart(fig3, use_container_width=True)

    lb = acorr_ljungbox(residuals, lags=[10], return_df=True)
    st.subheader("Ljung–Box Test")
    st.dataframe(lb, use_container_width=True)
    p_val = lb["lb_pvalue"].iloc[0]
    st.write("✅ Residuals resemble white noise (good model)." if p_val > 0.05 else "⚠️ Some autocorrelation remains — model may be improved.")

st.divider()
ui.conclusion_box(
    "ARIMA(1,1,1) captures the short-run persistence in NIFTY50 Close but, like any linear "
    "model, cannot adapt to sudden regime shifts. Its MAE/RMSE/R² here serve as the baseline "
    "the ANN model is measured against on the Model Comparison page."
)
