import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX

from utils import data_pipeline as dp
from utils import theme, ui

theme.inject_css()
ui.page_header(
    "Interactive Forecasting",
    "Project NIFTY50 trading volume forward, with a confidence band, on a horizon you choose.",
    "🔮",
)

if not st.session_state.get("data_ready"):
    st.warning("Go to the Home page and load the data first.")
    st.stop()

ui.objective_box(
    "To turn the SARIMA(1,1,1)(1,0,1,52) model — fit on the full history rather than a "
    "train/test split — into a genuinely forward-looking forecast with uncertainty bounds, "
    "so the reader can see not just a point estimate but a plausible range."
)

df_market = st.session_state["df_market"]
df_macro = st.session_state["df_macro"]
merged = dp.build_inner_merge(df_market, df_macro)
ts = merged.set_index("Date")["NIFTY50_Volume"].astype(float)

c1, c2 = st.columns([1, 3])
with c1:
    horizon = st.slider("Forecast horizon (weeks)", 4, 52, 30)
    ci_level = st.select_slider("Confidence level", options=[80, 90, 95, 99], value=95)

with st.spinner("Fitting SARIMA on the full history..."):
    full_fit = SARIMAX(
        ts, order=(1, 1, 1), seasonal_order=(1, 0, 1, 52),
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)

forecast_res = full_fit.get_forecast(steps=horizon)
mean_fc = forecast_res.predicted_mean
alpha = 1 - ci_level / 100
conf_int = forecast_res.conf_int(alpha=alpha)

fig = go.Figure()
fig.add_trace(go.Scatter(x=ts.index, y=ts.values, name="Historical Volume", line=dict(width=2)))
fig.add_trace(go.Scatter(x=mean_fc.index, y=mean_fc.values, name="Forecast", line=dict(width=2, dash="dash")))
fig.add_trace(go.Scatter(
    x=list(conf_int.index) + list(conf_int.index[::-1]),
    y=list(conf_int.iloc[:, 1]) + list(conf_int.iloc[:, 0][::-1]),
    fill="toself", fillcolor="rgba(0,194,168,0.15)", line=dict(width=0),
    name=f"{ci_level}% Confidence Interval", hoverinfo="skip",
))
fig.update_layout(title=f"NIFTY50 Trading Volume — {horizon}-Week Forecast", xaxis_title="Date", yaxis_title="Volume")
st.plotly_chart(theme.style_fig(fig), use_container_width=True)

ui.kpi_row([
    ("Horizon", f"{horizon} weeks"),
    ("Forecast at horizon end", f"{mean_fc.iloc[-1]:,.0f}"),
    (f"{ci_level}% Interval width (final week)", f"±{(conf_int.iloc[-1, 1] - mean_fc.iloc[-1]):,.0f}"),
])

ui.interpretation_box(
    "The confidence band widens further out — expected, since uncertainty compounds the "
    "further ahead the model projects. A wide band relative to the forecast value signals "
    "the point estimate should be treated as directional guidance, not a precise number."
)

forecast_df = pd.DataFrame({
    "Date": mean_fc.index, "Forecast": mean_fc.values,
    f"Lower {ci_level}%": conf_int.iloc[:, 0].values, f"Upper {ci_level}%": conf_int.iloc[:, 1].values,
})
st.dataframe(forecast_df, use_container_width=True)
st.download_button(
    "⬇️ Download forecast as CSV", forecast_df.to_csv(index=False),
    file_name=f"NIFTY50_Volume_{horizon}week_Forecast.csv", mime="text/csv",
)

st.divider()
ui.conclusion_box(
    "This forecast is a practical extension of the SARIMA benchmark evaluated earlier — useful "
    "for planning purposes, but it inherits every limitation discussed on the Model Comparison "
    "page: it assumes the historical linear/seasonal relationship continues to hold, and does "
    "not adapt to structural breaks the way a regime-aware or non-linear model might."
)
