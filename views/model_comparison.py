import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils import theme, ui

theme.inject_css()
ui.page_header(
    "Model Comparison",
    "Every forecasting model in this dashboard, side by side, on the same test period and metric set.",
    "🏆",
)

if not st.session_state.get("data_ready"):
    st.warning("Go to the Home page and load the data first.")
    st.stop()

ui.objective_box(
    "To bring ARIMA, SARIMA, VAR, GARCH, and the ANN together in one table using identical "
    "evaluation metrics (MAE, MSE, RMSE, MAPE, R²) computed on the identical 91-observation "
    "test period — so the 'best model' claim is actually a fair, apples-to-apples comparison."
)

comparison = st.session_state.get("final_comparison_table")

if comparison is None:
    st.info(
        "This table is populated once the ANN model has been trained. Open **ANN Forecasting "
        "Model** in the sidebar, click **Train / retrain ANN**, then return here."
    )
    st.stop()

st.dataframe(comparison, use_container_width=True)

valid = comparison.dropna(subset=["RMSE"])
best_model = valid["RMSE"].idxmin()
best_row = valid.loc[best_model]

ui.kpi_row([
    ("🏆 Best Model", best_model),
    ("RMSE", f"{best_row['RMSE']:,.0f}"),
    ("MAPE", f"{best_row['MAPE']:.2f}%"),
    ("R²", f"{best_row['R2']:.3f}"),
])

fig = px.bar(valid.reset_index(), x="index", y="RMSE", color="index",
             title="RMSE by model (lower is better)", labels={"index": "Model"})
fig.update_layout(showlegend=False)
st.plotly_chart(theme.style_fig(fig), use_container_width=True)

fig2 = px.bar(valid.reset_index(), x="index", y="R2", color="index",
              title="R² by model (higher is better)", labels={"index": "Model"})
fig2.update_layout(showlegend=False)
st.plotly_chart(theme.style_fig(fig2), use_container_width=True)

ui.conclusion_box(
    f"**{best_model}** achieves the lowest RMSE on this held-out test period. If it is the ANN, "
    "this indicates the non-linear, regime-dependent structure in NIFTY50 trading volume is "
    "real and material — a fixed linear specification (ARIMA/SARIMA/VAR) cannot fully capture "
    "it. If a classical model wins instead, that's an equally valid and interesting finding: it "
    "would suggest the extra flexibility of a neural network isn't paying off on a series this "
    "short, and a simpler, more interpretable model is the more defensible choice."
)
ui.business_box(
    "For a dissertation defense or stakeholder presentation, lead with this table: it "
    "demonstrates the core empirical contribution — a rigorous, leakage-free comparison "
    "between econometric and machine-learning approaches on the same market data."
)

st.download_button(
    "⬇️ Download Final_Model_Comparison.csv", comparison.to_csv(),
    file_name="Final_Model_Comparison.csv", mime="text/csv",
)
