import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from utils import data_pipeline as dp
from utils import theme, ui

theme.inject_css()
ui.page_header(
    "OLS vs. Gradient Descent vs. Ridge vs. Lasso",
    "Four ways to model NIFTY50 trading volume from an engineered feature set — which generalizes best?",
    "📦",
)

if not st.session_state.get("data_ready"):
    st.warning("Go to the Home page and load the data first.")
    st.stop()

ui.objective_box(
    "To compare a classical OLS regression against three optimization-based alternatives on "
    "the same feature-engineered dataset, and identify which regularization strategy (if any) "
    "improves out-of-sample accuracy for forecasting NIFTY50 trading volume."
)
ui.methodology_box(
    "Lag features, rolling means/volatility, interaction, and log terms are engineered from "
    "the VIF-reduced dataset. **Ridge** shrinks coefficients toward zero (handles remaining "
    "collinearity); **Lasso** can shrink some coefficients to exactly zero (automatic feature "
    "selection); **Gradient Descent** (SGDRegressor) fits the same linear form via iterative "
    "optimization rather than a closed-form solution — useful at larger scale, though here it's "
    "mainly a sanity check that the closed-form OLS solution is near-optimal."
)

if not st.session_state.get("data_ready"):
    st.warning("Go to the Home page and load the data first.")
    st.stop()

if "dated_reduced" not in st.session_state:
    st.info("Visit the VIF & Multicollinearity page first to build the reduced feature matrix.")
    st.stop()

dated_reduced = st.session_state["dated_reduced"]

if "NIFTY50_Volume" not in dated_reduced.columns:
    st.error("`NIFTY50_Volume` was removed during VIF reduction, so this page can't run at the current VIF threshold. Lower the threshold on the VIF page.")
    st.stop()

enhanced_df = dp.build_enhanced_regression_dataset(dated_reduced)
st.write(f"Enhanced feature-engineered dataset: **{enhanced_df.shape[0]} rows × {enhanced_df.shape[1]} columns** "
         "(lag features, moving averages, rolling volatility, interaction and log terms).")

tab1, tab2, tab3 = st.tabs(["Correlation", "Model Comparison", "Actual vs Predicted"])

with tab1:
    numeric_df = enhanced_df.drop(columns=["Date"], errors="ignore")
    corr_target = numeric_df.corr(numeric_only=True)["NIFTY50_Volume"].drop("NIFTY50_Volume").sort_values()
    fig = px.bar(corr_target, orientation="h", title="Correlation of features with NIFTY50 Volume")
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    with st.spinner("Fitting OLS, Gradient Descent, Ridge, and Lasso..."):
        result = dp.run_regularized_regression(enhanced_df)

    comparison = result["comparison"]
    st.dataframe(comparison, use_container_width=True)

    best = comparison.loc[comparison["R2"].idxmax()]
    st.success(f"Best model by R²: **{best['Model']}** (R² = {best['R2']})")
    ui.conclusion_box(
        "When Ridge/Lasso perform close to plain OLS, it indicates multicollinearity was "
        "already well-controlled by the earlier VIF step — regularization adds robustness "
        "rather than dramatically changing the fit. A visibly better regularized R² would "
        "instead suggest the engineered feature set still has meaningful collinearity."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Ridge best alpha", round(result["ridge_alpha"], 4))
    with c2:
        st.metric("Lasso best alpha", round(result["lasso_alpha"], 4))

    fig2 = px.bar(comparison, x="Model", y="R2", title="Regression model comparison (R²)")
    st.plotly_chart(fig2, use_container_width=True)

    st.session_state["regularized_regression_result"] = result

with tab3:
    if "regularized_regression_result" not in st.session_state:
        st.info("Open the Model Comparison tab first.")
    else:
        result = st.session_state["regularized_regression_result"]
        Y_test = result["Y_test"]
        model_choice = st.selectbox("Model", list(result["predictions"].keys()))
        Y_pred = result["predictions"][model_choice]

        fig = px.scatter(x=Y_test, y=Y_pred, labels={"x": "Actual", "y": "Predicted"}, title=f"{model_choice}: Actual vs Predicted")
        min_v, max_v = Y_test.min(), Y_test.max()
        fig.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v, line=dict(dash="dash", color="red"))
        st.plotly_chart(fig, use_container_width=True)

        residuals = Y_test.values - Y_pred
        fig2 = px.histogram(residuals, nbins=30, title=f"{model_choice}: residual distribution")
        st.plotly_chart(fig2, use_container_width=True)
