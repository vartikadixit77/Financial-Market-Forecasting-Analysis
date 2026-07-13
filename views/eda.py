import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils import data_pipeline as dp
from utils import theme, ui

theme.inject_css()
ui.page_header(
    "Exploratory Data Analysis",
    "Understanding the shape, distribution, and relationships in the market data before any modelling begins.",
    "🔍",
)

if not st.session_state.get("data_ready"):
    st.warning("Go to the Home page and load the data first.")
    st.stop()

ui.objective_box(
    "To understand the statistical properties of the merged market microstructure "
    "and macroeconomic dataset — distributions, correlations, outliers, and data quality — "
    "before selecting variables for modelling."
)

df_market = st.session_state["df_market"]
df_macro = st.session_state["df_macro"]
merged = dp.build_asof_merge(df_market, df_macro)

st.caption(
    f"Uses the `merge_asof` (backward-fill) merged dataset — {merged.shape[0]} rows × "
    f"{merged.shape[1]} columns — as in Step 1-2 and Step 4 of the Clustering & Regression notebook."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Target Distribution", "Correlation", "Outliers", "Skew/Kurtosis & Variance"]
)

with tab1:
    st.subheader("Basic information")
    c1, c2 = st.columns(2)
    c1.metric("Rows", merged.shape[0])
    c2.metric("Columns", merged.shape[1])
    st.dataframe(merged.head(), use_container_width=True)
    st.dataframe(merged.describe().T, use_container_width=True)

    st.subheader("Missing values")
    missing = pd.DataFrame({
        "Missing Values": merged.isnull().sum(),
        "Percentage": (merged.isnull().sum() / len(merged)) * 100,
    })
    missing = missing[missing["Missing Values"] > 0].sort_values("Percentage", ascending=False)
    if len(missing):
        st.dataframe(missing, use_container_width=True)
    else:
        st.success("No missing values.")

with tab2:
    target = st.selectbox("Target variable", [c for c in merged.columns if c != "Date"],
                           index=list(merged.columns).index("NIFTY50_Close") - 1 if "NIFTY50_Close" in merged.columns else 0)
    fig = px.histogram(merged, x=target, nbins=40, title=f"Distribution of {target}", marginal="box")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(merged, x="Date", y=target, title=f"{target} over time")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    numeric_cols = merged.select_dtypes(include="number").columns.tolist()
    target2 = st.selectbox("Correlate against", numeric_cols,
                            index=numeric_cols.index("NIFTY50_Close") if "NIFTY50_Close" in numeric_cols else 0,
                            key="corr_target")
    corr = merged[numeric_cols].corr()[target2].drop(target2).sort_values(ascending=False)
    top20 = corr.head(20)
    fig3 = px.bar(top20, orientation="h", title=f"Top correlated variables with {target2}")
    fig3.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    col = st.selectbox("Column for outlier check", numeric_cols, key="outlier_col")
    fig4 = px.box(merged, y=col, title=f"Boxplot: {col}")
    st.plotly_chart(fig4, use_container_width=True)

    q1, q3 = merged[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_outliers = ((merged[col] < lower) | (merged[col] > upper)).sum()
    st.metric("IQR outlier count", int(n_outliers))

with tab5:
    numeric_df = merged.select_dtypes(include="number")
    stats_df = pd.DataFrame({
        "Skewness": numeric_df.skew(),
        "Kurtosis": numeric_df.kurtosis(),
        "Variance": numeric_df.var(),
    }).sort_values("Variance", ascending=False)
    st.dataframe(stats_df, use_container_width=True, height=500)

st.divider()
ui.interpretation_box(
    "Market variables (index closes, volumes) are heavy-tailed and show visible outliers — "
    "typical of financial time series during volatile periods. Macro variables (repo rate, "
    "inflation) move much more slowly and in steps, reflecting how policy is set."
)
ui.conclusion_box(
    "The dataset is usable for modelling but carries multicollinearity (many correlated price "
    "series) and non-stationarity (trending levels) — both addressed in the next two stages: "
    "VIF-based feature reduction and stationarity testing before time-series modelling."
)
