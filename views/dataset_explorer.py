import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils import data_pipeline as dp
from utils import theme, ui

theme.inject_css()
ui.page_header(
    "Dataset Explorer",
    "Browse, filter, and download the underlying data behind every analysis in this dashboard.",
    "🗂️",
)

if not st.session_state.get("data_ready"):
    st.warning("Go to the Home page and load the data first.")
    st.stop()

df_market = st.session_state["df_market"]
df_macro = st.session_state["df_macro"]
merged = dp.build_inner_merge(df_market, df_macro)

ui.objective_box(
    "To give a transparent, self-service view of the exact data every model in this dashboard "
    "is trained on — before any transformation — so results can be sanity-checked against the "
    "raw numbers."
)

ui.kpi_row([
    ("Rows", merged.shape[0]),
    ("Columns", merged.shape[1]),
    ("Date Range", f"{merged['Date'].min().date()} → {merged['Date'].max().date()}"),
    ("Missing Cells", int(merged.isnull().sum().sum())),
])

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📋 Data Viewer", "🧬 Data Types", "❓ Missing Values", "📐 Summary Statistics", "🔥 Correlation Heatmap"]
)

with tab1:
    search = st.text_input("🔎 Filter columns (type part of a column name)", "")
    cols = [c for c in merged.columns if search.lower() in c.lower()] if search else merged.columns.tolist()
    min_date, max_date = merged["Date"].min().date(), merged["Date"].max().date()
    date_range = st.date_input(
        "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date
    view = merged[(merged["Date"].dt.date >= start_date) & (merged["Date"].dt.date <= end_date)][cols]
    st.dataframe(view, use_container_width=True, height=420)
    st.download_button(
        "⬇️ Download this view as CSV", view.to_csv(index=False),
        file_name="nifty_dataset_view.csv", mime="text/csv",
    )

with tab2:
    dtypes_df = pd.DataFrame({"Column": merged.columns, "Data Type": merged.dtypes.astype(str).values})
    st.dataframe(dtypes_df, use_container_width=True, height=420)

with tab3:
    missing = pd.DataFrame({
        "Missing Values": merged.isnull().sum(),
        "Percentage": (merged.isnull().sum() / len(merged)) * 100,
    })
    missing = missing[missing["Missing Values"] > 0].sort_values("Percentage", ascending=False)
    if len(missing):
        st.dataframe(missing, use_container_width=True)
        fig = px.bar(missing, y=missing.index, x="Percentage", orientation="h", title="Missing value % by column")
        st.plotly_chart(theme.style_fig(fig), use_container_width=True)
    else:
        st.success("✅ No missing values in the merged dataset.")

with tab4:
    st.dataframe(merged.describe().T, use_container_width=True, height=500)

with tab5:
    numeric_cols = merged.select_dtypes(include="number").columns.tolist()
    default_cols = [c for c in [
        "NIFTY50_Close", "NIFTY50_Volume", "SENSEX_Close", "USD_INR_Close",
        "GOLD_Close", "BRENT_CRUDE_Close", "RBI_Repo_Rate", "Inflation_Proxy_YoY",
    ] if c in numeric_cols]
    pick = st.multiselect("Variables to include", numeric_cols, default=default_cols or numeric_cols[:10])
    if len(pick) >= 2:
        corr = merged[pick].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         title="Correlation heatmap")
        st.plotly_chart(theme.style_fig(fig), use_container_width=True)
        ui.interpretation_box(
            "Values close to +1 or -1 indicate two variables move almost in lockstep — a red "
            "flag for multicollinearity if both are used as independent predictors in the same "
            "regression (addressed formally on the VIF page)."
        )
    else:
        st.info("Pick at least 2 variables to render the heatmap.")

st.divider()
ui.conclusion_box(
    "This explorer is the ground truth for every number reported elsewhere in the dashboard — "
    "if a model's result looks surprising, this is the place to check the raw inputs first."
)
