import streamlit as st
import plotly.express as px

from utils import data_pipeline as dp
from utils import theme, ui

theme.inject_css()
ui.page_header(
    "Multicollinearity — Variance Inflation Factor",
    "Removing redundant, highly correlated predictors so downstream models are stable and interpretable.",
    "🧮",
)

if not st.session_state.get("data_ready"):
    st.warning("Go to the Home page and load the data first.")
    st.stop()

ui.objective_box(
    "Financial datasets like this one contain many variables that move almost identically "
    "(e.g. NIFTY50 and SENSEX). Feeding all of them into a regression or clustering model "
    "at once causes multicollinearity, which inflates coefficient variance and makes the model "
    "unstable. VIF quantifies exactly how redundant each variable is."
)
ui.methodology_box(
    "For each variable, VIF = 1 / (1 - R²) where R² comes from regressing that variable on "
    "all the others. A VIF above the chosen threshold (commonly 10) signals the variable is "
    "largely explained by the rest of the feature set and can be safely dropped without losing "
    "much information."
)

df_market = st.session_state["df_market"]
df_macro = st.session_state["df_macro"]
merged = dp.build_inner_merge(df_market, df_macro)

st.caption(
    f"Uses the inner-join dataset ({merged.shape[0]} rows × {merged.shape[1]} columns) — "
    "the working dataset for every downstream stage, as in Step 5 of the Clustering & Regression notebook."
)

threshold = st.slider("VIF threshold", 5.0, 20.0, 10.0, 0.5)
result = dp.compute_vif(merged, vif_threshold=threshold)

vif_table, high_vif, low_vif = result["vif_table"], result["high_vif"], result["low_vif"]

c1, c2, c3 = st.columns(3)
c1.metric("Total predictors", len(vif_table))
c2.metric(f"High VIF (> {threshold})", len(high_vif))
c3.metric(f"Retained after removal", result["X_reduced"].shape[1])

tab1, tab2 = st.tabs(["VIF Table", "High vs Low VIF"])

with tab1:
    fig = px.bar(vif_table.head(30), x="VIF", y="Variable", orientation="h",
                 title="Top 30 variables by VIF")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=700)
    fig.add_vline(x=threshold, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(vif_table, use_container_width=True, height=400)

with tab2:
    st.subheader(f"High VIF variables (> {threshold}) — removed")
    st.dataframe(high_vif, use_container_width=True)
    st.subheader(f"Low VIF variables (≤ {threshold}) — retained")
    st.dataframe(low_vif, use_container_width=True)

st.session_state["dated_reduced"] = result["dated_reduced"]
st.session_state["X_reduced"] = result["X_reduced"]
st.session_state["vif_threshold_used"] = threshold

st.success(
    f"Reduced feature matrix ready: {result['X_reduced'].shape[0]} rows × "
    f"{result['X_reduced'].shape[1]} columns. This feeds PCA, clustering, and the "
    "regularized-regression / ANN pages."
)

with st.expander("Download reduced dataset"):
    st.download_button(
        "Download Dataset_After_VIF_Removal.csv",
        result["dated_reduced"].to_csv(index=False),
        file_name="Dataset_After_VIF_Removal.csv",
        mime="text/csv",
    )

st.divider()
ui.conclusion_box(
    "Removing high-VIF variables leaves a lean, low-redundancy feature set. This is the exact "
    "dataset used for PCA, clustering, the volume-based regressions, and the ANN model — "
    "keeping every downstream stage of the pipeline consistent."
)
