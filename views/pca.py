import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import data_pipeline as dp
from utils import theme, ui

theme.inject_css()
ui.page_header(
    "Principal Component Analysis",
    "Compressing the VIF-reduced feature set into a smaller number of uncorrelated dimensions for clustering.",
    "🧭",
)

if not st.session_state.get("data_ready"):
    st.warning("Go to the Home page and load the data first.")
    st.stop()

ui.objective_box(
    "Even after removing high-VIF variables, the remaining features still overlap. PCA "
    "re-expresses them as a smaller set of uncorrelated components that capture most of the "
    "original variance — giving clustering algorithms a cleaner, lower-dimensional space to work in."
)

if "X_reduced" not in st.session_state:
    st.info("Visit the VIF & Multicollinearity page first to build the reduced feature matrix.")
    st.stop()

X_reduced = st.session_state["X_reduced"]
variance_target = st.slider("Variance to retain", 0.80, 0.99, 0.95, 0.01)

pca_result = dp.compute_pca(X_reduced, variance=variance_target)

c1, c2 = st.columns(2)
c1.metric("Input dimensions", X_reduced.shape[1])
c2.metric("Components retained", pca_result["n_components"])

fig = go.Figure()
fig.add_trace(go.Bar(
    x=[f"PC{i+1}" for i in range(len(pca_result["explained_variance_ratio"]))],
    y=pca_result["explained_variance_ratio"],
    name="Explained variance",
))
fig.add_trace(go.Scatter(
    x=[f"PC{i+1}" for i in range(len(pca_result["cumulative_variance"]))],
    y=pca_result["cumulative_variance"],
    name="Cumulative variance",
    yaxis="y2",
    mode="lines+markers",
))
fig.update_layout(
    title="Explained variance by component",
    yaxis=dict(title="Explained variance ratio"),
    yaxis2=dict(title="Cumulative variance", overlaying="y", side="right", range=[0, 1.05]),
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("First two principal components")
pc_df = pd.DataFrame(pca_result["X_pca"][:, :2], columns=["PC1", "PC2"])
fig2 = px.scatter(pc_df, x="PC1", y="PC2", opacity=0.6, title="Data projected onto PC1 vs PC2")
st.plotly_chart(fig2, use_container_width=True)

st.session_state["X_pca"] = pca_result["X_pca"]
st.session_state["pca_variance_target"] = variance_target

st.success(f"PCA-transformed feature matrix ready: {pca_result['X_pca'].shape[0]} rows × {pca_result['n_components']} components. Feeds the Clustering page.")

st.divider()
ui.interpretation_box(
    "If the first 2-3 components already capture most of the cumulative variance, it means a "
    "small number of underlying market forces (e.g. broad index momentum, currency/commodity "
    "co-movement) drive most of the variation across all these variables together."
)
ui.conclusion_box(
    "The PCA-transformed feature matrix is what every clustering algorithm on the next page "
    "is actually trained on — not the raw features — which is standard practice for "
    "distance-based clustering on correlated financial data."
)
