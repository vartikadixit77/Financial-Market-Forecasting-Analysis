import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils import data_pipeline as dp
from utils import theme, ui

theme.inject_css()
ui.page_header(
    "Clustering — Market Regime Detection",
    "Grouping historical weeks into distinct market regimes using four unsupervised algorithms.",
    "🧩",
)

if not st.session_state.get("data_ready"):
    st.warning("Go to the Home page and load the data first.")
    st.stop()

ui.objective_box(
    "To identify whether the market moves through distinct, recurring 'regimes' (e.g. calm bull "
    "markets vs. high-stress crisis periods) using only the data's own structure — no labels are "
    "given to the algorithm. Four different clustering methods are compared so the conclusion "
    "doesn't rest on any single algorithm's assumptions."
)

if "X_pca" not in st.session_state or "dated_reduced" not in st.session_state:
    st.info("Visit the VIF page then the PCA page first to build the PCA feature matrix.")
    st.stop()

X_pca = st.session_state["X_pca"]
dated_reduced = st.session_state["dated_reduced"]

important_features = [c for c in [
    "NIFTY50_Close", "SENSEX_Close", "NIFTY_BANK_Close", "NIFTY_IT_Close",
    "RELIANCE_Close", "TCS_Close", "HDFCBANK_Close", "USD_INR_Close",
    "GOLD_Close", "BRENT_CRUDE_Close", "RBI_Repo_Rate", "Inflation_Proxy_YoY",
] if c in dated_reduced.columns]


def scatter_clusters(labels, title):
    df_plot = pd.DataFrame(X_pca[:, :2], columns=["PC1", "PC2"])
    df_plot["Cluster"] = labels.astype(str)
    fig = px.scatter(df_plot, x="PC1", y="PC2", color="Cluster", opacity=0.7, title=title)
    return fig


def profile_table(labels, colname):
    df = dated_reduced.copy()
    df[colname] = labels
    profile = df.groupby(colname).mean(numeric_only=True).round(2)
    return profile


ui.methodology_box(
    "All four algorithms run on the same PCA-transformed feature matrix. **K-Means** and "
    "**Hierarchical (Ward)** assume roughly spherical, similarly-sized clusters. **Gaussian "
    "Mixture** allows elliptical, overlapping clusters with soft assignment. **DBSCAN** finds "
    "clusters of arbitrary shape and explicitly flags outliers as noise rather than forcing "
    "them into a cluster. Comparing all four guards against conclusions that are an artefact "
    "of one algorithm's assumptions."
)

tabs = st.tabs(["K-Means", "Hierarchical", "Gaussian Mixture", "DBSCAN", "Comparison"])

# ---------------- K-Means ----------------
with tabs[0]:
    st.subheader("K-Means Clustering")
    ks, wcss, sil = dp.kmeans_selection_curves(X_pca)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(x=ks, y=wcss, markers=True, title="Elbow curve (WCSS)", labels={"x": "k", "y": "WCSS"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.line(x=ks, y=sil, markers=True, title="Silhouette score by k", labels={"x": "k", "y": "Silhouette"})
        st.plotly_chart(fig, use_container_width=True)

    k = st.slider("Number of clusters (k)", 2, 10, ks[int(np.argmax(sil))], key="kmeans_k")
    labels, score, model = dp.run_kmeans(X_pca, k)
    st.metric("Silhouette score", round(score, 4))
    st.plotly_chart(scatter_clusters(labels, f"K-Means clusters (k={k})"), use_container_width=True)

    st.subheader("Cluster distribution")
    st.bar_chart(pd.Series(labels).value_counts().sort_index())

    st.subheader("Cluster profile (key market variables)")
    profile = profile_table(labels, "KMeans_Cluster")
    st.dataframe(profile[important_features] if important_features else profile, use_container_width=True)

    st.session_state["kmeans_labels"] = labels

# ---------------- Hierarchical ----------------
with tabs[1]:
    st.subheader("Hierarchical (Agglomerative, Ward linkage)")
    k2 = st.slider("Number of clusters", 2, 10, 4, key="hier_k")
    labels2, score2, model2 = dp.run_hierarchical(X_pca, k2)
    st.metric("Silhouette score", round(score2, 4))
    st.plotly_chart(scatter_clusters(labels2, f"Hierarchical clusters (k={k2})"), use_container_width=True)

    st.subheader("Cluster distribution")
    st.bar_chart(pd.Series(labels2).value_counts().sort_index())

    st.subheader("Cluster profile (key market variables)")
    profile2 = profile_table(labels2, "Hier_Cluster")
    st.dataframe(profile2[important_features] if important_features else profile2, use_container_width=True)

    st.session_state["hier_labels"] = labels2

# ---------------- GMM ----------------
with tabs[2]:
    st.subheader("Gaussian Mixture Model")
    ks3, sil3 = dp.gmm_selection_curve(X_pca)
    fig = px.line(x=ks3, y=sil3, markers=True, title="Silhouette score by number of components")
    st.plotly_chart(fig, use_container_width=True)

    k3 = st.slider("Number of components", 2, 10, ks3[int(np.argmax(sil3))], key="gmm_k")
    labels3, score3, model3 = dp.run_gmm(X_pca, k3)
    st.metric("Silhouette score", round(score3, 4))
    st.plotly_chart(scatter_clusters(labels3, f"GMM clusters (k={k3})"), use_container_width=True)

    st.subheader("Cluster distribution")
    st.bar_chart(pd.Series(labels3).value_counts().sort_index())

    st.subheader("Cluster profile (key market variables)")
    profile3 = profile_table(labels3, "GMM_Cluster")
    st.dataframe(profile3[important_features] if important_features else profile3, use_container_width=True)

    st.session_state["gmm_labels"] = labels3

# ---------------- DBSCAN ----------------
with tabs[3]:
    st.subheader("DBSCAN")
    k_dist = dp.k_distance(X_pca, k=5)
    fig = px.line(y=k_dist, title="k-distance graph (look for the elbow)", labels={"y": "5th nearest neighbor distance", "index": "Points sorted"})
    st.plotly_chart(fig, use_container_width=True)

    eps = st.slider("eps", 0.1, float(max(2.0, np.percentile(k_dist, 95))), float(np.median(k_dist)), 0.1)
    min_samples = st.slider("min_samples", 2, 20, 5)
    labels4, score4, n_clusters4, noise4 = dp.run_dbscan(X_pca, eps, min_samples)

    c1, c2, c3 = st.columns(3)
    c1.metric("Clusters found", n_clusters4)
    c2.metric("Noise points", noise4)
    c3.metric("Silhouette score", round(score4, 4) if pd.notna(score4) else "N/A")

    st.plotly_chart(scatter_clusters(labels4, f"DBSCAN clusters (eps={eps}, min_samples={min_samples})"), use_container_width=True)

    st.subheader("Cluster distribution")
    st.bar_chart(pd.Series(labels4).value_counts().sort_index())

    st.session_state["dbscan_labels"] = labels4

# ---------------- Comparison ----------------
with tabs[4]:
    st.subheader("Comparative analysis of clustering algorithms")
    needed = ["kmeans_labels", "hier_labels", "gmm_labels", "dbscan_labels"]
    if not all(k in st.session_state for k in needed):
        st.info("Visit each clustering tab at least once so all four algorithms have run.")
    else:
        comp = dp.clustering_comparison(
            X_pca,
            st.session_state["kmeans_labels"],
            st.session_state["hier_labels"],
            st.session_state["gmm_labels"],
            st.session_state["dbscan_labels"],
        )
        st.dataframe(comp, use_container_width=True)

        best = comp.loc[comp["Silhouette Score"].idxmax()]
        st.success(f"Best algorithm by silhouette score: **{best['Algorithm']}** ({best['Silhouette Score']})")

        fig = px.bar(comp, x="Algorithm", y="Silhouette Score", title="Silhouette score comparison")
        st.plotly_chart(fig, use_container_width=True)

        ui.conclusion_box(
            f"**{best['Algorithm']}** produces the most internally consistent, well-separated "
            "clusters on this feature set (highest silhouette, and a healthy Calinski-Harabasz "
            "score). DBSCAN's noise points are worth a second look on their own — they typically "
            "correspond to sudden shock weeks that don't fit any recurring regime."
        )
        ui.business_box(
            "Each cluster can be read as a distinct market regime (e.g. calm, elevated stress, "
            "crisis). Knowing which regime the market is currently in helps calibrate risk "
            "limits, hedging, and expected volatility — a regime-aware model typically "
            "outperforms a single one-size-fits-all model across a full market cycle."
        )
