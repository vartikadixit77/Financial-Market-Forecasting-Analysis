"""
Shared data-loading and modelling pipeline for the NIFTY dissertation dashboard.

This module reproduces, as faithfully as possible, the logic used across the
four source notebooks:
  - Clustering & Regression Analysis (EDA, VIF, PCA, clustering, OLS+diagnostics, ARIMA)
  - Ordinary Least Squares (OLS) Regression Analysis (feature engineering + regularized regression)
  - Time Series Forecasting Workflow (AR/MA/ARMA/ARIMA/SARIMA, ARCH/GARCH, VAR, Granger)
  - ANN Forecasting Model (Keras ANN vs. econometric benchmarks)

One deliberate improvement over the original notebooks: instead of re-inferring
a synthetic weekly Date index (the notebooks did this because Colab intermediate
CSVs dropped the Date column), every function here carries the real Date column
through the whole pipeline. Numbers match the notebooks; the Date axis is exact
rather than reconstructed.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

DATE_COL = "Date"
MARKET_FILENAME = "market_microstructure_dataset.csv"
MACRO_FILENAME = "50 Macroeconomic Indicators rbi.xlsx"


# ---------------------------------------------------------------------------
# STEP 1-2 (Clustering notebook): Load & clean, merge_asof (EDA-only view)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_market(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_macro(file) -> pd.DataFrame:
    df = pd.read_excel(file, sheet_name="Weekly", header=3)
    df.columns = df.columns.astype(str).str.strip().str.replace("\n", " ", regex=False)
    if "Period" in df.columns:
        df.rename(columns={"Period": "Date"}, inplace=True)
    else:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df = df.loc[:, ~df.columns.duplicated(keep="last")]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # Coerce every non-Date column to numeric up front so "-" placeholders and stray
    # strings never leak into downstream displays/Arrow serialization.
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col].replace("-", np.nan), errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def build_asof_merge(df_market: pd.DataFrame, df_macro: pd.DataFrame) -> pd.DataFrame:
    """Step 2 of the Clustering notebook: merge_asof, backward direction (522 rows).
    Used only for the exploratory EDA section."""
    m = df_market.dropna(subset=["Date"]).drop_duplicates(subset="Date").sort_values("Date").reset_index(drop=True)
    a = df_macro.dropna(subset=["Date"]).drop_duplicates(subset="Date").sort_values("Date").reset_index(drop=True)
    merged = pd.merge_asof(m, a, on="Date", direction="backward")
    return merged


@st.cache_data(show_spinner=False)
def build_inner_merge(df_market: pd.DataFrame, df_macro: pd.DataFrame) -> pd.DataFrame:
    """Step 5 of the Clustering notebook: plain inner join on Date (452 rows).
    This is the working dataset for VIF, PCA, clustering, OLS and ARIMA."""
    merged = pd.merge(df_market, df_macro, on="Date", how="inner")
    return merged.sort_values("Date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# STEP 5 (Clustering notebook): Multicollinearity / VIF
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def compute_vif(merged_df: pd.DataFrame, vif_threshold: float = 10.0):
    """Reproduces the notebook's VIF computation on ALL numeric columns
    (Date excluded), then splits into high/low VIF and drops high-VIF vars."""
    X = merged_df.drop(columns=["Date"], errors="ignore")
    X = X.select_dtypes(include=["number"])
    X = X.replace("-", np.nan).apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.ffill().bfill()
    # Drop any column that is still entirely NaN or constant (zero variance breaks VIF)
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=True) > 1]

    X_const = add_constant(X)
    vif = pd.DataFrame()
    vif["Variable"] = X_const.columns
    vif["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
    vif = vif[vif["Variable"] != "const"]
    vif = vif.sort_values("VIF", ascending=False).reset_index(drop=True)

    high_vif = vif[vif["VIF"] > vif_threshold]
    low_vif = vif[vif["VIF"] <= vif_threshold]

    X_reduced = X.drop(columns=high_vif["Variable"].tolist())
    # Carry the real Date column alongside the reduced feature matrix
    dated_reduced = pd.concat([merged_df[["Date"]].reset_index(drop=True), X_reduced.reset_index(drop=True)], axis=1)

    return {
        "vif_table": vif,
        "high_vif": high_vif,
        "low_vif": low_vif,
        "X": X,
        "X_reduced": X_reduced,
        "dated_reduced": dated_reduced,
    }


# ---------------------------------------------------------------------------
# PCA (used inside the clustering section, 95% variance retained)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def compute_pca(X_reduced: pd.DataFrame, variance: float = 0.95):
    from sklearn.decomposition import PCA

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)

    pca = PCA(n_components=variance)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    return {
        "X_scaled": X_scaled,
        "X_pca": X_pca,
        "explained_variance_ratio": explained,
        "cumulative_variance": np.cumsum(explained),
        "n_components": X_pca.shape[1],
    }


# ---------------------------------------------------------------------------
# Clustering (KMeans / Hierarchical / GMM / DBSCAN)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def run_kmeans(X_pca: np.ndarray, k: int):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels) if k > 1 else np.nan
    return labels, score, model


@st.cache_data(show_spinner=False)
def kmeans_selection_curves(X_pca: np.ndarray, k_range=range(2, 11)):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    wcss, sil = [], []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_pca)
        wcss.append(model.inertia_)
        sil.append(silhouette_score(X_pca, labels))
    return list(k_range), wcss, sil


@st.cache_data(show_spinner=False)
def run_hierarchical(X_pca: np.ndarray, k: int):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = model.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    return labels, score, model


@st.cache_data(show_spinner=False)
def run_gmm(X_pca: np.ndarray, k: int):
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score

    model = GaussianMixture(n_components=k, random_state=42)
    labels = model.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    return labels, score, model


@st.cache_data(show_spinner=False)
def gmm_selection_curve(X_pca: np.ndarray, k_range=range(2, 11)):
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score

    sil = []
    for k in k_range:
        model = GaussianMixture(n_components=k, random_state=42)
        labels = model.fit_predict(X_pca)
        sil.append(silhouette_score(X_pca, labels))
    return list(k_range), sil


@st.cache_data(show_spinner=False)
def run_dbscan(X_pca: np.ndarray, eps: float, min_samples: int = 5):
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_pca)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = int(np.sum(labels == -1))
    valid = labels != -1
    if len(set(labels[valid])) > 1:
        score = silhouette_score(X_pca[valid], labels[valid])
    else:
        score = np.nan
    return labels, score, n_clusters, noise


@st.cache_data(show_spinner=False)
def k_distance(X_pca: np.ndarray, k: int = 5):
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=k).fit(X_pca)
    distances, _ = nbrs.kneighbors(X_pca)
    distances = np.sort(distances[:, k - 1])
    return distances


@st.cache_data(show_spinner=False)
def clustering_comparison(X_pca, kmeans_labels, hier_labels, gmm_labels, dbscan_labels):
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    rows = []
    for name, labels in [
        ("K-Means", kmeans_labels),
        ("Hierarchical", hier_labels),
        ("Gaussian Mixture", gmm_labels),
        ("DBSCAN", dbscan_labels),
    ]:
        labels = np.asarray(labels)
        valid = labels != -1
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise = int(np.sum(labels == -1))
        if len(set(labels[valid])) > 1:
            sil = silhouette_score(X_pca[valid], labels[valid])
            db = davies_bouldin_score(X_pca[valid], labels[valid])
            ch = calinski_harabasz_score(X_pca[valid], labels[valid])
        else:
            sil = db = ch = np.nan
        rows.append({
            "Algorithm": name, "Clusters": n_clusters,
            "Silhouette Score": round(sil, 3) if pd.notna(sil) else np.nan,
            "Davies-Bouldin Index": round(db, 3) if pd.notna(db) else np.nan,
            "Calinski-Harabasz Score": round(ch, 2) if pd.notna(ch) else np.nan,
            "Noise Points": noise,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# OLS - NIFTY50 Daily Return drivers (Clustering notebook Step: OLS section)
# ---------------------------------------------------------------------------

OLS_RETURN_TARGET = "NIFTY50_Daily_Return"
OLS_RETURN_FEATURES = [
    "USD_INR_Daily_Return",
    "GOLD_Daily_Return",
    "BRENT_CRUDE_Daily_Return",
    "RBI_Repo_Rate",
    "Inflation_Proxy_YoY",
    "Monetary_Policy_Shock",
]


@st.cache_data(show_spinner=False)
def run_ols_daily_return(merged_df: pd.DataFrame):
    import statsmodels.api as sm

    ols_df = merged_df[[OLS_RETURN_TARGET] + OLS_RETURN_FEATURES].copy().dropna()
    X = sm.add_constant(ols_df[OLS_RETURN_FEATURES])
    Y = ols_df[OLS_RETURN_TARGET]
    model = sm.OLS(Y, X).fit()
    predicted = model.predict(X)
    residuals = Y - predicted
    return {"model": model, "X": X, "Y": Y, "predicted": predicted, "residuals": residuals, "df": ols_df}


@st.cache_data(show_spinner=False)
def ols_diagnostics(_model, residuals):
    from statsmodels.stats.stattools import jarque_bera, durbin_watson
    from statsmodels.stats.diagnostic import het_breuschpagan

    jb_stat, jb_p, skew, kurt = jarque_bera(residuals)
    bp = het_breuschpagan(residuals, _model.model.exog)
    dw = durbin_watson(residuals)
    return {
        "jarque_bera": (jb_stat, jb_p, skew, kurt),
        "breusch_pagan": bp,
        "durbin_watson": dw,
    }


# ---------------------------------------------------------------------------
# Regularized Regression - NIFTY50 Volume (standalone OLS notebook)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def build_enhanced_regression_dataset(dated_reduced: pd.DataFrame):
    """Reproduces Step 4 (Feature Engineering) of the standalone OLS notebook,
    applied to the VIF-reduced dataset. Falls back gracefully if a source
    column was itself removed by VIF."""
    df = dated_reduced.copy()

    def has(col):
        return col in df.columns

    if has("NIFTY50_Volume"):
        df["NIFTY50_Volume_Lag1"] = df["NIFTY50_Volume"].shift(1)
        df["NIFTY50_Volume_Lag5"] = df["NIFTY50_Volume"].shift(5)
        df["NIFTY50_Volume_MA7"] = df["NIFTY50_Volume"].rolling(7).mean()
        df["NIFTY50_Volume_MA15"] = df["NIFTY50_Volume"].rolling(15).mean()
        df["NIFTY50_Volume_STD7"] = df["NIFTY50_Volume"].rolling(7).std()
    if has("RELIANCE_Volume") and has("TCS_Volume"):
        df["Reliance_TCS"] = df["RELIANCE_Volume"] * df["TCS_Volume"]
    if has("GOLD_Volume") and has("BRENT_CRUDE_Volume"):
        df["Gold_Brent"] = df["GOLD_Volume"] * df["BRENT_CRUDE_Volume"]
    if has("RELIANCE_Volume"):
        df["Log_RELIANCE_Volume"] = np.log1p(df["RELIANCE_Volume"])
    if has("TCS_Volume"):
        df["Log_TCS_Volume"] = np.log1p(df["TCS_Volume"])
    if has("HDFCBANK_Volume"):
        df["Log_HDFCBANK_Volume"] = np.log1p(df["HDFCBANK_Volume"])

    df = df.dropna().reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def run_regularized_regression(enhanced_df: pd.DataFrame, target: str = "NIFTY50_Volume"):
    import statsmodels.api as sm
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import SGDRegressor, RidgeCV, LassoCV
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    df = enhanced_df.drop(columns=["Date"], errors="ignore")
    Y = df[target]
    X = df.drop(columns=[target])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    # --- OLS ---
    X_train_c = sm.add_constant(X_train)
    X_test_c = sm.add_constant(X_test)
    ols_model = sm.OLS(Y_train, X_train_c).fit()
    Y_pred_ols = ols_model.predict(X_test_c)

    def metrics(y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return r2, rmse, mae, mape

    ols_r2, ols_rmse, ols_mae, ols_mape = metrics(Y_test, Y_pred_ols)

    # --- Scale for regularized models ---
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    sgd = SGDRegressor(loss="squared_error", learning_rate="adaptive", eta0=0.001, max_iter=5000, random_state=42)
    sgd.fit(X_train_s, Y_train)
    Y_pred_sgd = sgd.predict(X_test_s)
    sgd_r2, sgd_rmse, sgd_mae, sgd_mape = metrics(Y_test, Y_pred_sgd)

    ridge = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=5)
    ridge.fit(X_train_s, Y_train)
    Y_pred_ridge = ridge.predict(X_test_s)
    ridge_r2, ridge_rmse, ridge_mae, ridge_mape = metrics(Y_test, Y_pred_ridge)

    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(X_train_s, Y_train)
    Y_pred_lasso = lasso.predict(X_test_s)
    lasso_r2, lasso_rmse, lasso_mae, lasso_mape = metrics(Y_test, Y_pred_lasso)

    comparison = pd.DataFrame({
        "Model": ["OLS", "Gradient Descent", "Ridge", "Lasso"],
        "R2": [ols_r2, sgd_r2, ridge_r2, lasso_r2],
        "RMSE": [ols_rmse, sgd_rmse, ridge_rmse, lasso_rmse],
        "MAE": [ols_mae, sgd_mae, ridge_mae, lasso_mae],
        "MAPE": [ols_mape, sgd_mape, ridge_mape, lasso_mape],
    }).round(4)

    return {
        "comparison": comparison,
        "Y_test": Y_test,
        "predictions": {"OLS": Y_pred_ols, "Gradient Descent": Y_pred_sgd, "Ridge": Y_pred_ridge, "Lasso": Y_pred_lasso},
        "ridge_alpha": ridge.alpha_,
        "lasso_alpha": lasso.alpha_,
        "ols_model": ols_model,
    }


# ---------------------------------------------------------------------------
# Time series helpers shared by ARIMA(Close)/TSF-Volume/ARCH-GARCH/VAR pages
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def evaluate_forecast(actual, forecast):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    mae = mean_absolute_error(actual, forecast)
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    r2 = r2_score(actual, forecast)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2}
