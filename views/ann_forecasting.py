import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils import data_pipeline as dp
from utils import theme, ui

theme.inject_css()
ui.page_header(
    "ANN Forecasting Model — NIFTY50 Trading Volume",
    "Can a neural network beat classical econometrics at forecasting volume, on a fair, identical test set?",
    "🧠",
)

if not st.session_state.get("data_ready"):
    st.warning("Go to the Home page and load the data first.")
    st.stop()

ui.objective_box(
    "To build a lag-based feed-forward neural network using only the same information "
    "(past volume values) available to ARIMA/SARIMA/VAR, then benchmark it fairly — same "
    "train/test split, same metrics — against those classical models."
)
ui.methodology_box(
    "Each training example uses the previous N_LAGS weekly volume values to predict the next "
    "one — the same autoregressive information set ARIMA and SARIMA use internally. A held-out "
    "chronological validation slice drives early stopping so the network doesn't overfit the "
    "~300-row training window. GARCH is excluded from the final ranking because it forecasts "
    "return volatility, not volume — a different target variable entirely."
)

if not st.session_state.get("data_ready"):
    st.warning("Go to the Home page and load the data first.")
    st.stop()

df_market = st.session_state["df_market"]
df_macro = st.session_state["df_macro"]
merged = dp.build_inner_merge(df_market, df_macro)
ts = merged.set_index("Date")["NIFTY50_Volume"].astype(float)

train_size = int(len(ts) * 0.80)
train, test = ts.iloc[:train_size], ts.iloc[train_size:]

st.write(f"Training observations: **{len(train)}**  |  Testing observations: **{len(test)}**")

TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers, callbacks, optimizers, Sequential
except ImportError:
    TF_AVAILABLE = False

n_lags = st.slider("Number of lags (N_LAGS)", 3, 12, 6)
n_layers = st.slider("Hidden layers", 1, 3, 1)
n_units = st.slider("Units in first hidden layer (funnel-shaped)", 8, 64, 16, step=8)
dropout = st.slider("Dropout", 0.0, 0.3, 0.0, 0.05)
epochs = st.slider("Max epochs", 20, 300, 150, step=10)

run = st.button("Train / retrain ANN", type="primary")


def make_lag_features(values, index, n_lags, test_start_date):
    n = len(values)
    X_all, y_all, idx_all = [], [], []
    for t in range(n_lags, n):
        X_all.append(values[t - n_lags:t][::-1])
        y_all.append(values[t])
        idx_all.append(index[t])
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    idx_all = pd.DatetimeIndex(idx_all)
    is_test = idx_all >= test_start_date
    return X_all[~is_test], y_all[~is_test], X_all[is_test], y_all[is_test], idx_all[is_test]


X_train_full, y_train_full, X_test, y_test, test_idx = make_lag_features(
    ts.values.astype(float), ts.index, n_lags, test.index[0]
)

if len(X_test) != len(test):
    st.warning(f"Lag-window alignment gives {len(X_test)} test rows vs {len(test)} in the TS split "
               "(small mismatches can occur if N_LAGS reaches past the split boundary).")

if "ann_results" not in st.session_state:
    st.session_state["ann_results"] = None

if run:
    from sklearn.preprocessing import MinMaxScaler

    val_frac = 0.15
    n_val = max(1, int(len(X_train_full) * val_frac))
    X_train, y_train = X_train_full[:-n_val], y_train_full[:-n_val]
    X_val, y_val = X_train_full[-n_val:], y_train_full[-n_val:]

    x_scaler, y_scaler = MinMaxScaler(), MinMaxScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    X_test_s = x_scaler.transform(X_test)
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).flatten()

    if TF_AVAILABLE:
        import random, os
        os.environ["PYTHONHASHSEED"] = "42"
        random.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)

        with st.spinner("Training ANN (TensorFlow/Keras)..."):
            model = Sequential()
            model.add(layers.Input(shape=(n_lags,)))
            for i in range(n_layers):
                units = max(4, n_units // (2 ** i))
                model.add(layers.Dense(units, activation="relu", kernel_regularizer=regularizers.l2(1e-4)))
                if dropout > 0:
                    model.add(layers.Dropout(dropout))
            model.add(layers.Dense(1, activation="linear"))
            model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])

            es = callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
            history = model.fit(X_train_s, y_train_s, validation_data=(X_val_s, y_val_s),
                                 epochs=epochs, batch_size=16, callbacks=[es], verbose=0)

            y_pred_s = model.predict(X_test_s, verbose=0).flatten()
            y_pred = y_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
            history_df = pd.DataFrame({"loss": history.history["loss"], "val_loss": history.history["val_loss"]})
    else:
        st.info("TensorFlow isn't installed in this environment — using scikit-learn's MLPRegressor as a fallback (same lag features, similar shallow architecture).")
        from sklearn.neural_network import MLPRegressor

        hidden_layer_sizes = tuple(max(4, n_units // (2 ** i)) for i in range(n_layers))
        with st.spinner("Training MLPRegressor..."):
            model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation="relu",
                                  learning_rate_init=1e-3, max_iter=epochs, random_state=42,
                                  early_stopping=True, validation_fraction=val_frac)
            model.fit(X_train_s, y_train_s)
            y_pred_s = model.predict(X_test_s)
            y_pred = y_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
            history_df = pd.DataFrame({"loss": model.loss_curve_}) if hasattr(model, "loss_curve_") else None

    y_true = y_test[:len(y_pred)]
    ann_metrics = dp.evaluate_forecast(y_true, y_pred)

    st.session_state["ann_results"] = {
        "metrics": ann_metrics, "y_true": y_true, "y_pred": y_pred,
        "test_idx": test_idx[:len(y_pred)], "history_df": history_df,
    }

if st.session_state["ann_results"] is not None:
    res = st.session_state["ann_results"]
    m = res["metrics"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAE", f"{m['MAE']:,.0f}")
    c2.metric("RMSE", f"{m['RMSE']:,.0f}")
    c3.metric("MAPE", f"{m['MAPE']:.2f}%")
    c4.metric("R²", f"{m['R2']:.3f}")
    c5.metric("MSE", f"{m['MSE']:,.0f}")

    if res["history_df"] is not None:
        fig = px.line(res["history_df"], title="Training vs Validation Loss")
        st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=res["test_idx"], y=res["y_true"], name="Actual"))
    fig2.add_trace(go.Scatter(x=res["test_idx"], y=res["y_pred"], name="ANN Predicted"))
    fig2.update_layout(title="ANN: Actual vs Predicted (test set)")
    st.plotly_chart(fig2, use_container_width=True)

    residuals = res["y_true"] - res["y_pred"]
    fig3 = px.histogram(residuals, nbins=25, title="Distribution of ANN residuals")
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("Click **Train / retrain ANN** to fit the model with the settings above.")

st.divider()
st.header("Final Model Comparison")

if st.session_state["ann_results"] is None:
    st.info("Train the ANN above to populate the final comparison table.")
else:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.api import VAR

    @st.cache_data(show_spinner=False)
    def fit_benchmarks(_train_key):
        rows = {}
        arima_fit = ARIMA(train, order=(1, 1, 1)).fit()
        arima_fc = arima_fit.forecast(steps=len(test))
        rows["ARIMA(1,1,1)"] = dp.evaluate_forecast(test, arima_fc)

        sarima_fit = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 52),
                              enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        sarima_fc = sarima_fit.forecast(steps=len(test))
        rows["SARIMA(1,1,1)(1,0,1,52)"] = dp.evaluate_forecast(test, sarima_fc)

        var_cols = [c for c in ["NIFTY50_Volume", "USD_INR_Daily_Return", "GOLD_Daily_Return",
                                 "BRENT_CRUDE_Daily_Return", "Monetary_Policy_Shock", "Inflation_Proxy_YoY"]
                    if c in merged.columns]
        var_data = merged.set_index("Date")[var_cols].dropna()
        var_diff = var_data.diff().dropna()
        train_var = var_diff.loc[:train.index[-1]]
        test_var = var_diff.loc[train.index[-1]:].iloc[1:]

        try:
            lag_order = VAR(train_var).select_order(maxlags=10)
            best_lag = max(1, lag_order.selected_orders["bic"])
            var_model = VAR(train_var).fit(best_lag)
            var_fc = var_model.forecast(train_var.values[-best_lag:], steps=len(test_var))
            var_fc_df = pd.DataFrame(var_fc, columns=var_diff.columns, index=test_var.index)
            last_level = train.iloc[-1]
            var_level_fc = last_level + var_fc_df["NIFTY50_Volume"].cumsum()
            rows["VAR"] = dp.evaluate_forecast(test.iloc[:len(var_level_fc)], var_level_fc)
        except Exception:
            rows["VAR"] = {"MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "R2": np.nan}

        return rows

    with st.spinner("Refitting ARIMA / SARIMA / VAR benchmarks for the comparison table..."):
        benchmark_rows = fit_benchmarks(hash(tuple(train.values.round(2))))

    comparison = pd.DataFrame(benchmark_rows).T[["MAE", "MSE", "RMSE", "MAPE", "R2"]]
    ann_m = st.session_state["ann_results"]["metrics"]
    comparison.loc["ANN"] = [ann_m["MAE"], ann_m["MSE"], ann_m["RMSE"], ann_m["MAPE"], ann_m["R2"]]
    comparison.loc["GARCH(1,1)"] = [np.nan, np.nan, np.nan, np.nan, np.nan]
    comparison = comparison.sort_values("RMSE").round(3)

    st.session_state["final_comparison_table"] = comparison
    st.dataframe(comparison, use_container_width=True)
    st.caption("GARCH(1,1) models return volatility, not volume, so it isn't scored on this metric set — see the Volatility page.")

    best_model = comparison["RMSE"].idxmin()
    st.success(f"Best model on this test period by RMSE: **{best_model}**")
    ui.conclusion_box(
        "If the ANN wins, it's evidence that trading volume has non-linear, regime-dependent "
        "structure that fixed linear models (ARIMA/SARIMA/VAR) cannot fully capture. This "
        "should be read as a single-test-window result, not a guarantee across all periods — "
        "a more rigorous validation would repeat this comparison over several rolling windows."
    )
    ui.business_box(
        "A model that reliably forecasts trading volume supports better liquidity planning, "
        "execution scheduling for large trades, and market-impact estimation — all of which "
        "depend on anticipating when volume will spike or dry up."
    )

    fig4 = px.bar(comparison.reset_index(), x="index", y="RMSE", title="RMSE by model (lower is better)",
                  labels={"index": "Model"})
    st.plotly_chart(fig4, use_container_width=True)

    st.download_button("Download Final_Model_Comparison.csv", comparison.to_csv(), file_name="Final_Model_Comparison.csv")
