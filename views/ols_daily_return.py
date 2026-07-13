import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import scipy.stats as stats

from utils import data_pipeline as dp
from utils import theme, ui

theme.inject_css()
ui.page_header(
    "OLS Regression — Drivers of NIFTY50 Daily Return",
    "Which macro and cross-asset variables move NIFTY50 daily returns, and how strongly?",
    "📉",
)

if not st.session_state.get("data_ready"):
    st.warning("Go to the Home page and load the data first.")
    st.stop()

ui.objective_box(
    "To quantify, using classical linear regression, how much of the day-to-day movement in "
    "NIFTY50 returns can be explained by currency, commodity, and monetary-policy variables — "
    "and which of those relationships are statistically reliable rather than noise."
)

df_market = st.session_state["df_market"]
df_macro = st.session_state["df_macro"]
merged = dp.build_inner_merge(df_market, df_macro)

st.markdown(f"""
**Dependent variable:** `{dp.OLS_RETURN_TARGET}`
**Independent variables:** {', '.join(f'`{f}`' for f in dp.OLS_RETURN_FEATURES)}
""")

result = dp.run_ols_daily_return(merged)
model, X, Y, predicted, residuals = result["model"], result["X"], result["Y"], result["predicted"], result["residuals"]

tab1, tab2, tab3 = st.tabs(["Model Summary", "Predictions", "Diagnostic Tests"])

with tab1:
    st.text(model.summary().as_text())

    coefficients = pd.DataFrame({"Coefficient": model.params, "P-Value": model.pvalues})
    significant = coefficients[coefficients["P-Value"] < 0.05]

    st.subheader("Statistically significant variables (p < 0.05)")
    st.dataframe(significant, use_container_width=True)

    for var in significant.index:
        if var == "const":
            continue
        direction = "Positive" if significant.loc[var, "Coefficient"] > 0 else "Negative"
        st.write(f"- **{var}** → {direction} significant impact")

with tab2:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    r2 = r2_score(Y, predicted)
    adj_r2 = model.rsquared_adj
    rmse = np.sqrt(mean_squared_error(Y, predicted))
    mae = mean_absolute_error(Y, predicted)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", round(r2, 4))
    c2.metric("Adj. R²", round(adj_r2, 4))
    c3.metric("RMSE", round(rmse, 6))
    c4.metric("MAE", round(mae, 6))

    plot_df = pd.DataFrame({"Actual": Y.values, "Predicted": predicted.values})
    fig = px.line(plot_df, title="Actual vs Predicted Daily Returns")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(x=predicted, y=residuals, labels={"x": "Predicted", "y": "Residual"}, title="Residual plot")
    fig2.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    diag = dp.ols_diagnostics(model, residuals)
    jb_stat, jb_p, skew, kurt = diag["jarque_bera"]
    bp = diag["breusch_pagan"]
    dw = diag["durbin_watson"]

    st.subheader("Jarque–Bera Normality Test")
    c1, c2 = st.columns(2)
    c1.metric("JB Statistic", round(jb_stat, 4))
    c2.metric("P-value", round(jb_p, 4))
    st.write("✅ Residuals approximately normal." if jb_p > 0.05 else "❌ Residuals NOT normally distributed.")

    st.subheader("Breusch–Pagan Test (Homoscedasticity)")
    labels = ["LM Statistic", "LM p-value", "F Statistic", "F p-value"]
    for lab, val in zip(labels, bp):
        st.write(f"**{lab}:** {val:.4f}")
    st.write("✅ Homoscedasticity satisfied." if bp[1] > 0.05 else "❌ Evidence of heteroscedasticity.")

    st.subheader("Durbin–Watson Test (Autocorrelation)")
    st.metric("Durbin-Watson statistic", round(dw, 4))
    if 1.5 <= dw <= 2.5:
        st.write("✅ Little or no autocorrelation.")
    elif dw < 1.5:
        st.write("⚠️ Positive autocorrelation detected.")
    else:
        st.write("⚠️ Negative autocorrelation detected.")

    c1, c2 = st.columns(2)
    with c1:
        fig3 = px.histogram(residuals, nbins=30, title="Residual distribution")
        st.plotly_chart(fig3, use_container_width=True)
    with c2:
        qq = stats.probplot(residuals, dist="norm")
        qq_df = pd.DataFrame({"Theoretical": qq[0][0], "Sample": qq[0][1]})
        fig4 = px.scatter(qq_df, x="Theoretical", y="Sample", title="Q-Q Plot")
        st.plotly_chart(fig4, use_container_width=True)

st.divider()
ui.conclusion_box(
    "Daily NIFTY50 returns are only partially explained by these macro/cross-asset variables — "
    "expected, since daily equity returns are dominated by firm-specific and sentiment-driven "
    "noise that no macro regression can capture. The coefficients that ARE significant still "
    "give a defensible, econometrically grounded view of directional sensitivity (e.g. to "
    "currency or commodity shocks) for risk reporting."
)
