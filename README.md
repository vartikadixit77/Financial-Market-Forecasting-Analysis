# 📈 NIFTY Hybrid Econometric–ML Dashboard

**Financial Market Forecasting and Analysis using Machine Learning, Time Series, and Econometric Models**

An interactive Streamlit dashboard reproducing a full dissertation pipeline — EDA, VIF,
PCA, clustering, OLS regression, ARIMA/SARIMA, ARCH/GARCH, VAR & Granger causality, and
an ANN forecasting model — on live NSE/BSE market microstructure and RBI macroeconomic
data.

---

## 1. Project structure

```
nifty_dashboard/
├── app.py                     # Entry point — run this with `streamlit run app.py`
├── requirements.txt
├── .streamlit/
│   └── config.toml            # Theme (dark, finance-styled)
├── data/                      # Bundled sample dataset (CSV + Excel)
├── utils/
│   ├── data_pipeline.py       # All data loading & modelling logic
│   ├── theme.py               # Visual theme + dark/light toggle
│   └── ui.py                  # Reusable UI components (KPI cards, boxes, headers)
└── views/                     # One file per dashboard page
    ├── home.py
    ├── dataset_explorer.py
    ├── eda.py
    ├── vif.py
    ├── pca.py
    ├── clustering.py
    ├── ols_daily_return.py
    ├── regularized_regression.py
    ├── arima_close.py
    ├── time_series_workflow.py
    ├── volatility_arch_garch.py
    ├── var_granger.py
    ├── ann_forecasting.py
    ├── model_comparison.py
    ├── forecasting.py
    └── conclusion.py
```

---

## 2. Run it locally

**Requirements:** Python 3.10–3.12 recommended. (Python 3.13 works too, but TensorFlow
support for 3.13 is newer — if TensorFlow fails to install, see step 4 below; the ANN
page automatically falls back to a scikit-learn model.)

```bash
cd nifty_dashboard
pip install -r requirements.txt
streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`. The bundled
sample dataset in `data/` is used by default — no upload needed to get started. You can
also upload your own market/macro files from the Home page sidebar.

---

## 3. Using the dashboard

Navigate using the **grouped sidebar menu**:

| Group | Pages |
|---|---|
| **Overview** | Home, Dataset Explorer, EDA |
| **Feature Engineering** | VIF & Multicollinearity, PCA |
| **Modelling** | Clustering, OLS (Daily Return), Regularized Regression (Volume) |
| **Time Series & Volatility** | ARIMA (Close), Time Series Workflow (Volume), ARCH/GARCH, VAR & Granger |
| **Machine Learning** | ANN Forecasting Model |
| **Results** | Model Comparison, Interactive Forecasting, Conclusion |

Toggle **🌙 Dark / ☀️ Light** mode at the top of the sidebar at any time.

**Tip:** visit pages roughly in sidebar order — later stages (Clustering, Model
Comparison) depend on outputs from earlier ones (VIF, PCA, ANN training).

---

## 4. If TensorFlow won't install (e.g. on Python 3.13 or low disk space)

The ANN page works either way:

- **With TensorFlow** → trains an actual Keras feed-forward network (as in the notebook).
- **Without TensorFlow** → automatically falls back to `scikit-learn`'s `MLPRegressor`
  with an equivalent shallow architecture — same lag features, same charts, same metrics.

To skip TensorFlow entirely (smaller install, faster setup), just delete the
`tensorflow>=2.15` line from `requirements.txt` before running `pip install`.

---

## 5. Deploying to Streamlit Community Cloud (free, public HTTPS URL)

1. **Push this project to a GitHub repository** (public or private):
   ```bash
   cd nifty_dashboard
   git init
   git add .
   git commit -m "NIFTY hybrid econometric-ML dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```

2. **Go to** [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.

3. Click **"New app"**, then select:
   - Repository: `<your-username>/<your-repo>`
   - Branch: `main`
   - Main file path: `app.py`

4. (Optional) Under **Advanced settings**, you can set the Python version to match
   what you tested locally (3.10–3.12 recommended).

5. Click **Deploy**. Streamlit Cloud will install everything from `requirements.txt`
   and build the app — this takes a few minutes the first time (mostly downloading
   TensorFlow).

6. Once built, you'll get a public URL like:
   ```
   https://<your-app-name>.streamlit.app
   ```
   Anyone can open this in a browser — no Python or installation needed on their end.

**Notes for Streamlit Cloud:**
- Free-tier apps sleep after inactivity and wake on the next visit (takes ~30-60 seconds).
- If the free tier's memory limit is tight, consider removing `tensorflow` from
  `requirements.txt` before deploying — the ANN page's scikit-learn fallback is much
  lighter and works well within Streamlit Cloud's default resource limits.
- The bundled sample dataset in `data/` deploys with the repo, so the app works
  immediately for any visitor without needing to upload files.

---

## 6. Known environment quirks

- Very new `pandas`/`pyarrow` combinations have a known rare crash when Streamlit
  hashes certain object types for caching. `requirements.txt` pins tested version
  ranges to avoid this; if you ever see the app process crash unexpectedly, refresh
  the browser tab (Streamlit restarts the script) and it will resolve.
- On Windows, if `pip install` runs low on disk space partway through (TensorFlow is
  ~350MB), run `pip cache purge` to free space, then re-run `pip install -r
  requirements.txt` — pip resumes from what's already installed.

---

## 7. Academic notes

Every analytical page includes an **Objective**, an expandable **Methodology**, and an
**Interpretation / Conclusion** box written in plain English, so the dashboard can be
understood by a non-technical reviewer without reading the full dissertation. The
**Model Comparison** page automatically highlights the best-performing forecasting
model based on RMSE and explains why, and the **Conclusion** page summarizes major
findings, research contributions, limitations, and future scope.
