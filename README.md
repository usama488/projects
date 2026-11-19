# 📊 Auto EDA + ML Studio

A Streamlit app that takes **any tabular dataset** and runs:

1. **15-step EDA:** ingest → shape → schema → type coercion → missing → duplicates → descriptive stats → cardinality → distribution & normality → outliers → univariate → bivariate → correlation & VIF → target analysis → quality score.
2. **Four dimensions of analysis:** Composition, Distribution, Comparison, Relationship. Charts are picked automatically from the column types.
3. **Machine learning:** the task is detected for you (binary, multiclass or regression). It includes leakage checks, a preprocessing pipeline, AutoML leaderboard, k-fold CV, Grid/Random search, learning and validation curves with a bias/variance diagnosis, ROC/PR/confusion/threshold/calibration plots, residual plots, feature and permutation importance, PDP, optional SHAP, a single-row prediction form, batch prediction, and `.joblib` + model-card export.
4. **Unsupervised:** K-Means (elbow + silhouette), DBSCAN, Agglomerative clustering (with dendrogram), PCA explained variance, PCA/t-SNE 2D maps and cluster profiles.
5. **Deep learning:** an MLP builder (layers, neurons, activation, dropout, BatchNorm, optimizer, learning rate, batch size, **epochs**, LR scheduler, early stopping, weight decay, gradient clipping). Training updates **live loss/metric curves every epoch** and shows steps per epoch, iteration count, parameter count and the architecture diagram. You also get weight histograms, gradient norms, a rule-based training diagnosis (overfitting, underfitting, LR too high or too low), an experiment tracker, and a comparison against the best classical model.
6. **Learn mode:** a sidebar toggle that adds a plain-English explanation and example next to every concept, plus a searchable glossary of 70+ terms. There's also an interactive gradient-descent demo (learning rate × epochs) and an epoch/batch/iteration calculator.
7. **Export:** standalone HTML and Markdown reports, plus cleaned-data CSV downloads.

All interpretations are **rule-based**, so no API key or LLM is needed.

Demo datasets: seaborn (`titanic`, `tips`, `iris`, `penguins`, `diamonds`, `mpg`, `taxis`) and scikit-learn (`breast_cancer`, `wine`, `california_housing`). They need internet the first time and are cached in `demo_cache/` after that.

---

## Setup (Windows PowerShell)

```powershell
cd auto_eda_ml_studio
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # deep learning (optional but recommended)
pip install -r requirements-optional.txt                            # XGBoost, LightGBM, SHAP (optional)
streamlit run app.py
```

Or just double-click **`run_app.bat`**: it creates the venv, installs everything and starts the app.

macOS / Linux:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install torch
streamlit run app.py
```

The app opens at **http://localhost:8501**.

## Stopping the app

- Press **Ctrl + C** in the terminal that's running it.
- If it's running in the background (Windows PowerShell):
  ```powershell
  Get-NetTCPConnection -LocalPort 8501 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
  ```
- macOS/Linux: `lsof -ti:8501 | xargs kill`

## Using it

1. **🏠 Data:** load a demo or upload a CSV/Excel/JSON/Parquet file, then pick the target column. You can also apply cleaning (drop duplicates, impute, drop very empty columns, clip outliers) and download the cleaned data.
2. **🔍 15-Step EDA:** go one step at a time, or toggle *Show all 15 steps*. In step 12, turn on selection to box/lasso points and cross-filter.
3. **📊 4 Dimensions:** view the auto-generated chart gallery.
4. **🤖 Machine Learning:** set up the problem, then run AutoML, train, tune, view curves, explain and predict. The *Clustering & PCA* tab doesn't need a target.
5. **🧠 Deep Learning:** reuses the ML page's train/test split. Design the network, press *Train*, and watch every epoch.
6. **📚 Learn & Glossary:** look up concepts and play with the interactive demos.
7. **📤 Export:** download HTML or Markdown reports.

The **sidebar filters** apply to every page. For dark/light theme, use the ⋮ menu → Settings → Theme. The default theme is set in `.streamlit/config.toml`.

## Project structure

```
app.py                 entry point + sidebar navigation & filters
data.py           loading, demo datasets, column typing
eda.py            15 EDA steps, 4 dimensions, cleaning, quality score, rule-based findings
ml.py             pipelines, model catalog, CV, tuning, metrics, curves, explainability, clustering
dl.py             PyTorch MLP trained epoch-by-epoch (sklearn MLP fallback), diagnosis, plots
glossary.py       Learn-mode explanations
report.py         HTML / Markdown export
page_*.py           one file per page

```

## Tests

```bash


```

## Notes

- If PyTorch isn't installed, the Deep Learning page falls back to scikit-learn's MLP. It's still trained one epoch at a time with live curves, but has fewer options.
- Plots are drawn from a sample of up to 5,000 rows, and clustering uses up to 20,000. All statistics use the full data.
- Random seed 42 is used everywhere, so results are reproducible.
- Neural networks rarely beat gradient boosting on small tabular data. The app tells you which one won, and deep learning really pays off on large datasets, images, text and audio.
