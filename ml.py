"""Classical machine learning: task detection, pipelines, models, tuning, evaluation, explainability."""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn import metrics as M
from sklearn.base import is_regressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold,
                                     cross_validate, learning_curve, train_test_split, validation_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder, OrdinalEncoder,
                                   RobustScaler, StandardScaler)

from data import column_kind

TEMPLATE = "plotly_white"
SEED = 42

try:
    import xgboost  # noqa: F401
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    import lightgbm  # noqa: F401
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.under_sampling import RandomUnderSampler
    HAS_IMB = True
except Exception:
    HAS_IMB = False
try:
    import shap  # noqa: F401
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


# ------------------------------------------------------------------ task detection
def detect_task(y: pd.Series) -> str:
    y = y.dropna()
    if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y) and y.nunique() > 15:
        return "regression"
    return "binary" if y.nunique() == 2 else "multiclass"


def leakage_warnings(df: pd.DataFrame, target: str, features: list[str]) -> list[str]:
    warns = []
    y = df[target]
    for c in features:
        k = column_kind(df[c])
        if k == "id":
            warns.append(f"`{c}` looks like an ID column - it can't generalise, remove it.")
        if pd.api.types.is_numeric_dtype(df[c]) and pd.api.types.is_numeric_dtype(y):
            pair = df[[c, target]].dropna()
            if len(pair) > 10 and pair[c].nunique() > 1 and abs(pair[c].corr(pair[target])) > .95:
                warns.append(f"`{c}` correlates > 0.95 with the target - possible leakage (is it computed from the target?).")
        if detect_task(y) != "regression" and df[c].nunique() <= max(y.nunique() * 2, 2):
            ct = pd.crosstab(df[c], y)
            if len(ct) > 1 and (ct.max(axis=1) / ct.sum(axis=1)).min() == 1.0 and df[c].nunique() <= y.nunique() * 2:
                warns.append(f"`{c}` predicts the target perfectly - likely a duplicate of the target (e.g. 'alive' vs 'survived').")
    return warns


# ------------------------------------------------------------------ preprocessing
SCALERS = {"Standard": StandardScaler, "MinMax": MinMaxScaler, "Robust": RobustScaler, "None": None}


@dataclass
class Prepared:
    task: str
    target: str
    features: list[str]
    num_cols: list[str]
    cat_cols: list[str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    classes: list = field(default_factory=list)
    label_encoder: LabelEncoder | None = None
    raw_features: list = field(default_factory=list)

    def featurize(self, new: pd.DataFrame) -> pd.DataFrame:
        """Apply the same feature engineering as prepare() to new rows (for batch prediction)."""
        d = new.copy()
        for c in self.raw_features:
            if c + "_year" in self.num_cols and c in d:
                dt = pd.to_datetime(d[c], errors="coerce")
                d[c + "_year"], d[c + "_month"], d[c + "_dow"] = dt.dt.year, dt.dt.month, dt.dt.dayofweek
        missing = [c for c in self.features if c not in d.columns]
        if missing:
            raise KeyError(f"Missing columns: {missing}")
        X = d[self.features].copy()
        X[self.num_cols] = X[self.num_cols].apply(pd.to_numeric, errors="coerce").astype(float)
        for c in self.cat_cols:
            X[c] = X[c].astype(str).where(X[c].notna(), np.nan)
        return X


def prepare(df: pd.DataFrame, target: str, features: list[str], test_size: float = .2,
            seed: int = SEED) -> Prepared:
    d = df[features + [target]].dropna(subset=[target]).copy()
    task = detect_task(d[target])
    num_cols, cat_cols = [], []
    for c in features:
        if pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c + "_year"], d[c + "_month"], d[c + "_dow"] = d[c].dt.year, d[c].dt.month, d[c].dt.dayofweek
            num_cols += [c + "_year", c + "_month", c + "_dow"]
            d = d.drop(columns=c)
        elif pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c]):
            num_cols.append(c)
        else:
            d[c] = d[c].astype(str).where(d[c].notna(), np.nan)
            cat_cols.append(c)
    X = d[num_cols + cat_cols].copy()
    X[num_cols] = X[num_cols].astype(float)
    le = None
    if task == "regression":
        y = d[target].astype(float).to_numpy()
        classes = []
    else:
        le = LabelEncoder()
        y = le.fit_transform(d[target].astype(str))
        classes = list(le.classes_)
    strat = y if task != "regression" and pd.Series(y).value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)
    return Prepared(task, target, num_cols + cat_cols, num_cols, cat_cols, X_tr, X_te, y_tr, y_te, classes, le, list(features))


def make_preprocessor(p: Prepared, scaler: str = "Standard", encoder: str = "One-Hot",
                      num_impute: str = "median") -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy=num_impute))]
    if SCALERS.get(scaler):
        num_steps.append(("scale", SCALERS[scaler]()))
    if encoder == "One-Hot":
        enc = OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=False)
    else:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat_steps = [("impute", SimpleImputer(strategy="most_frequent")), ("encode", enc)]
    return ColumnTransformer([("num", Pipeline(num_steps), p.num_cols), ("cat", Pipeline(cat_steps), p.cat_cols)],
                             verbose_feature_names_out=False)


# ------------------------------------------------------------------ model catalog
def model_catalog(task: str) -> dict:
    """name -> (estimator factory, default param grid for tuning). Grid keys use the 'model__' prefix."""
    from sklearn import ensemble as E, linear_model as L, naive_bayes as NB, neighbors as N, svm as S, tree as T

    if task == "regression":
        cat = {
            "Linear Regression": (lambda: L.LinearRegression(), {}),
            "Ridge": (lambda: L.Ridge(random_state=SEED), {"model__alpha": [0.01, 0.1, 1, 10, 100]}),
            "Lasso": (lambda: L.Lasso(random_state=SEED, max_iter=5000), {"model__alpha": [0.001, 0.01, 0.1, 1]}),
            "ElasticNet": (lambda: L.ElasticNet(random_state=SEED, max_iter=5000),
                           {"model__alpha": [0.01, 0.1, 1], "model__l1_ratio": [0.2, 0.5, 0.8]}),
            "KNN": (lambda: N.KNeighborsRegressor(), {"model__n_neighbors": [3, 5, 9, 15], "model__weights": ["uniform", "distance"]}),
            "SVR": (lambda: S.SVR(), {"model__C": [0.1, 1, 10], "model__kernel": ["rbf", "linear"]}),
            "Decision Tree": (lambda: T.DecisionTreeRegressor(random_state=SEED), {"model__max_depth": [3, 5, 8, None], "model__min_samples_leaf": [1, 5, 20]}),
            "Random Forest": (lambda: E.RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
                              {"model__n_estimators": [100, 300], "model__max_depth": [None, 8, 16], "model__min_samples_leaf": [1, 3]}),
            "Gradient Boosting": (lambda: E.HistGradientBoostingRegressor(random_state=SEED),
                                  {"model__learning_rate": [0.03, 0.1, 0.3], "model__max_depth": [None, 4, 8]}),
        }
        if HAS_XGB:
            from xgboost import XGBRegressor
            cat["XGBoost"] = (lambda: XGBRegressor(n_estimators=300, learning_rate=.1, random_state=SEED, verbosity=0),
                              {"model__max_depth": [3, 6], "model__learning_rate": [0.05, 0.1]})
        if HAS_LGBM:
            from lightgbm import LGBMRegressor
            cat["LightGBM"] = (lambda: LGBMRegressor(random_state=SEED, verbose=-1), {"model__num_leaves": [15, 31, 63]})
        return cat
    cat = {
        "Logistic Regression": (lambda: L.LogisticRegression(max_iter=2000),
                                {"model__C": [0.01, 0.1, 1, 10], "model__class_weight": [None, "balanced"]}),
        "KNN": (lambda: N.KNeighborsClassifier(), {"model__n_neighbors": [3, 5, 9, 15], "model__weights": ["uniform", "distance"]}),
        "Naive Bayes": (lambda: NB.GaussianNB(), {"model__var_smoothing": [1e-9, 1e-7, 1e-5]}),
        "SVM": (lambda: S.SVC(probability=True, random_state=SEED), {"model__C": [0.1, 1, 10], "model__kernel": ["rbf", "linear"]}),
        "Decision Tree": (lambda: T.DecisionTreeClassifier(random_state=SEED), {"model__max_depth": [3, 5, 8, None], "model__min_samples_leaf": [1, 5, 20]}),
        "Random Forest": (lambda: E.RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
                          {"model__n_estimators": [100, 300], "model__max_depth": [None, 8, 16], "model__class_weight": [None, "balanced"]}),
        "Gradient Boosting": (lambda: E.HistGradientBoostingClassifier(random_state=SEED),
                              {"model__learning_rate": [0.03, 0.1, 0.3], "model__max_depth": [None, 4, 8]}),
    }
    if HAS_XGB:
        from xgboost import XGBClassifier
        cat["XGBoost"] = (lambda: XGBClassifier(n_estimators=300, learning_rate=.1, random_state=SEED, verbosity=0),
                          {"model__max_depth": [3, 6], "model__learning_rate": [0.05, 0.1]})
    if HAS_LGBM:
        from lightgbm import LGBMClassifier
        cat["LightGBM"] = (lambda: LGBMClassifier(random_state=SEED, verbose=-1), {"model__num_leaves": [15, 31, 63]})
    return cat


def build_pipeline(pre: ColumnTransformer, estimator, imbalance: str = "None"):
    if is_regressor(estimator):
        imbalance = "None"
    if imbalance in ("SMOTE", "Undersample") and HAS_IMB:
        sampler = SMOTE(random_state=SEED) if imbalance == "SMOTE" else RandomUnderSampler(random_state=SEED)
        return ImbPipeline([("pre", pre), ("sampler", sampler), ("model", estimator)])
    if imbalance == "class_weight" and hasattr(estimator, "class_weight"):
        estimator.set_params(class_weight="balanced")
    return Pipeline([("pre", pre), ("model", estimator)])


# ------------------------------------------------------------------ training & metrics
def cv_splitter(task: str, k: int):
    return KFold(k, shuffle=True, random_state=SEED) if task == "regression" else StratifiedKFold(k, shuffle=True, random_state=SEED)


def primary_metric(task: str) -> str:
    return {"regression": "r2", "binary": "roc_auc", "multiclass": "f1_macro"}[task]


def score_metrics(task: str, y_true, y_pred, y_proba=None) -> dict:
    if task == "regression":
        return {"R2": M.r2_score(y_true, y_pred), "MAE": M.mean_absolute_error(y_true, y_pred),
                "RMSE": float(np.sqrt(M.mean_squared_error(y_true, y_pred))),
                "MAPE_%": float(M.mean_absolute_percentage_error(y_true, y_pred) * 100)}
    avg = "binary" if task == "binary" else "macro"
    out = {"Accuracy": M.accuracy_score(y_true, y_pred),
           "Balanced Acc": M.balanced_accuracy_score(y_true, y_pred),
           "Precision": M.precision_score(y_true, y_pred, average=avg, zero_division=0),
           "Recall": M.recall_score(y_true, y_pred, average=avg, zero_division=0),
           "F1": M.f1_score(y_true, y_pred, average=avg, zero_division=0)}
    if y_proba is not None:
        try:
            out["ROC-AUC"] = M.roc_auc_score(y_true, y_proba[:, 1]) if task == "binary" else \
                M.roc_auc_score(y_true, y_proba, multi_class="ovr")
            out["Log loss"] = M.log_loss(y_true, y_proba)
        except ValueError:
            pass
    return out


def train_eval(p: Prepared, model_name: str, scaler="Standard", encoder="One-Hot", imbalance="None",
               cv_k: int = 5, params: dict | None = None) -> dict:
    est = model_catalog(p.task)[model_name][0]()
    if params:
        est.set_params(**{k.replace("model__", ""): v for k, v in params.items()})
    pipe = build_pipeline(make_preprocessor(p, scaler, encoder), est, imbalance)
    t0 = time.time()
    scoring = primary_metric(p.task)
    cv = cross_validate(pipe, p.X_train, p.y_train, cv=cv_splitter(p.task, cv_k), scoring=scoring,
                        return_train_score=True, n_jobs=1)
    pipe.fit(p.X_train, p.y_train)
    fit_s = time.time() - t0
    pred = pipe.predict(p.X_test)
    proba = pipe.predict_proba(p.X_test) if p.task != "regression" and hasattr(pipe, "predict_proba") else None
    mets = score_metrics(p.task, p.y_test, pred, proba)
    train_pred = pipe.predict(p.X_train)
    train_mets = score_metrics(p.task, p.y_train, train_pred,
                               pipe.predict_proba(p.X_train) if proba is not None else None)
    return {"name": model_name, "pipe": pipe, "pred": pred, "proba": proba, "test": mets, "train": train_mets,
            "cv_mean": float(np.mean(cv["test_score"])), "cv_std": float(np.std(cv["test_score"])),
            "cv_train_mean": float(np.mean(cv["train_score"])), "cv_scores": cv["test_score"],
            "cv_metric": scoring, "fit_seconds": fit_s}


def automl(p: Prepared, scaler="Standard", encoder="One-Hot", imbalance="None", cv_k=5, skip=(), progress=None) -> tuple[pd.DataFrame, dict]:
    rows, results = [], {}
    names = [n for n in model_catalog(p.task) if n not in skip]
    for i, n in enumerate(names):
        if progress:
            progress(i / len(names), f"Training {n} ...")
        try:
            r = train_eval(p, n, scaler, encoder, imbalance, cv_k)
        except Exception as e:
            rows.append({"model": n, "error": str(e)[:80]})
            continue
        results[n] = r
        rows.append({"model": n, f"CV {r['cv_metric']}": r["cv_mean"], "CV std": r["cv_std"],
                     **{f"test {k}": v for k, v in r["test"].items()}, "fit_s": r["fit_seconds"]})
    if progress:
        progress(1.0, "Done")
    lb = pd.DataFrame(rows)
    key = [c for c in lb.columns if c.startswith("CV ") and c != "CV std"]
    if key:
        lb = lb.sort_values(key[0], ascending=False).reset_index(drop=True)
    return lb.round(4), results


def tune(p: Prepared, model_name: str, grid: dict, method="Grid", n_iter=20, cv_k=5, scaler="Standard",
         encoder="One-Hot", imbalance="None") -> dict:
    est = model_catalog(p.task)[model_name][0]()
    pipe = build_pipeline(make_preprocessor(p, scaler, encoder), est, imbalance)
    cv = cv_splitter(p.task, cv_k)
    scoring = primary_metric(p.task)
    if method == "Grid":
        search = GridSearchCV(pipe, grid, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=True)
    else:
        search = RandomizedSearchCV(pipe, grid, n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=-1,
                                    random_state=SEED, return_train_score=True)
    t0 = time.time()
    search.fit(p.X_train, p.y_train)
    res = pd.DataFrame(search.cv_results_)
    cols = [c for c in res.columns if c.startswith("param_")] + ["mean_test_score", "std_test_score", "mean_train_score", "rank_test_score"]
    return {"best_params": search.best_params_, "best_score": search.best_score_, "results": res[cols].sort_values("rank_test_score"),
            "best_pipe": search.best_estimator_, "seconds": time.time() - t0, "metric": scoring}


# ------------------------------------------------------------------ diagnosis
def diagnose_fit(train_score: float, val_score: float, metric: str) -> tuple[str, str]:
    """Rule-based bias/variance verdict from train vs validation score (higher = better)."""
    gap = train_score - val_score
    good = {"r2": .7, "roc_auc": .8, "f1_macro": .75}.get(metric, .75)
    if gap > .1:
        return "Overfitting (high variance)", (f"Train {metric} {train_score:.3f} is much higher than validation {val_score:.3f}. "
                                               "Try: more regularisation, shallower trees, fewer features, more data.")
    if val_score < good and train_score < good:
        return "Underfitting (high bias)", (f"Both train ({train_score:.3f}) and validation ({val_score:.3f}) are low. "
                                            "Try: a more flexible model, more/better features, less regularisation.")
    return "Good fit", f"Train ({train_score:.3f}) and validation ({val_score:.3f}) are close and reasonably high."


def explain_metrics(task: str, m: dict, classes: list) -> list[str]:
    out = []
    if task == "regression":
        out.append(f"R2 = {m['R2']:.3f}: the model explains {max(m['R2'], 0):.0%} of the variance in the target.")
        out.append(f"On average predictions are off by {m['MAE']:.3f} (MAE); RMSE {m['RMSE']:.3f} punishes big misses more.")
        if m["RMSE"] > 1.5 * m["MAE"]:
            out.append("RMSE is much larger than MAE, so a few large errors dominate - check the residual plot for outliers.")
    else:
        out.append(f"Accuracy {m['Accuracy']:.1%}, balanced accuracy {m['Balanced Acc']:.1%}.")
        if m["Accuracy"] - m["Balanced Acc"] > .05:
            out.append("Balanced accuracy is clearly lower than accuracy - the model favours the majority class.")
        out.append(f"Precision {m['Precision']:.1%} = of the rows predicted positive, how many were right. "
                   f"Recall {m['Recall']:.1%} = of the real positives, how many were found.")
        if "ROC-AUC" in m:
            q = "excellent" if m["ROC-AUC"] > .9 else "good" if m["ROC-AUC"] > .8 else "fair" if m["ROC-AUC"] > .7 else "weak"
            out.append(f"ROC-AUC {m['ROC-AUC']:.3f} ({q}): the probability the model ranks a random positive above a random negative.")
    return out


# ------------------------------------------------------------------ plots
def confusion_fig(p: Prepared, pred, normalize=False):
    cm = M.confusion_matrix(p.y_test, pred, normalize="true" if normalize else None)
    labels = [str(c) for c in p.classes]
    return px.imshow(cm.round(3) if normalize else cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues",
                     labels=dict(x="Predicted", y="Actual"), title="Confusion matrix", template=TEMPLATE)


def roc_pr_figs(p: Prepared, proba):
    roc, pr = go.Figure(), go.Figure()
    ys = [(p.classes[1] if p.task == "binary" else c, (p.y_test == (1 if p.task == "binary" else i)).astype(int),
           proba[:, 1] if p.task == "binary" else proba[:, i])
          for i, c in (enumerate(p.classes[1:2]) if p.task == "binary" else enumerate(p.classes))]
    for name, yt, sc in ys:
        fpr, tpr, _ = M.roc_curve(yt, sc)
        roc.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {M.auc(fpr, tpr):.3f})")
        prec, rec, _ = M.precision_recall_curve(yt, sc)
        pr.add_scatter(x=rec, y=prec, mode="lines", name=f"{name} (AP {M.average_precision_score(yt, sc):.3f})")
    roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"), name="random")
    roc.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate", template=TEMPLATE)
    pr.update_layout(title="Precision-Recall curve", xaxis_title="Recall", yaxis_title="Precision", template=TEMPLATE)
    return roc, pr


def threshold_table(p: Prepared, proba, threshold: float) -> dict:
    pred = (proba[:, 1] >= threshold).astype(int)
    return score_metrics("binary", p.y_test, pred, proba), pred


def threshold_curve_fig(p: Prepared, proba):
    ts = np.linspace(.05, .95, 19)
    rows = []
    for t in ts:
        pr = (proba[:, 1] >= t).astype(int)
        rows.append({"threshold": t, "precision": M.precision_score(p.y_test, pr, zero_division=0),
                     "recall": M.recall_score(p.y_test, pr, zero_division=0), "f1": M.f1_score(p.y_test, pr, zero_division=0)})
    d = pd.DataFrame(rows).melt("threshold", var_name="metric")
    return px.line(d, x="threshold", y="value", color="metric", markers=True, title="Precision / Recall / F1 vs threshold", template=TEMPLATE)


def calibration_fig(p: Prepared, proba):
    from sklearn.calibration import calibration_curve
    frac, mean = calibration_curve(p.y_test, proba[:, 1], n_bins=10, strategy="quantile")
    fig = go.Figure()
    fig.add_scatter(x=mean, y=frac, mode="lines+markers", name="model")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"), name="perfect")
    fig.update_layout(title="Calibration curve (are probabilities trustworthy?)", xaxis_title="mean predicted probability",
                      yaxis_title="observed fraction positive", template=TEMPLATE)
    return fig


def regression_figs(p: Prepared, pred):
    d = pd.DataFrame({"actual": p.y_test, "predicted": pred})
    d["residual"] = d.actual - d.predicted
    lo, hi = d[["actual", "predicted"]].min().min(), d[["actual", "predicted"]].max().max()
    f1 = px.scatter(d, x="actual", y="predicted", opacity=.6, title="Predicted vs actual", template=TEMPLATE)
    f1.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="red"), name="perfect")
    f2 = px.scatter(d, x="predicted", y="residual", opacity=.6, title="Residuals vs predicted (should look like random noise)", template=TEMPLATE)
    f2.add_hline(y=0, line_dash="dash", line_color="red")
    f3 = px.histogram(d, x="residual", nbins=40, marginal="box", title="Residual distribution", template=TEMPLATE)
    return f1, f2, f3


def learning_curve_fig(p: Prepared, pipe, cv_k=5):
    sizes, tr, va = learning_curve(pipe, p.X_train, p.y_train, cv=cv_splitter(p.task, cv_k), scoring=primary_metric(p.task),
                                   train_sizes=np.linspace(.1, 1, 6), n_jobs=-1, shuffle=True, random_state=SEED)
    fig = go.Figure()
    for name, arr, col in (("train", tr, "#2E86AB"), ("validation", va, "#E4572E")):
        m, s = arr.mean(1), arr.std(1)
        fig.add_scatter(x=sizes, y=m, mode="lines+markers", name=name, line_color=col)
        fig.add_scatter(x=np.r_[sizes, sizes[::-1]], y=np.r_[m + s, (m - s)[::-1]], fill="toself", line=dict(width=0),
                        fillcolor=col, opacity=.15, showlegend=False, hoverinfo="skip")
    fig.update_layout(title="Learning curve (score vs amount of training data)", xaxis_title="training rows",
                      yaxis_title=primary_metric(p.task), template=TEMPLATE)
    verdict = diagnose_fit(tr.mean(1)[-1], va.mean(1)[-1], primary_metric(p.task))
    if va.mean(1)[-1] - va.mean(1)[-2] > .01:
        verdict = (verdict[0], verdict[1] + " The validation curve is still rising: more data would likely help.")
    return fig, verdict


def validation_curve_fig(p: Prepared, pipe, param: str, values: list, cv_k=5):
    tr, va = validation_curve(pipe, p.X_train, p.y_train, param_name=param, param_range=values,
                              cv=cv_splitter(p.task, cv_k), scoring=primary_metric(p.task), n_jobs=-1)
    xs = [str(v) for v in values]
    fig = go.Figure()
    fig.add_scatter(x=xs, y=tr.mean(1), mode="lines+markers", name="train")
    fig.add_scatter(x=xs, y=va.mean(1), mode="lines+markers", name="validation")
    best = xs[int(np.argmax(va.mean(1)))]
    fig.update_layout(title=f"Validation curve for {param.replace('model__', '')} (best = {best})",
                      xaxis_title=param.replace("model__", ""), yaxis_title=primary_metric(p.task), template=TEMPLATE)
    return fig, best


def feature_names(pipe) -> list[str]:
    try:
        return list(pipe.named_steps["pre"].get_feature_names_out())
    except Exception:
        return []


def builtin_importance(pipe) -> pd.DataFrame | None:
    model = pipe.named_steps["model"]
    names = feature_names(pipe)
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "coef_"):
        vals = np.abs(np.atleast_2d(model.coef_)).mean(0)
    else:
        return None
    if len(names) != len(vals):
        names = [f"f{i}" for i in range(len(vals))]
    return pd.DataFrame({"feature": names, "importance": vals}).sort_values("importance", ascending=False)


def permutation_fig(p: Prepared, pipe, n_repeats=5):
    r = permutation_importance(pipe, p.X_test, p.y_test, n_repeats=n_repeats, random_state=SEED,
                               scoring=primary_metric(p.task), n_jobs=-1)
    d = pd.DataFrame({"feature": p.X_test.columns, "importance": r.importances_mean, "std": r.importances_std})
    d = d.sort_values("importance", ascending=False)
    fig = px.bar(d.head(20), x="importance", y="feature", error_x="std", orientation="h", template=TEMPLATE,
                 title="Permutation importance (drop in score when a column is shuffled)")
    fig.update_yaxes(autorange="reversed")
    return fig, d


def pdp_fig(p: Prepared, pipe, feature: str):
    res = partial_dependence(pipe, p.X_train, [feature], kind="average", grid_resolution=30)
    avg = res["average"]
    xs = res["grid_values"][0]
    fig = go.Figure()
    if p.task == "multiclass":
        for i, c in enumerate(p.classes):
            fig.add_scatter(x=xs, y=avg[i], mode="lines", name=str(c))
    else:
        fig.add_scatter(x=xs, y=avg[0], mode="lines+markers", name="partial dependence")
    fig.update_layout(title=f"Partial dependence on `{feature}` (average effect, others held fixed)",
                      xaxis_title=feature, yaxis_title="prediction", template=TEMPLATE)
    return fig


def shap_fig(p: Prepared, pipe, max_rows=300):
    import shap
    pre, model = pipe.named_steps["pre"], pipe.named_steps["model"]
    Xt = pre.transform(p.X_test.iloc[:max_rows])
    names = feature_names(pipe)
    bg = pre.transform(p.X_train.sample(min(100, len(p.X_train)), random_state=SEED))
    f = (lambda X: model.predict_proba(X)[:, 1]) if p.task == "binary" else model.predict
    explainer = shap.Explainer(f, bg, feature_names=names)
    sv = explainer(Xt)
    vals = np.abs(sv.values).mean(0)
    d = pd.DataFrame({"feature": names, "mean_|SHAP|": vals}).sort_values("mean_|SHAP|", ascending=False).head(20)
    fig = px.bar(d, x="mean_|SHAP|", y="feature", orientation="h", template=TEMPLATE, title="SHAP global importance")
    fig.update_yaxes(autorange="reversed")
    return fig


# ------------------------------------------------------------------ export
def model_bytes(pipe, p: Prepared) -> bytes:
    buf = io.BytesIO()
    joblib.dump({"pipeline": pipe, "task": p.task, "target": p.target, "features": p.features,
                 "classes": p.classes}, buf)
    return buf.getvalue()


def model_card(r: dict, p: Prepared, dataset: str) -> str:
    lines = [f"# Model card - {r['name']}", "", f"- **Dataset:** {dataset}", f"- **Task:** {p.task}",
             f"- **Target:** `{p.target}`", f"- **Features ({len(p.features)}):** {', '.join(p.features)}",
             f"- **Train / test rows:** {len(p.X_train)} / {len(p.X_test)}",
             f"- **CV {r['cv_metric']}:** {r['cv_mean']:.4f} +/- {r['cv_std']:.4f}", "", "## Test metrics", ""]
    lines += [f"- {k}: {v:.4f}" for k, v in r["test"].items()]
    lines += ["", "## Interpretation", ""] + [f"- {t}" for t in explain_metrics(p.task, r["test"], p.classes)]
    lines += ["", "## Usage", "", "```python", "import joblib", "bundle = joblib.load('model.joblib')",
              "preds = bundle['pipeline'].predict(new_df[bundle['features']])", "```"]
    return "\n".join(lines)


# ------------------------------------------------------------------ unsupervised
def unsupervised_matrix(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, list[str]]:
    d = df[features].copy()
    nums = [c for c in features if pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c])]
    cats = [c for c in features if c not in nums]
    for c in cats:
        d[c] = d[c].astype(str)
    pre = ColumnTransformer([("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), nums),
                             ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                                               ("e", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cats)],
                            verbose_feature_names_out=False)
    X = pre.fit_transform(d)
    return X, list(pre.get_feature_names_out())


def elbow_silhouette(X: np.ndarray, kmax: int = 10):
    from sklearn.cluster import KMeans
    rows = []
    for k in range(2, kmax + 1):
        km = KMeans(k, n_init=10, random_state=SEED).fit(X)
        sample = np.random.default_rng(SEED).choice(len(X), min(len(X), 3000), replace=False)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": M.silhouette_score(X[sample], km.labels_[sample])})
    d = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_scatter(x=d.k, y=d.inertia, name="inertia (elbow)", mode="lines+markers")
    fig.add_scatter(x=d.k, y=d.silhouette, name="silhouette", mode="lines+markers", yaxis="y2")
    fig.update_layout(title="Choosing k: elbow (inertia) and silhouette", xaxis_title="k", template=TEMPLATE,
                      yaxis=dict(title="inertia"), yaxis2=dict(title="silhouette", overlaying="y", side="right"))
    return fig, int(d.loc[d.silhouette.idxmax(), "k"]), d


def run_clustering(X: np.ndarray, algo: str, k: int = 3, eps: float = .8, min_samples: int = 5):
    from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
    if algo == "KMeans":
        labels = KMeans(k, n_init=10, random_state=SEED).fit_predict(X)
    elif algo == "DBSCAN":
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    else:
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    sil = M.silhouette_score(X, labels) if 1 < n_clusters < len(X) else float("nan")
    db = M.davies_bouldin_score(X, labels) if 1 < n_clusters < len(X) else float("nan")
    return labels, {"clusters": n_clusters, "noise_points": int((labels == -1).sum()), "silhouette": sil, "davies_bouldin": db}


def projection_fig(X: np.ndarray, labels, method: str = "PCA", hover: pd.DataFrame | None = None):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    if method == "PCA":
        Z = PCA(2, random_state=SEED).fit_transform(X)
    else:
        n = min(len(X), 2000)
        idx = np.random.default_rng(SEED).choice(len(X), n, replace=False)
        X, labels = X[idx], np.asarray(labels)[idx]
        hover = hover.iloc[idx] if hover is not None else None
        Z = TSNE(2, random_state=SEED, perplexity=min(30, n - 1), init="pca").fit_transform(X)
    d = pd.DataFrame({"dim1": Z[:, 0], "dim2": Z[:, 1], "cluster": np.asarray(labels).astype(str)})
    if hover is not None:
        d = pd.concat([d, hover.reset_index(drop=True).iloc[:, :4]], axis=1)
    return px.scatter(d, x="dim1", y="dim2", color="cluster", hover_data=d.columns, opacity=.75,
                      title=f"Clusters projected to 2D with {method}", template=TEMPLATE)


def pca_variance_fig(X: np.ndarray):
    from sklearn.decomposition import PCA
    pca = PCA(random_state=SEED).fit(X)
    ev = pca.explained_variance_ratio_[:20]
    d = pd.DataFrame({"component": [f"PC{i + 1}" for i in range(len(ev))], "explained": ev, "cumulative": np.cumsum(ev)})
    fig = go.Figure()
    fig.add_bar(x=d.component, y=d.explained, name="explained variance")
    fig.add_scatter(x=d.component, y=d.cumulative, name="cumulative", mode="lines+markers")
    fig.update_layout(title="PCA explained variance", template=TEMPLATE, yaxis_tickformat=".0%")
    n95 = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), .95) + 1)
    return fig, n95


def dendrogram_fig(X: np.ndarray, max_rows: int = 200):
    import plotly.figure_factory as ff
    idx = np.random.default_rng(SEED).choice(len(X), min(len(X), max_rows), replace=False)
    fig = ff.create_dendrogram(X[idx], linkagefun=lambda x: __import__("scipy.cluster.hierarchy", fromlist=["linkage"]).linkage(x, "ward"))
    fig.update_layout(title=f"Hierarchical clustering dendrogram (Ward, {len(idx)} sampled rows)", template=TEMPLATE,
                      xaxis_showticklabels=False, height=450)
    return fig


def cluster_profile(df: pd.DataFrame, labels, features: list[str]) -> pd.DataFrame:
    d = df[features].copy()
    d["cluster"] = labels
    nums = [c for c in features if pd.api.types.is_numeric_dtype(d[c])]
    prof = d.groupby("cluster")[nums].mean().round(3)
    prof.insert(0, "n_rows", d.groupby("cluster").size())
    return prof
