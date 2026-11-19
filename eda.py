"""15-step EDA engine. Every step returns a dict with tables, figures and findings.

Findings are rule-based (no LLM): each is a dict {level, text} where level is
"info", "warn" or "good".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from data import categorical_cols, column_kind, column_kinds, numeric_cols, sample_for_plot

STEPS = [
    "1. Ingest & provenance",
    "2. Shape & memory footprint",
    "3. Schema & data types",
    "4. Type coercion",
    "5. Missing values",
    "6. Duplicates",
    "7. Descriptive statistics",
    "8. Cardinality & frequency",
    "9. Distribution, shape & normality",
    "10. Outliers",
    "11. Univariate explorer",
    "12. Bivariate explorer",
    "13. Correlation & multicollinearity",
    "14. Target-focused analysis",
    "15. Synthesis, quality score & export",
]

TEMPLATE = "plotly_white"


def F(level: str, text: str) -> dict:
    return {"level": level, "text": text}


# ---------------------------------------------------------------- step 1
def step_ingest(df: pd.DataFrame, source: str) -> dict:
    return {
        "source": source,
        "head": df.head(10),
        "tail": df.tail(10),
        "sample": df.sample(min(10, len(df)), random_state=0),
        "findings": [F("info", f"Loaded **{source}** with {len(df):,} rows and {df.shape[1]} columns.")],
    }


# ---------------------------------------------------------------- step 2
def step_shape(df: pd.DataFrame) -> dict:
    cells = df.size
    missing = int(df.isna().sum().sum())
    num = df.select_dtypes("number")
    inf = int(np.isinf(num.to_numpy(dtype=float, na_value=np.nan)).sum()) if not num.empty else 0
    populated = cells - missing - inf
    mem = df.memory_usage(deep=True).sum()
    opt = optimize_dtypes(df).memory_usage(deep=True).sum()
    fig = px.pie(names=["Populated", "Missing", "Infinite"], values=[populated, missing, inf],
                 hole=.5, title="Share of cells", template=TEMPLATE,
                 color_discrete_sequence=["#2E86AB", "#E4572E", "#F3A712"])
    findings = [F("info", f"The dataset holds {len(df):,} rows and {df.shape[1]} columns ({cells:,} cells).")]
    if missing == 0 and inf == 0:
        findings.append(F("good", "Every cell is populated - no missing or infinite values."))
    else:
        findings.append(F("warn", f"{missing:,} missing and {inf:,} infinite cells ({(missing + inf) / cells:.1%} of all cells)."))
    if mem > 0 and opt < mem * .7:
        findings.append(F("info", f"Downcasting types would shrink memory from {mem / 1e6:.2f} MB to {opt / 1e6:.2f} MB."))
    return {"rows": len(df), "cols": df.shape[1], "cells": cells, "missing": missing, "inf": inf,
            "memory_mb": mem / 1e6, "optimized_mb": opt / 1e6, "fig": fig, "findings": findings}


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_integer_dtype(s):
            out[c] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_float_dtype(s):
            out[c] = pd.to_numeric(s, downcast="float")
        elif not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_datetime64_any_dtype(s) and s.nunique() < 0.5 * len(s):
            out[c] = s.astype("category")
    return out


# ---------------------------------------------------------------- step 3
def step_schema(df: pd.DataFrame) -> dict:
    kinds = column_kinds(df)
    rows = []
    for c in df.columns:
        s = df[c]
        ex = s.dropna().iloc[0] if s.notna().any() else None
        rows.append({
            "column": c, "dtype": str(s.dtype), "kind": kinds[c], "distinct": s.nunique(),
            "missing": int(s.isna().sum()), "missing_%": round(s.isna().mean() * 100, 2),
            "example": str(ex)[:40], "usable_for_stats": kinds[c] not in ("id", "text"),
            "memory_kb": round(s.memory_usage(deep=True) / 1024, 1),
        })
    table = pd.DataFrame(rows)
    counts = table["kind"].value_counts().reset_index()
    counts.columns = ["kind", "count"]
    fig = px.bar(counts, x="kind", y="count", color="kind", title="Column kinds", template=TEMPLATE)
    findings = [F("info", ", ".join(f"{r['count']} {r['kind']}" for _, r in counts.iterrows()) + " columns.")]
    for c, k in kinds.items():
        if k == "id":
            findings.append(F("warn", f"`{c}` looks like an ID (all values unique) - exclude it from modelling."))
    return {"table": table, "fig": fig, "kinds": kinds, "findings": findings}


# ---------------------------------------------------------------- step 4
def step_coercion(df: pd.DataFrame) -> dict:
    sugg = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            nn = s.dropna().astype(str)
            if nn.empty:
                continue
            as_num = pd.to_numeric(nn.str.replace(",", "", regex=False), errors="coerce")
            if as_num.notna().mean() > .95:
                sugg.append({"column": c, "current": "object", "suggested": "numeric",
                             "reason": f"{as_num.notna().mean():.0%} of values parse as numbers"})
                continue
            as_dt = pd.to_datetime(nn.head(500), errors="coerce", format="mixed")
            if as_dt.notna().mean() > .95:
                sugg.append({"column": c, "current": "object", "suggested": "datetime",
                             "reason": "values parse as dates"})
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and s.nunique() == 2:
            sugg.append({"column": c, "current": str(s.dtype), "suggested": "boolean",
                         "reason": "only two distinct values"})
    table = pd.DataFrame(sugg, columns=["column", "current", "suggested", "reason"])
    findings = [F("good", "All columns are stored with sensible types.")] if table.empty else \
        [F("warn", f"{len(table)} column(s) may be stored with the wrong type - see suggestions.")]
    return {"table": table, "findings": findings}


def apply_coercion(df: pd.DataFrame, column: str, target: str) -> pd.DataFrame:
    out = df.copy()
    if target == "numeric":
        out[column] = pd.to_numeric(out[column].astype(str).str.replace(",", "", regex=False), errors="coerce")
    elif target == "datetime":
        out[column] = pd.to_datetime(out[column], errors="coerce", format="mixed")
    elif target == "boolean":
        vals = sorted(out[column].dropna().unique())
        out[column] = out[column].map({vals[0]: False, vals[-1]: True})
    return out


# ---------------------------------------------------------------- step 5
def step_missing(df: pd.DataFrame) -> dict:
    miss = df.isna().sum()
    table = pd.DataFrame({"missing": miss, "missing_%": (miss / len(df) * 100).round(2)})
    table = table[table.missing > 0].sort_values("missing", ascending=False)
    table["suggested_fix"] = [suggest_impute(df[c], p) for c, p in table["missing_%"].items()]
    findings, figs = [], []
    if table.empty:
        findings.append(F("good", "No missing values."))
    else:
        figs.append(px.bar(table.reset_index(names="column"), x="column", y="missing_%",
                           title="Missing values per column (%)", template=TEMPLATE, color="missing_%",
                           color_continuous_scale="Reds"))
        samp, _ = sample_for_plot(df, 400)
        mat = samp[table.index].isna().astype(int)
        figs.append(px.imshow(mat.T, aspect="auto", color_continuous_scale=["#EEF2F7", "#E4572E"],
                              title="Missingness pattern (red = missing, sample of rows)", template=TEMPLATE))
        for c, r in table.iterrows():
            lvl = "warn" if r["missing_%"] > 5 else "info"
            findings.append(F(lvl, f"`{c}` is {r['missing_%']}% missing -> {r['suggested_fix']}."))
    return {"table": table, "figs": figs, "findings": findings}


def suggest_impute(s: pd.Series, pct: float) -> str:
    if pct > 60:
        return "drop the column"
    if pd.api.types.is_numeric_dtype(s):
        return "impute median" if abs(s.skew()) > 1 else "impute mean"
    return "impute most frequent"


# ---------------------------------------------------------------- step 6
def step_duplicates(df: pd.DataFrame) -> dict:
    hashable = df.copy()
    for c in hashable.columns:
        if hashable[c].map(type).eq(list).any():
            hashable[c] = hashable[c].astype(str)
    mask = hashable.duplicated(keep=False)
    n = int(hashable.duplicated().sum())
    findings = [F("good", "No duplicate rows.")] if n == 0 else \
        [F("warn", f"{n} duplicate row(s) ({n / len(df):.1%}). Check whether they are real repeat events before dropping.")]
    return {"count": n, "rows": df[mask].head(50), "findings": findings}


# ---------------------------------------------------------------- step 7
def step_descriptive(df: pd.DataFrame) -> dict:
    num = df[numeric_cols(df)]
    if num.empty:
        return {"table": pd.DataFrame(), "fig": None, "findings": [F("info", "No numeric columns.")]}
    d = pd.DataFrame({
        "count": num.count(), "missing": num.isna().sum(), "mean": num.mean(), "std": num.std(),
        "cv": num.std() / num.mean().replace(0, np.nan), "min": num.min(),
        "p1": num.quantile(.01), "p5": num.quantile(.05), "q1": num.quantile(.25),
        "median": num.median(), "q3": num.quantile(.75), "p95": num.quantile(.95),
        "p99": num.quantile(.99), "max": num.max(), "skew": num.skew(), "kurtosis": num.kurt(),
    }).round(3)
    scaled = (num - num.min()) / (num.max() - num.min()).replace(0, np.nan) * 100
    samp, _ = sample_for_plot(scaled, 3000)
    fig = px.box(samp.melt(var_name="column", value_name="scaled_0_100"), x="column", y="scaled_0_100",
                 title="Spread of each numeric column on a common 0-100 scale", template=TEMPLATE, color="column")
    findings = []
    for c, r in d.iterrows():
        if pd.notna(r["cv"]) and abs(r["cv"]) > 1:
            findings.append(F("info", f"`{c}` varies a lot (CV = {r['cv']:.2f}): the std is larger than the mean."))
        if r["mean"] > 0 and r["median"] > 0 and r["mean"] / r["median"] > 1.25:
            findings.append(F("info", f"`{c}` mean ({r['mean']:.2f}) is well above its median ({r['median']:.2f}) - a long right tail."))
    if not findings:
        findings.append(F("good", "Numeric columns have moderate spread."))
    return {"table": d, "fig": fig, "findings": findings}


# ---------------------------------------------------------------- step 8
def step_cardinality(df: pd.DataFrame) -> dict:
    kinds = column_kinds(df)
    rows = []
    for c, k in kinds.items():
        if k in ("categorical", "boolean", "discrete", "text", "id"):
            vc = df[c].value_counts(dropna=True)
            if vc.empty:
                continue
            rows.append({"column": c, "kind": k, "distinct": df[c].nunique(), "most_common": str(vc.index[0]),
                         "top_share_%": round(vc.iloc[0] / vc.sum() * 100, 1),
                         "rare_levels(<1%)": int((vc / vc.sum() < .01).sum())})
    table = pd.DataFrame(rows)
    findings = []
    for _, r in table.iterrows():
        if r["top_share_%"] > 90:
            findings.append(F("warn", f"`{r['column']}` is nearly constant ({r['top_share_%']}% = '{r['most_common']}')."))
        if r["distinct"] > 30 and r["kind"] == "categorical":
            findings.append(F("info", f"`{r['column']}` has high cardinality ({r['distinct']} levels) - consider grouping rare levels."))
    if not findings and not table.empty:
        findings.append(F("good", "Categorical columns have healthy cardinality."))
    return {"table": table, "findings": findings}


def frequency_fig(df: pd.DataFrame, col: str, top: int = 20):
    vc = df[col].astype(str).value_counts().head(top).reset_index()
    vc.columns = [col, "count"]
    vc["share_%"] = (vc["count"] / len(df) * 100).round(1)
    vc["cum_%"] = vc["share_%"].cumsum()
    fig = go.Figure()
    fig.add_bar(x=vc[col], y=vc["count"], name="count", marker_color="#2E86AB")
    fig.add_scatter(x=vc[col], y=vc["cum_%"], name="cumulative %", yaxis="y2", mode="lines+markers",
                    line_color="#E4572E")
    fig.update_layout(title=f"Frequency & Pareto - {col}", template=TEMPLATE,
                      yaxis2=dict(overlaying="y", side="right", range=[0, 105], title="cumulative %"))
    top_row = vc.iloc[0]
    return fig, vc, F("info", f"'{top_row[col]}' is the largest level of `{col}` at {top_row['share_%']}% of rows.")


# ---------------------------------------------------------------- step 9
def step_distribution(df: pd.DataFrame) -> dict:
    rows = []
    for c in numeric_cols(df):
        s = df[c].dropna()
        if len(s) < 8 or s.nunique() < 3:
            continue
        samp = s.sample(min(len(s), 5000), random_state=0)
        test = "Shapiro-Wilk" if len(samp) <= 5000 else "D'Agostino"
        p = stats.shapiro(samp).pvalue if test == "Shapiro-Wilk" else stats.normaltest(samp).pvalue
        sk = s.skew()
        shape = "symmetric" if abs(sk) < .5 else ("right-skewed" if sk > 0 else "left-skewed")
        if abs(sk) >= 1:
            shape = "highly " + shape
        rows.append({"column": c, "skew": round(sk, 3), "kurtosis": round(s.kurt(), 3), "shape": shape,
                     "normality_test": test, "p_value": float(f"{p:.4g}"), "normal(p>0.05)": p > .05,
                     "suggested_transform": suggest_transform(s)})
    table = pd.DataFrame(rows)
    findings = []
    for _, r in table.iterrows():
        if "highly" in r["shape"]:
            findings.append(F("warn", f"`{r['column']}` is {r['shape']} (skew {r['skew']}). Try a {r['suggested_transform']} transform before linear models."))
    n_norm = int(table["normal(p>0.05)"].sum()) if not table.empty else 0
    findings.append(F("info", f"{n_norm} of {len(table)} numeric columns pass the normality test (p > 0.05)."))
    return {"table": table, "findings": findings}


def suggest_transform(s: pd.Series) -> str:
    sk = s.skew()
    if abs(sk) < .5:
        return "none"
    if sk > 0:
        return "log1p" if s.min() >= 0 else "Yeo-Johnson"
    return "square / Yeo-Johnson"


def distribution_figs(df: pd.DataFrame, col: str):
    s = df[col].dropna()
    hist = px.histogram(df, x=col, marginal="box", nbins=40, title=f"Histogram - {col}", template=TEMPLATE,
                        color_discrete_sequence=["#2E86AB"])
    samp = s.sample(min(len(s), 3000), random_state=0)
    (osm, osr), (slope, intercept, _) = stats.probplot(samp, dist="norm")
    qq = go.Figure()
    qq.add_scatter(x=osm, y=osr, mode="markers", name="data", marker=dict(size=4, color="#2E86AB"))
    qq.add_scatter(x=osm, y=slope * np.array(osm) + intercept, mode="lines", name="normal line",
                   line_color="#E4572E")
    qq.update_layout(title=f"Q-Q plot vs normal - {col}", xaxis_title="theoretical quantiles",
                     yaxis_title="sample quantiles", template=TEMPLATE)
    return hist, qq


# ---------------------------------------------------------------- step 10
def step_outliers(df: pd.DataFrame) -> dict:
    rows = []
    for c in numeric_cols(df):
        s = df[c].dropna()
        if s.nunique() < 5:
            continue
        q1, q3 = s.quantile([.25, .75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_iqr = int(((s < lo) | (s > hi)).sum())
        z = np.abs((s - s.mean()) / (s.std() or 1))
        rows.append({"column": c, "lower_fence": round(lo, 3), "upper_fence": round(hi, 3),
                     "iqr_outliers": n_iqr, "iqr_outliers_%": round(n_iqr / len(s) * 100, 2),
                     "z>3_outliers": int((z > 3).sum())})
    table = pd.DataFrame(rows)
    findings = []
    for _, r in table.iterrows():
        if r["iqr_outliers_%"] > 5:
            findings.append(F("warn", f"`{r['column']}` has {r['iqr_outliers']} IQR outliers ({r['iqr_outliers_%']}%). Investigate, cap at percentiles, or use robust models (trees, RobustScaler)."))
    if not findings:
        findings.append(F("good", "No column has more than 5% outliers by the IQR rule."))
    return {"table": table, "findings": findings}


def clip_outliers(df: pd.DataFrame, cols: list[str], lower: float = .01, upper: float = .99) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        lo, hi = out[c].quantile([lower, upper])
        out[c] = out[c].clip(lo, hi)
    return out


# ---------------------------------------------------------------- step 11
UNI_NUMERIC = ["Histogram", "Box", "Violin", "ECDF", "Density (KDE)"]
UNI_CATEG = ["Bar", "Pie", "Treemap"]


def univariate_fig(df: pd.DataFrame, col: str, kind: str, color: str | None = None, histnorm: str | None = None):
    d, _ = sample_for_plot(df, 20000)
    title = f"{kind} - {col}" + (f" by {color}" if color else "")
    if kind == "Histogram":
        return px.histogram(d, x=col, color=color, histnorm=histnorm, barmode="overlay", opacity=.75,
                            nbins=40, title=title, template=TEMPLATE)
    if kind == "Box":
        return px.box(d, x=color, y=col, color=color, points="outliers", title=title, template=TEMPLATE)
    if kind == "Violin":
        return px.violin(d, x=color, y=col, color=color, box=True, title=title, template=TEMPLATE)
    if kind == "ECDF":
        return px.ecdf(d, x=col, color=color, title=title, template=TEMPLATE)
    if kind == "Density (KDE)":
        fig = go.Figure()
        groups = [(None, d)] if not color else list(d.groupby(color))
        for name, g in groups:
            s = g[col].dropna()
            if s.nunique() < 2:
                continue
            xs = np.linspace(s.min(), s.max(), 200)
            fig.add_scatter(x=xs, y=stats.gaussian_kde(s)(xs), mode="lines", fill="tozeroy",
                            name=str(name) if name is not None else col)
        fig.update_layout(title=title, template=TEMPLATE, xaxis_title=col, yaxis_title="density")
        return fig
    vc = d[col].astype(str).value_counts().reset_index()
    vc.columns = [col, "count"]
    if kind == "Pie":
        return px.pie(vc.head(15), names=col, values="count", hole=.4, title=title, template=TEMPLATE)
    if kind == "Treemap":
        return px.treemap(vc.head(40), path=[col], values="count", title=title, template=TEMPLATE)
    if color:
        ct = d.groupby([col, color]).size().reset_index(name="count")
        ct[col] = ct[col].astype(str)
        ct[color] = ct[color].astype(str)
        return px.bar(ct, x=col, y="count", color=color, barmode="group", title=title, template=TEMPLATE)
    return px.bar(vc.head(30), x=col, y="count", title=title, template=TEMPLATE)


# ---------------------------------------------------------------- step 12
def bivariate(df: pd.DataFrame, x: str, y: str, color: str | None = None) -> tuple[go.Figure, list]:
    d, _ = sample_for_plot(df, 5000)
    xn, yn = x in numeric_cols(df), y in numeric_cols(df)
    findings = []
    if xn and yn:
        d = d.assign(_row=d.index)
        fig = px.scatter(d, x=x, y=y, color=color, trendline="ols" if not color else None, custom_data=["_row"],
                         opacity=.7, title=f"{y} vs {x}", template=TEMPLATE)
        pair = df[[x, y]].dropna()
        r, p = stats.pearsonr(pair[x], pair[y])
        rho, _ = stats.spearmanr(pair[x], pair[y])
        findings.append(F("info", f"Pearson r = {r:.3f} (p = {p:.3g}), Spearman rho = {rho:.3f}: {strength(r)} {'positive' if r > 0 else 'negative'} relationship."))
    elif xn != yn:
        num, cat = (x, y) if xn else (y, x)
        fig = px.box(d, x=cat, y=num, color=color or cat, points="outliers",
                     title=f"{num} by {cat}", template=TEMPLATE)
        groups = [g[num].dropna() for _, g in df.groupby(cat) if g[num].notna().sum() > 2]
        if len(groups) >= 2:
            fstat, p = stats.f_oneway(*groups)
            kw = stats.kruskal(*groups).pvalue
            means = df.groupby(cat)[num].mean().sort_values()
            verdict = "differs significantly" if p < .05 else "does not differ significantly"
            findings.append(F("warn" if p < .05 else "info",
                              f"`{num}` {verdict} across `{cat}` (ANOVA p = {p:.3g}, Kruskal p = {kw:.3g}). Highest mean: '{means.index[-1]}' ({means.iloc[-1]:.2f}); lowest: '{means.index[0]}' ({means.iloc[0]:.2f})."))
    else:
        ct = pd.crosstab(df[x], df[y])
        fig = px.imshow(ct, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                        title=f"Crosstab {x} x {y}", template=TEMPLATE)
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            v = np.sqrt(chi2 / (ct.to_numpy().sum() * (min(ct.shape) - 1)))
            findings.append(F("warn" if p < .05 else "info",
                              f"Chi-square p = {p:.3g}, Cramer's V = {v:.3f}: `{x}` and `{y}` are {'associated' if p < .05 else 'not clearly associated'} ({strength(v)})."))
    return fig, findings


def strength(r: float) -> str:
    a = abs(r)
    return "very strong" if a >= .8 else "strong" if a >= .6 else "moderate" if a >= .4 else "weak" if a >= .2 else "negligible"


# ---------------------------------------------------------------- step 13
def step_correlation(df: pd.DataFrame, method: str = "pearson") -> dict:
    num = df[numeric_cols(df)].dropna(axis=1, how="all")
    num = num.loc[:, num.nunique() > 1]
    if num.shape[1] < 2:
        return {"corr": pd.DataFrame(), "fig": None, "pairs": pd.DataFrame(), "vif": pd.DataFrame(),
                "findings": [F("info", "Need at least two numeric columns for correlation.")]}
    corr = num.corr(method=method)
    fig = px.imshow(corr.round(2), text_auto=True, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                    title=f"{method.title()} correlation", template=TEMPLATE, aspect="auto")
    iu = np.triu_indices_from(corr, 1)
    pairs = pd.DataFrame({"col_a": corr.index[iu[0]], "col_b": corr.columns[iu[1]], "r": corr.to_numpy()[iu]})
    pairs["abs_r"] = pairs.r.abs()
    pairs["strength"] = pairs.r.map(strength)
    pairs = pairs.sort_values("abs_r", ascending=False).head(15).round(3)
    vif = compute_vif(num)
    findings = []
    for _, r in pairs.head(3).iterrows():
        findings.append(F("info", f"`{r.col_a}` and `{r.col_b}`: r = {r.r} ({r.strength})."))
    for _, r in vif[vif.VIF > 10].iterrows():
        findings.append(F("warn", f"`{r.column}` has VIF {r.VIF:.1f} (> 10) - strong multicollinearity; linear model coefficients will be unstable."))
    return {"corr": corr, "fig": fig, "pairs": pairs, "vif": vif, "findings": findings}


def compute_vif(num: pd.DataFrame) -> pd.DataFrame:
    X = num.dropna()
    if X.shape[1] < 2 or len(X) < X.shape[1] + 2:
        return pd.DataFrame(columns=["column", "VIF"])
    X = (X - X.mean()) / X.std().replace(0, 1)
    out = []
    for c in X.columns:
        others = X.drop(columns=c).to_numpy()
        y = X[c].to_numpy()
        A = np.c_[np.ones(len(others)), others]
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ beta
        r2 = 1 - resid.var() / (y.var() or 1)
        out.append({"column": c, "VIF": round(1 / max(1 - r2, 1e-6), 2)})
    return pd.DataFrame(out).sort_values("VIF", ascending=False)


# ---------------------------------------------------------------- step 14
def step_target(df: pd.DataFrame, target: str) -> dict:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    y = df[target]
    is_reg = target in numeric_cols(df) and y.nunique() > 15
    feats = [c for c in df.columns if c != target and column_kind(df[c]) not in ("id", "text", "datetime")]
    d = df[feats + [target]].dropna(subset=[target])
    X = d[feats].copy()
    discrete = []
    for c in feats:
        if not pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(str).astype("category").cat.codes
            discrete.append(True)
        else:
            X[c] = X[c].fillna(X[c].median())
            discrete.append(False)
    yy = d[target] if is_reg else d[target].astype(str)
    fn = mutual_info_regression if is_reg else mutual_info_classif
    mi = fn(X, yy, discrete_features=discrete, random_state=0)
    mi = pd.DataFrame({"feature": feats, "mutual_information": mi}).sort_values("mutual_information", ascending=False)
    fig = px.bar(mi, x="mutual_information", y="feature", orientation="h", template=TEMPLATE,
                 title=f"Mutual information with `{target}` (captures non-linear signal)")
    fig.update_yaxes(autorange="reversed")
    findings = [F("info", f"Target `{target}` treated as **{'regression' if is_reg else 'classification'}**.")]
    if not mi.empty:
        findings.append(F("info", "Most informative features: " + ", ".join(f"`{f}`" for f in mi.feature.head(3)) + "."))
        weak = mi[mi.mutual_information < .005].feature.tolist()
        if weak:
            findings.append(F("info", "Near-zero signal (candidates to drop): " + ", ".join(f"`{w}`" for w in weak[:6]) + "."))
    if not is_reg:
        vc = y.value_counts(normalize=True)
        if vc.min() < .2:
            findings.append(F("warn", f"Class imbalance: smallest class '{vc.idxmin()}' is only {vc.min():.1%} of rows. Use stratified splits, F1/ROC-AUC, class weights or SMOTE."))
        bal = px.bar(vc.reset_index(), x=vc.index.name or "index", y="proportion", title="Target class balance",
                     template=TEMPLATE)
    else:
        bal = px.histogram(df, x=target, nbins=40, marginal="box", title="Target distribution", template=TEMPLATE)
    return {"mi": mi, "fig": fig, "balance_fig": bal, "is_regression": is_reg, "findings": findings}


# ---------------------------------------------------------------- 4 dimensions
def four_dimensions(df: pd.DataFrame) -> dict[str, list]:
    """Auto-pick representative charts for Composition, Distribution, Comparison, Relationship."""
    d, _ = sample_for_plot(df, 5000)
    nums, cats = numeric_cols(df), [c for c in categorical_cols(df, 20)]
    out = {"Composition": [], "Distribution": [], "Comparison": [], "Relationship": []}
    for c in cats[:3]:
        vc = d[c].astype(str).value_counts().reset_index()
        vc.columns = [c, "count"]
        out["Composition"].append(px.pie(vc, names=c, values="count", hole=.45, title=f"Composition of {c}", template=TEMPLATE))
    if len(cats) >= 2:
        ct = d.groupby([cats[0], cats[1]]).size().reset_index(name="count").astype({cats[0]: str, cats[1]: str})
        out["Composition"].append(px.bar(ct, x=cats[0], y="count", color=cats[1], title=f"{cats[1]} within {cats[0]} (stacked)", template=TEMPLATE))
    for c in nums[:4]:
        out["Distribution"].append(px.histogram(d, x=c, marginal="violin", nbins=40, title=f"Distribution of {c}", template=TEMPLATE))
    for cat in cats[:2]:
        for num in nums[:2]:
            g = d.groupby(cat)[num].agg(["mean", "std", "count"]).reset_index()
            g[cat] = g[cat].astype(str)
            out["Comparison"].append(px.bar(g, x=cat, y="mean", error_y="std", title=f"Mean {num} by {cat} (+/- std)", template=TEMPLATE, color=cat))
    dt = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if dt and nums:
        ts = df.set_index(dt[0])[nums[0]].resample("D").mean().reset_index()
        out["Comparison"].append(px.line(ts, x=dt[0], y=nums[0], title=f"{nums[0]} over time (daily mean)", template=TEMPLATE))
    if len(nums) >= 2:
        out["Relationship"].append(px.scatter_matrix(d, dimensions=nums[:5], color=cats[0] if cats else None,
                                                     title="Scatter matrix", template=TEMPLATE, height=700))
        out["Relationship"].append(px.scatter(d, x=nums[0], y=nums[1], color=cats[0] if cats else None,
                                              trendline="ols" if not cats else None,
                                              title=f"{nums[1]} vs {nums[0]}", template=TEMPLATE))
    return out


# ---------------------------------------------------------------- step 15
def quality_score(df: pd.DataFrame) -> dict:
    miss = df.isna().mean().mean()
    dup = df.duplicated().mean() if len(df) else 0
    out_tbl = step_outliers(df)["table"]
    out_rate = (out_tbl["iqr_outliers_%"].mean() / 100) if not out_tbl.empty else 0
    const = sum(df[c].nunique() <= 1 for c in df.columns) / max(df.shape[1], 1)
    score = 100 - 40 * min(miss * 3, 1) - 25 * min(dup * 4, 1) - 20 * min(out_rate * 4, 1) - 15 * const
    grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D" if score >= 60 else "F"
    return {"score": round(score, 1), "grade": grade,
            "components": {"missing_%": round(miss * 100, 2), "duplicates_%": round(dup * 100, 2),
                           "avg_outlier_%": round(out_rate * 100, 2), "constant_cols_%": round(const * 100, 2)}}


def all_findings(df: pd.DataFrame, target: str | None = None) -> list[dict]:
    fs = []
    for fn in (step_shape, step_schema, step_coercion, step_missing, step_duplicates, step_descriptive,
               step_cardinality, step_distribution, step_outliers):
        try:
            fs += fn(df)["findings"]
        except Exception as e:  # never let one step break the summary
            fs.append(F("warn", f"{fn.__name__} skipped: {e}"))
    try:
        fs += step_correlation(df)["findings"]
    except Exception:
        pass
    if target:
        try:
            fs += step_target(df, target)["findings"]
        except Exception:
            pass
    return fs


# ---------------------------------------------------------------- cleaning
def apply_cleaning(df: pd.DataFrame, drop_dups=False, impute=False, drop_high_missing=False, clip=False) -> tuple[pd.DataFrame, list[str]]:
    out, log = df.copy(), []
    if drop_dups:
        n = int(out.duplicated().sum())
        out = out.drop_duplicates().reset_index(drop=True)
        log.append(f"Dropped {n} duplicate rows.")
    if drop_high_missing:
        cols = [c for c in out.columns if out[c].isna().mean() > .6]
        out = out.drop(columns=cols)
        if cols:
            log.append(f"Dropped columns with >60% missing: {', '.join(cols)}.")
    if impute:
        for c in out.columns[out.isna().any()]:
            s = out[c]
            if pd.api.types.is_numeric_dtype(s):
                v = s.median() if abs(s.skew()) > 1 else s.mean()
            else:
                v = s.mode().iloc[0] if not s.mode().empty else "missing"
            out[c] = s.fillna(v)
            log.append(f"Imputed `{c}` with {v if not isinstance(v, float) else round(v, 3)}.")
    if clip:
        cols = [c for c in numeric_cols(out) if out[c].nunique() > 10]
        out = clip_outliers(out, cols)
        log.append(f"Clipped {len(cols)} numeric columns to the 1st-99th percentile.")
    return out, log
