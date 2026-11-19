"""Data loading, demo datasets and column typing."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent / "demo_cache"

# name -> (source, suggested target, description)
DEMO_DATASETS = {
    "titanic": ("seaborn", "survived", "Titanic passengers - binary classification (survived)"),
    "tips": ("seaborn", "tip", "Restaurant bills & tips - regression"),
    "iris": ("seaborn", "species", "Iris flowers - multiclass classification"),
    "penguins": ("seaborn", "species", "Palmer penguins - multiclass / clustering"),
    "diamonds": ("seaborn", "price", "Diamond prices - regression (53k rows)"),
    "mpg": ("seaborn", "mpg", "Car fuel efficiency - regression"),
    "taxis": ("seaborn", "total", "NYC taxi trips - regression with datetimes"),
    "breast_cancer": ("sklearn", "target", "Breast cancer diagnosis - binary classification"),
    "wine": ("sklearn", "target", "Wine cultivars - multiclass classification"),
    "california_housing": ("sklearn", "MedHouseVal", "California house values - regression"),
}


def _load_sklearn(name: str) -> pd.DataFrame:
    from sklearn import datasets

    if name == "breast_cancer":
        b = datasets.load_breast_cancer(as_frame=True)
        df = b.frame
        df["target"] = df["target"].map(dict(enumerate(b.target_names)))
        return df
    if name == "wine":
        b = datasets.load_wine(as_frame=True)
        df = b.frame
        df["target"] = df["target"].map(lambda i: f"class_{i}")
        return df
    if name == "california_housing":
        try:
            return datasets.fetch_california_housing(as_frame=True).frame
        except Exception:
            # Offline fallback: synthetic data with the same shape of problem
            rng = np.random.default_rng(0)
            n = 3000
            df = pd.DataFrame({
                "MedInc": rng.gamma(4, 1, n), "HouseAge": rng.integers(1, 52, n),
                "AveRooms": rng.normal(5, 1, n), "AveBedrms": rng.normal(1, .1, n),
                "Population": rng.gamma(3, 400, n), "AveOccup": rng.normal(3, .5, n),
                "Latitude": rng.uniform(32, 42, n), "Longitude": rng.uniform(-124, -114, n),
            })
            df["MedHouseVal"] = (0.4 * df.MedInc + 0.01 * df.HouseAge + rng.normal(0, .5, n)).clip(.15, 5)
            return df
    raise KeyError(name)


def load_demo(name: str) -> pd.DataFrame:
    """Load a demo dataset, caching it locally after the first download."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / f"{name}.pkl"
    if cached.exists():
        return pd.read_pickle(cached)
    source = DEMO_DATASETS[name][0]
    if source == "seaborn":
        import seaborn as sns
        df = sns.load_dataset(name)
    else:
        df = _load_sklearn(name)
    # seaborn returns some categoricals; plain object dtype is simpler downstream
    for c in df.columns:
        if isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype(object)
    df.to_pickle(cached)
    return df


def read_uploaded(name: str, raw: bytes) -> pd.DataFrame:
    """Read CSV / TSV / Excel / JSON / Parquet from uploaded bytes."""
    ext = name.lower().rsplit(".", 1)[-1]
    buf = io.BytesIO(raw)
    if ext in ("csv", "txt"):
        return pd.read_csv(buf, sep=None, engine="python")
    if ext == "tsv":
        return pd.read_csv(buf, sep="\t")
    if ext in ("xlsx", "xls"):
        return pd.read_excel(buf)
    if ext == "json":
        return pd.read_json(buf)
    if ext == "parquet":
        return pd.read_parquet(buf)
    raise ValueError(f"Unsupported file type: .{ext}")


def column_kind(s: pd.Series) -> str:
    """Classify a column as numeric / categorical / boolean / datetime / text / id."""
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    nun = s.nunique(dropna=True)
    if pd.api.types.is_numeric_dtype(s):
        if nun <= 2:
            return "boolean"
        if pd.api.types.is_integer_dtype(s) and nun <= 15:
            return "discrete"
        return "numeric"
    n = max(len(s.dropna()), 1)
    if nun == n and n > 50:
        return "id"
    avg_len = s.dropna().astype(str).str.len().mean() if n else 0
    if nun > 50 and avg_len > 30:
        return "text"
    return "categorical"


def column_kinds(df: pd.DataFrame) -> dict[str, str]:
    return {c: column_kind(df[c]) for c in df.columns}


def numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


def categorical_cols(df: pd.DataFrame, max_levels: int = 50) -> list[str]:
    kinds = column_kinds(df)
    return [c for c, k in kinds.items() if k in ("categorical", "boolean", "discrete") and df[c].nunique() <= max_levels]


def sample_for_plot(df: pd.DataFrame, n: int = 5000, seed: int = 42) -> tuple[pd.DataFrame, bool]:
    if len(df) > n:
        return df.sample(n, random_state=seed), True
    return df, False
