"""Shared UI helpers: state, chart display, findings, learn-mode tips, cached computations."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import eda
from glossary import GLOSSARY

ICON = {"warn": "⚠️", "good": "✅", "info": "💡"}


def init_state():
    defaults = {"df_raw": None, "df": None, "source": None, "target": None, "clean_log": [],
                "learn": True, "ml": {}, "dl_runs": [], "cluster": {}}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def get_df() -> pd.DataFrame | None:
    """The working dataframe after cleaning AND the sidebar filters."""
    df = st.session_state.get("df")
    if df is None:
        return None
    filt = st.session_state.get("filters", {})
    for col, val in filt.items():
        if col not in df.columns:
            continue
        if isinstance(val, tuple):
            df = df[df[col].between(*val) | df[col].isna()]
        elif val:
            df = df[df[col].astype(str).isin(val)]
    return df


def show(fig, key: str | None = None, **kw):
    if fig is None:
        return None
    return st.plotly_chart(fig, key=key, **kw)


def findings(fs: list[dict], title: str | None = None):
    if not fs:
        return
    if title:
        st.markdown(f"**{title}**")
    for f in fs:
        fn = {"warn": st.warning, "good": st.success, "info": st.info}[f["level"]]
        fn(f["text"], icon=ICON[f["level"]])


def tip(*terms: str):
    """In Learn mode, show short explanations for the given glossary terms."""
    if not st.session_state.get("learn"):
        return
    for t in terms:
        if t in GLOSSARY:
            exp, ex = GLOSSARY[t]
            st.caption(f"🎓 **{t}** — {exp}" + (f" *Example: {ex}*" if ex else ""))


def helptext(term: str) -> str | None:
    return GLOSSARY[term][0] if term in GLOSSARY else None


def need_data() -> pd.DataFrame | None:
    df = get_df()
    if df is None:
        st.info("👈 Load a dataset first on the **🏠 Data** page (pick a demo or upload your own file).")
        st.stop()
    if df.empty:
        st.warning("The sidebar filters removed every row. Loosen them to continue.")
        st.stop()
    return df


def download_df(df: pd.DataFrame, name: str, label: str = "⬇️ Download CSV", key: str | None = None):
    st.download_button(label, df.to_csv(index=True).encode(), file_name=f"{name}.csv", mime="text/csv", key=key)


# --------------------------------------------------------------- cached EDA steps
@st.cache_data(show_spinner=False)
def c_shape(df): return eda.step_shape(df)
@st.cache_data(show_spinner=False)
def c_schema(df): return eda.step_schema(df)
@st.cache_data(show_spinner=False)
def c_coercion(df): return eda.step_coercion(df)
@st.cache_data(show_spinner=False)
def c_missing(df): return eda.step_missing(df)
@st.cache_data(show_spinner=False)
def c_dups(df): return eda.step_duplicates(df)
@st.cache_data(show_spinner=False)
def c_desc(df): return eda.step_descriptive(df)
@st.cache_data(show_spinner=False)
def c_card(df): return eda.step_cardinality(df)
@st.cache_data(show_spinner=False)
def c_dist(df): return eda.step_distribution(df)
@st.cache_data(show_spinner=False)
def c_out(df): return eda.step_outliers(df)
@st.cache_data(show_spinner=False)
def c_corr(df, method): return eda.step_correlation(df, method)
@st.cache_data(show_spinner=False)
def c_target(df, target): return eda.step_target(df, target)
@st.cache_data(show_spinner=False)
def c_quality(df): return eda.quality_score(df)
@st.cache_data(show_spinner=False)
def c_findings(df, target): return eda.all_findings(df, target)
@st.cache_data(show_spinner=False)
def c_dims(df): return eda.four_dimensions(df)
