"""Auto EDA + ML Studio — Streamlit entry point.

Run:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Auto EDA + ML Studio", page_icon="📊", layout="wide")

import data, dl, ml  # noqa: E402
import page_data, page_dims, page_dl, page_eda, page_export, page_learn, page_ml  # noqa: E402
from common import init_state  # noqa: E402

PAGES = {
    "🏠 Data": page_data.render,
    "🔍 15-Step EDA": page_eda.render,
    "📊 4 Dimensions": page_dims.render,
    "🤖 Machine Learning": page_ml.render,
    "🧠 Deep Learning": page_dl.render,
    "📚 Learn & Glossary": page_learn.render,
    "📤 Export": page_export.render,
}


def sidebar_filters():
    df = st.session_state.df
    if df is None:
        return
    st.session_state.setdefault("filters", {})
    with st.sidebar.expander("🔎 Filters (apply to every page)"):
        cols = st.multiselect("Filter on columns", [c for c in df.columns if data.column_kind(df[c]) not in ("id", "text")],
                              default=list(st.session_state.filters))
        new = {}
        for c in cols:
            s = df[c]
            if pd.api.types.is_numeric_dtype(s) and s.nunique() > 15:
                lo, hi = float(s.min()), float(s.max())
                new[c] = st.slider(c, lo, hi, st.session_state.filters.get(c, (lo, hi)), key=f"f_{c}")
            elif pd.api.types.is_datetime64_any_dtype(s):
                continue
            else:
                opts = sorted(s.dropna().astype(str).unique())[:200]
                new[c] = st.multiselect(c, opts, default=st.session_state.filters.get(c, []), key=f"f_{c}")
        if new != st.session_state.filters:
            st.session_state.filters = new
            st.session_state.ml = {}
        if new:
            st.caption("Filtered rows are used everywhere. Clear the filters to use all data.")


def main():
    init_state()
    st.sidebar.title("📊 Auto EDA + ML Studio")
    page = st.sidebar.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.sidebar.toggle("🎓 Learn mode (explain every concept)", key="learn")
    sidebar_filters()
    if st.session_state.df is not None:
        st.sidebar.success(f"{st.session_state.source}\n\n{len(st.session_state.df):,} rows × {st.session_state.df.shape[1]} cols")
    with st.sidebar.expander("Environment"):
        st.caption(f"Deep learning: {dl.backend()}")
        st.caption(f"XGBoost: {'✅' if ml.HAS_XGB else '—'} · LightGBM: {'✅' if ml.HAS_LGBM else '—'} · "
                   f"SMOTE: {'✅' if ml.HAS_IMB else '—'} · SHAP: {'✅' if ml.HAS_SHAP else '—'}")
        st.caption("Dark / light theme: ⋮ menu (top-right) → Settings → Theme.")
    PAGES[page]()


main()
