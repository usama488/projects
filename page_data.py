import pandas as pd
import streamlit as st

import data, eda
from common import c_findings, c_quality, download_df, findings, get_df, tip


def load_into_state(df: pd.DataFrame, source: str, target: str | None):
    st.session_state.update(df_raw=df, df=df.copy(), source=source, target=target, clean_log=[],
                            ml={}, dl_runs=[], cluster={}, filters={})


def render():
    st.title("🏠 Data")
    st.caption("Load a demo dataset or upload your own tabular file. Everything else in the app works on this data.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Demo datasets")
        name = st.selectbox("Pick a demo dataset", list(data.DEMO_DATASETS),
                            format_func=lambda n: f"{n} — {data.DEMO_DATASETS[n][2]}")
        if st.button("Load demo", type="primary"):
            with st.spinner(f"Loading {name} ..."):
                try:
                    df = data.load_demo(name)
                except Exception as e:
                    st.error(f"Could not load {name}: {e}. Seaborn datasets need internet the first time.")
                    st.stop()
            load_into_state(df, f"demo: {name}", data.DEMO_DATASETS[name][1])
            st.rerun()
    with c2:
        st.subheader("Upload your data")
        up = st.file_uploader("CSV, TSV, Excel, JSON or Parquet", type=["csv", "tsv", "txt", "xlsx", "xls", "json", "parquet"])
        if up is not None and st.button("Load upload", type="primary"):
            try:
                df = data.read_uploaded(up.name, up.getvalue())
            except Exception as e:
                st.error(f"Could not read the file: {e}")
                st.stop()
            load_into_state(df, f"upload: {up.name}", None)
            st.rerun()

    df = get_df()
    if df is None:
        st.info("No data loaded yet.")
        return
    st.divider()
    st.subheader(f"📄 {st.session_state.source}")
    q = c_quality(df)
    m = st.columns(7)
    m[0].metric("Rows", f"{len(df):,}")
    m[1].metric("Columns", df.shape[1])
    m[2].metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    m[3].metric("Duplicates", int(df.duplicated().sum()))
    m[4].metric("Numeric", len(data.numeric_cols(df)))
    m[5].metric("Categorical", len(data.categorical_cols(df)))
    m[6].metric("Data quality", q["grade"], help=f"Score {q['score']}/100 — {q['components']}")

    cols = [c for c in df.columns]
    idx = cols.index(st.session_state.target) + 1 if st.session_state.target in cols else 0
    tgt = st.selectbox("🎯 Target column (what you want to predict — optional, used by EDA step 14 and ML)",
                       [None] + cols, index=idx)
    st.session_state.target = tgt

    with st.expander("⚠️ Automatic findings (rule-based, no AI key needed)", expanded=True):
        fs = c_findings(df, tgt)
        warn = [f for f in fs if f["level"] == "warn"]
        findings(warn[:8] or [{"level": "good", "text": "No warnings - the data looks clean."}])
        if len(warn) > 8:
            st.caption(f"+{len(warn) - 8} more warnings in the 15-step EDA.")

    st.subheader("Data viewer")
    show_cols = st.multiselect("Columns to show", cols, default=cols[:20])
    st.dataframe(df[show_cols] if show_cols else df, height=320)
    download_df(df, "data_current", key="dl_cur")

    st.subheader("🧹 Cleaning proposals")
    tip("Imputation", "Outlier")
    c = st.columns(4)
    dd = c[0].checkbox("Drop duplicate rows")
    im = c[1].checkbox("Impute missing values (median/mean/mode)")
    hm = c[2].checkbox("Drop columns > 60% missing")
    cl = c[3].checkbox("Clip outliers to 1st–99th percentile")
    b1, b2 = st.columns([1, 5])
    if b1.button("Apply cleaning"):
        new, log = eda.apply_cleaning(st.session_state.df, dd, im, hm, cl)
        st.session_state.df = new
        st.session_state.clean_log += log or ["Nothing to change."]
        st.session_state.ml, st.session_state.dl_runs = {}, []
        st.rerun()
    if b2.button("↩️ Reset to original data"):
        st.session_state.df = st.session_state.df_raw.copy()
        st.session_state.clean_log = []
        st.rerun()
    for line in st.session_state.clean_log:
        st.write("✔️", line)
    if st.session_state.clean_log:
        download_df(st.session_state.df, "data_cleaned", "⬇️ Download cleaned CSV", key="dl_clean")
