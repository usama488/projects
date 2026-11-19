import pandas as pd
import streamlit as st

import eda, report
import common as C
from common import need_data


def render():
    df = need_data()
    st.title("📤 Export")
    st.caption("Download the EDA (and ML results, if you trained models) as a standalone HTML or Markdown report, plus the data.")
    tgt = st.session_state.target
    with st.spinner("Collecting findings ..."):
        q = C.c_quality(df)
        fs = C.c_findings(df, tgt)
        tables = {"Schema": C.c_schema(df)["table"].set_index("column"),
                  "Descriptive statistics": C.c_desc(df)["table"],
                  "Missing values": C.c_missing(df)["table"],
                  "Distribution & normality": C.c_dist(df)["table"].set_index("column") if not C.c_dist(df)["table"].empty else pd.DataFrame(),
                  "Outliers": C.c_out(df)["table"].set_index("column") if not C.c_out(df)["table"].empty else pd.DataFrame()}
        corr = C.c_corr(df, "pearson")
        figs = [f for f in [C.c_shape(df)["fig"], C.c_desc(df)["fig"], corr["fig"]] if f is not None]
        if tgt:
            t = C.c_target(df, tgt)
            figs += [t["balance_fig"], t["fig"]]
        include_dims = st.checkbox("Include the 4-dimension charts (bigger file)", True)
        if include_dims:
            for v in C.c_dims(df).values():
                figs += v[:2]
    mls = {}
    S = st.session_state.ml
    if "leaderboard" in S:
        mls["AutoML leaderboard"] = S["leaderboard"]
    if "current" in S:
        r = S["current"]
        mls[f"Current model: {r['name']}"] = pd.DataFrame([r["test"]]).round(4)
    if st.session_state.dl_runs:
        r = st.session_state.dl_runs[-1]
        mls["Neural network (last run)"] = pd.DataFrame([{**r["test"], "epochs_run": r["last_epoch"], "best_epoch": r["best_epoch"]}]).round(4)
    title = st.text_input("Report title", f"EDA report — {st.session_state.source}")
    html_doc = report.build_html(title, st.session_state.source, q, fs, tables, figs, mls or None)
    md_doc = report.build_markdown(title, st.session_state.source, q, fs, tables, mls or None)
    c = st.columns(3)
    c[0].download_button("⬇️ HTML report", html_doc.encode(), "eda_report.html", "text/html", type="primary")
    c[1].download_button("⬇️ Markdown report", md_doc.encode(), "eda_report.md", "text/markdown")
    c[2].download_button("⬇️ Current data (CSV)", df.to_csv(index=False).encode(), "data.csv", "text/csv")
    st.markdown(f"**Preview** — grade **{q['grade']}**, {len(fs)} findings, {len(figs)} charts, {len(mls)} ML sections")
    with st.expander("Markdown preview"):
        st.markdown(md_doc[:6000] + ("\n\n…" if len(md_doc) > 6000 else ""))
