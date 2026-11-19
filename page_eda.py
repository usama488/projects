import pandas as pd
import streamlit as st

import data, eda
import common as C
from common import download_df, findings, need_data, show, tip


def render():
    df = need_data()
    st.title("🔍 15-Step EDA")
    all_on = st.toggle("Show all 15 steps at once", value=False)
    if all_on:
        steps = eda.STEPS
    else:
        steps = [st.selectbox("Step", eda.STEPS)]
    for s in steps:
        st.header(s)
        n = int(s.split(".")[0])
        try:
            RENDER[n](df)
        except Exception as e:  # keep the rest of the page alive
            st.error(f"This step failed on this dataset: {e}")
        if all_on:
            st.divider()


def s1(df):
    r = eda.step_ingest(df, st.session_state.source)
    findings(r["findings"])
    view = st.radio("Preview", ["First rows", "Last rows", "Random sample"], horizontal=True, key="s1v")
    n = st.slider("Rows to preview", 5, 100, 10, key="s1n")
    st.dataframe(df.head(n) if view == "First rows" else df.tail(n) if view == "Last rows" else df.sample(min(n, len(df)), random_state=1))


def s2(df):
    r = C.c_shape(df)
    m = st.columns(5)
    m[0].metric("Cells", f"{r['cells']:,}")
    m[1].metric("Populated", f"{r['cells'] - r['missing'] - r['inf']:,}")
    m[2].metric("Missing", f"{r['missing']:,}")
    m[3].metric("In memory", f"{r['memory_mb']:.2f} MB")
    m[4].metric("If optimised", f"{r['optimized_mb']:.2f} MB", f"{r['optimized_mb'] - r['memory_mb']:.2f} MB")
    show(r["fig"], key="s2f")
    findings(r["findings"])


def s3(df):
    r = C.c_schema(df)
    st.dataframe(r["table"], hide_index=True)
    download_df(r["table"], "schema", key="s3d")
    show(r["fig"], key="s3f")
    findings(r["findings"])


def s4(df):
    r = C.c_coercion(df)
    findings(r["findings"])
    if not r["table"].empty:
        st.dataframe(r["table"], hide_index=True)
        c1, c2 = st.columns(2)
        row = c1.selectbox("Apply a suggestion", r["table"].index, format_func=lambda i: f"{r['table'].column[i]} → {r['table'].suggested[i]}")
        if c2.button("Convert column"):
            t = r["table"].loc[row]
            st.session_state.df = eda.apply_coercion(st.session_state.df, t.column, t.suggested)
            st.session_state.clean_log.append(f"Converted `{t.column}` to {t.suggested}.")
            st.rerun()


def s5(df):
    r = C.c_missing(df)
    tip("Imputation")
    findings(r["findings"])
    if not r["table"].empty:
        st.dataframe(r["table"])
        for i, f in enumerate(r["figs"]):
            show(f, key=f"s5f{i}")
        st.caption("Apply imputation on the 🏠 Data page → Cleaning proposals.")


def s6(df):
    r = C.c_dups(df)
    findings(r["findings"])
    if r["count"]:
        with st.expander(f"Inspect duplicate rows (showing up to 50 of the rows involved)"):
            st.dataframe(r["rows"])
        if st.button("Drop duplicates now"):
            st.session_state.df = st.session_state.df.drop_duplicates().reset_index(drop=True)
            st.session_state.clean_log.append(f"Dropped {r['count']} duplicate rows.")
            st.rerun()


def s7(df):
    r = C.c_desc(df)
    tip("Skewness", "Kurtosis")
    if r["table"].empty:
        findings(r["findings"])
        return
    st.dataframe(r["table"])
    download_df(r["table"], "descriptive_stats", key="s7d")
    show(r["fig"], key="s7f")
    findings(r["findings"])


def s8(df):
    r = C.c_card(df)
    if not r["table"].empty:
        st.dataframe(r["table"], hide_index=True)
    findings(r["findings"])
    cats = [c for c in df.columns if df[c].nunique() <= 200]
    if cats:
        col = st.selectbox("Inspect a column's frequencies", cats, key="s8c")
        fig, vc, f = eda.frequency_fig(df, col)
        show(fig, key="s8f")
        findings([f])
        download_df(vc, f"freq_{col}", key="s8d")


def s9(df):
    r = C.c_dist(df)
    tip("Skewness", "Normality test", "p-value")
    if r["table"].empty:
        st.info("No numeric columns to test.")
        return
    st.dataframe(r["table"], hide_index=True)
    col = st.selectbox("Column", r["table"].column, key="s9c")
    h, q = eda.distribution_figs(df, col)
    c1, c2 = st.columns(2)
    with c1:
        show(h, key="s9h")
    with c2:
        show(q, key="s9q")
    st.caption("On the Q-Q plot, points on the red line = normal. Points curving up at the right = right skew / heavy tail.")
    findings(r["findings"])


def s10(df):
    r = C.c_out(df)
    tip("Outlier")
    if r["table"].empty:
        st.info("No continuous numeric columns.")
        return
    st.dataframe(r["table"], hide_index=True)
    col = st.selectbox("View outliers for", r["table"].column, key="s10c")
    show(eda.univariate_fig(df, col, "Box"), key="s10f")
    t = r["table"].set_index("column").loc[col]
    out_rows = df[(df[col] < t.lower_fence) | (df[col] > t.upper_fence)]
    with st.expander(f"{len(out_rows)} outlier rows in `{col}`"):
        st.dataframe(out_rows)
    if st.button(f"Clip `{col}` to 1st–99th percentile"):
        st.session_state.df = eda.clip_outliers(st.session_state.df, [col])
        st.session_state.clean_log.append(f"Clipped `{col}` to the 1st-99th percentile.")
        st.rerun()
    findings(r["findings"])


def s11(df):
    tip("Distribution", "Composition")
    c = st.columns(4)
    col = c[0].selectbox("Column", list(df.columns), key="s11c")
    is_num = col in data.numeric_cols(df) and df[col].nunique() > 15
    kind = c[1].selectbox("Chart", eda.UNI_NUMERIC if is_num else eda.UNI_CATEG, key="s11k")
    groups = [None] + data.categorical_cols(df, 12)
    color = c[2].selectbox("Split by (optional)", groups, key="s11g")
    norm = c[3].selectbox("Histogram scale", [None, "percent", "probability", "density"], key="s11n") if kind == "Histogram" else None
    show(eda.univariate_fig(df, col, kind, color if color != col else None, norm), key="s11f")
    s = df[col]
    if is_num:
        st.write(f"**{col}**: mean {s.mean():.3f}, median {s.median():.3f}, std {s.std():.3f}, skew {s.skew():.2f}, range {s.min():.3g} – {s.max():.3g}")
    else:
        vc = s.value_counts(normalize=True)
        st.write(f"**{col}**: {s.nunique()} levels; top = '{vc.index[0]}' ({vc.iloc[0]:.1%})")


def s12(df):
    tip("Relationship", "Correlation", "p-value")
    c = st.columns(3)
    cols = [x for x in df.columns if data.column_kind(df[x]) not in ("id", "text")]
    nums = data.numeric_cols(df)
    x = c[0].selectbox("X", cols, index=cols.index(nums[0]) if nums else 0, key="s12x")
    y = c[1].selectbox("Y", cols, index=cols.index(nums[1]) if len(nums) > 1 else 0, key="s12y")
    color = c[2].selectbox("Colour by", [None] + data.categorical_cols(df, 12), key="s12c")
    if x == y:
        st.info("Pick two different columns.")
        return
    fig, fs = eda.bivariate(df, x, y, color)
    sel = st.toggle("Enable click / box / lasso selection (cross-filter)", key="s12sel")
    if sel:
        ev = show(fig, key="s12f", on_select="rerun", selection_mode=("box", "lasso", "points"))
        pts = ev.selection.points if ev and ev.selection else []
        if pts:
            rows = [p["customdata"][0] for p in pts if p.get("customdata")]
            sub = df.loc[df.index.intersection(rows)]
            if sub.empty:
                st.info("Selection works on scatter plots (numeric X and Y).")
                return
            st.success(f"Selected {len(sub)} points — summary of the selection vs all data:")
            comp = pd.DataFrame({"selection_mean": sub[nums].mean(), "all_mean": df[nums].mean()})
            comp["diff_%"] = ((comp.selection_mean / comp.all_mean - 1) * 100).round(1)
            st.dataframe(comp)
            st.dataframe(sub, height=200)
    else:
        show(fig, key="s12f")
    findings(fs)


def s13(df):
    tip("Correlation", "Multicollinearity (VIF)")
    method = st.radio("Method", ["pearson", "spearman", "kendall"], horizontal=True, key="s13m")
    r = C.c_corr(df, method)
    if r["fig"] is None:
        findings(r["findings"])
        return
    show(r["fig"], key="s13f")
    c1, c2 = st.columns(2)
    c1.markdown("**Strongest pairs**")
    c1.dataframe(r["pairs"], hide_index=True)
    c2.markdown("**Variance Inflation Factor**")
    c2.dataframe(r["vif"], hide_index=True)
    findings(r["findings"])


def s14(df):
    tip("Mutual information", "Class imbalance")
    tgt = st.session_state.target
    tgt = st.selectbox("Target", list(df.columns), index=list(df.columns).index(tgt) if tgt in df.columns else 0, key="s14t")
    r = C.c_target(df, tgt)
    c1, c2 = st.columns(2)
    with c1:
        show(r["balance_fig"], key="s14b")
    with c2:
        show(r["fig"], key="s14f")
    findings(r["findings"])
    feat = st.selectbox("Compare a feature against the target", [c for c in df.columns if c != tgt], key="s14x")
    fig, fs = eda.bivariate(df, feat, tgt)
    show(fig, key="s14g")
    findings(fs)


def s15(df):
    q = C.c_quality(df)
    c = st.columns(5)
    c[0].metric("Quality grade", q["grade"], help="A ≥ 90, B ≥ 80, C ≥ 70, D ≥ 60, F below")
    c[1].metric("Score", f"{q['score']}/100")
    for i, (k, v) in enumerate(list(q["components"].items())[:3]):
        c[i + 2].metric(k, v)
    fs = C.c_findings(df, st.session_state.target)
    order = {"warn": 0, "info": 1, "good": 2}
    findings(sorted(fs, key=lambda f: order[f["level"]]), "All findings")
    st.caption("Export the full HTML / Markdown report on the 📤 Export page.")


RENDER = {1: s1, 2: s2, 3: s3, 4: s4, 5: s5, 6: s6, 7: s7, 8: s8, 9: s9, 10: s10, 11: s11, 12: s12, 13: s13, 14: s14, 15: s15}
