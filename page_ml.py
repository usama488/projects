import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import data, ml
from common import download_df, findings, helptext, need_data, show, tip


def _prepare_section(df):
    S = st.session_state.ml
    st.subheader("1 · Problem setup")
    cols = list(df.columns)
    tgt = st.session_state.target
    c = st.columns([2, 1, 1, 1])
    target = c[0].selectbox("🎯 Target", cols, index=cols.index(tgt) if tgt in cols else len(cols) - 1)
    st.session_state.target = target
    task = ml.detect_task(df[target])
    c[1].metric("Detected task", task)
    c[2].metric("Rows with target", int(df[target].notna().sum()))
    c[3].metric("Classes" if task != "regression" else "Target std",
                df[target].nunique() if task != "regression" else f"{df[target].std():.3g}")
    tip("Supervised learning")

    candidates = [x for x in cols if x != target]
    leak = ml.leakage_warnings(df, target, candidates)
    bad = {x for x in candidates if data.column_kind(df[x]) in ("id", "text")}
    bad |= {w.split("`")[1] for w in leak if "`" in w}
    if leak:
        tip("Data leakage")
        for w in leak:
            st.warning(w, icon="🚨")
    feats = st.multiselect("Features (suspicious columns are unticked automatically)", candidates,
                           default=[x for x in candidates if x not in bad])

    st.subheader("2 · Preprocessing & validation")
    c = st.columns(5)
    test_size = c[0].slider("Test size", .1, .4, .2, .05, help=helptext("Train / validation / test split"))
    cv_k = c[1].slider("CV folds (k)", 3, 10, 5, help=helptext("Cross-validation"))
    scaler = c[2].selectbox("Scaler", list(ml.SCALERS), help=helptext("Scaling"))
    encoder = c[3].selectbox("Encoder", ["One-Hot", "Ordinal"], help=helptext("Encoding"))
    imb_opts = ["None", "class_weight"] + (["SMOTE", "Undersample"] if ml.HAS_IMB else [])
    imbalance = c[4].selectbox("Imbalance handling", imb_opts if task != "regression" else ["None"], help=helptext("Class imbalance"))
    tip("Train / validation / test split", "Cross-validation", "Scaling", "Encoding")
    if not feats:
        st.warning("Select at least one feature.")
        st.stop()
    sig = (st.session_state.source, len(df), target, tuple(feats), test_size)
    if S.get("sig") != sig:
        S.clear()
        S["sig"] = sig
        S["p"] = ml.prepare(df, target, feats, test_size)
    S["cfg"] = dict(scaler=scaler, encoder=encoder, imbalance=imbalance, cv_k=cv_k)
    p = S["p"]
    st.caption(f"Train rows: {len(p.X_train)} · Test rows: {len(p.X_test)} · numeric features: {len(p.num_cols)} · "
               f"categorical features: {len(p.cat_cols)} · split is {'stratified' if task != 'regression' else 'random'} with seed {ml.SEED}")
    return p, S


def _eval_block(p, r, key):
    m = pd.DataFrame({"train": r["train"], "test": r["test"]}).T
    st.dataframe(m.round(4))
    c = st.columns(3)
    c[0].metric(f"CV {r['cv_metric']} (mean)", f"{r['cv_mean']:.4f}", help="Average validation score across the k folds")
    c[1].metric("CV std", f"{r['cv_std']:.4f}", help="Spread across folds — high = unstable model")
    c[2].metric("Fit time", f"{r['fit_seconds']:.2f}s")
    verdict, why = ml.diagnose_fit(r["cv_train_mean"], r["cv_mean"], r["cv_metric"])
    (st.success if verdict == "Good fit" else st.warning)(f"**{verdict}** — {why}")
    tip("Overfitting", "Underfitting")
    for t in ml.explain_metrics(p.task, r["test"], p.classes):
        st.info(t, icon="💡")
    if p.task == "regression":
        f1, f2, f3 = ml.regression_figs(p, r["pred"])
        a, b = st.columns(2)
        with a:
            show(f1, key=f"{key}rf1")
        with b:
            show(f2, key=f"{key}rf2")
        show(f3, key=f"{key}rf3")
        tip("R2", "MAE / RMSE", "Residual")
        return
    tip("Confusion matrix", "Precision", "Recall", "F1 score", "ROC-AUC")
    a, b = st.columns(2)
    with a:
        norm = st.toggle("Normalise confusion matrix", key=f"{key}norm")
        show(ml.confusion_fig(p, r["pred"], norm), key=f"{key}cm")
    if r["proba"] is not None:
        roc, pr = ml.roc_pr_figs(p, r["proba"])
        with b:
            show(roc, key=f"{key}roc")
        show(pr, key=f"{key}pr")
        if p.task == "binary":
            st.markdown("**Decision threshold**")
            tip("Decision threshold", "Calibration")
            t = st.slider("Threshold", .05, .95, .5, .05, key=f"{key}thr")
            mt, _ = ml.threshold_table(p, r["proba"], t)
            st.dataframe(pd.DataFrame([mt]).round(4), hide_index=True)
            a, b = st.columns(2)
            with a:
                show(ml.threshold_curve_fig(p, r["proba"]), key=f"{key}tc")
            with b:
                show(ml.calibration_fig(p, r["proba"]), key=f"{key}cal")
    rep = pd.DataFrame(__import__("sklearn.metrics", fromlist=["x"]).classification_report(
        p.y_test, r["pred"], target_names=[str(c) for c in p.classes], output_dict=True, zero_division=0)).T
    with st.expander("Per-class report"):
        st.dataframe(rep.round(3))


def render():
    df = need_data()
    st.title("🤖 Machine Learning")
    tabs = st.tabs(["🎯 Supervised", "🧩 Clustering & PCA (unsupervised)"])
    with tabs[1]:
        clustering(df)
    with tabs[0]:
        supervised(df)


def supervised(df):
    p, S = _prepare_section(df)
    cfg = S["cfg"]
    st.subheader("3 · Models")
    t = st.tabs(["🏁 AutoML leaderboard", "🎯 Train & evaluate", "🔧 Tune", "📈 Learning curves", "🔎 Explain", "🔮 Predict & export"])
    catalog = ml.model_catalog(p.task)

    with t[0]:
        st.caption("Trains every suitable model with the same preprocessing and k-fold CV, then ranks them.")
        skip = st.multiselect("Skip slow models", list(catalog), default=[m for m in ("SVM", "SVR") if m in catalog] if len(p.X_train) > 3000 else [])
        if st.button("🚀 Run AutoML", type="primary"):
            bar = st.progress(0.0, "starting")
            lb, res = ml.automl(p, cfg["scaler"], cfg["encoder"], cfg["imbalance"], cfg["cv_k"], skip,
                                progress=lambda f, m: bar.progress(f, m))
            S["leaderboard"], S["results"] = lb, res
            if res:
                S["current"] = res[lb.model.iloc[0]]
        if "leaderboard" in S:
            lb = S["leaderboard"]
            st.dataframe(lb, hide_index=True)
            key = [c for c in lb.columns if c.startswith("CV ") and c != "CV std"][0]
            show(px.bar(lb, x="model", y=key, error_y="CV std", color=key, title="Cross-validated score by model"), key="lbfig")
            best = lb.iloc[0]
            st.success(f"🏆 Best by {key}: **{best.model}** ({best[key]:.4f}). It's now the current model for the other tabs.")

    with t[1]:
        c = st.columns([2, 3])
        name = c[0].selectbox("Model", list(catalog), key="ml_model")
        params_txt = c[1].text_input("Optional hyperparameters as JSON", "{}", key="ml_params",
                                     help='e.g. {"max_depth": 5} - keys without the model__ prefix')
        if st.button("Train", type="primary", key="ml_train"):
            try:
                params = json.loads(params_txt or "{}")
            except json.JSONDecodeError:
                st.error("Hyperparameters must be valid JSON.")
                st.stop()
            with st.spinner(f"Training {name} with {cfg['cv_k']}-fold CV ..."):
                S["current"] = ml.train_eval(p, name, cfg["scaler"], cfg["encoder"], cfg["imbalance"], cfg["cv_k"], params)
        if "current" in S:
            r = S["current"]
            st.markdown(f"### Current model: {r['name']}")
            _eval_block(p, r, "cur")

    with t[2]:
        tip("Hyperparameter", "Grid / Random search")
        name = st.selectbox("Model to tune", list(catalog), key="tune_model")
        grid = st.text_area("Parameter grid (JSON, keys use the model__ prefix)", json.dumps(catalog[name][1], indent=1),
                            height=140, key=f"grid_{name}")
        c = st.columns(3)
        method = c[0].radio("Search", ["Grid", "Random"], horizontal=True)
        n_iter = c[1].number_input("Random iterations", 2, 200, 15, disabled=method == "Grid")
        if c[2].button("🔧 Run search", type="primary"):
            try:
                g = json.loads(grid)
            except json.JSONDecodeError:
                st.error("Invalid JSON grid.")
                st.stop()
            n = int(np.prod([len(v) for v in g.values()])) if g else 1
            with st.spinner(f"Searching {n if method == 'Grid' else n_iter} combinations x {cfg['cv_k']} folds ..."):
                S["tune"] = ml.tune(p, name, g, method, int(n_iter), cfg["cv_k"], cfg["scaler"], cfg["encoder"], cfg["imbalance"])
                S["tune"]["model"] = name
        if "tune" in S:
            tr = S["tune"]
            st.success(f"Best CV {tr['metric']} = **{tr['best_score']:.4f}** with {tr['best_params']} ({tr['seconds']:.1f}s)")
            st.dataframe(tr["results"].round(4), hide_index=True)
            if st.button("Use best params as the current model"):
                S["current"] = ml.train_eval(p, tr["model"], cfg["scaler"], cfg["encoder"], cfg["imbalance"], cfg["cv_k"], tr["best_params"])
                S["current"]["name"] += " (tuned)"
                st.success("Done — see 🎯 Train & evaluate.")

    cur = S.get("current")
    with t[3]:
        tip("Learning curve", "Bias-variance trade-off", "Validation curve")
        if not cur:
            st.info("Train a model first.")
        else:
            if st.button("Compute learning curve", key="lc"):
                with st.spinner("Training on growing subsets ..."):
                    S["lc"] = ml.learning_curve_fig(p, cur["pipe"], cfg["cv_k"])
            if "lc" in S:
                fig, (verdict, why) = S["lc"]
                show(fig, key="lcfig")
                (st.success if verdict == "Good fit" else st.warning)(f"**{verdict}** — {why}")
            st.markdown("**Validation curve** — sweep one hyperparameter")
            grid = catalog[cur["name"].replace(" (tuned)", "")][1]
            if grid:
                prm = st.selectbox("Parameter", list(grid), key="vcp")
                vals = [v for v in grid[prm]]
                if st.button("Compute validation curve", key="vc"):
                    with st.spinner("Sweeping ..."):
                        fig, best = ml.validation_curve_fig(p, cur["pipe"], prm, vals, cfg["cv_k"])
                    show(fig, key="vcfig")
                    st.info(f"Best validation score at {prm.replace('model__', '')} = {best}")
            else:
                st.caption("This model has no tunable parameters listed.")

    with t[4]:
        tip("Feature importance", "Partial dependence", "SHAP")
        if not cur:
            st.info("Train a model first.")
        else:
            imp = ml.builtin_importance(cur["pipe"])
            if imp is not None:
                fig = px.bar(imp.head(20), x="importance", y="feature", orientation="h",
                             title=f"Built-in importance — {cur['name']}")
                fig.update_yaxes(autorange="reversed")
                show(fig, key="imp")
            else:
                st.caption("This model has no built-in importance — use permutation importance.")
            if st.button("Compute permutation importance", key="perm"):
                with st.spinner("Shuffling each column ..."):
                    S["perm"] = ml.permutation_fig(p, cur["pipe"])
            if "perm" in S:
                show(S["perm"][0], key="permfig")
                top = S["perm"][1].iloc[0]
                st.info(f"Shuffling `{top.feature}` hurts the score most ({top.importance:.4f}) — it's the model's most relied-on column.", icon="💡")
            if p.num_cols:
                f = st.selectbox("Partial dependence for", p.num_cols, key="pdpf")
                show(ml.pdp_fig(p, cur["pipe"], f), key="pdp")
            if ml.HAS_SHAP and st.button("Compute SHAP (slow)", key="shap"):
                with st.spinner("SHAP ..."):
                    show(ml.shap_fig(p, cur["pipe"]), key="shapfig")
            elif not ml.HAS_SHAP:
                st.caption("Install `shap` to enable SHAP explanations.")

    with t[5]:
        if not cur:
            st.info("Train a model first.")
        else:
            st.markdown(f"**Single prediction with {cur['name']}**")
            src = p.X_train
            vals = {}
            cols = st.columns(3)
            for i, f in enumerate(p.features):
                with cols[i % 3]:
                    if f in p.num_cols:
                        vals[f] = st.number_input(f, value=float(np.nanmedian(src[f])), key=f"pred_{f}")
                    else:
                        opts = sorted(src[f].dropna().astype(str).unique().tolist())
                        mode = src[f].mode()
                        vals[f] = st.selectbox(f, opts, index=opts.index(str(mode.iloc[0])) if len(mode) else 0, key=f"pred_{f}")
            row = pd.DataFrame([vals])
            pred = cur["pipe"].predict(row)[0]
            if p.task == "regression":
                st.success(f"Predicted **{p.target}** = **{pred:.4f}**")
            else:
                proba = cur["pipe"].predict_proba(row)[0]
                st.success(f"Predicted **{p.target}** = **{p.classes[int(pred)]}**")
                show(px.bar(x=[str(c) for c in p.classes], y=proba, labels={"x": "class", "y": "probability"},
                            title="Class probabilities"), key="predprob")
            st.markdown("**Batch prediction** — upload a file with the same feature columns")
            up = st.file_uploader("File to score", type=["csv", "xlsx", "json", "parquet"], key="batch")
            if up:
                new = data.read_uploaded(up.name, up.getvalue())
                try:
                    X = p.featurize(new)
                except KeyError as e:
                    st.error(str(e))
                else:
                    out = new.copy()
                    pr = cur["pipe"].predict(X)
                    out[f"predicted_{p.target}"] = pr if p.task == "regression" else [p.classes[int(i)] for i in pr]
                    st.dataframe(out.head(50))
                    download_df(out, "predictions", key="dlpred")
            st.markdown("**Export**")
            c = st.columns(2)
            c[0].download_button("⬇️ Trained pipeline (.joblib)", ml.model_bytes(cur["pipe"], p), "model.joblib")
            c[1].download_button("⬇️ Model card (.md)", ml.model_card(cur, p, st.session_state.source), "model_card.md")


def clustering(df):
    st.caption("No target needed. Finds natural groups and projects the data to 2D.")
    tip("Unsupervised learning", "K-Means", "Elbow method", "Silhouette score", "DBSCAN", "PCA", "t-SNE")
    S = st.session_state.cluster
    cands = [c for c in df.columns if data.column_kind(df[c]) not in ("id", "text", "datetime")]
    default = [c for c in data.numeric_cols(df) if c != st.session_state.target][:8]
    feats = st.multiselect("Features to cluster on", cands, default=default, key="cl_feats")
    if len(feats) < 2:
        st.info("Pick at least two features.")
        return
    d = df[feats].dropna(how="all")
    if len(d) > 20000:
        d = d.sample(20000, random_state=ml.SEED)
        st.caption("Using a 20,000-row sample.")
    X, names = ml.unsupervised_matrix(d, feats)
    a, b = st.columns(2)
    with a:
        fig, n95 = ml.pca_variance_fig(X)
        show(fig, key="pcav")
        st.caption(f"{n95} principal components keep 95% of the variance (from {X.shape[1]} encoded features).")
    with b:
        if st.button("Find best k (elbow + silhouette)"):
            with st.spinner("Trying k = 2..10"):
                S["elbow"] = ml.elbow_silhouette(X, 10)
        if "elbow" in S:
            show(S["elbow"][0], key="elbow")
            st.info(f"Highest silhouette at k = {S['elbow'][1]}.", icon="💡")
    c = st.columns(4)
    algo = c[0].selectbox("Algorithm", ["KMeans", "Agglomerative", "DBSCAN"])
    k = c[1].slider("k (clusters)", 2, 12, S.get("elbow", (None, 3))[1])
    eps = c[2].slider("DBSCAN eps", .1, 5.0, .8, .1, disabled=algo != "DBSCAN")
    proj = c[3].selectbox("2D projection", ["PCA", "t-SNE"])
    labels, stats_ = ml.run_clustering(X, algo, k, eps)
    m = st.columns(4)
    m[0].metric("Clusters", stats_["clusters"])
    m[1].metric("Noise points", stats_["noise_points"])
    m[2].metric("Silhouette", f"{stats_['silhouette']:.3f}", help=helptext("Silhouette score"))
    m[3].metric("Davies-Bouldin", f"{stats_['davies_bouldin']:.3f}", help="Lower is better")
    show(ml.projection_fig(X, labels, proj, d), key="proj")
    prof = ml.cluster_profile(d, labels, feats)
    st.markdown("**Cluster profiles (mean of each numeric feature)**")
    st.dataframe(prof)
    big = prof.drop(columns="n_rows")
    if not big.empty and len(big) > 1:
        z = (big - big.mean()) / big.std().replace(0, 1)
        for cl, row in z.iterrows():
            hi, lo = row.idxmax(), row.idxmin()
            st.write(f"• Cluster **{cl}** ({prof.loc[cl, 'n_rows']} rows): highest relative `{hi}`, lowest relative `{lo}`.")
    if algo == "Agglomerative":
        show(ml.dendrogram_fig(X), key="dendro")
    tgt = st.session_state.target
    if tgt in df.columns and df[tgt].nunique() <= 20:
        ct = pd.crosstab(pd.Series(labels, index=d.index, name="cluster"), df.loc[d.index, tgt])
        show(px.imshow(ct, text_auto=True, title=f"Clusters vs `{tgt}` (did clustering rediscover the labels?)",
                       color_continuous_scale="Blues"), key="clvt")
    download_df(d.assign(cluster=labels), "clustered_data", key="dlcl")
