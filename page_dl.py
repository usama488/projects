import time

import pandas as pd
import plotly.express as px
import streamlit as st

import dl, ml
from common import findings, helptext, need_data, show, tip


def render():
    df = need_data()
    st.title("🧠 Deep Learning — neural network builder")
    st.caption(f"Backend: **{dl.backend()}**. Trains a multi-layer perceptron (MLP) on the same split as the ML page, "
               "epoch by epoch, with live curves.")
    tip("Deep learning", "Neural network (MLP)")
    S = st.session_state.ml
    if "p" not in S:
        st.info("Set the target and features on the 🤖 Machine Learning page first (the same train/test split is reused).")
        return
    p = S["p"]
    st.write(f"Target **{p.target}** · task **{p.task}** · {len(p.features)} features · {len(p.X_train)} training rows")

    # ------------------------------------------------------------ builder
    st.subheader("1 · Architecture")
    c = st.columns(4)
    n_layers = c[0].slider("Hidden layers", 1, 6, 2, help="More layers = can learn more complex patterns, but overfits more easily.")
    hidden = []
    with c[1]:
        for i in range(n_layers):
            hidden.append(st.select_slider(f"Neurons in layer {i + 1}", [4, 8, 16, 32, 64, 128, 256, 512],
                                           value=[128, 64, 32, 16, 8, 8][i], key=f"h{i}"))
    act = c[2].selectbox("Activation", dl.ACTIVATIONS, help=helptext("Activation function"))
    dropout = c[2].slider("Dropout", 0.0, 0.7, 0.2, 0.05, help=helptext("Dropout"))
    bn = c[3].checkbox("Batch normalization", True, help=helptext("Batch normalization"))
    tip("Neuron / unit", "Weights & biases", "Activation function", "Dropout", "Batch normalization")

    st.subheader("2 · Training settings")
    c = st.columns(4)
    opt = c[0].selectbox("Optimizer", dl.OPTIMIZERS, help=helptext("Optimizer"))
    lr = c[0].select_slider("Learning rate", [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0], 1e-3,
                            format_func=lambda v: f"{v:g}", help=helptext("Learning rate"))
    epochs = c[1].number_input("Epochs", 1, 2000, 150, help=helptext("Epoch"))
    batch = c[1].select_slider("Batch size", [8, 16, 32, 64, 128, 256, 512], 32, help=helptext("Batch / batch size"))
    wd = c[2].select_slider("Weight decay (L2)", [0.0, 1e-5, 1e-4, 1e-3, 1e-2], 1e-4, format_func=lambda v: f"{v:g}",
                            help=helptext("Weight decay"))
    sched = c[2].selectbox("LR scheduler", dl.SCHEDULERS, help=helptext("Learning-rate scheduler"))
    es = c[3].checkbox("Early stopping", True, help=helptext("Early stopping"))
    patience = c[3].number_input("Patience (epochs)", 2, 200, 15, disabled=not es)
    with st.expander("Advanced"):
        a = st.columns(4)
        val_split = a[0].slider("Validation split (from train)", .1, .4, .2, .05)
        loss = a[1].selectbox("Regression loss", ["MSE", "Huber"], disabled=p.task != "regression")
        clip = a[2].number_input("Gradient clipping (0 = off)", 0.0, 100.0, 0.0, help=helptext("Gradient clipping"))
        seed = a[3].number_input("Random seed", 0, 99999, 42)
    cfg = dl.DLConfig(hidden=hidden, activation=act, dropout=dropout, batchnorm=bn, optimizer=opt, lr=lr, weight_decay=wd,
                      batch_size=batch, epochs=int(epochs), scheduler=sched, early_stopping=es, patience=int(patience),
                      val_split=val_split, loss=loss, grad_clip=clip, seed=int(seed))
    tip("Epoch", "Batch / batch size", "Iteration / step", "Learning rate", "Optimizer", "Early stopping", "Loss function")

    n_tr = int(len(p.X_train) * (1 - val_split))
    info = dl.steps_info(n_tr, cfg)
    loss_name = {"binary": "Binary cross-entropy", "multiclass": "Cross-entropy", "regression": loss}[p.task]
    m = st.columns(5)
    m[0].metric("Training rows", info["train_rows"])
    m[1].metric("Steps / epoch", info["steps_per_epoch"], help="ceil(training rows / batch size) = weight updates per epoch")
    m[2].metric("Max iterations", f"{info['max_iterations']:,}", help="steps per epoch x epochs")
    m[3].metric("Loss function", loss_name)
    in_dim = ml.make_preprocessor(p).fit(p.X_train.head(500)).transform(p.X_train.head(5)).shape[1]
    m[4].metric("Trainable parameters", f"{dl.count_params(in_dim, cfg, dl.output_dim(p)):,}")
    show(dl.architecture_fig(in_dim, cfg, dl.output_dim(p)), key="arch")

    # ------------------------------------------------------------ live training
    st.subheader("3 · Train")
    b1, b2 = st.columns([1, 4])
    go_ = b1.button("▶️ Train network", type="primary")
    b2.caption("Press **Stop** in the top-right corner (or any button) to interrupt — curves trained so far are kept.")
    if go_:
        live(p, cfg)
    runs = st.session_state.dl_runs
    if st.session_state.get("dl_partial"):
        hist = pd.DataFrame(st.session_state.dl_partial)
        st.warning(f"Last training was interrupted at epoch {len(hist)}. Curves so far:")
        show(dl.curves_fig(hist, "loss"), key="partial")
    if runs:
        report(p, runs[-1], "last")
        compare(p, runs)


def live(p, cfg):
    st.session_state.dl_partial = []
    status = st.empty()
    bar = st.progress(0.0)
    a, b = st.columns(2)
    loss_ph, met_ph = a.empty(), b.empty()
    table_ph = st.empty()
    hist, snaps, setup, every = [], {}, None, 1
    t0 = time.time()
    for ev in dl.train_stream(p, cfg):
        if ev["type"] == "setup":
            setup = ev
            every = max(1, cfg.epochs // 60)  # redraw at most ~60 times
            continue
        if ev["type"] == "epoch":
            w = ev.pop("weights")
            if w is not None and (ev["epoch"] == 1 or ev["epoch"] % every == 0 or ev["improved"]):
                snaps[ev["epoch"]] = w
            hist.append(ev)
            st.session_state.dl_partial = hist
            bar.progress(min(ev["epoch"] / cfg.epochs, 1.0),
                         f"Epoch {ev['epoch']}/{cfg.epochs} · train loss {ev['train_loss']:.4f} · val loss {ev['val_loss']:.4f} · best epoch {ev['best_epoch']}")
            if ev["epoch"] % every == 0 or ev["epoch"] == 1:
                h = pd.DataFrame(hist)
                with loss_ph:
                    show(dl.curves_fig(h, "loss", ev["best_epoch"]), key=f"lloss{ev['epoch']}")
                with met_ph:
                    show(dl.curves_fig(h, "metric", ev["best_epoch"]), key=f"lmet{ev['epoch']}")
                table_ph.dataframe(h[["epoch", "train_loss", "val_loss", "train_metric", "val_metric", "lr", "seconds"]].tail(8).round(4),
                                   hide_index=True)
            continue
        done = ev
    status.success(f"Finished in {time.time() - t0:.1f}s — {done['last_epoch']} epochs, best epoch {done['best_epoch']}"
                   + (" (early stopping triggered, best weights restored)" if done["stopped_early"] else ""))
    st.session_state.dl_partial = []
    run = {"id": len(st.session_state.dl_runs) + 1, "cfg": cfg, "history": pd.DataFrame(hist), "snaps": snaps,
           "setup": setup, **{k: done[k] for k in ("net", "best_epoch", "last_epoch", "stopped_early", "pred", "proba", "test")}}
    st.session_state.dl_runs.append(run)
    loss_ph.empty(), met_ph.empty(), table_ph.empty(), bar.empty()


def report(p, run, key):
    st.subheader(f"4 · Results — run #{run['id']}")
    h = run["history"]
    stop = run["last_epoch"] if run["stopped_early"] else None
    a, b = st.columns(2)
    with a:
        show(dl.curves_fig(h, "loss", run["best_epoch"], stop), key=f"{key}loss")
    with b:
        show(dl.curves_fig(h, "metric", run["best_epoch"], stop), key=f"{key}met")
    tip("Train vs validation loss", "Overfitting", "Underfitting")
    findings(dl.diagnose(h, p.task), "Training diagnosis")
    st.markdown("**Test-set metrics (best weights)**")
    st.dataframe(pd.DataFrame([run["test"]]).round(4), hide_index=True)
    for t in ml.explain_metrics(p.task, run["test"], p.classes):
        st.info(t, icon="💡")
    with st.expander("Epoch-by-epoch table"):
        st.dataframe(h.drop(columns=["improved"]).round(5), hide_index=True)
        st.download_button("⬇️ Training history CSV", h.to_csv(index=False).encode(), f"dl_history_run{run['id']}.csv")
    a, b = st.columns(2)
    with a:
        show(dl.lr_grad_fig(h), key=f"{key}lr")
        tip("Learning-rate scheduler", "Vanishing / exploding gradients")
    with b:
        if run["snaps"]:
            layer = st.selectbox("Layer weights", list(next(iter(run["snaps"].values())).keys()), key=f"{key}layer")
            show(dl.weights_fig(run["snaps"], layer), key=f"{key}w")
    if p.task != "regression":
        show(ml.confusion_fig(p, run["pred"]), key=f"{key}cm")
    else:
        show(ml.regression_figs(p, run["pred"])[0], key=f"{key}reg")


def compare(p, runs):
    st.subheader("5 · Experiment tracker")
    rows = []
    for r in runs:
        c = r["cfg"]
        rows.append({"run": r["id"], "layers": "-".join(map(str, c.hidden)), "act": c.activation, "dropout": c.dropout,
                     "optimizer": c.optimizer, "lr": c.lr, "batch": c.batch_size, "epochs_run": r["last_epoch"],
                     "best_epoch": r["best_epoch"], "best_val_loss": r["history"].val_loss.min(),
                     **{f"test {k}": v for k, v in r["test"].items()}})
    tbl = pd.DataFrame(rows)
    st.dataframe(tbl.round(4), hide_index=True)
    if len(runs) > 1:
        long = pd.concat([r["history"].assign(run=f"#{r['id']}") for r in runs])
        show(px.line(long, x="epoch", y="val_loss", color="run", title="Validation loss — all runs"), key="cmpruns")
    lb = st.session_state.ml.get("leaderboard")
    best_cls = None
    if lb is not None and not lb.empty:
        key = [c for c in lb.columns if c.startswith("test ")][0]
        best_cls = lb.iloc[0]
    metric = "R2" if p.task == "regression" else ("ROC-AUC" if p.task == "binary" else "F1")
    best_nn = max(runs, key=lambda r: r["test"].get(metric, -1e9))
    msg = f"Best neural net: run #{best_nn['id']} with test {metric} {best_nn['test'].get(metric, float('nan')):.4f}."
    if best_cls is not None and f"test {metric}" in best_cls:
        cl = best_cls[f"test {metric}"]
        msg += f" Best classical model ({best_cls.model}): {cl:.4f}. "
        msg += ("The neural net wins here." if best_nn["test"].get(metric, -1) > cl else
                "Tree ensembles often beat neural nets on small tabular datasets — DL shines with large data, images, text and audio.")
    else:
        msg += " Run AutoML on the ML page to compare with classical models."
    st.info(msg, icon="⚖️")
