"""Deep learning for tabular data: a configurable MLP trained epoch by epoch.

`train_stream()` is a generator that yields one log dict per epoch, so the UI can
draw live loss curves. PyTorch is used when installed; otherwise scikit-learn's
MLP is trained with partial_fit, one epoch at a time, behind the same interface.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

import ml

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

TEMPLATE = "plotly_white"


@dataclass
class DLConfig:
    hidden: list[int] = field(default_factory=lambda: [64, 32])
    activation: str = "ReLU"
    dropout: float = 0.2
    batchnorm: bool = True
    optimizer: str = "Adam"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    batch_size: int = 32
    epochs: int = 100
    scheduler: str = "None"
    early_stopping: bool = True
    patience: int = 10
    val_split: float = 0.2
    loss: str = "auto"
    grad_clip: float = 0.0
    seed: int = 42
    track_weights: bool = True


ACTIVATIONS = ["ReLU", "LeakyReLU", "Tanh", "Sigmoid", "GELU", "ELU"]
OPTIMIZERS = ["Adam", "AdamW", "SGD", "SGD + momentum", "RMSprop"]
SCHEDULERS = ["None", "StepLR", "ReduceLROnPlateau", "CosineAnnealing"]


def backend() -> str:
    return "PyTorch" if HAS_TORCH else "scikit-learn MLP (PyTorch not installed)"


# ------------------------------------------------------------------ helpers
def _prep_arrays(p: ml.Prepared, cfg: DLConfig):
    pre = ml.make_preprocessor(p, "Standard", "One-Hot")
    X_tr_df, X_val_df, y_tr, y_val = train_test_split(
        p.X_train, p.y_train, test_size=cfg.val_split, random_state=cfg.seed,
        stratify=p.y_train if p.task != "regression" else None)
    X_tr = pre.fit_transform(X_tr_df).astype(np.float32)
    X_val = pre.transform(X_val_df).astype(np.float32)
    X_te = pre.transform(p.X_test).astype(np.float32)
    y_mu, y_sd = 0.0, 1.0
    if p.task == "regression":  # scale the target so the loss is well-conditioned
        y_mu, y_sd = float(np.mean(y_tr)), float(np.std(y_tr) or 1)
    return pre, X_tr, X_val, X_te, np.asarray(y_tr), np.asarray(y_val), y_mu, y_sd


def output_dim(p: ml.Prepared) -> int:
    return len(p.classes) if p.task == "multiclass" else 1


def count_params(in_dim: int, cfg: DLConfig, out_dim: int) -> int:
    total, prev = 0, in_dim
    for h in cfg.hidden:
        total += prev * h + h + (2 * h if cfg.batchnorm else 0)
        prev = h
    return total + prev * out_dim + out_dim


def steps_info(n_train: int, cfg: DLConfig) -> dict:
    spe = math.ceil(n_train / cfg.batch_size)
    return {"train_rows": n_train, "batch_size": cfg.batch_size, "steps_per_epoch": spe,
            "epochs": cfg.epochs, "max_iterations": spe * cfg.epochs}


# ------------------------------------------------------------------ torch model
if HAS_TORCH:
    def build_mlp(in_dim: int, out_dim: int, cfg: DLConfig) -> "nn.Module":
        act = {"ReLU": nn.ReLU, "LeakyReLU": nn.LeakyReLU, "Tanh": nn.Tanh, "Sigmoid": nn.Sigmoid,
               "GELU": nn.GELU, "ELU": nn.ELU}[cfg.activation]
        layers, prev = [], in_dim
        for h in cfg.hidden:
            layers.append(nn.Linear(prev, h))
            if cfg.batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def _optimizer(model, cfg: DLConfig):
        params = model.parameters()
        if cfg.optimizer == "Adam":
            return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        if cfg.optimizer == "AdamW":
            return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        if cfg.optimizer == "SGD":
            return torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        if cfg.optimizer == "SGD + momentum":
            return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, nesterov=True, weight_decay=cfg.weight_decay)
        return torch.optim.RMSprop(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    def _scheduler(opt, cfg: DLConfig):
        if cfg.scheduler == "StepLR":
            return torch.optim.lr_scheduler.StepLR(opt, step_size=max(cfg.epochs // 4, 1), gamma=.5)
        if cfg.scheduler == "ReduceLROnPlateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.5, patience=max(cfg.patience // 2, 2))
        if cfg.scheduler == "CosineAnnealing":
            return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
        return None

    def _loss_fn(task: str, cfg: DLConfig):
        if task == "binary":
            return nn.BCEWithLogitsLoss()
        if task == "multiclass":
            return nn.CrossEntropyLoss()
        return nn.HuberLoss() if cfg.loss == "Huber" else nn.MSELoss()


class TrainedNet:
    """Wraps a trained network + preprocessor so it predicts like a sklearn pipeline."""

    def __init__(self, model, pre, task, classes, y_mu, y_sd, kind):
        self.model, self.pre, self.task, self.classes = model, pre, task, classes
        self.y_mu, self.y_sd, self.kind = y_mu, y_sd, kind

    def _raw(self, X: pd.DataFrame) -> np.ndarray:
        Xt = self.pre.transform(X).astype(np.float32)
        if self.kind == "torch":
            self.model.eval()
            with torch.no_grad():
                return self.model(torch.from_numpy(Xt)).numpy()
        return Xt

    def predict_proba(self, X):
        if self.kind != "torch":
            return self.model.predict_proba(self._raw(X))
        out = self._raw(X)
        if self.task == "binary":
            p1 = 1 / (1 + np.exp(-out[:, 0]))
            return np.c_[1 - p1, p1]
        e = np.exp(out - out.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def predict(self, X):
        if self.task == "regression":
            out = self._raw(X)
            out = out[:, 0] if self.kind == "torch" else self.model.predict(out)
            return out * self.y_sd + self.y_mu
        return self.predict_proba(X).argmax(1)


# ------------------------------------------------------------------ training stream
def train_stream(p: ml.Prepared, cfg: DLConfig, should_stop=lambda: False):
    """Yield {'type': 'setup' | 'epoch' | 'done', ...} dictionaries."""
    np.random.seed(cfg.seed)
    pre, X_tr, X_val, X_te, y_tr, y_val, y_mu, y_sd = _prep_arrays(p, cfg)
    out_dim = output_dim(p)
    yield {"type": "setup", "in_dim": X_tr.shape[1], "out_dim": out_dim,
           "params": count_params(X_tr.shape[1], cfg, out_dim), "steps": steps_info(len(X_tr), cfg),
           "val_rows": len(X_val), "backend": backend()}
    if HAS_TORCH:
        yield from _train_torch(p, cfg, pre, X_tr, X_val, X_te, y_tr, y_val, y_mu, y_sd, out_dim, should_stop)
    else:
        yield from _train_sklearn(p, cfg, pre, X_tr, X_val, X_te, y_tr, y_val, y_mu, y_sd, should_stop)


def _metric(task, y_true, raw_or_pred, y_mu=0.0, y_sd=1.0):
    """Main per-epoch metric: accuracy for classification, R2 for regression."""
    from sklearn.metrics import accuracy_score, r2_score
    if task == "regression":
        return r2_score(y_true, raw_or_pred * y_sd + y_mu)
    return accuracy_score(y_true, raw_or_pred)


def _train_torch(p, cfg, pre, X_tr, X_val, X_te, y_tr, y_val, y_mu, y_sd, out_dim, should_stop):
    torch.manual_seed(cfg.seed)
    model = build_mlp(X_tr.shape[1], out_dim, cfg)
    opt, loss_fn = _optimizer(model, cfg), _loss_fn(p.task, cfg)
    sched = _scheduler(opt, cfg)

    def tt(y):
        if p.task == "binary":
            return torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        if p.task == "multiclass":
            return torch.tensor(y, dtype=torch.long)
        return torch.tensor((y - y_mu) / y_sd, dtype=torch.float32).unsqueeze(1)

    Xtr_t, Xval_t, ytr_t, yval_t = torch.from_numpy(X_tr), torch.from_numpy(X_val), tt(y_tr), tt(y_val)

    def decode(out):
        if p.task == "binary":
            return (out[:, 0] > 0).int().numpy()
        if p.task == "multiclass":
            return out.argmax(1).numpy()
        return out[:, 0].numpy()

    best, best_state, best_epoch, wait = np.inf, None, 0, 0
    g = torch.Generator().manual_seed(cfg.seed)
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=g)
        tot, grad_norms = 0.0, []
        for i in range(0, len(perm), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            if len(idx) < 2 and cfg.batchnorm:
                continue  # BatchNorm needs >1 row per batch
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip if cfg.grad_clip > 0 else 1e9))
            grad_norms.append(gn)
            opt.step()
            tot += loss.item() * len(idx)
        train_loss = tot / len(Xtr_t)
        model.eval()
        with torch.no_grad():
            out_tr, out_val = model(Xtr_t), model(Xval_t)
            val_loss = loss_fn(out_val, yval_t).item()
        train_m = _metric(p.task, y_tr, decode(out_tr), y_mu, y_sd)
        val_m = _metric(p.task, y_val, decode(out_val), y_mu, y_sd)
        lr = opt.param_groups[0]["lr"]
        if sched is not None:
            sched.step(val_loss) if cfg.scheduler == "ReduceLROnPlateau" else sched.step()
        improved = val_loss < best - 1e-6
        if improved:
            best, best_epoch, wait = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        weights = None
        if cfg.track_weights:
            weights = {f"layer{j}": m.weight.detach().numpy().ravel().copy()
                       for j, m in enumerate([m for m in model if isinstance(m, nn.Linear)])}
        log = {"type": "epoch", "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "train_metric": train_m, "val_metric": val_m, "lr": lr, "seconds": time.time() - t0,
               "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0, "best_epoch": best_epoch,
               "improved": improved, "weights": weights}
        yield log
        if not math.isfinite(train_loss):
            break
        if (cfg.early_stopping and wait >= cfg.patience) or should_stop():
            break
    stopped_early = cfg.early_stopping and wait >= cfg.patience
    if best_state is not None:
        model.load_state_dict(best_state)
    net = TrainedNet(model, pre, p.task, p.classes, y_mu, y_sd, "torch")
    yield _finish(p, net, best_epoch, epoch, stopped_early)


def _train_sklearn(p, cfg, pre, X_tr, X_val, X_te, y_tr, y_val, y_mu, y_sd, should_stop):
    from sklearn.metrics import log_loss, mean_squared_error
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    act = {"ReLU": "relu", "Tanh": "tanh", "Sigmoid": "logistic"}.get(cfg.activation, "relu")
    solver = "sgd" if cfg.optimizer.startswith("SGD") else "adam"
    kw = dict(hidden_layer_sizes=tuple(cfg.hidden), activation=act, solver=solver, alpha=cfg.weight_decay,
              learning_rate_init=cfg.lr, batch_size=cfg.batch_size, random_state=cfg.seed,
              momentum=cfg.momentum if cfg.optimizer == "SGD + momentum" else 0.0)
    reg = p.task == "regression"
    model = MLPRegressor(**kw) if reg else MLPClassifier(**kw)
    classes = np.arange(len(p.classes)) if not reg else None
    best, best_epoch, wait, best_model = np.inf, 0, 0, None
    import copy
    ytr_fit = (y_tr - y_mu) / y_sd if reg else y_tr
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.partial_fit(X_tr, ytr_fit) if reg else model.partial_fit(X_tr, ytr_fit, classes=classes)
        if reg:
            tl = mean_squared_error(ytr_fit, model.predict(X_tr))
            vl = mean_squared_error((y_val - y_mu) / y_sd, model.predict(X_val))
            tm, vm = _metric("regression", y_tr, model.predict(X_tr), y_mu, y_sd), _metric("regression", y_val, model.predict(X_val), y_mu, y_sd)
        else:
            tl = log_loss(y_tr, model.predict_proba(X_tr), labels=classes)
            vl = log_loss(y_val, model.predict_proba(X_val), labels=classes)
            tm, vm = _metric(p.task, y_tr, model.predict(X_tr)), _metric(p.task, y_val, model.predict(X_val))
        improved = vl < best - 1e-6
        if improved:
            best, best_epoch, wait, best_model = vl, epoch, 0, copy.deepcopy(model)
        else:
            wait += 1
        weights = {f"layer{j}": w.ravel().copy() for j, w in enumerate(model.coefs_)} if cfg.track_weights else None
        yield {"type": "epoch", "epoch": epoch, "train_loss": tl, "val_loss": vl, "train_metric": tm, "val_metric": vm,
               "lr": cfg.lr, "seconds": time.time() - t0, "grad_norm": float("nan"), "best_epoch": best_epoch,
               "improved": improved, "weights": weights}
        if (cfg.early_stopping and wait >= cfg.patience) or should_stop():
            break
    net = TrainedNet(best_model or model, pre, p.task, p.classes, y_mu, y_sd, "sklearn")
    yield _finish(p, net, best_epoch, epoch, cfg.early_stopping and wait >= cfg.patience)


def _finish(p, net, best_epoch, last_epoch, stopped_early):
    pred = net.predict(p.X_test)
    proba = net.predict_proba(p.X_test) if p.task != "regression" else None
    return {"type": "done", "net": net, "best_epoch": best_epoch, "last_epoch": last_epoch,
            "stopped_early": stopped_early, "pred": pred, "proba": proba,
            "test": ml.score_metrics(p.task, p.y_test, pred, proba)}


# ------------------------------------------------------------------ diagnosis & plots
def diagnose(history: pd.DataFrame, task: str) -> list[dict]:
    """Rule-based training diagnosis from the per-epoch history."""
    out = []
    if history.empty:
        return out
    tl, vl = history.train_loss.to_numpy(), history.val_loss.to_numpy()
    if not np.isfinite(tl).all() or tl[-1] > 10 * tl[0]:
        return [{"level": "warn", "text": "Loss exploded / became NaN -> learning rate is too high. Lower it 10x or add gradient clipping."}]
    n = len(history)
    best = int(np.argmin(vl))
    if n - best - 1 >= 5 and tl[-1] < tl[best]:
        out.append({"level": "warn", "text": f"Overfitting: validation loss bottomed out at epoch {best + 1} and rose for "
                                             f"{n - best - 1} epochs while training loss kept falling. Use early stopping, more dropout, "
                                             "weight decay, or a smaller network."})
    rel_drop = (tl[0] - tl[-1]) / (abs(tl[0]) + 1e-9)
    if n >= 10 and rel_drop < .05:
        out.append({"level": "warn", "text": "Loss barely moved -> learning rate too low, or features need scaling. Try a 3-10x higher LR."})
    good = .8 if task != "regression" else .6
    vm = history.val_metric.iloc[best]
    if vm < good and history.train_metric.iloc[best] < good:
        metric = "accuracy" if task != "regression" else "R2"
        out.append({"level": "warn", "text": f"Underfitting: train and validation {metric} are both low. Add layers/neurons, train more epochs, "
                                             "reduce dropout/weight decay, or engineer better features."})
    gap = vl[best] - tl[best]
    if not out:
        out.append({"level": "good", "text": f"Healthy training: best validation loss {vl[best]:.4f} at epoch {best + 1}; "
                                             f"train/val gap {gap:.4f}."})
    if n > 3 and np.std(np.diff(vl[-10:])) > abs(np.mean(vl[-10:])) * .2:
        out.append({"level": "info", "text": "Validation loss is noisy - try a larger batch size or a smaller learning rate."})
    return out


def curves_fig(history: pd.DataFrame, which: str = "loss", best_epoch: int | None = None, stop_epoch: int | None = None):
    fig = go.Figure()
    a, b = (("train_loss", "val_loss") if which == "loss" else ("train_metric", "val_metric"))
    fig.add_scatter(x=history.epoch, y=history[a], name="train", mode="lines", line_color="#2E86AB")
    fig.add_scatter(x=history.epoch, y=history[b], name="validation", mode="lines", line_color="#E4572E")
    if best_epoch:
        fig.add_vline(x=best_epoch, line_dash="dash", line_color="green", annotation_text=f"best epoch {best_epoch}")
    if stop_epoch:
        fig.add_vline(x=stop_epoch, line_dash="dot", line_color="grey", annotation_text="early stop", annotation_position="bottom right")
    ttl = "Loss per epoch" if which == "loss" else "Metric per epoch (accuracy / R2)"
    fig.update_layout(title=ttl, xaxis_title="epoch", template=TEMPLATE, height=340, margin=dict(t=50, b=40))
    return fig


def lr_grad_fig(history: pd.DataFrame):
    fig = go.Figure()
    fig.add_scatter(x=history.epoch, y=history.lr, name="learning rate", mode="lines")
    if history.grad_norm.notna().any():
        fig.add_scatter(x=history.epoch, y=history.grad_norm, name="mean gradient norm", mode="lines", yaxis="y2")
    fig.update_layout(title="Learning rate & gradient norm", xaxis_title="epoch", template=TEMPLATE, height=320,
                      yaxis=dict(title="learning rate (log)", type="log"), yaxis2=dict(title="grad norm", overlaying="y", side="right"))
    return fig


def weights_fig(weight_snapshots: dict[int, dict], layer: str):
    fig = go.Figure()
    epochs = sorted(weight_snapshots)
    pick = sorted(set([epochs[0], epochs[len(epochs) // 2], epochs[-1]]))
    for e in pick:
        w = weight_snapshots[e].get(layer)
        if w is not None:
            fig.add_histogram(x=w, name=f"epoch {e}", opacity=.55, nbinsx=50)
    fig.update_layout(barmode="overlay", title=f"Weight distribution of {layer} over training", template=TEMPLATE, height=320)
    return fig


def architecture_fig(in_dim: int, cfg: DLConfig, out_dim: int):
    layers = [("Input", in_dim)] + [(f"Dense {h}\n{cfg.activation}", h) for h in cfg.hidden] + [("Output", out_dim)]
    fig = go.Figure()
    max_show = 10
    for li, (name, n) in enumerate(layers):
        shown = min(n, max_show)
        ys = np.linspace(-1, 1, shown) if shown > 1 else np.array([0.0])
        if li < len(layers) - 1:
            nxt = min(layers[li + 1][1], max_show)
            ys2 = np.linspace(-1, 1, nxt) if nxt > 1 else np.array([0.0])
            for y1 in ys:
                for y2 in ys2:
                    fig.add_scatter(x=[li, li + 1], y=[y1, y2], mode="lines", line=dict(width=.3, color="#B0BEC5"),
                                    hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=[li] * shown, y=ys, mode="markers", marker=dict(size=14, color="#2E86AB" if 0 < li < len(layers) - 1 else "#E4572E"),
                        showlegend=False, hovertext=f"{name}: {n} units", hoverinfo="text")
        label = name.replace("\n", "<br>") + (f"<br>({n} units)" if n > max_show else "")
        fig.add_annotation(x=li, y=1.25, text=label, showarrow=False)
    extras = []
    if cfg.batchnorm:
        extras.append("BatchNorm")
    if cfg.dropout:
        extras.append(f"Dropout {cfg.dropout}")
    fig.update_layout(title="Network architecture" + (f" (each hidden layer: {', '.join(extras)})" if extras else ""),
                      template=TEMPLATE, height=360, xaxis=dict(visible=False), yaxis=dict(visible=False, range=[-1.3, 1.5]))
    return fig
