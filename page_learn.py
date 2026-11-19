import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from glossary import GLOSSARY

SECTIONS = {
    "Data & EDA": ["EDA", "Composition", "Distribution", "Comparison", "Relationship", "Skewness", "Kurtosis", "Outlier",
                   "Correlation", "Multicollinearity (VIF)", "Mutual information", "p-value", "Normality test", "Data leakage"],
    "Machine learning": ["Machine learning", "Supervised learning", "Unsupervised learning", "Train / validation / test split",
                         "Cross-validation", "Stratification", "Scaling", "Encoding", "Imputation", "Class imbalance", "SMOTE",
                         "Hyperparameter", "Grid / Random search", "Regularization", "Overfitting", "Underfitting",
                         "Bias-variance trade-off", "Learning curve", "Validation curve"],
    "Metrics": ["Accuracy", "Precision", "Recall", "F1 score", "ROC-AUC", "Confusion matrix", "Decision threshold", "Calibration",
                "R2", "MAE / RMSE", "Residual", "Feature importance", "Partial dependence", "SHAP"],
    "Clustering": ["K-Means", "Elbow method", "Silhouette score", "DBSCAN", "PCA", "t-SNE"],
    "Deep learning": ["Deep learning", "Neural network (MLP)", "Neuron / unit", "Weights & biases", "Activation function", "Epoch",
                      "Batch / batch size", "Iteration / step", "Loss function", "Gradient descent", "Backpropagation", "Learning rate",
                      "Optimizer", "Learning-rate scheduler", "Dropout", "Batch normalization", "Weight decay", "Early stopping",
                      "Gradient clipping", "Vanishing / exploding gradients", "Train vs validation loss"],
}


def render():
    st.title("📚 Learn mode & glossary")
    st.caption("Turn on **🎓 Learn mode** in the sidebar to see these explanations next to every control in the app.")
    q = st.text_input("Search a term", placeholder="e.g. epoch, recall, dropout")
    for sec, terms in SECTIONS.items():
        hits = [t for t in terms if not q or q.lower() in t.lower() or q.lower() in GLOSSARY[t][0].lower()]
        if not hits:
            continue
        with st.expander(f"{sec} ({len(hits)})", expanded=bool(q)):
            for t in hits:
                exp, ex = GLOSSARY[t]
                st.markdown(f"**{t}** — {exp}" + (f"  \n*Example:* {ex}" if ex else ""))

    st.divider()
    st.header("🧪 Interactive: how learning rate & epochs shape training")
    st.caption("Gradient descent on a simple bowl-shaped loss L(w) = (w − 3)². Watch what the step size does.")
    c = st.columns(3)
    lr = c[0].select_slider("Learning rate", [0.01, 0.05, 0.1, 0.3, 0.7, 0.95, 1.05], 0.1)
    epochs = c[1].slider("Epochs (steps)", 1, 60, 20)
    w0 = c[2].slider("Starting weight", -6.0, 10.0, -4.0)
    ws = [w0]
    for _ in range(epochs):
        ws.append(ws[-1] - lr * 2 * (ws[-1] - 3))
        if abs(ws[-1]) > 1e6:
            break
    ws = np.array(ws)
    grid = np.linspace(-8, 14, 200)
    fig = go.Figure()
    fig.add_scatter(x=grid, y=(grid - 3) ** 2, mode="lines", name="loss curve", line_color="#9FB3C8")
    fig.add_scatter(x=ws, y=(ws - 3) ** 2, mode="lines+markers", name="weight after each epoch",
                    marker=dict(size=8, color=np.arange(len(ws)), colorscale="Viridis"))
    fig.update_layout(xaxis_title="weight w", yaxis_title="loss", yaxis_range=[0, 130], xaxis_range=[-8, 14], height=380)
    st.plotly_chart(fig)
    final = (ws[-1] - 3) ** 2
    if lr >= 1:
        st.error("Learning rate too high: every step overshoots further — the loss explodes (diverges).")
    elif lr > .5:
        st.warning(f"Large learning rate: the weight zig-zags across the minimum. Final loss {final:.4f}.")
    elif final > .01:
        st.info(f"Final loss {final:.4f} — still descending. More epochs (or a higher learning rate) needed.")
    else:
        st.success(f"Converged to w ≈ {ws[-1]:.3f} (loss {final:.5f}) in {len(ws) - 1} epochs.")

    st.header("🧮 Epoch / batch / iteration calculator")
    c = st.columns(3)
    rows = c[0].number_input("Training rows", 1, 10_000_000, 1000)
    bs = c[1].number_input("Batch size", 1, 100_000, 32)
    ep = c[2].number_input("Epochs", 1, 10_000, 50)
    spe = int(np.ceil(rows / bs))
    st.write(f"**{spe}** iterations per epoch (ceil({rows} / {bs})) × **{ep}** epochs = **{spe * ep:,}** weight updates. "
             f"Each row is seen **{ep}** times.")
    st.dataframe(pd.DataFrame({"epoch": range(1, min(ep, 5) + 1), "rows seen so far": [rows * e for e in range(1, min(ep, 5) + 1)],
                               "updates so far": [spe * e for e in range(1, min(ep, 5) + 1)]}), hide_index=True)
