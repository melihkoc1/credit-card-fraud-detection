"""Metrics, cross-validation loops, bootstrap CI, analyst-capacity table and net benefit."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from src.data import RANDOM_STATE

METRICS = ["PR-AUC", "ROC-AUC", "Top 0.5% capture"]
SCHEME_STRATIFIED = "Stratified 5-fold"
SCHEME_TIME = "Time-based"


def top_k_capture(y_true, score, q=0.005):
    """Share of all frauds found among the top ``q`` fraction of transactions by score."""
    y_true = np.asarray(y_true)
    n = max(1, int(len(y_true) * q))
    idx = np.argsort(-np.asarray(score))[:n]
    return y_true[idx].sum() / y_true.sum()


def compute_metrics(y_true, p):
    """PR-AUC, ROC-AUC and top-0.5% capture for scores ``p``."""
    return {
        "PR-AUC": average_precision_score(y_true, p),
        "ROC-AUC": roc_auc_score(y_true, p),
        "Top 0.5% capture": top_k_capture(y_true, p),
    }


def score_model(model, X_te, y_te):
    """Metrics of a fitted model's predicted fraud probability on (X_te, y_te)."""
    p = model.predict_proba(X_te)[:, 1]
    return compute_metrics(y_te, p)


def stratified_splitter():
    return StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)


def time_splitter():
    """Expanding-window splitter; rows must be sorted by Time."""
    return TimeSeriesSplit(n_splits=4)


def run_cv(builder, splitter, X, y, scheme):
    """Cross-validate every model returned by ``builder(y_train_fold)``.

    A fresh model is built and fitted on each fold's training part only, so
    resampling (SMOTE) and class-ratio settings never see the validation part.
    """
    rows = []
    for fold, (tr, va) in enumerate(splitter.split(X, y), 1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        for name, model in builder(y_tr).items():
            s = score_model(model.fit(X_tr, y_tr), X_va, y_va)
            s.update(
                {
                    "Model": name,
                    "Fold": fold,
                    "Scheme": scheme,
                    "Val frauds": int(y_va.sum()),
                }
            )
            rows.append(s)
    return pd.DataFrame(rows)


def summarize_cv(cv_results):
    """Mean and std of each metric per (scheme, model)."""
    return cv_results.groupby(["Scheme", "Model"])[METRICS].agg(["mean", "std"]).round(3)


def bootstrap_ci(y_true, p, n_boot=500, alpha=0.05, seed=RANDOM_STATE):
    """Percentile bootstrap CI over transactions (resampled independently).

    Resamples without any fraud are discarded. Because transactions are
    resampled independently, fraud clustering is ignored and the true
    uncertainty is probably larger.
    Returns a DataFrame with the point estimate and lower/upper bounds.
    """
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    rng = np.random.default_rng(seed)
    idx_list = [rng.integers(0, len(y_true), len(y_true)) for _ in range(n_boot)]
    idx_list = [i for i in idx_list if y_true[i].sum() > 0]
    boot = pd.DataFrame([compute_metrics(y_true[i], p[i]) for i in idx_list])
    return pd.DataFrame(
        {
            "Point estimate": pd.Series(compute_metrics(y_true, p)),
            "Lower": boot.quantile(alpha / 2),
            "Upper": boot.quantile(1 - alpha / 2),
        }
    )


def capacity_row(y_true, p, amount, q):
    """Review the top ``q`` fraction of transactions by score and report what is caught.

    Precision = frauds caught / transactions reviewed;
    Recall = frauds caught / all frauds;
    Amount ratio = fraud amount caught / total fraud amount.
    """
    y_true = np.asarray(y_true)
    amount = np.asarray(amount)
    order = np.argsort(-np.asarray(p))
    n = int(len(y_true) * q)
    sel = order[:n]
    tp = y_true[sel].sum()
    total_frauds = y_true.sum()
    total_amount = amount[y_true == 1].sum()
    return {
        "Reviewed (%)": round(q * 100, 2),
        "Reviewed transactions": n,
        "Frauds caught": int(tp),
        "Precision": round(tp / n, 3),
        "Recall": round(tp / total_frauds, 3),
        "Amount caught ratio": round(amount[sel][y_true[sel] == 1].sum() / total_amount, 3),
    }


def capacity_table(y_true, p, amount, fractions=(0.001, 0.0025, 0.005, 0.01, 0.02)):
    return pd.DataFrame([capacity_row(y_true, p, amount, q) for q in fractions])


def net_benefit(y_true, p, amount, q, cost):
    """Fraud amount caught in the top ``q`` fraction minus ``cost`` per reviewed transaction.

    ``cost`` is an assumption (in the same unit as Amount); chargebacks and
    customer friction are not modelled.
    """
    y_true = np.asarray(y_true)
    amount = np.asarray(amount)
    order = np.argsort(-np.asarray(p))
    n = int(len(y_true) * q)
    sel = order[:n]
    return amount[sel][y_true[sel] == 1].sum() - cost * n


def best_review_fraction(y_true, p, amount, cost, fractions=None):
    """Review fraction (on the grid) with the highest net benefit, and that benefit."""
    if fractions is None:
        fractions = np.linspace(0.0005, 0.05, 100)
    values = np.array([net_benefit(y_true, p, amount, q, cost) for q in fractions])
    i = values.argmax()
    return fractions[i], values[i]


def plot_net_benefit(y_true, p, amount, costs=(1, 5, 25), path=None):
    """Net benefit vs. share of transactions reviewed, one curve per assumed cost."""
    qs = np.linspace(0.0005, 0.05, 100)
    plt.figure(figsize=(8, 4.5))
    for c in costs:
        f = np.array([net_benefit(y_true, p, amount, q, c) for q in qs])
        i = f.argmax()
        plt.plot(qs * 100, f, label=f"Review cost = {c} units (best: {qs[i] * 100:.2f}%)")
        plt.scatter(qs[i] * 100, f[i])
    plt.xlabel("Share of transactions reviewed (%)")
    plt.ylabel("Net benefit (units)")
    plt.title("Net benefit by analyst capacity (assumed costs)")
    plt.legend()
    plt.grid(alpha=0.3)
    if path is not None:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
