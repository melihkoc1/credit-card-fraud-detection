"""SHAP explanations for the final tree model."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def shap_values(model, X):
    """SHAP values (log-odds scale) of a fitted tree model for the positive class."""
    explainer = shap.TreeExplainer(model)
    sv = explainer(X.astype(float))
    return sv.values if sv.values.ndim == 2 else sv.values[:, :, 1]


def mean_abs_shap(vals, columns, y_true):
    """Mean |SHAP| per feature over all transactions and over frauds only."""
    y_true = np.asarray(y_true)
    all_tx = pd.Series(np.abs(vals).mean(axis=0), index=columns)
    frauds = pd.Series(np.abs(vals[y_true == 1]).mean(axis=0), index=columns)
    return pd.DataFrame({"All transactions": all_tx, "Frauds only": frauds})


def top_features(importance, by, n=10):
    """Top-n rows of the importance table sorted by column ``by``."""
    return importance.sort_values(by, ascending=False).head(n).round(3)


def plot_summary(vals, X, path=None, max_display=15):
    """Beeswarm summary plot; saved to ``path`` if given."""
    shap.summary_plot(vals, X.astype(float), max_display=max_display, show=False)
    if path is not None:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
