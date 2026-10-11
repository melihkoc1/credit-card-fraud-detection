"""Model builders: logistic regression, LightGBM, SMOTE pipelines, Isolation Forest."""
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.data import RANDOM_STATE


def _log_amount_transformer():
    """ColumnTransformer applying log1p to Amount; other columns pass through."""
    return ColumnTransformer(
        [("amt", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), ["Amount"])],
        remainder="passthrough",
    )


def make_logreg(**kw):
    """Logistic regression on log1p(Amount) + standardised features."""
    return Pipeline(
        [
            ("pre", _log_amount_transformer()),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, **kw)),
        ]
    )


def make_lgbm(**kw):
    """LightGBM with near-default settings (no hyperparameter tuning was done)."""
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbose=-1,
        **kw,
    )


def build_models(y_tr):
    """Baseline models. scale_pos_weight is the square root of the class ratio."""
    spw = float(np.sqrt((y_tr == 0).sum() / (y_tr == 1).sum()))
    return {
        "Logistic Regression": make_logreg(),
        "Logistic Reg. (balanced)": make_logreg(class_weight="balanced"),
        "LightGBM": make_lgbm(),
        "LightGBM (scale_pos_weight=sqrt ratio)": make_lgbm(scale_pos_weight=spw),
    }


def make_smote_model(base):
    """imblearn Pipeline: SMOTE is applied only when fitting, never at predict time.

    SMOTE uses distances, so Amount is log-transformed and everything is scaled first.
    """
    return ImbPipeline(
        [
            ("pre", _log_amount_transformer()),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(sampling_strategy=0.1, k_neighbors=5, random_state=RANDOM_STATE)),
            ("model", base),
        ]
    )


class IsoScorer:
    """Isolation Forest: uses no labels, outputs an anomaly score (higher = more anomalous)."""

    def __init__(self, **kw):
        self.m = IsolationForest(**kw)

    def fit(self, X, y=None):
        self.m.fit(X)
        return self

    def predict_proba(self, X):
        s = -self.m.score_samples(X)
        return np.column_stack([-s, s])


def build_models_v2(y_tr):
    """SMOTE variants and the unsupervised Isolation Forest (y_tr kept for a common signature)."""
    return {
        "Logistic Reg. + SMOTE": make_smote_model(
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        ),
        "LightGBM + SMOTE": make_smote_model(make_lgbm()),
        "Isolation Forest (unsupervised)": IsoScorer(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
