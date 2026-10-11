import numpy as np
from imblearn.over_sampling import SMOTE

from src import models
from src.data import split_features_target
from src.evaluate import run_cv, stratified_splitter


class RecordingSMOTE(SMOTE):
    """SMOTE that records how many rows it is asked to resample."""

    seen = []

    def fit_resample(self, X, y):
        RecordingSMOTE.seen.append(len(X))
        return super().fit_resample(X, y)


def test_smote_sees_only_the_training_part_of_each_fold(synthetic_df, monkeypatch):
    X, y = split_features_target(synthetic_df)
    monkeypatch.setattr(models, "SMOTE", RecordingSMOTE)
    RecordingSMOTE.seen = []

    splitter = stratified_splitter()
    expected_train_sizes = [len(tr) for tr, _ in splitter.split(X, y)]

    builder = lambda y_tr: {"smote": models.make_smote_model(models.LogisticRegression(max_iter=500))}
    cv = run_cv(builder, splitter, X, y, "Stratified 5-fold")

    assert RecordingSMOTE.seen == expected_train_sizes      # validation rows never reach SMOTE
    assert all(s < len(X) for s in RecordingSMOTE.seen)
    assert len(cv) == 5


def test_smote_pipeline_does_not_resample_at_predict_time(synthetic_df):
    X, y = split_features_target(synthetic_df)
    pipe = models.make_smote_model(models.LogisticRegression(max_iter=500)).fit(X.iloc[:1000], y.iloc[:1000])
    assert pipe.predict_proba(X.iloc[1000:]).shape == (len(X) - 1000, 2)


def test_all_models_fit_and_score(synthetic_df):
    X, y = split_features_target(synthetic_df)
    for builder in (models.build_models, models.build_models_v2):
        for name, m in builder(y).items():
            p = m.fit(X, y).predict_proba(X)[:, 1]
            assert p.shape == (len(X),) and np.isfinite(p).all(), name
