import numpy as np
import pandas as pd
import pytest


def make_synthetic(n=1500, fraud_rate=0.05, seed=0):
    """Small creditcard-like frame with unique Time values and informative V1."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    df = pd.DataFrame(
        {
            "Time": rng.permutation(n).astype(float),
            "V1": rng.normal(size=n) + 2.5 * y,
            "V2": rng.normal(size=n) - 1.5 * y,
            "V3": rng.normal(size=n),
            "Amount": np.round(rng.gamma(2.0, 40.0, size=n), 2),
            "Class": y,
        }
    )
    return df


@pytest.fixture
def synthetic_df():
    return make_synthetic()
