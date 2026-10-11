import numpy as np
import pytest

from src.evaluate import (
    bootstrap_ci,
    capacity_row,
    net_benefit,
    top_k_capture,
)


def hand_example():
    """20 transactions ranked by index (index 0 has the highest score).

    Frauds at ranks 0, 2, 7, 15 with amounts 100, 50, 25, 25 (total 200);
    every legitimate transaction has amount 10.
    """
    y = np.zeros(20, dtype=int)
    amount = np.full(20, 10.0)
    for i, a in zip([0, 2, 7, 15], [100, 50, 25, 25]):
        y[i] = 1
        amount[i] = a
    score = np.arange(20, 0, -1, dtype=float)
    return y, score, amount


def test_top_k_capture_matches_hand_calculation():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    assert top_k_capture(y, score, q=0.3) == pytest.approx(2 / 3)   # top 3 contain 2 of 3 frauds
    assert top_k_capture(y, score, q=0.01) == pytest.approx(1 / 3)  # at least one row is reviewed


def test_capacity_row_precision_recall_amount():
    y, score, amount = hand_example()
    r = capacity_row(y, score, amount, 0.25)    # reviews 5 rows: ranks 0-4, frauds at 0 and 2
    assert r["Reviewed transactions"] == 5
    assert r["Frauds caught"] == 2
    assert r["Precision"] == pytest.approx(2 / 5)
    assert r["Recall"] == pytest.approx(2 / 4)
    assert r["Amount caught ratio"] == pytest.approx(150 / 200)

    r = capacity_row(y, score, amount, 0.5)     # reviews 10 rows: frauds at 0, 2, 7
    assert r["Frauds caught"] == 3
    assert r["Precision"] == pytest.approx(0.3)
    assert r["Recall"] == pytest.approx(0.75)
    assert r["Amount caught ratio"] == pytest.approx(175 / 200)


def test_net_benefit_matches_hand_calculation():
    y, score, amount = hand_example()
    assert net_benefit(y, score, amount, 0.25, cost=5) == pytest.approx(150 - 5 * 5)
    assert net_benefit(y, score, amount, 0.5, cost=10) == pytest.approx(175 - 10 * 10)


def test_bootstrap_interval_is_ordered_and_reproducible():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.05).astype(int)
    p = np.clip(0.3 * y + rng.random(2000) * 0.7, 0, 1)
    ci = bootstrap_ci(y, p, n_boot=100)
    assert (ci["Lower"] <= ci["Upper"]).all()
    again = bootstrap_ci(y, p, n_boot=100)
    assert ci.equals(again)
