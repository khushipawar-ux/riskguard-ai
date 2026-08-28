"""
tests.test_threshold
~~~~~~~~~~~~~~~~~~~~
Unit tests for ThresholdOptimizer and risk policies (Phase 4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskguard.models.threshold import ThresholdEvaluation, ThresholdOptimizer


@pytest.fixture
def synthetic_scores() -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic ground truth and calibrated predictions."""
    rng = np.random.RandomState(42)
    n = 500
    y_true = np.zeros(n, dtype=int)
    y_true[:25] = 1  # 5% fraud

    # Fraud cases have higher probabilities
    y_prob = np.zeros(n)
    y_prob[y_true == 0] = rng.beta(1, 10, size=n - 25)
    y_prob[y_true == 1] = rng.beta(8, 2, size=25)

    return y_true, y_prob


class TestThresholdOptimizer:
    """Test suite for ThresholdOptimizer."""

    def test_find_best_f1_threshold(self, synthetic_scores: tuple[np.ndarray, np.ndarray]) -> None:
        y_true, y_prob = synthetic_scores
        optimizer = ThresholdOptimizer()
        thresh, best_f1 = optimizer.find_best_f1_threshold(y_true, y_prob)

        assert 0.0 < thresh < 1.0
        assert 0.0 < best_f1 <= 1.0

    def test_find_target_precision_threshold(self, synthetic_scores: tuple[np.ndarray, np.ndarray]) -> None:
        y_true, y_prob = synthetic_scores
        optimizer = ThresholdOptimizer()
        thresh, recall = optimizer.find_target_precision_threshold(y_true, y_prob, min_precision=0.80)

        assert 0.0 < thresh < 1.0
        assert recall is not None
        assert 0.0 < recall <= 1.0

    def test_find_cost_optimal_threshold(self, synthetic_scores: tuple[np.ndarray, np.ndarray]) -> None:
        y_true, y_prob = synthetic_scores
        optimizer = ThresholdOptimizer(default_cost_fn=500, default_cost_fp=25)
        thresh, min_cost = optimizer.find_cost_optimal_threshold(y_true, y_prob)

        assert 0.0 < thresh < 1.0
        assert min_cost >= 0

    def test_evaluate_threshold(self, synthetic_scores: tuple[np.ndarray, np.ndarray]) -> None:
        y_true, y_prob = synthetic_scores
        optimizer = ThresholdOptimizer()
        res = optimizer.evaluate_threshold(y_true, y_prob, threshold=0.50)

        assert isinstance(res, ThresholdEvaluation)
        assert res.threshold == 0.50
        assert res.true_positives + res.false_negatives == 25
        assert res.true_negatives + res.false_positives == 475

    def test_compare_policies(self, synthetic_scores: tuple[np.ndarray, np.ndarray]) -> None:
        y_true, y_prob = synthetic_scores
        optimizer = ThresholdOptimizer()
        df = optimizer.compare_policies(y_true, y_prob)

        assert isinstance(df, pd.DataFrame)
        assert "Standard Baseline (0.50)" in df.index
        assert "Max F1 Policy" in df.index
        assert "Cost-Optimal Policy" in df.index
        assert "precision" in df.columns
        assert "recall" in df.columns
