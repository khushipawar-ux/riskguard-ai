"""Tests for riskguard.models.evaluator — ModelEvaluator and EvaluationResult."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from riskguard.models.evaluator import EvaluationResult, ModelEvaluator


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def evaluator() -> ModelEvaluator:
    return ModelEvaluator()


@pytest.fixture()
def fraud_data() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic labels and probabilities with realistic class imbalance."""
    rng = np.random.default_rng(42)
    n = 1_000
    y_true = np.where(np.arange(n) < 20, 1, 0)  # 2% fraud
    # Fraud cases get higher scores; legit cases get lower.
    y_prob = np.where(
        y_true == 1,
        rng.uniform(0.5, 1.0, n),
        rng.uniform(0.0, 0.4, n),
    )
    return y_true, y_prob


@pytest.fixture()
def perfect_data() -> tuple[np.ndarray, np.ndarray]:
    """A classifier that perfectly separates fraud from legit."""
    y_true = np.array([0] * 990 + [1] * 10)
    y_prob = np.where(y_true == 1, 1.0, 0.0)
    return y_true, y_prob


@pytest.fixture()
def worst_data() -> tuple[np.ndarray, np.ndarray]:
    """A classifier that gives fraud a lower score than legit (inverted)."""
    y_true = np.array([0] * 990 + [1] * 10)
    y_prob = np.where(y_true == 1, 0.0, 1.0)  # perfectly wrong
    return y_true, y_prob


# ── ModelEvaluator.compute_metrics ────────────────────────────────────────────

class TestComputeMetrics:
    def test_returns_evaluation_result(self, evaluator, fraud_data) -> None:
        y_true, y_prob = fraud_data
        result = evaluator.compute_metrics(y_true, y_prob)
        assert isinstance(result, EvaluationResult)

    def test_pr_auc_between_0_and_1(self, evaluator, fraud_data) -> None:
        y_true, y_prob = fraud_data
        result = evaluator.compute_metrics(y_true, y_prob)
        assert 0.0 <= result.pr_auc <= 1.0

    def test_perfect_classifier_pr_auc_is_1(self, evaluator, perfect_data) -> None:
        y_true, y_prob = perfect_data
        result = evaluator.compute_metrics(y_true, y_prob)
        assert result.pr_auc == pytest.approx(1.0, abs=1e-6)

    def test_best_f1_ge_f1_at_threshold(self, evaluator, fraud_data) -> None:
        """Best F1 (optimised over thresholds) must be >= F1 at the default 0.5."""
        y_true, y_prob = fraud_data
        result = evaluator.compute_metrics(y_true, y_prob, operating_threshold=0.5)
        assert result.best_f1 >= result.f1_at_threshold - 1e-9

    def test_best_threshold_in_0_1(self, evaluator, fraud_data) -> None:
        y_true, y_prob = fraud_data
        result = evaluator.compute_metrics(y_true, y_prob)
        assert 0.0 <= result.best_threshold <= 1.0

    def test_confusion_matrix_shape(self, evaluator, fraud_data) -> None:
        y_true, y_prob = fraud_data
        result = evaluator.compute_metrics(y_true, y_prob)
        assert len(result.confusion) == 2
        assert len(result.confusion[0]) == 2
        assert len(result.confusion[1]) == 2

    def test_confusion_matrix_counts_sum_to_n(self, evaluator, fraud_data) -> None:
        y_true, y_prob = fraud_data
        result = evaluator.compute_metrics(y_true, y_prob)
        total = sum(result.confusion[i][j] for i in range(2) for j in range(2))
        assert total == len(y_true)

    def test_accepts_pandas_series(self, evaluator, fraud_data) -> None:
        y_true, y_prob = fraud_data
        y_series = pd.Series(y_true)
        result = evaluator.compute_metrics(y_series, y_prob)
        assert isinstance(result, EvaluationResult)

    def test_operating_threshold_stored(self, evaluator, fraud_data) -> None:
        y_true, y_prob = fraud_data
        result = evaluator.compute_metrics(y_true, y_prob, operating_threshold=0.3)
        assert result.operating_threshold == pytest.approx(0.3)


# ── ModelEvaluator.recall_at_precision ────────────────────────────────────────

class TestRecallAtPrecision:
    def test_returns_float_for_achievable_target(self, evaluator, fraud_data) -> None:
        y_true, y_prob = fraud_data
        result = evaluator.recall_at_precision(y_true, y_prob, min_precision=0.5)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_returns_none_when_precision_truly_unachievable(self, evaluator) -> None:
        """Precision target of 1.01 (above any valid value) must return None,
        since no real threshold can exceed 1.0 precision."""
        rng = np.random.default_rng(7)
        n = 1_000
        y_true = np.where(np.arange(n) < 10, 1, 0)
        y_prob = rng.uniform(0, 1, n)
        result = evaluator.recall_at_precision(y_true, y_prob, min_precision=1.01)
        assert result is None

    def test_constant_predictor_returns_zero_recall_at_high_precision(
        self, evaluator
    ) -> None:
        """sklearn includes a sentinel (precision=1.0, recall=0.0) representing
        'predict nothing', so a constant-probability classifier achieves
        recall=0.0 at 90% precision rather than returning None."""
        n = 10_000
        y_true = np.where(np.arange(n) < 100, 1, 0)
        y_prob = np.full(n, 0.01)  # constant — no useful discrimination
        result = evaluator.recall_at_precision(y_true, y_prob, min_precision=0.90)
        # recall=0 at precision=1.0 is the degenerate "predict nothing" point.
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_perfect_classifier_achieves_full_recall_at_80p(
        self, evaluator, perfect_data
    ) -> None:
        y_true, y_prob = perfect_data
        result = evaluator.recall_at_precision(y_true, y_prob, min_precision=0.80)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_higher_precision_target_lowers_or_equals_recall(
        self, evaluator, fraud_data
    ) -> None:
        y_true, y_prob = fraud_data
        r80 = evaluator.recall_at_precision(y_true, y_prob, min_precision=0.80)
        r90 = evaluator.recall_at_precision(y_true, y_prob, min_precision=0.90)
        if r80 is not None and r90 is not None:
            # Stricter precision target → recall can only stay same or drop.
            assert r80 >= r90 - 1e-9


# ── ModelEvaluator.threshold_curve ────────────────────────────────────────────

class TestThresholdCurve:
    def test_returns_dataframe(self, evaluator, fraud_data) -> None:
        y_true, y_prob = fraud_data
        curve = evaluator.threshold_curve(y_true, y_prob)
        assert isinstance(curve, pd.DataFrame)

    def test_required_columns_present(self, evaluator, fraud_data) -> None:
        y_true, y_prob = fraud_data
        curve = evaluator.threshold_curve(y_true, y_prob)
        assert {"threshold", "precision", "recall", "f1"}.issubset(curve.columns)

    def test_threshold_sorted_ascending(self, evaluator, fraud_data) -> None:
        y_true, y_prob = fraud_data
        curve = evaluator.threshold_curve(y_true, y_prob)
        assert list(curve["threshold"].values) == sorted(curve["threshold"].values)

    def test_precision_recall_in_0_1(self, evaluator, fraud_data) -> None:
        y_true, y_prob = fraud_data
        curve = evaluator.threshold_curve(y_true, y_prob)
        assert curve["precision"].between(0, 1).all()
        assert curve["recall"].between(0, 1).all()

    def test_f1_in_0_1(self, evaluator, fraud_data) -> None:
        y_true, y_prob = fraud_data
        curve = evaluator.threshold_curve(y_true, y_prob)
        assert curve["f1"].between(0, 1).all()

    def test_nonempty(self, evaluator, fraud_data) -> None:
        y_true, y_prob = fraud_data
        curve = evaluator.threshold_curve(y_true, y_prob)
        assert len(curve) > 0


# ── EvaluationResult ──────────────────────────────────────────────────────────

class TestEvaluationResult:
    def test_str_representation(self) -> None:
        result = EvaluationResult(
            pr_auc=0.75,
            f1_at_threshold=0.60,
            best_f1=0.65,
            best_threshold=0.35,
            recall_at_80p=0.45,
            operating_threshold=0.5,
            confusion=[[100, 5], [3, 12]],
        )
        s = str(result)
        assert "0.7500" in s

    def test_none_recall_at_80p_displayed(self) -> None:
        result = EvaluationResult(
            pr_auc=0.1,
            f1_at_threshold=0.1,
            best_f1=0.1,
            best_threshold=0.5,
            recall_at_80p=None,
            operating_threshold=0.5,
            confusion=[[90, 10], [8, 2]],
        )
        assert result.recall_at_80p is None
