"""Tests for riskguard.models.imbalance and riskguard.models.comparison."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from riskguard.data.validator import EXPECTED_COLUMNS
from riskguard.models.comparison import (
    ComparisonResult,
    ImbalanceComparison,
    StrategyResult,
)
from riskguard.models.imbalance import (
    ClassWeighting,
    ImbalanceStrategy,
    RandomUndersampling,
    SmoteOversampling,
)
from riskguard.models.trainer import DataSplitter


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_imbalanced_df(n_legit: int = 500, n_fraud: int = 20, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic imbalanced credit card data."""
    rng = np.random.default_rng(seed)
    rows = n_legit + n_fraud
    data: dict = {
        "Time": rng.uniform(0, 172_800, rows),
        "Amount": rng.uniform(0, 500, rows),
        "Class": [0] * n_legit + [1] * n_fraud,
    }
    for i in range(1, 29):
        if i == 14:
            data[f"V{i}"] = np.concatenate([
                rng.normal(0, 1, n_legit),
                rng.normal(-4, 1, n_fraud),
            ])
        else:
            data[f"V{i}"] = rng.standard_normal(rows)
    return pd.DataFrame(data)[EXPECTED_COLUMNS]


@pytest.fixture()
def sample_data() -> tuple[np.ndarray, np.ndarray]:
    """Transformed feature matrix and labels."""
    rng = np.random.default_rng(42)
    n_legit, n_fraud = 200, 10
    X = rng.standard_normal((n_legit + n_fraud, 31))
    y = np.array([0] * n_legit + [1] * n_fraud)
    return X, y


@pytest.fixture()
def split_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train/test split of synthetic raw data."""
    df = _make_imbalanced_df(n_legit=400, n_fraud=20, seed=42)
    splitter = DataSplitter(test_size=0.25, random_seed=42)
    return splitter.split(df)


# ── ImbalanceStrategy Tests ───────────────────────────────────────────────────

class TestClassWeighting:
    def test_fit_resample_leaves_data_unchanged(self, sample_data) -> None:
        X, y = sample_data
        strategy = ClassWeighting()
        X_res, y_res = strategy.fit_resample(X, y)

        assert X_res.shape == X.shape
        assert len(y_res) == len(y)
        np.testing.assert_array_equal(X_res, X)
        np.testing.assert_array_equal(y_res, y)

    def test_requires_class_weight_true(self) -> None:
        strategy = ClassWeighting()
        assert strategy.requires_class_weight is True
        assert strategy.name == "class_weight=balanced"


class TestSmoteOversampling:
    def test_fit_resample_increases_fraud_count(self, sample_data) -> None:
        X, y = sample_data
        initial_fraud = int(y.sum())
        initial_total = len(y)

        strategy = SmoteOversampling(sampling_strategy=0.2, random_seed=42, k_neighbors=3)
        X_res, y_res = strategy.fit_resample(X, y)

        res_fraud = int(y_res.sum())
        assert res_fraud > initial_fraud
        assert len(y_res) > initial_total
        assert X_res.shape[0] == len(y_res)
        assert X_res.shape[1] == X.shape[1]

    def test_requires_class_weight_false(self) -> None:
        strategy = SmoteOversampling()
        assert strategy.requires_class_weight is False


class TestRandomUndersampling:
    def test_fit_resample_reduces_legit_count(self, sample_data) -> None:
        X, y = sample_data
        initial_legit = int((y == 0).sum())
        initial_fraud = int(y.sum())

        strategy = RandomUndersampling(sampling_strategy=0.2, random_seed=42)
        X_res, y_res = strategy.fit_resample(X, y)

        res_legit = int((y_res == 0).sum())
        res_fraud = int(y_res.sum())

        assert res_legit < initial_legit
        assert res_fraud == initial_fraud
        assert len(y_res) < len(y)
        assert X_res.shape[0] == len(y_res)

    def test_requires_class_weight_false(self) -> None:
        strategy = RandomUndersampling()
        assert strategy.requires_class_weight is False


# ── ImbalanceComparison Tests ─────────────────────────────────────────────────

class TestImbalanceComparison:
    def test_run_evaluates_all_strategies(self, split_data) -> None:
        X_train, X_test, y_train, y_test = split_data

        strategies = [
            ClassWeighting(random_seed=42),
            SmoteOversampling(sampling_strategy=0.2, random_seed=42, k_neighbors=3),
            RandomUndersampling(sampling_strategy=0.2, random_seed=42),
        ]
        comparison = ImbalanceComparison(strategies=strategies, random_seed=42)
        result = comparison.run(X_train, X_test, y_train, y_test)

        assert isinstance(result, ComparisonResult)
        assert len(result.strategy_results) == 3
        assert isinstance(result.winner, ImbalanceStrategy)

    def test_results_ranked_by_pr_auc_descending(self, split_data) -> None:
        X_train, X_test, y_train, y_test = split_data

        comparison = ImbalanceComparison(random_seed=42)
        result = comparison.run(X_train, X_test, y_train, y_test)

        pr_aucs = [sr.eval_result.pr_auc for sr in result.strategy_results]
        assert pr_aucs == sorted(pr_aucs, reverse=True)
        assert result.strategy_results[0].strategy.name == result.winner.name

    def test_summary_df_structure(self, split_data) -> None:
        X_train, X_test, y_train, y_test = split_data

        comparison = ImbalanceComparison(random_seed=42)
        result = comparison.run(X_train, X_test, y_train, y_test)

        df = result.summary_df
        assert isinstance(df, pd.DataFrame)
        assert "PR-AUC" in df.columns
        assert "Best F1" in df.columns
        assert "Winner" in df.columns
        assert df["Winner"].sum() == 1  # exactly one winner
