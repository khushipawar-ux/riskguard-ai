"""Tests for riskguard.eda.analysis pure functions."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from riskguard.data.validator import EXPECTED_COLUMNS
from riskguard.eda.analysis import (
    ImbalanceStats,
    class_imbalance_stats,
    correlation_with_target,
    feature_separability,
    per_class_stats,
    temporal_fraud_rate,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Synthetic 200-row DataFrame matching the creditcard schema."""
    rng = np.random.default_rng(0)
    n_legit, n_fraud = 190, 10
    rows = n_legit + n_fraud
    data: dict = {
        "Time": np.concatenate([
            rng.uniform(0, 172800, n_legit),
            rng.uniform(0, 50000, n_fraud),   # fraud clusters early
        ]),
        "Amount": np.concatenate([
            rng.uniform(10, 500, n_legit),
            rng.uniform(0, 50, n_fraud),       # fraud is smaller
        ]),
        "Class": [0] * n_legit + [1] * n_fraud,
    }
    for i in range(1, 29):
        # V14 is strongly anti-correlated with fraud in the real dataset
        if i == 14:
            data[f"V{i}"] = np.concatenate([
                rng.normal(0, 1, n_legit),
                rng.normal(-5, 1, n_fraud),
            ])
        else:
            data[f"V{i}"] = rng.standard_normal(rows)
    return pd.DataFrame(data)[EXPECTED_COLUMNS]


# ── class_imbalance_stats ─────────────────────────────────────────────────────

class TestClassImbalanceStats:
    def test_returns_named_tuple(self, sample_df):
        result = class_imbalance_stats(sample_df)
        assert isinstance(result, ImbalanceStats)

    def test_correct_counts(self, sample_df):
        result = class_imbalance_stats(sample_df)
        assert result.legit_count == 190
        assert result.fraud_count == 10

    def test_percentages_sum_to_100(self, sample_df):
        result = class_imbalance_stats(sample_df)
        assert abs(result.legit_pct + result.fraud_pct - 100.0) < 1e-6

    def test_ratio_is_legit_over_fraud(self, sample_df):
        result = class_imbalance_stats(sample_df)
        assert abs(result.ratio - 19.0) < 1e-6


# ── per_class_stats ───────────────────────────────────────────────────────────

class TestPerClassStats:
    def test_returns_dataframe(self, sample_df):
        result = per_class_stats(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_both_classes_present(self, sample_df):
        result = per_class_stats(sample_df)
        assert 0 in result.index
        assert 1 in result.index

    def test_custom_columns(self, sample_df):
        result = per_class_stats(sample_df, columns=["Amount"])
        assert "Amount" in result.columns.get_level_values(0)


# ── temporal_fraud_rate ───────────────────────────────────────────────────────

class TestTemporalFraudRate:
    def test_returns_dataframe_with_expected_columns(self, sample_df):
        result = temporal_fraud_rate(sample_df)
        assert {"Fraud", "Total", "FraudRate"}.issubset(result.columns)

    def test_fraud_rate_between_0_and_100(self, sample_df):
        result = temporal_fraud_rate(sample_df)
        assert result["FraudRate"].between(0, 100).all()

    def test_hour_index_within_window(self, sample_df):
        window = 24
        result = temporal_fraud_rate(sample_df, window_hours=window)
        assert result.index.max() < window


# ── feature_separability ──────────────────────────────────────────────────────

class TestFeatureSeparability:
    def test_returns_series(self, sample_df):
        result = feature_separability(sample_df)
        assert isinstance(result, pd.Series)

    def test_sorted_descending(self, sample_df):
        result = feature_separability(sample_df)
        assert list(result.values) == sorted(result.values, reverse=True)

    def test_v14_is_top_feature(self, sample_df):
        """V14 was engineered to have the largest gap — should rank first."""
        result = feature_separability(sample_df)
        assert result.index[0] == "V14"

    def test_all_values_non_negative(self, sample_df):
        result = feature_separability(sample_df)
        assert (result >= 0).all()


# ── correlation_with_target ───────────────────────────────────────────────────

class TestCorrelationWithTarget:
    def test_returns_series(self, sample_df):
        result = correlation_with_target(sample_df)
        assert isinstance(result, pd.Series)

    def test_sorted_ascending(self, sample_df):
        result = correlation_with_target(sample_df)
        assert list(result.values) == sorted(result.values)

    def test_target_not_in_result(self, sample_df):
        result = correlation_with_target(sample_df)
        assert "Class" not in result.index

    def test_values_in_minus1_to_1(self, sample_df):
        result = correlation_with_target(sample_df)
        assert result.between(-1, 1).all()

    def test_v14_is_most_negative(self, sample_df):
        """V14 is anti-correlated with Class in the synthetic fixture."""
        result = correlation_with_target(sample_df)
        assert result.index[0] == "V14"
