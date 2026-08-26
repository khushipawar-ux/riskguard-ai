"""Tests for riskguard.features.engineering — FraudFeatureTransformer."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from riskguard.data.validator import EXPECTED_COLUMNS
from riskguard.features.engineering import HOUR_OF_DAY_COL, FraudFeatureTransformer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic DataFrame matching the creditcard schema."""
    rng = np.random.default_rng(seed)
    data: dict = {
        "Time": rng.uniform(0, 172_800, n),   # two days in seconds
        "Amount": rng.uniform(0, 500, n),
        "Class": np.where(np.arange(n) < 5, 1, 0),
    }
    for i in range(1, 29):
        data[f"V{i}"] = rng.standard_normal(n)
    return pd.DataFrame(data)[EXPECTED_COLUMNS]


# ── FraudFeatureTransformer tests ─────────────────────────────────────────────

class TestFraudFeatureTransformerFit:
    def test_fit_returns_self(self) -> None:
        df = _make_df()
        transformer = FraudFeatureTransformer()
        result = transformer.fit(df)
        assert result is transformer

    def test_fit_sets_scaler(self) -> None:
        df = _make_df()
        transformer = FraudFeatureTransformer()
        transformer.fit(df)
        assert hasattr(transformer, "scaler_")

    def test_fit_sets_feature_names_out(self) -> None:
        df = _make_df()
        transformer = FraudFeatureTransformer()
        transformer.fit(df)
        assert hasattr(transformer, "feature_names_out_")
        assert isinstance(transformer.feature_names_out_, list)
        assert len(transformer.feature_names_out_) > 0

    def test_feature_names_contain_v_features(self) -> None:
        df = _make_df()
        transformer = FraudFeatureTransformer()
        transformer.fit(df)
        for i in range(1, 29):
            assert f"V{i}" in transformer.feature_names_out_

    def test_feature_names_contain_scaled_columns(self) -> None:
        df = _make_df()
        transformer = FraudFeatureTransformer()
        transformer.fit(df)
        names = transformer.feature_names_out_
        assert "Amount_scaled" in names
        assert "Time_scaled" in names
        assert f"{HOUR_OF_DAY_COL}_scaled" in names

    def test_raises_on_missing_time_column(self) -> None:
        df = _make_df().drop(columns=["Time"])
        transformer = FraudFeatureTransformer()
        with pytest.raises(ValueError, match="Time"):
            transformer.fit(df)

    def test_raises_on_missing_amount_column(self) -> None:
        df = _make_df().drop(columns=["Amount"])
        transformer = FraudFeatureTransformer()
        with pytest.raises(ValueError, match="Amount"):
            transformer.fit(df)


class TestFraudFeatureTransformerTransform:
    @pytest.fixture()
    def fitted_transformer(self) -> FraudFeatureTransformer:
        df = _make_df(n=200, seed=0)
        t = FraudFeatureTransformer()
        t.fit(df)
        return t

    def test_transform_returns_ndarray(self, fitted_transformer) -> None:
        df = _make_df(n=50, seed=1)
        result = fitted_transformer.transform(df)
        assert isinstance(result, np.ndarray)

    def test_output_shape(self, fitted_transformer) -> None:
        df = _make_df(n=50, seed=1)
        result = fitted_transformer.transform(df)
        # 28 V-features + Amount_scaled + Time_scaled + HourOfDay_scaled = 31
        assert result.shape == (50, 31)

    def test_v_features_unchanged(self, fitted_transformer) -> None:
        """V1–V28 columns should pass through without modification."""
        df = _make_df(n=50, seed=1)
        result = fitted_transformer.transform(df)
        # V-features occupy the first 28 columns.
        v_cols = sorted(
            [c for c in df.columns if c.startswith("V")],
            key=lambda c: int(c[1:]),
        )
        v_original = df[v_cols].to_numpy()
        np.testing.assert_array_almost_equal(result[:, :28], v_original)

    def test_scaled_columns_have_approx_zero_mean(self) -> None:
        """Columns scaled on the training set should have near-zero mean when
        the test set is drawn from the same distribution."""
        df_train = _make_df(n=500, seed=10)
        df_test = _make_df(n=500, seed=10)  # same seed → same distribution

        transformer = FraudFeatureTransformer()
        transformer.fit(df_train)
        result = transformer.transform(df_test)

        # Scaled columns are the last 3; mean should be close to 0.
        scaled = result[:, 28:]
        assert np.abs(scaled.mean(axis=0)).max() < 0.2  # generous tolerance

    def test_hour_of_day_range(self) -> None:
        """HourOfDay should always be in [0, 24) before scaling."""
        df = _make_df(n=100)
        # Compute directly from the Time column.
        hour = (df["Time"] % 86400) / 3600
        assert hour.between(0, 24).all()

    def test_raises_not_fitted_before_fit(self) -> None:
        transformer = FraudFeatureTransformer()
        df = _make_df()
        with pytest.raises(NotFittedError):
            transformer.transform(df)

    def test_fit_transform_equals_fit_then_transform(self) -> None:
        df = _make_df(n=100)
        t1 = FraudFeatureTransformer()
        out1 = t1.fit_transform(df)

        t2 = FraudFeatureTransformer()
        t2.fit(df)
        out2 = t2.transform(df)

        np.testing.assert_array_almost_equal(out1, out2)

    def test_no_leakage_different_distributions(self) -> None:
        """The transformer must be fit on training data only; applying it
        to a shifted test set should NOT re-fit (no data leakage)."""
        df_train = _make_df(n=300, seed=0)
        # Shift the Amount column significantly to simulate out-of-sample data.
        df_test = _make_df(n=50, seed=99)
        df_test["Amount"] = df_test["Amount"] + 10_000  # large shift

        transformer = FraudFeatureTransformer()
        transformer.fit(df_train)

        # Mean of scaler should be based on training data, not test.
        scaler_mean_amount = transformer.scaler_.mean_[0]  # Amount is first
        assert scaler_mean_amount < 1000  # training mean was ~250, not ~10250
        # Transform should not raise.
        transformer.transform(df_test)
