"""
tests.test_trees
~~~~~~~~~~~~~~~~
Unit and integration tests for XGBoost and LightGBM models (Phase 4).
"""

from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
import pytest

from riskguard.models.trees import LightGBMFraudModel, XGBoostFraudModel


@pytest.fixture
def synthetic_fraud_df() -> tuple[pd.DataFrame, pd.Series]:
    """Create a synthetic dataset mimicking credit card fraud structure."""
    rng = np.random.RandomState(42)
    n_samples = 200
    n_fraud = 15

    data = {f"V{i}": rng.randn(n_samples) for i in range(1, 29)}
    data["Time"] = rng.uniform(0, 172800, n_samples)
    data["Amount"] = rng.exponential(scale=50, size=n_samples)

    y = np.zeros(n_samples, dtype=int)
    fraud_indices = rng.choice(n_samples, size=n_fraud, replace=False)
    y[fraud_indices] = 1

    # Shift distribution for fraud samples on V14 and V4
    df = pd.DataFrame(data)
    df.loc[y == 1, "V14"] -= 3.0
    df.loc[y == 1, "V4"] += 2.5

    return df, pd.Series(y, name="Class")


class TestXGBoostFraudModel:
    """Test suite for XGBoostFraudModel."""

    def test_initialization(self) -> None:
        model = XGBoostFraudModel(n_estimators=50, max_depth=3)
        assert model.n_estimators == 50
        assert model.max_depth == 3
        assert not model.is_fitted
        assert model.model_name == "XGBoost Fraud Model"

    def test_fit_and_predict_proba(self, synthetic_fraud_df: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = synthetic_fraud_df
        model = XGBoostFraudModel(n_estimators=10, max_depth=2, random_seed=42)
        model.fit(X, y)

        assert model.is_fitted
        probs = model.predict_proba(X)
        assert isinstance(probs, np.ndarray)
        assert len(probs) == len(X)
        assert np.all((probs >= 0.0) & (probs <= 1.0))

        # Check fraud samples have higher mean score than legit samples
        mean_fraud = probs[y == 1].mean()
        mean_legit = probs[y == 0].mean()
        assert mean_fraud > mean_legit

    def test_predict_with_threshold(self, synthetic_fraud_df: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = synthetic_fraud_df
        model = XGBoostFraudModel(n_estimators=10, max_depth=2, random_seed=42)
        model.fit(X, y)

        model.optimal_threshold = 0.8
        preds_high = model.predict(X)
        preds_low = model.predict(X, threshold=0.1)

        assert np.sum(preds_low) >= np.sum(preds_high)

    def test_feature_names_and_importances(self, synthetic_fraud_df: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = synthetic_fraud_df
        model = XGBoostFraudModel(n_estimators=10, max_depth=2, random_seed=42)
        model.fit(X, y)

        names = model.feature_names
        assert len(names) == 31  # 28 V's + Amount_scaled + Time_scaled + HourOfDay_scaled
        assert "V14" in names
        assert "HourOfDay_scaled" in names

        importances = model.feature_importances
        assert len(importances) == len(names)

    def test_save_and_load(self, synthetic_fraud_df: tuple[pd.DataFrame, pd.Series], tmp_path: pathlib.Path) -> None:
        X, y = synthetic_fraud_df
        model = XGBoostFraudModel(n_estimators=10, max_depth=2, random_seed=42)
        model.fit(X, y)
        model.optimal_threshold = 0.65

        save_path = model.save(tmp_path)
        assert save_path.exists()

        loaded = XGBoostFraudModel.load(save_path)
        assert loaded.is_fitted
        assert loaded.optimal_threshold == 0.65

        np.testing.assert_allclose(model.predict_proba(X), loaded.predict_proba(X))

    def test_unfitted_raises_runtime_error(self, synthetic_fraud_df: tuple[pd.DataFrame, pd.Series]) -> None:
        X, _ = synthetic_fraud_df
        model = XGBoostFraudModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(X)


class TestLightGBMFraudModel:
    """Test suite for LightGBMFraudModel."""

    def test_fit_and_predict(self, synthetic_fraud_df: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = synthetic_fraud_df
        model = LightGBMFraudModel(n_estimators=10, num_leaves=15, random_seed=42)
        model.fit(X, y)

        assert model.is_fitted
        assert model.model_name == "LightGBM Fraud Model"

        probs = model.predict_proba(X)
        assert len(probs) == len(X)
        assert np.all((probs >= 0.0) & (probs <= 1.0))

    def test_save_and_load(self, synthetic_fraud_df: tuple[pd.DataFrame, pd.Series], tmp_path: pathlib.Path) -> None:
        X, y = synthetic_fraud_df
        model = LightGBMFraudModel(n_estimators=10, num_leaves=15, random_seed=42)
        model.fit(X, y)

        save_path = model.save(tmp_path)
        loaded = LightGBMFraudModel.load(save_path)
        assert loaded.is_fitted
        np.testing.assert_allclose(model.predict_proba(X), loaded.predict_proba(X))
