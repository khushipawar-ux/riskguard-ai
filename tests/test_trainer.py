"""
tests.test_trainer
~~~~~~~~~~~~~~~~~~
Unit tests for ModelTrainer and Stratified Cross-Validation (Phase 4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskguard.models.trainer import CVResult, DataSplitter, ModelTrainer
from riskguard.models.trees import XGBoostFraudModel


@pytest.fixture
def synthetic_fraud_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create a synthetic dataset for cross-validation tests."""
    rng = np.random.RandomState(42)
    n_samples = 150
    n_fraud = 15

    data = {f"V{i}": rng.randn(n_samples) for i in range(1, 29)}
    data["Time"] = rng.uniform(0, 100000, n_samples)
    data["Amount"] = rng.exponential(scale=40, size=n_samples)

    y = np.zeros(n_samples, dtype=int)
    fraud_indices = rng.choice(n_samples, size=n_fraud, replace=False)
    y[fraud_indices] = 1

    df = pd.DataFrame(data)
    df.loc[y == 1, "V14"] -= 2.5
    return df, pd.Series(y, name="Class")


class TestModelTrainer:
    """Test suite for ModelTrainer."""

    def test_cross_validate(self, synthetic_fraud_data: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = synthetic_fraud_data
        trainer = ModelTrainer(n_splits=3, random_seed=42)

        cv_result = trainer.cross_validate(
            XGBoostFraudModel,
            X,
            y,
            params={"n_estimators": 5, "max_depth": 2},
        )

        assert isinstance(cv_result, CVResult)
        assert len(cv_result.fold_metrics) == 3
        assert 0.0 <= cv_result.mean_pr_auc <= 1.0
        assert len(cv_result.oof_predictions) == len(X)

    def test_train_final_model(self, synthetic_fraud_data: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = synthetic_fraud_data
        trainer = ModelTrainer(random_seed=42)

        model = trainer.train_final_model(
            XGBoostFraudModel,
            X,
            y,
            params={"n_estimators": 5, "max_depth": 2},
            optimal_policy="max_f1",
        )

        assert model.is_fitted
        assert 0.0 < model.optimal_threshold < 1.0
