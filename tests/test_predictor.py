"""
tests.test_predictor
~~~~~~~~~~~~~~~~~~~~
Unit tests for FraudPredictor and Phase 6 packaging.
"""

from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
import pytest

from riskguard.explainability.shap_analyzer import ShapAnalyzer
from riskguard.inference.predictor import FraudPredictor, PredictionResult
from riskguard.models.trees import XGBoostFraudModel


@pytest.fixture
def trained_predictor_and_data() -> tuple[FraudPredictor, pd.DataFrame, pd.Series]:
    """Create a trained FraudPredictor on synthetic data."""
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
    df.loc[y == 1, "V14"] -= 3.0
    df.loc[y == 1, "V4"] += 2.5

    y_series = pd.Series(y, name="Class")
    model = XGBoostFraudModel(n_estimators=10, max_depth=2, random_seed=42)
    model.fit(df, y_series)
    model.optimal_threshold = 0.70

    explainer = ShapAnalyzer(model=model, background_samples=50, random_seed=42)
    explainer.fit(df)

    predictor = FraudPredictor(model=model, explainer=explainer, threshold=0.70)
    return predictor, df, y_series


class TestFraudPredictor:
    """Test suite for FraudPredictor."""

    def test_predict_dict_input(self, trained_predictor_and_data: tuple[FraudPredictor, pd.DataFrame, pd.Series]) -> None:
        predictor, df, _ = trained_predictor_and_data
        tx_dict = df.iloc[0].to_dict()

        result = predictor.predict(tx_dict, transaction_id="TX_1001")
        assert isinstance(result, PredictionResult)
        assert result.transaction_id == "TX_1001"
        assert 0.0 <= result.fraud_probability <= 1.0
        assert result.risk_level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        assert result.decision in ("ALLOW", "MANUAL_REVIEW", "FLAG_FRAUD")
        assert len(result.top_reasons) <= 3
        assert result.latency_ms > 0

        d = result.to_dict()
        assert d["transaction_id"] == "TX_1001"
        assert "fraud_probability" in d
        assert "top_reasons" in d

    def test_predict_series_input(self, trained_predictor_and_data: tuple[FraudPredictor, pd.DataFrame, pd.Series]) -> None:
        predictor, df, _ = trained_predictor_and_data
        series_input = df.iloc[1]

        result = predictor.predict(series_input)
        assert isinstance(result, PredictionResult)
        assert 0.0 <= result.fraud_probability <= 1.0

    def test_predict_dataframe_input(self, trained_predictor_and_data: tuple[FraudPredictor, pd.DataFrame, pd.Series]) -> None:
        predictor, df, _ = trained_predictor_and_data
        df_row = df.iloc[[2]]

        result = predictor.predict(df_row)
        assert isinstance(result, PredictionResult)

    def test_predict_batch(self, trained_predictor_and_data: tuple[FraudPredictor, pd.DataFrame, pd.Series]) -> None:
        predictor, df, _ = trained_predictor_and_data
        batch = df.iloc[:5]

        results = predictor.predict_batch(batch)
        assert len(results) == 5
        assert all(isinstance(r, PredictionResult) for r in results)

    def test_predict_batch_df(self, trained_predictor_and_data: tuple[FraudPredictor, pd.DataFrame, pd.Series]) -> None:
        predictor, df, _ = trained_predictor_and_data
        batch = df.iloc[:10]

        res_df = predictor.predict_batch_df(batch)
        assert isinstance(res_df, pd.DataFrame)
        assert len(res_df) == 10
        assert "Fraud_Probability" in res_df.columns
        assert "Decision" in res_df.columns
        assert "Top_Risk_Drivers" in res_df.columns

    def test_threshold_override(self, trained_predictor_and_data: tuple[FraudPredictor, pd.DataFrame, pd.Series]) -> None:
        predictor, df, _ = trained_predictor_and_data
        row = df.iloc[0].to_dict()

        # Score with very high threshold (should ALLOW)
        res_high = predictor.predict(row, threshold_override=0.99)
        assert res_high.threshold_used == 0.99

        # Score with very low threshold (should FLAG)
        res_low = predictor.predict(row, threshold_override=0.01)
        assert res_low.threshold_used == 0.01

    def test_save_and_load_bundle(
        self,
        trained_predictor_and_data: tuple[FraudPredictor, pd.DataFrame, pd.Series],
        tmp_path: pathlib.Path,
    ) -> None:
        predictor, df, _ = trained_predictor_and_data
        export_dir = tmp_path / "bundle"

        predictor.save_bundle(export_dir)
        assert (export_dir / "model.joblib").exists()
        assert (export_dir / "metadata.json").exists()

        loaded_predictor = FraudPredictor.load_bundle(export_dir)
        assert loaded_predictor.threshold == predictor.threshold

        row = df.iloc[0].to_dict()
        res_orig = predictor.predict(row)
        res_loaded = loaded_predictor.predict(row)

        assert np.isclose(res_orig.fraud_probability, res_loaded.fraud_probability)
        assert res_orig.decision == res_loaded.decision

    def test_missing_columns_fallback(self, trained_predictor_and_data: tuple[FraudPredictor, pd.DataFrame, pd.Series]) -> None:
        predictor, _, _ = trained_predictor_and_data
        # Missing many V-columns
        sparse_tx = {"Amount": 100.0, "Time": 50000.0, "V4": 1.5}

        result = predictor.predict(sparse_tx)
        assert isinstance(result, PredictionResult)
        assert 0.0 <= result.fraud_probability <= 1.0
