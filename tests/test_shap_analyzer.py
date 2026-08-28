"""
tests.test_shap_analyzer
~~~~~~~~~~~~~~~~~~~~~~~~
Unit and integration tests for ShapAnalyzer and explainability layer (Phase 5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskguard.explainability.shap_analyzer import ShapAnalyzer, TransactionExplanation
from riskguard.models.trees import XGBoostFraudModel


@pytest.fixture
def fitted_tree_and_data() -> tuple[XGBoostFraudModel, pd.DataFrame, pd.Series]:
    """Train a fast XGBoostFraudModel on synthetic data."""
    rng = np.random.RandomState(42)
    n_samples = 120
    n_fraud = 12

    data = {f"V{i}": rng.randn(n_samples) for i in range(1, 29)}
    data["Time"] = rng.uniform(0, 100000, n_samples)
    data["Amount"] = rng.exponential(scale=30, size=n_samples)

    y = np.zeros(n_samples, dtype=int)
    fraud_indices = rng.choice(n_samples, size=n_fraud, replace=False)
    y[fraud_indices] = 1

    df = pd.DataFrame(data)
    df.loc[y == 1, "V14"] -= 3.0
    df.loc[y == 1, "V4"] += 2.0

    y_series = pd.Series(y, name="Class")
    model = XGBoostFraudModel(n_estimators=10, max_depth=2, random_seed=42)
    model.fit(df, y_series)

    return model, df, y_series


class TestShapAnalyzer:
    """Test suite for ShapAnalyzer."""

    def test_fit_initializes_explainer(self, fitted_tree_and_data: tuple[XGBoostFraudModel, pd.DataFrame, pd.Series]) -> None:
        model, df, _ = fitted_tree_and_data
        analyzer = ShapAnalyzer(model=model, background_samples=50, random_seed=42)
        analyzer.fit(df)

        assert analyzer._explainer is not None
        assert len(analyzer._feature_names) == 31

    def test_compute_shap_values(self, fitted_tree_and_data: tuple[XGBoostFraudModel, pd.DataFrame, pd.Series]) -> None:
        model, df, _ = fitted_tree_and_data
        analyzer = ShapAnalyzer(model=model, background_samples=50, random_seed=42)
        analyzer.fit(df)

        shap_vals = analyzer.compute_shap_values(df.iloc[:10])
        assert isinstance(shap_vals, np.ndarray)
        assert shap_vals.shape == (10, 31)

    def test_compute_global_importance(self, fitted_tree_and_data: tuple[XGBoostFraudModel, pd.DataFrame, pd.Series]) -> None:
        model, df, _ = fitted_tree_and_data
        analyzer = ShapAnalyzer(model=model, background_samples=50, random_seed=42)
        analyzer.fit(df)

        global_df = analyzer.compute_global_importance(df.iloc[:30])
        assert isinstance(global_df, pd.DataFrame)
        assert "Feature" in global_df.columns
        assert "Mean_Abs_SHAP" in global_df.columns
        assert "Importance_Pct" in global_df.columns
        assert len(global_df) == 31
        # Should be sorted descending
        assert global_df["Mean_Abs_SHAP"].is_monotonic_decreasing

    def test_explain_transaction(self, fitted_tree_and_data: tuple[XGBoostFraudModel, pd.DataFrame, pd.Series]) -> None:
        model, df, _ = fitted_tree_and_data
        analyzer = ShapAnalyzer(model=model, background_samples=50, random_seed=42)
        analyzer.fit(df)

        single_row = df.iloc[[0]]
        explanation = analyzer.explain_transaction(single_row, top_n=3)

        assert isinstance(explanation, TransactionExplanation)
        assert 0.0 <= explanation.fraud_probability <= 1.0
        assert len(explanation.risk_drivers) <= 3
        assert len(explanation.protective_factors) <= 3
        assert isinstance(explanation.human_readable_summary, str)

        d = explanation.to_dict()
        assert "fraud_probability" in d
        assert "risk_drivers" in d

    def test_explain_flagged_batch(self, fitted_tree_and_data: tuple[XGBoostFraudModel, pd.DataFrame, pd.Series]) -> None:
        model, df, _ = fitted_tree_and_data
        analyzer = ShapAnalyzer(model=model, background_samples=50, random_seed=42)
        analyzer.fit(df)

        batch = df.iloc[:4]
        explanations = analyzer.explain_flagged_batch(batch, top_n=2)
        assert len(explanations) == 4

    def test_unfitted_raises(self, fitted_tree_and_data: tuple[XGBoostFraudModel, pd.DataFrame, pd.Series]) -> None:
        model, df, _ = fitted_tree_and_data
        analyzer = ShapAnalyzer(model=model)
        with pytest.raises(RuntimeError, match="not fitted"):
            analyzer.explain_transaction(df.iloc[[0]])
