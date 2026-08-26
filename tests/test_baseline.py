"""Tests for riskguard.models.baseline — LogisticRegressionBaseline."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from riskguard.data.validator import EXPECTED_COLUMNS
from riskguard.models.baseline import LogisticRegressionBaseline


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_df(n_legit: int = 200, n_fraud: int = 10, seed: int = 42) -> pd.DataFrame:
    """Synthetic DataFrame matching the creditcard schema."""
    rng = np.random.default_rng(seed)
    rows = n_legit + n_fraud
    data: dict = {
        "Time": rng.uniform(0, 172_800, rows),
        "Amount": rng.uniform(0, 500, rows),
        "Class": [0] * n_legit + [1] * n_fraud,
    }
    for i in range(1, 29):
        if i == 14:
            # V14 strongly separates fraud from legit — mirrors real dataset.
            data[f"V{i}"] = np.concatenate([
                rng.normal(0, 1, n_legit),
                rng.normal(-5, 1, n_fraud),
            ])
        else:
            data[f"V{i}"] = rng.standard_normal(rows)
    return pd.DataFrame(data)[EXPECTED_COLUMNS]


@pytest.fixture()
def df() -> pd.DataFrame:
    return _make_df()


@pytest.fixture()
def split(df) -> tuple:
    """Return (X_train, X_test, y_train, y_test)."""
    from riskguard.models.trainer import DataSplitter
    splitter = DataSplitter(test_size=0.20, random_seed=42)
    return splitter.split(df)


@pytest.fixture()
def fitted_model(split) -> LogisticRegressionBaseline:
    X_train, _, y_train, _ = split
    model = LogisticRegressionBaseline(random_seed=42)
    model.fit(X_train, y_train)
    return model


# ── Construction ──────────────────────────────────────────────────────────────

class TestLogisticRegressionBaselineInit:
    def test_default_construction(self) -> None:
        model = LogisticRegressionBaseline()
        assert model.max_iter == 1_000
        assert model.solver == "lbfgs"
        assert model.C == 1.0

    def test_custom_params_stored(self) -> None:
        model = LogisticRegressionBaseline(max_iter=500, C=0.1, random_seed=99)
        assert model.max_iter == 500
        assert model.C == pytest.approx(0.1)
        assert model.random_seed == 99


# ── Fit ───────────────────────────────────────────────────────────────────────

class TestFit:
    def test_fit_returns_self(self, split) -> None:
        X_train, _, y_train, _ = split
        model = LogisticRegressionBaseline()
        result = model.fit(X_train, y_train)
        assert result is model

    def test_fit_does_not_raise(self, split) -> None:
        X_train, _, y_train, _ = split
        model = LogisticRegressionBaseline()
        model.fit(X_train, y_train)  # Should not raise.

    def test_fit_with_large_dataset(self) -> None:
        df_large = _make_df(n_legit=1_000, n_fraud=50)
        from riskguard.models.trainer import DataSplitter
        X_train, _, y_train, _ = DataSplitter().split(df_large)
        model = LogisticRegressionBaseline()
        model.fit(X_train, y_train)


# ── predict_proba ─────────────────────────────────────────────────────────────

class TestPredictProba:
    def test_returns_ndarray(self, fitted_model, split) -> None:
        _, X_test, _, _ = split
        y_prob = fitted_model.predict_proba(X_test)
        assert isinstance(y_prob, np.ndarray)

    def test_output_shape(self, fitted_model, split) -> None:
        _, X_test, _, _ = split
        y_prob = fitted_model.predict_proba(X_test)
        assert y_prob.shape == (len(X_test),)

    def test_probabilities_in_0_1(self, fitted_model, split) -> None:
        _, X_test, _, _ = split
        y_prob = fitted_model.predict_proba(X_test)
        assert (y_prob >= 0.0).all() and (y_prob <= 1.0).all()

    def test_fraud_gets_higher_scores_on_average(self, fitted_model, split) -> None:
        """With V14 strongly discriminating, fraud rows should score higher."""
        _, X_test, _, y_test = split
        y_prob = fitted_model.predict_proba(X_test)
        mean_fraud = y_prob[y_test == 1].mean() if (y_test == 1).any() else 0.5
        mean_legit = y_prob[y_test == 0].mean()
        assert mean_fraud > mean_legit

    def test_raises_not_fitted(self, split) -> None:
        _, X_test, _, _ = split
        model = LogisticRegressionBaseline()
        with pytest.raises((NotFittedError, AttributeError)):
            model.predict_proba(X_test)


# ── predict ───────────────────────────────────────────────────────────────────

class TestPredict:
    def test_returns_ndarray_of_ints(self, fitted_model, split) -> None:
        _, X_test, _, _ = split
        y_pred = fitted_model.predict(X_test)
        assert isinstance(y_pred, np.ndarray)
        assert set(y_pred).issubset({0, 1})

    def test_low_threshold_flags_more_positives(self, fitted_model, split) -> None:
        _, X_test, _, _ = split
        pred_low = fitted_model.predict(X_test, threshold=0.1)
        pred_high = fitted_model.predict(X_test, threshold=0.9)
        assert pred_low.sum() >= pred_high.sum()

    def test_threshold_0_flags_everything(self, fitted_model, split) -> None:
        _, X_test, _, _ = split
        y_pred = fitted_model.predict(X_test, threshold=0.0)
        # threshold=0 means prob >= 0 is always True.
        assert y_pred.sum() == len(X_test)

    def test_threshold_1_flags_nothing(self, fitted_model, split) -> None:
        _, X_test, _, _ = split
        y_pred = fitted_model.predict(X_test, threshold=1.0 + 1e-9)
        # threshold slightly above 1 → nothing flagged.
        assert y_pred.sum() == 0


# ── Save / Load ───────────────────────────────────────────────────────────────

class TestSaveLoad:
    def test_save_creates_file(self, fitted_model, tmp_path) -> None:
        path = fitted_model.save(tmp_path)
        assert path.exists()
        assert path.suffix == ".joblib"

    def test_load_preserves_predictions(self, fitted_model, split, tmp_path) -> None:
        _, X_test, _, _ = split
        original_probs = fitted_model.predict_proba(X_test)

        path = fitted_model.save(tmp_path)
        loaded = LogisticRegressionBaseline.load(path)
        loaded_probs = loaded.predict_proba(X_test)

        np.testing.assert_array_almost_equal(original_probs, loaded_probs)


# ── feature_names property ────────────────────────────────────────────────────

class TestFeatureNames:
    def test_feature_names_is_list(self, fitted_model) -> None:
        assert isinstance(fitted_model.feature_names, list)

    def test_feature_names_length(self, fitted_model) -> None:
        # 28 V-features + Amount_scaled + Time_scaled + HourOfDay_scaled = 31
        assert len(fitted_model.feature_names) == 31

    def test_feature_names_raises_before_fit(self) -> None:
        model = LogisticRegressionBaseline()
        with pytest.raises((NotFittedError, AttributeError)):
            _ = model.feature_names


# ── coef_ property ────────────────────────────────────────────────────────────

class TestCoef:
    def test_coef_shape(self, fitted_model) -> None:
        coef = fitted_model.coef_
        # Binary classification: shape [1, n_features].
        assert coef.shape == (1, 31)
