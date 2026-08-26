"""
riskguard.features.engineering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Feature engineering pipeline (Phase 2+).

Transforms applied here:

* Scale ``Amount`` and ``Time`` using :class:`sklearn.preprocessing.StandardScaler`
  — V1–V28 are already PCA-standardised, so they are left unchanged.
* Add ``HourOfDay`` — derived from the ``Time`` column
  (``(Time % 86400) / 3600``).  EDA showed fraud clusters at specific hours.

All transformers are sklearn-compatible and fit only on training data to
prevent any leakage into the test fold.

Usage::

    from riskguard.features.engineering import FraudFeatureTransformer

    transformer = FraudFeatureTransformer()
    X_train_t = transformer.fit_transform(X_train)
    X_test_t  = transformer.transform(X_test)
    print(transformer.feature_names_out_)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from riskguard.utils.logging import get_logger

logger = get_logger(__name__)

# Columns that need standard-scaling.
_SCALE_COLS: list[str] = ["Amount", "Time"]

# Name for the engineered temporal feature.
HOUR_OF_DAY_COL: str = "HourOfDay"


class FraudFeatureTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for the Credit Card Fraud feature set.

    Pipeline:

    1. Compute ``HourOfDay = (Time % 86400) / 3600``.
    2. ``StandardScaler`` applied to ``Amount``, ``Time``, and ``HourOfDay``
       (fit only on training data).
    3. V1–V28 columns are passed through unchanged.

    The output is a :class:`numpy.ndarray` with columns ordered as:
    ``[V1, ..., V28, Amount_scaled, Time_scaled, HourOfDay_scaled]``.

    Attributes:
        feature_names_out_: List of output column names (set after ``fit``).
        scaler_: Fitted :class:`~sklearn.preprocessing.StandardScaler`
                 instance (set after ``fit``).
    """

    def __init__(self) -> None:
        # sklearn convention: constructor sets hyper-params only; no fitting.
        pass

    # ── sklearn interface ──────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y=None) -> "FraudFeatureTransformer":
        """Fit the scaler on *X* (training data only).

        Args:
            X:  DataFrame containing at least ``Time``, ``Amount``,
                and ``V1``–``V28`` columns.
            y:  Ignored — present for sklearn pipeline compatibility.

        Returns:
            Self (for method chaining).
        """
        self._validate_input(X)

        # Build the full column set (V-features + Amount + Time + HourOfDay).
        v_cols = [c for c in X.columns if c.startswith("V")]
        self._v_cols: list[str] = sorted(v_cols, key=lambda c: int(c[1:]))

        # Columns that will be scaled (Amount + Time + HourOfDay).
        self._scale_col_order: list[str] = _SCALE_COLS + [HOUR_OF_DAY_COL]

        # Fit scaler on training Amount, Time, and HourOfDay.
        X_aug = self._add_hour_of_day(X)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X_aug[self._scale_col_order])

        # Record output column names for SHAP and downstream use.
        scaled_names = [f"{c}_scaled" for c in self._scale_col_order]
        self.feature_names_out_: list[str] = self._v_cols + scaled_names

        logger.debug(
            "FraudFeatureTransformer fitted — %d features out: %s",
            len(self.feature_names_out_),
            self.feature_names_out_,
        )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        """Apply the fitted transformer to *X*.

        Args:
            X:  DataFrame with the same schema as the training data.
            y:  Ignored.

        Returns:
            :class:`numpy.ndarray` of shape ``(n_samples, n_features)``.

        Raises:
            sklearn.exceptions.NotFittedError: If called before ``fit``.
        """
        check_is_fitted(self, ["scaler_", "feature_names_out_"])
        self._validate_input(X)

        X_aug = self._add_hour_of_day(X)

        # Scale Amount, Time, HourOfDay.
        scaled = self.scaler_.transform(X_aug[self._scale_col_order])

        # Concatenate V-features (unchanged) with scaled columns.
        v_array = X_aug[self._v_cols].to_numpy(dtype=np.float64)
        result = np.concatenate([v_array, scaled], axis=1)

        logger.debug(
            "Transformed %d samples → shape %s", result.shape[0], result.shape
        )
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _add_hour_of_day(X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of *X* with the ``HourOfDay`` column appended."""
        X_copy = X.copy()
        X_copy[HOUR_OF_DAY_COL] = (X_copy["Time"] % 86400) / 3600
        return X_copy

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> None:
        """Raise :exc:`ValueError` if required columns are missing."""
        required = {"Time", "Amount"}
        missing = required - set(X.columns)
        if missing:
            raise ValueError(
                f"FraudFeatureTransformer requires columns {sorted(missing)}, "
                f"but they were not found in the input DataFrame."
            )
