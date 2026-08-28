"""
riskguard.models.baseline
~~~~~~~~~~~~~~~~~~~~~~~~~
Logistic Regression baseline model (Phase 2).

Uses ``class_weight='balanced'`` to handle extreme class imbalance
(1 : 578 fraud-to-legit ratio) without any data augmentation.  This
establishes an interpretable performance floor before the stronger
XGBoost/LightGBM models are introduced in Phase 4.

The baseline bundles the feature-engineering transformer and the
logistic regression into a single sklearn :class:`~sklearn.pipeline.Pipeline`
so that:

1. Only training data is ever seen during ``fit``.
2. Calling ``predict_proba(X_test)`` applies the same transformations
   automatically — no risk of applying the wrong scaler.
3. The pipeline is serialisable as a single artefact with ``joblib``.

Usage::

    from riskguard.models.baseline import LogisticRegressionBaseline

    model = LogisticRegressionBaseline()
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)          # fraud probabilities
    y_pred = model.predict(X_test, threshold=0.5) # binary predictions
    print(model.feature_names)                    # for SHAP
"""

from __future__ import annotations

import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from riskguard.features.engineering import FraudFeatureTransformer
from riskguard.utils.logging import get_logger

logger = get_logger(__name__)

# Model artefact filename — kept as a constant to avoid magic strings.
_ARTEFACT_NAME: str = "baseline_model.joblib"


class LogisticRegressionBaseline:
    """Interpretable Logistic Regression baseline for card fraud detection.

    Wraps a :class:`~sklearn.pipeline.Pipeline` that chains:
    1. :class:`~riskguard.features.engineering.FraudFeatureTransformer`
    2. :class:`~sklearn.linear_model.LogisticRegression` with
       ``class_weight='balanced'``

    Args:
        max_iter:    Maximum number of solver iterations (default 1 000).
        solver:      Solver algorithm (default ``'lbfgs'``).
        C:           Inverse of regularisation strength (default 1.0).
        random_seed: Random seed for reproducible results.
    """

    def __init__(
        self,
        max_iter: int = 1_000,
        solver: str = "lbfgs",
        C: float = 1.0,
        random_seed: int = 42,
    ) -> None:
        self.max_iter = max_iter
        self.solver = solver
        self.C = C
        self.random_seed = random_seed

        self._pipeline: Pipeline = self._build_pipeline()

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LogisticRegressionBaseline":
        """Fit the feature transformer and logistic regression on training data.

        Args:
            X_train: Feature DataFrame (must contain Time, Amount, V1–V28).
            y_train: Binary fraud labels (0 = legit, 1 = fraud).

        Returns:
            Self (for method chaining).
        """
        logger.info(
            "Fitting baseline LogisticRegression — samples=%d, fraud=%d (%.4f%%)",
            len(X_train),
            int(y_train.sum()),
            y_train.mean() * 100,
        )
        self._pipeline.fit(X_train, y_train)
        logger.info("Baseline model fitted successfully.")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return the predicted fraud probability for each sample.

        Args:
            X: Feature DataFrame with the same schema as training data.

        Returns:
            1-D :class:`numpy.ndarray` of shape ``(n_samples,)`` with
            fraud probabilities in [0, 1].

        Raises:
            sklearn.exceptions.NotFittedError: If called before ``fit``.
        """
        check_is_fitted(self._pipeline)
        proba = self._pipeline.predict_proba(X)
        # Column 1 corresponds to class=1 (fraud).
        return proba[:, 1]

    def predict(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Return binary predictions at *threshold*.

        Lower thresholds increase recall at the cost of precision —
        the appropriate operating point is a business decision.

        Args:
            X:         Feature DataFrame.
            threshold: Probability cut-off above which a transaction is
                       flagged as fraud (default 0.5).

        Returns:
            1-D integer :class:`numpy.ndarray` (0 = allow, 1 = flag).
        """
        y_prob = self.predict_proba(X)
        return (y_prob >= threshold).astype(int)

    def save(self, output_dir: pathlib.Path) -> pathlib.Path:
        """Serialise the fitted pipeline to *output_dir*.

        Args:
            output_dir: Directory where the artefact will be written.

        Returns:
            Absolute path of the saved file.
        """
        check_is_fitted(self._pipeline)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / _ARTEFACT_NAME
        joblib.dump(self._pipeline, path)
        logger.info("Baseline model saved to: %s", path.resolve())
        return path

    @classmethod
    def load(cls, artefact_path: pathlib.Path) -> "LogisticRegressionBaseline":
        """Load a previously saved baseline model from *artefact_path*.

        Args:
            artefact_path: Path to the ``.joblib`` file.

        Returns:
            A :class:`LogisticRegressionBaseline` with the pipeline restored.
        """
        instance = cls.__new__(cls)
        instance._pipeline = joblib.load(artefact_path)
        logger.info("Baseline model loaded from: %s", artefact_path.resolve())
        return instance

    @property
    def feature_names(self) -> list[str]:
        """Output feature names after transformation.

        Only available after ``fit``.  Used for SHAP explanations in Phase 5.

        Returns:
            List of feature name strings.

        Raises:
            sklearn.exceptions.NotFittedError: If called before ``fit``.
        """
        check_is_fitted(self._pipeline)
        transformer: FraudFeatureTransformer = self._pipeline.named_steps["transformer"]
        return transformer.feature_names_out_

    @property
    def coef_(self) -> np.ndarray:
        """Logistic Regression coefficients (shape: [1, n_features]).

        Only available after ``fit``.  Useful for quick interpretability.
        """
        check_is_fitted(self._pipeline)
        lr: LogisticRegression = self._pipeline.named_steps["classifier"]
        return lr.coef_

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_pipeline(self) -> Pipeline:
        """Construct the sklearn Pipeline object."""
        return Pipeline(
            steps=[
                ("transformer", FraudFeatureTransformer()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=self.max_iter,
                        solver=self.solver,
                        C=self.C,
                        random_state=self.random_seed,
                    ),
                ),
            ]
        )
