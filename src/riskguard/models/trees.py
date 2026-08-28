"""
riskguard.models.trees
~~~~~~~~~~~~~~~~~~~~~~
Production tree-based models for tabular fraud detection (Phase 4).

Wraps XGBoost and LightGBM into clean, scikit-learn compatible classes that:
* Integrate feature engineering (:class:`~riskguard.features.engineering.FraudFeatureTransformer`)
  directly into an end-to-end pipeline.
* Automatically compute and handle class imbalance (e.g. ``scale_pos_weight``).
* Persist models, optimal decision thresholds, and metadata via joblib.
* Expose boosters for SHAP TreeExplainer (Phase 5).

Usage::

    from riskguard.models.trees import XGBoostFraudModel, LightGBMFraudModel

    # Train XGBoost
    xgb = XGBoostFraudModel(n_estimators=100, max_depth=4, learning_rate=0.1)
    xgb.fit(X_train, y_train)
    y_prob = xgb.predict_proba(X_test)

    # Train LightGBM
    lgb = LightGBMFraudModel(n_estimators=100, num_leaves=31, learning_rate=0.05)
    lgb.fit(X_train, y_train)
    y_prob = lgb.predict_proba(X_test)
"""

from __future__ import annotations

import pathlib
from abc import ABC, abstractmethod
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from riskguard.features.engineering import FraudFeatureTransformer
from riskguard.utils.logging import get_logger

logger = get_logger(__name__)


class BaseTreeFraudModel(BaseEstimator, ClassifierMixin, ABC):
    """Abstract base class for tree-based fraud detection models.

    Integrates feature transformation and estimator training, providing
    consistent serialization and prediction APIs.
    """

    def __init__(self) -> None:
        self.transformer: FraudFeatureTransformer = FraudFeatureTransformer()
        self.optimal_threshold: float = 0.5
        self.is_fitted: bool = False

    @abstractmethod
    def _create_estimator(self, scale_pos_weight: float) -> Any:
        """Instantiate the underlying tree classifier."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable name of the model architecture."""

    @property
    @abstractmethod
    def estimator(self) -> Any:
        """The underlying tree classifier instance."""

    # ── Fitting & Prediction ──────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        scale_pos_weight: float | str | None = "auto",
    ) -> BaseTreeFraudModel:
        """Fit feature transformer and the tree classifier.

        Args:
            X: Raw feature DataFrame before feature engineering.
            y: Binary target labels (0 = legit, 1 = fraud).
            scale_pos_weight: Weight for positive class. If "auto", computed as
                ``count(0) / count(1)``. If ``None``, defaults to 1.0.

        Returns:
            ``self`` fitted instance.
        """
        y_arr = np.asarray(y, dtype=int)
        n_pos = int(np.sum(y_arr == 1))
        n_neg = int(np.sum(y_arr == 0))

        if n_pos == 0:
            raise ValueError("Training data contains zero positive (fraud) instances.")
        if n_neg == 0:
            raise ValueError("Training data contains zero negative (legit) instances.")

        if scale_pos_weight == "auto":
            calculated_spw = float(n_neg / n_pos)
        elif scale_pos_weight is None:
            calculated_spw = 1.0
        else:
            calculated_spw = float(scale_pos_weight)

        logger.info(
            "Fitting %s — samples=%d (neg=%d, pos=%d, spw=%.2f)",
            self.model_name,
            len(y_arr),
            n_neg,
            n_pos,
            calculated_spw,
        )

        # 1. Transform features
        X_trans = self.transformer.fit_transform(X)

        # 2. Instantiate and train estimator
        self._init_estimator(calculated_spw)
        self.estimator.fit(X_trans, y_arr)
        self.is_fitted = True

        logger.info("%s fit complete.", self.model_name)
        return self

    @abstractmethod
    def _init_estimator(self, scale_pos_weight: float) -> None:
        """Instantiate the estimator using the computed scale_pos_weight."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fraud probabilities for input samples.

        Args:
            X: Raw feature DataFrame.

        Returns:
            1-D array of predicted probabilities for class 1 (fraud), shape (n_samples,).
        """
        self._check_is_fitted()
        X_trans = self.transformer.transform(X)
        proba = self.estimator.predict_proba(X_trans)
        # Return probability of the positive class (column 1)
        return proba[:, 1]

    def predict(
        self,
        X: pd.DataFrame,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Predict binary fraud labels using the specified or optimal decision threshold.

        Args:
            X: Raw feature DataFrame.
            threshold: Decision cut-off in [0, 1]. Defaults to ``self.optimal_threshold``.

        Returns:
            1-D array of integer predictions (0 = legit, 1 = fraud).
        """
        prob = self.predict_proba(X)
        cutoff = self.optimal_threshold if threshold is None else threshold
        return (prob >= cutoff).astype(int)

    def get_transformed_features(self, X: pd.DataFrame) -> np.ndarray:
        """Transform raw DataFrame into processed feature matrix."""
        self._check_is_fitted()
        return self.transformer.transform(X)

    @property
    def feature_names(self) -> list[str]:
        """Names of transformed features."""
        self._check_is_fitted()
        return self.transformer.feature_names_out_

    @property
    def feature_importances(self) -> np.ndarray:
        """Feature importances from the underlying tree model."""
        self._check_is_fitted()
        return getattr(self.estimator, "feature_importances_", np.array([]))

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, destination: str | pathlib.Path) -> pathlib.Path:
        """Save model pipeline, optimal threshold, and metadata to disk.

        Args:
            destination: Target file path or directory.

        Returns:
            Resolved path of the saved artefact.
        """
        self._check_is_fitted()
        dest_path = pathlib.Path(destination)
        if dest_path.is_dir() or not dest_path.suffix:
            dest_path.mkdir(parents=True, exist_ok=True)
            filename = f"{self.model_name.lower().replace(' ', '_')}.joblib"
            filepath = dest_path / filename
        else:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            filepath = dest_path

        payload = {
            "model_name": self.model_name,
            "transformer": self.transformer,
            "estimator": self.estimator,
            "optimal_threshold": self.optimal_threshold,
            "feature_names": self.feature_names,
            "params": self.get_params(),
        }
        joblib.dump(payload, filepath)
        logger.info("Saved %s artefact to %s", self.model_name, filepath.resolve())
        return filepath

    @classmethod
    def load(cls, filepath: str | pathlib.Path) -> BaseTreeFraudModel:
        """Load a persisted model artefact from disk."""
        path = pathlib.Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path.resolve()}")

        payload = joblib.load(path)
        instance = cls()
        instance.transformer = payload["transformer"]
        instance._set_loaded_estimator(payload["estimator"])
        instance.optimal_threshold = payload.get("optimal_threshold", 0.5)
        instance.is_fitted = True
        logger.info("Loaded %s from %s", instance.model_name, path.resolve())
        return instance

    @abstractmethod
    def _set_loaded_estimator(self, estimator: Any) -> None:
        """Assign loaded estimator to internal field."""

    def _check_is_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.model_name} is not fitted yet. Call .fit() or .load() first."
            )


# ── XGBoost Implementation ────────────────────────────────────────────────────


class XGBoostFraudModel(BaseTreeFraudModel):
    """Production XGBoost model for fraud detection.

    Args:
        n_estimators: Number of boosting trees.
        max_depth: Maximum tree depth for base learners.
        learning_rate: Boosting learning rate (eta).
        subsample: Subsample ratio of training instances.
        colsample_bytree: Subsample ratio of columns when constructing each tree.
        random_seed: Random state for reproducibility.
        scale_pos_weight: Explicit positive class weighting or None (auto-computed in fit).
        **kwargs: Additional parameters passed to :class:`xgboost.XGBClassifier`.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_seed: int = 42,
        scale_pos_weight: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_seed = random_seed
        self.scale_pos_weight = scale_pos_weight
        self.extra_kwargs = kwargs
        self._clf: Any | None = None

    @property
    def model_name(self) -> str:
        return "XGBoost Fraud Model"

    @property
    def estimator(self) -> Any:
        return self._clf

    def _init_estimator(self, scale_pos_weight: float) -> None:
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for XGBoostFraudModel. "
                "Install it with: pip install xgboost"
            ) from exc

        spw = self.scale_pos_weight if self.scale_pos_weight is not None else scale_pos_weight
        self._clf = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=spw,
            random_state=self.random_seed,
            eval_metric="logloss",
            tree_method="hist",
            **self.extra_kwargs,
        )

    def _create_estimator(self, scale_pos_weight: float) -> Any:
        self._init_estimator(scale_pos_weight)
        return self._clf

    def _set_loaded_estimator(self, estimator: Any) -> None:
        self._clf = estimator


# ── LightGBM Implementation ───────────────────────────────────────────────────


class LightGBMFraudModel(BaseTreeFraudModel):
    """Production LightGBM model for fraud detection.

    Args:
        n_estimators: Number of boosting iterations.
        num_leaves: Maximum tree leaves for base learners.
        max_depth: Maximum tree depth.
        learning_rate: Boosting learning rate.
        subsample: Subsample ratio of training instances.
        colsample_bytree: Subsample ratio of columns.
        random_seed: Random state for reproducibility.
        scale_pos_weight: Positive class weight.
        **kwargs: Additional parameters passed to :class:`lightgbm.LGBMClassifier`.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_seed: int = 42,
        scale_pos_weight: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_seed = random_seed
        self.scale_pos_weight = scale_pos_weight
        self.extra_kwargs = kwargs
        self._clf: Any | None = None

    @property
    def model_name(self) -> str:
        return "LightGBM Fraud Model"

    @property
    def estimator(self) -> Any:
        return self._clf

    def _init_estimator(self, scale_pos_weight: float) -> None:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "lightgbm is required for LightGBMFraudModel. "
                "Install it with: pip install lightgbm"
            ) from exc

        # Use explicitly provided scale_pos_weight, or 1.0 default if None was passed in constructor
        spw = self.scale_pos_weight if self.scale_pos_weight is not None else 1.0
        self._clf = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=spw,
            random_state=self.random_seed,
            verbosity=-1,
            **self.extra_kwargs,
        )

    def _create_estimator(self, scale_pos_weight: float) -> Any:
        self._init_estimator(scale_pos_weight)
        return self._clf

    def _set_loaded_estimator(self, estimator: Any) -> None:
        self._clf = estimator
