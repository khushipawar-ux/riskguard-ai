"""
riskguard.models.imbalance
~~~~~~~~~~~~~~~~~~~~~~~~~~
Imbalance-handling strategies for the fraud detection pipeline (Phase 3).

Each strategy is applied **only to the training fold** — the test set is
never touched, oversampled, or undersampled.  Applying any resampler to
the test set would be a form of data leakage and invalidate all metrics.

Strategies implemented
-----------------------
* ``ClassWeighting``  — Pass ``class_weight='balanced'`` to the classifier.
  No data is added or removed.  Computationally cheapest.
* ``SmoteOversampling`` — Synthetic Minority Over-sampling TEchnique.
  Generates synthetic fraud examples to balance the training fold.
* ``RandomUndersampling`` — Randomly discard majority-class (legit) rows
  until the desired ratio is reached.

All strategies expose the same interface::

    handler = SmoteOversampling(sampling_strategy=0.1, random_seed=42)
    X_res, y_res = handler.fit_resample(X_train_transformed, y_train)

This makes strategies interchangeable and easy to compare.

Usage::

    from riskguard.models.imbalance import (
        ClassWeighting,
        SmoteOversampling,
        RandomUndersampling,
        ImbalanceStrategy,
    )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from riskguard.utils.logging import get_logger

logger = get_logger(__name__)


# ── Abstract base ─────────────────────────────────────────────────────────────


class ImbalanceStrategy(ABC):
    """Abstract base class for all imbalance-handling strategies.

    Subclasses must implement :meth:`fit_resample` and expose a
    ``name`` attribute used in comparison reports.
    """

    name: str

    @abstractmethod
    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray | pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resample (or pass-through) the training data.

        Args:
            X: Transformed feature matrix (already passed through
               :class:`~riskguard.features.engineering.FraudFeatureTransformer`).
            y: Binary training labels (0 = legit, 1 = fraud).

        Returns:
            Tuple of ``(X_resampled, y_resampled)`` — both are
            :class:`numpy.ndarray` instances.
        """

    @property
    def requires_class_weight(self) -> bool:
        """Whether this strategy sets ``class_weight='balanced'``
        on the downstream classifier."""
        return False


# ── Strategy: Class Weighting ─────────────────────────────────────────────────


@dataclass
class ClassWeighting(ImbalanceStrategy):
    """Pass ``class_weight='balanced'`` to the classifier without
    resampling the training data.

    This is the cheapest strategy — no synthetic samples are created and
    no data is discarded.  It instructs the loss function to penalise
    misclassified fraud examples more heavily, proportional to the
    class imbalance.

    Args:
        random_seed: Kept for interface consistency; not used internally.
    """

    name: str = "class_weight=balanced"
    random_seed: int = 42

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray | pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pass the training data through unchanged.

        Class weighting is applied directly to the classifier, not to the
        data.  This method is a no-op so all strategies share the same
        interface.
        """
        y_arr = np.asarray(y, dtype=int)
        fraud = int(y_arr.sum())
        logger.info(
            "[%s] No resampling — passing %d samples through (fraud=%d).",
            self.name,
            len(y_arr),
            fraud,
        )
        return np.asarray(X, dtype=float), y_arr

    @property
    def requires_class_weight(self) -> bool:
        return True


# ── Strategy: SMOTE Oversampling ──────────────────────────────────────────────


@dataclass
class SmoteOversampling(ImbalanceStrategy):
    """Synthetic Minority Over-sampling TEchnique (SMOTE).

    Generates synthetic fraud examples by interpolating between existing
    fraud samples in feature space.  The training fold is expanded; the
    test fold is never modified.

    .. warning::
        Always split **before** applying SMOTE, never after.  Applying
        SMOTE to the full dataset before splitting leaks synthetic test
        samples into training, inflating all metrics.

    Args:
        sampling_strategy: Target fraud fraction in the resampled training
            set (default 0.1 → 10 % fraud).
        random_seed:       Seed for reproducible synthetic samples.
        k_neighbors:       Number of nearest neighbours used by SMOTE
            (default 5 — the SMOTE paper default).
    """

    name: str = "SMOTE oversampling"
    sampling_strategy: float = 0.1
    random_seed: int = 42
    k_neighbors: int = 5

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray | pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE to generate synthetic fraud examples.

        Requires ``imbalanced-learn`` (``imblearn``) to be installed.

        Args:
            X: Transformed feature matrix.
            y: Binary training labels.

        Returns:
            Resampled ``(X_res, y_res)`` with more fraud examples.

        Raises:
            ImportError: If ``imbalanced-learn`` is not installed.
        """
        try:
            # pyrefly: ignore [missing-import]
            from imblearn.over_sampling import SMOTE
        except ImportError as exc:
            raise ImportError(
                "imbalanced-learn is required for SMOTE. "
                "Install it with: pip install imbalanced-learn"
            ) from exc

        y_arr = np.asarray(y, dtype=int)
        before_fraud = int(y_arr.sum())

        smote = SMOTE(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_seed,
            k_neighbors=self.k_neighbors,
        )
        X_res, y_res = smote.fit_resample(np.asarray(X, dtype=float), y_arr)
        after_fraud = int(y_res.sum())

        logger.info(
            "[%s] Resampled: %d -> %d samples | fraud: %d -> %d",
            self.name,
            len(y_arr),
            len(y_res),
            before_fraud,
            after_fraud,
        )
        return X_res, y_res


# ── Strategy: Random Undersampling ────────────────────────────────────────────


@dataclass
class RandomUndersampling(ImbalanceStrategy):
    """Randomly discard majority-class (legit) examples from training.

    This is the most aggressive strategy — it permanently discards real
    data.  Risk: losing potentially discriminating legit examples.

    Args:
        sampling_strategy: Target fraud fraction in the undersampled set
            (default 0.1 → 10 % fraud after undersampling).
        random_seed:       Seed for the random discard.
    """

    name: str = "random undersampling"
    sampling_strategy: float = 0.1
    random_seed: int = 42

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray | pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Randomly discard legit examples until the target ratio is reached.

        Requires ``imbalanced-learn`` (``imblearn``) to be installed.

        Args:
            X: Transformed feature matrix.
            y: Binary training labels.

        Returns:
            Undersampled ``(X_res, y_res)``.

        Raises:
            ImportError: If ``imbalanced-learn`` is not installed.
        """
        try:
            # pyrefly: ignore [missing-import]
            from imblearn.under_sampling import RandomUnderSampler
        except ImportError as exc:
            raise ImportError(
                "imbalanced-learn is required for RandomUndersampling. "
                "Install it with: pip install imbalanced-learn"
            ) from exc

        y_arr = np.asarray(y, dtype=int)
        before_legit = int((y_arr == 0).sum())
        before_fraud = int(y_arr.sum())

        rus = RandomUnderSampler(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_seed,
        )
        X_res, y_res = rus.fit_resample(np.asarray(X, dtype=float), y_arr)
        after_legit = int((y_res == 0).sum())

        logger.info(
            "[%s] Undersampled: legit %d -> %d | fraud %d (unchanged) | "
            "new ratio=1:%.0f",
            self.name,
            before_legit,
            after_legit,
            before_fraud,
            after_legit / before_fraud if before_fraud else 0,
        )
        return X_res, y_res
