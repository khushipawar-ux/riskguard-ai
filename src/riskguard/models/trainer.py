"""
riskguard.models.trainer
~~~~~~~~~~~~~~~~~~~~~~~~
Model training and cross-validation utilities (Phase 2 & Phase 4).

Responsibilities
----------------
* Produce a **stratified** train/test split — critical given the 1:578 fraud
  ratio; random splits can leave entire folds with zero fraud cases.
* Validate that both splits contain fraud examples.
* Run Stratified K-Fold Cross-Validation on tree models (XGBoost, LightGBM)
  to estimate out-of-fold PR-AUC and evaluate stability.
* Compare models fairly across folds and held-out test sets.

Usage::

    from riskguard.models.trainer import DataSplitter, ModelTrainer
    from riskguard.models.trees import XGBoostFraudModel, LightGBMFraudModel

    splitter = DataSplitter(test_size=0.2, random_seed=42)
    X_train, X_test, y_train, y_test = splitter.split(df)

    trainer = ModelTrainer(n_splits=5, random_seed=42)
    xgb_cv = trainer.cross_validate(XGBoostFraudModel, X_train, y_train, {"max_depth": 4, "n_estimators": 100})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Type

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from riskguard.models.evaluator import EvaluationResult, ModelEvaluator
from riskguard.models.threshold import ThresholdOptimizer
from riskguard.utils.logging import get_logger

logger = get_logger(__name__)

# Column names expected from the Credit Card Fraud dataset.
_TARGET_COL: str = "Class"
_DROP_COLS: list[str] = ["Class"]


class SplitError(RuntimeError):
    """Raised when a stratified split cannot satisfy the fraud-presence
    constraint (e.g. too few fraud samples for the requested test size)."""


class DataSplitter:
    """Produce a stratified train / test split for the fraud detection dataset.

    Using sklearn's ``train_test_split`` with ``stratify=y`` ensures that the
    fraud-to-legit ratio is preserved in both splits even when fraud is
    less than 0.2 % of all records.

    Args:
        test_size:   Fraction of data held out for testing (default 0.2).
        random_seed: Controls reproducibility of the split.
    """

    def __init__(
        self,
        test_size: float = 0.20,
        random_seed: int = 42,
    ) -> None:
        if not 0.0 < test_size < 1.0:
            raise ValueError(
                f"test_size must be in (0, 1), got {test_size!r}"
            )
        self.test_size = test_size
        self.random_seed = random_seed

    # ── Public API ────────────────────────────────────────────────────────────

    def split(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split *df* into train and test sets.

        Args:
            df: Raw (pre-transformation) DataFrame containing a ``Class``
                column as the binary target.

        Returns:
            Tuple of ``(X_train, X_test, y_train, y_test)`` where
            ``X_*`` are DataFrames without the target column and
            ``y_*`` are integer Series.

        Raises:
            SplitError: If either split ends up with zero fraud examples.
            ValueError: If the ``Class`` column is absent.
        """
        if _TARGET_COL not in df.columns:
            raise ValueError(
                f"DataFrame must contain a '{_TARGET_COL}' column, "
                f"but found only: {list(df.columns)}"
            )

        X = df.drop(columns=_DROP_COLS)
        y = df[_TARGET_COL].astype(int)

        fraud_total = int(y.sum())
        logger.info(
            "Splitting %d rows (fraud=%d, %.4f%%) — test_size=%.0f%%",
            len(df),
            fraud_total,
            fraud_total / len(df) * 100,
            self.test_size * 100,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_seed,
            stratify=y,
        )

        self._validate_split(y_train, y_test)

        logger.info(
            "Split complete — train: %d (fraud=%d) | test: %d (fraud=%d)",
            len(X_train),
            int(y_train.sum()),
            len(X_test),
            int(y_test.sum()),
        )
        return X_train, X_test, y_train, y_test

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _validate_split(y_train: pd.Series, y_test: pd.Series) -> None:
        """Raise :exc:`SplitError` if either split lacks fraud examples."""
        train_fraud = int(y_train.sum())
        test_fraud = int(y_test.sum())
        issues: list[str] = []
        if train_fraud == 0:
            issues.append("training set contains no fraud examples")
        if test_fraud == 0:
            issues.append("test set contains no fraud examples")
        if issues:
            raise SplitError(
                "Stratified split failed: "
                + "; ".join(issues)
                + ". Try increasing the dataset size or reducing test_size."
            )


# ── Cross-Validation & Model Training ─────────────────────────────────────────


@dataclass
class FoldMetric:
    """Performance metrics for a single CV fold."""

    fold: int
    pr_auc: float
    best_f1: float
    best_threshold: float
    recall_at_80p: float | None


@dataclass
class CVResult:
    """Cross-validation summary across all folds."""

    model_name: str
    params: dict[str, Any]
    fold_metrics: list[FoldMetric]
    mean_pr_auc: float
    std_pr_auc: float
    mean_best_f1: float
    oof_predictions: np.ndarray = field(repr=False)
    oof_y_true: np.ndarray = field(repr=False)


class ModelTrainer:
    """Orchestrates Stratified K-Fold CV and training for fraud classifiers."""

    def __init__(self, n_splits: int = 5, random_seed: int = 42) -> None:
        self.n_splits = n_splits
        self.random_seed = random_seed
        self.evaluator = ModelEvaluator()
        self.threshold_optimizer = ThresholdOptimizer()

    def cross_validate(
        self,
        model_class: Type[Any],
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        params: dict[str, Any] | None = None,
    ) -> CVResult:
        """Perform Stratified K-Fold Cross-Validation on training features.

        Args:
            model_class: Class of the model to instantiate (e.g. XGBoostFraudModel).
            X: Training feature DataFrame.
            y: Training target labels.
            params: Dictionary of hyperparameters to pass to the model constructor.

        Returns:
            :class:`CVResult` containing fold-by-fold and aggregated performance metrics.
        """
        init_params = params.copy() if params else {}
        init_params["random_seed"] = self.random_seed

        y_arr = np.asarray(y, dtype=int)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)

        oof_probs = np.zeros(len(y_arr), dtype=float)
        fold_metrics: list[FoldMetric] = []

        logger.info(
            "Starting %d-fold Stratified CV for %s...",
            self.n_splits,
            model_class.__name__,
        )

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_arr), 1):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

            # Train on fold
            model = model_class(**init_params)
            model.fit(X_tr, y_tr)

            # Predict on val fold
            val_probs = model.predict_proba(X_val)
            oof_probs[val_idx] = val_probs

            eval_res = self.evaluator.compute_metrics(y_val, val_probs)
            fm = FoldMetric(
                fold=fold,
                pr_auc=eval_res.pr_auc,
                best_f1=eval_res.best_f1,
                best_threshold=eval_res.best_threshold,
                recall_at_80p=eval_res.recall_at_80p,
            )
            fold_metrics.append(fm)
            logger.info("  Fold %d/%d — PR-AUC: %.4f | Best F1: %.4f", fold, self.n_splits, fm.pr_auc, fm.best_f1)

        pr_aucs = [fm.pr_auc for fm in fold_metrics]
        f1s = [fm.best_f1 for fm in fold_metrics]
        mean_pr_auc = float(np.mean(pr_aucs))
        std_pr_auc = float(np.std(pr_aucs))
        mean_best_f1 = float(np.mean(f1s))

        logger.info(
            "CV Complete for %s — Mean PR-AUC: %.4f (+/- %.4f) | Mean Best F1: %.4f",
            model_class.__name__,
            mean_pr_auc,
            std_pr_auc,
            mean_best_f1,
        )

        return CVResult(
            model_name=model_class.__name__,
            params=init_params,
            fold_metrics=fold_metrics,
            mean_pr_auc=mean_pr_auc,
            std_pr_auc=std_pr_auc,
            mean_best_f1=mean_best_f1,
            oof_predictions=oof_probs,
            oof_y_true=y_arr,
        )

    def train_final_model(
        self,
        model_class: Type[Any],
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        params: dict[str, Any] | None = None,
        optimal_policy: str = "max_f1",
    ) -> Any:
        """Train a final production model on full training data and calibrate its threshold.

        Args:
            model_class: Class of the model (e.g. XGBoostFraudModel).
            X_train: Full training feature DataFrame.
            y_train: Full training binary target labels.
            params: Final chosen hyperparameters.
            optimal_policy: Policy for setting ``optimal_threshold`` ("max_f1", "precision_80", "cost").

        Returns:
            Fitted and calibrated model instance.
        """
        init_params = params.copy() if params else {}
        init_params["random_seed"] = self.random_seed
        model = model_class(**init_params)
        model.fit(X_train, y_train)

        # Compute train predictions to calibrate default threshold
        train_probs = model.predict_proba(X_train)
        if optimal_policy == "precision_80":
            thresh, _ = self.threshold_optimizer.find_target_precision_threshold(y_train, train_probs, min_precision=0.80)
        elif optimal_policy == "cost":
            thresh, _ = self.threshold_optimizer.find_cost_optimal_threshold(y_train, train_probs)
        else:
            thresh, _ = self.threshold_optimizer.find_best_f1_threshold(y_train, train_probs)

        model.optimal_threshold = thresh
        logger.info(
            "Final %s trained and calibrated (policy=%s, threshold=%.4f).",
            model.model_name,
            optimal_policy,
            thresh,
        )
        return model
