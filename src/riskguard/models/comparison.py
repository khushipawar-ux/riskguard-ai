"""
riskguard.models.comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Imbalance-strategy comparison engine (Phase 3).

Runs each :class:`~riskguard.models.imbalance.ImbalanceStrategy` against
the same held-out test set and collects PR-AUC, F1, and recall metrics so
a single winner can be selected objectively.

Design principles
-----------------
* The feature transformer is fit **once** on the training data and reused
  across all strategies — it is the resampler that changes, not the scaler.
* The test set is transformed but **never resampled**.
* A fixed Logistic Regression is used so the comparison isolates the
  effect of the imbalance strategy rather than conflating it with model
  choice (which is Phase 4's responsibility).
* The comparison result is a ranked :class:`ComparisonResult` with an
  explicit :attr:`~ComparisonResult.winner` and a summary DataFrame,
  making the selection transparent and auditable.

Usage::

    from riskguard.models.comparison import ImbalanceComparison
    from riskguard.models.imbalance import (
        ClassWeighting, SmoteOversampling, RandomUndersampling
    )

    comparison = ImbalanceComparison(random_seed=42)
    result = comparison.run(X_train, X_test, y_train, y_test)
    print(result.winner.name)
    print(result.summary_df)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from riskguard.features.engineering import FraudFeatureTransformer
from riskguard.models.evaluator import EvaluationResult, ModelEvaluator
from riskguard.models.imbalance import (
    ClassWeighting,
    ImbalanceStrategy,
    RandomUndersampling,
    SmoteOversampling,
)
from riskguard.utils.logging import get_logger

logger = get_logger(__name__)

# Default strategies to compare — can be overridden in the constructor.
_DEFAULT_STRATEGIES: list[ImbalanceStrategy] = [
    ClassWeighting(random_seed=42),
    SmoteOversampling(sampling_strategy=0.1, random_seed=42),
    RandomUndersampling(sampling_strategy=0.1, random_seed=42),
]

# Logistic Regression hyper-parameters used for all strategy trials.
_LR_MAX_ITER: int = 1_000
_LR_C: float = 1.0
_LR_SOLVER: str = "lbfgs"


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass
class StrategyResult:
    """Outcome for a single imbalance strategy.

    Attributes:
        strategy:         The :class:`~riskguard.models.imbalance.ImbalanceStrategy`
                          that was evaluated.
        eval_result:      Full :class:`~riskguard.models.evaluator.EvaluationResult`
                          on the held-out test set.
        train_sample_count: Number of training samples after resampling.
        train_fraud_count:  Number of fraud samples after resampling.
    """

    strategy: ImbalanceStrategy
    eval_result: EvaluationResult
    train_sample_count: int
    train_fraud_count: int


@dataclass
class ComparisonResult:
    """Aggregated results from comparing all imbalance strategies.

    Attributes:
        strategy_results: Ordered list (by PR-AUC descending) of
                          :class:`StrategyResult` for each strategy.
        winner:           Strategy with the highest PR-AUC.
        summary_df:       DataFrame with one row per strategy for easy
                          reporting.
    """

    strategy_results: list[StrategyResult]
    winner: ImbalanceStrategy
    summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        if not self.summary_df.shape[0]:
            self.summary_df = self._build_summary()

    def _build_summary(self) -> pd.DataFrame:
        rows = []
        for sr in self.strategy_results:
            er = sr.eval_result
            rows.append(
                {
                    "Strategy": sr.strategy.name,
                    "PR-AUC": round(er.pr_auc, 4),
                    "Best F1": round(er.best_f1, 4),
                    "F1 @ t=0.50": round(er.f1_at_threshold, 4),
                    "Recall@80%prec": (
                        round(er.recall_at_80p, 4)
                        if er.recall_at_80p is not None
                        else "N/A"
                    ),
                    "Train samples": sr.train_sample_count,
                    "Train fraud": sr.train_fraud_count,
                    "Winner": sr.strategy.name == self.winner.name,
                }
            )
        return pd.DataFrame(rows).set_index("Strategy")


# ── Comparison engine ─────────────────────────────────────────────────────────


class ImbalanceComparison:
    """Run each imbalance strategy against the same test set and rank results.

    The comparison is fair because:

    * The same train/test split is used for every strategy.
    * The :class:`~riskguard.features.engineering.FraudFeatureTransformer`
      is fit only once on the raw training features (before resampling).
    * Resampling is applied to the **already-transformed** training matrix
      so synthetic feature values are generated in the scaled space.
    * The test set is only **transformed**, never resampled.

    Args:
        strategies:  List of strategies to compare.  Defaults to
                     ClassWeighting, SMOTE, and RandomUndersampling.
        random_seed: Seed for the Logistic Regression classifier.
    """

    def __init__(
        self,
        strategies: list[ImbalanceStrategy] | None = None,
        random_seed: int = 42,
    ) -> None:
        self.strategies = strategies or _DEFAULT_STRATEGIES
        self.random_seed = random_seed
        self._evaluator = ModelEvaluator()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> ComparisonResult:
        """Run all strategies and return a ranked :class:`ComparisonResult`.

        Args:
            X_train: Training features (raw, pre-transformation DataFrame).
            X_test:  Test features (raw, pre-transformation DataFrame).
            y_train: Training labels.
            y_test:  Test labels.

        Returns:
            :class:`ComparisonResult` with strategy results ranked by PR-AUC.
        """
        logger.info(
            "Starting imbalance comparison — %d strategies, "
            "train=%d, test=%d, fraud_train=%d",
            len(self.strategies),
            len(X_train),
            len(X_test),
            int(y_train.sum()),
        )

        # Fit the transformer once on training data only.
        transformer = FraudFeatureTransformer()
        X_train_t = transformer.fit_transform(X_train)
        X_test_t = transformer.transform(X_test)

        y_train_arr = np.asarray(y_train, dtype=int)
        y_test_arr = np.asarray(y_test, dtype=int)

        strategy_results: list[StrategyResult] = []
        for strategy in self.strategies:
            sr = self._evaluate_strategy(
                strategy, X_train_t, X_test_t, y_train_arr, y_test_arr
            )
            strategy_results.append(sr)

        # Rank by PR-AUC (descending).
        strategy_results.sort(key=lambda r: r.eval_result.pr_auc, reverse=True)
        winner = strategy_results[0].strategy

        logger.info(
            "Comparison complete. Winner: '%s' (PR-AUC=%.4f)",
            winner.name,
            strategy_results[0].eval_result.pr_auc,
        )

        return ComparisonResult(
            strategy_results=strategy_results,
            winner=winner,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _evaluate_strategy(
        self,
        strategy: ImbalanceStrategy,
        X_train_t: np.ndarray,
        X_test_t: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> StrategyResult:
        """Resample, train, and evaluate a single strategy."""
        logger.info("--- Evaluating strategy: %s ---", strategy.name)

        # Resample (or pass-through for ClassWeighting).
        X_res, y_res = strategy.fit_resample(X_train_t, y_train)

        # Build and fit a Logistic Regression (with or without class_weight).
        class_weight = "balanced" if strategy.requires_class_weight else None
        clf = LogisticRegression(
            class_weight=class_weight,
            max_iter=_LR_MAX_ITER,
            C=_LR_C,
            solver=_LR_SOLVER,
            random_state=self.random_seed,
        )
        clf.fit(X_res, y_res)

        # Predict probabilities on the untouched test set.
        y_prob = clf.predict_proba(X_test_t)[:, 1]

        eval_result = self._evaluator.compute_metrics(
            y_test, y_prob, operating_threshold=0.5
        )

        logger.info(
            "[%s] PR-AUC=%.4f | Best F1=%.4f | Recall@80%%prec=%s",
            strategy.name,
            eval_result.pr_auc,
            eval_result.best_f1,
            f"{eval_result.recall_at_80p:.4f}"
            if eval_result.recall_at_80p is not None
            else "N/A",
        )

        return StrategyResult(
            strategy=strategy,
            eval_result=eval_result,
            train_sample_count=len(y_res),
            train_fraud_count=int(y_res.sum()),
        )
