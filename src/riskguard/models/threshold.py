"""
riskguard.models.threshold
~~~~~~~~~~~~~~~~~~~~~~~~~~
Decision threshold tuning and risk policy optimization (Phase 4).

In fraud detection under extreme class imbalance, the standard threshold
of 0.50 is almost always suboptimal. The decision threshold *is* the
business risk policy.

Policies implemented:
* **Max F1** — Balances precision and recall objectively.
* **Target Precision (e.g. >=80% / >=90%)** — Guarantees risk analyst review
  efficiency by ensuring high positive predictive value.
* **Cost Optimal** — Minimises expected financial loss based on the cost
  of missed fraud (False Negative) vs. review/friction cost (False Positive).

Usage::

    from riskguard.models.threshold import ThresholdOptimizer

    optimizer = ThresholdOptimizer()
    best_t, best_f1 = optimizer.find_best_f1_threshold(y_test, y_prob)
    target_t, rec = optimizer.find_target_precision_threshold(y_test, y_prob, min_precision=0.80)
    cost_t, min_cost = optimizer.find_cost_optimal_threshold(y_test, y_prob, cost_fn=500, cost_fp=25)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from riskguard.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ThresholdEvaluation:
    """Evaluation summary for a specific decision threshold."""

    threshold: float
    precision: float
    recall: float
    f1: float
    total_cost: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": round(self.threshold, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "total_cost": round(self.total_cost, 2),
            "TP": self.true_positives,
            "FP": self.false_positives,
            "TN": self.true_negatives,
            "FN": self.false_negatives,
        }


class ThresholdOptimizer:
    """Optimize classification thresholds for business and performance criteria."""

    def __init__(self, default_cost_fn: float = 500.0, default_cost_fp: float = 25.0) -> None:
        """Initialize ThresholdOptimizer with default cost parameters.

        Args:
            default_cost_fn: Estimated cost of a missed fraud transaction (FN).
            default_cost_fp: Estimated cost of reviewing a false positive (FP).
        """
        self.default_cost_fn = default_cost_fn
        self.default_cost_fp = default_cost_fp

    def find_best_f1_threshold(
        self,
        y_true: pd.Series | np.ndarray,
        y_prob: np.ndarray,
    ) -> tuple[float, float]:
        """Find the decision threshold that maximizes the F1 score.

        Args:
            y_true: Ground truth binary labels.
            y_prob: Predicted fraud probabilities.

        Returns:
            Tuple of ``(optimal_threshold, max_f1_score)``.
        """
        y_true_arr = np.asarray(y_true, dtype=int)
        y_prob_arr = np.asarray(y_prob, dtype=float)

        precisions, recalls, thresholds = precision_recall_curve(y_true_arr, y_prob_arr)
        # thresholds is 1 element shorter than precisions/recalls
        prec = precisions[:-1]
        rec = recalls[:-1]

        denom = prec + rec
        with np.errstate(invalid="ignore"):
            f1_scores = np.where(denom > 0, 2 * prec * rec / denom, 0.0)

        if len(f1_scores) == 0:
            return 0.5, 0.0

        best_idx = int(np.argmax(f1_scores))
        best_threshold = float(thresholds[best_idx])
        best_f1 = float(f1_scores[best_idx])

        logger.info("Optimal F1 threshold: %.4f (F1=%.4f)", best_threshold, best_f1)
        return best_threshold, best_f1

    def find_target_precision_threshold(
        self,
        y_true: pd.Series | np.ndarray,
        y_prob: np.ndarray,
        min_precision: float = 0.80,
    ) -> tuple[float, float | None]:
        """Find the threshold that achieves at least `min_precision` with maximum recall.

        Args:
            y_true: Ground truth binary labels.
            y_prob: Predicted fraud probabilities.
            min_precision: Minimum required precision (default 0.80).

        Returns:
            Tuple of ``(threshold, recall)``. If target precision is never met,
            returns ``(0.5, None)``.
        """
        y_true_arr = np.asarray(y_true, dtype=int)
        y_prob_arr = np.asarray(y_prob, dtype=float)

        precisions, recalls, thresholds = precision_recall_curve(y_true_arr, y_prob_arr)
        prec = precisions[:-1]
        rec = recalls[:-1]

        valid_mask = prec >= min_precision
        if not np.any(valid_mask):
            logger.warning("Target precision %.2f cannot be achieved.", min_precision)
            return 0.5, None

        # Among those satisfying min_precision, choose the one with highest recall
        # (if multiple, choose the lowest threshold among max recall)
        valid_indices = np.where(valid_mask)[0]
        best_idx = valid_indices[np.argmax(rec[valid_indices])]

        target_threshold = float(thresholds[best_idx])
        achieved_recall = float(rec[best_idx])

        logger.info(
            "Target precision (>=%.2f) threshold: %.4f (recall=%.4f, precision=%.4f)",
            min_precision,
            target_threshold,
            achieved_recall,
            float(prec[best_idx]),
        )
        return target_threshold, achieved_recall

    def find_cost_optimal_threshold(
        self,
        y_true: pd.Series | np.ndarray,
        y_prob: np.ndarray,
        cost_fn: float | None = None,
        cost_fp: float | None = None,
        num_thresholds: int = 200,
    ) -> tuple[float, float]:
        """Find the threshold minimizing expected cost = (cost_fn * FN) + (cost_fp * FP).

        Args:
            y_true: Ground truth binary labels.
            y_prob: Predicted fraud probabilities.
            cost_fn: Cost of a missed fraud (False Negative).
            cost_fp: Cost of a false alarm / analyst review (False Positive).
            num_thresholds: Grid search resolution.

        Returns:
            Tuple of ``(optimal_threshold, min_total_cost)``.
        """
        fn_cost = cost_fn if cost_fn is not None else self.default_cost_fn
        fp_cost = cost_fp if cost_fp is not None else self.default_cost_fp

        y_true_arr = np.asarray(y_true, dtype=int)
        y_prob_arr = np.asarray(y_prob, dtype=float)

        threshold_grid = np.linspace(0.01, 0.99, num_thresholds)
        costs = []

        for t in threshold_grid:
            y_pred = (y_prob_arr >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
            total_cost = (fn * fn_cost) + (fp * fp_cost)
            costs.append(total_cost)

        best_idx = int(np.argmin(costs))
        best_threshold = float(threshold_grid[best_idx])
        min_cost = float(costs[best_idx])

        logger.info(
            "Cost-optimal threshold: %.4f (min_cost=$%.2f | cost_fn=$%.2f, cost_fp=$%.2f)",
            best_threshold,
            min_cost,
            fn_cost,
            fp_cost,
        )
        return best_threshold, min_cost

    def evaluate_threshold(
        self,
        y_true: pd.Series | np.ndarray,
        y_prob: np.ndarray,
        threshold: float,
        cost_fn: float | None = None,
        cost_fp: float | None = None,
    ) -> ThresholdEvaluation:
        """Compute full metrics and financial impact at a specific threshold."""
        fn_cost = cost_fn if cost_fn is not None else self.default_cost_fn
        fp_cost = cost_fp if cost_fp is not None else self.default_cost_fp

        y_true_arr = np.asarray(y_true, dtype=int)
        y_prob_arr = np.asarray(y_prob, dtype=float)
        y_pred = (y_prob_arr >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
        prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
        rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
        f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))
        total_cost = float((fn * fn_cost) + (fp * fp_cost))

        return ThresholdEvaluation(
            threshold=threshold,
            precision=prec,
            recall=rec,
            f1=f1,
            total_cost=total_cost,
            true_positives=int(tp),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn),
        )

    def compare_policies(
        self,
        y_true: pd.Series | np.ndarray,
        y_prob: np.ndarray,
        cost_fn: float | None = None,
        cost_fp: float | None = None,
    ) -> pd.DataFrame:
        """Compare standard default threshold (0.50) with optimized risk policies."""
        best_f1_t, _ = self.find_best_f1_threshold(y_true, y_prob)
        target_p80_t, _ = self.find_target_precision_threshold(y_true, y_prob, min_precision=0.80)
        target_p90_t, _ = self.find_target_precision_threshold(y_true, y_prob, min_precision=0.90)
        cost_t, _ = self.find_cost_optimal_threshold(y_true, y_prob, cost_fn, cost_fp)

        policies = [
            ("Standard Baseline (0.50)", 0.50),
            ("Max F1 Policy", best_f1_t),
            ("Target Precision >=80%", target_p80_t),
            ("Target Precision >=90%", target_p90_t),
            ("Cost-Optimal Policy", cost_t),
        ]

        rows = []
        for policy_name, thresh in policies:
            eval_res = self.evaluate_threshold(y_true, y_prob, thresh, cost_fn, cost_fp)
            d = eval_res.to_dict()
            d["Policy"] = policy_name
            rows.append(d)

        df = pd.DataFrame(rows).set_index("Policy")
        return df
