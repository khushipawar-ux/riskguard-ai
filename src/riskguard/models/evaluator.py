"""
riskguard.models.evaluator
~~~~~~~~~~~~~~~~~~~~~~~~~~
Metric computation for the fraud detection use case (Phase 2+).

**Accuracy is intentionally absent** — on a 1:578 imbalanced dataset a
model predicting "always legit" achieves 99.83 % accuracy while catching
zero fraud.

Metrics computed here:

* **Precision-Recall AUC** (primary) — area under the PR curve.
* **F1 score** — harmonic mean of precision and recall at the operating
  threshold.
* **Best-threshold F1** — F1 at the threshold that maximises F1 on the
  given data.
* **Recall at fixed precision** — business-relevant query:
  "What is our recall if we require at least X % precision?"
* **Confusion matrix** — raw true/false positive/negative counts.
* **Threshold curve** — DataFrame of precision, recall, F1 at every
  decision threshold; enables the analyst to choose a risk policy.

Usage::

    from riskguard.models.evaluator import ModelEvaluator

    evaluator = ModelEvaluator()
    result = evaluator.compute_metrics(y_test, y_prob)
    print(result)
    recall = evaluator.recall_at_precision(y_test, y_prob, min_precision=0.80)
    curve  = evaluator.threshold_curve(y_test, y_prob)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)

from riskguard.utils.logging import get_logger

logger = get_logger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class EvaluationResult:
    """Aggregated evaluation metrics for a binary fraud classifier.

    Attributes:
        pr_auc:           Precision-Recall area under the curve (primary
                          metric; higher is better).
        f1_at_threshold:  F1 score at *operating_threshold*.
        best_f1:          Maximum F1 achievable across all thresholds.
        best_threshold:   Probability cut-off that yields *best_f1*.
        recall_at_80p:    Recall when minimum precision is 80 %.
                          ``None`` if 80 % precision is unachievable.
        operating_threshold: The threshold used for *f1_at_threshold*
                          and the confusion matrix (default 0.5).
        confusion:        2-D list [[TN, FP], [FN, TP]].
    """

    pr_auc: float
    f1_at_threshold: float
    best_f1: float
    best_threshold: float
    recall_at_80p: float | None
    operating_threshold: float
    confusion: list[list[int]]

    def __str__(self) -> str:  # pragma: no cover
        recall_str = (
            f"{self.recall_at_80p:.4f}" if self.recall_at_80p is not None else "N/A"
        )
        return (
            f"PR-AUC              : {self.pr_auc:.4f}\n"
            f"F1 @ {self.operating_threshold:.2f} threshold : {self.f1_at_threshold:.4f}\n"
            f"Best F1             : {self.best_f1:.4f}  (threshold={self.best_threshold:.4f})\n"
            f"Recall @ ≥80% prec  : {recall_str}\n"
            f"Confusion matrix    : TN={self.confusion[0][0]:,}  FP={self.confusion[0][1]:,}"
            f"  FN={self.confusion[1][0]:,}  TP={self.confusion[1][1]:,}"
        )


# ── Evaluator class ───────────────────────────────────────────────────────────


class ModelEvaluator:
    """Compute fraud-detection-specific evaluation metrics.

    All methods are stateless — pass ``y_true`` and ``y_prob`` each time.
    This makes the evaluator reusable across models without re-instantiation.
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def compute_metrics(
        self,
        y_true: pd.Series | np.ndarray,
        y_prob: np.ndarray,
        operating_threshold: float = 0.5,
    ) -> EvaluationResult:
        """Compute the full evaluation metric suite.

        Args:
            y_true:              Binary ground-truth labels (0 = legit,
                                 1 = fraud).
            y_prob:              Predicted fraud probabilities in [0, 1].
            operating_threshold: Decision cut-off for the confusion matrix
                                 and ``f1_at_threshold`` (default 0.5).

        Returns:
            :class:`EvaluationResult` populated with all metrics.
        """
        y_true_arr = np.asarray(y_true, dtype=int)
        y_prob_arr = np.asarray(y_prob, dtype=float)

        pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))

        # Threshold curve for best-F1 search.
        precision_arr, recall_arr, thresholds = precision_recall_curve(
            y_true_arr, y_prob_arr
        )
        f1_arr = self._f1_from_pr(precision_arr[:-1], recall_arr[:-1])
        best_idx = int(np.argmax(f1_arr)) if len(f1_arr) > 0 else 0
        best_f1 = float(f1_arr[best_idx]) if len(f1_arr) > 0 else 0.0
        best_threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5

        # F1 at the chosen operating threshold.
        y_pred = (y_prob_arr >= operating_threshold).astype(int)
        f1_at_threshold = float(f1_score(y_true_arr, y_pred, zero_division=0))

        # Recall at ≥80 % precision.
        recall_80p = self.recall_at_precision(y_true_arr, y_prob_arr, min_precision=0.80)

        # Confusion matrix.
        cm = confusion_matrix(y_true_arr, y_pred).tolist()

        result = EvaluationResult(
            pr_auc=pr_auc,
            f1_at_threshold=f1_at_threshold,
            best_f1=best_f1,
            best_threshold=best_threshold,
            recall_at_80p=recall_80p,
            operating_threshold=operating_threshold,
            confusion=cm,
        )

        logger.info(
            "Evaluation — PR-AUC=%.4f | best F1=%.4f (t=%.3f) | "
            "F1@%.2f=%.4f | Recall@80%%prec=%s",
            pr_auc,
            best_f1,
            best_threshold,
            operating_threshold,
            f1_at_threshold,
            f"{recall_80p:.4f}" if recall_80p is not None else "N/A",
        )
        return result

    def recall_at_precision(
        self,
        y_true: pd.Series | np.ndarray,
        y_prob: np.ndarray,
        min_precision: float = 0.80,
    ) -> float | None:
        """Return the recall achievable at *min_precision* or higher.

        Answers the business question: "If we require that at least
        ``min_precision * 100`` % of flagged transactions are genuine fraud,
        what fraction of all fraud do we catch?"

        Args:
            y_true:        Binary ground-truth labels.
            y_prob:        Predicted fraud probabilities.
            min_precision: Minimum acceptable precision threshold (0–1).

        Returns:
            The maximum recall at which precision ≥ *min_precision*, or
            ``None`` if the precision target is never met.
        """
        y_true_arr = np.asarray(y_true, dtype=int)
        y_prob_arr = np.asarray(y_prob, dtype=float)

        precision_arr, recall_arr, _ = precision_recall_curve(
            y_true_arr, y_prob_arr
        )

        # Mask where precision meets the target (last element is always 1.0
        # precision with recall=0; sklearn convention).
        mask = precision_arr >= min_precision
        if not mask.any():
            return None

        max_recall = float(recall_arr[mask].max())
        return max_recall

    def threshold_curve(
        self,
        y_true: pd.Series | np.ndarray,
        y_prob: np.ndarray,
    ) -> pd.DataFrame:
        """Return precision, recall, and F1 at every decision threshold.

        Useful for visualising the cost trade-off: the analyst can pick a
        threshold based on their acceptable false-positive rate.

        Args:
            y_true: Binary ground-truth labels.
            y_prob: Predicted fraud probabilities.

        Returns:
            :class:`pandas.DataFrame` with columns
            ``["threshold", "precision", "recall", "f1"]``, one row per
            threshold value (sorted ascending by threshold).
        """
        y_true_arr = np.asarray(y_true, dtype=int)
        y_prob_arr = np.asarray(y_prob, dtype=float)

        precision_arr, recall_arr, thresholds = precision_recall_curve(
            y_true_arr, y_prob_arr
        )

        # sklearn returns len(thresholds) == len(precision) - 1.
        prec = precision_arr[:-1]
        rec = recall_arr[:-1]
        f1_vals = self._f1_from_pr(prec, rec)

        df = pd.DataFrame(
            {
                "threshold": thresholds,
                "precision": prec,
                "recall": rec,
                "f1": f1_vals,
            }
        ).sort_values("threshold").reset_index(drop=True)

        return df

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _f1_from_pr(
        precision: np.ndarray, recall: np.ndarray
    ) -> np.ndarray:
        """Compute F1 from precision and recall arrays, handling zero denominators."""
        denom = precision + recall
        # Avoid division by zero.
        with np.errstate(invalid="ignore"):
            f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
        return f1.astype(float)
