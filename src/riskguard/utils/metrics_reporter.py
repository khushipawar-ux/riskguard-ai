"""
riskguard.utils.metrics_reporter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Formats and persists evaluation results for human consumption.

Keeps all presentation logic out of :mod:`riskguard.models.evaluator`,
which is purely computational.

Responsibilities
----------------
* Render :class:`~riskguard.models.evaluator.EvaluationResult` as a
  markdown table.
* Save the markdown report to the configured output directory.
* Plot and save the threshold decision curve (precision / recall / F1 vs
  threshold) for risk-policy analysis.

Usage::

    from riskguard.utils.metrics_reporter import MetricsReporter
    from riskguard.models.evaluator import ModelEvaluator

    evaluator = ModelEvaluator()
    result = evaluator.compute_metrics(y_test, y_prob)
    curve  = evaluator.threshold_curve(y_test, y_prob)

    reporter = MetricsReporter(output_dir=pathlib.Path("outputs/phase2"))
    reporter.save_markdown(result, model_name="Logistic Regression Baseline")
    reporter.save_threshold_chart(curve, model_name="Logistic Regression Baseline")
"""

from __future__ import annotations

import pathlib
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd

from riskguard.models.evaluator import EvaluationResult
from riskguard.utils.logging import get_logger
from riskguard.utils.plotting import (
    BLUE,
    FG,
    GRID_CLR,
    RED,
    apply_theme,
    save_figure,
)

logger = get_logger(__name__)

_CHART_FILENAME: str = "baseline_threshold_curve.png"
_REPORT_FILENAME: str = "baseline_metrics.md"

# Ensure theme is applied once at module load.
apply_theme()


class MetricsReporter:
    """Format and persist evaluation metrics for Phase 2.

    Args:
        output_dir: Directory where the markdown report and chart will be
                    written.  Created automatically if it does not exist.
    """

    def __init__(self, output_dir: pathlib.Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def save_markdown(
        self,
        result: EvaluationResult,
        model_name: str = "Baseline",
        extra_notes: str = "",
    ) -> pathlib.Path:
        """Write *result* as a markdown table to ``baseline_metrics.md``.

        Args:
            result:      Populated :class:`~riskguard.models.evaluator.EvaluationResult`.
            model_name:  Human-readable model name shown in the report header.
            extra_notes: Optional free-text appended at the bottom.

        Returns:
            Path of the written file.
        """
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        cm = result.confusion

        # Guard against confusion matrices that don't have 2×2 shape.
        try:
            tn, fp = int(cm[0][0]), int(cm[0][1])
            fn, tp = int(cm[1][0]), int(cm[1][1])
        except (IndexError, TypeError):
            tn = fp = fn = tp = 0

        recall_80p_str = (
            f"{result.recall_at_80p:.4f}"
            if result.recall_at_80p is not None
            else "N/A (precision target unachievable)"
        )

        md = f"""# RiskGuard AI — Phase 2 Baseline Metrics

**Model:** {model_name}
**Generated:** {timestamp}

> ⚠️ Accuracy is intentionally excluded — on a 1:578 imbalanced dataset
> it is a misleading metric. PR-AUC is the primary optimisation target.

## Evaluation Results

| Metric | Value |
|--------|-------|
| **PR-AUC** (primary) | **{result.pr_auc:.4f}** |
| F1 @ threshold {result.operating_threshold:.2f} | {result.f1_at_threshold:.4f} |
| Best F1 | {result.best_f1:.4f} |
| Best-F1 threshold | {result.best_threshold:.4f} |
| Recall @ ≥80 % precision | {recall_80p_str} |

## Confusion Matrix (threshold = {result.operating_threshold:.2f})

|  | Predicted: Legit | Predicted: Fraud |
|--|-----------------|-----------------|
| **Actual: Legit** | TN = {tn:,} | FP = {fp:,} |
| **Actual: Fraud** | FN = {fn:,} | TP = {tp:,} |

- **False Positive Rate:** {fp / (fp + tn) * 100:.2f}% of legit transactions incorrectly flagged
- **True Positive Rate (Recall):** {tp / (tp + fn) * 100:.2f}% of fraud caught

## Interpretation

- **PR-AUC** close to 1.0 means the model achieves high precision *and*
  high recall simultaneously across all thresholds.
- The **operating threshold** ({result.operating_threshold:.2f}) is the default; see the
  threshold curve chart for alternative risk policies.
- **Recall @ ≥80 % precision** answers: "If we review only transactions the
  model is 80 %+ confident are fraud, how much fraud do we catch?"
"""

        if extra_notes:
            md += f"\n## Notes\n\n{extra_notes}\n"

        path = self.output_dir / _REPORT_FILENAME
        path.write_text(md, encoding="utf-8")
        logger.info("Metrics report written: %s", path.resolve())
        return path

    def save_threshold_chart(
        self,
        curve: pd.DataFrame,
        model_name: str = "Baseline",
    ) -> pathlib.Path:
        """Plot the precision / recall / F1 vs threshold curve and save it.

        This chart is the visual representation of the risk policy:
        the analyst can pick the threshold that matches their acceptable
        false-positive rate.

        Args:
            curve:       DataFrame from
                         :meth:`~riskguard.models.evaluator.ModelEvaluator.threshold_curve`.
            model_name:  Used in the chart title.

        Returns:
            Path of the saved PNG.
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            curve["threshold"],
            curve["precision"],
            color=BLUE,
            linewidth=1.8,
            label="Precision",
        )
        ax.plot(
            curve["threshold"],
            curve["recall"],
            color=RED,
            linewidth=1.8,
            label="Recall",
        )
        ax.plot(
            curve["threshold"],
            curve["f1"],
            color="#A8D8A8",
            linewidth=2.2,
            linestyle="--",
            label="F1",
        )

        # Mark the F1-maximising threshold.
        best_idx = curve["f1"].idxmax()
        best_t = float(curve.loc[best_idx, "threshold"])
        best_f1 = float(curve.loc[best_idx, "f1"])
        ax.axvline(
            best_t,
            color=FG,
            linewidth=0.8,
            linestyle=":",
            alpha=0.7,
            label=f"Best-F1 threshold={best_t:.3f}",
        )
        ax.annotate(
            f"F1={best_f1:.3f}",
            xy=(best_t, best_f1),
            xytext=(best_t + 0.05, best_f1 - 0.08),
            color=FG,
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color=FG, lw=0.8),
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Decision Threshold", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(
            f"{model_name} — Threshold Decision Curve\n"
            "(Precision / Recall / F1 across all operating thresholds)",
            fontsize=12,
        )
        ax.legend(loc="center right", fontsize=9)
        ax.grid(True, color=GRID_CLR, linewidth=0.5)

        path = self.output_dir / _CHART_FILENAME
        save_figure(fig, path)
        logger.info("Threshold curve chart saved: %s", path.resolve())
        return path
