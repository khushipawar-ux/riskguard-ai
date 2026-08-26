"""
riskguard.utils.comparison_reporter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Formats and persists Phase 3 imbalance-comparison results.

Separates presentation from computation — all charting and markdown
generation lives here so :mod:`riskguard.models.comparison` stays focused
on evaluation logic.

Responsibilities
----------------
* Render :class:`~riskguard.models.comparison.ComparisonResult` as a
  rich markdown table with winner highlighted.
* Save a bar-chart comparing PR-AUC across all strategies.
* Save per-strategy threshold curves on a shared axes for direct
  visual comparison.

Usage::

    from riskguard.utils.comparison_reporter import ComparisonReporter
    from riskguard.models.evaluator import ModelEvaluator

    reporter = ComparisonReporter(output_dir=pathlib.Path("outputs/phase3"))
    reporter.save_markdown(result)
    reporter.save_prauc_bar_chart(result)
    reporter.save_threshold_curves(result, evaluator, X_test_t, y_test)
"""

from __future__ import annotations

import pathlib
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from riskguard.models.comparison import ComparisonResult
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

apply_theme()

# Palette for multi-strategy charts (one colour per strategy).
_STRATEGY_COLOURS: list[str] = ["#4FC3F7", "#EF5350", "#A8D8A8", "#FFA726"]


class ComparisonReporter:
    """Save Phase 3 comparison results as markdown and charts.

    Args:
        output_dir: Directory where all outputs are written.
    """

    def __init__(self, output_dir: pathlib.Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def save_markdown(self, result: ComparisonResult) -> pathlib.Path:
        """Write the comparison summary to ``imbalance_comparison.md``.

        Args:
            result: :class:`~riskguard.models.comparison.ComparisonResult`
                    from :class:`~riskguard.models.comparison.ImbalanceComparison`.

        Returns:
            Path of the written file.
        """
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        winner_name = result.winner.name

        # Build markdown table rows.
        table_rows = ""
        for sr in result.strategy_results:
            er = sr.eval_result
            recall_str = (
                f"{er.recall_at_80p:.4f}"
                if er.recall_at_80p is not None
                else "N/A"
            )
            win_marker = " ✅ **Winner**" if sr.strategy.name == winner_name else ""
            table_rows += (
                f"| {sr.strategy.name}{win_marker} "
                f"| **{er.pr_auc:.4f}**"
                f"| {er.best_f1:.4f}"
                f"| {er.f1_at_threshold:.4f}"
                f"| {recall_str}"
                f"| {sr.train_sample_count:,}"
                f"| {sr.train_fraud_count:,} |\n"
            )

        md = f"""# RiskGuard AI — Phase 3: Imbalance Strategy Comparison

**Generated:** {timestamp}

> All strategies are evaluated on the **same held-out test set** (20% of data).
> The test set is never resampled — only the training fold is modified.
> Accuracy is excluded; PR-AUC is the primary selection metric.

## Summary Table

| Strategy | PR-AUC | Best F1 | F1 @ t=0.50 | Recall @ >=80% prec | Train samples | Train fraud |
|----------|--------|---------|-------------|---------------------|--------------|-------------|
{table_rows}

## Winner: `{winner_name}`

The winning strategy is applied to the final model training in subsequent phases.

## Strategy Notes

### class_weight=balanced
Instructs the classifier loss function to penalise fraud misclassification
proportionally to the 1:578 imbalance ratio.  No data is added or removed.
Computationally cheapest and zero risk of synthetic data artefacts.

### SMOTE oversampling
Generates synthetic fraud examples by interpolating between real fraud
neighbours in feature space.  Expands the training set. Risk: synthetic
samples may not reflect real fraud patterns.

### random undersampling
Randomly discards legit transactions until the target ratio is reached.
Shrinks the training set.  Risk: discarding real patterns from the majority
class.

## Phase 4 Implication

The winner's resampling strategy feeds directly into the XGBoost / LightGBM
hyperparameter search in Phase 4.
"""

        path = self.output_dir / "imbalance_comparison.md"
        path.write_text(md, encoding="utf-8")
        logger.info("Comparison report written: %s", path.resolve())
        return path

    def save_prauc_bar_chart(self, result: ComparisonResult) -> pathlib.Path:
        """Save a horizontal bar chart comparing PR-AUC across strategies.

        Args:
            result: Comparison result (strategies already ranked by PR-AUC).

        Returns:
            Path of the saved PNG.
        """
        strategies = [sr.strategy.name for sr in result.strategy_results]
        pr_aucs = [sr.eval_result.pr_auc for sr in result.strategy_results]
        colours = [
            _STRATEGY_COLOURS[i % len(_STRATEGY_COLOURS)]
            for i in range(len(strategies))
        ]

        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.barh(strategies, pr_aucs, color=colours, edgecolor=GRID_CLR, height=0.5)

        # Annotate bar values.
        for bar, val in zip(bars, pr_aucs):
            ax.text(
                val + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                color=FG,
                fontsize=10,
            )

        # Winner marker.
        winner_name = result.winner.name
        for i, name in enumerate(strategies):
            if name == winner_name:
                ax.text(
                    0.002,
                    i,
                    "WINNER",
                    va="center",
                    ha="left",
                    color="#A8D8A8",
                    fontsize=8,
                    fontweight="bold",
                )

        ax.set_xlim(0, max(pr_aucs) * 1.20)
        ax.set_xlabel("Precision-Recall AUC (higher is better)", fontsize=11)
        ax.set_title(
            "Phase 3 — Imbalance Strategy Comparison\n"
            "(same train/test split; test set never resampled)",
            fontsize=12,
        )
        ax.grid(True, axis="x", color=GRID_CLR, linewidth=0.5)
        ax.invert_yaxis()

        path = self.output_dir / "imbalance_prauc_comparison.png"
        save_figure(fig, path)
        logger.info("PR-AUC bar chart saved: %s", path.resolve())
        return path

    def save_threshold_curves(
        self,
        result: ComparisonResult,
        threshold_curves: dict[str, pd.DataFrame],
    ) -> pathlib.Path:
        """Plot precision/recall/F1 curves for all strategies on shared axes.

        Args:
            result:           Comparison result.
            threshold_curves: Mapping of strategy name -> threshold DataFrame
                              (from :meth:`~riskguard.models.evaluator.ModelEvaluator.threshold_curve`).

        Returns:
            Path of the saved PNG.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        metrics = ["precision", "recall", "f1"]
        titles = ["Precision vs Threshold", "Recall vs Threshold", "F1 vs Threshold"]

        for ax, metric, title in zip(axes, metrics, titles):
            for i, sr in enumerate(result.strategy_results):
                name = sr.strategy.name
                if name not in threshold_curves:
                    continue
                curve = threshold_curves[name]
                colour = _STRATEGY_COLOURS[i % len(_STRATEGY_COLOURS)]
                lw = 2.5 if name == result.winner.name else 1.4
                ls = "-" if name == result.winner.name else "--"
                label = f"{name} {'(winner)' if name == result.winner.name else ''}"
                ax.plot(
                    curve["threshold"],
                    curve[metric],
                    color=colour,
                    linewidth=lw,
                    linestyle=ls,
                    label=label.strip(),
                    alpha=0.9,
                )

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Threshold", fontsize=10)
            ax.set_title(title, fontsize=10)
            ax.grid(True, color=GRID_CLR, linewidth=0.5)

        axes[0].set_ylabel("Score", fontsize=10)
        axes[2].legend(loc="upper right", fontsize=8)
        fig.suptitle(
            "Phase 3 — Threshold Decision Curves by Strategy",
            fontsize=13,
        )
        plt.tight_layout()

        path = self.output_dir / "imbalance_threshold_curves.png"
        save_figure(fig, path)
        logger.info("Threshold curves chart saved: %s", path.resolve())
        return path
