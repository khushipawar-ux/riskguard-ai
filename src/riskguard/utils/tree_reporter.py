"""
riskguard.utils.tree_reporter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Markdown reporting and visual chart generation for Phase 4 (Tree Models & Threshold Tuning).
"""

from __future__ import annotations

import pathlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from riskguard.models.evaluator import EvaluationResult
from riskguard.models.trainer import CVResult
from riskguard.utils.logging import get_logger
from riskguard.utils.plotting import BG, BLUE, FG, GRID_CLR, RED, apply_theme, save_figure

logger = get_logger(__name__)


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    """Format DataFrame as markdown table without depending on tabulate."""
    headers = [df.index.name or ""] + list(df.columns)
    col_aligns = [":---"] + [":---:" for _ in df.columns]
    header_line = "| " + " | ".join(headers) + " |"
    align_line = "| " + " | ".join(col_aligns) + " |"

    rows = []
    for idx, row in df.iterrows():
        row_vals = [str(idx)] + [str(v) for v in row.values]
        rows.append("| " + " | ".join(row_vals) + " |")

    return "\n".join([header_line, align_line] + rows)


class TreeReporter:
    """Generates markdown reports and comparative visualisations for Phase 4."""

    def __init__(self, output_dir: str | pathlib.Path) -> None:
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        apply_theme()

    def save_markdown_report(
        self,
        cv_results: list[CVResult],
        test_evals: dict[str, EvaluationResult],
        policy_df: pd.DataFrame,
        best_model_name: str,
    ) -> pathlib.Path:
        """Write Phase 4 comprehensive model comparison report to disk."""
        filepath = self.output_dir / "model_comparison.md"

        # Build CV table
        cv_rows = []
        for cv in cv_results:
            cv_rows.append(
                f"| **{cv.model_name}** | {cv.mean_pr_auc:.4f} ± {cv.std_pr_auc:.4f} | {cv.mean_best_f1:.4f} |"
            )
        cv_table = "\n".join(cv_rows)

        # Build Test table
        test_rows = []
        for name, er in test_evals.items():
            r80 = f"{er.recall_at_80p:.4f}" if er.recall_at_80p is not None else "N/A"
            cm = er.confusion
            cm_str = f"TN={cm[0][0]:,} FP={cm[0][1]:,} FN={cm[1][0]:,} TP={cm[1][1]:,}"
            test_rows.append(
                f"| **{name}** | {er.pr_auc:.4f} | {er.best_f1:.4f} | {er.f1_at_threshold:.4f} | {r80} | {cm_str} |"
            )
        test_table = "\n".join(test_rows)

        # Policy table
        policy_table = _df_to_markdown_table(policy_df)

        content = f"""# Phase 4 — Stronger Tree Models & Decision Threshold Tuning

## Executive Summary
This phase evaluates gradient boosted decision tree architectures (**XGBoost** and **LightGBM**) against the **Logistic Regression Baseline**, applying **Stratified 5-Fold Cross-Validation** and explicit **decision threshold optimization**.

- **Selected Best Model:** `{best_model_name}`
- **Primary Metric:** Precision-Recall AUC (PR-AUC)
- **Imbalance Handling:** Built-in positive class scale weighting (`scale_pos_weight`)

---

## 1. Stratified 5-Fold Cross-Validation Results

| Model Architecture | 5-Fold Mean PR-AUC | 5-Fold Mean Best F1 |
|:---|:---:|:---:|
{cv_table}

---

## 2. Held-Out Test Set Performance

| Model | PR-AUC (Primary) | Best F1 | F1 @ t=0.50 | Recall @ ≥80% Prec | Confusion Matrix (t=0.50) |
|:---|:---:|:---:|:---:|:---:|:---|
{test_table}

---

## 3. Decision Threshold & Risk Policy Comparison (`{best_model_name}`)

The default threshold of 0.50 is suboptimal under extreme class imbalance. Below is the performance under distinct business policies:

{policy_table}

---

## 4. Key Takeaways
1. **Tree Models vs Linear Baseline:** Tree-based gradient boosting models capture non-linear relationships and interactions between anonymized PCA features without requiring explicit interaction term engineering.
2. **Threshold Tuning as Risk Policy:** Calibrating the operating threshold directly controls the precision/recall trade-off based on business operational constraints (e.g. analyst capacity vs fraud loss exposure).
"""
        filepath.write_text(content.strip(), encoding="utf-8")
        logger.info("Saved Phase 4 markdown report to %s", filepath.resolve())
        return filepath

    def save_pr_curves(
        self,
        curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
    ) -> pathlib.Path:
        """Plot and save comparative Precision-Recall curves using dark theme."""
        filepath = self.output_dir / "pr_curves_comparison.png"
        fig, ax = plt.subplots(figsize=(9, 6))

        palette = [FG, BLUE, "#81C784", "#FFB74D"]
        for i, (name, (y_true, y_prob, pr_auc)) in enumerate(curves.items()):
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            color = palette[i % len(palette)]
            ax.plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.4f})", color=color, linewidth=2.2)

        ax.set_xlabel("Recall (Fraction of Fraud Caught)", fontsize=11, fontweight="bold", color=FG)
        ax.set_ylabel("Precision (True Fraud / Total Flagged)", fontsize=11, fontweight="bold", color=FG)
        ax.set_title("Precision-Recall Curve Comparison (Held-Out Test Set)", fontsize=13, fontweight="bold", color=FG, pad=12)
        ax.legend(loc="lower left", frameon=True, fontsize=10)
        ax.set_xlim([0.0, 1.02])
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, linestyle="--", alpha=0.5, color=GRID_CLR)

        save_figure(fig, filepath)
        logger.info("Saved PR curves comparison to %s", filepath.resolve())
        return filepath

    def save_policy_chart(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        best_threshold: float,
        model_name: str,
    ) -> pathlib.Path:
        """Plot precision, recall, and F1 across decision thresholds using dark theme."""
        filepath = self.output_dir / "threshold_policy_curves.png"
        prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
        p = prec[:-1]
        r = rec[:-1]
        denom = p + r
        with np.errstate(invalid="ignore"):
            f1 = np.where(denom > 0, 2 * p * r / denom, 0.0)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(thresholds, p, label="Precision", color=BLUE, linewidth=2.0)
        ax.plot(thresholds, r, label="Recall", color="#81C784", linewidth=2.0)
        ax.plot(thresholds, f1, label="F1 Score", color="#FFB74D", linewidth=2.2)

        ax.axvline(
            best_threshold,
            color=RED,
            linestyle="--",
            linewidth=1.8,
            label=f"Optimal Threshold = {best_threshold:.3f}",
        )
        ax.axvline(
            0.50,
            color="#8B949E",
            linestyle=":",
            linewidth=1.5,
            label="Default Threshold = 0.50",
        )

        ax.set_xlabel("Decision Cut-Off Threshold", fontsize=11, fontweight="bold", color=FG)
        ax.set_ylabel("Metric Score", fontsize=11, fontweight="bold", color=FG)
        ax.set_title(f"Risk Policy Threshold Tuning — {model_name}", fontsize=13, fontweight="bold", color=FG, pad=12)
        ax.legend(loc="best", frameon=True, fontsize=10)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, linestyle="--", alpha=0.5, color=GRID_CLR)

        save_figure(fig, filepath)
        logger.info("Saved threshold policy chart to %s", filepath.resolve())
        return filepath
