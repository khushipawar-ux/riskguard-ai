"""
riskguard.utils.explainability_reporter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Markdown reporting and visual chart generation for Phase 5 (SHAP Interpretability).
"""

from __future__ import annotations

import pathlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from riskguard.explainability.shap_analyzer import TransactionExplanation
from riskguard.utils.logging import get_logger
from riskguard.utils.plotting import BG, BLUE, FG, GRID_CLR, RED, apply_theme, save_figure

logger = get_logger(__name__)


class ExplainabilityReporter:
    """Generates markdown reports and SHAP visualisations for Phase 5."""

    def __init__(self, output_dir: str | pathlib.Path) -> None:
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        apply_theme()

    def save_markdown_report(
        self,
        global_df: pd.DataFrame,
        sample_explanations: list[TransactionExplanation],
        model_name: str,
    ) -> pathlib.Path:
        """Write Phase 5 interpretability report to disk."""
        filepath = self.output_dir / "interpretability_report.md"

        top10_table = global_df.head(10).to_markdown(index=False)

        # Build sample transaction breakdowns
        tx_sections = []
        for i, exp in enumerate(sample_explanations, 1):
            drivers = "\n".join(
                [f"  - **{rd.feature_name}** (`val={rd.feature_value:.2f}`): +{rd.shap_value:.4f} SHAP impact ({rd.description})" for rd in exp.risk_drivers]
            )
            protect = "\n".join(
                [f"  - **{pf.feature_name}** (`val={pf.feature_value:.2f}`): {pf.shap_value:.4f} SHAP impact ({pf.description})" for pf in exp.protective_factors]
            )

            status = "🚨 **FLAGGED AS HIGH RISK / FRAUD**" if exp.is_flagged else "✅ **CLEARED / LOW RISK**"

            tx_sections.append(
                f"### Case #{i} — {status}\n\n"
                f"- **Model Fraud Probability:** `{exp.fraud_probability * 100:.2f}%`\n"
                f"- **Model Base Value:** `{exp.base_value:.4f}`\n\n"
                f"**Top Risk Drivers (Pushing Score Higher):**\n"
                f"{drivers if drivers else '  - None'}\n\n"
                f"**Top Protective Factors (Pushing Score Lower):**\n"
                f"{protect if protect else '  - None'}\n\n"
                f"> **Risk Analyst Summary:**\n> {exp.human_readable_summary}\n"
            )

        tx_body = "\n---\n\n".join(tx_sections)

        content = f"""# Phase 5 — SHAP Interpretability & Risk Explainability

## Executive Summary
A black-box prediction is insufficient for enterprise risk operations. Risk analysts need to know **why** a transaction was flagged before initiating a block or customer outreach.

This phase implements **SHAP (SHapley Additive exPlanations)** on the `{model_name}` to produce:
1. **Global feature importance** across thousands of transactions.
2. **Local per-transaction breakdowns** decomposing any individual score into contributing drivers.

---

## 1. Global Feature Importance (Top 10 Drivers)

{top10_table}

---

## 2. Sample Case Explanations

{tx_body}

---

## 3. Risk Team Decision Workflow
1. **Automated Flag:** Model scores transaction $\ge$ optimal threshold.
2. **Instant Reason Extraction:** `ShapAnalyzer` computes top risk drivers in $<5$ ms.
3. **Analyst Action:** Analyst evaluates human-readable drivers to verify patterns (e.g. extreme PCA deviations, abnormal timing) and takes action.
"""
        filepath.write_text(content.strip(), encoding="utf-8")
        logger.info("Saved Phase 5 markdown report to %s", filepath.resolve())
        return filepath

    def save_global_importance_chart(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 15,
    ) -> pathlib.Path:
        """Plot and save horizontal bar chart of global SHAP feature importance in dark theme."""
        filepath = self.output_dir / "shap_global_importance.png"
        top_df = importance_df.head(top_n).sort_values("Mean_Abs_SHAP", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.barh(top_df["Feature"], top_df["Mean_Abs_SHAP"], color=BLUE, edgecolor="#1E88E5", height=0.65)

        # Add percentage labels
        for bar, pct in zip(bars, top_df["Importance_Pct"]):
            width = bar.get_width()
            ax.text(
                width + (max(top_df["Mean_Abs_SHAP"]) * 0.01),
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%",
                va="center",
                fontsize=9,
                color=FG,
                fontweight="bold",
            )

        ax.set_xlabel("Mean Absolute SHAP Value (Average Impact on Model Output)", fontsize=11, fontweight="bold", color=FG)
        ax.set_ylabel("Feature", fontsize=11, fontweight="bold", color=FG)
        ax.set_title(f"Top {top_n} Global Features by SHAP Importance", fontsize=13, fontweight="bold", color=FG, pad=12)
        ax.grid(True, axis="x", linestyle="--", alpha=0.5, color=GRID_CLR)

        save_figure(fig, filepath)
        logger.info("Saved global SHAP importance chart to %s", filepath.resolve())
        return filepath

    def save_local_waterfall_chart(
        self,
        explanation: TransactionExplanation,
        case_id: int = 1,
    ) -> pathlib.Path:
        """Plot local feature contribution breakdown for a single transaction in dark theme."""
        filepath = self.output_dir / f"shap_waterfall_case_{case_id}.png"

        all_factors = explanation.risk_drivers + explanation.protective_factors
        all_factors.sort(key=lambda f: abs(f.shap_value), reverse=False)

        names = [f.feature_name for f in all_factors]
        values = [f.shap_value for f in all_factors]
        colors = [RED if v > 0 else "#81C784" for v in values]

        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.barh(names, values, color=colors, height=0.6)

        ax.axvline(0, color=FG, linestyle="-", linewidth=1.0)
        ax.set_xlabel("SHAP Impact on Fraud Score", fontsize=11, fontweight="bold", color=FG)
        ax.set_ylabel("Feature", fontsize=11, fontweight="bold", color=FG)

        title_status = "FLAGGED FRAUD" if explanation.is_flagged else "CLEARED LEGIT"
        ax.set_title(
            f"Case #{case_id} [{title_status}] — Probability: {explanation.fraud_probability * 100:.1f}%\n"
            f"Red = Pushes Fraud Risk Higher | Green = Pushes Legit",
            fontsize=12,
            fontweight="bold",
            color=FG,
            pad=12,
        )
        ax.grid(True, axis="x", linestyle="--", alpha=0.5, color=GRID_CLR)

        save_figure(fig, filepath)
        logger.info("Saved local SHAP chart to %s", filepath.resolve())
        return filepath
