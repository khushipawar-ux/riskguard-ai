#!/usr/bin/env python
"""
scripts/run_explainability.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Entry point for Phase 5: SHAP Interpretability & Risk Explainability.

Pipeline:
1. Load & validate dataset.
2. Stratified train/test split.
3. Load or train best model (XGBoostFraudModel).
4. Initialize ShapAnalyzer with background sample distribution.
5. Compute global feature importance via mean absolute SHAP values.
6. Score test transactions and identify flagged high-risk cases.
7. Generate per-transaction risk breakdowns (risk drivers & protective factors).
8. Save global charts, local waterfall charts, and markdown reports to ``outputs/phase5/``.

Usage::

    python scripts/run_explainability.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from riskguard.config import Settings
from riskguard.data.loader import DataLoadError, DataLoader
from riskguard.data.validator import validate_schema
from riskguard.explainability.shap_analyzer import ShapAnalyzer
from riskguard.models.trainer import DataSplitter
from riskguard.models.trees import XGBoostFraudModel
from riskguard.utils.explainability_reporter import ExplainabilityReporter
from riskguard.utils.logging import get_logger

logger = get_logger("run_explainability")
_PHASE5_SUBDIR: str = "phase5"
_PHASE4_SUBDIR: str = "phase4"


def main() -> int:
    """Run the full Phase 5 SHAP explainability pipeline.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    settings = Settings()
    output_dir = settings.output_dir / _PHASE5_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info("  RiskGuard AI  |  Phase 5 — SHAP Interpretability Layer")
    logger.info("=" * 65)
    logger.info("Output dir  : %s", output_dir.resolve())
    logger.info("Random seed : %d", settings.random_seed)

    # ── 1. Load Dataset ───────────────────────────────────────────────────────
    loader = DataLoader(
        local_path=settings.dataset_path,
        kaggle_slug=settings.kaggle_dataset_slug,
    )
    try:
        df = loader.load()
    except DataLoadError as exc:
        logger.error("Failed to load dataset: %s", exc)
        return 1

    # ── 2. Validate Schema ────────────────────────────────────────────────────
    val = validate_schema(df)
    if not val.valid:
        logger.error("Validation failed:\n%s", val)
        return 1

    # ── 3. Stratified Split ───────────────────────────────────────────────────
    splitter = DataSplitter(test_size=0.20, random_seed=settings.random_seed)
    X_train, X_test, y_train, y_test = splitter.split(df)

    # ── 4. Load or Train Champion Model ───────────────────────────────────────
    phase4_model_path = settings.output_dir / _PHASE4_SUBDIR / "xgboost_fraud_model.joblib"
    if phase4_model_path.exists():
        logger.info("Loading trained champion model from %s...", phase4_model_path)
        model = XGBoostFraudModel.load(phase4_model_path)
    else:
        logger.info("Phase 4 artefact not found. Training fresh XGBoostFraudModel...")
        model = XGBoostFraudModel(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.1,
            random_seed=settings.random_seed,
        )
        model.fit(X_train, y_train)

    # ── 5. Fit SHAP Analyzer ──────────────────────────────────────────────────
    logger.info("Initializing ShapAnalyzer with background sample reference...")
    analyzer = ShapAnalyzer(
        model=model,
        background_samples=500,
        random_seed=settings.random_seed,
    )
    analyzer.fit(X_train)

    # ── 6. Global Feature Importance ──────────────────────────────────────────
    # Sample 1,500 test records for fast global SHAP analysis
    test_sample_size = min(1500, len(X_test))
    X_test_sample = X_test.sample(n=test_sample_size, random_state=settings.random_seed)
    global_df = analyzer.compute_global_importance(X_test_sample)

    # ── 7. Local Explanations for Flagged High-Risk Transactions ──────────────
    logger.info("Scoring test dataset and finding flagged fraud cases...")
    test_probs = model.predict_proba(X_test)
    y_test_arr = np.asarray(y_test, dtype=int)

    # Find genuine fraud cases that model correctly flagged
    flagged_mask = (test_probs >= model.optimal_threshold) & (y_test_arr == 1)
    flagged_indices = np.where(flagged_mask)[0]

    if len(flagged_indices) == 0:
        # Fallback to highest scoring test cases
        flagged_indices = np.argsort(test_probs)[-3:]

    sample_cases_to_explain = flagged_indices[:3]
    sample_explanations = []

    reporter = ExplainabilityReporter(output_dir=output_dir)

    for i, idx in enumerate(sample_cases_to_explain, 1):
        row = X_test.iloc[[idx]]
        explanation = analyzer.explain_transaction(row, top_n=4)
        sample_explanations.append(explanation)
        reporter.save_local_waterfall_chart(explanation, case_id=i)

    # ── 8. Persist Global Charts & Reports ────────────────────────────────────
    reporter.save_markdown_report(
        global_df=global_df,
        sample_explanations=sample_explanations,
        model_name=model.model_name,
    )
    reporter.save_global_importance_chart(global_df, top_n=15)

    # ── 9. Summary Log ────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("PHASE 5 COMPLETED SUCCESSFULLY")
    logger.info("=" * 65)
    logger.info("Top Global Risk Drivers:")
    for idx, row in global_df.head(5).iterrows():
        logger.info("  %d. %-15s (Mean |SHAP|=%.4f, %5.1f%%)", idx + 1, row["Feature"], row["Mean_Abs_SHAP"], row["Importance_Pct"])
    logger.info("-" * 65)
    logger.info("Local Explanations Generated : %d cases", len(sample_explanations))
    logger.info("Phase 5 outputs written to   : %s", output_dir.resolve())
    logger.info("  Report                     : %s", output_dir / "interpretability_report.md")
    logger.info("  Global SHAP Chart          : %s", output_dir / "shap_global_importance.png")
    logger.info("=" * 65)

    return 0


if __name__ == "__main__":
    sys.exit(main())
