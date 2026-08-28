#!/usr/bin/env python
"""
scripts/run_inference.py
~~~~~~~~~~~~~~~~~~~~~~~~
CLI Entry point for Phase 6: Packaging & Inference Engine.

Demonstrates:
1. Loading the production champion model.
2. Scoring representative transactions (Low risk vs Suspicious vs Actual Fraud).
3. Outputting decision, probability, and top-3 SHAP reason codes.
4. Exporting self-contained production bundle to ``outputs/phase6/``.

Usage::

    python scripts/run_inference.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from riskguard.config import Settings
from riskguard.data.loader import DataLoadError, DataLoader
from riskguard.explainability.shap_analyzer import ShapAnalyzer
from riskguard.inference.predictor import FraudPredictor
from riskguard.models.trainer import DataSplitter
from riskguard.models.trees import XGBoostFraudModel
from riskguard.utils.logging import get_logger

logger = get_logger("run_inference")
_PHASE6_SUBDIR: str = "phase6"
_PHASE4_SUBDIR: str = "phase4"


def main() -> int:
    """Run Phase 6 inference and packaging demonstration.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    settings = Settings()
    output_dir = settings.output_dir / _PHASE6_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info("  RiskGuard AI  |  Phase 6 — Packaging & Inference Engine")
    logger.info("=" * 65)
    logger.info("Output dir  : %s", output_dir.resolve())

    # ── 1. Load Dataset & Train/Test Split ─────────────────────────────────────
    loader = DataLoader(
        local_path=settings.dataset_path,
        kaggle_slug=settings.kaggle_dataset_slug,
    )
    try:
        df = loader.load()
    except DataLoadError as exc:
        logger.error("Failed to load dataset: %s", exc)
        return 1

    splitter = DataSplitter(test_size=0.20, random_seed=settings.random_seed)
    X_train, X_test, y_train, y_test = splitter.split(df)
    y_test_arr = np.asarray(y_test, dtype=int)

    # ── 2. Load Model & Explainer ─────────────────────────────────────────────
    model_path = settings.output_dir / _PHASE4_SUBDIR / "xgboost_fraud_model.joblib"
    if model_path.exists():
        logger.info("Loading champion model from %s...", model_path)
        model = XGBoostFraudModel.load(model_path)
    else:
        logger.info("Training XGBoostFraudModel...")
        model = XGBoostFraudModel(n_estimators=120, max_depth=4, learning_rate=0.1, random_seed=42)
        model.fit(X_train, y_train)

    explainer = ShapAnalyzer(model=model, background_samples=500, random_seed=42)
    explainer.fit(X_train)

    predictor = FraudPredictor(
        model=model,
        explainer=explainer,
        threshold=model.optimal_threshold,
    )

    # ── 3. Evaluate 3 Archetypal Transaction Cases ────────────────────────────
    # Case 1: Known legitimate transaction
    legit_idx = np.where(y_test_arr == 0)[0][0]
    tx_legit = X_test.iloc[legit_idx].to_dict()

    # Case 2: Suspicious synthetic high-amount midnight transaction
    tx_suspicious = tx_legit.copy()
    tx_suspicious["Amount"] = 850.00
    tx_suspicious["Time"] = 14400.0  # 4 AM
    tx_suspicious["V4"] = 2.4
    tx_suspicious["V14"] = -2.1

    # Case 3: Confirmed actual fraud transaction
    fraud_idx = np.where(y_test_arr == 1)[0][0]
    tx_fraud = X_test.iloc[fraud_idx].to_dict()

    cases = [
        ("Legitimate Transaction", tx_legit),
        ("Suspicious Midnight Purchase", tx_suspicious),
        ("Confirmed Fraud Case", tx_fraud),
    ]

    logger.info("-" * 65)
    logger.info("LIVE TRANSACTION SCORING DEMONSTRATION")
    logger.info("-" * 65)

    scoring_results = []
    for label, tx_data in cases:
        res = predictor.predict(tx_data, transaction_id=label)
        scoring_results.append(res.to_dict())

        logger.info("[*] Transaction: %s", label)
        logger.info("  Fraud Probability : %.2f%%", res.fraud_probability * 100)
        logger.info("  Risk Level        : %s", res.risk_level)
        logger.info("  Action Decision   : %s (Threshold=%.4f)", res.decision, res.threshold_used)
        logger.info("  Latency           : %.2f ms", res.latency_ms)
        logger.info("  Top 3 Risk Drivers:")
        for r in res.top_reasons:
            logger.info("    - %-15s (+%.4f SHAP) : %s", r.feature, r.shap_impact, r.explanation)
        logger.info("  Analyst Summary   : %s", res.narrative_summary)
        logger.info("-" * 65)

    # ── 4. Batch Scoring Sample ───────────────────────────────────────────────
    logger.info("Scoring sample batch of 25 test transactions...")
    batch_df = predictor.predict_batch_df(X_test.head(25))
    batch_csv_path = output_dir / "sample_batch_scored.csv"
    batch_df.to_csv(batch_csv_path, index=False)
    logger.info("Batch results saved to: %s", batch_csv_path)

    # ── 5. Export Self-Contained Production Bundle ─────────────────────────────
    bundle_path = predictor.save_bundle(output_dir / "production_bundle")
    logger.info("Exported self-contained bundle to: %s", bundle_path)

    # Verify bundle loading
    loaded_predictor = FraudPredictor.load_bundle(bundle_path)
    test_res = loaded_predictor.predict(tx_fraud)
    assert test_res.decision == res.decision
    logger.info("Bundle round-trip load & prediction verified successfully!")

    # Save summary report
    report_path = output_dir / "inference_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model.model_name,
                "operating_threshold": model.optimal_threshold,
                "archetype_cases": scoring_results,
            },
            f,
            indent=2,
        )

    logger.info("=" * 65)
    logger.info("PHASE 6 PACKAGING COMPLETED SUCCESSFULLY")
    logger.info("=" * 65)
    logger.info("Production Bundle  : %s", bundle_path)
    logger.info("Batch Scoring CSV  : %s", batch_csv_path)
    logger.info("Inference Summary  : %s", report_path)
    logger.info("Interactive UI     : Run `streamlit run src/riskguard/ui/app.py`")
    logger.info("=" * 65)

    return 0


if __name__ == "__main__":
    sys.exit(main())
