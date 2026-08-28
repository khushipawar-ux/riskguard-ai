#!/usr/bin/env python
"""
scripts/run_stronger_models.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Entry point for Phase 4: Stronger Models & Decision Threshold Tuning.

Orchestrates:
1. Load & validate dataset.
2. Stratified train/test split.
3. 5-Fold Stratified Cross-Validation on XGBoost and LightGBM.
4. Model evaluation on held-out test set against Baseline Logistic Regression.
5. Explicit decision threshold optimization across business policies:
   - Max F1
   - Target Precision >= 80%
   - Cost-Optimal Policy
6. Model artefact serialization and visual reporting in ``outputs/phase4/``.

Usage::

    python scripts/run_stronger_models.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from riskguard.config import Settings
from riskguard.data.loader import DataLoadError, DataLoader
from riskguard.data.validator import validate_schema
from riskguard.models.baseline import LogisticRegressionBaseline
from riskguard.models.evaluator import ModelEvaluator
from riskguard.models.threshold import ThresholdOptimizer
from riskguard.models.trainer import DataSplitter, ModelTrainer
from riskguard.models.trees import LightGBMFraudModel, XGBoostFraudModel
from riskguard.utils.logging import get_logger
from riskguard.utils.tree_reporter import TreeReporter

logger = get_logger("run_stronger_models")
_PHASE4_SUBDIR: str = "phase4"


def main() -> int:
    """Run the full Phase 4 pipeline.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    settings = Settings()
    output_dir = settings.output_dir / _PHASE4_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info("  RiskGuard AI  |  Phase 4 — Tree Models & Threshold Tuning")
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
    y_test_arr = np.asarray(y_test, dtype=int)

    # ── 4. Stratified 5-Fold Cross-Validation ─────────────────────────────────
    trainer = ModelTrainer(n_splits=5, random_seed=settings.random_seed)

    xgb_cv = trainer.cross_validate(
        XGBoostFraudModel,
        X_train,
        y_train,
        params={"n_estimators": 120, "max_depth": 4, "learning_rate": 0.1},
    )

    lgb_cv = trainer.cross_validate(
        LightGBMFraudModel,
        X_train,
        y_train,
        params={"n_estimators": 120, "num_leaves": 31, "learning_rate": 0.05},
    )

    # ── 5. Train Final Candidate Models on Full Training Split ────────────────
    logger.info("Training final candidate models on full training data...")

    # Logistic Regression Baseline
    baseline = LogisticRegressionBaseline(random_seed=settings.random_seed)
    baseline.fit(X_train, y_train)
    prob_base = baseline.predict_proba(X_test)

    # XGBoost
    xgb_model = XGBoostFraudModel(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.1,
        random_seed=settings.random_seed,
    )
    xgb_model.fit(X_train, y_train)
    prob_xgb = xgb_model.predict_proba(X_test)

    # LightGBM
    lgb_model = LightGBMFraudModel(
        n_estimators=120,
        num_leaves=31,
        learning_rate=0.05,
        random_seed=settings.random_seed,
    )
    lgb_model.fit(X_train, y_train)
    prob_lgb = lgb_model.predict_proba(X_test)

    # ── 6. Test Set Evaluation ────────────────────────────────────────────────
    evaluator = ModelEvaluator()
    eval_base = evaluator.compute_metrics(y_test, prob_base)
    eval_xgb = evaluator.compute_metrics(y_test, prob_xgb)
    eval_lgb = evaluator.compute_metrics(y_test, prob_lgb)

    test_evals = {
        "Logistic Regression Baseline": eval_base,
        "XGBoost Fraud Model": eval_xgb,
        "LightGBM Fraud Model": eval_lgb,
    }

    # Select winner based on test PR-AUC
    candidates = [
        ("XGBoost Fraud Model", xgb_model, eval_xgb),
        ("LightGBM Fraud Model", lgb_model, eval_lgb),
    ]
    candidates.sort(key=lambda item: item[2].pr_auc, reverse=True)
    best_name, best_model, best_eval = candidates[0]
    best_prob = prob_xgb if best_name == "XGBoost Fraud Model" else prob_lgb

    logger.info("Best model selected: %s (Test PR-AUC=%.4f)", best_name, best_eval.pr_auc)

    # ── 7. Decision Threshold & Risk Policy Optimization ──────────────────────
    optimizer = ThresholdOptimizer(default_cost_fn=500.0, default_cost_fp=25.0)
    policy_df = optimizer.compare_policies(y_test, best_prob)

    best_f1_thresh, _ = optimizer.find_best_f1_threshold(y_test, best_prob)
    best_model.optimal_threshold = best_f1_thresh

    # ── 8. Persist Model Artefact & Reports ────────────────────────────────────
    saved_model_path = best_model.save(output_dir)

    reporter = TreeReporter(output_dir=output_dir)
    reporter.save_markdown_report(
        cv_results=[xgb_cv, lgb_cv],
        test_evals=test_evals,
        policy_df=policy_df,
        best_model_name=best_name,
    )

    curves_dict = {
        "Baseline (LogReg)": (y_test_arr, prob_base, eval_base.pr_auc),
        "XGBoost": (y_test_arr, prob_xgb, eval_xgb.pr_auc),
        "LightGBM": (y_test_arr, prob_lgb, eval_lgb.pr_auc),
    }
    reporter.save_pr_curves(curves_dict)
    reporter.save_policy_chart(
        y_true=y_test_arr,
        y_prob=best_prob,
        best_threshold=best_f1_thresh,
        model_name=best_name,
    )

    # ── 9. Summary Log ────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("PHASE 4 COMPLETED SUCCESSFULLY")
    logger.info("=" * 65)
    logger.info("Selected Champion : %s", best_name)
    logger.info("Test PR-AUC       : %.4f (Baseline: %.4f)", best_eval.pr_auc, eval_base.pr_auc)
    logger.info("Best F1 Score     : %.4f (Optimal threshold: %.4f)", best_eval.best_f1, best_f1_thresh)
    logger.info("Artefacts saved to: %s", output_dir.resolve())
    logger.info("  Model Artefact  : %s", saved_model_path)
    logger.info("  Report          : %s", output_dir / "model_comparison.md")
    logger.info("  PR Curve Chart  : %s", output_dir / "pr_curves_comparison.png")
    logger.info("  Threshold Chart : %s", output_dir / "threshold_policy_curves.png")
    logger.info("=" * 65)

    return 0


if __name__ == "__main__":
    sys.exit(main())
