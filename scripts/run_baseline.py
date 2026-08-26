#!/usr/bin/env python
"""
scripts/run_baseline.py
~~~~~~~~~~~~~~~~~~~~~~~
Entry point for Phase 2: Baseline Model.

Orchestration only — no business logic lives here.
All computation is delegated to the ``riskguard`` package modules.

Pipeline
--------
1. Load the Credit Card Fraud dataset (local path or kagglehub).
2. Validate schema.
3. Stratified train / test split.
4. Fit the Logistic Regression baseline (includes feature engineering).
5. Evaluate on the held-out test set.
6. Save metrics report, threshold curve chart, and model artefact to
   ``outputs/phase2/``.

Usage::

    python scripts/run_baseline.py
    DATASET_PATH=/path/to/creditcard.csv python scripts/run_baseline.py
    LOG_LEVEL=DEBUG python scripts/run_baseline.py
"""

from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

# Ensure src/ is on the path when running without `pip install -e .`
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from riskguard.config import Settings
from riskguard.data.loader import DataLoadError, DataLoader
from riskguard.data.validator import validate_schema
from riskguard.models.baseline import LogisticRegressionBaseline
from riskguard.models.evaluator import ModelEvaluator
from riskguard.models.trainer import DataSplitter
from riskguard.utils.logging import get_logger
from riskguard.utils.metrics_reporter import MetricsReporter

logger = get_logger("run_baseline")

# Phase 2 sub-directory within the configured output root.
_PHASE2_SUBDIR: str = "phase2"


def main() -> int:
    """Run the full Phase 2 baseline pipeline.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    settings = Settings()
    output_dir = settings.output_dir / _PHASE2_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    _log_header(settings, output_dir)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    loader = DataLoader(
        local_path=settings.dataset_path,
        kaggle_slug=settings.kaggle_dataset_slug,
    )
    try:
        df = loader.load()
    except DataLoadError as exc:
        logger.error("Failed to load dataset: %s", exc)
        return 1

    # ── 2. Validate ───────────────────────────────────────────────────────────
    result = validate_schema(df)
    if not result.valid:
        logger.error("Dataset validation failed:\n%s", result)
        return 1
    logger.info(
        "Dataset: %d rows × %d columns  |  missing=%d",
        df.shape[0],
        df.shape[1],
        df.isnull().sum().sum(),
    )

    # ── 3. Stratified split ───────────────────────────────────────────────────
    splitter = DataSplitter(
        test_size=0.20,
        random_seed=settings.random_seed,
    )
    X_train, X_test, y_train, y_test = splitter.split(df)

    logger.info(
        "Train: %d samples (fraud=%d, %.4f%%)",
        len(X_train),
        int(y_train.sum()),
        y_train.mean() * 100,
    )
    logger.info(
        "Test : %d samples (fraud=%d, %.4f%%)",
        len(X_test),
        int(y_test.sum()),
        y_test.mean() * 100,
    )

    # ── 4. Fit baseline model ─────────────────────────────────────────────────
    model = LogisticRegressionBaseline(random_seed=settings.random_seed)
    model.fit(X_train, y_train)

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    evaluator = ModelEvaluator()
    y_prob = model.predict_proba(X_test)

    eval_result = evaluator.compute_metrics(y_test, y_prob, operating_threshold=0.5)
    curve = evaluator.threshold_curve(y_test, y_prob)

    recall_at_best_t = evaluator.recall_at_precision(y_test, y_prob, min_precision=0.80)

    _log_results(eval_result, recall_at_best_t)

    # ── 6. Persist outputs ────────────────────────────────────────────────────
    reporter = MetricsReporter(output_dir=output_dir)
    reporter.save_markdown(
        eval_result,
        model_name="Logistic Regression Baseline (class_weight='balanced')",
        extra_notes=(
            "- Feature engineering: Amount + Time scaled with StandardScaler; "
            "HourOfDay added as an engineered feature.\n"
            "- Next phase (Phase 3) will compare SMOTE, undersampling, and class "
            "weighting on XGBoost / LightGBM to improve PR-AUC further.\n"
            f"- Model feature names: {model.feature_names}"
        ),
    )
    reporter.save_threshold_chart(
        curve,
        model_name="Logistic Regression Baseline",
    )
    model_path = model.save(output_dir)

    logger.info("=" * 60)
    logger.info("Phase 2 outputs written to: %s", output_dir.resolve())
    logger.info("  Metrics report : %s", output_dir / "baseline_metrics.md")
    logger.info("  Threshold chart: %s", output_dir / "baseline_threshold_curve.png")
    logger.info("  Model artefact : %s", model_path)
    logger.info("=" * 60)
    return 0


# ── Private helpers ───────────────────────────────────────────────────────────


def _log_header(settings: Settings, output_dir: Path) -> None:
    logger.info("=" * 60)
    logger.info("  RiskGuard AI  |  Phase 2 — Baseline Model")
    logger.info("=" * 60)
    logger.info("Random seed    : %d", settings.random_seed)
    logger.info("Output dir     : %s", output_dir.resolve())


def _log_results(eval_result, recall_at_best_t) -> None:
    logger.info("-" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("-" * 60)
    logger.info("PR-AUC (primary)       : %.4f", eval_result.pr_auc)
    logger.info("Best F1                : %.4f  (threshold=%.4f)", eval_result.best_f1, eval_result.best_threshold)
    logger.info("F1 @ threshold=0.50    : %.4f", eval_result.f1_at_threshold)
    if eval_result.recall_at_80p is not None:
        logger.info("Recall @ >=80%% prec   : %.4f", eval_result.recall_at_80p)
    else:
        logger.info("Recall @ >=80%% prec   : N/A (precision target unachievable)")
    cm = eval_result.confusion
    if len(cm) == 2 and len(cm[0]) == 2:
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        logger.info(
            "Confusion (t=0.50)     : TN=%d  FP=%d  FN=%d  TP=%d", tn, fp, fn, tp
        )
    logger.info("-" * 60)


if __name__ == "__main__":
    sys.exit(main())
