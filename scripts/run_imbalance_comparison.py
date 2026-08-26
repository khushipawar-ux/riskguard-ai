#!/usr/bin/env python
"""
scripts/run_imbalance_comparison.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Entry point for Phase 3: Imbalance Strategy Comparison.

Orchestration only — all logic is delegated to the ``riskguard`` package.

Pipeline
--------
1. Load and validate the Credit Card Fraud dataset.
2. Stratified 80/20 train/test split.
3. Run ImbalanceComparison across three strategies:
   - class_weight='balanced'  (baseline, no data modification)
   - SMOTE oversampling        (synthetic minority samples)
   - Random undersampling      (discard majority samples)
4. Rank strategies by PR-AUC on the held-out test set.
5. Save comparison report, PR-AUC bar chart, threshold curves to
   ``outputs/phase3/``.

Usage::

    python scripts/run_imbalance_comparison.py
    DATASET_PATH=/path/to/creditcard.csv python scripts/run_imbalance_comparison.py
    LOG_LEVEL=DEBUG python scripts/run_imbalance_comparison.py
"""

from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from riskguard.config import Settings
from riskguard.data.loader import DataLoadError, DataLoader
from riskguard.data.validator import validate_schema
from riskguard.features.engineering import FraudFeatureTransformer
from riskguard.models.comparison import ImbalanceComparison
from riskguard.models.evaluator import ModelEvaluator
from riskguard.models.imbalance import (
    ClassWeighting,
    RandomUndersampling,
    SmoteOversampling,
)
from riskguard.models.trainer import DataSplitter
from riskguard.utils.comparison_reporter import ComparisonReporter
from riskguard.utils.logging import get_logger

logger = get_logger("run_imbalance_comparison")

_PHASE3_SUBDIR: str = "phase3"


def main() -> int:
    """Run the Phase 3 imbalance comparison pipeline.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    settings = Settings()
    output_dir = settings.output_dir / _PHASE3_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info("  RiskGuard AI  |  Phase 3 -- Imbalance Strategy Comparison")
    logger.info("=" * 65)
    logger.info("Output dir  : %s", output_dir.resolve())
    logger.info("Random seed : %d", settings.random_seed)

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
    val = validate_schema(df)
    if not val.valid:
        logger.error("Dataset validation failed:\n%s", val)
        return 1
    logger.info(
        "Dataset: %d rows x %d columns | missing=%d",
        df.shape[0], df.shape[1], df.isnull().sum().sum(),
    )

    # ── 3. Stratified split ───────────────────────────────────────────────────
    splitter = DataSplitter(test_size=0.20, random_seed=settings.random_seed)
    X_train, X_test, y_train, y_test = splitter.split(df)

    # ── 4. Run comparison ─────────────────────────────────────────────────────
    strategies = [
        ClassWeighting(random_seed=settings.random_seed),
        SmoteOversampling(sampling_strategy=0.1, random_seed=settings.random_seed),
        RandomUndersampling(sampling_strategy=0.1, random_seed=settings.random_seed),
    ]
    comparison = ImbalanceComparison(
        strategies=strategies,
        random_seed=settings.random_seed,
    )
    result = comparison.run(X_train, X_test, y_train, y_test)

    # ── 5. Build threshold curves for the chart ───────────────────────────────
    # We need to re-transform the test set to compute per-strategy curves.
    transformer = FraudFeatureTransformer()
    transformer.fit(X_train)
    X_test_t = transformer.transform(X_test)
    evaluator = ModelEvaluator()

    # Re-run each strategy's classifier predictions to get threshold curves.
    # (Results already cached in strategy_results — we only need the curves.)
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    threshold_curves: dict[str, object] = {}
    for sr in result.strategy_results:
        strategy = sr.strategy
        X_train_t = transformer.fit_transform(X_train)
        X_res, y_res = strategy.fit_resample(X_train_t, np.asarray(y_train, dtype=int))
        clf = LogisticRegression(
            class_weight="balanced" if strategy.requires_class_weight else None,
            max_iter=1_000,
            solver="lbfgs",
            random_state=settings.random_seed,
        )
        clf.fit(X_res, y_res)
        y_prob = clf.predict_proba(X_test_t)[:, 1]
        threshold_curves[strategy.name] = evaluator.threshold_curve(y_test, y_prob)

    # ── 6. Report & persist ───────────────────────────────────────────────────
    reporter = ComparisonReporter(output_dir=output_dir)
    reporter.save_markdown(result)
    reporter.save_prauc_bar_chart(result)
    reporter.save_threshold_curves(result, threshold_curves)

    # ── 7. Log summary ────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("PHASE 3 RESULTS")
    logger.info("=" * 65)
    for sr in result.strategy_results:
        er = sr.eval_result
        win = " <-- WINNER" if sr.strategy.name == result.winner.name else ""
        recall_str = f"{er.recall_at_80p:.4f}" if er.recall_at_80p is not None else "N/A"
        logger.info(
            "  %-30s  PR-AUC=%.4f | Best F1=%.4f | Recall@80%%prec=%s%s",
            sr.strategy.name,
            er.pr_auc,
            er.best_f1,
            recall_str,
            win,
        )
    logger.info("=" * 65)
    logger.info("Winner: %s", result.winner.name)
    logger.info("Phase 3 outputs written to: %s", output_dir.resolve())
    logger.info("  Report  : %s", output_dir / "imbalance_comparison.md")
    logger.info("  Chart 1 : %s", output_dir / "imbalance_prauc_comparison.png")
    logger.info("  Chart 2 : %s", output_dir / "imbalance_threshold_curves.png")
    logger.info("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
