#!/usr/bin/env python
"""
scripts/run_eda.py
~~~~~~~~~~~~~~~~~~
Entry point for Phase 1: Setup & Exploratory Data Analysis.

Orchestration only -- no business logic lives here.
All analysis is delegated to ``riskguard.eda.analysis`` and
all chart rendering to ``riskguard.eda.visualizer``.

Usage::

    python scripts/run_eda.py
    python scripts/run_eda.py            # uses .env / environment variables
    DATASET_PATH=/path/to/creditcard.csv python scripts/run_eda.py
"""

import sys
import warnings

# Suppress noisy third-party warnings before any local imports.
warnings.filterwarnings("ignore")

# Ensure src/ is on the path when running without `pip install -e .`
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from riskguard.config import Settings
from riskguard.data.loader import DataLoader, DataLoadError
from riskguard.data.validator import validate_schema
from riskguard.eda import analysis as eda
from riskguard.eda import visualizer as viz
from riskguard.utils.logging import get_logger

logger = get_logger("run_eda")


def main() -> int:
    """Run the full Phase 1 EDA pipeline.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    settings = Settings()
    logger.info("=" * 60)
    logger.info("  RiskGuard AI  |  Phase 1 -- EDA")
    logger.info("=" * 60)
    logger.info("Output directory: %s", settings.output_dir.resolve())

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
        logger.error("Dataset failed validation:\n%s", result)
        return 1

    logger.info(
        "Dataset: %d rows x %d columns | missing=%d",
        df.shape[0], df.shape[1], df.isnull().sum().sum(),
    )

    # ── 3. Analyse ────────────────────────────────────────────────────────────
    stats = eda.class_imbalance_stats(df)
    logger.info(
        "Class imbalance -- Non-Fraud: %d (%.4f%%)  |  Fraud: %d (%.4f%%)  |  Ratio 1:%.0f",
        stats.legit_count, stats.legit_pct,
        stats.fraud_count, stats.fraud_pct,
        stats.ratio,
    )

    per_class = eda.per_class_stats(df)
    logger.info("Per-class Amount & Time statistics:\n%s", per_class.to_string())

    hourly = eda.temporal_fraud_rate(df)
    peak_hour = int(hourly["FraudRate"].idxmax())
    logger.info(
        "Peak fraud rate: %.2f%% at hour %d",
        hourly["FraudRate"].max(), peak_hour,
    )

    separability = eda.feature_separability(df)
    logger.info("Top 5 discriminating features: %s", separability.head(5).index.tolist())

    class_corr = eda.correlation_with_target(df)
    logger.info(
        "Top positive correlations with Class:\n%s",
        class_corr.tail(5).to_string(),
    )
    logger.info(
        "Top negative correlations with Class:\n%s",
        class_corr.head(5).to_string(),
    )

    # ── 4. Visualise ──────────────────────────────────────────────────────────
    output_dir = settings.output_dir

    viz.plot_class_imbalance(stats, output_dir)
    viz.plot_amount_time_distributions(df, output_dir)
    viz.plot_temporal_patterns(hourly, output_dir)
    viz.plot_vfeature_kdes(df, separability.head(12).index.tolist(), separability, output_dir)
    viz.plot_correlation_analysis(df, class_corr, output_dir)
    viz.plot_feature_separability(separability, output_dir)

    # ── 5. Write summary report ────────────────────────────────────────────────
    _write_summary(df, stats, hourly, class_corr, output_dir)

    logger.info("=" * 60)
    logger.info("Phase 1 EDA complete.  Charts written to: %s", output_dir.resolve())
    logger.info("=" * 60)
    return 0


def _write_summary(df, stats, hourly, class_corr, output_dir: Path) -> None:
    """Persist a markdown EDA summary alongside the charts."""
    ratio = int(stats.ratio)
    peak_hour = int(hourly["FraudRate"].idxmax())
    fraud_df = df[df["Class"] == 1]
    legit_df = df[df["Class"] == 0]

    md = f"""# RiskGuard AI -- Phase 1 EDA Summary

## Dataset
| Property | Value |
|---|---|
| Rows | {df.shape[0]:,} |
| Columns | {df.shape[1]} |
| Missing Values | {df.isnull().sum().sum()} |

## Class Distribution
| Class | Count | Percentage |
|---|---|---|
| Non-Fraud (0) | {stats.legit_count:,} | {stats.legit_pct:.4f}% |
| Fraud (1) | {stats.fraud_count:,} | {stats.fraud_pct:.4f}% |
| **Imbalance Ratio** | **1 : {ratio}** | |

## Amount Statistics
| Metric | Non-Fraud | Fraud |
|---|---|---|
| Median | ${legit_df['Amount'].median():.2f} | ${fraud_df['Amount'].median():.2f} |
| Mean | ${legit_df['Amount'].mean():.2f} | ${fraud_df['Amount'].mean():.2f} |
| Max | ${legit_df['Amount'].max():.2f} | ${fraud_df['Amount'].max():.2f} |

## Temporal Insights
- Dataset spans **{df['Time'].max()/3600:.1f} hours** (~2 days)
- Peak fraud rate: **{hourly['FraudRate'].max():.2f}%** at hour index {peak_hour}

## Top Discriminating Features (Pearson r with Class)
```
Positive (fraud indicator):
{class_corr.tail(5).to_string()}

Negative (fraud suppressor):
{class_corr.head(5).to_string()}
```

## Phase 2 Actions
1. Stratified train/test split -- mandatory given 1:{ratio} imbalance
2. Discard accuracy metric -- use PR-AUC, F1, Recall@Precision
3. Scale Amount & Time (V-features already PCA-standardised)
4. Baseline: Logistic Regression with `class_weight='balanced'`
5. Candidate feature: hour-of-day (elevated fraud rates in certain hours)
"""
    summary_path = output_dir / "eda_summary.md"
    summary_path.write_text(md, encoding="utf-8")
    logger.info("EDA summary written: %s", summary_path)


if __name__ == "__main__":
    sys.exit(main())
