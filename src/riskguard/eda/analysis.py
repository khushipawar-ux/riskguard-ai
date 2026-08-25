"""
riskguard.eda.analysis
~~~~~~~~~~~~~~~~~~~~~~
Pure statistical analysis functions for the EDA phase.

Design principles
-----------------
* **No side effects** — functions only read DataFrames and return results.
* **No I/O** — no file writes, no logging, no chart rendering.
* **Fully testable** — every function accepts a DataFrame and returns a
  plain Python / pandas / numpy object.

Callers (scripts, visualiser) are responsible for logging and persistence.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd


# ── Named return types ────────────────────────────────────────────────────────

class ImbalanceStats(NamedTuple):
    """Summary statistics for class imbalance."""
    legit_count: int
    fraud_count: int
    legit_pct: float
    fraud_pct: float
    ratio: float  # legit / fraud


# ── Public functions ──────────────────────────────────────────────────────────

def class_imbalance_stats(df: pd.DataFrame, target: str = "Class") -> ImbalanceStats:
    """Compute class counts and imbalance ratio.

    Args:
        df:     Dataset containing the target column.
        target: Name of the binary target column (0 = legit, 1 = fraud).

    Returns:
        :class:`ImbalanceStats` with counts and percentages.
    """
    counts = df[target].value_counts()
    norm = df[target].value_counts(normalize=True) * 100
    legit_cnt = int(counts[0])
    fraud_cnt = int(counts[1])
    return ImbalanceStats(
        legit_count=legit_cnt,
        fraud_count=fraud_cnt,
        legit_pct=float(norm[0]),
        fraud_pct=float(norm[1]),
        ratio=legit_cnt / fraud_cnt,
    )


def per_class_stats(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    target: str = "Class",
) -> pd.DataFrame:
    """Descriptive statistics broken down by class.

    Args:
        df:      Dataset.
        columns: Columns to describe.  Defaults to ``["Amount", "Time"]``.
        target:  Binary target column name.

    Returns:
        Multi-level :class:`pandas.DataFrame` from ``groupby(...).describe()``.
    """
    cols = columns or ["Amount", "Time"]
    return df.groupby(target)[cols].describe().round(2)


def temporal_fraud_rate(
    df: pd.DataFrame,
    time_col: str = "Time",
    target: str = "Class",
    window_hours: int = 48,
) -> pd.DataFrame:
    """Compute per-hour fraud count and rate across a rolling time window.

    Args:
        df:           Dataset with a ``Time`` column in seconds.
        time_col:     Name of the elapsed-seconds column.
        target:       Binary target column name.
        window_hours: Modular window size (default 48 = 2-day dataset).

    Returns:
        DataFrame with columns ``Fraud``, ``Total``, ``FraudRate``
        indexed by ``Hour`` (0-based integer).
    """
    hour_col = (df[time_col] / 3600).astype(int) % window_hours
    tmp = df.assign(Hour=hour_col)
    hourly = (
        tmp.groupby("Hour")[target]
        .agg(["sum", "count"])
        .rename(columns={"sum": "Fraud", "count": "Total"})
    )
    hourly["FraudRate"] = hourly["Fraud"] / hourly["Total"] * 100
    return hourly


def feature_separability(
    df: pd.DataFrame,
    features: list[str] | None = None,
    target: str = "Class",
) -> pd.Series:
    """Compute a Cohen's-d-proxy separability score for each feature.

    A higher score means the feature distributions for fraud and non-fraud
    are more distinct (i.e. the feature is more discriminating).

    Formula:
        ``|mean_fraud - mean_legit| / pooled_std``

    Args:
        df:       Dataset.
        features: Feature columns to evaluate.  Defaults to V1-V28 + Amount.
        target:   Binary target column name.

    Returns:
        :class:`pandas.Series` indexed by feature name, sorted descending.
    """
    if features is None:
        features = [f"V{i}" for i in range(1, 29)] + ["Amount"]

    fraud_df = df[df[target] == 1]
    legit_df = df[df[target] == 0]

    scores: dict[str, float] = {}
    for col in features:
        pooled_std = float(
            np.sqrt(
                (fraud_df[col].std() ** 2 + legit_df[col].std() ** 2) / 2
            )
        ) + 1e-8
        scores[col] = abs(float(fraud_df[col].mean()) - float(legit_df[col].mean())) / pooled_std

    return pd.Series(scores).sort_values(ascending=False)


def correlation_with_target(
    df: pd.DataFrame,
    features: list[str] | None = None,
    target: str = "Class",
) -> pd.Series:
    """Pearson correlation of each feature with *target*, sorted ascending.

    Args:
        df:       Dataset.
        features: Columns to include.  Defaults to V1-V28 + Amount + Time.
        target:   Binary target column name.

    Returns:
        :class:`pandas.Series` of correlations sorted from most-negative
        to most-positive.
    """
    if features is None:
        features = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]

    cols = features + [target]
    corr_matrix = df[cols].corr()
    return corr_matrix[target].drop(target).sort_values()
