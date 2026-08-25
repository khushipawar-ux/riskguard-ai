"""
riskguard.eda.visualizer
~~~~~~~~~~~~~~~~~~~~~~~~
Chart-generation functions for the EDA phase.

Design principles
-----------------
* Accepts **pre-computed** DataFrames / Series from :mod:`riskguard.eda.analysis`.
* Does **not** perform statistical computation itself.
* Writes chart files to disk via :func:`riskguard.utils.plotting.save_figure`.
* Returns the saved :class:`pathlib.Path` for logging / testing.

All six EDA charts are generated here.
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

from riskguard.eda.analysis import ImbalanceStats
from riskguard.utils.plotting import (
    BG, BLUE, FG, GRID_CLR, RED,
    apply_theme,
    save_figure,
)
from riskguard.utils.logging import get_logger

logger = get_logger(__name__)

# Apply theme once at module import.
apply_theme()

_LABELS = ["Non-Fraud", "Fraud"]
_COLOURS = [BLUE, RED]


# ── Chart 1 ───────────────────────────────────────────────────────────────────

def plot_class_imbalance(
    stats: ImbalanceStats, output_dir: pathlib.Path
) -> pathlib.Path:
    """Bar chart (log scale) + donut showing the class imbalance.

    Args:
        stats:      Computed by :func:`riskguard.eda.analysis.class_imbalance_stats`.
        output_dir: Directory to save the chart.

    Returns:
        Path of the saved PNG.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Class Distribution -- Extreme Imbalance",
        fontsize=14, fontweight="bold", color=FG, y=1.02,
    )

    counts = [stats.legit_count, stats.fraud_count]

    # Bar (log scale)
    bars = ax1.bar(_LABELS, counts, color=_COLOURS, width=0.45, zorder=3)
    ax1.set_yscale("log")
    ax1.set_ylabel("Count (log scale)")
    ax1.set_title("Transaction Counts (log scale)", fontsize=11)
    ax1.grid(axis="y", alpha=0.4)
    for bar, cnt in zip(bars, counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.5,
            f"{cnt:,}",
            ha="center", fontweight="bold", color=FG, fontsize=10,
        )

    # Donut
    _, _, autotexts = ax2.pie(
        counts, labels=_LABELS, colors=_COLOURS,
        autopct="%1.3f%%", startangle=90,
        wedgeprops={"width": 0.55, "edgecolor": BG, "linewidth": 2},
        textprops={"color": FG},
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color(FG)
        at.set_fontweight("bold")
    ax2.set_title("Proportion (donut)", fontsize=11)

    fig.tight_layout()
    dest = output_dir / "01_class_imbalance.png"
    save_figure(fig, dest)
    logger.info("Saved: %s", dest)
    return dest


# ── Chart 2 ───────────────────────────────────────────────────────────────────

def plot_amount_time_distributions(
    df: pd.DataFrame, output_dir: pathlib.Path
) -> pathlib.Path:
    """Four-panel chart: Amount KDE, Amount box (log), Time histogram, Time CDF.

    Args:
        df:         Full dataset with ``Amount``, ``Time``, and ``Class``.
        output_dir: Directory to save the chart.

    Returns:
        Path of the saved PNG.
    """
    fraud_df = df[df["Class"] == 1]
    legit_df = df[df["Class"] == 0]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "Amount & Time Distributions -- Fraud vs. Non-Fraud",
        fontsize=14, fontweight="bold", color=FG, y=1.01,
    )
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.3)

    # Amount KDE
    ax1 = fig.add_subplot(gs[0, 0])
    legit_df["Amount"].clip(upper=legit_df["Amount"].quantile(0.99)).plot.kde(
        ax=ax1, color=BLUE, lw=2, label="Non-Fraud"
    )
    fraud_df["Amount"].clip(upper=fraud_df["Amount"].quantile(0.99)).plot.kde(
        ax=ax1, color=RED, lw=2, label="Fraud"
    )
    ax1.set_title("Transaction Amount -- KDE")
    ax1.set_xlabel("Amount ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Amount boxplot (log scale)
    ax2 = fig.add_subplot(gs[0, 1])
    box_data = [legit_df["Amount"].values + 0.01, fraud_df["Amount"].values + 0.01]
    try:
        bp = ax2.boxplot(
            box_data, tick_labels=_LABELS, patch_artist=True,
            medianprops={"color": "white", "lw": 2},
            flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
        )
    except TypeError:
        bp = ax2.boxplot(
            box_data, labels=_LABELS, patch_artist=True,
            medianprops={"color": "white", "lw": 2},
            flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
        )
    for patch, colour in zip(bp["boxes"], _COLOURS):
        patch.set_facecolor(colour)
        patch.set_alpha(0.6)
    ax2.set_yscale("log")
    ax2.set_title("Transaction Amount -- Box (log)")
    ax2.set_ylabel("Amount ($, log)")
    ax2.grid(True, alpha=0.3, axis="y")

    # Time histogram
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(legit_df["Time"] / 3600, bins=48, color=BLUE, alpha=0.6, label="Non-Fraud", density=True)
    ax3.hist(fraud_df["Time"] / 3600, bins=48, color=RED,  alpha=0.8, label="Fraud",     density=True)
    ax3.set_title("Time (hours since start)")
    ax3.set_xlabel("Hours")
    ax3.set_ylabel("Density")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Time CDF
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(
        np.sort(legit_df["Time"].values / 3600),
        np.linspace(0, 1, len(legit_df)),
        color=BLUE, lw=2, label="Non-Fraud",
    )
    ax4.plot(
        np.sort(fraud_df["Time"].values / 3600),
        np.linspace(0, 1, len(fraud_df)),
        color=RED, lw=2, label="Fraud",
    )
    ax4.set_title("CDF -- Time")
    ax4.set_xlabel("Hours")
    ax4.set_ylabel("CDF")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    fig.tight_layout()
    dest = output_dir / "02_amount_time_distributions.png"
    save_figure(fig, dest)
    logger.info("Saved: %s", dest)
    return dest


# ── Chart 3 ───────────────────────────────────────────────────────────────────

def plot_temporal_patterns(
    hourly: pd.DataFrame, output_dir: pathlib.Path
) -> pathlib.Path:
    """Two-panel chart: hourly volume vs. fraud count + fraud-rate line.

    Args:
        hourly:     Output of :func:`riskguard.eda.analysis.temporal_fraud_rate`.
        output_dir: Directory to save the chart.

    Returns:
        Path of the saved PNG.
    """
    peak_hour = int(hourly["FraudRate"].idxmax())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        "Temporal Fraud Patterns -- Hourly Breakdown",
        fontsize=14, fontweight="bold", color=FG,
    )

    ax1.bar(hourly.index, hourly["Total"],     color=BLUE, alpha=0.5, label="Total Txns")
    ax1.bar(hourly.index, hourly["Fraud"]*100, color=RED,  alpha=0.9, label="Fraud x100 (scaled)")
    ax1.set_ylabel("Count")
    ax1.set_title("Volume vs. Fraud Count per Hour", fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.fill_between(hourly.index, hourly["FraudRate"], color=RED, alpha=0.25)
    ax2.plot(hourly.index, hourly["FraudRate"], color=RED, lw=2)
    ax2.axvline(peak_hour, color="yellow", ls="--", lw=1.2, alpha=0.75)
    ax2.text(
        peak_hour + 0.3,
        hourly["FraudRate"].max() * 0.93,
        f"Peak h={peak_hour}",
        color="yellow", fontsize=9,
    )
    ax2.set_xlabel("Hour Index")
    ax2.set_ylabel("Fraud Rate (%)")
    ax2.set_title("Fraud Rate per Hour", fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    dest = output_dir / "03_temporal_patterns.png"
    save_figure(fig, dest)
    logger.info("Saved: %s | Peak fraud rate %.2f%% at hour %d", dest, hourly["FraudRate"].max(), peak_hour)
    return dest


# ── Chart 4 ───────────────────────────────────────────────────────────────────

def plot_vfeature_kdes(
    df: pd.DataFrame,
    top_features: list[str],
    separability: pd.Series,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    """4x3 grid of KDE plots for the most discriminating PCA features.

    Args:
        df:            Full dataset.
        top_features:  Ordered list of top-12 feature names.
        separability:  Cohen's-d Series from :func:`feature_separability`.
        output_dir:    Directory to save the chart.

    Returns:
        Path of the saved PNG.
    """
    fraud_df = df[df["Class"] == 1]
    legit_df = df[df["Class"] == 0]

    fig, axes = plt.subplots(4, 3, figsize=(16, 13))
    fig.suptitle(
        "Top 12 PCA Features -- Fraud vs. Non-Fraud KDE",
        fontsize=14, fontweight="bold", color=FG, y=1.01,
    )

    for ax, col in zip(axes.flatten(), top_features[:12]):
        lo = df[col].quantile(0.005)
        hi = df[col].quantile(0.995)
        legit_df[col].clip(lo, hi).plot.kde(ax=ax, color=BLUE, lw=2, label="Non-Fraud")
        fraud_df[col].clip(lo, hi).plot.kde(ax=ax, color=RED,  lw=2, label="Fraud")
        ax.set_title(f"{col}  (d={separability.get(col, 0):.2f})", fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    dest = output_dir / "04_vfeature_distributions.png"
    save_figure(fig, dest)
    logger.info("Saved: %s", dest)
    return dest


# ── Chart 5 ───────────────────────────────────────────────────────────────────

def plot_correlation_analysis(
    df: pd.DataFrame,
    class_corr: pd.Series,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    """Heatmap of V-feature inter-correlations + bar of each feature vs. Class.

    Args:
        df:         Full dataset (used to compute heatmap).
        class_corr: Output of :func:`correlation_with_target`, sorted ascending.
        output_dir: Directory to save the chart.

    Returns:
        Path of the saved PNG.
    """
    v_feats = [f"V{i}" for i in range(1, 29)]
    subset = df[v_feats].corr()
    mask = np.triu(np.ones_like(subset, dtype=bool))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [3, 1]}
    )
    fig.suptitle(
        "Correlation Analysis -- V-Features & Class",
        fontsize=14, fontweight="bold", color=FG, y=1.01,
    )

    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(
        subset, mask=mask, cmap=cmap, center=0, ax=ax1,
        linewidths=0.3, linecolor=BG, cbar_kws={"shrink": 0.7},
    )
    ax1.set_title("V-Feature Inter-Correlation (lower triangle)", fontsize=11)
    ax1.tick_params(labelsize=7)

    bar_colours = [RED if v > 0 else BLUE for v in class_corr.values]
    ax2.barh(class_corr.index, class_corr.values, color=bar_colours, edgecolor=BG)
    ax2.axvline(0, color=FG, lw=0.8)
    ax2.set_title("Correlation with 'Class'", fontsize=11)
    ax2.set_xlabel("Pearson r")
    ax2.grid(True, alpha=0.3, axis="x")
    ax2.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    dest = output_dir / "05_correlation_analysis.png"
    save_figure(fig, dest)
    logger.info("Saved: %s", dest)
    return dest


# ── Chart 6 ───────────────────────────────────────────────────────────────────

def plot_feature_separability(
    separability: pd.Series, output_dir: pathlib.Path
) -> pathlib.Path:
    """Horizontal bar chart ranking features by Cohen's-d separability.

    Args:
        separability: Output of :func:`feature_separability`, sorted descending.
        output_dir:   Directory to save the chart.

    Returns:
        Path of the saved PNG.
    """
    sorted_sep = separability.sort_values(ascending=True)
    median_val = float(sorted_sep.median())
    bar_colours = [RED if v > median_val else BLUE for v in sorted_sep.values]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(sorted_sep.index, sorted_sep.values, color=bar_colours, edgecolor=BG)
    ax.axvline(
        median_val, color="yellow", ls="--", lw=1.2, alpha=0.7,
        label=f"Median = {median_val:.2f}",
    )
    ax.set_title(
        "Feature Separability (Cohen d proxy) -- Higher = More Discriminating",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("Separability Score")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    dest = output_dir / "06_feature_separability.png"
    save_figure(fig, dest)
    logger.info("Saved: %s", dest)
    return dest
