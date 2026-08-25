"""
riskguard.utils.plotting
~~~~~~~~~~~~~~~~~~~~~~~~
Shared dark-theme configuration and figure-save helper.

All chart-generating modules call ``apply_theme()`` once at import time
and use ``save_figure()`` so style choices live in exactly one place.
"""

import pathlib

import matplotlib
import matplotlib.pyplot as plt

# Palette — import these constants everywhere instead of repeating hex codes.
BG: str = "#0D1117"
FG: str = "#E6EDF3"
BLUE: str = "#4FC3F7"
RED: str = "#EF5350"
GRID_CLR: str = "#21262D"
LEGEND_BG: str = "#161B22"


def apply_theme() -> None:
    """Apply the RiskGuard dark theme to :mod:`matplotlib` globally.

    Call once — typically at the top of a script or visualiser module.
    Uses ``Agg`` backend so charts render without a display (CI / servers).
    """
    matplotlib.use("Agg")
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "axes.edgecolor": GRID_CLR,
            "axes.labelcolor": FG,
            "axes.titlecolor": FG,
            "xtick.color": FG,
            "ytick.color": FG,
            "text.color": FG,
            "grid.color": GRID_CLR,
            "grid.linewidth": 0.6,
            "legend.facecolor": LEGEND_BG,
            "legend.edgecolor": GRID_CLR,
            "font.family": "DejaVu Sans",
            "figure.dpi": 150,
        }
    )


def save_figure(fig: plt.Figure, path: pathlib.Path) -> None:
    """Save *fig* to *path* and close it to free memory.

    Args:
        fig:  The matplotlib figure to persist.
        path: Full file path including extension (e.g. ``outputs/chart.png``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
