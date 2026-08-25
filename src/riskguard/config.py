"""
riskguard.config
~~~~~~~~~~~~~~~~
Central configuration loaded from environment variables (or a ``.env`` file).

Every other module imports ``Settings`` rather than reading ``os.environ``
directly, so configuration changes only require editing one place.

Usage::

    from riskguard.config import Settings
    settings = Settings()
    print(settings.output_dir)
"""

import os
import pathlib
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load .env if present — does nothing when running in an environment where
# variables are already set (CI, Docker, etc.).
load_dotenv()


@dataclass
class Settings:
    """Application-wide configuration resolved from environment variables.

    All attributes have sensible defaults so the project works out of the box
    with no `.env` file.
    """

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_path: pathlib.Path | None = field(default=None)
    """Absolute or relative path to a local ``creditcard.csv``.
    When ``None`` the loader falls back to ``kagglehub``."""

    kaggle_dataset_slug: str = field(default="mlg-ulb/creditcardfraud")
    """Kaggle dataset identifier used by ``kagglehub``."""

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: pathlib.Path = field(default=pathlib.Path("outputs"))
    """Root directory for all generated charts, reports, and model artefacts."""

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = field(default="INFO")
    """Python logging level (DEBUG / INFO / WARNING / ERROR)."""

    # ── Reproducibility ───────────────────────────────────────────────────────
    random_seed: int = field(default=42)
    """Fixed seed for train/test splits, samplers, and model initialisations."""

    def __post_init__(self) -> None:
        # Override defaults with environment variables where set.
        raw_path = os.environ.get("DATASET_PATH", "")
        if raw_path:
            self.dataset_path = pathlib.Path(raw_path)

        self.kaggle_dataset_slug = os.environ.get(
            "KAGGLE_DATASET_SLUG", self.kaggle_dataset_slug
        )
        self.output_dir = pathlib.Path(
            os.environ.get("OUTPUT_DIR", str(self.output_dir))
        )
        self.log_level = os.environ.get("LOG_LEVEL", self.log_level)
        seed_env = os.environ.get("RANDOM_SEED")
        if seed_env is not None:
            self.random_seed = int(seed_env)

        # Ensure output directory exists.
        self.output_dir.mkdir(parents=True, exist_ok=True)
