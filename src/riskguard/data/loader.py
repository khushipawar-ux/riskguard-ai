"""
riskguard.data.loader
~~~~~~~~~~~~~~~~~~~~~
Dataset acquisition layer.

Responsibilities
----------------
* Try to load ``creditcard.csv`` from a local path.
* Fall back to ``kagglehub`` automatic download when no local file is found.
* Raise a typed :exc:`DataLoadError` on unrecoverable failure.

No analysis or transformation happens here — this module is purely I/O.
"""

import pathlib
from typing import Optional

import pandas as pd

from riskguard.utils.logging import get_logger

logger = get_logger(__name__)


class DataLoadError(RuntimeError):
    """Raised when the dataset cannot be located or loaded."""


class DataLoader:
    """Loads the Credit Card Fraud dataset from a local path or Kaggle.

    Args:
        local_path:         Optional explicit path to ``creditcard.csv``.
        kaggle_slug:        Kaggle dataset slug passed to ``kagglehub``.
        fallback_paths:     Additional local paths to probe before attempting
                            a remote download.
    """

    _DEFAULT_FALLBACKS: tuple[pathlib.Path, ...] = (
        pathlib.Path("creditcard.csv"),
        pathlib.Path("data") / "creditcard.csv",
    )

    def __init__(
        self,
        local_path: Optional[pathlib.Path] = None,
        kaggle_slug: str = "mlg-ulb/creditcardfraud",
        fallback_paths: Optional[tuple[pathlib.Path, ...]] = None,
    ) -> None:
        self._local_path = local_path
        self._kaggle_slug = kaggle_slug
        self._fallback_paths = fallback_paths or self._DEFAULT_FALLBACKS

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """Return the raw dataset as a :class:`pandas.DataFrame`.

        Resolution order:
        1. Explicit ``local_path`` (if set).
        2. ``fallback_paths`` probed in sequence.
        3. ``kagglehub`` remote download.

        Raises:
            DataLoadError: When no source is available.
        """
        csv = self._resolve_csv_path()
        logger.info("Loading dataset from %s", csv)
        df = pd.read_csv(csv)
        logger.info(
            "Dataset loaded: %d rows x %d columns", df.shape[0], df.shape[1]
        )
        return df

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_csv_path(self) -> pathlib.Path:
        """Return the first resolvable CSV path or raise :exc:`DataLoadError`."""
        # 1. Explicit path.
        if self._local_path is not None:
            if self._local_path.exists():
                logger.debug("Using explicit local path: %s", self._local_path)
                return self._local_path
            raise DataLoadError(
                f"Explicit dataset path does not exist: {self._local_path}"
            )

        # 2. Fallback paths.
        for candidate in self._fallback_paths:
            if candidate.exists():
                logger.debug("Found local dataset at fallback path: %s", candidate)
                return candidate

        # 3. Remote download.
        return self._download_via_kagglehub()

    def _download_via_kagglehub(self) -> pathlib.Path:
        """Attempt to download the dataset via ``kagglehub``."""
        logger.info(
            "No local dataset found. Downloading via kagglehub (slug=%s) ...",
            self._kaggle_slug,
        )
        try:
            import kagglehub  # optional dependency — only needed for download

            download_dir = kagglehub.dataset_download(self._kaggle_slug)
            csv = pathlib.Path(download_dir) / "creditcard.csv"
            if not csv.exists():
                raise DataLoadError(
                    f"kagglehub download succeeded but creditcard.csv not found in {download_dir}"
                )
            logger.info("Download complete: %s", csv)
            return csv
        except ImportError as exc:
            raise DataLoadError(
                "kagglehub is not installed. Run `pip install kagglehub` or "
                "place creditcard.csv in the project root."
            ) from exc
        except Exception as exc:
            raise DataLoadError(
                f"kagglehub download failed: {exc}\n"
                "Place creditcard.csv in the project root and retry, or set "
                "KAGGLE_USERNAME / KAGGLE_KEY environment variables."
            ) from exc
