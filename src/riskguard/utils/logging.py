"""
riskguard.utils.logging
~~~~~~~~~~~~~~~~~~~~~~~
Centralised logger factory.  All modules call ``get_logger(__name__)``
instead of using bare ``print()`` statements.
"""

import logging
import os
import sys
from typing import Optional


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Return a configured :class:`logging.Logger` for *name*.

    The log level is resolved (in priority order) from:
    1. The *level* argument, if supplied.
    2. The ``LOG_LEVEL`` environment variable.
    3. ``INFO`` as the default.

    Args:
        name:  Typically ``__name__`` of the calling module.
        level: Optional override (e.g. ``"DEBUG"``).

    Returns:
        A logger that writes to *stdout* with a timestamped format.
    """
    resolved_level = level or os.environ.get("LOG_LEVEL", "INFO")

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured — avoid duplicate handlers

    logger.setLevel(resolved_level.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(resolved_level.upper())

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger
