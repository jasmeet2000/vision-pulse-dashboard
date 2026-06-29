"""
VisionPulse Dashboard — Logging Configuration

Provides a centralized logging factory that configures:
    - Rotating file handler  → logs/visionpulse.log
    - Console (stderr) handler for development visibility
    - Consistent formatting across all modules

Usage::

    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Camera opened on device 0")
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from config.settings import AppSettings, LoggingSettings

# ---------------------------------------------------------------------------
# Module-level state — guards against duplicate handler attachment
# ---------------------------------------------------------------------------
_initialized: bool = False
_settings: LoggingSettings = AppSettings().logging


def _ensure_log_directory(log_dir: Path) -> None:
    """Create the log directory if it does not exist."""
    log_dir.mkdir(parents=True, exist_ok=True)


def _setup_root_logger(settings: LoggingSettings) -> None:
    """
    Configure the root logger with a file handler and console handler.

    Called once on first ``get_logger()`` invocation.  Subsequent calls
    are no-ops thanks to the ``_initialized`` guard.
    """
    global _initialized  # noqa: PLW0603
    if _initialized:
        return

    _ensure_log_directory(settings.log_dir)

    log_path = settings.log_dir / settings.log_file
    level = getattr(logging, settings.log_level.upper(), logging.DEBUG)

    # ---- Formatter --------------------------------------------------------
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ---- Rotating file handler --------------------------------------------
    file_handler = RotatingFileHandler(
        filename=str(log_path),
        maxBytes=settings.max_bytes,
        backupCount=settings.backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # ---- Console handler --------------------------------------------------
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)  # Less verbose on console
    console_handler.setFormatter(formatter)

    # ---- Root logger ------------------------------------------------------
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    _initialized = True
    logging.getLogger(__name__).debug(
        "Logging initialized — file: %s, level: %s", log_path, settings.log_level
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a named logger, initializing the logging subsystem on first call.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        A configured logger instance.
    """
    _setup_root_logger(_settings)
    return logging.getLogger(name)
