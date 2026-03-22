"""
Centralised logging configuration for AI Math Olympiad Solver.

Every module imports `get_logger(__name__)` to obtain a named logger
that writes simultaneously to the console and to a rotating file.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


_LOG_DIR = Path("logs")
_LOG_FILE = _LOG_DIR / "app.log"
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_BACKUP_COUNT = 5
_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_initialized: bool = False


def _setup_root_logger() -> None:
    """Configure the root logger once for the entire process."""
    global _initialized
    if _initialized:
        return

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(fmt=_FMT, datefmt=_DATE_FMT)

    # ── Console handler ──────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # ── Rotating file handler ────────────────────────────────────────────
    file_handler = RotatingFileHandler(
        filename=_LOG_FILE,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    _setup_root_logger()
    return logging.getLogger(name)
