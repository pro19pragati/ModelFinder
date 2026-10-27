"""Logging helpers shared by the API layer and the experiment workers."""

from __future__ import annotations

import logging
import sys
from typing import Optional

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-32s | %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once (safe to call multiple times)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    # Optuna is chatty; keep warnings only.
    logging.getLogger("optuna").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


def quiet_external_loggers() -> None:
    for name in ("optuna", "matplotlib", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)


def short_id(value: Optional[str], length: int = 8) -> str:
    if not value:
        return "-"
    return value if len(value) <= length else value[:length]
