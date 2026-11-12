"""Structured logging setup shared by scripts and pipeline stages."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """Format records as single-line JSON (for machine-readable logs)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(
    level: str = "INFO", log_file: Path | None = None, json_format: bool = False
) -> None:
    """Configure the root ``roadmarking`` logger once (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    formatter: logging.Formatter = (
        JsonFormatter()
        if json_format
        else logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    )
    logger = logging.getLogger("roadmarking")
    logger.setLevel(level.upper())
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``roadmarking`` namespace."""
    return logging.getLogger(name if name.startswith("roadmarking") else f"roadmarking.{name}")
