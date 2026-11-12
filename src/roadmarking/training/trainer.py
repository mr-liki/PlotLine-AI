"""Mask2Former fine-tuning loop (Phase 5, not implemented)."""

from __future__ import annotations

from roadmarking.config import Config


def train(cfg: Config) -> None:
    """Fine-tune Mask2Former Swin-S on road-marking classes from ``classes.yaml``."""
    raise NotImplementedError("Fine-tuning is implemented in Phase 5.")
