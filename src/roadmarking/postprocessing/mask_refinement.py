"""Stage 5 — mask cleanup, tile merging, instance separation (Phase 6, not implemented)."""

from __future__ import annotations

from roadmarking.inference.predictor import InstancePrediction


def clean_mask(prediction: InstancePrediction, min_area_px: int) -> InstancePrediction | None:
    """Remove small fragments and refine the boundary of one instance mask."""
    raise NotImplementedError("Mask cleanup is implemented in Phase 6.")


def merge_tiles(
    predictions: list[InstancePrediction], iou_threshold: float
) -> list[InstancePrediction]:
    """Merge instances duplicated across overlapping tiles."""
    raise NotImplementedError("Overlapping-tile merging is implemented in Phase 6.")
