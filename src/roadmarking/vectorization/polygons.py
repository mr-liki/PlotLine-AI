"""Stage 6 — polygon extraction for other markings (Phase 7, not implemented)."""

from __future__ import annotations

from shapely.geometry import Polygon

from roadmarking.inference.predictor import InstancePrediction


def extract_polygons(prediction: InstancePrediction) -> list[Polygon]:
    """Trace contours (with holes) of a mask and return world-coordinate polygons."""
    raise NotImplementedError("Polygon vectorization is implemented in Phase 7.")
