"""Stage 6 — longitudinal marking centerlines (Phase 7, not implemented)."""

from __future__ import annotations

from shapely.geometry import LineString

from roadmarking.inference.predictor import InstancePrediction


def extract_centerline(prediction: InstancePrediction) -> list[LineString]:
    """Skeletonize a mask and fit a 2D centerline in world coordinates.

    Broken-line dashes stay separate geometries; grouping is handled in Stage 7.
    """
    raise NotImplementedError("Centerline vectorization is implemented in Phase 7.")
