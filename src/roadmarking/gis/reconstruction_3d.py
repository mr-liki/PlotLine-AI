"""Stage 8 — Z recovery from nearby road-surface points (Phase 8, not implemented)."""

from __future__ import annotations

from shapely.geometry.base import BaseGeometry

from roadmarking.io.pointcloud import PointCloud


def add_z_from_surface(geom: BaseGeometry, surface: PointCloud, radius_m: float) -> BaseGeometry:
    """Return a 3D copy of ``geom`` with Z sampled from valid road-surface points."""
    raise NotImplementedError("3D reconstruction is implemented in Phase 8.")
