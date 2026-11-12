"""Exact native-grid geometry: world/pixel mapping, polyline buffering, coverage rasters.

Conventions (identical to Phase 3): the affine maps pixel *corners*; pixel (row, col)
covers [col, col+1) x [row, row+1) in continuous pixel space and its centre is at
(col + 0.5, row + 0.5). Rasterization is centre-sampled (``all_touched=False``) on a
supersampled grid, so coverage fractions are geometric, not "any touch" dilations.
"""

from __future__ import annotations

import math

import numpy as np
import shapely
from affine import Affine
from rasterio import features as rio_features
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry


def tile_affine(tile: dict) -> Affine:
    a, b, c, d, e, f = tile["affine"]
    return Affine(a, b, c, d, e, f)


def tile_box(tile: dict) -> Polygon:
    return box(*tile["native_xy_bbox"])


def world_to_pixel(affine: Affine, x, y):
    """Continuous (row, col); integer values are pixel corners, +0.5 is a centre."""
    inv = ~affine
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    col = inv.a * x + inv.b * y + inv.c
    row = inv.d * x + inv.e * y + inv.f
    return row, col


def pixel_center_to_world(affine: Affine, row, col):
    c = np.asarray(col, dtype=float) + 0.5
    r = np.asarray(row, dtype=float) + 0.5
    return affine.a * c + affine.b * r + affine.c, affine.d * c + affine.e * r + affine.f


def pixel_index(affine: Affine, x, y):
    """Integer (row, col) of the pixel containing (x, y)."""
    row, col = world_to_pixel(affine, x, y)
    return np.floor(row).astype(np.int64), np.floor(col).astype(np.int64)


def flatten_2d(geom: BaseGeometry) -> BaseGeometry:
    return shapely.force_2d(geom)


def line_parts(geom: BaseGeometry) -> list[LineString]:
    geom = flatten_2d(geom)
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    raise TypeError(f"Expected a line geometry, got {geom.geom_type}")


def polygon_parts(geom: BaseGeometry) -> list[Polygon]:
    geom = flatten_2d(geom)
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    raise TypeError(f"Expected a polygon geometry, got {geom.geom_type}")


def painted_area_from_centerline(geom: BaseGeometry, width: float) -> BaseGeometry:
    """Buffer a centre-axis polyline by half its physical width.

    Flat caps keep the painted end exactly at the digitised endpoint (no extra half-width
    beyond each dash end); mitre joins with a limit avoid rounding curves while bounding
    spikes at sharp vertices. Each part is buffered separately, so dashes stored as
    separate parts are not bridged. Width must be in the coordinate units of ``geom``.
    """
    if not (isinstance(width, int | float) and math.isfinite(width) and width > 0):
        raise ValueError(f"Invalid painted width: {width!r}")
    pieces = [
        part.buffer(width / 2.0, cap_style="flat", join_style="mitre", mitre_limit=2.0)
        for part in line_parts(geom)
        if part.length > 0
    ]
    if not pieces:
        raise ValueError("Line has no nonzero-length part")
    return shapely.union_all(pieces)


def pixel_window(affine: Affine, geom: BaseGeometry, shape: tuple[int, int], pad: int = 1):
    """Clipped (row0, row1, col0, col1) pixel window containing ``geom``; None if outside."""
    minx, miny, maxx, maxy = geom.bounds
    rows, cols = world_to_pixel(affine, [minx, maxx, minx, maxx], [miny, miny, maxy, maxy])
    r0 = max(math.floor(rows.min()) - pad, 0)
    r1 = min(math.ceil(rows.max()) + pad, shape[0])
    c0 = max(math.floor(cols.min()) - pad, 0)
    c1 = min(math.ceil(cols.max()) + pad, shape[1])
    if r0 >= r1 or c0 >= c1:
        return None
    return r0, r1, c0, c1


def coverage_fraction(
    geom: BaseGeometry, affine: Affine, shape: tuple[int, int], supersample: int = 4
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Fraction (0..1) of each pixel covered by ``geom`` inside its bounding window.

    Holes are preserved because rasterio burns polygon interiors with even-odd rings.
    Returns (coverage_window, window) or None when the geometry misses the grid.
    """
    if supersample < 1:
        raise ValueError("supersample must be >= 1")
    geom = flatten_2d(geom)
    if geom.is_empty:
        return None
    window = pixel_window(affine, geom, shape)
    if window is None:
        return None
    r0, r1, c0, c1 = window
    k = supersample
    sub = affine @ Affine.translation(c0, r0) @ Affine.scale(1.0 / k)
    burned = rio_features.rasterize(
        [(geom, 1)],
        out_shape=((r1 - r0) * k, (c1 - c0) * k),
        transform=sub,
        fill=0,
        all_touched=False,
        dtype="uint8",
    )
    cov = burned.reshape(r1 - r0, k, c1 - c0, k).mean(axis=(1, 3), dtype=np.float64)
    if not cov.any():
        return None
    return cov.astype(np.float32), window
