"""Window extraction, nearest-neighbour spacing and BEV preview rasters.

All rasters here are *observations*: a pixel is valid only if at least one point fell in
it. Any filled/interpolated values are produced by :func:`fill_for_display`, which returns
a separate mask so filled pixels can never be mistaken for measured ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine

from roadmarking.io.pointcloud import iter_laz_chunks
from roadmarking.rasterization.rasterizer import north_up_transform

WINDOW_FIELDS = (
    "x",
    "y",
    "z",
    "intensity",
    "return_number",
    "number_of_returns",
    "classification",
    "point_source_id",
    "gps_time",
    "red",
    "green",
    "blue",
)


@dataclass(frozen=True)
class Window:
    name: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x >= self.x_min) & (x < self.x_max) & (y >= self.y_min) & (y < self.y_max)

    @property
    def size(self) -> tuple[float, float]:
        return self.x_max - self.x_min, self.y_max - self.y_min

    @classmethod
    def centered(cls, name: str, cx: float, cy: float, half: float) -> Window:
        return cls(name, cx - half, cy - half, cx + half, cy + half)


def read_windows(path: str | Path, windows: list[Window]) -> dict[str, dict[str, np.ndarray]]:
    """Stream ``path`` once (chunk-table reader) and return the points inside each window."""
    import laspy

    with laspy.open(Path(path), mode="r") as reader:
        dims = set(reader.header.point_format.dimension_names)
    parts: dict[str, dict[str, list[np.ndarray]]] = {
        w.name: {f: [] for f in WINDOW_FIELDS} for w in windows
    }
    for pts in iter_laz_chunks(path):
        x, y = np.asarray(pts.x), np.asarray(pts.y)
        for w in windows:
            m = w.contains(x, y)
            if not m.any():
                continue
            for f in WINDOW_FIELDS:
                if f in ("x", "y", "z") or f in dims:
                    parts[w.name][f].append(np.asarray(getattr(pts, f))[m])
    return {
        name: {f: np.concatenate(v) for f, v in fields.items() if v}
        for name, fields in parts.items()
    }


def nn_spacing(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """2D nearest-neighbour distance statistics (exact, KD-tree).

    Coincident XY points (distance 0, e.g. multiple returns of one pulse) are reported
    separately and excluded from the spacing percentiles.
    """
    from scipy.spatial import cKDTree

    xy = np.column_stack([x, y])
    if len(xy) < 2:
        return {"points": len(xy)}
    d, _ = cKDTree(xy).query(xy, k=2)
    nn = d[:, 1]
    pos = nn[nn > 0]
    q = np.percentile(pos, [5, 25, 50, 75, 95]) if pos.size else [np.nan] * 5
    return {
        "points": len(xy),
        "coincident_xy_pct": round(100.0 * float((nn == 0).mean()), 3),
        "nn_m": {
            k: round(float(v), 4)
            for k, v in zip(["p5", "p25", "p50", "p75", "p95"], q, strict=True)
        },
        "nn_mean_m": round(float(pos.mean()), 4) if pos.size else None,
    }


@dataclass
class BevRaster:
    """Observed BEV raster. ``count == 0`` marks no-data pixels (no measurement)."""

    transform: Affine
    count: np.ndarray
    i_max: np.ndarray
    i_mean: np.ndarray
    i_min: np.ndarray
    z_min: np.ndarray

    @property
    def nodata(self) -> np.ndarray:
        return self.count == 0

    @property
    def shape(self) -> tuple[int, int]:
        return self.count.shape


def bev_raster(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, intensity: np.ndarray, window: Window, gsd: float
) -> BevRaster:
    """Bin points into a north-up grid with origin (window.x_min, window.y_max)."""
    w, h = window.size
    nx, ny = int(np.ceil(w / gsd - 1e-9)), int(np.ceil(h / gsd - 1e-9))
    transform = north_up_transform(window.x_min, window.y_max, gsd)
    col = np.floor((x - window.x_min) / gsd).astype(np.int64)
    row = np.floor((window.y_max - y) / gsd).astype(np.int64)
    ok = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
    flat = row[ok] * nx + col[ok]
    inten = intensity[ok].astype(np.float64)
    n = nx * ny
    count = np.bincount(flat, minlength=n)
    i_sum = np.bincount(flat, weights=inten, minlength=n)
    i_max = np.full(n, np.nan)
    i_min = np.full(n, np.nan)
    z_min = np.full(n, np.nan)
    has = count > 0
    i_max[has], i_min[has], z_min[has] = -np.inf, np.inf, np.inf
    np.maximum.at(i_max, flat, inten)
    np.minimum.at(i_min, flat, inten)
    np.minimum.at(z_min, flat, z[ok].astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        i_mean = np.where(has, i_sum / np.maximum(count, 1), np.nan)
    shape = (ny, nx)
    return BevRaster(
        transform,
        count.reshape(shape),
        i_max.reshape(shape),
        i_mean.reshape(shape),
        i_min.reshape(shape),
        z_min.reshape(shape),
    )


def fill_for_display(img: np.ndarray, max_distance_px: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-valid fill of NaN pixels within ``max_distance_px`` — DISPLAY ONLY.

    Returns ``(filled, filled_mask)``; ``filled_mask`` marks pixels that were not observed.
    """
    from scipy import ndimage

    nan = np.isnan(img)
    if not nan.any():
        return img.copy(), np.zeros_like(nan)
    dist, (ri, ci) = ndimage.distance_transform_edt(nan, return_indices=True)
    filled = img[ri, ci]
    use = nan & (dist <= max_distance_px)
    out = np.where(use, filled, img)
    return out, use


def robust_scale(
    img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0, valid: np.ndarray | None = None
) -> np.ndarray:
    """Map values to [0, 1] using percentiles of valid pixels (NaN stays NaN)."""
    vals = img[~np.isnan(img)] if valid is None else img[valid & ~np.isnan(img)]
    if vals.size == 0:
        return np.full_like(img, np.nan, dtype=np.float64)
    lo, hi = np.percentile(vals, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)
