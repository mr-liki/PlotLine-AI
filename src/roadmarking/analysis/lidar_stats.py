"""Streaming per-file statistics for LAS/LAZ point clouds.

Files are read in chunks with ``laspy`` so that memory use is bounded regardless of file
size. Input files are opened read-only and never modified.

Everything collected here is an *observation* of the file contents. Derived quantities
(e.g. density per grid cell) are computed from exact counts, not estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from roadmarking.io.pointcloud import ChunkReadReport, iter_laz_chunks
from roadmarking.utils.logging import get_logger

log = get_logger(__name__)

INTENSITY_BINS = 65536  # full uint16 range -> exact percentiles from the histogram


@dataclass
class GridSpec:
    """North-up grid aligned to a fixed origin (x_min, y_max) with square cells."""

    x_min: float
    y_max: float
    cell: float
    nx: int
    ny: int

    @classmethod
    def covering(cls, mins: np.ndarray, maxs: np.ndarray, cell: float) -> GridSpec:
        if cell <= 0:
            raise ValueError("cell must be > 0")
        x_min = np.floor(mins[0] / cell) * cell
        y_max = np.ceil(maxs[1] / cell) * cell
        nx = int(np.floor((maxs[0] - x_min) / cell)) + 1
        ny = int(np.floor((y_max - mins[1]) / cell)) + 1
        return cls(float(x_min), float(y_max), float(cell), nx, ny)

    def index(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (row, col) integer indices; points outside the grid are clipped."""
        col = np.floor((x - self.x_min) / self.cell).astype(np.int64)
        row = np.floor((self.y_max - y) / self.cell).astype(np.int64)
        return np.clip(row, 0, self.ny - 1), np.clip(col, 0, self.nx - 1)

    def inside(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """True where (x, y) falls inside the grid (``index`` clips; use this to mask)."""
        col = np.floor((np.asarray(x) - self.x_min) / self.cell)
        row = np.floor((self.y_max - np.asarray(y)) / self.cell)
        return (col >= 0) & (col < self.nx) & (row >= 0) & (row < self.ny)

    def cell_center(self, row: np.ndarray, col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = self.x_min + (np.asarray(col) + 0.5) * self.cell
        y = self.y_max - (np.asarray(row) + 0.5) * self.cell
        return x, y

    def as_dict(self) -> dict[str, Any]:
        return {
            "x_min": self.x_min,
            "y_max": self.y_max,
            "cell_m": self.cell,
            "nx": self.nx,
            "ny": self.ny,
        }


@dataclass
class GridAccumulator:
    """Per-cell exact counts and simple aggregates, accumulated chunk by chunk."""

    spec: GridSpec
    count: np.ndarray = field(init=False)
    single_return: np.ndarray = field(init=False)
    intensity_sum: np.ndarray = field(init=False)
    z_min: np.ndarray = field(init=False)
    z_max: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        shape = (self.spec.ny, self.spec.nx)
        self.count = np.zeros(shape, np.int32)
        self.single_return = np.zeros(shape, np.int32)
        self.intensity_sum = np.zeros(shape, np.float64)
        self.z_min = np.full(shape, np.inf, np.float32)
        self.z_max = np.full(shape, -np.inf, np.float32)

    def add(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray, intensity: np.ndarray, single: np.ndarray
    ) -> None:
        row, col = self.spec.index(x, y)
        flat = row * self.spec.nx + col
        n = self.spec.nx * self.spec.ny
        self.count += np.bincount(flat, minlength=n).reshape(self.count.shape).astype(np.int32)
        self.single_return += (
            np.bincount(flat, weights=single, minlength=n)
            .reshape(self.count.shape)
            .astype(np.int32)
        )
        self.intensity_sum += np.bincount(flat, weights=intensity, minlength=n).reshape(
            self.count.shape
        )
        zf = z.astype(np.float32)
        np.minimum.at(self.z_min.reshape(-1), flat, zf)
        np.maximum.at(self.z_max.reshape(-1), flat, zf)

    @property
    def occupied(self) -> np.ndarray:
        return self.count > 0

    def mean_intensity(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.count > 0, self.intensity_sum / np.maximum(self.count, 1), np.nan)

    def z_range(self) -> np.ndarray:
        return np.where(self.count > 0, self.z_max - self.z_min, np.nan)


@dataclass
class FileStats:
    """Streaming statistics of one point-cloud file."""

    point_count: int = 0
    intensity_hist: np.ndarray = field(default_factory=lambda: np.zeros(INTENSITY_BINS, np.int64))
    class_counts: np.ndarray = field(default_factory=lambda: np.zeros(256, np.int64))
    return_matrix: np.ndarray = field(default_factory=lambda: np.zeros((8, 8), np.int64))
    scan_angle_hist: np.ndarray = field(default_factory=lambda: np.zeros(256, np.int64))
    user_data_counts: np.ndarray = field(default_factory=lambda: np.zeros(256, np.int64))
    flag_counts: dict[str, int] = field(default_factory=dict)
    point_source_counts: dict[int, int] = field(default_factory=dict)
    xyz_min: np.ndarray = field(default_factory=lambda: np.full(3, np.inf))
    xyz_max: np.ndarray = field(default_factory=lambda: np.full(3, -np.inf))
    gps_time_min: float = float("inf")
    gps_time_max: float = float("-inf")
    rgb_min: np.ndarray | None = None
    rgb_max: np.ndarray | None = None
    rgb_all_zero: int = 0
    grid: GridAccumulator | None = None
    read_error: str | None = None
    chunk_report: ChunkReadReport | None = None

    def update(self, pts: Any, dims: set[str]) -> None:
        x, y, z = np.asarray(pts.x), np.asarray(pts.y), np.asarray(pts.z)
        n = x.shape[0]
        if n == 0:
            return
        self.point_count += n
        inten = np.asarray(pts.intensity)
        self.intensity_hist += np.bincount(inten, minlength=INTENSITY_BINS)
        self.class_counts += np.bincount(np.asarray(pts.classification), minlength=256)
        rn = np.clip(np.asarray(pts.return_number), 0, 7)
        nr = np.clip(np.asarray(pts.number_of_returns), 0, 7)
        self.return_matrix += np.bincount(rn * 8 + nr, minlength=64).reshape(8, 8)
        if "scan_angle_rank" in dims:
            sa = np.asarray(pts.scan_angle_rank).astype(np.int16) + 128
            self.scan_angle_hist += np.bincount(sa, minlength=256)
        if "user_data" in dims:
            self.user_data_counts += np.bincount(np.asarray(pts.user_data), minlength=256)
        for flag in ("synthetic", "key_point", "withheld", "edge_of_flight_line"):
            if flag in dims:
                self.flag_counts[flag] = self.flag_counts.get(flag, 0) + int(
                    np.count_nonzero(np.asarray(getattr(pts, flag)))
                )
        if "point_source_id" in dims:
            ids, cnt = np.unique(np.asarray(pts.point_source_id), return_counts=True)
            for i, c in zip(ids.tolist(), cnt.tolist(), strict=True):
                self.point_source_counts[i] = self.point_source_counts.get(i, 0) + c
        xyz = np.stack([x.min(), y.min(), z.min()]), np.stack([x.max(), y.max(), z.max()])
        self.xyz_min = np.minimum(self.xyz_min, xyz[0])
        self.xyz_max = np.maximum(self.xyz_max, xyz[1])
        if "gps_time" in dims:
            g = np.asarray(pts.gps_time)
            self.gps_time_min = min(self.gps_time_min, float(g.min()))
            self.gps_time_max = max(self.gps_time_max, float(g.max()))
        if {"red", "green", "blue"} <= dims:
            rgb = np.stack([np.asarray(pts.red), np.asarray(pts.green), np.asarray(pts.blue)])
            mn, mx = rgb.min(axis=1), rgb.max(axis=1)
            self.rgb_min = mn if self.rgb_min is None else np.minimum(self.rgb_min, mn)
            self.rgb_max = mx if self.rgb_max is None else np.maximum(self.rgb_max, mx)
            self.rgb_all_zero += int(np.count_nonzero(rgb.max(axis=0) == 0))
        if self.grid is not None:
            self.grid.add(x, y, z, inten, (nr == 1).astype(np.float64))


def collect_file_stats(path: str | Path, grid_cell: float | None = 1.0) -> FileStats:
    """Stream ``path`` once (chunk by chunk) and accumulate :class:`FileStats` + grid."""
    import laspy

    path = Path(path)
    with laspy.open(path, mode="r") as reader:
        hdr = reader.header
        dims = set(hdr.point_format.dimension_names)
    stats = FileStats()
    if grid_cell is not None and hdr.point_count > 0:
        stats.grid = GridAccumulator(GridSpec.covering(hdr.mins, hdr.maxs, grid_cell))
    report = ChunkReadReport(path, 0, 0)
    try:
        for pts in iter_laz_chunks(path, report):
            stats.update(pts, dims)
    except Exception as exc:  # unreadable beyond chunk level: keep what was read, flag it
        stats.read_error = f"{type(exc).__name__}: {exc}"
        log.error(
            "%s: read stopped after %d of %d points: %s",
            path.name,
            stats.point_count,
            hdr.point_count,
            stats.read_error,
        )
    stats.chunk_report = report
    if (report.bad_chunks or report.corrupt_chunks) and stats.read_error is None:
        stats.read_error = (
            f"{len(report.bad_chunks)} unreadable and "
            f"{len(report.corrupt_chunks)} corrupt LAZ chunk(s); "
            f"{report.dropped_points} points discarded"
        )
    log.info("%s: %d points streamed", path.name, stats.point_count)
    return stats


# --------------------------------------------------------------------------- summaries
def hist_percentiles(hist: np.ndarray, qs: list[float]) -> dict[str, int]:
    """Exact percentiles (lower interpolation) of integer data given its full histogram."""
    total = int(hist.sum())
    if total == 0:
        return {}
    cdf = np.cumsum(hist)
    out: dict[str, int] = {}
    for q in qs:
        rank = max(1, int(np.ceil(q / 100.0 * total)))
        out[f"p{q:g}"] = int(np.searchsorted(cdf, rank))
    return out


def intensity_summary(hist: np.ndarray) -> dict[str, Any]:
    total = int(hist.sum())
    if total == 0:
        return {"count": 0}
    values = np.nonzero(hist)[0]
    vmax = int(values.max())
    mean = float((np.arange(hist.size) * hist).sum() / total)
    # "Saturated" = at the observed maximum value AND that maximum is a sensor ceiling
    # (all bits set for 8/12/16-bit encodings). Otherwise report 0 and the top count.
    ceilings = {255, 4095, 65535}
    top_count = int(hist[vmax])
    return {
        "count": total,
        "min": int(values.min()),
        "max": vmax,
        "mean": round(mean, 2),
        "distinct_values": int(values.size),
        "zero_count": int(hist[0]),
        "zero_pct": round(100.0 * hist[0] / total, 4),
        "at_max_count": top_count,
        "at_max_pct": round(100.0 * top_count / total, 4),
        "saturated_pct": round(100.0 * top_count / total, 4) if vmax in ceilings else 0.0,
        "percentiles": hist_percentiles(hist, [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]),
        "value_step_gcd": int(np.gcd.reduce(values[values > 0])) if (values > 0).any() else None,
    }


def density_summary(grid: GridAccumulator, mask: np.ndarray | None = None) -> dict[str, Any]:
    """Points per m² over grid cells (optionally restricted by ``mask``).

    Cell density = exact count in the cell / cell area. Cells with zero points are
    reported separately (they are *missing coverage*, not low density).
    """
    area = grid.spec.cell**2
    sel = grid.occupied if mask is None else (mask & grid.occupied)
    d = grid.count[sel] / area
    if d.size == 0:
        return {"cells": 0}
    q = np.percentile(d, [1, 5, 25, 50, 75, 95, 99])
    med = float(q[3])
    return {
        "cells": int(d.size),
        "area_m2": float(d.size * area),
        "pts_per_m2": {
            k: round(float(v), 2)
            for k, v in zip(["p1", "p5", "p25", "p50", "p75", "p95", "p99"], q, strict=True)
        },
        "mean_pts_per_m2": round(float(d.mean()), 2),
        "cv": round(float(d.std() / d.mean()), 3),
        # Spacing implied by density for a uniform pattern: 1/sqrt(density).
        "implied_spacing_m_at_median": round(1.0 / np.sqrt(med), 4) if med > 0 else None,
        "cells_below_half_median_pct": round(100.0 * float((d < 0.5 * med).mean()), 2),
    }
