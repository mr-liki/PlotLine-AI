"""Acquisition geometry from the 1 m grid: corridor axis, length, swath width, platform
speed and point rate, and point density as a function of across-track offset.

Values are derived from exact per-cell counts and GPS times; anything that depends on the
derived corridor axis (length, width, speed, offsets) is an *estimate*.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from roadmarking.analysis.lidar_stats import GridAccumulator


def _axis_coords(grid: GridAccumulator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Along/across coordinates of occupied cells w.r.t. the point-weighted principal axis."""
    rows, cols = np.nonzero(grid.count > 0)
    x, y = grid.spec.cell_center(rows, cols)
    w = grid.count[rows, cols].astype(np.float64)
    xy = np.column_stack([x, y])
    mu = np.average(xy, axis=0, weights=w)
    cov = np.cov((xy - mu).T, aweights=w)
    _, evecs = np.linalg.eigh(cov)
    rel = xy - mu
    return rel @ evecs[:, 1], rel @ evecs[:, 0], w


def corridor_geometry(
    grid: GridAccumulator, gps_time: tuple[float, float], valid_fraction: float
) -> dict[str, Any]:
    """Corridor length, swath width, speed and point rate (estimates)."""
    along, across, _ = _axis_coords(grid)
    length = float(np.percentile(along, 99.5) - np.percentile(along, 0.5))
    bins = np.floor(along / 20.0).astype(np.int64)
    widths = []
    for b in np.unique(bins):
        a = across[bins == b]
        if a.size > 50:
            widths.append(float(np.percentile(a, 99) - np.percentile(a, 1)))
    dur = float(gps_time[1] - gps_time[0])
    n = int(grid.count.sum())
    return {
        "kind": "estimate (principal-axis corridor; straight-axis approximation)",
        "corridor_length_m": round(length, 1),
        "swath_width_m_p5_p50_p95": (
            [round(float(np.percentile(widths, q)), 1) for q in (5, 50, 95)] if widths else None
        ),
        "gps_duration_s": round(dur, 2),
        "platform_speed_m_s": round(length / dur, 2) if dur > 0 else None,
        "point_rate_pts_s": round(n / dur / max(valid_fraction, 1e-9)) if dur > 0 else None,
    }


def across_track_density(
    grid: GridAccumulator,
    bin_m: float = 10.0,
    along_bin_m: float = 25.0,
    max_offset_m: float = 90.0,
) -> list[dict[str, Any]]:
    """Median points/m² of occupied cells per across-track offset bin.

    The offset is measured from the point-weighted median across-position of each
    ``along_bin_m`` slice (follows gentle curvature).
    """
    along, across, w = _axis_coords(grid)
    b = np.floor(along / along_bin_m).astype(np.int64)
    off = np.empty_like(across)
    for k in np.unique(b):
        m = b == k
        order = np.argsort(across[m])
        cw = np.cumsum(w[m][order])
        centre = across[m][order][np.searchsorted(cw, cw[-1] / 2)]
        off[m] = across[m] - centre
    out = []
    area = grid.spec.cell**2
    for lo in np.arange(-max_offset_m, max_offset_m, bin_m):
        m = (off >= lo) & (off < lo + bin_m)
        if m.sum() >= 200:
            out.append(
                {
                    "offset_m": [float(lo), float(lo + bin_m)],
                    "median_pts_per_m2": float(np.median(w[m]) / area),
                    "cells": int(m.sum()),
                }
            )
    return out
