"""Road-surface *assessment* heuristics on a 1 m grid (Phase 2 analysis only).

This is not the road-surface extraction algorithm (Phase 3). It flags grid cells that are
plausibly part of a large, smooth, open hard surface so that density and intensity
can be measured where road markings actually lie, rather than over the whole bounding box.
The result is a *candidate* mask: it also contains other flat open surfaces (car parks,
pavements, bare ground) and must be reported as an estimate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from roadmarking.analysis.lidar_stats import GridAccumulator


@dataclass(frozen=True)
class SurfaceCriteria:
    min_points: int = 4  # per 1 m cell, to trust z-range / return stats
    min_single_return_frac: float = 0.95  # vegetation produces multiple returns
    max_z_range_m: float = 0.15  # flat within the cell (excludes kerbs, cars, walls)
    max_slope: float = 0.12  # rise/run between neighbouring cells (~7 degrees)
    # Keep only large connected smooth surfaces. Roofs are cut off from the ground by
    # walls (non-flat / steep cells) and form separate, smaller components. A height-above-
    # local-minimum test was rejected: it removes roads on embankments (measured on the
    # client data: 80 % of a visible carriageway rejected).
    min_component_m2: int = 200

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def candidate_surface_mask(
    grid: GridAccumulator, criteria: SurfaceCriteria | None = None
) -> dict[str, np.ndarray]:
    """Return the candidate mask and each individual criterion mask (all ``(ny, nx)`` bool)."""
    c = criteria or SurfaceCriteria()
    if grid.spec.cell != 1.0:
        raise ValueError("criteria are defined for a 1 m grid")
    occ = grid.count >= c.min_points
    with np.errstate(invalid="ignore", divide="ignore"):
        single = np.where(occ, grid.single_return / np.maximum(grid.count, 1), 0.0)
    zmin = np.where(grid.count > 0, grid.z_min, np.nan).astype(np.float64)
    zr = grid.z_range()

    # Slope from z_min differences to valid 4-neighbours (max absolute difference / 1 m).
    slope = np.zeros_like(zmin)
    for axis in (0, 1):
        for shift in (1, -1):
            nb = np.roll(zmin, shift, axis=axis)
            d = np.abs(zmin - nb)
            d = np.where(np.isnan(d), 0.0, d)
            slope = np.maximum(slope, d)

    masks = {
        "enough_points": occ,
        "single_return": single >= c.min_single_return_frac,
        "flat_cell": np.nan_to_num(zr, nan=np.inf) <= c.max_z_range_m,
        "low_slope": slope <= c.max_slope,
    }
    smooth = np.logical_and.reduce(list(masks.values()))
    lab, n = ndimage.label(smooth)  # 4-connectivity: diagonal leaks across walls excluded
    large = np.zeros_like(smooth)
    if n:
        sizes = ndimage.sum(smooth, lab, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, bool)
        keep[1:] = sizes >= c.min_component_m2
        large = keep[lab]
    masks["large_component"] = large
    masks["candidate"] = smooth & large
    return masks


def mask_fractions(masks: dict[str, np.ndarray], occupied: np.ndarray) -> dict[str, float]:
    """Percentage of occupied cells passing each criterion."""
    n = max(int(occupied.sum()), 1)
    return {k: round(100.0 * float((v & occupied).sum()) / n, 2) for k, v in masks.items()}
