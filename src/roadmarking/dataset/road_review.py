"""Phase 4 step 3: review of Phase 3 road-surface candidates (read-only, no reprocessing).

Evidence sources: (a) per-tile 5 cm layers (normalized, observed, road_state); (b) the
Phase 2 1 m all-point grid (count, intensity sum) against the Phase 3 1 m surface state,
which shows what the geometric selection excluded. Flags are review prompts, not truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy import ndimage

BRIGHT = 0.5  # normalized intensity regarded as paint-like for review statistics
EDGE_PX = 20  # 1 m band at 5 cm


def tile_statistics(tile: dict) -> dict:
    with rasterio.open(tile["layers"]["road_state"]) as s:
        road = s.read(1) > 0
    with rasterio.open(tile["layers"]["observed"]) as s:
        obs = s.read(1) > 0
    with rasterio.open(tile["layers"]["normalized"]) as s:
        norm = s.read(1)
    n = road.size
    cand = int(road.sum())
    bright = obs & (norm >= BRIGHT)
    core = ndimage.binary_erosion(road, structure=np.ones((3, 3)), iterations=EDGE_PX)
    edge = road & ~core
    labels, ncomp = ndimage.label(road, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel())[1:] if ncomp else np.array([])
    stats = {
        "tile_id": tile["tile_id"],
        "source": Path(tile["source_file"]).stem,
        "candidate_pct": 100 * cand / n,
        "observed_pct": 100 * obs.sum() / n,
        "missing_in_candidate_pct": 100 * (road & ~obs).sum() / cand if cand else None,
        "bright_pct_of_observed": 100 * bright.sum() / obs.sum() if obs.sum() else None,
        "bright_in_edge_band_pct": (
            100 * (bright & edge).sum() / bright.sum() if bright.sum() else None
        ),
        "candidate_components": int(ncomp),
        "small_components_lt_1m2": int((sizes < 400).sum()),
        "observed_outside_candidate": int((obs & ~road).sum()),
    }
    flags = []
    if cand and stats["missing_in_candidate_pct"] > 50:
        flags.append("sparse_observation_in_candidate")
    if stats["candidate_pct"] < 2:
        flags.append("little_candidate_surface")
    if ncomp > 3:
        flags.append("fragmented_candidate")
    if (
        stats["bright_in_edge_band_pct"] is not None
        and stats["bright_in_edge_band_pct"] > 30
        and bright.sum() > 2000
    ):
        flags.append("paint_like_pixels_at_candidate_edge")
    if stats["observed_outside_candidate"]:
        flags.append("observations_outside_candidate")
    if stats["observed_pct"] < NEAR_EMPTY_PCT:
        flags.append("near_empty_tile")
    stats["flags"] = flags
    return stats


NEAR_EMPTY_PCT = 1.0  # tiles with <1 % observed pixels are excluded from training


def pick_review_tiles(stats: list[dict]) -> list[str]:
    """Most informative examples: edge-cut paint, fragmentation, sparse, and a clean tile."""
    usable = [s for s in stats if s["observed_pct"] >= 5]
    picks = []
    for key in ("bright_in_edge_band_pct", "candidate_components", "missing_in_candidate_pct"):
        ranked = sorted(usable, key=lambda s: (-(s[key] or 0), s["tile_id"]))
        picks += [s["tile_id"] for s in ranked if s["tile_id"] not in picks][:1]
    clean = [s["tile_id"] for s in sorted(usable, key=lambda s: s["tile_id"]) if not s["flags"]]
    picks += [t for t in clean[len(clean) // 2 : len(clean) // 2 + 1] if t not in picks]
    return picks


def source_comparison(source: str, phase2_cache: Path, surface_tif: Path) -> dict | None:
    """All-point 1 m intensity vs 1 m candidate state on the identical 1 m lattice."""
    grid_npz = phase2_cache / f"{source}_grid.npz"
    meta_path = phase2_cache / f"{source}.json"
    if not (grid_npz.is_file() and surface_tif.is_file() and meta_path.is_file()):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    g = np.load(grid_npz)
    with rasterio.open(surface_tif) as s:
        state = s.read(1)
        transform = s.transform
    origin = (meta["grid"]["x_min"], meta["grid"]["y_max"])
    if g["count"].shape != state.shape or (transform.c, transform.f) != origin:
        return {"comparable": False, "reason": "Phase 2 grid and surface state lattices differ"}
    count = g["count"]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_i = np.where(count > 0, g["intensity_sum"] / count, np.nan)
    cand = state > 0
    occupied = count > 0
    near = ndimage.binary_dilation(cand, iterations=5) & ~cand & occupied  # within 5 m
    p95_cand = float(np.nanpercentile(mean_i[cand], 95)) if cand.any() else None
    bright_near = near & (mean_i >= p95_cand) if p95_cand else np.zeros_like(near)
    labels, ncomp = ndimage.label(cand, structure=np.ones((3, 3)))
    sizes = np.sort(np.bincount(labels.ravel())[1:])[::-1] if ncomp else np.array([])
    # Flat rejected cells (z range < 10 cm) within 15 m of the candidate: likely road-like
    # surface excluded by the geometric selection (or adjacent flat terrain - needs review).
    zr = np.where(occupied, g["z_max"] - g["z_min"], np.inf)
    flat_near = (zr < 0.10) & (count >= 8) & ndimage.binary_dilation(cand, iterations=15) & ~cand
    flat_lab, nflat = ndimage.label(flat_near, structure=np.ones((3, 3)))
    flat_sizes = np.bincount(flat_lab.ravel())[1:] if nflat else np.array([])
    return {
        "flat_rejected_cells_within_15m": int(flat_near.sum()),
        "flat_rejected_regions_ge_100m2": int((flat_sizes >= 100).sum()),
        "flat_rejected_cells_in_regions_ge_100m2": int(flat_sizes[flat_sizes >= 100].sum()),
        "comparable": True,
        "occupied_cells": int(occupied.sum()),
        "candidate_cells": int(cand.sum()),
        "rejected_occupied_cells": int((occupied & ~cand).sum()),
        "rejected_cells_within_5m_of_candidate": int(near.sum()),
        "candidate_cell_mean_intensity_p95": p95_cand,
        "bright_rejected_cells_within_5m": int(bright_near.sum()),
        "candidate_components": int(ncomp),
        "largest_component_cells": int(sizes[0]) if len(sizes) else 0,
        "cells_outside_largest_component": int(sizes[1:].sum()) if len(sizes) > 1 else 0,
        "_arrays": (mean_i, cand, bright_near, flat_near, transform),
    }


def overview_figure(path: Path, source: str, comp: dict) -> None:
    mean_i, cand, bright_near, flat_near, t = comp["_arrays"]
    h, w = cand.shape
    ext = (t.c, t.c + w * t.a, t.f + h * t.e, t.f)
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    vmax = np.nanpercentile(mean_i, 99)
    for ax in axes:
        ax.imshow(
            np.log1p(mean_i), cmap="gray", vmax=np.log1p(vmax), extent=ext, interpolation="nearest"
        )
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.set_xlabel("native X")
        ax.set_ylabel("native Y")
    overlay = np.zeros((h, w, 4))
    overlay[cand] = [0, 0.8, 1, 0.45]
    overlay[flat_near] = [1, 0.85, 0, 0.8]
    overlay[bright_near] = [1, 0.1, 0.1, 1]
    axes[1].imshow(overlay, extent=ext, interpolation="nearest")
    axes[0].set_title(f"{source}: all-point 1 m mean intensity (log)")
    axes[1].set_title(
        "cyan = Phase 3 candidate; yellow = flat (<10 cm z-range) rejected cells within 15 m;\n"
        "red = bright rejected cells within 5 m",
        fontsize=9,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def tile_side_by_side(path: Path, tiles: list[dict], stats: dict[str, dict]) -> None:
    fig, axes = plt.subplots(len(tiles), 3, figsize=(15, 5 * len(tiles)), squeeze=False)
    for row, tile in zip(axes, tiles, strict=True):
        x0, y0, x1, y1 = tile["native_xy_bbox"]
        ext = (x0, x1, y0, y1)
        with rasterio.open(tile["layers"]["normalized"]) as s:
            norm = s.read(1)
        with rasterio.open(tile["layers"]["road_state"]) as s:
            road = s.read(1)
        with rasterio.open(tile["layers"]["observed"]) as s:
            obs = s.read(1)
        cmap = plt.get_cmap("gray").with_extremes(bad="#ff00ff")
        row[0].imshow(
            np.where(norm < -1, np.nan, norm),
            cmap=cmap,
            vmin=0,
            vmax=1,
            extent=ext,
            interpolation="nearest",
        )
        row[1].imshow(road, cmap="viridis", vmin=0, vmax=2, extent=ext, interpolation="nearest")
        row[2].imshow(obs, cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="nearest")
        s = stats[tile["tile_id"]]
        row[0].set_title(f"{tile['tile_id']}\nnormalized intensity", fontsize=8)
        row[1].set_title(
            f"road_state candidate ({s['candidate_pct']:.1f}%)\n"
            f"flags: {', '.join(s['flags']) or 'none'}",
            fontsize=8,
        )
        row[2].set_title(f"observed ({s['observed_pct']:.1f}%)", fontsize=8)
        for ax in row:
            ax.ticklabel_format(useOffset=False, style="plain")
            ax.tick_params(labelsize=6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90)
    plt.close(fig)
