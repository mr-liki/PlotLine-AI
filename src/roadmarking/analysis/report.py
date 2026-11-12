"""Phase 2 report builder: derives density, intensity, road-surface and preview results
from the cached streaming statistics (``scripts/inspect_lidar.py stats``) and a second,
window-only pass over each readable file.

Every number written here is either an exact count/statistic over the decoded valid
points, or explicitly labelled as an estimate (``"kind": "estimate"``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from roadmarking.analysis import figures as fig
from roadmarking.analysis.acquisition import across_track_density, corridor_geometry
from roadmarking.analysis.lidar_stats import (
    GridAccumulator,
    GridSpec,
    density_summary,
)
from roadmarking.analysis.sampling import (
    Window,
    bev_raster,
    fill_for_display,
    nn_spacing,
    read_windows,
    robust_scale,
)
from roadmarking.analysis.surface import SurfaceCriteria, candidate_surface_mask, mask_fractions
from roadmarking.utils.logging import get_logger

log = get_logger(__name__)

PREVIEW_GSDS = (0.025, 0.05, 0.10)
PREVIEW_HALF_M = 20.0  # 40 m x 40 m preview windows
SAMPLE_HALF_M = 4.0  # 8 m x 8 m density/spacing samples (>= 90 % road cover)
SAMPLE_MIN_ROAD_FRAC = 0.9
SAMPLES_PER_FILE = 8
PREVIEWS_PER_FILE = 2
MARKING_WIDTH_M = 0.15
CORRIDOR_BUFFER_M = 5  # road candidate + 5 m keeps kerbs and edge markings
TILE_OPTIONS = ((0.05, 1024, 128), (0.05, 1024, 256), (0.025, 1024, 128), (0.10, 1024, 128))
# 4 float32 channels (max, mean intensity, z_min, normalised) + uint16 count = 18 B/px
RASTER_BYTES_PER_PX = 18
BLOCK_M = 100  # along-track blocks for regional intensity variation


@dataclass
class LoadedFile:
    name: str
    path: Path
    record: dict[str, Any]
    grid: GridAccumulator
    intensity_hist: np.ndarray


def _load(cache: Path, raw_dir: Path) -> tuple[list[LoadedFile], list[dict[str, Any]]]:
    loaded, all_records = [], []
    for jf in sorted(cache.glob("*.json")):
        rec = json.loads(jf.read_text(encoding="utf-8"))
        all_records.append(rec)
        npz = cache / f"{jf.stem}_grid.npz"
        if "grid" not in rec or not npz.exists() or rec.get("streamed_point_count", 0) == 0:
            continue
        d = np.load(npz)
        g = rec["grid"]
        acc = GridAccumulator(GridSpec(g["x_min"], g["y_max"], g["cell_m"], g["nx"], g["ny"]))
        acc.count, acc.single_return = d["count"], d["single_return"]
        acc.intensity_sum, acc.z_min, acc.z_max = d["intensity_sum"], d["z_min"], d["z_max"]
        loaded.append(LoadedFile(rec["file"], raw_dir / rec["file"], rec, acc, d["intensity_hist"]))
    return loaded, all_records


def _cell_intensity_contrast(lf: LoadedFile, cand: np.ndarray) -> np.ndarray:
    """Per-cell mean intensity / median of candidate cells within 15 m (NaN off-road)."""
    mi = lf.grid.mean_intensity()
    road = np.where(cand, mi, np.nan)
    bg = _block_median(road, 15)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cand, road / bg, np.nan)


def _pick_windows(
    lf: LoadedFile, cand: np.ndarray, contrast: np.ndarray, rng: np.random.Generator
) -> tuple[list[Window], list[Window], dict]:
    """Preview windows: highest density of bright (paint-like) road cells, plus one
    median-scoring window to avoid cherry-picking. Samples: random 8 m boxes with >= 90 %
    road-candidate cover (only road-candidate points inside them are measured)."""
    spec = lf.grid.spec
    bright = np.nan_to_num(contrast, nan=0.0) > 1.6
    k = int(2 * PREVIEW_HALF_M)
    score = ndimage.uniform_filter(bright.astype(np.float32), size=k, mode="constant")
    road_frac = ndimage.uniform_filter(cand.astype(np.float32), size=k, mode="constant")
    score = np.where(road_frac > 0.2, score, 0)
    previews: list[Window] = []
    taken = np.zeros_like(cand)
    order = np.argsort(score, axis=None)[::-1]
    for flat in order:
        if len(previews) >= PREVIEWS_PER_FILE - 1 or score.flat[flat] <= 0:
            break
        r, c = np.unravel_index(flat, score.shape)
        if taken[r, c]:
            continue
        cx, cy = spec.cell_center(np.array([r]), np.array([c]))
        previews.append(
            Window.centered(
                f"{Path(lf.name).stem}_p{len(previews) + 1}_dense",
                float(cx[0]),
                float(cy[0]),
                PREVIEW_HALF_M,
            )
        )
        taken[max(0, r - 3 * k) : r + 3 * k, max(0, c - 3 * k) : c + 3 * k] = True
    positive = np.flatnonzero((score > 0).ravel() & ~taken.ravel())
    if positive.size:
        med_val = np.median(score.flat[positive])
        flat = positive[np.argmin(np.abs(score.flat[positive] - med_val))]
        r, c = np.unravel_index(flat, score.shape)
        cx, cy = spec.cell_center(np.array([r]), np.array([c]))
        previews.append(
            Window.centered(
                f"{Path(lf.name).stem}_p{len(previews) + 1}_typical",
                float(cx[0]),
                float(cy[0]),
                PREVIEW_HALF_M,
            )
        )
    # Density samples: 10 m boxes fully inside candidate road surface.
    cover = ndimage.uniform_filter(cand.astype(np.float32), size=int(2 * SAMPLE_HALF_M))
    rows, cols = np.nonzero((cover >= SAMPLE_MIN_ROAD_FRAC) & cand)
    samples: list[Window] = []
    if rows.size:
        pick = rng.choice(rows.size, size=min(SAMPLES_PER_FILE, rows.size), replace=False)
        for i, j in enumerate(pick):
            cx, cy = spec.cell_center(rows[j : j + 1], cols[j : j + 1])
            samples.append(
                Window.centered(
                    f"{Path(lf.name).stem}_s{i + 1}", float(cx[0]), float(cy[0]), SAMPLE_HALF_M
                )
            )
    meta = {
        "bright_cell_rule": "candidate cell mean intensity > 1.6 x local road median",
        "bright_cells": int(bright.sum()),
    }
    return previews, samples, meta


def _regional_intensity(lf: LoadedFile, cand: np.ndarray) -> dict[str, Any]:
    """Median road-cell mean intensity per 100 m block (variation across the corridor)."""
    mi = lf.grid.mean_intensity()
    ny, nx = cand.shape
    vals = []
    for r0 in range(0, ny, BLOCK_M):
        for c0 in range(0, nx, BLOCK_M):
            sel = cand[r0 : r0 + BLOCK_M, c0 : c0 + BLOCK_M]
            if sel.sum() >= 500:
                vals.append(float(np.median(mi[r0 : r0 + BLOCK_M, c0 : c0 + BLOCK_M][sel])))
    v = np.array(vals)
    if v.size == 0:
        return {"blocks": 0}
    return {
        "blocks": int(v.size),
        "block_median_intensity": {
            "min": round(float(v.min()), 1),
            "p50": round(float(np.median(v)), 1),
            "max": round(float(v.max()), 1),
        },
        "max_over_min": round(float(v.max() / max(v.min(), 1e-9)), 2),
        "values": v.round(1),
    }


def _write_geotiff(path: Path, img: np.ndarray, transform, nodata: float) -> None:
    import rasterio

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=img.shape[1],
        height=img.shape[0],
        count=1,
        dtype="float32",
        transform=transform,
        nodata=nodata,
        compress="deflate",
    ) as dst:  # no CRS: none is defined in the source
        dst.write(np.where(np.isnan(img), nodata, img).astype(np.float32), 1)


def _window_analysis(
    pts: dict[str, np.ndarray],
    w: Window,
    lf: LoadedFile,
    cand: np.ndarray,
    out_dir: Path,
    render: bool,
) -> dict[str, Any]:
    """Density/spacing on road-candidate points; optional multi-GSD previews."""
    res: dict[str, Any] = {"window": w.name, "bounds": [w.x_min, w.y_min, w.x_max, w.y_max]}
    if "x" not in pts:
        res["points"] = 0
        return res
    x, y, z, inten = pts["x"], pts["y"], pts["z"], pts["intensity"].astype(np.float64)
    r, c = lf.grid.spec.index(x, y)
    on_road = cand[r, c]
    res["points"] = int(x.size)
    res["road_candidate_points"] = int(on_road.sum())
    road_area = float(np.unique(r[on_road] * lf.grid.spec.nx + c[on_road]).size)
    res["road_candidate_area_m2"] = road_area
    if on_road.sum() > 10:
        res["road_density_pts_m2"] = round(on_road.sum() / max(road_area, 1.0), 2)
        res["nn_spacing_road"] = nn_spacing(x[on_road], y[on_road])
        res["nn_spacing_road_single_returns"] = nn_spacing(
            x[on_road & (pts["number_of_returns"] == 1)],
            y[on_road & (pts["number_of_returns"] == 1)],
        )
        ri = inten[on_road]
        res["road_intensity"] = {
            f"p{q}": float(np.percentile(ri, q)) for q in (1, 5, 50, 95, 99, 99.9)
        }
        if "point_source_id" in pts:
            ids, cnt = np.unique(pts["point_source_id"][on_road], return_counts=True)
            res["road_points_by_point_source_id"] = {
                int(i): int(n) for i, n in zip(ids, cnt, strict=True)
            }
            per = {}
            for i in ids[cnt >= 2000]:
                sel = on_road & (pts["point_source_id"] == i)
                per[int(i)] = {
                    "p50": float(np.median(inten[sel])),
                    "p95": float(np.percentile(inten[sel], 95)),
                }
            res["road_intensity_by_point_source_id"] = per
        if "gps_time" in pts:
            g = np.sort(pts["gps_time"][on_road])
            gaps = np.flatnonzero(np.diff(g) > 10.0)
            res["road_passes_gps_gap_10s"] = int(gaps.size + 1)
    if not render:
        return res

    # Road mask at pixel level (from 1 m candidate cells) for per-GSD coverage metrics.
    res["gsd"] = {}
    for gsd in PREVIEW_GSDS:
        bev = bev_raster(x, y, z, inten, w, gsd)
        cols_px = np.arange(bev.shape[1])
        rows_px = np.arange(bev.shape[0])
        cx = w.x_min + (cols_px + 0.5) * gsd
        cy = w.y_max - (rows_px + 0.5) * gsd
        px_x = np.repeat(cx[None, :], bev.shape[0], 0)
        px_y = np.repeat(cy[:, None], bev.shape[1], 1)
        gr, gc = lf.grid.spec.index(px_x, px_y)
        # index() clips to the edge; pixels outside the surveyed grid are never road.
        road_px = cand[gr, gc] & lf.grid.spec.inside(px_x, px_y)
        nod = bev.nodata
        counts_road = bev.count[road_px]
        mean_road = float(counts_road.mean()) if road_px.any() else 0.0
        res["gsd"][str(gsd)] = {
            "pixels": int(nod.size),
            "road_pixels": int(road_px.sum()),
            "road_nodata_pct": (
                round(100.0 * float(nod[road_px].mean()), 2) if road_px.any() else None
            ),
            # Share of empty pixels expected from random (Poisson) sampling at the measured
            # mean density; observed >> expected means systematic gaps (scan pattern/holes).
            "road_nodata_pct_poisson_expected": round(100.0 * float(np.exp(-mean_road)), 2),
            "road_mean_points_per_pixel": round(mean_road, 3),
            "road_points_per_pixel": (
                {f"p{q}": float(np.percentile(counts_road, q)) for q in (5, 25, 50, 75, 95)}
                if road_px.any()
                else None
            ),
            "marking_width_px": round(MARKING_WIDTH_M / gsd, 2),
        }
        stem = f"{w.name}_gsd{round(gsd * 1000):03d}mm"
        valid = ~nod
        scaled = robust_scale(bev.i_max, 1, 99.5, valid=valid & road_px)
        filled, fmask = fill_for_display(scaled, max_distance_px=1)
        dens = np.where(valid, bev.count.astype(float), np.nan)
        fig.save_overview(
            out_dir / f"{stem}.png",
            [
                (
                    "max intensity — observed only (magenta = no data)",
                    fig.gray_with_nodata(scaled, nod),
                    {},
                ),
                (
                    "display fill ≤1 px (cyan tint = NOT observed)",
                    fig.gray_with_nodata(filled, nod & ~fmask, fmask),
                    {},
                ),
                ("points per pixel", dens, {"cmap": "viridis"}),
                (
                    "road candidate (1 m cells)",
                    road_px.astype(float),
                    {"cmap": "gray", "vmin": 0, "vmax": 1},
                ),
            ],
            bev.transform,
            f"{w.name} | GSD {gsd * 100:.1f} cm | origin x={w.x_min:.3f} y={w.y_max:.3f} "
            f"(source CRS, undefined in file)",
        )
        _write_geotiff(out_dir / f"{stem}_imax.tif", bev.i_max, bev.transform, -1.0)
        _write_geotiff(out_dir / f"{stem}_count.tif", bev.count.astype(float), bev.transform, -1.0)
        if gsd == 0.05:
            res["normalization_figure"] = _normalization_figure(
                bev, road_px, out_dir / f"{w.name}_normalization.png", w
            )
    return res


def _normalization_figure(bev, road_px: np.ndarray, path: Path, w: Window) -> dict[str, Any]:
    """Raw vs global-percentile vs local-background normalisation (observed pixels only)."""
    img = bev.i_max
    valid = ~bev.nodata
    rv = img[valid & road_px]
    if rv.size < 100:
        return {}
    raw = robust_scale(img, 0, 100, valid=valid & road_px)
    glob = robust_scale(img, 1, 99, valid=valid & road_px)
    # Local background: median of ALL observed pixels per 2.5 m block (a 15 cm marking
    # cannot move a 2.5 m block median). Using road pixels only would leave the
    # background undefined wherever the road mask is false.
    win = max(3, round(2.5 / bev.transform.a) | 1)
    bg = _block_median(np.where(valid, img, np.nan), win)
    with np.errstate(invalid="ignore", divide="ignore"):
        local = img / bg
    local_s = robust_scale(local, 1, 99, valid=valid & road_px)
    fig.save_overview(
        path,
        [
            ("raw (min–max)", fig.gray_with_nodata(raw, bev.nodata), {}),
            ("global p1–p99 of road pixels", fig.gray_with_nodata(glob, bev.nodata), {}),
            (
                "local: I / road median (2.5 m blocks)",
                fig.gray_with_nodata(local_s, bev.nodata),
                {},
            ),
        ],
        bev.transform,
        f"{w.name} — intensity normalisation candidates (GSD 5 cm)",
    )
    lv = local[valid & road_px]
    lv = lv[np.isfinite(lv)]
    return {
        "road_bg_median_raw": float(np.median(rv)),
        "road_p99_over_p50_raw": round(float(np.percentile(rv, 99) / np.median(rv)), 2),
        "road_p99_over_p50_local": (
            round(float(np.percentile(lv, 99) / np.median(lv)), 2) if lv.size else None
        ),
    }


def _block_median(img: np.ndarray, b: int) -> np.ndarray:
    """NaN-aware median over non-overlapping b x b blocks, broadcast back to full size.

    A fast approximation of a local background level (exact medians per block).
    """
    import warnings

    ny, nx = img.shape
    py, px = -ny % b, -nx % b
    pad = np.pad(img, ((0, py), (0, px)), constant_values=np.nan)
    blocks = pad.reshape(pad.shape[0] // b, b, pad.shape[1] // b, b).swapaxes(1, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(blocks.reshape(*blocks.shape[:2], -1), axis=2)
    return np.repeat(np.repeat(med, b, axis=0), b, axis=1)[:ny, :nx]


def _file_overview(lf: LoadedFile, masks: dict[str, np.ndarray], out_dir: Path) -> str:
    g = lf.grid
    occ = g.occupied
    mi = g.mean_intensity()
    road_vals = mi[masks["candidate"]]
    scaled = robust_scale(mi, 1, 99, valid=masks["candidate"] if road_vals.size else None)
    dens = np.where(occ, g.count.astype(float), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        single = np.where(occ, g.single_return / np.maximum(g.count, 1), np.nan)
    from affine import Affine

    tr = Affine(g.spec.cell, 0, g.spec.x_min, 0, -g.spec.cell, g.spec.y_max)
    name = f"{Path(lf.name).stem}_overview_1m.png"
    fig.save_overview(
        out_dir / name,
        [
            ("mean intensity, 1 m (magenta = no points)", fig.gray_with_nodata(scaled, ~occ), {}),
            (
                "points per m² (log)",
                dens,
                {"cmap": "viridis", "norm": fig.LogNorm(1, max(2, np.nanmax(dens)))},
            ),
            ("single-return fraction", single, {"cmap": "magma", "vmin": 0, "vmax": 1}),
            (
                "road-surface candidate",
                masks["candidate"].astype(float),
                {"cmap": "gray", "vmin": 0, "vmax": 1},
            ),
        ],
        tr,
        f"{lf.name} — 1 m grid overview (exact per-cell counts)",
    )
    return name


def build_report(cfg: Any, files: list[Path], seed: int = 42) -> dict[str, Any]:
    reports = cfg.path("outputs.reports")
    cache = reports / "lidar_cache"
    prev_dir = reports / "lidar_previews"
    prev_dir.mkdir(parents=True, exist_ok=True)
    loaded, records = _load(cache, cfg.path("data.raw_pointclouds"))
    rng = np.random.default_rng(seed)
    criteria = SurfaceCriteria()
    result: dict[str, Any] = {
        "files": [],
        "criteria": criteria.as_dict(),
        "preview_gsds_m": list(PREVIEW_GSDS),
    }
    all_hists: dict[str, np.ndarray] = {}
    dens_series: dict[str, np.ndarray] = {}
    road_dens_series: dict[str, np.ndarray] = {}
    regional: dict[str, np.ndarray] = {}

    for rec in records:
        if not any(lf.name == rec["file"] for lf in loaded):
            result["files"].append(
                {
                    "file": rec["file"],
                    "size_bytes": rec.get("size_bytes"),
                    "usable": False,
                    "error": rec.get("error"),
                    "header": rec.get("header"),
                }
            )

    for lf in loaded:
        log.info("report: %s", lf.name)
        masks = candidate_surface_mask(lf.grid, criteria)
        cand = masks["candidate"]
        occ = lf.grid.occupied
        mi = lf.grid.mean_intensity()
        entry: dict[str, Any] = {k: v for k, v in lf.record.items() if k != "grid"}
        entry["usable"] = True
        entry["grid_1m"] = lf.record["grid"]
        bbox_area = (lf.record["xyz_max"][0] - lf.record["xyz_min"][0]) * (
            lf.record["xyz_max"][1] - lf.record["xyz_min"][1]
        )
        entry["density"] = {
            "bbox_area_m2": round(bbox_area, 1),
            "bbox_pts_per_m2_NOT_road_density": round(
                lf.record["streamed_point_count"] / bbox_area, 2
            ),
            "occupied_cell_fraction_of_grid_pct": round(100.0 * float(occ.mean()), 2),
            "all_occupied_cells": density_summary(lf.grid),
            "road_candidate_cells": density_summary(lf.grid, cand),
        }
        entry["road_surface"] = {
            "criteria_pass_pct_of_occupied_cells": mask_fractions(masks, occ),
            "candidate_area_m2": int(cand.sum()),
            "kind": "estimate (heuristic candidate mask, not an extraction)",
        }
        # Gaps inside the road corridor: empty cells enclosed by road-candidate surface.
        closed = ndimage.binary_closing(cand, structure=np.ones((5, 5)))
        holes = closed & ~occ
        entry["road_surface"]["empty_1m_cells_inside_road_corridor"] = int(holes.sum())
        entry["intensity_road_cells"] = (
            {f"p{q}": round(float(np.percentile(mi[cand], q)), 1) for q in (1, 5, 50, 95, 99)}
            if cand.any()
            else None
        )
        reg = _regional_intensity(lf, cand)
        if "values" in reg:
            regional[Path(lf.name).stem] = reg.pop("values")
        entry["intensity_regional_100m_blocks"] = reg
        entry["overview_figure"] = _file_overview(lf, masks, prev_dir)
        valid_frac = lf.record["streamed_point_count"] / max(lf.record["header"]["point_count"], 1)
        entry["acquisition"] = corridor_geometry(lf.grid, lf.record["gps_time"], valid_frac)
        entry["acquisition"]["across_track_density"] = across_track_density(lf.grid)
        corridor = ndimage.binary_dilation(cand, iterations=CORRIDOR_BUFFER_M)
        entry["road_corridor"] = {
            "buffer_m": CORRIDOR_BUFFER_M,
            "area_m2": int(corridor.sum()),
            "tiles": {
                f"{gsd}m_{size}px_overlap{ov}px": _tile_count(corridor, gsd, size, ov)
                for gsd, size, ov in TILE_OPTIONS
            },
        }

        contrast = _cell_intensity_contrast(lf, cand)
        previews, samples, meta = _pick_windows(lf, cand, contrast, rng)
        entry["window_selection"] = meta
        pts = read_windows(lf.path, previews + samples)
        entry["previews"] = [
            _window_analysis(pts.get(w.name, {}), w, lf, cand, prev_dir, True) for w in previews
        ]
        entry["density_samples"] = [
            _window_analysis(pts.get(w.name, {}), w, lf, cand, prev_dir, False) for w in samples
        ]
        result["files"].append(entry)

        all_hists[Path(lf.name).stem] = lf.intensity_hist
        dens_series[Path(lf.name).stem + " all"] = lf.grid.count[occ].astype(float)
        road_dens_series[Path(lf.name).stem] = lf.grid.count[cand].astype(float)

    if all_hists:
        vmax = int(max(np.nonzero(h)[0].max() for h in all_hists.values()))
        fig.save_histograms(
            prev_dir / "intensity_histograms.png",
            all_hists,
            vmax,
            "Intensity distribution per file (all valid points)",
        )
        fig.save_density_hist(
            prev_dir / "road_density_hist.png",
            road_dens_series,
            "Points per m² on road-candidate 1 m cells",
        )
        fig.save_density_hist(
            prev_dir / "all_density_hist.png",
            dens_series,
            "Points per m² on all occupied 1 m cells",
        )
        if regional:
            fig.save_boxplot(
                prev_dir / "regional_road_intensity.png",
                regional,
                "median road-cell mean intensity",
                "Road intensity per 100 m block, per file",
            )
    result["usable_files"] = len(loaded)
    result["survey_totals"] = _survey_totals(result)
    return result


def _tile_count(corridor: np.ndarray, gsd: float, size_px: int, overlap_px: int) -> int:
    """Axis-aligned tiles (fixed global stride grid) that touch the 1 m road corridor."""
    stride_m = (size_px - overlap_px) * gsd
    rows, cols = np.nonzero(corridor)
    ty = np.floor(rows / stride_m).astype(np.int64)
    tx = np.floor(cols / stride_m).astype(np.int64)
    return int(np.unique(ty * 1_000_000 + tx).size)


def _survey_totals(result: dict[str, Any]) -> dict[str, Any]:
    """Dataset totals and raster/tile budgets (estimates labelled)."""
    files = result["files"]
    usable = [f for f in files if f.get("usable")]
    corridor_m2 = sum(f["road_corridor"]["area_m2"] for f in usable)
    tiles: dict[str, int] = {}
    for f in usable:
        for k, v in f["road_corridor"]["tiles"].items():
            tiles[k] = tiles.get(k, 0) + v
    raster = {}
    for gsd in (0.025, 0.05, 0.10):
        px = corridor_m2 / gsd**2
        raster[str(gsd)] = {
            "corridor_megapixels": round(px / 1e6, 1),
            "in_memory_GB_at_18_bytes_per_px": round(px * RASTER_BYTES_PER_PX / 1e9, 2),
            "bbox_gigapixels_if_rasterised_naively": round(
                sum(f["density"]["bbox_area_m2"] for f in usable) / gsd**2 / 1e9, 2
            ),
        }
    return {
        "files_total": len(files),
        "files_usable": len(usable),
        "laz_bytes_total": sum(f.get("size_bytes") or 0 for f in files),
        "header_points_total": sum((f.get("header") or {}).get("point_count", 0) for f in files),
        "valid_points_total": sum(f.get("streamed_point_count", 0) for f in usable),
        "corridor_length_m_total": round(
            sum(f["acquisition"]["corridor_length_m"] for f in usable), 1
        ),
        "road_candidate_area_m2_total": sum(f["road_surface"]["candidate_area_m2"] for f in usable),
        "road_corridor_area_m2_total": corridor_m2,
        "tiles_touching_corridor": tiles,
        "raster_budget": raster,
        "raster_bytes_per_px_assumed": RASTER_BYTES_PER_PX,
        "kind": "estimate (corridor = road candidate dilated by buffer; tile grid axis-aligned)",
    }
