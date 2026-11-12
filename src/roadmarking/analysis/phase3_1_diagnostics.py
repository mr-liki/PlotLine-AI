"""Phase 3.1 step 1: why did Phase 3 (v1) accept or reject road cells?

Uses the saved Phase 3 1 m surface model (exact per-cell OLS statistics) to attribute
every rejection to the first failing v1 criterion, and one streaming read of the
original LAZ to render all-point intensity for a few windows. Read-only.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy import ndimage

from roadmarking.io.pointcloud import iter_pointcloud_chunks

REASONS = {
    0: "no points",
    1: "accepted (v1 candidate)",
    2: "fewer than min_points",
    3: "plane RMS above max_rms",
    4: "slope above max_slope",
    5: "neighbour step or small component",
}
REASON_COLORS = ListedColormap(["#000000", "#2ca02c", "#ff00ff", "#d62728", "#ff7f0e", "#1f77b4"])


def rejection_reasons(model: dict, criteria: dict) -> np.ndarray:
    n, rms = model["count"], model["rms"]
    slope = np.hypot(model["a"], model["b"])
    out = np.zeros(n.shape, np.uint8)
    obs = n > 0
    out[obs] = 5
    out[obs & (slope > criteria["max_slope"])] = 4
    out[obs & (rms > criteria["max_rms_m"])] = 3
    out[obs & (n < criteria["min_points"])] = 2
    out[model["state"] > 0] = 1
    return out


def band_statistics(model: dict, criteria: dict, band_cells: int = 15) -> dict:
    reasons = rejection_reasons(model, criteria)
    cand = model["state"] > 0
    near = ndimage.binary_dilation(cand, iterations=band_cells) & ~cand & (model["count"] > 0)
    counts = np.bincount(reasons[near], minlength=6)
    return dict(
        band_cells=band_cells,
        rejected_occupied_cells_in_band=int(near.sum()),
        by_first_failing_criterion={REASONS[i]: int(counts[i]) for i in range(2, 6)},
        rms_percentiles_in_band=np.percentile(
            model["rms"][near & (model["count"] >= 8)], [25, 50, 75]
        ).tolist(),
        rms_percentiles_candidates=np.percentile(model["rms"][cand], [25, 50, 75]).tolist(),
        points_per_cell_in_band=np.percentile(model["count"][near], [25, 50, 75]).tolist(),
        points_per_cell_candidates=np.percentile(model["count"][cand], [25, 50, 75]).tolist(),
    )


def read_windows(laz: Path, boxes: dict[str, list[float]], chunk_size=2_000_000) -> dict:
    parts = {k: [] for k in boxes}
    for _, cloud in iter_pointcloud_chunks(laz, chunk_size):
        x, y = cloud.xyz[:, 0], cloud.xyz[:, 1]
        for k, (x0, y0, x1, y1) in boxes.items():
            m = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
            if m.any():
                parts[k].append(np.column_stack((cloud.xyz[m], cloud.intensity[m])))
    return {k: np.concatenate(v) if v else np.empty((0, 4)) for k, v in parts.items()}


def window_diagnostic(
    path: Path,
    name: str,
    tile: dict,
    pts: np.ndarray,
    model: dict,
    transform,
    criteria: dict,
    calibration: dict,
) -> dict:
    """All-point intensity | v1 boundary + rejected smooth cells | reasons | cross-section."""
    import rasterio

    x0, y0, x1, y1 = tile["native_xy_bbox"]
    res = tile["resolution"]
    h, w = tile["height"], tile["width"]
    cols = np.floor((pts[:, 0] - x0) / res).astype(int)
    rows = np.floor((y1 - pts[:, 1]) / res).astype(int)
    ok = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    img = np.full((h, w), -1.0)
    np.maximum.at(img, (rows[ok], cols[ok]), pts[ok, 3])
    lo, hi = calibration["low"], calibration["high"]
    norm = np.where(img >= 0, np.clip((img - lo) / max(hi - lo, 1024), 0, 1), np.nan)

    reasons = rejection_reasons(model, criteria)
    c0 = int(np.floor((x0 - transform.c) / transform.a))
    r0 = int(np.floor((transform.f - y1) / -transform.e))
    nc, nr = int(np.ceil(w * res)), int(np.ceil(h * res))
    sub = reasons[max(r0, 0) : r0 + nr, max(c0, 0) : c0 + nc]
    cand = sub == 1
    strict = (
        (model["count"] >= criteria["min_points"])
        & (model["rms"] <= criteria["max_rms_m"])
        & (np.hypot(model["a"], model["b"]) <= criteria["max_slope"])
    )[max(r0, 0) : r0 + nr, max(c0, 0) : c0 + nc]
    smooth_rejected = strict & ~cand
    ext1 = (
        transform.c + max(c0, 0),
        transform.c + max(c0, 0) + sub.shape[1],
        transform.f - max(r0, 0) - sub.shape[0],
        transform.f - max(r0, 0),
    )
    with rasterio.open(tile["layers"]["observed"]) as s:
        observed = s.read(1) > 0
    with rasterio.open(tile["layers"]["road_state"]) as s:
        road = s.read(1) > 0
    missing = road & ~observed

    fig, ax = plt.subplots(1, 4, figsize=(26, 6.6))
    cmap = plt.get_cmap("gray").with_extremes(bad="#400040")
    ext = (x0, x1, y0, y1)
    ax[0].imshow(norm, cmap=cmap, vmin=0, vmax=1, extent=ext, interpolation="nearest")
    ax[0].set_title(
        "ALL points: max intensity (Phase 3 calibration), dark purple = no point", fontsize=9
    )
    ax[1].imshow(norm, cmap=cmap, vmin=0, vmax=1, extent=ext, interpolation="nearest")
    ov = np.zeros((*sub.shape, 4))
    ov[smooth_rejected] = (1, 0.8, 0, 0.55)
    ax[1].imshow(ov, extent=ext1, interpolation="nearest")
    ax[1].contour(
        cand, levels=[0.5], colors=["#00e5ff"], linewidths=1.0, extent=ext1, origin="upper"
    )
    mv = np.zeros((h, w, 4))
    mv[missing] = (1, 0, 1, 0.5)
    ax[1].imshow(mv, extent=ext, interpolation="nearest")
    ax[1].set_title(
        "cyan = v1 boundary; amber = rejected smooth cells;\n"
        "magenta = candidate pixels without observations",
        fontsize=9,
    )
    ax[2].imshow(sub, cmap=REASON_COLORS, vmin=-0.5, vmax=5.5, extent=ext1, interpolation="nearest")
    ax[2].set_title(
        "first failing v1 criterion: green accepted, magenta few points,\n"
        "red RMS, orange slope, blue step/small component",
        fontsize=9,
    )
    # Cross-section perpendicular to the candidate axis through the window centre.
    rr, cc = np.nonzero(road)
    if len(rr) > 10:
        cx = x0 + (cc + 0.5) * res
        cy = y1 - (rr + 0.5) * res
        c = np.c_[cx, cy]
        mu = c.mean(0)
        _, _, vt = np.linalg.svd(c - mu, full_matrices=False)
        d = vt[0]
        nrm = np.array([-d[1], d[0]])
        along = (pts[:, :2] - mu) @ d
        across = (pts[:, :2] - mu) @ nrm
        sel = np.abs(along) < 1.0
        ax[3].scatter(across[sel], pts[sel, 2], c=np.clip(pts[sel, 3], lo, hi), s=1, cmap="viridis")
        ca = (c - mu) @ nrm
        ax[3].axvspan(np.percentile(ca, 0.5), np.percentile(ca, 99.5), color="orange", alpha=0.15)
        ax[3].set_title(
            "cross-section (|along| < 1 m), colour = intensity;\norange = v1 candidate extent",
            fontsize=9,
        )
        ax[3].set_xlabel("across-track offset (native units)")
        ax[3].set_ylabel("Z")
    for a in ax[:3]:
        a.ticklabel_format(useOffset=False, style="plain")
        a.tick_params(labelsize=6)
    fig.suptitle(f"Phase 3 (v1) diagnosis — {name} — {tile['tile_id']}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=85)
    plt.close(fig)
    counts = np.bincount(sub.ravel(), minlength=6)
    return dict(
        tile_id=tile["tile_id"],
        points=len(pts),
        cells_by_reason={REASONS[i]: int(counts[i]) for i in range(6)},
        rejected_smooth_cells=int(smooth_rejected.sum()),
        candidate_pixels_without_observation=int(missing.sum()),
    )


def run_diagnostics(cfg, phase3_runs: dict, cases: dict, out_dir: Path) -> dict:
    import rasterio

    result = {}
    manifest = {
        t["tile_id"]: t
        for t in json.loads(
            (cfg.path("data.processed") / "phase3/metadata/tile_manifest.json").read_text()
        )
    }
    for source, items in cases.items():
        run = phase3_runs[source]
        masks = Path(run["output_directories"]["road_masks"])
        with np.load(masks / "surface_model.npz") as d:
            model = {k: d[k] for k in d.files}
        with rasterio.open(masks / "surface_state_1m.tif") as s:
            transform = s.transform
        criteria = run["criteria"]
        meta = json.loads((Path(run["output_directories"]["metadata"]) / "source.json").read_text())
        boxes = {name: manifest[tid]["native_xy_bbox"] for name, tid in items if tid in manifest}
        pts = read_windows(cfg.path("data.raw_pointclouds") / f"{source}.laz", boxes)
        windows = {}
        for name, tid in items:
            if tid in manifest:
                windows[name] = window_diagnostic(
                    out_dir / f"diagnosis_{source.split('_')[-1]}_{name}.png",
                    name,
                    manifest[tid],
                    pts[name],
                    model,
                    transform,
                    criteria,
                    meta["normalization"],
                )
        result[source] = dict(band=band_statistics(model, criteria), windows=windows)
    return result
