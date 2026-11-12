"""Phase 3.1: before/after comparison of road extraction runs on one global grid.

All comparisons use identical input points and the identical 1 m / 5 cm lattices of the
two runs. Coverage differences are diagnostic observations; without verified road
boundaries they are not recall or precision measurements.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from scipy import ndimage

from roadmarking.preprocessing.road_surface import (
    CONFIDENCE_NAMES,
    EVIDENCE_NAMES,
)
from roadmarking.rasterization.pipeline import BLOCK, read_mosaic

CONF_COLORS = ListedColormap(["#000000", "#7f7f7f", "#ff00ff", "#ffbf00", "#2ca02c", "#1f77b4"])


def load_run(processing_json: Path) -> dict:
    result = json.loads(Path(processing_json).read_text(encoding="utf-8"))
    masks = Path(result["output_directories"]["road_masks"])
    with rasterio.open(masks / "surface_state_1m.tif") as src:
        state = src.read(1)
        transform = src.transform
    out = dict(result=result, state=state, transform=transform, masks=masks)
    if (masks / "refinement_model.npz").is_file():
        with np.load(masks / "refinement_model.npz") as data:
            out["refined"] = {k: data[k] for k in data.files}
    with np.load(masks / "surface_model.npz") as data:
        out["model"] = {k: data[k] for k in data.files}
    out["blocks_dir"] = Path(result["output_directories"]["rasters"]) / "blocks"
    out["manifest"] = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    out["histogram"] = np.load(Path(result["reports"]) / "intensity_histogram.npy")
    return out


def _align(a: dict, b: dict):
    """Crop both 1 m state grids to their common lattice window (identical origins)."""
    ta, tb = a["transform"], b["transform"]
    if (ta.a, ta.e) != (tb.a, tb.e):
        raise ValueError("Different cell sizes")
    dc = round((tb.c - ta.c) / ta.a)
    dr = round((tb.f - ta.f) / ta.e)
    if abs(ta.c + dc * ta.a - tb.c) > 1e-9 or abs(ta.f + dr * ta.e - tb.f) > 1e-9:
        raise ValueError("1 m lattices are not aligned")
    ha, wa = a["state"].shape
    hb, wb = b["state"].shape
    r0, c0 = max(0, dr), max(0, dc)
    r1, c1 = min(ha, dr + hb), min(wa, dc + wb)
    sa = (slice(r0, r1), slice(c0, c1))
    sb = (slice(r0 - dr, r1 - dr), slice(c0 - dc, c1 - dc))
    return sa, sb


def boundary_stats(mask: np.ndarray, cell: float) -> dict:
    _lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    filled = ndimage.binary_fill_holes(mask)
    _hole_lab, holes = ndimage.label(filled & ~mask)
    edges = int((mask[:, 1:] != mask[:, :-1]).sum() + (mask[1:, :] != mask[:-1, :]).sum())
    return dict(
        area=float(mask.sum() * cell**2),
        components=int(n),
        interior_holes=int(holes),
        interior_hole_area=float((filled & ~mask).sum() * cell**2),
        boundary_length=float(edges * cell),
        boundary_per_area=float(edges / max(mask.sum(), 1)),
    )


def bright_points(hist: np.ndarray, thresholds: dict[str, int]) -> dict:
    return {k: int(hist[v:].sum()) for k, v in thresholds.items()}


def cell_comparison(before: dict, after: dict) -> dict:
    sa, sb = _align(before, after)
    cell = abs(before["transform"].a)
    v1 = before["state"][sa] > 0
    v2 = after["state"][sb] > 0
    out = dict(
        before=boundary_stats(v1, cell),
        after=boundary_stats(v2, cell),
        recovered_cells=int((v2 & ~v1).sum()),
        removed_cells=int((v1 & ~v2).sum()),
        unchanged_cells=int((v1 & v2).sum()),
    )
    ref = after.get("refined")
    if ref is not None:
        conf = ref["confidence"][sb]
        ev = ref["evidence"][sb]
        out["after_confidence_cells"] = {
            CONFIDENCE_NAMES[i]: int(v)
            for i, v in enumerate(np.bincount(conf.ravel(), minlength=6))
        }
        rec = v2 & ~v1
        rem = v1 & ~v2
        out["recovered_by_evidence"] = {
            name: int(((ev & bit) > 0)[rec].sum()) for bit, name in EVIDENCE_NAMES.items()
        }
        out["removed_by_evidence"] = {
            name: int(((ev & bit) > 0)[rem].sum()) for bit, name in EVIDENCE_NAMES.items()
        }
        out["removed_now_uncertain"] = int((conf[rem] == 3).sum())
        above = ref["above"][sb][rec].astype(np.int64)
        inl = ref["inliers"][sb][rec].astype(np.int64)
        out["recovered_cell_outlier_share"] = (
            float(above.sum() / max(1, above.sum() + inl.sum())) if rec.any() else None
        )
    return out


def removed_cell_analysis(before: dict, after: dict) -> dict:
    """Where are v1 cells that v2 no longer selects, and are they coplanar with the road?

    Rebuilds the corridor reference from the saved v1 surface model (no point access)
    and reports distance to the corridor core and plane offset from it."""
    from roadmarking.analysis.lidar_stats import GridSpec
    from roadmarking.preprocessing.road_surface import (
        CorridorRefiner,
        RefinementCriteria,
        RoadCriteria,
        SurfaceModel,
    )

    res = before["result"]
    grid = json.loads((before["masks"] / "candidate_regions.native.json").read_text())["grid"]
    m = before["model"]
    model = SurfaceModel(
        GridSpec(**grid),
        RoadCriteria(**res["criteria"]),
        m["state"],
        m["a"],
        m["b"],
        m["z"],
        m["rms"],
        m["count"],
    )
    ref = CorridorRefiner(model, RefinementCriteria(**after["result"]["refinement"]["criteria"]))
    sa, sb = _align(before, after)
    if sa != sb or before["state"].shape != after["state"].shape:
        return {"comparable": False}
    rem = (before["state"] > 0) & ~(after["state"] > 0)
    if not rem.any():
        return {"comparable": True, "removed_cells": 0}
    rr, cc = np.indices(rem.shape)
    off = np.abs(m["z"] - ref._reference_at_centres(rr, cc))
    near = rem & (ref.dist <= 4)
    return dict(
        comparable=True,
        removed_cells=int(rem.sum()),
        distance_to_core_cells_percentiles_10_50_90=np.percentile(
            ref.dist[rem], [10, 50, 90]
        ).tolist(),
        abs_plane_offset_percentiles_10_50_90=np.percentile(off[rem], [10, 50, 90]).tolist(),
        removed_within_4_cells=int(near.sum()),
        removed_within_4_cells_abs_offset_percentiles_10_50_90=(
            np.percentile(off[near], [10, 50, 90]).tolist() if near.any() else None
        ),
        corridor_components=ref.components,
    )


def _densify(coords, step=0.25):
    pts = np.asarray(coords, dtype=float)[:, :2]
    if len(pts) < 2:
        return pts, 0.0
    seg = np.hypot(*np.diff(pts, axis=0).T)
    length = float(seg.sum())
    if length == 0:
        return pts[:1], 0.0
    s = np.r_[0, np.cumsum(seg)]
    t = np.arange(0, length, step)
    return np.c_[np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], length


def paint_line_support(lines: list, before: dict, after: dict) -> dict:
    """Independent paint evidence: share of externally traced marking vertices that lie
    inside each corridor. Lines come from the Naksha PlotLine tracer (intensity-based
    edge/dash tracing), which shares no code or threshold with v1/v2 selection.
    Lines >= 20 m are treated as edge/solid lines, 2.5-12 m as dashes."""
    sa, sb = _align(before, after)
    v1 = np.zeros_like(after["state"], dtype=bool)
    v1[sb] = before["state"][sa] > 0
    v2 = after["state"] > 0
    conf = after["refined"]["confidence"]
    t = after["transform"]
    groups: dict[str, list] = {"edge_or_solid": [], "dash": [], "other": []}
    for coords in lines:
        pts, length = _densify(coords)
        if not len(pts):
            continue
        kind = "edge_or_solid" if length >= 20 else "dash" if 2.5 <= length <= 12 else "other"
        groups[kind].append(pts)
    out = {}
    h, w = v2.shape
    for kind, parts in groups.items():
        if not parts:
            out[kind] = dict(lines=0)
            continue
        pts = np.vstack(parts)
        c = np.floor((pts[:, 0] - t.c) / t.a).astype(int)
        r = np.floor((pts[:, 1] - t.f) / t.e).astype(int)
        ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        r, c = r[ok], c[ok]
        n = max(len(r), 1)
        out[kind] = dict(
            lines=len(parts),
            vertices_0_25m=int(ok.sum()),
            outside_grid=int((~ok).sum()),
            inside_v1_pct=100 * float(v1[r, c].sum()) / n,
            inside_v2_pct=100 * float(v2[r, c].sum()) / n,
            inside_v2_uncertain_pct=100 * float((conf[r, c] == 3).sum()) / n,
            v2_only_pct=100 * float((v2[r, c] & ~v1[r, c]).sum()) / n,
            v1_only_pct=100 * float((v1[r, c] & ~v2[r, c]).sum()) / n,
        )
    return out


def block_pixels(run: dict) -> dict[tuple[int, int], Path]:
    blocks = {}
    for f in run["blocks_dir"].glob("b*_*.npz"):
        br, bc = f.stem[1:].split("_")
        blocks[(int(br), int(bc))] = f
    return blocks


def pixel_comparison(before: dict, after: dict) -> dict:
    """Observed 5 cm pixels on the shared global grid (blocks are grid-aligned)."""
    tb = before["result"]
    ta = after["result"]
    if tb["source"] != ta["source"]:
        raise ValueError("Different sources")
    bb, ba = block_pixels(before), block_pixels(after)
    obs_b = obs_a = added = removed = cand_b = cand_a = miss_b = miss_a = 0
    for key in sorted(set(bb) | set(ba)):
        ob = oa = rb = ra = None
        if key in bb:
            with np.load(bb[key]) as d:
                ob, rb = d["observed"] > 0, d["road_state"] > 0
        if key in ba:
            with np.load(ba[key]) as d:
                oa, ra = d["observed"] > 0, d["road_state"] > 0
        z = np.zeros((BLOCK, BLOCK), bool)
        ob = z if ob is None else ob
        oa = z if oa is None else oa
        rb = z if rb is None else rb
        ra = z if ra is None else ra
        obs_b += int(ob.sum())
        obs_a += int(oa.sum())
        added += int((oa & ~ob).sum())
        removed += int((ob & ~oa).sum())
        cand_b += int(rb.sum())
        cand_a += int(ra.sum())
        miss_b += int((rb & ~ob).sum())
        miss_a += int((ra & ~oa).sum())
    return dict(
        observed_pixels_before=obs_b,
        observed_pixels_after=obs_a,
        observed_pixels_added=added,
        observed_pixels_removed=removed,
        candidate_pixels_before=cand_b,
        candidate_pixels_after=cand_a,
        missing_in_candidate_pct_before=100 * miss_b / max(cand_b, 1),
        missing_in_candidate_pct_after=100 * miss_a / max(cand_a, 1),
    )


def grid_identity(before: dict, after: dict) -> dict:
    """Tiles present in both runs must have identical affines and dimensions."""
    mb = {t["tile_id"]: t for t in before["manifest"]}
    ma = {t["tile_id"]: t for t in after["manifest"]}
    common = sorted(set(mb) & set(ma))
    mismatched = [
        t
        for t in common
        if mb[t]["affine"] != ma[t]["affine"]
        or (mb[t]["height"], mb[t]["width"]) != (ma[t]["height"], ma[t]["width"])
    ]
    return dict(
        tiles_before=len(mb),
        tiles_after=len(ma),
        common_tiles=len(common),
        new_tiles=sorted(set(ma) - set(mb)),
        dropped_tiles=sorted(set(mb) - set(ma)),
        affine_or_size_mismatches=mismatched,
        identical_grid=not mismatched,
    )


def _mosaic(run, row, col, h, w, names):
    blocks = set(block_pixels(run))
    return read_mosaic(run["blocks_dir"], blocks, row, col, h, w, names)


def window_figure(path: Path, title: str, before: dict, after: dict, tile: dict) -> dict:
    """Same 5 cm window from both runs: intensity before/after, change, confidence."""
    r, c, h, w = tile["row_offset"], tile["col_offset"], tile["height"], tile["width"]
    b = _mosaic(before, r, c, h, w, ["normalized", "observed", "road_state"])
    a = _mosaic(after, r, c, h, w, ["normalized", "observed", "road_state", "road_confidence"])
    x0, y0, x1, y1 = tile["native_xy_bbox"]
    ext = (x0, x1, y0, y1)
    cmap = plt.get_cmap("gray").with_extremes(bad="#ff00ff")
    fig, ax = plt.subplots(1, 4, figsize=(24, 6.4))
    for k, (d, lab) in enumerate(((b, "Phase 3 (v1)"), (a, "Phase 3.1 (v2)"))):
        img = np.where(d["normalized"] < -1, np.nan, d["normalized"])
        ax[k].imshow(img, cmap=cmap, vmin=0, vmax=1, extent=ext, interpolation="nearest")
        ax[k].contour(
            d["road_state"] > 0,
            levels=[0.5],
            colors=["#00e5ff"],
            linewidths=0.7,
            extent=ext,
            origin="upper",
        )
        ax[k].set_title(f"{lab}: normalized intensity, cyan = candidate boundary", fontsize=9)
    change = np.zeros((h, w, 3))
    change[b["observed"] > 0] = (0.35, 0.35, 0.35)
    change[(a["observed"] > 0) & ~(b["observed"] > 0)] = (0.1, 0.9, 0.2)
    change[(b["observed"] > 0) & ~(a["observed"] > 0)] = (0.95, 0.15, 0.15)
    ax[2].imshow(change, extent=ext, interpolation="nearest")
    ax[2].set_title("observed pixels: grey both, green added by v2, red removed", fontsize=9)
    ax[3].imshow(
        a["road_confidence"],
        cmap=CONF_COLORS,
        vmin=-0.5,
        vmax=5.5,
        extent=ext,
        interpolation="nearest",
    )
    ax[3].set_title(
        "v2 evidence: black none, grey rejected, magenta insufficient,\n"
        "amber uncertain (not selected), green high, blue ROI",
        fontsize=9,
    )
    for axis in ax:
        axis.ticklabel_format(useOffset=False, style="plain")
        axis.tick_params(labelsize=6)
    fig.suptitle(f"{title} — {tile['tile_id']} (identical points and grid; diagnostic only)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=85)
    plt.close(fig)
    bright_b = int(((b["normalized"] >= 0.5) & (b["observed"] > 0)).sum())
    bright_a = int(((a["normalized"] >= 0.5) & (a["observed"] > 0)).sum())
    return dict(
        tile_id=tile["tile_id"],
        observed_before=int((b["observed"] > 0).sum()),
        observed_after=int((a["observed"] > 0).sum()),
        bright_pixels_before=bright_b,
        bright_pixels_after=bright_a,
        uncertain_pixels=int((a["road_confidence"] == 3).sum()),
    )


def overview_figure(path: Path, source: str, before: dict, after: dict) -> None:
    sa, sb = _align(before, after)
    conf = after["refined"]["confidence"][sb]
    v1 = before["state"][sa] > 0
    t = after["transform"]
    _h, _w = conf.shape
    ext = (
        t.c + sb[1].start * t.a,
        t.c + sb[1].stop * t.a,
        t.f + sb[0].stop * t.e,
        t.f + sb[0].start * t.e,
    )
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.imshow(conf, cmap=CONF_COLORS, vmin=-0.5, vmax=5.5, extent=ext, interpolation="nearest")
    ax.contour(v1, levels=[0.5], colors=["#ff3030"], linewidths=0.5, extent=ext, origin="upper")
    ax.set_title(
        f"{source}: v2 1 m evidence (green high, amber uncertain, grey rejected,\n"
        "magenta insufficient); red line = Phase 3 v1 candidate boundary",
        fontsize=10,
    )
    ax.ticklabel_format(useOffset=False, style="plain")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def proposed_roi(path: Path, source: str, after: dict) -> dict:
    """Polygonise v2 HIGH and UNCERTAIN cells as PROPOSED_UNREVIEWED review layers."""
    import geopandas as gpd
    from rasterio.features import shapes
    from shapely.geometry import shape

    conf = after["refined"]["confidence"]
    t = after["transform"]
    layers = {}
    for name, code in (("proposed_road_corridor", 4), ("uncertain_candidates", 3)):
        geoms = [shape(g) for g, v in shapes(conf.astype(np.uint8), mask=conf == code, transform=t)]
        gdf = gpd.GeoDataFrame(
            {
                "source": [source] * len(geoms),
                "evidence_class": [CONFIDENCE_NAMES[code]] * len(geoms),
                "review_status": ["PROPOSED_UNREVIEWED"] * len(geoms),
                "extraction_version": [after["result"]["configuration_version"]] * len(geoms),
                "coordinate_system": ["native XY; CRS_UNCONFIRMED"] * len(geoms),
                "area_native_units2": [g.area for g in geoms],
            },
            geometry=geoms,
            crs=None,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if len(gdf):
            gdf.to_file(path, layer=name, driver="GPKG")
        layers[name] = dict(polygons=len(gdf), area=float(gdf.area.sum()) if len(gdf) else 0.0)
    return dict(path=str(path), layers=layers, review_status="PROPOSED_UNREVIEWED")
