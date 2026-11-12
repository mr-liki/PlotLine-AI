"""Raster/vector coordinate-frame verification. Diagnoses; never transforms.

Axis-swap/sign hypotheses are evaluated only as diagnostics and are never applied.
Displacement is measured from intensity cross-sections of painted lines, which is an
independent observation of the paint position in the raster.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import shapely
from shapely import STRtree
from shapely.geometry import LineString, box

from roadmarking.dataset.geometry import line_parts, tile_box, world_to_pixel


def raster_footprints(tiles: list[dict]) -> dict[str, shapely.Geometry]:
    by_source: dict[str, list] = {}
    for tile in tiles:
        by_source.setdefault(Path(tile["source_file"]).stem, []).append(tile_box(tile))
    return {k: shapely.union_all(v) for k, v in sorted(by_source.items())}


def lidar_header_extents(cache_dir: Path) -> dict[str, dict]:
    """Header XYZ bounds of every LAS/LAZ inspected in Phase 2 (incl. damaged files)."""
    out = {}
    for path in sorted(Path(cache_dir).glob("*.json")):
        header = json.loads(path.read_text(encoding="utf-8")).get("header") or {}
        mins, maxs = header.get("mins"), header.get("maxs")
        valid = bool(mins and maxs and all(abs(v) < 1e300 for v in mins + maxs))
        out[path.stem] = {
            "valid_header_bounds": valid,
            "xyz_min": mins if valid else None,
            "xyz_max": maxs if valid else None,
            "crs_wkt": header.get("crs_wkt"),
        }
    return out


HYPOTHESES = {
    "identity": lambda x, y: (x, y),
    "swap_xy": lambda x, y: (y, x),
    "negate_x": lambda x, y: (-x, y),
    "negate_y": lambda x, y: (x, -y),
    "negate_xy": lambda x, y: (-x, -y),
    "swap_negate_x": lambda x, y: (-y, x),
    "swap_negate_y": lambda x, y: (y, -x),
}


def axis_hypotheses(vector_bounds, raster_bounds) -> dict[str, bool]:
    """Would any pure axis swap/sign flip make bounding boxes intersect? Diagnostic only."""
    minx, miny, maxx, maxy = vector_bounds
    target = box(*raster_bounds)
    out = {}
    for name, fn in HYPOTHESES.items():
        xs, ys = zip(*(fn(x, y) for x, y in ((minx, miny), (maxx, maxy))), strict=True)
        out[name] = bool(box(min(xs), min(ys), max(xs), max(ys)).intersects(target))
    return out


def feature_overlap(features, tiles: list[dict]) -> dict:
    """Count features intersecting any tile footprint using a tile STRtree."""
    boxes = [tile_box(t) for t in tiles]
    tree = STRtree(boxes)
    geoms = shapely.force_2d(features.geometry.values)
    valid = np.array([g is not None and not g.is_empty for g in geoms])
    feat_idx, tile_idx = tree.query(geoms[valid], predicate="intersects")
    hit_features = np.unique(np.nonzero(valid)[0][feat_idx])
    pairs = {}
    for f, t in zip(np.nonzero(valid)[0][feat_idx], tile_idx, strict=True):
        pairs.setdefault(int(f), []).append(tiles[int(t)]["tile_id"])
    return {
        "features_total": len(features),
        "features_intersecting_tiles": len(hit_features),
        "tiles_with_features": len(np.unique(tile_idx)),
        "feature_tiles": pairs,
    }


def perpendicular_offsets(
    line: LineString,
    image: np.ndarray,
    affine,
    half_window: float = 0.6,
    spacing: float = 0.5,
    min_peak: float = 0.0,
) -> np.ndarray:
    """Signed lateral offsets (native units) of the intensity peak from a centreline.

    Samples the raster at pixel centres along perpendicular profiles every ``spacing``
    units, ignoring NaN/no-data. Positive = left of the line direction. A profile with
    no finite samples or a peak below ``min_peak`` is skipped.
    """
    res = abs(affine.a)
    offsets = np.arange(-half_window, half_window + res / 2, res / 2)
    out = []
    length = line.length
    for d in np.arange(spacing / 2, length, spacing):
        p0 = line.interpolate(max(d - res, 0))
        p1 = line.interpolate(min(d + res, length))
        dx, dy = p1.x - p0.x, p1.y - p0.y
        norm = math.hypot(dx, dy)
        if norm == 0:
            continue
        nx, ny = -dy / norm, dx / norm
        c = line.interpolate(d)
        xs, ys = c.x + offsets * nx, c.y + offsets * ny
        rows, cols = world_to_pixel(affine, xs, ys)
        rows, cols = np.floor(rows).astype(int), np.floor(cols).astype(int)
        inside = (rows >= 0) & (rows < image.shape[0]) & (cols >= 0) & (cols < image.shape[1])
        vals = np.full(len(offsets), np.nan)
        vals[inside] = image[rows[inside], cols[inside]]
        if not np.isfinite(vals).any() or np.nanmax(vals) < min_peak:
            continue
        # Intensity-weighted centroid of the upper half of the profile: robust to 1-pixel noise.
        peak = np.nanmax(vals)
        base = np.nanmin(vals)
        w = np.where(np.isfinite(vals), np.clip(vals - (base + peak) / 2, 0, None), 0)
        if w.sum() <= 0:
            continue
        out.append(float((w * offsets).sum() / w.sum()))
    return np.asarray(out)


def measure_line_displacement(lines, image, affine, **kwargs) -> dict:
    values = np.concatenate(
        [perpendicular_offsets(p, image, affine, **kwargs) for g in lines for p in line_parts(g)]
        or [np.empty(0)]
    )
    if not len(values):
        return {"samples": 0}
    return {
        "samples": len(values),
        "median_signed": float(np.median(values)),
        "median_abs": float(np.median(np.abs(values))),
        "p90_abs": float(np.percentile(np.abs(values), 90)),
    }


def compare_frames(features, tiles: list[dict], lidar_extents: dict) -> dict:
    """Extent/unit/axis comparison between annotations and Phase 3 rasters."""
    footprints = raster_footprints(tiles)
    raster_union = shapely.union_all(list(footprints.values()))
    rb = raster_union.bounds
    out = {
        "raster_crs_status": sorted({t["crs_status"] for t in tiles}),
        "raster_bounds": list(rb),
        "raster_footprint_area_native2": float(raster_union.area),
        "raster_bounds_by_source": {k: list(v.bounds) for k, v in footprints.items()},
        "raster_resolution": sorted({t["resolution"] for t in tiles}),
        "raster_axis": "north-up; affine e < 0 (row increases toward -Y)",
        "lidar_header_extents": lidar_extents,
        "annotation_files": {},
    }
    for name, group in features.groupby("source_file"):
        vb = [float(v) for v in group.total_bounds]
        zs = shapely.get_coordinates(group.geometry.values, include_z=True)[:, 2]
        out["annotation_files"][name] = {
            "declared_crs": sorted({str(c) for c in group["declared_crs"]}),
            "bounds": vb,
            "z_range": (
                [float(np.nanmin(zs)), float(np.nanmax(zs))] if np.isfinite(zs).any() else None
            ),
            "bbox_intersects_raster_bbox": bool(box(*vb).intersects(box(*rb))),
            "gap_to_raster_footprint_native": float(box(*vb).distance(raster_union)),
            "centroid_offset_native": [
                (vb[0] + vb[2]) / 2 - (rb[0] + rb[2]) / 2,
                (vb[1] + vb[3]) / 2 - (rb[1] + rb[3]) / 2,
            ],
            "extent_size_native": [vb[2] - vb[0], vb[3] - vb[1]],
            "axis_hypotheses_bbox_intersect": axis_hypotheses(vb, rb),
        }
        lidar_hits = {
            k: bool(
                v["valid_header_bounds"]
                and box(*vb).intersects(
                    box(v["xyz_min"][0], v["xyz_min"][1], v["xyz_max"][0], v["xyz_max"][1])
                )
            )
            for k, v in lidar_extents.items()
        }
        out["annotation_files"][name]["bbox_intersects_any_lidar_header"] = lidar_hits
    overlap = feature_overlap(features, tiles)
    out["feature_overlap"] = {k: v for k, v in overlap.items() if k != "feature_tiles"}
    out["_feature_tiles"] = overlap["feature_tiles"]
    return out
