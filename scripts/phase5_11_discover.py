"""Phase 5.11 Stages 2-3: rank new training windows, then verify ACTUAL point coverage.

Windows are proposed from client annotation content, but a window is retained only after
its assigned LAZ file is fully decoded (strict point-count integrity) and the rasterised
observations show meaningful observed paint. Imagery reuses the frozen shared normalization,
native 0.05 grid and gamma .7 rendering. No synthetic paint; nothing is filled.
"""

import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import shapely
from affine import Affine
from shapely.geometry import box

from phase5_11_common import EDGE, MERGE, ROOT, client_features, load, reserved_geography
from roadmarking.dataset.coverage import paint_contrast, solid_shape
from roadmarking.dataset.labels import LabelFeature
from roadmarking.dataset.verified_masks import reviewed_masks
from roadmarking.io.pointcloud import ChunkReadReport, iter_laz_chunks
from roadmarking.io.source_inventory import fingerprint, safe_header
from roadmarking.preprocessing.intensity import IntensityCalibration
from roadmarking.rasterization.pipeline import dump, write_tif
from roadmarking.rasterization.rasterizer import RasterAccumulator, world_to_pixel
from roadmarking.visualization.intensity_render import save_intensity_png, write_png_worldfile

HALF = 25.6
GUARD = 500.0
VERIFY_LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 40
PACKAGE_LIMIT = 16
MAX_SOURCE_ATTEMPTS = 3
TRAINED_JOB = "Job_20250708_0834"


def window_stats(region, hits, features):
    solids, dashes, merges, edges, unsupported, interchange = [], 0, 0, 0, 0.0, 0
    for j in hits:
        f = features[j]
        if f["visual_id"] == 1:
            clipped = f["centerline"].intersection(region)
            shape, length, rate = solid_shape(clipped)
            if length >= 5:
                solids.append(
                    {
                        "shape": shape,
                        "length": length,
                        "rate": rate,
                        "edge": f["designation"] in EDGE,
                    }
                )
        elif f["visual_id"] == 2:
            dashes += 1
        else:
            unsupported += f["geometry"].intersection(region).area
        merges += f["designation"] in MERGE
        edges += f["designation"] in EDGE
        interchange += f["layer"] == "merged"
    curved = sum(s["shape"] == "curved" for s in solids)
    return {
        "solid_lines": len(solids),
        "curved_solids": curved,
        "max_turning_deg_per_50": max((s["rate"] for s in solids), default=0.0),
        "solid_length": sum(s["length"] for s in solids),
        "edge_lines": sum(s["edge"] for s in solids),
        "merge_lines": merges,
        "dashes": dashes,
        "unsupported_area": unsupported,
        "interchange_features": interchange,
    }


def score(s):
    if s["solid_lines"] == 0 and s["dashes"] < 6:
        return -1.0
    value = (
        3.0 * s["curved_solids"]
        + 1.5 * s["edge_lines"]
        + 1.0 * s["solid_lines"]
        + 1.5 * min(s["merge_lines"], 4)
        + 0.12 * min(s["dashes"], 30)
    )
    return value - (2.0 if s["unsupported_area"] > 40 else 0.0)


def tag(s):
    if s["curved_solids"]:
        return "curved_solid"
    if s["merge_lines"]:
        return "merge_diverge"
    if s["edge_lines"]:
        return "solid_edge"
    if s["solid_lines"]:
        return "straight_solid"
    return "broken_lines"


def main():
    began = time.perf_counter()
    latest = load(ROOT / "outputs/reports/phase5_11/latest.json")
    out = Path(latest["report_dir"])
    data = ROOT / "data/processed/phase5_11_review" / latest["run_id"]
    data.mkdir(parents=True, exist_ok=True)
    features, layers, audit = client_features()
    regions, extra_ids, _ = reserved_geography()
    reserved_ids = set(extra_ids)
    train_ids = set()
    for r in regions:
        (train_ids if r["role"] == "train" else reserved_ids).update(r["ids"])
    footprints = [r["box"] for r in regions]
    tree = shapely.STRtree([f["geometry"] for f in features])
    raw = [
        r
        for r in load(Path(audit["report_directory"]) / "raw_integrity.json")
        if "backup" not in r["path"].lower()
    ]

    # ---------------- candidate centres from client content
    centres = []
    for f in features:
        if f["visual_id"] == 1:
            line = f["centerline"]
            for d in np.arange(0, line.length, 25.0):
                p = line.interpolate(d)
                centres.append((p.x, p.y))
        elif f["visual_id"] == 2 and int(f["global_id"][:8], 16) % 5 == 0:
            c = f["geometry"].centroid
            centres.append((c.x, c.y))
    candidates = []
    for x, y in centres:
        region = box(x - HALF, y - HALF, x + HALF, y + HALF)
        if min(region.distance(b) for b in footprints) < GUARD:
            continue
        hits = tree.query(region, predicate="intersects")
        ids = {features[i]["global_id"] for i in hits}
        if ids & reserved_ids:
            continue
        files = [
            r
            for r in raw
            if r["mins"][0] <= x - HALF
            and r["maxs"][0] >= x + HALF
            and r["mins"][1] <= y - HALF
            and r["maxs"][1] >= y + HALF
        ]
        if not files:
            continue
        stats = window_stats(region, hits, features)
        value = score(stats)
        if value <= 0:
            continue
        jobs = {Path(r["path"]).name.split("_Track")[0] for r in files}
        new_job = bool(jobs - {TRAINED_JOB})
        candidates.append(
            {
                "x": x,
                "y": y,
                "region": region,
                "hits": hits,
                "stats": stats,
                "score": value + (1.5 if new_job else 0.0),
                "files": files,
                "new_acquisition_available": new_job,
                "tag": tag(stats),
            }
        )
    # ---------------- greedy diverse selection (independent sections, category quotas)
    candidates.sort(key=lambda c: -c["score"])
    quotas = {
        "curved_solid": 14,
        "merge_diverge": 8,
        "solid_edge": 8,
        "straight_solid": 4,
        "broken_lines": 6,
    }
    chosen, used = [], Counter()
    for c in candidates:
        if len(chosen) >= VERIFY_LIMIT:
            break
        if used[c["tag"]] >= quotas[c["tag"]]:
            continue
        if any(c["region"].distance(o["region"]) < GUARD for o in chosen):
            continue
        chosen.append(c)
        used[c["tag"]] += 1
    dump(
        out / "candidate_ranking.json",
        {
            "candidate_windows_considered": len(candidates),
            "selected_for_point_verification": len(chosen),
            "guard_native": GUARD,
            "quotas": quotas,
            "selected": [
                {
                    "centre": [c["x"], c["y"]],
                    "tag": c["tag"],
                    "score": c["score"],
                    "stats": c["stats"],
                    "files": [Path(f["path"]).name for f in c["files"]],
                }
                for c in chosen
            ],
        },
    )
    print(f"{len(candidates)} candidate windows; verifying {len(chosen)}", flush=True)
    if "dry" in sys.argv:
        return

    # ---------------- one source per window (never merged); fall back to the next
    # header-containing file when the first choice lacks actual observations.
    for k, c in enumerate(chosen):
        c["name"] = f"P511_{k + 1:02d}"
        c["queue"] = sorted(
            c["files"],
            key=lambda r: (
                Path(r["path"]).name.split("_Track")[0] == TRAINED_JOB,
                -r["declared_bbox_density_points_per_native_square"],
            ),
        )
        c["attempts"] = []
    calibration_path = next(
        p
        for p in load(ROOT / "outputs/reports/phase5_6a/20261106T093757339115Z/frozen_model.json")[
            "artifact_sha256"
        ]
        if p.endswith("shared_normalization.json")
    )
    cal = load(calibration_path)
    calibration = IntensityCalibration(
        **{
            k: cal[k]
            for k in ("low", "high", "percentile_low", "percentile_high", "statistics", "method")
        }
    )
    scans = []
    for attempt in range(MAX_SOURCE_ATTEMPTS):
        by_file = defaultdict(list)
        for c in chosen:
            if not c.get("meaningful") and attempt < len(c["queue"]):
                c["source"] = c["queue"][attempt]
                by_file[c["source"]["path"]].append(c)
        if not by_file:
            break
        for path, windows in by_file.items():
            scan = render_file(path, windows, raw, calibration, features, attempt)
            scans.append(scan)
            dump(out / "source_integrity.json", scans)
            done = sum(c.get("meaningful", False) for c in windows)
            print(
                f"[attempt {attempt + 1}] {Path(path).name}: {scan['actual_points']:,} pts, "
                f"ok={scan['full_source_integrity_passed']}, {scan['seconds']:.0f}s, "
                f"meaningful {done}/{len(windows)}",
                flush=True,
            )
    verified = [c for c in chosen if c.get("meaningful")]
    # ---------------- final package selection: unique new identities, priority first
    verified.sort(key=lambda c: -c["score"])
    package, seen = [], set()
    for c in verified:
        if len(package) >= PACKAGE_LIMIT:
            break
        new = {p["global_id"] for p in c["observed_ids"]} - seen - train_ids
        if len(new) < 3:
            continue
        package.append(c)
        seen |= {p["global_id"] for p in c["observed_ids"]}
    rows, unsupported_rows, footprint_rows, tiles = [], [], [], []
    references = []
    for index, item in enumerate(layers):
        copy = data / "references" / f"client_{index}.gpkg"
        copy.parent.mkdir(exist_ok=True)
        if not copy.exists():
            shutil.copyfile(item["local_snapshot"], copy)
        references.append(
            {
                "path": str(copy),
                "layer": item["layer"],
                "source": item["source"],
                "sha256": item["sha256"],
            }
        )
    for c in package:
        name, affine, layers_ = c["name"], c["affine"], c["layers"]
        layers_["nodata"] = (~layers_["observed"].astype(bool)).astype(np.uint8)
        layers_["padding"] = np.zeros((1024, 1024), np.uint8)
        paths = {}
        for key, array in layers_.items():
            target = data / "rasters" / name / (key + ".tif")
            write_tif(target, array, affine, None)
            paths[key] = str(target)
        png = data / "images" / (name + ".png")
        save_intensity_png(png, layers_["normalized"], layers_["observed"], gamma=0.7)
        write_png_worldfile(png, affine)
        observed_lookup = {p["global_id"]: p for p in c["observed_ids"]}
        for j in c["hits"]:
            f = features[j]
            geom = f["geometry"].intersection(c["region"])
            if geom.is_empty or geom.area <= 0:
                continue
            base = {
                "global_id": f["global_id"],
                "source_identity": f["source_identity"],
                "candidate_id": name,
                "client_layer": f["layer"],
            }
            if f["visual_id"] in (1, 2):
                obs = observed_lookup.get(f["global_id"], {})
                rows.append(
                    {
                        **base,
                        "visual_id": f["visual_id"],
                        "class_name": "solid" if f["visual_id"] == 1 else "dash",
                        "designation": f["designation"],
                        "width": f["width"],
                        "observed_pixels": obs.get("observed_pixels", 0),
                        "review_status": "NEW_TRAINING_REVIEW_PENDING",
                        "training_eligible": 0,
                        "geometry": geom,
                    }
                )
            else:
                unsupported_rows.append(
                    {
                        **base,
                        "code": str(f["code"]),
                        "policy": "EXCLUDE_WITH_MARGIN_NOT_BACKGROUND",
                        "geometry": geom,
                    }
                )
        footprint_rows.append(
            {
                "candidate_id": name,
                "priority_tag": c["tag"],
                "source_file": Path(c["source"]["path"]).name,
                "observed_fraction": c["observed_fraction"],
                "observed_ids": len(c["observed_ids"]),
                "review_status": "PENDING_FIRST_APPROVAL",
                "geometry": c["region"],
            }
        )
        tiles.append(
            {
                "candidate_id": name,
                "priority_tag": c["tag"],
                "bounds": list(c["region"].bounds),
                "affine": list(affine)[:6],
                "source": c["source"]["path"],
                "acquisition": Path(c["source"]["path"]).name.split("_Track")[0],
                "layers": paths,
                "images": {"grayscale": str(png)},
                "content_stats": c["stats"],
                "observed_fraction": c["observed_fraction"],
                "observed_paint_pixels": c["observed_paint_pixels"],
                "observed_solid_ids": sum(p["class"] == "solid" for p in c["observed_ids"]),
                "observed_dash_ids": sum(p["class"] == "dash" for p in c["observed_ids"]),
                "observed_ids": c["observed_ids"],
                "bright_background_pixels": c["bright_background_pixels"],
                "source_global_ids": sorted({features[j]["global_id"] for j in c["hits"]}),
                "shares_ids_with_existing_training": sorted(
                    {p["global_id"] for p in c["observed_ids"]} & train_ids
                ),
                "minimum_distance_to_reserved_or_trained": min(
                    c["region"].distance(b) for b in footprints
                ),
                "approval_status": "PENDING_FIRST_APPROVAL",
                "training_eligible": False,
            }
        )
    store = data / "candidate_outlines.gpkg"
    if store.exists():
        store.unlink()
    for layer, frame_rows in (
        ("candidate_paint_footprints", rows),
        ("unsupported_marking_regions", unsupported_rows),
        ("review_region_footprints", footprint_rows),
    ):
        if frame_rows:
            pyogrio.write_dataframe(
                gpd.GeoDataFrame(frame_rows, geometry="geometry", crs=None),
                store,
                layer=layer,
                driver="GPKG",
                promote_to_multi=True,
                append=store.exists(),
            )
    unique_new = {
        p["global_id"]: p["class"]
        for t in tiles
        for p in t["observed_ids"]
        if p["global_id"] not in train_ids
    }
    spec = {
        "output": str(data),
        "candidate_store": str(store),
        "references": references,
        "tiles": tiles,
        "verification": {
            "windows_point_verified": len(chosen),
            "windows_meaningful": len(verified),
            "windows_packaged": len(package),
            "meaningful_but_not_packaged": [c["name"] for c in verified if c not in package],
            "failed_observation_check": [
                {
                    "name": c["name"],
                    "centre": [c["x"], c["y"]],
                    "tag": c["tag"],
                    "attempts": c["attempts"],
                }
                for c in chosen
                if not c.get("meaningful")
            ],
            "attempts_per_window": {c["name"]: c["attempts"] for c in chosen},
        },
        "new_unique_observed_identities": {
            "total": len(unique_new),
            "solid": sum(v == "solid" for v in unique_new.values()),
            "dash": sum(v == "dash" for v in unique_new.values()),
        },
        "elapsed_seconds": time.perf_counter() - began,
        "status": "REAL_DATA_REVIEW_REQUIRED",
        "training_eligible": False,
    }
    dump(out / "new_region_candidates.json", spec)
    dump(data / "review_spec.json", spec)
    dump(
        ROOT / "outputs/reports/phase5_11/latest.json",
        {**latest, "data_dir": str(data), "status": "REVIEW_PACKAGE_PENDING_QGIS"},
    )
    print(
        json.dumps(
            {
                k: spec[k]
                for k in ("verification", "new_unique_observed_identities", "elapsed_seconds")
            },
            indent=1,
        ),
        flush=True,
    )


def render_file(path, windows, raw, calibration, features, attempt, *, observed_only=False):
    """Fully decode one LAZ file once and rasterise every window assigned to it."""
    record = next(r for r in raw if r["path"] == path)
    source = Path(path)
    before = fingerprint(source)
    if before != record["fingerprint"]:
        raise ValueError(f"Source changed since inventory: {path}")
    header = safe_header(source)
    accumulators = {c["name"]: RasterAccumulator((1024, 1024)) for c in windows}
    affines = {
        c["name"]: Affine(0.05, 0, c["region"].bounds[0], 0, -0.05, c["region"].bounds[3])
        for c in windows
    }
    progress = ChunkReadReport(source, 0, 0)
    tick = time.perf_counter()
    for points in iter_laz_chunks(source, progress):
        x, y = np.asarray(points.x), np.asarray(points.y)
        intensity = np.asarray(points.intensity)
        for c in windows:
            x0, y0, x1, y1 = c["region"].bounds
            keep = (x >= x0) & (x < x1) & (y > y0) & (y <= y1)
            if keep.any():
                cols, rows = world_to_pixel(affines[c["name"]], x[keep], y[keep])
                inside = (cols >= 0) & (rows >= 0) & (cols < 1024) & (rows < 1024)
                accumulators[c["name"]].add(cols[inside], rows[inside], intensity[keep][inside])
    ok = (
        progress.points_read == header.point_count
        and not progress.dropped_points
        and fingerprint(source) == before
    )
    scan = {
        "source": path,
        "fingerprint": before,
        "attempt": attempt + 1,
        "expected_points": int(header.point_count),
        "actual_points": progress.points_read,
        "dropped_points": progress.dropped_points,
        "full_source_integrity_passed": ok,
        "windows": [c["name"] for c in windows],
        "seconds": time.perf_counter() - tick,
    }
    if not ok:
        return scan
    for c in windows:
        layers_ = accumulators[c["name"]].layers(calibration)
        observed = layers_["observed"].astype(bool)
        affine = affines[c["name"]]
        fs = [
            LabelFeature(
                i + 1,
                features[j]["global_id"],
                features[j]["visual_id"],
                "polygon",
                features[j]["geometry"].intersection(c["region"]),
            )
            for i, j in enumerate(c["hits"])
            if features[j]["visual_id"] in (1, 2)
        ]
        labels, _ = reviewed_masks(
            affine, observed.shape, observed, np.ones_like(observed), fs, completeness=c["region"]
        )
        gray = np.asarray(layers_["normalized"], float)
        paint = labels.instance > 0
        per_id = []
        for entry in labels.instances:
            if not entry["local_id"]:
                continue
            m = (labels.instance == entry["local_id"]) & observed
            per_id.append(
                {
                    "global_id": entry["global_id"],
                    "class": "solid" if entry["visual_id"] == 1 else "dash",
                    "observed_pixels": int(m.sum()),
                    "contrast_normalized": paint_contrast(gray, m, observed, exclude=paint & ~m),
                }
            )
        observed_ids = [p for p in per_id if p["observed_pixels"] >= 4]
        paint_pixels = int((paint & observed).sum())
        background = observed & ~binary_dilate(paint, 3)
        threshold = np.percentile(gray[paint & observed], 25) if paint_pixels else np.inf
        result = {
            "source": path,
            "observed_fraction": float(observed.mean()),
            "observed_paint_pixels": paint_pixels,
            "observed_ids": len(observed_ids),
        }
        c["attempts"].append(result)
        meaningful = (
            result["observed_fraction"] >= 0.1 and paint_pixels >= 64 and len(observed_ids) >= 3
        )
        if meaningful or (observed_only and observed.any()):
            c.update(
                observed_fraction=result["observed_fraction"],
                observed_paint_pixels=paint_pixels,
                observed_ids=observed_ids,
                bright_background_pixels=int((gray[background] >= threshold).sum()),
                layers=layers_,
                affine=affine,
                meaningful=True,
            )
    return scan


def binary_dilate(mask, iterations):
    from scipy.ndimage import binary_dilation

    return binary_dilation(mask, iterations=iterations)


if __name__ == "__main__":
    main()
