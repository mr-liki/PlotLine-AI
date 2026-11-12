"""Stage 8 Task 3: search the verified acquisitions for genuinely new solid identities.

For every client solid (visual class 1) the script records whether its ID was used in any
role (training, development, benchmark, consumed test, Stage-6, Stage-7J candidates),
whether it duplicates or continues a used solid under another client ID, and which
integrity-verified LAZ files cover it. Unused, non-duplicate solids are then checked for
ACTUAL observed-pixel support by fully decoding every covering file once (no header or
bounding-box shortcut): observed pixels inside the painted polygon and the supported
centreline length in 1-unit bins. Nothing is rendered for review here and no model runs.
"""

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import shapely
from shapely.geometry import box

sys.argv = sys.argv[:1]
from phase5_11_common import (  # noqa: E402
    EDGE,
    MERGE,
    ROOT,
    client_features,
    load,
    reserved_geography,
)
from roadmarking.dataset.coverage import solid_shape  # noqa: E402
from roadmarking.io.pointcloud import ChunkReadReport, iter_laz_chunks  # noqa: E402
from roadmarking.io.source_inventory import fingerprint, safe_header  # noqa: E402
from roadmarking.rasterization.pipeline import dump  # noqa: E402

OUT = ROOT / "outputs/reports/stage8/8_3_solid_discovery"
PIXEL = 0.05
MIN_LENGTH = 10.0  # units; shorter solids are not useful independent identities
SUPPORTED_BIN = 3  # observed paint pixels needed for a 1-unit centreline bin
GUARD = 500.0


def used_geography():
    """Every footprint and ID used so far, by role (metadata only)."""
    regions, extra, _ = reserved_geography()
    out = [
        {"id": r["id"], "role": r["role"], "box": box(*r["bounds"]), "ids": set(r["ids"])}
        for r in regions
    ]
    for phase, role in (
        ("phase5_11", "train_approved"),
        ("phase5_12", "train_approved"),
        ("stage6_review", "stage6"),
        ("stage7_new_test", "stage7j_candidate_pending_review"),
    ):
        spec = load(
            Path(load(ROOT / f"outputs/reports/{phase}/latest.json")["report_dir"])
            / "new_region_candidates.json"
        )
        for t in spec["tiles"]:
            r = role
            if phase == "stage6_review":
                r = "fresh_dev" if t["candidate_id"] == "S6_01" else "train_approved"
            out.append(
                {
                    "id": t["candidate_id"],
                    "role": r,
                    "box": box(*t["bounds"]),
                    "ids": set(t["source_global_ids"]),
                }
            )
    return out, set(extra)


def bad_sources():
    bad = set()
    for p in ("phase5_11", "phase5_12", "stage6_review", "stage7_new_test"):
        rd = Path(load(ROOT / f"outputs/reports/{p}/latest.json")["report_dir"])
        for s in load(rd / "source_integrity.json"):
            if not s["full_source_integrity_passed"]:
                bad.add(s["source"])
    return bad


def scan_file(path, record, items):
    ("Decode one LAZ file fully; collect observed pixel keys inside each " "solid polygon.")
    source = Path(path)
    before = fingerprint(source)
    if before != record["fingerprint"]:
        raise ValueError(f"Source changed since inventory: {path}")
    header = safe_header(source)
    keys = defaultdict(list)
    progress = ChunkReadReport(source, 0, 0)
    tick = time.perf_counter()
    for points in iter_laz_chunks(source, progress):
        x, y = np.asarray(points.x), np.asarray(points.y)
        for it in items:
            x0, y0, x1, y1 = it["bounds"]
            sel = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
            if not sel.any():
                continue
            xs, ys = x[sel], y[sel]
            inside = shapely.contains_xy(it["geometry"], xs, ys)
            if inside.any():
                col = np.floor((xs[inside] - x0) / PIXEL).astype(np.int64)
                row = np.floor((y1 - ys[inside]) / PIXEL).astype(np.int64)
                keys[it["global_id"]].append(np.unique(row * 10_000_000 + col))
    ok = (
        progress.points_read == header.point_count
        and not progress.dropped_points
        and fingerprint(source) == before
    )
    return (
        {k: np.unique(np.concatenate(v)) for k, v in keys.items()},
        {
            "source": path,
            "expected_points": int(header.point_count),
            "actual_points": progress.points_read,
            "dropped_points": progress.dropped_points,
            "full_source_integrity_passed": bool(ok),
            "solids_scanned": len(items),
            "seconds": time.perf_counter() - tick,
        },
    )


def support(item, pixel_keys):
    """Observed paint pixels and supported centreline length (1-unit bins)."""
    if pixel_keys is None or not len(pixel_keys):
        return 0, 0.0
    x0, _, _, y1 = item["bounds"]
    rows, cols = pixel_keys // 10_000_000, pixel_keys % 10_000_000
    xs, ys = x0 + (cols + 0.5) * PIXEL, y1 - (rows + 0.5) * PIXEL
    along = shapely.line_locate_point(item["centerline"], shapely.points(xs, ys))
    bins = np.bincount(np.floor(along).astype(int))
    return len(pixel_keys), float((bins >= SUPPORTED_BIN).sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    features, _, audit = client_features()
    used, extra = used_geography()
    used_ids = set(extra).union(*(r["ids"] for r in used))
    train_boxes = [r["box"] for r in used if r["role"].startswith("train")]
    all_boxes = [r["box"] for r in used]
    solids = [f for f in features if f["visual_id"] == 1]
    used_solids = [f for f in solids if f["global_id"] in used_ids]
    used_tree = shapely.STRtree([f["geometry"] for f in used_solids])
    raw = [
        r
        for r in load(Path(audit["report_directory"]) / "raw_integrity.json")
        if "backup" not in r["path"].lower()
    ]
    bad = bad_sources()
    raw = [r for r in raw if r["path"] not in bad]
    rows = []
    for f in solids:
        geom = f["geometry"]
        kind, _, curv = solid_shape(f["centerline"])
        rec = {
            "global_id": f["global_id"],
            "layer": f["layer"],
            "designation": f["designation"],
            "length": float(f["centerline"].length),
            "shape": kind,
            "curvature_deg_per_50": float(curv),
            "edge_line": f["designation"] in EDGE,
            "merge_line": f["designation"] in MERGE,
            "used": f["global_id"] in used_ids,
            "roles": sorted({r["role"] for r in used if f["global_id"] in r["ids"]}),
            "distance_to_training_footprint": float(min(geom.distance(b) for b in train_boxes)),
            "distance_to_any_used_footprint": float(min(geom.distance(b) for b in all_boxes)),
        }
        near = [
            used_solids[i]
            for i in used_tree.query(geom, predicate="dwithin", distance=1.0)
            if used_solids[i]["global_id"] != f["global_id"]
        ]
        rec["overlaps_used_solid"] = any(
            geom.intersection(u["geometry"]).area > 0.2 * max(1e-9, geom.area) for u in near
        )
        rec["touches_used_solid_within_1"] = bool(near)
        cover = [
            r
            for r in raw
            if r["mins"][0] <= geom.bounds[2]
            and r["maxs"][0] >= geom.bounds[0]
            and r["mins"][1] <= geom.bounds[3]
            and r["maxs"][1] >= geom.bounds[1]
        ]
        rec["header_files"] = [r["path"] for r in cover]
        rec["acquisitions"] = sorted({Path(p).name.split("_Track")[0] for p in rec["header_files"]})
        rec["identity_independent"] = (
            not rec["used"]
            and not rec["overlaps_used_solid"]
            and not rec["touches_used_solid_within_1"]
        )
        rows.append(rec)
    candidates = [
        r
        for r in rows
        if r["identity_independent"] and r["length"] >= MIN_LENGTH and r["header_files"]
    ]
    by_id = {f["global_id"]: f for f in solids}
    per_file = defaultdict(list)
    for r in candidates:
        f = by_id[r["global_id"]]
        for p in r["header_files"]:
            per_file[p].append(
                {
                    "global_id": f["global_id"],
                    "geometry": f["geometry"],
                    "centerline": f["centerline"],
                    "bounds": f["geometry"].bounds,
                }
            )
    merged, scans = defaultdict(list), []
    for k, (path, items) in enumerate(sorted(per_file.items())):
        record = next(r for r in raw if r["path"] == path)
        keys, scan = scan_file(path, record, items)
        scans.append(scan)
        if scan["full_source_integrity_passed"]:
            for gid, v in keys.items():
                merged[gid].append(v)
        print(
            f"[{k + 1}/{len(per_file)}] {Path(path).name}: "
            f"ok={scan['full_source_integrity_passed']} "
            f"{scan['seconds']:.0f}s",
            flush=True,
        )
    for r in candidates:
        f = by_id[r["global_id"]]
        keys = np.unique(np.concatenate(merged[r["global_id"]])) if merged[r["global_id"]] else None
        item = {"bounds": f["geometry"].bounds, "centerline": f["centerline"]}
        px, supported = support(item, keys)
        r["observed_paint_pixels"] = px
        r["painted_area_pixels"] = int(f["geometry"].area / PIXEL**2)
        r["observed_fraction"] = px / max(1, r["painted_area_pixels"])
        r["supported_length"] = supported
        r["supported_fraction_of_length"] = supported / max(1e-9, r["length"])
        r["observed_supported"] = supported >= MIN_LENGTH
        r["eligible_for_independent_test"] = (
            r["observed_supported"] and r["distance_to_any_used_footprint"] >= GUARD
        )
    summary = {
        "client_solids": len(rows),
        "used_solid_ids": sum(r["used"] for r in rows),
        "unused_solid_ids": sum(not r["used"] for r in rows),
        "unused_but_overlapping_or_touching_used_solid": sum(
            (not r["used"]) and not r["identity_independent"] for r in rows
        ),
        "identity_independent": sum(r["identity_independent"] for r in rows),
        "identity_independent_long_enough_with_header_coverage": len(candidates),
        "files_decoded": len(scans),
        "files_failed_integrity": sum(not s["full_source_integrity_passed"] for s in scans),
        "observed_supported": sum(r.get("observed_supported", False) for r in candidates),
        "observed_supported_edge_lines": sum(
            r.get("observed_supported", False) and r["edge_line"] for r in candidates
        ),
        "observed_supported_curved": sum(
            r.get("observed_supported", False) and r["shape"] == "curved" for r in candidates
        ),
        "observed_supported_merge": sum(
            r.get("observed_supported", False) and r["merge_line"] for r in candidates
        ),
        "eligible_for_independent_test_ge_500_from_all_used": sum(
            r.get("eligible_for_independent_test", False) for r in candidates
        ),
        "by_acquisition_observed_supported": dict(
            Counter(a for r in candidates if r.get("observed_supported") for a in r["acquisitions"])
        ),
        "excluded_integrity_failed_sources": sorted(bad),
        "elapsed_seconds": time.perf_counter() - began,
    }
    dump(OUT / "solid_inventory.json", {"summary": summary, "solids": rows, "scans": scans})
    frame = [
        {
            **{
                k: (", ".join(v) if isinstance(v, list) else v)
                for k, v in r.items()
                if k != "header_files"
            },
            "geometry": by_id[r["global_id"]]["geometry"],
        }
        for r in candidates
    ]
    if frame:
        store = OUT / "independent_solid_candidates.gpkg"
        if store.exists():
            store.unlink()
        pyogrio.write_dataframe(
            gpd.GeoDataFrame(frame, geometry="geometry", crs=None),
            store,
            layer="independent_solid_candidates",
            driver="GPKG",
            promote_to_multi=True,
        )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
