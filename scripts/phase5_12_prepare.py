"""Phase 5.12: package the 10 point-verified Phase 5.11 reserve windows for human review.

Each window is re-rendered from the ONE source file that already passed verification in
Phase 5.11 (no share-wide search). Strict full-point integrity is re-checked. Nothing here
approves anything; every region stays PENDING_FIRST_APPROVAL / training_eligible=0.
"""

import json
import shutil
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import shapely
from shapely.geometry import box

from phase5_11_common import ROOT, client_features, load, reserved_geography
from phase5_11_discover import HALF, render_file, tag, window_stats
from roadmarking.preprocessing.intensity import IntensityCalibration
from roadmarking.rasterization.pipeline import dump, sha256, write_tif
from roadmarking.visualization.intensity_render import save_intensity_png, write_png_worldfile

RESERVE = [f"P511_{k}" for k in range(17, 27)]
GUARD = 500.0


def main():
    began = time.perf_counter()
    p511 = load(ROOT / "outputs/reports/phase5_11/latest.json")
    out511 = Path(p511["report_dir"])
    ranking = load(out511 / "candidate_ranking.json")
    spec511 = load(out511 / "new_region_candidates.json")
    attempts = spec511["verification"]["attempts_per_window"]
    assert set(RESERVE) == set(spec511["verification"]["meaningful_but_not_packaged"])
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = ROOT / "outputs/reports/phase5_12" / run_id
    data = ROOT / "data/processed/phase5_12_review" / run_id
    out.mkdir(parents=True)
    data.mkdir(parents=True)

    features, layers, audit = client_features()
    tree = shapely.STRtree([f["geometry"] for f in features])
    regions, extra, _ = reserved_geography()
    # The 16 approved Phase 5.11 regions are now training geography.
    for t in spec511["tiles"]:
        regions.append(
            {
                "id": t["candidate_id"],
                "role": "train_phase5_11",
                "bounds": t["bounds"],
                "box": box(*t["bounds"]),
                "ids": t["source_global_ids"],
            }
        )
    reserved_ids = set(extra)
    train_ids = set()
    for r in regions:
        (train_ids if r["role"].startswith("train") else reserved_ids).update(r["ids"])
    raw = [
        r
        for r in load(Path(audit["report_directory"]) / "raw_integrity.json")
        if "backup" not in r["path"].lower()
    ]

    windows = []
    for k, old in enumerate(RESERVE, start=1):
        entry = ranking["selected"][int(old.split("_")[1]) - 1]
        x, y = entry["centre"]
        region = box(x - HALF, y - HALF, x + HALF, y + HALF)
        hits = tree.query(region, predicate="intersects")
        stats = window_stats(region, hits, features)
        assert tag(stats) == entry["tag"], "reserve content changed since Phase 5.11"
        source = attempts[old][-1]["source"]
        distances = defaultdict(lambda: np.inf)
        for r in regions:
            distances[r["role"]] = min(distances[r["role"]], region.distance(r["box"]))
        ids = {features[j]["global_id"] for j in hits}
        windows.append(
            {
                "name": f"P512_{k:02d}",
                "reserve_of": old,
                "x": x,
                "y": y,
                "region": region,
                "hits": hits,
                "stats": stats,
                "tag": entry["tag"],
                "source_path": source,
                "attempts": [],
                "distance_by_role": dict(distances),
                "shared_reserved_ids": sorted(ids & reserved_ids),
                "shared_training_ids": sorted(ids & train_ids),
            }
        )
    for w in windows:
        others = [o["region"].distance(w["region"]) for o in windows if o is not w]
        w["min_distance_to_other_reserve"] = min(others)
        leaks = [
            role
            for role, d in w["distance_by_role"].items()
            if not role.startswith("train") and d < GUARD
        ]
        if leaks or w["shared_reserved_ids"]:
            raise ValueError(
                f"{w['name']} violates independence: {leaks} "
                f"{len(w['shared_reserved_ids'])} reserved IDs"
            )

    cal_path = next(
        p
        for p in load(
            ROOT / "outputs/reports/phase5_6a/20261106T093757339115Z/" "frozen_model.json"
        )["artifact_sha256"]
        if p.endswith("shared_normalization.json")
    )
    cal = load(cal_path)
    calibration = IntensityCalibration(
        **{
            k: cal[k]
            for k in ("low", "high", "percentile_low", "percentile_high", "statistics", "method")
        }
    )
    by_file, scans = defaultdict(list), []
    for w in windows:
        by_file[w["source_path"]].append(w)
    for path, group in by_file.items():
        scan = render_file(path, group, raw, calibration, features, 0)
        scans.append(scan)
        print(
            f"{Path(path).name}: ok={scan['full_source_integrity_passed']} "
            f"{scan['seconds']:.0f}s",
            flush=True,
        )
        if not scan["full_source_integrity_passed"]:
            raise ValueError(f"Previously verified source now fails integrity: {path}")
    dump(out / "source_integrity.json", scans)
    missing = [w["name"] for w in windows if not w.get("meaningful")]
    if missing:
        raise ValueError(f"Re-render no longer meets observation criteria: {missing}")

    references = []
    for index, item in enumerate(layers):
        copy = data / "references" / f"client_{index}.gpkg"
        copy.parent.mkdir(exist_ok=True)
        shutil.copyfile(item["local_snapshot"], copy)
        references.append(
            {
                "path": str(copy),
                "layer": item["layer"],
                "source": item["source"],
                "sha256": item["sha256"],
            }
        )
    rows, unsupported_rows, footprint_rows, tiles = [], [], [], []
    for w in windows:
        name, affine, lay = w["name"], w["affine"], w["layers"]
        observed = lay["observed"].astype(bool)
        lay["nodata"] = (~observed).astype(np.uint8)
        lay["padding"] = np.zeros(observed.shape, np.uint8)
        paths = {}
        for key, array in lay.items():
            target = data / "rasters" / name / (key + ".tif")
            write_tif(target, array, affine, None)
            paths[key] = str(target)
        png = data / "images" / (name + ".png")
        save_intensity_png(png, lay["normalized"], lay["observed"], gamma=0.7)
        write_png_worldfile(png, affine)
        obs = {p["global_id"]: p for p in w["observed_ids"]}
        supported, unsupported_codes, low_obs = Counter(), Counter(), []
        unobserved_area = total_area = 0.0
        for j in w["hits"]:
            f = features[j]
            geom = f["geometry"].intersection(w["region"])
            if geom.is_empty or geom.area <= 0:
                continue
            base = {
                "global_id": f["global_id"],
                "source_identity": f["source_identity"],
                "client_sig": f["client_sig"],
                "marking_group": f["group"],
                "candidate_id": name,
                "client_layer": f["layer"],
            }
            if f["visual_id"] in (1, 2):
                cls = "solid" if f["visual_id"] == 1 else "dash"
                supported[cls] += 1
                px = obs.get(f["global_id"], {}).get("observed_pixels", 0)
                if f["global_id"] not in obs:
                    low_obs.append(f["global_id"])
                expected = geom.area / 0.0025
                total_area += expected
                unobserved_area += max(0.0, expected - px)
                rows.append(
                    {
                        **base,
                        "visual_id": f["visual_id"],
                        "class_name": cls,
                        "designation": f["designation"],
                        "width": f["width"],
                        "observed_pixels": px,
                        "review_status": "PHASE5_12_REVIEW_PENDING",
                        "training_eligible": 0,
                        "geometry": geom,
                    }
                )
            else:
                unsupported_codes[str(f["code"])] += 1
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
                "reserve_of": w["reserve_of"],
                "priority_tag": w["tag"],
                "source_file": Path(w["source_path"]).name,
                "observed_fraction": w["observed_fraction"],
                "review_status": "PENDING_FIRST_APPROVAL",
                "geometry": w["region"],
            }
        )
        new_ids = [p for p in w["observed_ids"] if p["global_id"] not in train_ids]
        tiles.append(
            {
                "candidate_id": name,
                "reserve_of": w["reserve_of"],
                "priority_tag": w["tag"],
                "bounds": list(w["region"].bounds),
                "affine": list(affine)[:6],
                "source": w["source_path"],
                "acquisition": Path(w["source_path"]).name.split("_Track")[0],
                "layers": paths,
                "images": {"grayscale": str(png)},
                "content_stats": w["stats"],
                "observed_fraction": w["observed_fraction"],
                "observed_paint_pixels": w["observed_paint_pixels"],
                "observed_solid_ids": sum(p["class"] == "solid" for p in w["observed_ids"]),
                "observed_dash_ids": sum(p["class"] == "dash" for p in w["observed_ids"]),
                "new_observed_solid_ids": sum(p["class"] == "solid" for p in new_ids),
                "new_observed_dash_ids": sum(p["class"] == "dash" for p in new_ids),
                "observed_ids": w["observed_ids"],
                "supported_classes_in_footprint": dict(supported),
                "unsupported_codes_in_footprint": dict(unsupported_codes),
                "targets_without_observed_pixels": low_obs,
                "target_paint_area_unobserved_fraction": unobserved_area / max(1.0, total_area),
                "bright_background_pixels": w["bright_background_pixels"],
                "source_global_ids": sorted({features[j]["global_id"] for j in w["hits"]}),
                "shares_ids_with_existing_training": w["shared_training_ids"],
                "shared_ids_with_dev_benchmark_or_test": w["shared_reserved_ids"],
                "minimum_distance_by_role_native": w["distance_by_role"],
                "minimum_distance_to_other_reserve_native": w["min_distance_to_other_reserve"],
                "approval_status": "PENDING_FIRST_APPROVAL",
                "training_eligible": False,
            }
        )
    store = data / "candidate_outlines.gpkg"
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
    # Editable reviewer workspace (separate file so the read-only store hash never changes).
    edits = data / "review_edits.gpkg"
    review_rows = [
        {
            **{
                k: r[k] for k in ("global_id", "candidate_id", "class_name", "designation", "width")
            },
            "review_decision": "",
            "reviewer_note": "",
            "geometry": r["geometry"],
        }
        for r in rows
    ]
    pyogrio.write_dataframe(
        gpd.GeoDataFrame(review_rows, geometry="geometry", crs=None),
        edits,
        layer="reviewer_outline_check",
        driver="GPKG",
        promote_to_multi=True,
    )
    pyogrio.write_dataframe(
        gpd.GeoDataFrame({"candidate_id": [], "issue": [], "note": []}, geometry=[], crs=None),
        edits,
        layer="reviewer_missing_paint",
        driver="GPKG",
        geometry_type="MultiPolygon",
        append=True,
    )
    spec = {
        "phase": "5.12",
        "output": str(data),
        "candidate_store": str(store),
        "candidate_store_sha256": sha256(store),
        "review_edits_store": str(edits),
        "references": references,
        "tiles": tiles,
        "source_integrity": scans,
        "new_unique_observed_identities": {
            "solid": len(
                {
                    p["global_id"]
                    for t in tiles
                    for p in t["observed_ids"]
                    if p["class"] == "solid" and p["global_id"] not in train_ids
                }
            ),
            "dash": len(
                {
                    p["global_id"]
                    for t in tiles
                    for p in t["observed_ids"]
                    if p["class"] == "dash" and p["global_id"] not in train_ids
                }
            ),
        },
        "elapsed_seconds": time.perf_counter() - began,
        "status": "REAL_DATA_REVIEW_REQUIRED",
        "training_eligible": False,
    }
    dump(out / "new_region_candidates.json", spec)
    dump(data / "review_spec.json", spec)
    dump(
        ROOT / "outputs/reports/phase5_12/latest.json",
        {
            "run_id": run_id,
            "report_dir": str(out),
            "data_dir": str(data),
            "status": "REVIEW_PACKAGE_PENDING_QGIS",
        },
    )
    print(
        json.dumps(
            {
                "report_dir": str(out),
                "new_unique_observed_identities": spec["new_unique_observed_identities"],
                "regions": [
                    (
                        t["candidate_id"],
                        t["reserve_of"],
                        t["priority_tag"],
                        t["observed_solid_ids"],
                        t["observed_dash_ids"],
                        round(t["observed_fraction"], 2),
                    )
                    for t in tiles
                ],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
