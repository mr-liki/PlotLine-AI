"""Stage 6: diversity-ranked new regions for human review (no approval, no training).

Candidates are 51.2 m windows from client content, >=500 units from every existing footprint
(including all 26 approved Phase 5.11/5.12 training regions) and free of development /
benchmark / protected-test IDs. Selection is farthest-point (diversity-first) in a
standardized content + acquisition + geography space, seeded by the labelled regions — not
by model uncertainty. Point coverage is verified by fully decoding the assigned LAZ file.
One packaged region (the most distant from all training) is proposed as a fresh development
region; the user decides its role.
"""

import json
import shutil
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import shapely
from shapely.geometry import box

sys.argv = sys.argv[:1]  # phase5_11_discover reads argv at import
from phase5_11_common import ROOT, client_features, load, reserved_geography  # noqa: E402
from phase5_11_discover import HALF, render_file, tag, window_stats  # noqa: E402
from roadmarking.preprocessing.intensity import IntensityCalibration  # noqa: E402
from roadmarking.rasterization.pipeline import dump, sha256, write_tif  # noqa: E402
from roadmarking.visualization.intensity_render import (  # noqa: E402
    save_intensity_png,
    write_png_worldfile,
)

GUARD = 500.0
VERIFY = 14
PACKAGE = 10
TRAINED_JOB = "Job_20250708_0834"


def vector(stats, x, y, new_job):
    return np.array(
        [
            stats["curved_solids"],
            stats["edge_lines"],
            stats["merge_lines"],
            min(stats["dashes"], 30),
            float(stats["interchange_features"] > 0),
            stats["unsupported_area"] / 50.0,
            float(new_job),
            x / 1000.0,
            y / 1000.0,
        ]
    )


def main():
    began = time.perf_counter()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = ROOT / "outputs/reports/stage6_review" / run_id
    data = ROOT / "data/processed/stage6_review" / run_id
    out.mkdir(parents=True)
    data.mkdir(parents=True)
    features, layers, audit = client_features()
    tree = shapely.STRtree([f["geometry"] for f in features])
    regions, extra, _ = reserved_geography()
    for phase in ("phase5_11", "phase5_12"):
        spec = load(
            Path(load(ROOT / f"outputs/reports/{phase}/latest.json")["report_dir"])
            / "new_region_candidates.json"
        )
        for t in spec["tiles"]:
            regions.append(
                {
                    "id": t["candidate_id"],
                    "role": "train_approved",
                    "bounds": t["bounds"],
                    "box": box(*t["bounds"]),
                    "ids": t["source_global_ids"],
                }
            )
    reserved_ids, train_ids = set(extra), set()
    for r in regions:
        (train_ids if r["role"].startswith("train") else reserved_ids).update(r["ids"])
    footprints = [r["box"] for r in regions]
    raw = [
        r
        for r in load(Path(audit["report_directory"]) / "raw_integrity.json")
        if "backup" not in r["path"].lower()
    ]
    bad = {
        s["source"]
        for p in ("phase5_11", "phase5_12")
        for s in load(
            Path(load(ROOT / f"outputs/reports/{p}/latest.json")["report_dir"])
            / "source_integrity.json"
        )
        if not s["full_source_integrity_passed"]
    }
    raw = [r for r in raw if r["path"] not in bad]
    centres = []
    for f in features:
        if f["visual_id"] == 1:
            for d in np.arange(0, f["centerline"].length, 25.0):
                p = f["centerline"].interpolate(d)
                centres.append((p.x, p.y))
        elif f["visual_id"] == 2 and int(f["global_id"][:8], 16) % 5 == 0:
            c = f["geometry"].centroid
            centres.append((c.x, c.y))
    cands = []
    for x, y in centres:
        region = box(x - HALF, y - HALF, x + HALF, y + HALF)
        if min(region.distance(b) for b in footprints) < GUARD:
            continue
        hits = tree.query(region, predicate="intersects")
        if {features[i]["global_id"] for i in hits} & reserved_ids:
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
        st = window_stats(region, hits, features)
        if st["solid_lines"] == 0 and st["dashes"] < 6:
            continue
        jobs = {Path(r["path"]).name.split("_Track")[0] for r in files}
        cands.append(
            {
                "x": x,
                "y": y,
                "region": region,
                "hits": hits,
                "stats": st,
                "files": files,
                "new_job": bool(jobs - {TRAINED_JOB}),
                "tag": tag(st),
            }
        )
    # Labelled-set feature vectors seed the farthest-point sampler.
    seed_vecs = []
    for r in regions:
        if not r["role"].startswith("train"):
            continue
        hits = tree.query(r["box"], predicate="intersects")
        c = r["box"].centroid
        seed_vecs.append(vector(window_stats(r["box"], hits, features), c.x, c.y, False))
    vecs = np.array([vector(c["stats"], c["x"], c["y"], c["new_job"]) for c in cands])
    allv = np.vstack([vecs, seed_vecs])
    mu, sd = allv.mean(0), allv.std(0) + 1e-9
    z, seeds = (vecs - mu) / sd, (np.array(seed_vecs) - mu) / sd
    dist = np.min(np.linalg.norm(z[:, None] - seeds[None], axis=-1), axis=1)
    chosen = []
    while len(chosen) < VERIFY and dist.max() > 0:
        i = int(np.argmax(dist))
        if all(cands[i]["region"].distance(cands[j]["region"]) >= GUARD for j in chosen):
            chosen.append(i)
            dist = np.minimum(dist, np.linalg.norm(z - z[i], axis=1))
        dist[i] = -1
    windows = []
    for k, i in enumerate(chosen, start=1):
        c = cands[i]
        c.update(
            name=f"S6_{k:02d}",
            attempts=[],
            queue=sorted(
                c["files"],
                key=lambda r: (
                    Path(r["path"]).name.split("_Track")[0] == TRAINED_JOB,
                    -r["declared_bbox_density_points_per_native_square"],
                ),
            ),
        )
        windows.append(c)
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
    scans = []
    for attempt in range(3):
        by_file = defaultdict(list)
        for w in windows:
            if not w.get("meaningful") and attempt < len(w["queue"]):
                w["source"] = w["queue"][attempt]
                by_file[w["source"]["path"]].append(w)
        for path, group in by_file.items():
            scans.append(render_file(path, group, raw, calibration, features, attempt))
            print(f"{Path(path).name}: ok={scans[-1]['full_source_integrity_passed']}", flush=True)
    dump(out / "source_integrity.json", scans)
    verified = [w for w in windows if w.get("meaningful")][:PACKAGE]
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
    for w in verified:
        lay, affine = w["layers"], w["affine"]
        observed = lay["observed"].astype(bool)
        lay["nodata"] = (~observed).astype(np.uint8)
        lay["padding"] = np.zeros(observed.shape, np.uint8)
        paths = {}
        for key, array in lay.items():
            target = data / "rasters" / w["name"] / (key + ".tif")
            write_tif(target, array, affine, None)
            paths[key] = str(target)
        png = data / "images" / (w["name"] + ".png")
        save_intensity_png(png, lay["normalized"], lay["observed"], gamma=0.7)
        write_png_worldfile(png, affine)
        obs = {p["global_id"]: p for p in w["observed_ids"]}
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
                "candidate_id": w["name"],
                "client_layer": f["layer"],
            }
            if f["visual_id"] in (1, 2):
                rows.append(
                    {
                        **base,
                        "visual_id": f["visual_id"],
                        "class_name": "solid" if f["visual_id"] == 1 else "dash",
                        "designation": f["designation"],
                        "width": f["width"],
                        "observed_pixels": obs.get(f["global_id"], {}).get("observed_pixels", 0),
                        "review_status": "STAGE6_REVIEW_PENDING",
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
                "candidate_id": w["name"],
                "priority_tag": w["tag"],
                "source_file": Path(w["source"]["path"]).name,
                "observed_fraction": w["observed_fraction"],
                "review_status": "PENDING_FIRST_APPROVAL",
                "geometry": w["region"],
            }
        )
        dmin = min(w["region"].distance(r["box"]) for r in regions if r["role"].startswith("train"))
        tiles.append(
            {
                "candidate_id": w["name"],
                "priority_tag": w["tag"],
                "bounds": list(w["region"].bounds),
                "affine": list(affine)[:6],
                "source": w["source"]["path"],
                "acquisition": Path(w["source"]["path"]).name.split("_Track")[0],
                "layers": paths,
                "images": {"grayscale": str(png)},
                "content_stats": w["stats"],
                "observed_fraction": w["observed_fraction"],
                "observed_paint_pixels": w["observed_paint_pixels"],
                "observed_solid_ids": sum(p["class"] == "solid" for p in w["observed_ids"]),
                "observed_dash_ids": sum(p["class"] == "dash" for p in w["observed_ids"]),
                "new_observed_solid_ids": sum(
                    p["class"] == "solid" and p["global_id"] not in train_ids
                    for p in w["observed_ids"]
                ),
                "new_observed_dash_ids": sum(
                    p["class"] == "dash" and p["global_id"] not in train_ids
                    for p in w["observed_ids"]
                ),
                "observed_ids": w["observed_ids"],
                "source_global_ids": sorted({features[j]["global_id"] for j in w["hits"]}),
                "shares_ids_with_existing_training": sorted(
                    {p["global_id"] for p in w["observed_ids"]} & train_ids
                ),
                "min_distance_to_training_native": dmin,
                "min_distance_to_any_reserved_native": min(
                    w["region"].distance(b) for b in footprints
                ),
                "approval_status": "PENDING_FIRST_APPROVAL",
                "training_eligible": False,
            }
        )
    fresh = (
        max(tiles, key=lambda t: t["min_distance_to_training_native"])["candidate_id"]
        if tiles
        else None
    )
    for t in tiles:
        t["proposed_role"] = (
            "FRESH_DEVELOPMENT_CANDIDATE" if t["candidate_id"] == fresh else "TRAINING_CANDIDATE"
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
    edits = data / "review_edits.gpkg"
    pyogrio.write_dataframe(
        gpd.GeoDataFrame(
            [
                {
                    k: r[k]
                    for k in ("global_id", "candidate_id", "class_name", "designation", "width")
                }
                | {"review_decision": "", "reviewer_note": "", "geometry": r["geometry"]}
                for r in rows
            ],
            geometry="geometry",
            crs=None,
        ),
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
        "stage": 6,
        "output": str(data),
        "candidate_store": str(store),
        "candidate_store_sha256": sha256(store),
        "review_edits_store": str(edits),
        "references": references,
        "tiles": tiles,
        "source_integrity": scans,
        "candidates_considered": len(cands),
        "verified": len(verified),
        "selection": "diversity-first farthest-point sampling seeded by labelled regions",
        "fresh_development_candidate": fresh,
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
    dump(
        ROOT / "outputs/reports/stage6_review/latest.json",
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
                k: spec[k]
                for k in (
                    "candidates_considered",
                    "verified",
                    "fresh_development_candidate",
                    "new_unique_observed_identities",
                )
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
