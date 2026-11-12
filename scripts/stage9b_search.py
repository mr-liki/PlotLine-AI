"""Bounded source-backed text search outside proposed development/protected data."""

import sys
from collections import defaultdict
from pathlib import Path

import shapely
from shapely.geometry import box

sys.argv = sys.argv[:1]
from phase5_11_common import ROOT, load  # noqa: E402
from phase5_11_discover import render_file, window_stats  # noqa: E402
from roadmarking.preprocessing.intensity import IntensityCalibration  # noqa: E402
from roadmarking.rasterization.pipeline import dump, sha256, write_tif  # noqa: E402
from roadmarking.visualization.intensity_render import (  # noqa: E402
    save_intensity_png,
    write_png_worldfile,
)
from stage9_common import context, identity_group  # noqa: E402


def main():
    base = ROOT / "outputs/reports/stage9b"
    latest = load(base / "latest.json")
    out = Path(latest["report_dir"])
    data = ROOT / "data/processed/stage9b_review" / latest["run_id"]
    data.mkdir(parents=True, exist_ok=True)
    ctx = context()
    stage9 = load(
        Path(load(ROOT / "outputs/reports/stage9/latest.json")["report_dir"])
        / "new_region_candidates.json"
    )
    forbidden = [r["box"] for r in ctx["used"] if not r["role"].startswith("train")]
    forbidden += [
        box(*t["bounds"]) for t in stage9["tiles"] if t["proposed_role"] == "development_candidate"
    ]
    dev_groups = {
        identity_group(ctx, u) for r in ctx["used"] if "dev" in r["role"] for u in r["ids"]
    }
    dev_groups |= {
        g
        for t in stage9["tiles"]
        if t["proposed_role"] == "development_candidate"
        for g in t["physical_groups"]
    }
    prior_seen = [
        box(*r["tile"]["bounds"]) for r in load(out / "cached_training_search_inventory.json")
    ]
    prior_seen += [box(*t["bounds"]) for t in stage9["tiles"]]
    raw = load(Path(ctx["source"]["report_directory"]) / "raw_integrity.json")
    passed = {
        r["source"]
        for r in load(ROOT / "outputs/reports/stage8/8_3_solid_discovery/solid_inventory.json")[
            "scans"
        ]
        if r["full_source_integrity_passed"]
    }
    raw = [r for r in raw if r["path"] in passed]
    tree = shapely.STRtree([f["geometry"] for f in ctx["features"]])
    candidates = []
    for f in ctx["features"]:
        if f["visual_id"] not in (3, 4):
            continue
        point = f["geometry"].centroid
        region = box(point.x - 25.6, point.y - 25.6, point.x + 25.6, point.y + 25.6)
        distance = min(region.distance(g) for g in forbidden)
        if distance < 500 or any(
            region.intersection(g).area > 0.25 * region.area for g in prior_seen
        ):
            continue
        hits = tree.query(region, predicate="intersects")
        groups = {
            identity_group(ctx, ctx["features"][int(i)]["global_id"])
            for i in hits
            if ctx["features"][int(i)]["visual_id"] in (1, 2)
        }
        if groups & dev_groups:
            continue
        arrows = sum(ctx["features"][int(i)]["visual_id"] in (3, 4) for i in hits)
        files = [
            r
            for r in raw
            if r["mins"][0] <= point.x - 25.6
            and r["maxs"][0] >= point.x + 25.6
            and r["mins"][1] <= point.y - 25.6
            and r["maxs"][1] >= point.y + 25.6
        ]
        if arrows < 2 or not files:
            continue
        candidates.append(
            {
                "region": region,
                "hits": hits,
                "arrows": arrows,
                "files": files,
                "distance": distance,
                "groups": groups,
            }
        )
    chosen = []
    for c in sorted(candidates, key=lambda c: (-c["arrows"], -c["distance"])):
        if all(c["region"].distance(w["region"]) >= 500 for w in chosen):
            c.update(
                name=f"TXT_SEARCH_{len(chosen)+1:02}",
                attempts=[],
                queue=sorted(
                    c["files"],
                    key=lambda r: (
                        "Job_20250708_1218" not in Path(r["path"]).name,
                        -r["declared_bbox_density_points_per_native_square"],
                    ),
                ),
            )
            chosen.append(c)
        if len(chosen) >= 8:
            break
    frozen = load(ROOT / "outputs/reports/phase5_6a/20261106T093757339115Z/frozen_model.json")
    path = next(p for p in frozen["artifact_sha256"] if p.endswith("shared_normalization.json"))
    assert sha256(path) == frozen["artifact_sha256"][path]
    cal = load(path)
    calibration = IntensityCalibration(
        **{
            k: cal[k]
            for k in ("low", "high", "percentile_low", "percentile_high", "statistics", "method")
        }
    )
    scans = []
    for attempt in range(2):
        per_file = defaultdict(list)
        for w in chosen:
            if not w.get("meaningful") and attempt < len(w["queue"]):
                w["source"] = w["queue"][attempt]
                per_file[w["source"]["path"]].append(w)
        for source, group in per_file.items():
            scan = render_file(source, group, raw, calibration, ctx["features"], attempt)
            scans.append(scan)
            dump(out / "text_search_source_integrity.json", scans)
            print(
                f"{Path(source).name}: full integrity={scan['full_source_integrity_passed']}; observed windows={sum(bool(w.get('meaningful')) for w in group)}/{len(group)}",  # noqa: E501
                flush=True,
            )
    tiles = []
    for w in chosen:
        if not w.get("meaningful"):
            continue
        layers = w["layers"]
        observed = layers["observed"].astype(bool)
        layers["nodata"] = (~observed).astype("uint8")
        paths = {}
        for key, array in layers.items():
            path = data / "search_rasters" / w["name"] / f"{key}.tif"
            write_tif(path, array, w["affine"], None)
            paths[key] = str(path)
        png = data / "search_images" / f"{w['name']}.png"
        save_intensity_png(png, layers["normalized"], observed, gamma=0.7)
        write_png_worldfile(png, w["affine"])
        tiles.append(
            {
                "candidate_id": w["name"],
                "bounds": list(w["region"].bounds),
                "affine": list(w["affine"])[:6],
                "source": w["source"]["path"],
                "acquisition": Path(w["source"]["path"]).name.split("_Track")[0],
                "layers": paths,
                "images": {"grayscale": str(png)},
                "observed_fraction": w["observed_fraction"],
                "distance_to_development_protected_historical_native": w["distance"],
                "physical_groups": sorted(w["groups"]),
                "client_area_stats": window_stats(w["region"], w["hits"], ctx["features"]),
                "role": "TRAINING_TEXT_SEARCH_ONLY_NOT_APPROVED",
                "text_visibility": "PENDING_VISUAL_INSPECTION",
            }
        )
    dump(
        out / "text_training_search.json",
        {
            "client_text_features_available": 0,
            "cached_training_windows_inspected": 35,
            "metadata_candidate_windows": len(candidates),
            "source_backed_search_windows": tiles,
            "source_integrity": scans,
            "search_rule": "Arrow-rich client geometry guides imagery search only; arrows never become road_text labels. All windows are >=500 native units from development/protected/historical footprints and their line groups are disjoint from development groups. No supervised labels exported.",  # noqa: E501
        },
    )
    dump(base / "latest.json", {**latest, "data_dir": str(data)})
    print(
        f"Rendered {len(tiles)} source-backed new training search windows; text is not assumed present"  # noqa: E501
    )


if __name__ == "__main__":
    main()
