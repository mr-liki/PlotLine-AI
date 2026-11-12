"""Search fresh, geographically separated LiDAR windows for real road text."""

from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import shapely
from shapely.geometry import box

from phase5_11_common import ROOT, load
from phase5_11_discover import render_file
from roadmarking.preprocessing.intensity import IntensityCalibration
from roadmarking.rasterization.pipeline import dump, sha256, write_tif
from roadmarking.visualization.intensity_render import save_intensity_png, write_png_worldfile
from stage9_common import context, identity_group

base = ROOT / "outputs/reports/stage9c_2"
out = base / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
out.mkdir(parents=True)
dump(base / "latest.json", {"run": str(out), "status": "SEARCHING"})
ctx = context()
reports = ROOT / "outputs/reports"
old9 = Path(load(reports / "stage9/latest.json")["report_dir"])
old9b = Path(load(reports / "stage9b/latest.json")["report_dir"])
stage9 = load(old9 / "new_region_candidates.json")["tiles"]
search9b = load(old9b / "text_training_search.json")
historical = [r["box"] for r in ctx["used"]] + [box(*t["bounds"]) for t in stage9]
prior_search = [box(*t["bounds"]) for t in search9b["source_backed_search_windows"]]
forbidden_groups = {
    identity_group(ctx, uid)
    for r in ctx["used"]
    if "dev" in r["role"] or "protected" in r["role"]
    for uid in r["ids"]
} | ctx["protected_groups"]
forbidden_groups.update(group for tile in stage9 for group in tile["physical_groups"])

raw = load(Path(ctx["source"]["report_directory"]) / "raw_integrity.json")
verified_paths = {
    row["source"]
    for row in load(reports / "stage8/8_3_solid_discovery/solid_inventory.json")["scans"]
    if row["full_source_integrity_passed"]
}
raw = [row for row in raw if row["path"] in verified_paths]
features = ctx["features"]
tree = shapely.STRtree([item["geometry"] for item in features])
candidates = []
for feature in features:
    if feature["visual_id"] not in (3, 4):
        continue
    center = feature["geometry"].centroid
    region = box(center.x - 25.6, center.y - 25.6, center.x + 25.6, center.y + 25.6)
    distance = min(region.distance(item) for item in historical)
    if distance < 50 or any(region.intersects(item) for item in prior_search):
        continue
    hits = tree.query(region, predicate="intersects")
    groups = {
        identity_group(ctx, features[int(i)]["global_id"])
        for i in hits
        if features[int(i)]["visual_id"] in (1, 2)
    }
    if groups & forbidden_groups:
        continue
    left, bottom, right, top = region.bounds
    sources = [
        row
        for row in raw
        if row["mins"][0] <= left
        and row["mins"][1] <= bottom
        and row["maxs"][0] >= right
        and row["maxs"][1] >= top
    ]
    if sources:
        arrows = sum(features[int(i)]["visual_id"] in (3, 4) for i in hits)
        candidates.append(
            dict(
                region=region,
                hits=hits,
                groups=groups,
                distance=distance,
                arrows=arrows,
                sources=sources,
                attempts=[],
            )
        )

selected = []
by_source = defaultdict(list)
for item in sorted(candidates, key=lambda w: (-w["arrows"], -w["distance"])):
    if any(item["region"].centroid.distance(w["region"].centroid) < 10 for w in selected):
        continue
    sources = sorted(
        item["sources"], key=lambda row: (-len(by_source[row["path"]]), Path(row["path"]).name)
    )
    source = sources[0]
    if len(by_source[source["path"]]) >= 3:
        continue
    item.update(name=f"R92_{len(selected)+1:02}", source=source)
    by_source[source["path"]].append(item)
    selected.append(item)
    if len(selected) == 12:
        break

inventory = {
    "verified_source_paths": [row["path"] for row in raw],
    "metadata_candidate_windows": len(candidates),
    "planned_windows": [
        dict(
            id=w["name"],
            bounds=list(w["region"].bounds),
            source=w["source"]["path"],
            nearest_reserved_distance_native=w["distance"],
            line_groups=sorted(w["groups"]),
        )
        for w in selected
    ],
    "reserved_footprints_checked": len(historical),
    "prior_text_search_windows_excluded": len(prior_search),
    "future_mixed_500_test_inventory_supplied": False,
}
dump(out / "source_inventory.json", inventory)
frozen = load(reports / "phase5_6a/20261106T093757339115Z/frozen_model.json")
cal_path = next(
    path for path in frozen["artifact_sha256"] if path.endswith("shared_normalization.json")
)
assert sha256(cal_path) == frozen["artifact_sha256"][cal_path]
cal = load(cal_path)
calibration = IntensityCalibration(
    **{
        key: cal[key]
        for key in ("low", "high", "percentile_low", "percentile_high", "statistics", "method")
    }
)
scans, tiles = [], []
for path, windows in by_source.items():
    if not windows:
        continue
    try:
        scan = render_file(path, windows, raw, calibration, features, 0, observed_only=True)
    except (OSError, ValueError, RuntimeError) as exc:
        scan = {"source": path, "full_source_integrity_passed": False, "error": str(exc)}
    scans.append(scan)
    print(Path(path).name, scan["full_source_integrity_passed"], flush=True)
    if scan["full_source_integrity_passed"]:
        for window in windows:
            if not window.get("meaningful"):
                continue
            layers = window["layers"]
            observed = layers["observed"].astype(bool)
            layers["nodata"] = (~observed).astype("uint8")
            folder = out / window["name"]
            paths = {}
            for key, array in layers.items():
                raster = folder / f"{key}.tif"
                write_tif(raster, array, window["affine"], None)
                paths[key] = str(raster)
            png = folder / "intensity.png"
            save_intensity_png(png, layers["normalized"], observed, gamma=0.7)
            write_png_worldfile(png, window["affine"])
            tiles.append(
                dict(
                    id=window["name"],
                    source=path,
                    acquisition=Path(path).name.split("_Track")[0],
                    bounds=list(window["region"].bounds),
                    affine=list(window["affine"])[:6],
                    observed_fraction=float(observed.mean()),
                    layers=paths,
                    image=str(png),
                    line_groups=sorted(window["groups"]),
                    reserved_distance_native=window["distance"],
                )
            )
    dump(out / "source_scan_results.json", {"scans": scans, "rendered_windows": tiles})
dump(out / "source_scan_results.json", {"scans": scans, "rendered_windows": tiles})
print(f"{len(tiles)} intensity windows rendered", flush=True)
