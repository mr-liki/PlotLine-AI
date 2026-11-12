"""Source-backed review candidates only; no training label export or inference."""

import shutil
import sys
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import rasterio
import shapely
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import box

sys.argv = sys.argv[:1]
from phase5_11_common import ROOT, load  # noqa: E402
from phase5_11_discover import render_file, tag, window_stats  # noqa: E402
from roadmarking.io.source_inventory import fingerprint  # noqa: E402
from roadmarking.preprocessing.intensity import IntensityCalibration  # noqa: E402
from roadmarking.rasterization.pipeline import dump, sha256, write_tif  # noqa: E402
from roadmarking.visualization.intensity_render import (  # noqa: E402
    save_intensity_png,
    write_png_worldfile,
)
from stage9_common import BASE, context, identity_group, out_dir  # noqa: E402


def support_stats(window, ctx):
    observed = window["layers"]["observed"].astype(bool)
    affine = window["affine"]
    by_class = {"solid": np.zeros_like(observed), "dash": np.zeros_like(observed)}
    values = []
    for row in window["observed_ids"]:
        f = ctx["by_id"][row["global_id"]]
        mask = (
            rasterize(
                [(f["geometry"], 1)], out_shape=observed.shape, transform=affine, dtype="uint8"
            ).astype(bool)
            & observed
        )
        r, c = np.nonzero(mask)
        x, y = affine * (c + 0.5, r + 0.5)
        along = shapely.line_locate_point(f["centerline"], shapely.points(x, y))
        length = int(np.count_nonzero(np.bincount(np.floor(along).astype(int)) >= 3))
        by_class[row["class"]] |= mask
        values.append(
            {
                **row,
                "physical_group": identity_group(ctx, row["global_id"]),
                "supported_length_native_bins": length,
                "client_width_native": f["width"],
                "median_paint_normalized_intensity": (
                    float(np.median(window["layers"]["normalized"][mask])) if mask.any() else None
                ),
            }
        )
    crops = []
    for r in (0, 512):
        for c in (0, 512):
            sl = (slice(r, r + 512), slice(c, c + 512))
            valid = float(observed[sl].mean()) >= 0.1
            solid, dash = int(by_class["solid"][sl].sum()), int(by_class["dash"][sl].sum())
            crops.append(
                {
                    "window": [r, c, 512, 512],
                    "eligible_after_first_geometry_review": valid and solid + dash >= 64,
                    "observed_solid_paint_pixels": solid,
                    "observed_dash_paint_pixels": dash,
                }
            )
    return {
        "per_id": values,
        "nonoverlapping_512_crop_candidates": crops,
        "eligible_nonoverlap_crops_after_approval": sum(
            c["eligible_after_first_geometry_review"] for c in crops
        ),
        "hard_negative_review_candidate_pixels": window["bright_background_pixels"],
        "background_verified": False,
    }


def main():
    out = out_dir()
    ctx = context()
    matrix = load(out / "opportunity_matrix.json")
    data = ROOT / "data/processed/stage9_review" / out.name
    data.mkdir(parents=True, exist_ok=True)
    raw = load(Path(ctx["source"]["report_directory"]) / "raw_integrity.json")
    raw = {r["path"]: r for r in raw}
    tree = shapely.STRtree([f["geometry"] for f in ctx["features"]])
    windows = []
    for k, row in enumerate(matrix["selected"], 1):
        region = box(*row["bounds"])
        hits = tree.query(region, predicate="intersects")
        preferred = (
            "Job_20250924_1354"
            if row["proposed_role"] == "training_candidate"
            else "Job_20250708_1218"
        )
        queue = sorted(
            [raw[p] for p in row["covering_integrity_passed_sources"]],
            key=lambda r: (
                Path(r["path"]).name.split("_Track")[0] != preferred,
                -r["declared_bbox_density_points_per_native_square"],
            ),
        )
        windows.append(
            {
                "name": f"N9_{k:02d}",
                "region": region,
                "hits": hits,
                "stats": window_stats(region, hits, ctx["features"]),
                "tag": tag(window_stats(region, hits, ctx["features"])),
                "queue": queue,
                "attempts": [],
                "opportunity": row,
            }
        )
    frozen = load(ROOT / "outputs/reports/phase5_6a/20261106T093757339115Z/frozen_model.json")
    cal_path = next(p for p in frozen["artifact_sha256"] if p.endswith("shared_normalization.json"))
    cal = load(cal_path)
    assert sha256(cal_path) == frozen["artifact_sha256"][cal_path]
    calibration = IntensityCalibration(
        **{
            k: cal[k]
            for k in ("low", "high", "percentile_low", "percentile_high", "statistics", "method")
        }
    )
    scans = []
    previous = load(BASE / "latest.json").get("previous_report_dir")
    if previous and (Path(previous) / "new_region_candidates.json").exists():
        old = load(Path(previous) / "new_region_candidates.json")
        for w in windows:
            matches = [
                t
                for t in old["tiles"]
                if t["bounds"] == list(w["region"].bounds)
                and t["proposed_role"] == w["opportunity"]["proposed_role"]
            ]
            if not matches:
                continue
            tile = matches[0]
            scan = next(
                s
                for s in old["source_integrity"]
                if s["source"] == tile["source"] and s["full_source_integrity_passed"]
            )
            assert fingerprint(Path(tile["source"])) == scan["fingerprint"]
            layers = {}
            for key, path in tile["layers"].items():
                with rasterio.open(path) as ds:
                    layers[key] = ds.read(1)
            w.update(
                {
                    "layers": layers,
                    "affine": Affine(*tile["affine"]),
                    "source": raw[tile["source"]],
                    "meaningful": True,
                    "verified_stage9": True,
                    "observed_fraction": tile["observed_fraction"],
                    "observed_paint_pixels": tile["observed_paint_pixels"],
                    "observed_ids": tile["observed_ids"],
                    "support_stats": tile["support_statistics"],
                    "bright_background_pixels": tile["support_statistics"][
                        "hard_negative_review_candidate_pixels"
                    ],
                    "reuse_provenance": {
                        "manifest": str(Path(previous) / "new_region_candidates.json"),
                        "raster_sha256": {path: sha256(path) for path in tile["layers"].values()},
                    },
                }
            )
            scans.append({**scan, "integrity_record_reused": True})
            print(
                f"{w['name']}: reused source-verified rasters from prior Stage-9 draft; source fingerprint unchanged",  # noqa: E501
                flush=True,
            )
    for attempt in range(3):
        per_file = defaultdict(list)
        for w in windows:
            if not w.get("verified_stage9") and attempt < len(w["queue"]):
                w["source"] = w["queue"][attempt]
                w.pop("meaningful", None)
                per_file[w["source"]["path"]].append(w)
        for path, group in per_file.items():
            scan = render_file(
                path, group, list(raw.values()), calibration, ctx["features"], attempt
            )
            scans.append(scan)
            for w in group:
                if not w.get("meaningful") or not scan["full_source_integrity_passed"]:
                    continue
                stats = support_stats(w, ctx)
                new = set(w["opportunity"]["new_solid_groups"])
                new_supported = [
                    r
                    for r in stats["per_id"]
                    if r["class"] == "solid"
                    and r["physical_group"] in new
                    and r["supported_length_native_bins"] >= 10
                ]
                dash = [r for r in stats["per_id"] if r["class"] == "dash"]
                w["verified_stage9"] = bool(new_supported) and (
                    bool(dash) or w["opportunity"]["proposed_role"] == "training_candidate"
                )
                w["support_stats"] = stats
            dump(out / "source_integrity.json", scans)
            print(
                f"{Path(path).name}: integrity={scan['full_source_integrity_passed']}; verified={sum(bool(w.get('verified_stage9')) for w in group)}/{len(group)}",  # noqa: E501
                flush=True,
            )
    verified = [w for w in windows if w.get("verified_stage9")]
    dump(
        out / "review_discovery.json",
        {
            "selected": len(windows),
            "verified": len(verified),
            "failures": [
                {"region": w["name"], "attempts": w["attempts"]}
                for w in windows
                if not w.get("verified_stage9")
            ],
            "no_supervised_labels_exported": True,
            "frozen_normalization": {"path": cal_path, "sha256": sha256(cal_path)},
        },
    )
    if not verified:
        print("No new suitable region passed observed-support verification")
        return
    references = []
    for k, layer in enumerate(ctx["layers"]):
        target = data / "references" / f"client_{k}.gpkg"
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(layer["local_snapshot"], target)
        references.append(
            {
                "path": str(target),
                "layer": layer["layer"],
                "source": layer["source"],
                "sha256": layer["sha256"],
            }
        )
    paint, unsupported, footprints, tiles = [], [], [], []
    for w in verified:
        observed = w["layers"]["observed"].astype(bool)
        w["layers"]["nodata"] = (~observed).astype("uint8")
        w["layers"]["padding"] = np.zeros_like(observed, dtype="uint8")
        paths = {}
        for key, array in w["layers"].items():
            path = data / "rasters" / w["name"] / f"{key}.tif"
            write_tif(path, array, w["affine"], None)
            paths[key] = str(path)
        png = data / "images" / f"{w['name']}.png"
        save_intensity_png(png, w["layers"]["normalized"], observed, gamma=0.7)
        write_png_worldfile(png, w["affine"])
        per_id = {p["global_id"]: p for p in w["observed_ids"]}
        role = w["opportunity"]["proposed_role"]
        for j in w["hits"]:
            f = ctx["features"][int(j)]
            geom = f["geometry"].intersection(w["region"])
            if geom.is_empty or geom.area <= 0:
                continue
            row = {
                "global_id": f["global_id"],
                "candidate_id": w["name"],
                "physical_group": identity_group(ctx, f["global_id"]),
                "proposed_role": role,
                "source_identity": f["source_identity"],
                "source_file": w["source"]["path"],
                "source_acquisition": Path(w["source"]["path"]).name.split("_Track")[0],
                "geometry": geom,
            }
            if f["visual_id"] in (1, 2):
                paint.append(
                    {
                        **row,
                        "visual_id": f["visual_id"],
                        "class_name": "solid" if f["visual_id"] == 1 else "dash",
                        "designation": f["designation"],
                        "width": f["width"],
                        "observed_pixels": per_id.get(f["global_id"], {}).get("observed_pixels", 0),
                        "review_status": "FIRST_APPROVAL_REQUIRED",
                        "training_eligible": 0,
                    }
                )
            else:
                unsupported.append(
                    {
                        **row,
                        "code": str(f["code"]),
                        "policy": "AMBIGUOUS_OR_UNSUPPORTED_EXCLUDE_NOT_BACKGROUND",
                    }
                )
        footprints.append(
            {
                "candidate_id": w["name"],
                "proposed_role": role,
                "source_file": w["source"]["path"],
                "source_acquisition": Path(w["source"]["path"]).name.split("_Track")[0],
                "review_status": "FIRST_APPROVAL_REQUIRED",
                "geometry": w["region"],
            }
        )
        tiles.append(
            {
                "candidate_id": w["name"],
                "priority_tag": role + "_" + w["tag"],
                "proposed_role": role,
                "bounds": list(w["region"].bounds),
                "affine": list(w["affine"])[:6],
                "source": w["source"]["path"],
                "acquisition": Path(w["source"]["path"]).name.split("_Track")[0],
                "layers": paths,
                "images": {"grayscale": str(png)},
                "content_stats": w["stats"],
                "observed_fraction": w["observed_fraction"],
                "observed_paint_pixels": w["observed_paint_pixels"],
                "observed_ids": w["observed_ids"],
                "source_global_ids": sorted(
                    ctx["features"][int(j)]["global_id"] for j in w["hits"]
                ),
                "physical_groups": w["opportunity"]["physical_groups"],
                "new_solid_groups": w["opportunity"]["new_solid_groups"],
                "support_statistics": w["support_stats"],
                "split_audit": w["opportunity"],
                "raster_cache_reuse": w.get("reuse_provenance"),
                "approval_status": "PENDING_FIRST_HUMAN_REVIEW",
                "training_eligible": False,
            }
        )
    store = data / "candidate_outlines.gpkg"
    for name, rows in (
        ("candidate_paint_footprints", paint),
        ("unsupported_marking_regions", unsupported),
        ("review_region_footprints", footprints),
    ):
        if rows:
            pyogrio.write_dataframe(
                gpd.GeoDataFrame(rows, geometry="geometry", crs=None),
                store,
                layer=name,
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
                    for k in (
                        "global_id",
                        "candidate_id",
                        "class_name",
                        "designation",
                        "width",
                        "physical_group",
                        "proposed_role",
                        "geometry",
                    )
                }
                | {"review_decision": "", "reviewer_note": ""}
                for r in paint
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
        "stage": "9",
        "output": str(data),
        "candidate_store": str(store),
        "candidate_store_sha256": sha256(store),
        "review_edits_store": str(edits),
        "references": references,
        "tiles": tiles,
        "source_integrity": scans,
        "status": "NEW_REGIONS_ONLY_FIRST_HUMAN_REVIEW_REQUIRED",
        "training_eligible": False,
    }
    dump(out / "new_region_candidates.json", spec)
    dump(BASE / "latest.json", {**load(BASE / "latest.json"), "review_data_dir": str(data)})
    for opportunity in matrix["opportunities"]:
        for tile in tiles:
            if opportunity["opportunity_id"] == tile["split_audit"]["opportunity_id"]:
                opportunity.update(
                    {
                        "actual_window_observed_paint_pixels": tile["observed_paint_pixels"],
                        "eligible_nonoverlap_crops": tile["support_statistics"][
                            "eligible_nonoverlap_crops_after_approval"
                        ],
                        "intensity_characteristics": tile["support_statistics"]["per_id"],
                        "review_region": tile["candidate_id"],
                    }
                )
    dump(out / "opportunity_matrix.json", matrix)
    print(f"New review package: {len(tiles)} regions; no approved/test region reused or changed")


if __name__ == "__main__":
    main()
