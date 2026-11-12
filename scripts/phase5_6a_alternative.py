"""Freeze Phase 5.5 and prepare reserved test regions; no inference before new review."""

import hashlib
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import shapely
from affine import Affine
from shapely.geometry import box

from roadmarking.dataset.geometry import painted_area_from_centerline
from roadmarking.dataset.phase55 import write_attestation
from roadmarking.dataset.taxonomy import Taxonomy, to_float
from roadmarking.io.pointcloud import ChunkReadReport, iter_laz_chunks
from roadmarking.io.source_inventory import fingerprint, safe_header
from roadmarking.preprocessing.intensity import IntensityCalibration
from roadmarking.rasterization.pipeline import dump, sha256, write_tif
from roadmarking.rasterization.rasterizer import RasterAccumulator, world_to_pixel
from roadmarking.visualization.intensity_render import save_intensity_png, write_png_worldfile


def main():
    root = Path(__file__).resolve().parents[1]
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    report = root / "outputs/reports/phase5_6a" / run_id
    output = root / "data/processed/phase5_6a_test_review" / run_id
    report.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    context = {"run_id": run_id, "report_dir": str(report), "output": str(output)}
    dump(report.parent / "latest.json", context)
    prior = json.loads((root / "outputs/reports/phase5_5/latest.json").read_text())
    prior_report = Path(prior["report_dir"])
    inference_path = prior_report / "selected_inference.json"
    inference = json.loads(inference_path.read_text())
    checkpoint = Path(inference["checkpoint"])
    weight = checkpoint / "model/model.safetensors"
    if sha256(weight) != inference["weight_sha256"]:
        raise ValueError("Selected checkpoint weights changed")
    model_config = json.loads((checkpoint / "model/config.json").read_text())
    if model_config["id2label"] != {"0": "longitudinal_solid", "1": "longitudinal_dash"}:
        raise ValueError("Unexpected model class mapping")
    processor = json.loads((checkpoint / "processor/preprocessor_config.json").read_text())
    if processor["do_resize"]:
        raise ValueError("Native inference requires do_resize=False")
    post_path = Path(inference["postprocessing_override"])
    postprocessing = json.loads(post_path.read_text())
    old = json.loads(
        Path(
            json.loads((root / "outputs/reports/phase5/production_pilot_latest.json").read_text())[
                "report"
            ]
        ).read_text()
    )
    calibration_path = Path(old["output_root"]) / "metadata/shared_normalization.json"
    calibration_json = json.loads(calibration_path.read_text())
    calibration = IntensityCalibration(
        **{
            k: calibration_json[k]
            for k in ("low", "high", "percentile_low", "percentile_high", "statistics", "method")
        }
    )
    frozen_files = [
        weight,
        inference_path,
        post_path,
        calibration_path,
        *sorted((checkpoint / "model").glob("*.json")),
        *sorted((checkpoint / "processor").glob("*.json")),
        root / "src/roadmarking/inference/calibration.py",
        root / "src/roadmarking/training/continuous.py",
        root / "src/roadmarking/visualization/intensity_render.py",
    ]
    frozen = {
        "checkpoint": str(checkpoint),
        "artifact_sha256": {str(p): sha256(p) for p in frozen_files},
        "classes": model_config["id2label"],
        "processor": processor,
        "postprocessing": postprocessing,
        "normalization": calibration_json,
        "gamma": 0.7,
        "pixel_size_native_units": 0.05,
        "crop_configuration": {"size": 512, "stride": 256, "padding": "excluded"},
        "raw_proposal_floor": 0.25,
        "mask_probability_threshold": 0.5,
        "minimum_observed_mask_pixels": 4,
        "nodata": "Only observed source pixels; no filling; exclude unsupported and boundary areas",
        "global_reconciliation": (
            "Frozen same-class mask IoU NMS .3 on global valid pixels; " "no union"
        ),
        "global_reconciliation_status": "Planned before test inference; not run before approval",
        "test_tuning_forbidden": True,
        "training_forbidden": True,
    }
    freeze_hash = write_attestation(report / "frozen_model.json", frozen)
    phase54 = json.loads((root / "outputs/reports/phase5_4/latest.json").read_text())
    reservation_path = Path(phase54["report_dir"]) / "future_test_region_reservations.json"
    reservations = json.loads(reservation_path.read_text())["regions"]
    dataset = Path(prior["dataset"])
    manifest_path = dataset / "metadata/dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    reservations[-1]["raw_candidate"] = reservations[-1]["raw_candidate"].replace(
        "Track01_24.laz", "Track01_52.laz"
    )
    used_ids = set().union(*(set(r["source_global_ids"]) for r in manifest["regions"]))
    audit = []
    for r in reservations:
        distances = [
            box(*r["bounds"]).distance(box(*u["native_xy_bbox"])) for u in manifest["regions"]
        ]
        shared = set(r["reference_annotation_ids"]) & used_ids
        row = {
            "region": r["reservation_id"],
            "minimum_train_development_distance": min(distances),
            "shared_train_development_ids": sorted(shared),
            "footprint_guard": 10.0,
            "independent_from_train_and_development": not shared and min(distances) > 10,
        }
        if not row["independent_from_train_and_development"]:
            raise ValueError("Reserved test leakage detected")
        audit.append(row)
    between = []
    for i, a in enumerate(reservations):
        for b in reservations[:i]:
            between.append(
                {
                    "regions": [a["reservation_id"], b["reservation_id"]],
                    "distance": box(*a["bounds"]).distance(box(*b["bounds"])),
                    "shared_source_ids": sorted(
                        set(a["reference_annotation_ids"]) & set(b["reference_annotation_ids"])
                    ),
                }
            )
    dump(
        report / "independence_audit.json",
        {
            "training_manifest": str(manifest_path),
            "training_manifest_sha256": sha256(manifest_path),
            "reservations_sha256": sha256(reservation_path),
            "regions": audit,
            "between_test_regions": between,
            "test_region_count_is_not_independent_sample_count": True,
            "note": (
                "Long source features link test regions; "
                "retain source IDs for geographic deduplication"
            ),
        },
    )
    source_audit = json.loads(
        Path(
            json.loads((root / "outputs/reports/phase5/source_audit_latest.json").read_text())[
                "report"
            ]
        ).read_text()
    )
    raw_inventory = json.loads(
        (Path(source_audit["report_directory"]) / "raw_integrity.json").read_text()
    )
    taxonomy = Taxonomy.load(root / "configs/phase4_taxonomy.yaml")
    features, references = [], []
    for index, source in enumerate(source_audit["annotation_layers"]):
        local = Path(source["local_snapshot"])
        if sha256(local) != source["sha256"]:
            raise ValueError("Client reference snapshot changed")
        copy = output / "references" / f"client_{index}.gpkg"
        copy.parent.mkdir(exist_ok=True)
        shutil.copyfile(local, copy)
        references.append(
            {
                "path": str(copy),
                "layer": source["layer"],
                "source": source["source"],
                "sha256": source["sha256"],
            }
        )
        frame = gpd.read_file(local, layer=source["layer"], fid_as_index=True)
        duplicates = frame.geometry.to_wkb().duplicated(keep=False)
        for fid, row in frame.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty or not geom.is_valid or duplicates.loc[fid]:
                continue
            line = geom.geom_type in {"LineString", "MultiLineString"}
            decision = (
                taxonomy.classify_line(row.to_dict())
                if line
                else taxonomy.classify_polygon(row.to_dict())
            )
            width = to_float(row.get("LARGURA")) if line else None
            if line and (width is None or width <= 0 or decision.visual_id not in (1, 2)):
                continue
            identity = f"{source['sha256']}|{source['layer']}|fid={fid}"
            paint = painted_area_from_centerline(geom, width) if line else shapely.force_2d(geom)
            features.append(
                {
                    "global_id": hashlib.sha256(identity.encode()).hexdigest(),
                    "source_identity": identity,
                    "visual_id": decision.visual_id or -1,
                    "width": width,
                    "geometry_kind": "LINE_DERIVED_DRAFT" if line else "CLIENT_POLYGON",
                    "geometry": paint,
                }
            )
    rows, tiles, scans = [], [], []
    began = time.perf_counter()
    for region in reservations:
        name, bounds = region["reservation_id"], region["bounds"]
        x0, y0, x1, y1 = bounds
        affine = Affine(0.05, 0, x0, 0, -0.05, y1)
        shape = box(*bounds)
        path = Path(region["raw_candidate"])
        record = next(r for r in raw_inventory if r["path"] == str(path))
        before = fingerprint(path)
        if before != record["fingerprint"]:
            raise ValueError("Reserved source changed since inventory")
        header = safe_header(path)
        if not box(header.mins[0], header.mins[1], header.maxs[0], header.maxs[1]).intersects(
            shape
        ):
            raise ValueError("Reserved source does not intersect requested footprint")
        accumulator = RasterAccumulator((1024, 1024))
        progress = ChunkReadReport(path, 0, 0)
        tick = time.perf_counter()
        print(f"Strict full-source rendering {name}: {header.point_count:,} points", flush=True)
        for points in iter_laz_chunks(path, progress):
            x, y = np.asarray(points.x), np.asarray(points.y)
            keep = (x >= x0) & (x < x1) & (y > y0) & (y <= y1)
            if keep.any():
                cols, rr = world_to_pixel(affine, x[keep], y[keep])
                inside = (cols >= 0) & (rr >= 0) & (cols < 1024) & (rr < 1024)
                accumulator.add(
                    cols[inside], rr[inside], np.asarray(points.intensity)[keep][inside]
                )
        if (
            progress.points_read != header.point_count
            or progress.dropped_points
            or before != fingerprint(path)
        ):
            raise ValueError("Reserved source strict full-point integrity failed")
        try:
            crs = header.parse_crs()
        except Exception:
            crs = None
        scans.append(
            {
                "region": name,
                "source": str(path),
                "fingerprint": before,
                "expected_points": int(header.point_count),
                "actual_points": progress.points_read,
                "dropped_points": progress.dropped_points,
                "full_source_integrity_passed": True,
                "crs_wkt": None if crs is None else crs.to_wkt(),
                "coordinate_units": "UNCONFIRMED" if crs is None else str(crs.axis_info),
                "seconds": time.perf_counter() - tick,
            }
        )
        layers = accumulator.layers(calibration)
        layers.update(
            nodata=(~layers["observed"].astype(bool)).astype(np.uint8),
            padding=np.zeros((1024, 1024), np.uint8),
        )
        paths = {}
        for key, array in layers.items():
            target = output / "rasters" / name / (key + ".tif")
            write_tif(target, array, affine, None)
            paths[key] = str(target)
        png = output / "images" / (name + ".png")
        save_intensity_png(png, layers["normalized"], layers["observed"], gamma=0.7)
        write_png_worldfile(png, affine)
        hits = [f for f in features if f["geometry"].intersects(shape)]
        actual_ids = {f["global_id"] for f in hits}
        if actual_ids & used_ids:
            raise ValueError(
                "Actual painted footprint reveals previously undetected source-ID leakage"
            )
        for feature in hits:
            paint = feature["geometry"].intersection(shape)
            if paint.is_empty or paint.area <= 0:
                continue
            rows.append(
                {
                    **feature,
                    "geometry": paint,
                    "candidate_id": name,
                    "review_status": "TEST_GROUND_TRUTH_REVIEW_PENDING",
                    "training_eligible": 0,
                }
            )
        tiles.append(
            {
                "candidate_id": name,
                "bounds": bounds,
                "affine": list(affine)[:6],
                "source": str(path),
                "layers": paths,
                "images": {"grayscale": str(png)},
                "observed_pixels": int(layers["observed"].sum()),
                "source_global_ids": sorted(actual_ids),
                "annotation_footprints": len(hits),
                "reservation_ids_not_in_painted_candidates": sorted(
                    set(region["reference_annotation_ids"]) - actual_ids
                ),
                "additional_painted_candidate_ids": sorted(
                    actual_ids - set(region["reference_annotation_ids"])
                ),
                "approval_status": "PENDING_NEW_TEST_REGION_ONLY",
                "training_eligible": False,
            }
        )
        dump(report / "source_integrity.json", scans)
    store = output / "candidate_outlines.gpkg"
    pyogrio.write_dataframe(
        gpd.GeoDataFrame(rows, geometry="geometry", crs=None),
        store,
        layer="candidate_paint_footprints",
        driver="GPKG",
        promote_to_multi=True,
    )
    spec = {
        "output": str(output),
        "tiles": tiles,
        "references": references,
        "candidate_store": str(store),
        "source_integrity": scans,
        "freeze_sha256": freeze_hash,
        "candidate_windows": len(tiles),
        "unique_candidate_annotation_ids": len({r["global_id"] for r in rows}),
        "elapsed_seconds": time.perf_counter() - began,
        "status": "TEST_PREPARATION_REQUIRED",
        "inference_performed": False,
    }
    dump(report / "new_region_candidates.json", spec)
    dump(output / "review_spec.json", spec)
    context.update(status="TEST_PREPARATION_REQUIRED")
    dump(report.parent / "latest.json", context)
    print(
        json.dumps({"report": str(report), "output": str(output), "status": spec["status"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
