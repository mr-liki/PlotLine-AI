"""Read-only matched LiDAR/GIS expansion into NEW-region review candidates, not labels."""

import hashlib
import json
import shutil
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import shapely
from affine import Affine
from shapely.geometry import box
from shapely.ops import unary_union

from roadmarking.dataset.geometry import painted_area_from_centerline
from roadmarking.dataset.taxonomy import Taxonomy, to_float
from roadmarking.io.pointcloud import ChunkReadReport, iter_laz_chunks
from roadmarking.io.source_inventory import fingerprint, safe_header
from roadmarking.preprocessing.intensity import IntensityCalibration
from roadmarking.rasterization.pipeline import dump, sha256, write_tif
from roadmarking.rasterization.rasterizer import RasterAccumulator, world_to_pixel
from roadmarking.visualization.intensity_render import save_intensity_png, write_png_worldfile


def main():
    root = Path(__file__).resolve().parents[1]
    context = json.loads((root / "outputs/reports/phase5_4/latest.json").read_text())
    reports = Path(context["report_dir"])
    source_audit = json.loads(
        Path(
            json.loads((root / "outputs/reports/phase5/source_audit_latest.json").read_text())[
                "report"
            ]
        ).read_text()
    )
    old = json.loads(
        Path(
            json.loads((root / "outputs/reports/phase5/production_pilot_latest.json").read_text())[
                "report"
            ]
        ).read_text()
    )
    approved = unary_union([box(*t["bounds"]) for t in old["tiles"]])
    raw = json.loads((Path(source_audit["report_directory"]) / "raw_integrity.json").read_text())
    base = root / "data/processed/phase5_4_candidates" / context["run_id"]
    output = base
    attempt = 1
    while output.exists():
        attempt += 1
        output = base.with_name(base.name + f"_attempt{attempt}")
    output.mkdir(parents=True, exist_ok=False)
    taxonomy = Taxonomy.load(root / "configs/phase4_taxonomy.yaml")
    features, references = [], []
    for index, layer in enumerate(source_audit["annotation_layers"]):
        local = Path(layer["local_snapshot"])
        if sha256(local) != layer["sha256"]:
            raise ValueError("Audited reference snapshot changed")
        copy = output / "references" / f"client_{index}.gpkg"
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, copy)
        references.append(
            {
                "path": str(copy),
                "layer": layer["layer"],
                "source": layer["source"],
                "sha256": layer["sha256"],
            }
        )
        frame = gpd.read_file(local, layer=layer["layer"], fid_as_index=True)
        duplicates = frame.geometry.to_wkb().duplicated(keep=False)
        for fid, row in frame.iterrows():
            geometry = row.geometry
            if (
                geometry is None
                or geometry.is_empty
                or not geometry.is_valid
                or duplicates.loc[fid]
            ):
                continue
            identity = f"{layer['sha256']}|{layer['layer']}|fid={fid}"
            line = geometry.geom_type in {"LineString", "MultiLineString"}
            decision = (
                taxonomy.classify_line(row.to_dict())
                if line
                else taxonomy.classify_polygon(row.to_dict())
            )
            width = to_float(row.get("LARGURA")) if line else None
            if line and (width is None or width <= 0 or decision.visual_id not in (1, 2)):
                continue
            paint = (
                painted_area_from_centerline(geometry, width)
                if line
                else shapely.force_2d(geometry)
            )
            features.append(
                {
                    "global_id": hashlib.sha256(identity.encode()).hexdigest(),
                    "source_identity": identity,
                    "geometry": geometry,
                    "paint": paint,
                    "visual_id": decision.visual_id or -1,
                    "class_flags": str(decision.flags),
                    "client_code": decision.client_code,
                    "width": width,
                    "geometry_kind": "LINE_DERIVED_DRAFT" if line else "ORIGINAL_CLIENT_POLYGON",
                }
            )
    tree = shapely.STRtree([f["geometry"] for f in features])
    old_paths = {s["path"] for s in old["full_source_scans"]}
    other_sources = []
    for record in raw:
        if (
            record["path"] in old_paths
            or "backup" in record["path"].lower()
            or record["status"] != "SAMPLE_QC_PASS_FULL_SCAN_PENDING"
        ):
            continue
        footprint = box(record["mins"][0], record["mins"][1], record["maxs"][0], record["maxs"][1])
        if footprint.distance(approved) < 500:
            continue
        hits = tree.query(footprint, predicate="intersects")
        if len(hits) >= 50:
            other_sources.append((record["point_count"], -len(hits), record))
    extra = min(other_sources, key=lambda item: (item[0], item[1]))[2] if other_sources else None
    selected_sources = []
    for scan in old["full_source_scans"]:
        if scan["group"] not in {"RIGHT", "LEFT"}:
            continue
        matching = next(r for r in raw if r["path"] == scan["path"])
        selected_sources.append((scan["group"], matching, 3))
    if extra:
        selected_sources.append(("DISTINCT_SOURCE", extra, 2))
    candidates = []
    for group, record, limit in selected_sources:
        footprint = box(record["mins"][0], record["mins"][1], record["maxs"][0], record["maxs"][1])
        pool = {}
        for index in tree.query(footprint, predicate="intersects"):
            feature = features[index]
            if feature["visual_id"] != 2:
                continue
            center = feature["geometry"].interpolate(0.5, normalized=True)
            x0 = np.floor((center.x - 25.6) / 38.4) * 38.4
            y0 = np.ceil((center.y + 25.6) / 38.4) * 38.4
            region = box(x0, y0 - 51.2, x0 + 51.2, y0)
            if region.distance(approved) < 100 or not footprint.covers(region):
                continue
            hits = tree.query(region, predicate="intersects")
            pool[(x0, y0)] = (len(hits), region, hits)
        chosen = []
        for (x0, y0), (_count, region, hits) in sorted(
            pool.items(), key=lambda item: (-item[1][0], item[0])
        ):
            if any(region.distance(item["region"]) < 100 for item in chosen):
                continue
            candidate = {
                "candidate_id": f"{group}_{len(chosen) + 1}",
                "region": region,
                "affine": Affine(0.05, 0, x0, 0, -0.05, y0),
                "hits": hits,
                "source": record,
                "accumulator": RasterAccumulator((1024, 1024)),
            }
            chosen.append(candidate)
            if len(chosen) == limit:
                break
        candidates.extend(chosen)
    calibration_json = json.loads(
        (Path(old["output_root"]) / "metadata/shared_normalization.json").read_text()
    )
    calibration = IntensityCalibration(
        **{
            key: calibration_json[key]
            for key in ("low", "high", "percentile_low", "percentile_high", "statistics", "method")
        }
    )
    scans = []
    start = time.perf_counter()
    for group, record, _ in selected_sources:
        selected = [c for c in candidates if c["source"]["path"] == record["path"]]
        if not selected:
            continue
        path = Path(record["path"])
        before = fingerprint(path)
        if before != record["fingerprint"]:
            raise ValueError("Selected raw source changed; refresh only this source before reuse")
        header = safe_header(path)
        progress = ChunkReadReport(path, 0, 0)
        tick = time.perf_counter()
        print(
            f"Rasterizing {group}: {header.point_count:,} source points "
            f"into {len(selected)} NEW windows",
            flush=True,
        )
        for points in iter_laz_chunks(path, progress):
            x, y = np.asarray(points.x), np.asarray(points.y)
            for candidate in selected:
                x0, y0, x1, y1 = candidate["region"].bounds
                keep = (x >= x0) & (x < x1) & (y > y0) & (y <= y1)
                if keep.any():
                    cols, rows = world_to_pixel(candidate["affine"], x[keep], y[keep])
                    inside = (cols >= 0) & (rows >= 0) & (cols < 1024) & (rows < 1024)
                    candidate["accumulator"].add(
                        cols[inside], rows[inside], np.asarray(points.intensity)[keep][inside]
                    )
        passed = (
            progress.points_read == header.point_count
            and not progress.dropped_points
            and before == fingerprint(path)
        )
        scans.append(
            {
                "source": str(path),
                "actual_valid_points": progress.points_read,
                "expected_points": int(header.point_count),
                "dropped_points": progress.dropped_points,
                "full_selected_source_integrity_passed": bool(passed),
                "seconds": time.perf_counter() - tick,
                "fingerprint": before,
            }
        )
        dump(output / "source_integrity.json", scans)
        if not passed:
            raise ValueError("New candidate source failed strict integrity")
    approved_manifest = json.loads((root / "outputs/reports/phase5_3/report.json").read_text())
    prior_ids = {
        f["properties"]["global_id"]
        for f in json.loads(
            (Path(approved_manifest["dataset"]) / "metadata/approved_geometry.geojson").read_text()
        )["features"]
    }
    tiles, draft_rows = [], []
    for candidate in candidates:
        name, affine = candidate["candidate_id"], candidate["affine"]
        layers = candidate["accumulator"].layers(calibration)
        folder = output / "rasters" / name
        paths = {}
        for key, array in layers.items():
            path = folder / (key + ".tif")
            write_tif(path, array, affine, None)
            paths[key] = str(path)
        gray = output / "images" / (name + ".png")
        save_intensity_png(gray, layers["normalized"], layers["observed"], gamma=0.7)
        write_png_worldfile(gray, affine)
        ids = []
        for index in candidate["hits"]:
            feature = features[index]
            paint = feature["paint"].intersection(candidate["region"])
            if paint.is_empty or paint.area <= 0:
                continue
            ids.append(feature["global_id"])
            draft_rows.append(
                {
                    **{k: v for k, v in feature.items() if k not in {"geometry", "paint"}},
                    "candidate_id": name,
                    "review_status": "NEW_REGION_REVIEW_PENDING",
                    "training_eligible": 0,
                    "geometry": paint,
                }
            )
        tiles.append(
            {
                "candidate_id": name,
                "bounds": list(candidate["region"].bounds),
                "affine": list(affine)[:6],
                "source": candidate["source"]["path"],
                "images": {"grayscale": str(gray)},
                "layers": paths,
                "observed_pixels": int(layers["observed"].sum()),
                "annotation_footprints": len(ids),
                "source_global_ids": ids,
                "shared_ids_with_original_approved_scope": sorted(set(ids) & prior_ids),
                "region_completeness_approved": False,
                "new_paint_conversion_approved": False,
                "training_eligible": False,
                "all_survey_passes_merged": False,
                "normalization": (
                    "Existing shared source calibration, fixed gamma 0.7; no per-tile fitting"
                ),
            }
        )
    store = output / "candidate_outlines.gpkg"
    if draft_rows:
        pyogrio.write_dataframe(
            gpd.GeoDataFrame(draft_rows, geometry="geometry", crs=None),
            store,
            layer="candidate_paint_footprints",
            driver="GPKG",
            promote_to_multi=True,
        )
    summary = {
        "output": str(output),
        "tiles": tiles,
        "references": references,
        "candidate_store": str(store),
        "candidate_windows": len(tiles),
        "unique_candidate_annotation_ids": len({r["global_id"] for r in draft_rows}),
        "candidate_footprint_appearances": len(draft_rows),
        "training_eligible_new_regions": 0,
        "source_integrity": scans,
        "elapsed_seconds": time.perf_counter() - start,
        "review_scope": (
            "NEW geographically distinct windows only; "
            "existing six approved windows are untouched"
        ),
        "next_requirement": (
            "Review new conversion alignment and visible-region completeness "
            "before adding these NEW regions to supervised training"
        ),
    }
    dump(output / "review_spec.json", summary)
    dump(reports / "new_region_candidates.json", summary)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "output",
                    "candidate_windows",
                    "unique_candidate_annotation_ids",
                    "training_eligible_new_regions",
                    "elapsed_seconds",
                )
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
