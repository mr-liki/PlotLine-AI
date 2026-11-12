"""Reuse QC-final client linework as attributed draft paint areas on complete pilot images."""

import hashlib
import json
import os
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

import geopandas as gpd
import pyogrio
import rasterio
from shapely.geometry import MultiPolygon, box
from shapely.ops import unary_union

from roadmarking.dataset.alignment import measure_line_displacement
from roadmarking.dataset.annotation_store import FIELDS, create_store, frame_id
from roadmarking.dataset.annotation_validation import validate_store
from roadmarking.dataset.annotation_workspace import taxonomy
from roadmarking.dataset.geometry import line_parts, painted_area_from_centerline, tile_affine
from roadmarking.dataset.labels import LabelFeature
from roadmarking.dataset.qa import tile_overlay
from roadmarking.dataset.taxonomy import Taxonomy, to_float
from roadmarking.dataset.verified_masks import reviewed_masks
from roadmarking.dataset.writer import read_tile_inputs, write_sample
from roadmarking.rasterization.pipeline import dump, sha256, write_tif


def run(root):
    report_dir = root / "outputs/reports/phase5"
    pointer = json.loads((report_dir / "production_pilot_latest.json").read_text())
    output = Path(pointer["output_root"])
    pilot = json.loads(Path(pointer["report"]).read_text())
    audit = json.loads(
        Path(
            json.loads((report_dir / "source_audit_latest.json").read_text())["report"]
        ).read_text()
    )
    attestation = json.loads((report_dir / "source_qc_attestation.json").read_text())
    if attestation["source_annotation_hashes"] != audit["source_annotation_hashes"]:
        raise ValueError("User QC attestation does not match audited source identities")
    if not all(x["full_integrity_passed"] for x in pilot["full_source_scans"]):
        raise ValueError("Rejecting pilot with failed full-source integrity")
    workspace = output / ("approval_workspace_" + time.strftime("%Y%m%dT%H%M%S"))
    for folder in ("annotation_projects", "annotations_draft", "metadata", "quality_control"):
        (workspace / folder).mkdir(parents=True, exist_ok=False)
    classes, client_taxonomy = taxonomy(root), Taxonomy.load(root / "configs/phase4_taxonomy.yaml")
    tiles = json.loads((output / "manifests/raster_tiles.json").read_text())
    # QGIS may cache raster statistics. Materialize owned display copies only.
    display_tiles, source_hashes = [], {}
    for tile in tiles:
        tile["source_group"] = tile["source"]
        tile["source"] = "CORRECTED_SURVEY"
        tile.update(
            source_file=tile["raw_source"],
            native_xy_bbox=tile["bounds"],
            crs_status="CRS_UNCONFIRMED",
        )
        display = json.loads(json.dumps(tile))
        directory = workspace / "annotation_projects/rasters" / tile["tile_id"]
        directory.mkdir(parents=True)
        for section, keys in (
            ("images", ("grayscale",)),
            ("layers", ("normalized",)),
            ("masks", ("observed", "valid")),
        ):
            for key in keys:
                original = Path(tile[section][key])
                source_hashes[str(original)] = sha256(original)
                target = directory / original.name
                shutil.copyfile(original, target)
                if original.suffix == ".png":
                    shutil.copyfile(original.with_suffix(".pgw"), target.with_suffix(".pgw"))
                display[section][key] = str(target)
        display_tiles.append(display)
    store = create_store(
        workspace / "annotations_draft/road_marking_annotations.gpkg", tiles, classes
    )
    regions = {"CORRECTED_SURVEY": unary_union([box(*t["bounds"]) for t in tiles])}
    drafts, source_records, refs, excluded = [], [], [], []
    for number, layer in enumerate(audit["annotation_layers"]):
        local = Path(layer["local_snapshot"])
        source_hashes[str(local)] = layer["sha256"]
        if sha256(local) != layer["sha256"]:
            raise ValueError("Changed source snapshot")
        target = workspace / "annotation_projects" / f"original_client_qc_{number}.gpkg"
        shutil.copyfile(local, target)
        refs.append(
            {
                "path": str(target),
                "layer": layer["layer"],
                "name": "QC-final original: " + local.name,
            }
        )
        frame = gpd.read_file(local, layer=layer["layer"], fid_as_index=True)
        duplicates = frame.geometry.to_wkb().duplicated(keep=False)
        for fid, row in frame.iterrows():
            geometry = row.geometry
            if geometry is None or geometry.is_empty or not geometry.is_valid:
                excluded.append(
                    {"source": layer["source"], "fid": int(fid), "reason": "INVALID_GEOMETRY"}
                )
                continue
            if geometry.geom_type not in {"LineString", "MultiLineString"}:
                continue  # Original QC-final polygons remain visible; AI-code mappings unresolved.
            decision = client_taxonomy.classify_line(row.to_dict())
            width = to_float(row.get("LARGURA"))
            if (
                decision.visual_id not in (1, 2)
                or width is None
                or width <= 0
                or duplicates.loc[fid]
            ):
                excluded.append(
                    {
                        "source": layer["source"],
                        "fid": int(fid),
                        "reason": "WIDTH_CLASS_OR_DUPLICATE_REQUIRES_ADJUDICATION",
                    }
                )
                continue
            identity = f"{layer['sha256']}|{layer['layer']}|fid={fid}"
            gid = hashlib.sha256(identity.encode()).hexdigest()
            area = painted_area_from_centerline(geometry, width)
            for group, region in regions.items():
                if not area.intersects(region):
                    continue
                paint = area.intersection(region)
                if paint.is_empty or paint.area <= 0:
                    continue
                if paint.geom_type == "Polygon":
                    paint = MultiPolygon([paint])
                if paint.geom_type != "MultiPolygon":
                    continue
                anchor = next(
                    t for t in tiles if t["source"] == group and paint.intersects(box(*t["bounds"]))
                )
                record = dict.fromkeys(FIELDS)
                record.update(
                    {
                        "annotation_id": gid + ":" + group,
                        "global_instance_id": gid,
                        "source_track": group,
                        "source_raster_id": anchor["tile_id"],
                        "source_tile_id": anchor["tile_id"],
                        "visual_class_id": decision.visual_id,
                        "visual_class_name": decision.visual_class,
                        "client_code": decision.client_code,
                        "geometry_type": "MultiPolygon",
                        "group_id": str(row.get("ID_GRUPO", "")),
                        "marking_width": width,
                        "annotation_source": "BUFFER_DERIVED_FROM_QC_FINAL_SOURCE",
                        "review_status": "PROPOSED_UNREVIEWED",
                        "width_verified": 0,
                        "mask_reviewed": 0,
                        "source_coordinate_frame": frame_id(group),
                        "purpose": "PAINT",
                        "confidence_or_quality_note": (
                            "Original geometry/attributes QC-final per user; "
                            "new pixel-area conversion pending QA"
                        ),
                        "comment": (
                            f"{identity}; ID_SIG={row.get('ID_SIG')}; no coordinate transform"
                        ),
                        "geometry": paint,
                    }
                )
                drafts.append(record)
                source_records.append(
                    {
                        "global_id": gid,
                        "group": group,
                        "geometry": geometry,
                        "paint": paint,
                        "visual_id": decision.visual_id,
                        "identity": identity,
                        "width": width,
                    }
                )
    if drafts:
        pyogrio.write_dataframe(
            gpd.GeoDataFrame(drafts, geometry="geometry", crs=None),
            store,
            layer="paint_polygons",
            append=True,
        )
    dump(workspace / "metadata/class_taxonomy.json", classes)
    dump(workspace / "metadata/source_qc_attestation.json", attestation)
    validation = validate_store(store)
    dump(workspace / "metadata/annotation_validation.json", validation)
    if validation["global_errors"] or validation["invalid_features"]:
        raise ValueError(
            "Draft workspace validation failed; inspect metadata/annotation_validation.json"
        )
    converted, offsets = [], []
    # These masks visualize the conversion. They are never primary supervised labels.
    mask_output = workspace / "conversion_review_only"
    for tile in tiles:
        inputs = read_tile_inputs(tile)
        with rasterio.open(tile["masks"]["valid"]) as ds:
            valid = ds.read(1).astype(bool)
        selected = [
            r
            for r in source_records
            if r["group"] == tile["source"] and r["paint"].intersects(box(*tile["bounds"]))
        ]
        features = [
            LabelFeature(index + 1, r["global_id"], r["visual_id"], "polygon", r["paint"])
            for index, r in enumerate(source_records)
            if r in selected
        ]
        labels, categories = reviewed_masks(
            tile_affine(tile), (1024, 1024), inputs["observed"], valid, features
        )
        converted.append(
            write_sample(
                mask_output,
                tile,
                inputs,
                labels,
                {
                    "role": "QC_SOURCE_DERIVED_CONVERSION_REVIEW_ONLY_NOT_TRAINING_LABELS",
                    "source_annotations_qc_final": True,
                    "new_paint_area_approval": False,
                    "stock_loss_eligible": False,
                    "reviewed_background_pixels": 0,
                    "source_qc_attestation": str(report_dir / "source_qc_attestation.json"),
                },
            )
        )
        write_tif(
            mask_output / "validity_masks" / f"{tile['tile_id']}_categories.tif",
            categories,
            tile_affine(tile),
            None,
        )
        tile_overlay(
            output / "qa_previews" / f"{tile['tile_id']}_conversion.png",
            tile,
            inputs["normalized"],
            [r["geometry"] for r in selected],
            labels,
            "QC-final source / derived area QA ONLY / " + tile["tile_id"],
        )
        clipped = [
            part
            for r in selected
            for part in line_parts(r["geometry"].intersection(box(*tile["bounds"])))
        ]
        offset = measure_line_displacement(
            clipped,
            inputs["normalized"],
            tile_affine(tile),
            half_window=0.5,
            spacing=0.5,
            min_peak=0.1,
        )
        offsets.append({"tile_id": tile["tile_id"], "intensity_peak_offset_proxy": offset})
    dump(
        mask_output / "metadata/dataset_manifest.json",
        {
            "samples": converted,
            "role": "CONVERSION_REVIEW_ONLY",
            "training_ready": False,
            "source_annotations_qc_final": True,
            "approved_training_samples": 0,
        },
    )
    spec_path = workspace / "annotation_projects/qgis_spec.json"
    dump(
        spec_path,
        {
            "output_root": str(workspace),
            "tiles": display_tiles,
            "store": str(store),
            "taxonomy": classes,
            "reference_layers": refs,
        },
    )
    subprocess.run(
        [
            "C:/Program Files/QGIS 3.44.7/bin/python-qgis.bat",
            str(root / "scripts/create_qgis_annotation_project.py"),
            str(spec_path),
        ],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        check=True,
    )
    changed = [p for p, h in source_hashes.items() if sha256(Path(p)) != h]
    if changed:
        raise ValueError(f"Changed protected inputs: {changed}")
    summary = {
        "source_annotations_qc_final": True,
        "source_approval_basis": attestation,
        "derived_draft_paint_areas": len(drafts),
        "conversion_review_pairs": len(converted),
        "approved_training_pairs": 0,
        "mask_instance_counts_by_visual_class": dict(
            Counter(r["visual_id"] for r in source_records)
        ),
        "exclusions": excluded,
        "offset_proxies": offsets,
        "qgis_project": str(workspace / "annotation_projects/road_marking_annotation.qgs"),
        "annotation_store": str(store),
        "protected_hash_mismatches": len(changed),
        "label_state_pixels": {r["tile_id"]: r["label_state_pixels"] for r in converted},
        "next_action": (
            "Review derived outlines and complete-region background on full-point images; "
            "reject missing observations from stock training"
        ),
    }
    dump(output / "metadata/annotation_conversion_report.json", summary)
    dump(
        report_dir / "annotation_conversion_latest.json",
        {"report": str(output / "metadata/annotation_conversion_report.json")},
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("derived_draft_paint_areas", "conversion_review_pairs", "qgis_project")
            },
            indent=2,
        )
    )
    return summary


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[1])
