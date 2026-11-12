"""Prepare a human-review workspace from the corrected-source alignment pilot.

The bounded point samples are diagnostic imagery, never production-ready tiles.
Original client shapes are immutable reference candidates, not approved paint masks.
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import yaml
from PIL import Image
from shapely.validation import explain_validity

from roadmarking.dataset.annotation_store import (
    CHECKLIST,
    FIELDS,
    LAYERS,
    STATUSES,
    VERSION,
    create_store,
)
from roadmarking.dataset.annotation_validation import validate_store
from roadmarking.dataset.annotation_workspace import (
    WorkspaceMemoryMonitor,
    ignore_compatibility,
    taxonomy,
)
from roadmarking.dataset.pipeline import converter_selftest
from roadmarking.dataset.verified_export import export_dataset
from roadmarking.rasterization.pipeline import dump, sha256, write_tif


def main():
    root = Path(__file__).resolve().parents[1]
    started = time.perf_counter()
    alignment = json.loads(
        (root / "outputs/reports/phase4_corrected_source_alignment/report.json").read_text()
    )
    audit = json.loads(Path(alignment["header_audit"]).read_text())
    run_id = time.strftime("%Y%m%dT%H%M%S") + "_corrected_source_review"
    output = root / "data/processed/phase4_verified" / run_id
    reports = root / "outputs/reports/phase4_verified"
    history = reports / "history" / run_id
    history.mkdir(parents=True)
    for name in ("report.json", "report.md", "latest.json"):
        if (reports / name).exists():
            shutil.copyfile(reports / name, history / name)
    for folder in (
        "annotation_projects",
        "annotations_draft",
        "annotations_verified",
        "images",
        "instance_masks",
        "validity_masks",
        "metadata",
        "splits",
        "quality_control",
    ):
        (output / folder).mkdir(parents=True, exist_ok=False)
    with WorkspaceMemoryMonitor() as monitor:
        tiles, counts = [], []
        protected = {}
        for entry in alignment["entries"]:
            tid = "corrected_" + entry["group"]
            directory = output / "annotation_projects/rasters" / tid
            directory.mkdir(parents=True)
            layers = {}
            with rasterio.open(entry["rasters"]["normalized"]) as source:
                transform, bounds = source.transform, list(source.bounds)
            for name, original in entry["rasters"].items():
                protected[original] = sha256(Path(original))
                target = directory / (name + ".tif")
                with rasterio.open(original) as source:
                    pixels = source.read(1)
                write_tif(target, pixels, transform, None)
                layers[name] = str(target)
            with rasterio.open(layers["observed"]) as source:
                observed = source.read(1).astype(bool)
            for name, array in (
                ("valid", np.ones_like(observed)),
                ("filled_mask", np.zeros_like(observed)),
            ):
                target = directory / (name + ".tif")
                write_tif(target, array.astype("uint8"), transform, None)
                layers[name] = str(target)
            image = directory / "grayscale.png"
            protected[entry["image"]] = sha256(Path(entry["image"]))
            shutil.copyfile(entry["image"], image)
            shutil.copyfile(Path(entry["image"]).with_suffix(".pgw"), image.with_suffix(".pgw"))
            rgb = directory / "rgb.png"
            with Image.open(image) as source:
                source.convert("RGB").save(rgb)
            tiles.append(
                {
                    "tile_id": tid,
                    "source": entry["group"],
                    "bounds": bounds,
                    "affine": list(transform)[:6],
                    "width": 1024,
                    "height": 1024,
                    "resolution": 0.05,
                    "layers": layers,
                    "images": {"grayscale": str(image), "rgb": str(rgb)},
                    "masks": {"observed": layers["observed"], "valid": layers["valid"]},
                    "raw_source": entry["source"],
                    "point_windows": entry["point_windows"],
                    "role": "BOUNDED_SAMPLE_REVIEW_ONLY_NOT_PRODUCTION_TILE",
                }
            )
            counts.append(
                {
                    "tile_id": tid,
                    "observed_unknown": int(observed.sum()),
                    "nodata": int((~observed).sum()),
                    "padding": 0,
                    "verified_paint": 0,
                    "verified_background": 0,
                }
            )
        references, invalid, reuse = [], [], []
        for number, layer in enumerate(audit["annotation_layers"]):
            original = Path(layer["readable_inspection_copy"])
            if sha256(original) != layer["sha256"]:
                raise ValueError("Source annotation hash changed")
            protected[str(original)] = layer["sha256"]
            target = output / "annotation_projects" / f"client_original_{number}.gpkg"
            shutil.copyfile(original, target)
            references.append(
                {
                    "path": str(target),
                    "layer": layer["layer"],
                    "name": f"Original {number}: {original.name} - UNREVIEWED",
                }
            )
            frame = gpd.read_file(original, layer=layer["layer"], fid_as_index=True)
            for fid, row in frame[~frame.geometry.is_valid].iterrows():
                invalid.append(
                    {
                        "source": str(original),
                        "layer": layer["layer"],
                        "fid": int(fid),
                        "reason": explain_validity(row.geometry),
                    }
                )
            reuse.append(
                {
                    "original": layer["path"],
                    "local": str(original),
                    "sha256": layer["sha256"],
                    "layer": layer["layer"],
                    "features": len(frame),
                    "valid_features": layer["valid_features"],
                    "coordinate_operation": "IDENTITY_NO_TRANSFORM",
                    "pixel_area_review_status": "PENDING_HUMAN_REVIEW",
                }
            )
        classes = taxonomy(root)
        store = create_store(
            output / "annotations_draft/road_marking_annotations.gpkg", tiles, classes
        )
        dump(output / "metadata/class_taxonomy.json", classes)
        dump(
            output / "metadata/annotation_schema.json",
            {
                "version": VERSION,
                "layers": LAYERS,
                "fields": list(FIELDS),
                "statuses": list(STATUSES),
                "review_checklist": list(CHECKLIST),
                "CRS": "CRS_UNCONFIRMED",
                "width_units": "native units",
            },
        )
        dump(
            output / "metadata/class_distribution_report.json",
            {
                "verified_instances_by_visual_id": {
                    str(c["id"]): 0 for c in classes["visual_classes"]
                },
                "visibly_present_in_sample": ["longitudinal_solid", "longitudinal_dash"],
                "candidate_class_metadata": [
                    item["class_attributes"] for item in audit["annotation_layers"]
                ],
                "catalog_confirmed": False,
                "other_visual_classes_located_in_sample": False,
            },
        )
        dump(
            output / "metadata/review_status_report.json",
            {
                "original_client_candidates": 16521,
                "human_verified_pixel_area_instances": 0,
                "draft_store_features": 0,
                "rejected": 0,
                "uncertain": 0,
                "unreviewed_regions": len(tiles),
                "verified_regions": 0,
            },
        )
        dump(
            output / "metadata/image_mask_alignment_report.json",
            {
                "native_affine": True,
                "transform_applied": False,
                "production_masks_generated": 0,
                "source_alignment_report": alignment,
            },
        )
        demo = converter_selftest(output / "quality_control/SYNTHETIC_ONLY")
        dump(
            output / "quality_control/SYNTHETIC_ONLY/selftest.json",
            {
                "role": "SYNTHETIC_NOT_VERIFIED_NOT_TRAINING_DATA",
                **demo,
            },
        )
        dump(
            output / "metadata/human_review_checklist_TEMPLATE.json",
            dict.fromkeys(CHECKLIST, False),
        )
        dump(output / "metadata/source_references.json", tiles)
        dump(
            output / "metadata/region_review_plan.json",
            {
                "regions": counts,
                "candidate_regions": 3,
                "planning_target": "20-30 independent regions, not yet available as complete tiles",
                "targets": [
                    "solid",
                    "dash",
                    "edge",
                    "faded paint",
                    "bright nonpaint",
                    "sparse returns",
                    "curves",
                    "symbols only if actually visible",
                ],
                "production_requirement": (
                    "Complete-file QC and calibrated raster generation required"
                ),
            },
        )
        dump(output / "metadata/model_compatibility.json", ignore_compatibility())
        dump(output / "metadata/annotation_validation.json", validate_store(store))
        exported = export_dataset(store, output / "metadata/export_gate")
        dump(output / "metadata/dataset_manifest.json", exported)
        dump(output / "metadata/instance_id_mapping.json", {})
        dump(output / "splits/spatial_split_report.json", exported["splits"])
        spec = {
            "output_root": str(output),
            "tiles": tiles,
            "store": str(store),
            "taxonomy": classes,
            "reference_layers": references,
        }
        spec_path = output / "annotation_projects/qgis_spec.json"
        dump(spec_path, spec)
        subprocess.run(
            [
                "C:/Program Files/QGIS 3.44.7/bin/python-qgis.bat",
                str(root / "scripts/create_qgis_annotation_project.py"),
                str(spec_path),
            ],
            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
            check=True,
            capture_output=True,
            text=True,
        )
        changed = [path for path, digest in protected.items() if sha256(Path(path)) != digest]
        if changed:
            raise ValueError(f"Protected inputs changed: {changed}")
    reuse_manifest = {
        "raw_root": alignment["raw_root"],
        "inputs": reuse,
        "invalid_geometry": invalid,
        "transform_applied": False,
        "verified_pixel_area_instances": 0,
        "forced_fit_copies_used": False,
    }
    dump(
        root / "outputs/reports/phase4_corrected_source_alignment/annotation_reuse_manifest.json",
        reuse_manifest,
    )
    report = {
        "status": "ANNOTATION REQUIRED",
        "run_id": run_id,
        "output_root": str(output),
        "qgis_project": str(output / "annotation_projects/road_marking_annotation.qgs"),
        "raw_root": alignment["raw_root"],
        "annotation_root": alignment["annotation_root"],
        "source_relationship": "IDENTITY_NATIVE_XY_NO_TRANSFORM",
        "source_files_with_readable_headers": 173,
        "candidate_annotations": 16521,
        "annotation_store": str(store),
        "annotations": validate_store(store),
        "input_regions": tiles,
        "training_ready": False,
        "mismatched_client_annotations_used_as_ground_truth": False,
        "protected_unchanged": not changed,
        "protected_hashes": protected,
        "valid_annotations_intersecting_headers": 16520,
        "invalid_geometry": invalid,
        "alignment_report": str(
            root / "outputs/reports/phase4_corrected_source_alignment/report.json"
        ),
        "region_pixel_categories": counts,
        "verified_instances": 0,
        "real_image_mask_pairs": exported["real_image_mask_pairs"],
        "production_training_tiles": 0,
        "training_started": False,
        "inference_started": False,
        "weights_downloaded": False,
        "forced_fit_used": False,
        "elapsed_seconds": time.perf_counter() - started,
        "sampled_peak_rss_bytes": monitor.peak,
        "protected_input_hash_mismatches": len(changed),
        "qgis_validation": json.loads((output / "metadata/qgis_validation.json").read_text()),
        "blockers": [
            "Human painted-area/class/instance approval not recorded",
            "Centerlines require verified native widths and buffered-area review",
            "One invalid source polygon requires adjudication",
            "Complete-file integrity and production raster coverage not established",
            "Missing authoritative class catalog and unsupported visual classes",
            "Stock Mask2Former loss does not ignore partial-review pixels",
            "Client CRS confirmation and independent spatial evaluation pending",
        ],
    }
    dump(reports / "report.json", report)
    dump(reports / "latest.json", {"run_id": run_id, "output_root": str(output)})
    sections = {
        "A. Input readiness": "Corrected raw share matches original client XY. All 16,520 valid "
        "features intersect raw footprints. Three real point-window overlays support placement. "
        "No shift, rotation or reprojection was applied. Earlier A6 forced-fit copies excluded.",
        "B. Annotation workspace": f"Open `{report['qgis_project']}`. Three native 1024-pixel "
        "review samples, four read-only original client layers, editable audited schema and "
        "three bookmarks. All display inputs are workspace-owned. Samples have partial point "
        "coverage and diagnostic normalization; they are not complete production tiles.",
        "C. Visual class taxonomy": "Stable visual IDs 1 solid and 2 dash enabled for this "
        "pilot; training IDs 0 and 1. IDs 3-9 reserved. Catalog mappings remain provisional.",
        "D. Annotation progress": "16,521 original candidates; 16,520 valid, one invalid. "
        "Zero human pixel-area approvals recorded. Original geometry is available without "
        "redigitization for review and copying into draft layers; manual authorship alone "
        "does not verify segmentation width, completeness or review provenance.",
        "E. Mask conversion": "Zero real approved pairs. Existing converter and synthetic "
        "regression tests retained; no synthetic labels counted as client ground truth.",
        "F. Training dataset quality": "All observed pixels remain unknown until reviewed; "
        "missing returns are NoData. No verified background, padding or paint asserted. "
        "See region_pixel_categories in report.json for counts.",
        "G. Dataset splits": "No split assigned without approved samples. Existing geographic "
        "overlap and global-identity guards retained. Three crops do not prove "
        "survey independence.",
        "H. Model compatibility": "Installed processor checked without weights or model forward. "
        "Gray replicated RGB, no resize, one rescale. Partial-review masks rejected for stock "
        "training: pixel_mask does not mask Hungarian matching or BCE/Dice losses.",
        "I. Execution performance": f"{report['elapsed_seconds']:.2f} seconds; sampled aggregate "
        f"peak RSS {monitor.peak / 2**20:.1f} MiB (includes QGIS child).",
        "J. Automated tests": "See validation_results.json for the current regression, lint "
        "and environment-check results; QGIS project saved and reopened with all layers valid.",
        "K. Missing training classes": "No class has verified training examples yet. Original "
        "COD_RST/designation frequencies are candidate metadata, not verified visual coverage. "
        "Arrows, characters and other catalog categories need explicit visual/catalog review.",
        "L. Remaining blockers": "\n".join("- " + item for item in report["blockers"]),
        "M. Phase 5 readiness": "ANNOTATION REQUIRED. No training, inference, weights download "
        "or final 3D digitization performed. Older Phase 3.3 two-track workspace preserved "
        "in report history; it must not receive these spatially separate client annotations.",
    }
    text = (
        "# Phase 4 — corrected-source review workspace\n\n"
        + "\n\n".join(f"## {title}\n\n{body}" for title, body in sections.items())
        + "\n"
    )
    (reports / "report.md").write_text(text, encoding="utf-8")
    (output / "README.md").write_text(text, encoding="utf-8")
    readiness = root / "outputs/reports/training_readiness"
    readiness.mkdir(parents=True, exist_ok=True)
    old = readiness / "history" / run_id
    old.mkdir(parents=True)
    for name in ("report.json", "report.md"):
        if (readiness / name).exists():
            shutil.copyfile(readiness / name, old / name)
    dump(
        readiness / "report.json",
        {
            "allow_training": False,
            "status": report["status"],
            "current_report": str(reports / "report.json"),
            "blockers": report["blockers"],
        },
    )
    (readiness / "report.md").write_text(text, encoding="utf-8")
    config = root / "configs/phase4_network_sources.yaml"
    active = yaml.safe_load(config.read_text())
    assert active["candidate_raw_roots"] == [alignment["raw_root"]]
    print(report["qgis_project"])


if __name__ == "__main__":
    main()
