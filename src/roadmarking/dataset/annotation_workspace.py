"""Prepare a small real-data annotation workspace, then stop at the human gate."""

import inspect
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import psutil
import rasterio
import yaml
from PIL import Image
from shapely.geometry import box

from roadmarking.dataset.annotation_store import (
    CHECKLIST,
    FIELDS,
    LAYERS,
    STATUSES,
    VERSION,
    create_store,
)
from roadmarking.dataset.annotation_validation import validate_store
from roadmarking.dataset.labels import LabelFeature
from roadmarking.dataset.pipeline import converter_selftest
from roadmarking.dataset.verified_export import export_dataset
from roadmarking.dataset.verified_masks import processor_targets, reviewed_masks
from roadmarking.rasterization.pipeline import MemoryMonitor, dump, sha256


class WorkspaceMemoryMonitor(MemoryMonitor):
    """Sample aggregate RSS of the parent and its QGIS child processes."""

    def sample(self):
        parent = psutil.Process()
        while not self.stop.is_set():
            processes = [parent, *parent.children(recursive=True)]
            total = 0
            for process in processes:
                try:
                    total += process.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            self.peak = max(self.peak, total)
            self.stop.wait(0.1)


def taxonomy(root):
    raw = yaml.safe_load((root / "configs/phase4_taxonomy.yaml").read_text(encoding="utf-8"))
    descriptions = {
        "longitudinal_solid": (
            "Visible continuous painted stripe; clipped dash/solid boundaries need "
            "context review"
        ),
        "longitudinal_dash": (
            "One physical dash, never bridge its unpainted gap; group related " "dashes"
        ),
        "arrow": (
            "Actual painted arrow outline; direction may require later " "catalog/context mapping"
        ),
        "detour_arrow": (
            "Reserved existing class; visually overlapping arrow definitions need "
            "catalog adjudication"
        ),
        "diagonal_stripe": "One painted hatched stripe, grouped with its hatch region",
        "give_way": "Reserved existing visual category; exact catalog outline remains unconfirmed",
        "colour_band": "Single-channel intensity cannot verify original paint colour",
        "safety_marking": (
            "Reserved existing category; exact visual definition requires client " "catalog"
        ),
        "character": "One painted letter/digit preserving holes, grouped into an inscription",
    }
    classes = []
    for c in raw["visual_classes"]:
        codes = [
            code
            for code, entry in raw["other_marks"]["codes"].items()
            if entry["visual"] == c["name"]
        ]
        classes.append(
            {
                **c,
                "training_class_id": c["id"] - 1,
                "description": descriptions[c["name"]],
                "annotation_geometry": (
                    "painted-area polygon preferred; verified-native-width line optional"
                    if c["id"] <= 2
                    else "painted-area polygon"
                ),
                "appearance": descriptions[c["name"]],
                "related_client_codes": codes,
                "client_mapping_status": "TO_CONFIRM_FROM_MISSING_AUTHORITATIVE_CATALOG",
                "enabled_for_annotation": c["id"] in (1, 2),
                "grouping": (
                    "one feature/global instance per physical dash, stripe or character; "
                    "group_id links related features"
                ),
                "ambiguity": (
                    "Human context review required; do not infer functional GIS designation"
                    " from intensity alone"
                ),
                "review_requirements": list(CHECKLIST),
            }
        )
    return {
        "version": "phase4-verified-taxonomy-v1",
        "legacy_version": raw["version"],
        "visual_classes": classes,
        "annotation_visual_ids_unchanged": True,
        "training_id_mapping": (
            "explicit training_class_id = stored visual_class_id - 1; legacy 0 "
            "background is not an object target"
        ),
        "active_pilot_scope": [1, 2],
        "catalog_status": "NOT_FOUND_IN_PROJECT",
        "catalog_code_hints_only": raw["other_marks"]["codes"],
        "longitudinal_client_reference": raw["longitudinal"]["designations"],
        "scope_status": "No class is trainable until verified examples and evaluation exist",
    }


def independent_groups(tiles):
    groups = []
    for tile in tiles:
        geom = box(*tile["bounds"])
        touches = [g for g in groups if any(geom.intersects(box(*t["bounds"])) for t in g)]
        combined = [tile]
        for group in touches:
            combined.extend(group)
            groups.remove(group)
        groups.append(combined)
    return [[t["tile_id"] for t in group] for group in groups]


def ignore_compatibility():
    """Demonstrate the loss limitation using fixed tensors, not a model forward pass."""
    import torch
    import transformers
    import transformers.models.mask2former.modeling_mask2former as model
    from affine import Affine

    affine = Affine(0.05, 0, 0, 0, -0.05, 1.6)
    observed = np.ones((32, 32), bool)
    feat = LabelFeature(1, "SYNTHETIC_ONLY", 1, "polygon", box(0.4, 0.4, 0.55, 1.2))
    labels, _ = reviewed_masks(affine, observed.shape, observed, observed, [feat])
    rgb = np.zeros((32, 32, 3), np.uint8)
    encoded = processor_targets(rgb, labels, inspection_only=True)
    target = encoded["mask_labels"][0][0]
    logits = torch.zeros_like(target).flatten()[None]
    changed = logits.clone()
    changed[0, 0] = 10  # an UNKNOWN source pixel, converted to zero in this binary target
    baseline = float(model.sigmoid_cross_entropy_loss(logits, target.flatten()[None], 1))
    altered = float(model.sigmoid_cross_entropy_loss(changed, target.flatten()[None], 1))
    rejected = False
    try:
        processor_targets(rgb, labels)
    except ValueError:
        rejected = True
    result = {
        "transformers_version": transformers.__version__,
        "processor_class_labels": encoded["class_labels"][0].tolist(),
        "pixel_values_shape": list(encoded["pixel_values"].shape),
        "unknown_pixel_target": float(target[0, 0]),
        "fixed_tensor_loss_before": baseline,
        "fixed_tensor_loss_after_unknown_change": altered,
        "unknown_changes_stock_loss": altered != baseline,
        "partial_sample_training_gate_rejects": rejected,
        "loss_masks_signature": str(inspect.signature(model.Mask2FormerLoss.loss_masks)),
        "matcher_signature": str(inspect.signature(model.Mask2FormerHungarianMatcher.forward)),
        "loss_implementation_file": inspect.getfile(model),
        "loss_implementation_sha256": sha256(Path(inspect.getfile(model))),
        "model_weights_loaded": False,
        "model_forward_or_training_run": False,
        "policy": (
            "Export partial masks for review only. Stock targets require every "
            "pixel to be verified paint/background. Otherwise exclude from training"
            " until matcher, point sampling, BCE/Dice and auxiliary losses have a "
            "validated validity mask strategy."
        ),
    }
    assert result["unknown_changes_stock_loss"] and rejected
    return result


def prepare(root, *, qgis=True):
    root = Path(root).resolve()
    started = time.perf_counter()
    latest = json.loads((root / "outputs/reports/phase3_3/latest.json").read_text())
    source = Path(latest["output_root"])
    tiles = json.loads((source / "model_tiles_1024/tile_manifest.json").read_text())
    completed = json.loads((source / "metadata/completion.json").read_text())
    protected = dict(completed["artifacts"])
    for base in (
        root / "outputs/reports/phase3_3",
        root / "outputs/reports/phase4",
        root / "data/raw/annotations",
        root / "data/processed/phase4",
    ):
        protected.update({str(p): sha256(p) for p in base.rglob("*") if p.is_file()})
    for path, digest in protected.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Existing input hash mismatch: {path}")
    run_id = time.strftime("%Y%m%dT%H%M%S") + "_annotation_v2"
    output = root / "data/processed/phase4_verified" / run_id
    report_dir = root / "outputs/reports/phase4_verified"
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
    report_dir.mkdir(parents=True, exist_ok=True)
    with WorkspaceMemoryMonitor() as monitor:
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
                "width_units": "native coordinate units, never assumed metres",
            },
        )
        checklist = dict.fromkeys(CHECKLIST, False)
        dump(output / "metadata/human_review_checklist_TEMPLATE.json", checklist)
        audit = []
        for tile in tiles:
            with rasterio.open(tile["layers"]["normalized"]) as ds:
                if ds.crs is not None or ds.shape != (1024, 1024):
                    raise ValueError("Unexpected source geometry")
                observed = rasterio.open(tile["masks"]["observed"])
                valid = rasterio.open(tile["masks"]["valid"])
                obs, val = observed.read(1).astype(bool), valid.read(1).astype(bool)
                observed.close()
                valid.close()
            with Image.open(tile["images"]["rgb"]) as im:
                rgb = np.asarray(im)
            if (
                rgb.shape != (1024, 1024, 3)
                or not np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
                or not np.array_equal(rgb[:, :, 1], rgb[:, :, 2])
            ):
                raise ValueError("Unclean RGB source")
            audit.append(
                {
                    "tile_id": tile["tile_id"],
                    "source": tile["source"],
                    "observed_pixels": int(obs.sum()),
                    "unknown_pixels": int(obs.sum()),
                    "nodata_pixels": int((val & ~obs).sum()),
                    "padding_pixels": int((~val).sum()),
                    "verified_background_pixels": 0,
                    "verified_paint_pixels": 0,
                    "review_status": "UNREVIEWED",
                    "negative_example_status": "NOT_VERIFIED",
                    "source_layers": tile["layers"],
                    "image": tile["images"]["rgb"],
                }
            )
        dump(output / "images/source_references.json", tiles)
        dump(
            output / "metadata/region_review_plan.json",
            {
                "regions": audit,
                "independent_groups": independent_groups(tiles),
                "target_regions": "20-30 independently useful regions when available",
                "limitation": (
                    "Only nine Phase 3.3 inputs; overlapping tiles form six independent "
                    "footprint groups. No duplicated crops invented to reach the target."
                ),
                "review_targets": [
                    "solid",
                    "dash",
                    "edge",
                    "curve/context",
                    "faded-paint search",
                    "bright nonpaint search",
                    "sparse observations",
                    "old extraction boundary",
                    "crossing/overpass context",
                ],
                "additional_data_needed": (
                    "More independent regions and confirmed examples of unlocated catalog "
                    "classes"
                ),
            },
        )
        demo = converter_selftest(output / "quality_control/SYNTHETIC_ONLY")
        dump(
            output / "quality_control/SYNTHETIC_ONLY/selftest.json",
            {"role": "SYNTHETIC_NOT_VERIFIED_NOT_TRAINING_DATA", **demo},
        )
        compatibility = ignore_compatibility()
        dump(output / "metadata/model_compatibility.json", compatibility)
        exported = export_dataset(store, output / "metadata/empty_export_gate")
        dump(output / "metadata/dataset_manifest.json", exported)
        dump(output / "metadata/instance_id_mapping.json", {})
        dump(output / "splits/spatial_split_report.json", exported["splits"])
        # QGIS/GDAL may cache display statistics into files opened read-only.
        # Only small owned display artifacts, never earlier phase files, enter QGIS.
        display_tiles = []
        for tile in tiles:
            display = json.loads(json.dumps(tile))
            directory = output / "annotation_projects/rasters" / tile["tile_id"]
            directory.mkdir(parents=True)
            for filename, section, key in (
                (tile["images"]["grayscale"], "images", "grayscale"),
                (tile["layers"]["normalized"], "layers", "normalized"),
                (tile["masks"]["observed"], "masks", "observed"),
                (tile["masks"]["valid"], "masks", "valid"),
            ):
                original = Path(filename)
                target = directory / original.name
                if section == "layers" and key == "normalized":
                    # A single-source VRT can delegate statistics to its TIFF source.
                    # Materialize only these nine small native crops for safe GUI use.
                    target = directory / "normalized_annotation_copy.tif"
                    from roadmarking.rasterization.pipeline import write_tif

                    with rasterio.open(original) as ds:
                        pixels, transform = ds.read(1), ds.transform
                    write_tif(target, pixels, transform, None)
                else:
                    shutil.copyfile(original, target)
                if original.suffix == ".png":
                    shutil.copyfile(original.with_suffix(".pgw"), target.with_suffix(".pgw"))
                display[section][key] = str(target)
            display_tiles.append(display)
        spec = {
            "output_root": str(output),
            "tiles": display_tiles,
            "store": str(store),
            "taxonomy": classes,
        }
        spec_path = output / "annotation_projects/qgis_spec.json"
        dump(spec_path, spec)
        if qgis:
            launcher = Path("C:/Program Files/QGIS 3.44.7/bin/python-qgis.bat")
            env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
            subprocess.run(
                [
                    str(launcher),
                    str(root / "scripts/create_qgis_annotation_project.py"),
                    str(spec_path),
                ],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
        validation = validate_store(store)
        qgis_validation = (
            json.loads((output / "metadata/qgis_validation.json").read_text()) if qgis else {}
        )
        dump(output / "metadata/review_status_report.json", validation)
        dump(
            output / "metadata/image_mask_alignment_report.json",
            {
                "real_image_mask_pairs": 0,
                "real_annotation_alignment_verified": False,
                "source_grid_checks": qgis_validation,
                "synthetic_converter_checks": demo["checks"],
            },
        )
        shutil.copyfile(
            root / "docs/phase4_verified_annotations.md",
            output / "annotation_projects/ANNOTATION_GUIDELINES.md",
        )
        class_coverage = [
            {
                "visual_class_id": c["id"],
                "name": c["name"],
                "training_class_id": c["training_class_id"],
                "visible_evidence": (
                    "SUGGESTED_BY_INTENSITY_REVIEW_NOT_ANNOTATED"
                    if c["id"] in (1, 2)
                    else "NOT_LOCATED_IN_CURRENT_PILOT"
                ),
                "verified_instances": 0,
                "needs_more_labels": True,
                "additional_data_likely_needed": c["id"] not in (1, 2),
            }
            for c in classes["visual_classes"]
        ]
        unchanged = all(sha256(Path(path)) == digest for path, digest in protected.items())
        if not unchanged:
            raise ValueError("Protected prior output changed")
        report = {
            "status": "ANNOTATION REQUIRED",
            "output_root": str(output),
            "run_id": run_id,
            "annotation_store": str(store),
            "qgis_project": str(output / "annotation_projects/road_marking_annotation.qgs"),
            "input_regions": audit,
            "independent_region_groups": independent_groups(tiles),
            "candidate_regions": len(tiles),
            "annotations": validation,
            "real_image_mask_pairs": 0,
            "class_coverage": class_coverage,
            "quality_metrics": {
                "annotated_regions": 0,
                "reviewed_regions": 0,
                "verified_regions": 0,
                "rejected_regions": 0,
                "uncertain_regions": 0,
                "verified_marking_instances": 0,
                "verified_background_pixels": 0,
                "polygon_annotations": 0,
                "polyline_derived_masks": 0,
                "average_instances_per_annotated_image": None,
                "cross_tile_verified_identity_status": "NOT_MEASURABLE_NO_VERIFIED_FEATURES",
            },
            "taxonomy": classes,
            "model_compatibility": compatibility,
            "synthetic_converter_demo": demo,
            "synthetic_is_real_ground_truth": False,
            "spatial_splits": exported["splits"],
            "training_ready": False,
            "mismatched_client_annotations_used_as_ground_truth": False,
            "catalog": {"status": "NOT_FOUND_IN_PROJECT", "exact_code_meanings_confirmed": False},
            "preparation_seconds": time.perf_counter() - started,
            "peak_rss_bytes": monitor.peak,
            "memory_scope": "sampled aggregate RSS parent plus recursive QGIS child processes",
            "protected_hashes": protected,
            "protected_unchanged": unchanged,
            "validation": {
                "pytest": "PENDING",
                "health": "PENDING",
                "ruff": "PENDING",
                "black": "PENDING",
            },
            "blockers": [
                "Human painted-area annotations and attributed verification required",
                "No verified background/completeness regions",
                (
                    "Stock mask loss cannot ignore unknown/NoData/padding; partial samples "
                    "excluded pending validated loss/matcher strategy"
                ),
                "Native CRS and physical units unconfirmed",
                "Authoritative client catalog unavailable; mappings remain hints",
                (
                    "Only six independent footprint groups; no defensible evaluation split "
                    "or coverage of all classes"
                ),
            ],
        }
    dump(output / "metadata/protected_hashes.json", protected)
    dump(output / "metadata/class_distribution_report.json", class_coverage)
    dump(report_dir / "latest.json", {"run_id": run_id, "output_root": str(output)})
    dump(report_dir / "report.json", report)
    write_report(report, report_dir / "report.md")
    return report


def write_report(report, path):
    output = Path(report["output_root"])
    text = [
        "# Phase 4 verified annotation workflow",
        "",
        f"Phase 5 readiness: **{report['status']}**.",
        "",
        "## A. Input readiness",
        "",
        (
            "Nine original Phase 3.3 1024/256 tiles from both intact tracks; exact "
            "0.05 native-unit affine and grayscale RGB. Native units/CRS "
            "unconfirmed. The four spatially mismatched client GeoPackages are "
            "excluded from ground truth. No authoritative catalog PDF/document was "
            "found in the project."
        ),
        "",
        "## B. Annotation workspace",
        "",
        f"Open `{report['qgis_project']}` in QGIS. "
        f"Editable store: `{report['annotation_store']}`. "
        "See docs/phase4_verified_annotations.md for drawing and human review commands. "
        "Paint, centerlines and review areas are distinct from planning footprints. "
        "Native coordinates are unchanged; no EPSG is assigned. "
        "Audit tables retain geometry history.",
        "",
        "## C. Visual class taxonomy",
        "",
        (
            "Existing visual IDs 1–9 retained; explicit model target IDs 0–8 "
            "are stored separately. Only existing solid/dash categories enabled for"
            " this pilot; all other reserved classes remain unlocated and disabled."
            " Client code hints retain their TO_CONFIRM status. No new absent class"
            " was invented."
        ),
        "",
        "## D. Annotation progress",
        "",
        f"{report['candidate_regions']} candidate tiles, "
        f"{len(report['independent_region_groups'])} independent footprint groups; "
        "zero annotated/reviewed/verified/rejected/uncertain real features and zero verified "
        "background regions. Twenty to thirty independent regions are a future planning target, "
        "not met by duplicating crops. All observed pixels remain unknown.",
        "",
        "## E. Mask conversion",
        "",
        (
            "Zero real image-mask pairs generated: no genuinely reviewed "
            "annotations are available. Existing converter and geometry utilities "
            "reused; isolated SYNTHETIC_ONLY converter examples preserve holes, "
            "narrow lines, separate dashes and cross-tile identity. Synthetic "
            "examples are not verified labels. Export tools only accept matching "
            "attributed human review events; geometry validation alone cannot "
            "approve anything."
        ),
        "",
        "## F. Training dataset quality",
        "",
        (
            "Per-region observed/unknown/NoData/padding counts and image references"
            " are in JSON. These are tile appearances and must not be summed as "
            "unique observations across overlaps. Separate canonical validity codes"
            " distinguish paint, verified background, unknown, missing returns, "
            "padding, boundary/conflict and uncertainty. Conversion retains polygon"
            " holes and original geometries. A centerline requires explicit native-"
            "unit width verification plus paint-mask review. Colored QA images are "
            "separate from clean model inputs."
        ),
        "",
        "## G. Dataset splits",
        "",
        (
            "No train/validation/test split assigned without verified data. "
            "Existing spatial splitter reused with overlap and global-ID guards "
            "across sources, no long-instance exception, plus explicit negative-"
            "example planning tokens. Small/insufficient evaluations stay "
            "provisional. Do not use the held-out test region to tune annotations "
            "or models."
        ),
        "",
        "## H. Model compatibility",
        "",
        (
            "Installed Transformers target API checked locally without weights. "
            "do_resize=False, uint8 grayscale RGB rescaled once by 1/255 and "
            "standard processor normalization. Background/ignored map value 0 is "
            "removed from object targets, but unknown pixels become zero in each "
            "remaining binary instance target. Fixed-tensor BCE evidence confirms "
            "changing an UNKNOWN location changes stock loss. Processor pixel_mask "
            "does not solve this. Partial review/NoData/padding samples are "
            "forbidden for stock training; a future validated "
            "matcher/point/BCE/Dice/auxiliary loss masking design is required. No "
            "head changes, training or inference performed."
        ),
        "",
        "## I. Execution performance",
        "",
        f"Preparation {report['preparation_seconds']:.3f} seconds; "
        f"sampled aggregate parent/QGIS RSS {report['peak_rss_bytes']/2**20:.1f} MiB. "
        "Includes workspace, validation, synthetic diagnostics and QGIS project creation, "
        "not model processing.",
        "",
        "## J. Automated tests",
        "",
        "```json",
        json.dumps({k: v for k, v in report["validation"].items() if k != "healthcheck"}, indent=2),
        "```",
        "",
        "## K. Missing training classes",
        "",
        (
            "Every class has zero verified examples. Solid/dash evidence is visible"
            " but not human-labelled. Arrows, detour arrows, diagonal/give-"
            "way/safety marks, characters and colour bands have not been "
            "established in this pilot; additional data/catalog review and manual "
            "examples are required. Grayscale cannot establish paint colour."
        ),
        "",
        "## L. Remaining blockers",
        "",
        *[f"- {b}" for b in report["blockers"]],
        "",
        "## M. Phase 5 readiness",
        "",
        (
            "ANNOTATION REQUIRED. Usable QGIS workspace and gated export tools are "
            "delivered. This is not a completed training dataset. Human review, "
            "class/coordinate confirmation, loss-validity handling and defensible "
            "geographic evaluation remain necessary."
        ),
        "",
        f"Workspace: `{output}`",
        "",
    ]
    Path(path).write_text("\n".join(text), encoding="utf-8")
