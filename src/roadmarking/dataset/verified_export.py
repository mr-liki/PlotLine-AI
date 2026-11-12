"""Versioned export from attributed human reviews; no proposals become ground truth."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from shapely.ops import unary_union

from roadmarking.dataset.annotation_store import metadata, read_annotations
from roadmarking.dataset.annotation_validation import validate_store
from roadmarking.dataset.geometry import tile_affine
from roadmarking.dataset.labels import LabelFeature
from roadmarking.dataset.qa import tile_overlay
from roadmarking.dataset.splits import spatial_split
from roadmarking.dataset.verified_masks import processor_targets, reviewed_masks
from roadmarking.dataset.writer import read_tile_inputs, write_sample
from roadmarking.rasterization.pipeline import dump, sha256, write_tif


def export_dataset(store, output):
    """Produces review masks even when stock loss cannot safely consume them."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    before = sha256(Path(store))
    validation = validate_store(store)
    if validation["invalid_features"] or validation["global_errors"]:
        raise ValueError("Fix invalid or stale VERIFIED features before export")
    meta = metadata(store)
    records = read_annotations(store)
    eligible = {r["annotation_id"] for r in validation["records"] if r["eligible"]}
    reviewed = [r for r in records if r["annotation_id"] in eligible]
    paint = [r for r in reviewed if r["layer"] != "review_areas"]
    global_ids = {
        gid: index + 1 for index, gid in enumerate(sorted(r["global_instance_id"] for r in paint))
    }
    samples, tile_instances, tile_classes = [], {}, {}
    source_protected = {}
    for tile in meta["tiles"]:
        source = tile["source"]
        features = [
            LabelFeature(
                global_ids[r["global_instance_id"]],
                r["global_instance_id"],
                int(r["visual_class_id"]),
                "line" if r["layer"] == "paint_centerlines" else "polygon",
                r["geometry"],
                r["marking_width"],
                r["group_id"],
            )
            for r in paint
            if r["source_track"] == source
        ]
        areas = [
            r
            for r in reviewed
            if r["source_track"] == source
            and r["layer"] == "review_areas"
            and r["purpose"] in {"COMPLETE_REGION", "BACKGROUND"}
        ]
        uncertain = [
            r["geometry"]
            for r in records
            if r["source_track"] == source
            and r["layer"] == "review_areas"
            and r["purpose"] == "UNCERTAIN"
            and r["review_status"] != "REJECTED"
        ]
        if not features and not areas:
            continue
        for path in (*tile["layers"].values(), *tile["masks"].values(), *tile["images"].values()):
            source_protected[path] = sha256(Path(path))
        inputs = read_tile_inputs(tile)
        with rasterio.open(tile["masks"]["valid"]) as ds:
            valid = ds.read(1).astype(bool)
        shape = (tile["height"], tile["width"])
        completeness = unary_union([r["geometry"] for r in areas]) if areas else None
        labels, categories = reviewed_masks(
            tile_affine(tile),
            shape,
            inputs["observed"],
            valid,
            features,
            completeness,
            unary_union(uncertain),
        )
        if not ((labels.semantic != 255) & valid).any():
            continue
        with Image.open(tile["images"]["rgb"]) as image:
            rgb = np.asarray(image)
        if not np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) or not np.array_equal(
            rgb[:, :, 1], rgb[:, :, 2]
        ):
            raise ValueError("Original image is not replicated grayscale")
        safe = bool((labels.semantic != 255).all())
        if safe:
            processor_targets(rgb, labels)
        record = write_sample(
            output,
            tile,
            inputs,
            labels,
            {
                "annotation_store_sha256": before,
                "rgb_image_reference": tile["images"]["rgb"],
                "rgb_sha256": sha256(Path(tile["images"]["rgb"])),
                "image_tif_role": "FULL_PRECISION_AUXILIARY_NOT_RGB_MODEL_INPUT",
                "stock_loss_eligible": safe,
                "review_provenance": [
                    {
                        k: r[k]
                        for k in (
                            "annotation_id",
                            "global_instance_id",
                            "reviewer_id",
                            "review_timestamp",
                        )
                    }
                    for r in reviewed
                    if r["source_track"] == source
                ],
                "training_target_mapping": {str(k): k - 1 for k in range(1, 10)},
            },
        )
        validity = output / "validity_masks" / tile["tile_id"]
        for name, array in {
            "categories": categories,
            "valid": valid,
            "observed": inputs["observed"],
            "ignore": labels.ignore,
        }.items():
            write_tif(validity / f"{name}.tif", array.astype(np.uint8), tile_affine(tile), None)
        record["validity_masks"] = {
            k: str((validity / f"{k}.tif").relative_to(output))
            for k in ("categories", "valid", "observed", "ignore")
        }
        record["validity_sha256"] = {
            key: sha256(output / relative) for key, relative in record["validity_masks"].items()
        }
        dump(output / "metadata/tiles" / f"{tile['tile_id']}.json", record)
        tile_overlay(
            output / "quality_control" / f"{tile['tile_id']}.png",
            tile,
            inputs["normalized"],
            [f.geometry for f in features],
            labels,
            f"HUMAN-REVIEWED annotations / {source} / {tile['tile_id']}",
        )
        samples.append(record)
        ids = {r["instance_uid"] for r in labels.instances if r["pixels"]}
        # A split-only token includes verified negative images in the existing splitter.
        tile_instances[tile["tile_id"]] = ids or {-len(samples)}
        tile_classes[tile["tile_id"]] = Counter(
            r["visual_id"] for r in labels.instances if r["pixels"]
        )
    split = {"status": "UNASSIGNED_INSUFFICIENT_VERIFIED_DATA", "assignment": {}}
    if len(samples) >= 3:
        chosen = [t for t in meta["tiles"] if t["tile_id"] in tile_instances]
        split = spatial_split(chosen, tile_instances, tile_classes, block_tiles=1, guard=25.6)
        split["negative_tokens_are_not_markings"] = True
        for summary in split["splits"].values():
            ids = (
                set().union(*(tile_instances[tid] for tid in summary["tile_ids"]))
                if summary["tile_ids"]
                else set()
            )
            summary["unique_instances"] = len({i for i in ids if i > 0})
        if split["insufficient_splits"]:
            split["status"] = "PILOT_ONLY_INSUFFICIENT_EVALUATION"
    manifest = {
        "status": (
            "ANNOTATION_REQUIRED" if not samples else "REVIEW_MASKS_EXPORTED_NOT_TRAINING_READY"
        ),
        "samples": samples,
        "real_image_mask_pairs": len(samples),
        "global_instance_ids": global_ids,
        "verified_instances_by_class": dict(Counter(r["visual_class_name"] for r in paint)),
        "annotation_validation": validation,
        "splits": split,
        "stock_loss_eligible_samples": sum(s["stock_loss_eligible"] for s in samples),
        "category_codes": {
            0: "NoData",
            1: "unknown",
            2: "verified_background",
            3: "verified_paint",
            4: "boundary_ignore",
            5: "conflict_ignore",
            6: "padding",
            7: "uncertain_ignore",
        },
        "training_ready": False,
    }
    dump(output / "metadata/dataset_manifest.json", manifest)
    dump(output / "metadata/instance_id_mapping.json", global_ids)
    dump(output / "splits/spatial_split.json", split)
    if sha256(Path(store)) != before or any(
        sha256(Path(p)) != h for p, h in source_protected.items()
    ):
        raise ValueError("Export changed source inputs")
    return manifest


def check_dataset(root):
    root = Path(root)
    manifest = json.loads((root / "metadata/dataset_manifest.json").read_text())
    for sample in manifest["samples"]:
        if sha256(Path(sample["rgb_image_reference"])) != sample["rgb_sha256"]:
            raise ValueError("Model RGB source changed")
        for key, relative in sample["validity_masks"].items():
            if sha256(root / relative) != sample["validity_sha256"][key]:
                raise ValueError("Validity mask hash mismatch")
            with rasterio.open(root / relative) as ds:
                if ds.shape != (
                    sample["height"],
                    sample["width"],
                ) or not ds.transform.almost_equals(tile_affine(sample)):
                    raise ValueError("Validity grid mismatch")
        for key, relative in sample["files"].items():
            path = root / relative
            if sha256(path) != sample["file_sha256"][key]:
                raise ValueError(f"Dataset hash mismatch: {path}")
            with rasterio.open(path) as ds:
                if ds.shape != (
                    sample["height"],
                    sample["width"],
                ) or not ds.transform.almost_equals(tile_affine(sample)):
                    raise ValueError("Image/mask geometry mismatch")
    return {
        "samples_checked": len(manifest["samples"]),
        "integrity_passed": True,
        "training_ready": manifest["training_ready"],
    }
