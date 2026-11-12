"""Inspect actual export supervision and geography before weights/GPU training."""

import json
from pathlib import Path

import numpy as np
import rasterio
from shapely.geometry import box

from roadmarking.dataset.annotation_validation import validate_store
from roadmarking.dataset.labels import BACKGROUND, INSTANCE
from roadmarking.dataset.verified_export import check_dataset
from roadmarking.rasterization.pipeline import sha256


class DatasetNotReadyError(ValueError):
    """An actionable dataset prerequisite failed; no model should be loaded."""


def inspect_training_dataset(root, *, annotation_store=None, guard=10.0, allowed_visual_ids=(1, 2)):
    root = Path(root)
    manifest_path = root / "metadata/dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    samples = manifest.get("samples", [])
    errors, accepted = [], []
    eligible_ids, store_hash = set(), None
    if annotation_store is None:
        errors.append("NO_AUDITABLE_PAINT_AREA_APPROVAL_STORE")
    else:
        validation = validate_store(annotation_store)
        store_hash = sha256(Path(annotation_store))
        eligible_ids = {r["annotation_id"] for r in validation["records"] if r["eligible"]}
        if validation["invalid_features"] or validation["global_errors"]:
            errors.append("INVALID_OR_STALE_ANNOTATION_APPROVAL_STORE")
    if not samples:
        errors.append("NO_APPROVED_IMAGE_MASK_PAIRS")
    elif manifest.get("role") == "CONVERSION_REVIEW_ONLY":
        errors.append("CONVERSION_REVIEW_MASKS_ARE_NOT_APPROVED_SUPERVISION")
    else:
        try:
            check_dataset(root)
        except (KeyError, OSError, ValueError) as exc:
            errors.append(f"DATASET_INTEGRITY: {exc}")
    seen_tiles = set()
    for sample in samples:
        reasons = []
        tid = sample["tile_id"]
        if tid in seen_tiles:
            reasons.append("DUPLICATE_TILE_ID")
        seen_tiles.add(tid)
        if not sample.get("stock_loss_eligible"):
            reasons.append("NOT_STOCK_LOSS_ELIGIBLE")
        if not sample.get("review_provenance"):
            reasons.append("NO_PAINT_AREA_REVIEW_PROVENANCE")
        if not store_hash or sample.get("annotation_store_sha256") != store_hash:
            reasons.append("APPROVAL_STORE_HASH_MISMATCH_OR_MISSING")
        for record in sample.get("review_provenance", []):
            if record.get("annotation_id") not in eligible_ids:
                reasons.append("NO_CURRENT_HUMAN_REVIEW_EVENT_FOR_ANNOTATION")
            if not all(record.get(k) for k in ("annotation_id", "reviewer_id", "review_timestamp")):
                reasons.append("INCOMPLETE_REVIEW_PROVENANCE")
        arrays = {}
        try:
            for name in ("state", "semantic", "instance"):
                with rasterio.open(root / sample["files"][name]) as src:
                    arrays[name] = src.read(1)
            state, semantic, instance = (arrays[k] for k in ("state", "semantic", "instance"))
            if not np.isin(state, [BACKGROUND, INSTANCE]).all():
                reasons.append("UNKNOWN_NODATA_BOUNDARY_OR_PADDING_PIXELS")
            if (semantic == 255).any() or not np.isin(semantic, [0, *allowed_visual_ids]).all():
                reasons.append("IGNORE_OR_UNSUPPORTED_CLASS_IDS")
            if not np.array_equal(instance > 0, state == INSTANCE):
                reasons.append("INSTANCE_STATE_INCONSISTENCY")
            if not np.array_equal(semantic > 0, state == INSTANCE):
                reasons.append("SEMANTIC_STATE_INCONSISTENCY")
        except (KeyError, OSError, ValueError) as exc:
            reasons.append(f"MASK_READ_FAILURE: {exc}")
        if reasons:
            errors.extend(f"{tid}:{reason}" for reason in reasons)
        else:
            accepted.append(sample)
    split_info = manifest.get("splits", {})
    assignment = split_info.get("assignment", {})
    if not assignment:
        errors.append("NO_INDEPENDENT_SPATIAL_SPLITS")
    counts = {
        split: sum(assignment.get(s["tile_id"]) == split for s in accepted)
        for split in ("train", "val", "test")
    }
    for split in ("train", "val"):
        if not counts[split]:
            errors.append("NO_APPROVED_" + split.upper() + "_SAMPLES")
    # Independent recomputation prevents a manifest readiness flag bypass.
    for index, sample in enumerate(accepted):
        a_split = assignment.get(sample["tile_id"])
        a_ids = {r["global_id"] for r in sample.get("instances", []) if r.get("pixels")}
        for other in accepted[index + 1 :]:
            b_split = assignment.get(other["tile_id"])
            if not a_split or not b_split or a_split == b_split:
                continue
            a_box, b_box = box(*sample["native_xy_bbox"]), box(*other["native_xy_bbox"])
            if a_box.intersection(b_box).area > 0 or a_box.distance(b_box) < guard:
                errors.append("SPATIAL_SPLIT_GUARD_VIOLATION")
            b_ids = {r["global_id"] for r in other.get("instances", []) if r.get("pixels")}
            if a_ids & b_ids:
                errors.append("GLOBAL_INSTANCE_SPLIT_LEAKAGE")
    return {
        "training_ready": not errors,
        "errors": sorted(set(errors)),
        "candidate_samples": len(samples),
        "fully_supervised_samples": len(accepted),
        "split_counts": counts,
        "guard_native_units": guard,
        "test_evaluation_available": bool(counts["test"]),
        "weights_loaded": False,
        "gpu_training_run": False,
    }


def require_training_dataset(root, **kwargs):
    result = inspect_training_dataset(root, **kwargs)
    if not result["training_ready"]:
        raise DatasetNotReadyError("; ".join(result["errors"]))
    return result
