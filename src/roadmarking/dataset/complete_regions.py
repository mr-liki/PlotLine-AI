"""Materialize the explicit Phase 5.3 completeness attestation and supervised crops."""

import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation
from shapely.geometry import shape

from roadmarking.dataset.approved_export import validate_approved_export
from roadmarking.dataset.labels import BACKGROUND, INSTANCE, NO_DATA, UNKNOWN
from roadmarking.rasterization.pipeline import dump, sha256, write_tif

COMPLETENESS_STATEMENT = (
    "All painted road-marking outlines in the six QGIS review windows have been inspected "
    "and approved. Every visible target road marking within those six reviewed regions "
    "is represented in the approved annotations. No additional manual annotation or "
    "repeated QGIS review is required for those same regions."
)


def prepare_complete_regions(source, output, report_dir):
    source, output, report_dir = Path(source).resolve(), Path(output).resolve(), Path(report_dir)
    if output.exists():
        raise FileExistsError(output)
    validate_approved_export(source)
    previous = json.loads((source / "metadata/dataset_manifest.json").read_text())
    reconciliation = json.loads((source / "metadata/approval_reconciliation.json").read_text())
    features = json.loads((source / "metadata/approved_geometry.geojson").read_text())["features"]
    output.mkdir(parents=True)
    approval = {
        "record_type": "USER_ATTESTATION_OF_COMPLETE_VISIBLE_TARGET_COVERAGE",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "reviewer_id": "project_user",
        "source": "User Phase 5.3 approval confirmation in this chat",
        "statement": COMPLETENESS_STATEMENT,
        "review_window_ids": [s["tile_id"] for s in previous["samples"]],
        "prior_outline_approval": reconciliation["approval_path"],
        "prior_outline_approval_sha256": reconciliation["approval_sha256"],
        "approved_artifact_sha256": reconciliation["approval"]["artifact_sha256"],
        "raster_and_source_artifact_sha256": previous["protected_inputs"],
        "source_dataset_manifest_sha256": sha256(source / "metadata/dataset_manifest.json"),
        "scope": "VISIBLE_TARGET_MARKINGS_WITHIN_THE_SIX_REVIEWED_WINDOWS_ONLY",
        "excludes": [
            "NoData",
            "padding",
            "ambiguous rasterization boundaries",
            "unresolved symbol-class footprints and conservative exclusion margins",
        ],
        "does_not_approve_unobserved_paint": True,
    }
    dump(output / "metadata/region_completeness_attestation.json", approval)
    for filename in (
        "approved_geometry.geojson",
        "approval_reconciliation.json",
        "class_mapping.json",
    ):
        shutil.copyfile(source / "metadata" / filename, output / "metadata" / filename)
    unsupported = [
        shape(f["geometry"]) for f in features if f["properties"]["visual_id"] not in (1, 2)
    ]
    tile_records, crops = [], []
    group_lookup = {
        tid: index
        for index, group in enumerate(previous["splits"]["candidate_components"])
        for tid in group
    }
    for sample in previous["samples"]:
        arrays = {}
        for key in ("state", "semantic", "instance"):
            with rasterio.open(source / sample["files"][key]) as ds:
                arrays[key] = ds.read(1)
        with rasterio.open(source / sample["files"]["image"]) as ds:
            observed = ds.read(2).astype(bool)
        with rasterio.open(source / "validity_masks" / sample["tile_id"] / "valid.tif") as ds:
            coverage = ds.read(1).astype(bool)
        affine = Affine(*sample["affine"])
        state = arrays["state"].copy()
        state[(state == UNKNOWN) & observed & coverage] = BACKGROUND
        if unsupported:
            exclusion = rasterize(
                [(g, 1) for g in unsupported],
                out_shape=state.shape,
                transform=affine,
                all_touched=True,
                dtype="uint8",
            ).astype(bool)
            exclusion = binary_dilation(exclusion, iterations=2)
        else:
            exclusion = np.zeros(state.shape, bool)
        state[exclusion & observed & coverage] = UNKNOWN
        state[~observed | ~coverage] = NO_DATA
        valid = np.isin(state, [BACKGROUND, INSTANCE]) & np.isin(arrays["semantic"], [0, 1, 2, 255])
        # Pixels newly promoted from UNKNOWN to BACKGROUND still have old semantic
        # value 255. State, rather than that historical semantic value, is authoritative.
        valid &= ~exclusion & observed & coverage
        valid[(arrays["state"] == INSTANCE) & ~np.isin(arrays["semantic"], [1, 2])] = False
        tile_dir = output / "tiles" / sample["tile_id"]
        state_path = tile_dir / "state.tif"
        valid_path = tile_dir / "valid_supervision.tif"
        background_path = tile_dir / "verified_background.tif"
        for path, array in (
            (state_path, state),
            (valid_path, valid),
            (background_path, state == BACKGROUND),
        ):
            write_tif(path, array.astype(np.uint8), affine, None)
        rgb = np.asarray(Image.open(source / sample["rgb_file"]))
        tile_records.append(
            {
                "tile_id": sample["tile_id"],
                "group": group_lookup[sample["tile_id"]],
                "affine": sample["affine"],
                "native_xy_bbox": sample["native_xy_bbox"],
                "source_sample": sample,
                "files": {
                    "state": str(state_path.relative_to(output)),
                    "validity": str(valid_path.relative_to(output)),
                },
                "verified_background_pixels": int((state == BACKGROUND).sum()),
                "valid_paint_pixels": int(((state == INSTANCE) & valid).sum()),
                "ignored_pixels": int((~valid).sum()),
                "unsupported_class_exclusion_pixels": int((exclusion & observed).sum()),
            }
        )
        for size in (512, 384, 256):
            stride = size
            starts = list(range(0, 1024 - size + 1, stride))
            if starts[-1] != 1024 - size:
                starts.append(1024 - size)
            for row in starts:
                for col in starts:
                    window = np.s_[row : row + size, col : col + size]
                    crop_valid = valid[window].copy()
                    local_instance = arrays["instance"][window]
                    targets, classes, identities = [], [], []
                    for instance in sample["instances"]:
                        if not instance["local_id"] or instance["visual_id"] not in (1, 2):
                            continue
                        mask = (local_instance == instance["local_id"]) & crop_valid
                        if 0 < mask.sum() < 4:
                            crop_valid[mask] = False
                            continue
                        if not mask.any():
                            continue
                        targets.append(mask)
                        classes.append(instance["visual_id"] - 1)
                        identities.append(instance["global_id"])
                    if not targets or crop_valid.mean() < 0.1:
                        continue
                    # Crop references retain all 95 source identities; only supported,
                    # observed positive instances enter this two-class pilot's targets.
                    masks = np.stack(targets).astype(np.uint8)
                    masks[:, ~crop_valid] = 0
                    name = f"{sample['tile_id']}_s{size}_r{row}_c{col}"
                    path = output / "crops" / (name + ".npz")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        path,
                        rgb=rgb[window],
                        masks=masks,
                        labels=np.asarray(classes, np.int64),
                        validity=crop_valid,
                    )
                    crop_affine = affine * Affine.translation(col, row)
                    crops.append(
                        {
                            "crop_id": name,
                            "file": str(path.relative_to(output)),
                            "sha256": sha256(path),
                            "tile_id": sample["tile_id"],
                            "group": group_lookup[sample["tile_id"]],
                            "size": size,
                            "window": [row, col, size, size],
                            "affine": list(crop_affine)[:6],
                            "native_xy_bbox": [
                                crop_affine.c,
                                crop_affine.f - size * 0.05,
                                crop_affine.c + size * 0.05,
                                crop_affine.f,
                            ],
                            "global_instance_ids": identities,
                            "labels": classes,
                            "complete_visible_region": True,
                            "valid_pixels": int(crop_valid.sum()),
                            "foreground_pixels": int(masks.any(0).sum()),
                        }
                    )
    # Whole connected groups, not individual overlapping windows, define allocation.
    group_weight = Counter()
    for crop in crops:
        if crop["size"] == 512:
            group_weight[crop["group"]] += crop["foreground_pixels"]
    ordered = sorted(group_weight, key=lambda g: (-group_weight[g], g))
    training_groups = ordered[:1]
    validation_groups = ordered[1:2]
    for crop in crops:
        crop["split"] = "train" if crop["group"] in training_groups else "val"
    for a in [c for c in crops if c["split"] == "train"]:
        for b in [c for c in crops if c["split"] == "val"]:
            if set(a["global_instance_ids"]) & set(b["global_instance_ids"]):
                raise ValueError("Cross-split physical instance leakage")
            from shapely.geometry import box

            if box(*a["native_xy_bbox"]).distance(box(*b["native_xy_bbox"])) <= 10:
                raise ValueError("Cross-split crop footprint guard violation")
    manifest = {
        "dataset_version": output.name,
        "source_dataset": str(source),
        "source_manifest_sha256": sha256(source / "metadata/dataset_manifest.json"),
        "status": "DATASET_READY",
        "region_completeness_attested": True,
        "attestation": "metadata/region_completeness_attestation.json",
        "attestation_sha256": sha256(output / "metadata/region_completeness_attestation.json"),
        "preserved_outline_count": len(features),
        "tiles": tile_records,
        "crops": crops,
        "training_classes": {"0": "longitudinal_solid", "1": "longitudinal_dash"},
        "excluded_visual_classes": [3, 6],
        "unsupported_exclusion_margin_pixels": 2,
        "splits": {
            "train_groups": training_groups,
            "val_groups": validation_groups,
            "test_groups": [],
            "cross_split_shared_instances": 0,
            "cross_split_guard_violations": 0,
            "guard_native_units": 10,
        },
        "criterion": {
            "name": "ValidMask2FormerCriterion",
            "max_points": 4096,
            "unmatched_class_policy": "NO_OBJECT_ONLY_IF_ENTIRE_CROP_VALID",
            "all_auxiliary_outputs_supervised": True,
        },
        "pixel_counts_are_overlapping_tile_appearances": True,
        "verified_background_pixels": sum(t["verified_background_pixels"] for t in tile_records),
        "valid_paint_pixels": sum(t["valid_paint_pixels"] for t in tile_records),
        "ignored_pixels": sum(t["ignored_pixels"] for t in tile_records),
    }
    if not any(c["split"] == "train" for c in crops):
        raise ValueError("No supported supervised training crop")
    dump(output / "metadata/dataset_manifest.json", manifest)
    dump(report_dir / "region_completeness_attestation.json", approval)
    dump(report_dir / "dataset_preparation.json", manifest)
    return manifest


def check_complete_dataset(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "metadata/dataset_manifest.json").read_text())
    source = Path(manifest["source_dataset"])
    if sha256(source / "metadata/dataset_manifest.json") != manifest["source_manifest_sha256"]:
        raise ValueError("Source dataset changed")
    validate_approved_export(source)
    if sha256(root / manifest["attestation"]) != manifest["attestation_sha256"]:
        raise ValueError("Completeness attestation changed")
    attestation = json.loads((root / manifest["attestation"]).read_text())
    previous = json.loads((source / "metadata/dataset_manifest.json").read_text())
    if attestation["record_type"] != "USER_ATTESTATION_OF_COMPLETE_VISIBLE_TARGET_COVERAGE" or set(
        attestation["review_window_ids"]
    ) != {s["tile_id"] for s in previous["samples"]}:
        raise ValueError("Region completeness scope differs from reviewed windows")
    if sha256(root / "metadata/approved_geometry.geojson") != sha256(
        source / "metadata/approved_geometry.geojson"
    ):
        raise ValueError("Approved geometry/identity archive changed")
    tile_validity = {}
    for tile in manifest["tiles"]:
        old = tile["source_sample"]
        with rasterio.open(source / old["files"]["image"]) as ds:
            observed = ds.read(2).astype(bool)
        with rasterio.open(source / old["files"]["state"]) as ds:
            old_state = ds.read(1)
        with rasterio.open(source / old["files"]["semantic"]) as ds:
            old_semantic = ds.read(1)
        with rasterio.open(root / tile["files"]["validity"]) as ds:
            tile_validity[tile["tile_id"]] = ds.read(1).astype(bool)
        valid = tile_validity[tile["tile_id"]]
        if (valid & ~observed).any() or (valid & np.isin(old_state, [4, 5])).any():
            raise ValueError("NoData or ambiguous boundary acquired supervision")
        if (valid & (old_state == INSTANCE) & ~np.isin(old_semantic, [1, 2])).any():
            raise ValueError("Unsupported paint acquired background supervision")
    for crop in manifest["crops"]:
        path = root / crop["file"]
        if sha256(path) != crop["sha256"]:
            raise ValueError("Supervised crop changed")
        with np.load(path) as arrays:
            valid, masks = arrays["validity"], arrays["masks"]
            row, col, height, width = crop["window"]
            expected = tile_validity[crop["tile_id"]][row : row + height, col : col + width]
            if (valid & ~expected).any():
                raise ValueError("Crop acquired unapproved pixel supervision")
            if masks[:, ~valid].any() or not valid.any():
                raise ValueError("Invalid supervised mask/NoData separation")
            if arrays["rgb"].shape != (crop["size"], crop["size"], 3):
                raise ValueError("Native crop size changed")
            if not np.array_equal(arrays["rgb"], np.repeat(arrays["rgb"][..., :1], 3, axis=2)):
                raise ValueError("Expected replicated grayscale")
            if not np.isin(arrays["labels"], [0, 1]).all():
                raise ValueError("Unsupported class in training targets")
    return {
        "training_ready": True,
        "status": "DATASET_READY",
        "crop_count": len(manifest["crops"]),
        "preserved_outlines": manifest["preserved_outline_count"],
        "splits": manifest["splits"],
    }
