"""Hash-bound seven-region approval, observed painted targets and geographic allocation."""

import hashlib
import json
import os
import stat
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation
from shapely.geometry import box
from shapely.ops import unary_union

from roadmarking.dataset.approved_export import spatial_components
from roadmarking.dataset.complete_regions import check_complete_dataset
from roadmarking.dataset.labels import BACKGROUND, INSTANCE, NO_DATA, UNKNOWN, LabelFeature
from roadmarking.dataset.verified_masks import reviewed_masks
from roadmarking.rasterization.pipeline import dump, sha256, write_tif

STATEMENT = (
    "The user personally inspected all seven Phase 5.4 QGIS review regions: the existing "
    "road-marking outlines are correctly positioned, every visible painted road marking "
    "in all seven regions is represented, and no additional QGIS review or redrawing is "
    "required. The original six Phase 5.3 regions and their 95 outlines remain approved."
)


def write_attestation(path, payload):
    """Write once and mark read-only; content hashes detect subsequent modification."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    os.chmod(path, stat.S_IREAD)
    return sha256(path)


def verify_hashes(hashes):
    for filename, expected in hashes.items():
        if sha256(Path(filename)) != expected:
            raise ValueError(f"Reviewed artifact changed: {filename}")


def allocate_regions(regions, reservations, guard=10.0):
    """Whole-footprint and full source-ID components; never split overlapping crops."""
    samples = [
        {
            "tile_id": r["tile_id"],
            "native_xy_bbox": r["native_xy_bbox"],
            "instances": [{"global_id": uid} for uid in r["source_global_ids"]],
        }
        for r in regions
    ]
    components = spatial_components(samples, guard)
    lookup = {r["tile_id"]: r for r in regions}
    assignment, audit = {}, []
    for number, component in enumerate(components):
        existing = {lookup[tid].get("existing_split") for tid in component} - {None}
        if len(existing) > 1:
            raise ValueError("New region bridges protected train and validation identities")
        # Existing protected split takes precedence. Any genuinely independent new
        # component is reserved for validation instead of automatically becoming train.
        split = next(iter(existing)) if existing else "val"
        for tid in component:
            assignment[tid] = split
        audit.append(
            {
                "component": number,
                "region_ids": component,
                "split": split,
                "reason": (
                    "Existing split retained through identity/footprint linkage"
                    if existing
                    else "New independent component assigned to validation"
                ),
            }
        )
    for region in regions:
        for reserved in reservations:
            if set(region["source_global_ids"]) & set(reserved["reference_annotation_ids"]):
                raise ValueError(
                    "Approved region shares an ID with a protected future-test reservation"
                )
            if box(*region["native_xy_bbox"]).distance(box(*reserved["bounds"])) <= guard:
                raise ValueError("Approved region violates a future-test geographic guard")
    return {
        "components": audit,
        "assignment": assignment,
        "guard_native_units": guard,
        "cross_split_shared_ids": 0,
        "cross_split_footprint_violations": 0,
        "future_test_regions_used": False,
        "future_test_region_count": len(reservations),
    }


def prepare(root, output, reports):
    root, output, reports = Path(root), Path(output).resolve(), Path(reports).resolve()
    context = json.loads((root / "outputs/reports/phase5_4/latest.json").read_text())
    old_reports = Path(context["report_dir"])
    review_path = old_reports / "new_review_artifact_manifest.json"
    review = json.loads(review_path.read_text())
    verify_hashes(review["artifact_sha256"])
    spec = json.loads((old_reports / "new_region_candidates.json").read_text())
    parent = Path(context["expanded_dataset"])
    check_complete_dataset(parent)
    old = json.loads((parent / "metadata/dataset_manifest.json").read_text())
    reserved_path = old_reports / "future_test_region_reservations.json"
    reservations = json.loads(reserved_path.read_text())["regions"]
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    reports.mkdir(parents=True, exist_ok=True)
    frame = gpd.read_file(
        spec["candidate_store"], layer="candidate_paint_footprints", fid_as_index=True
    )
    if (
        frame.geometry.isna().any()
        or frame.geometry.is_empty.any()
        or not frame.geometry.is_valid.all()
    ):
        raise ValueError(
            "Reviewed paint geometry is invalid; do not repair or transfer approval silently"
        )
    geometry = [
        {
            "derived_fid": int(fid),
            "region_id": row.candidate_id,
            "global_id": row.global_id,
            "source_identity": row.source_identity,
            "visual_id": int(row.visual_id),
            "supplied_width": (
                None if row.width is None or np.isnan(row.width) else float(row.width)
            ),
            "geometry_sha256": hashlib.sha256(row.geometry.wkb).hexdigest(),
        }
        for fid, row in frame.iterrows()
    ]
    prior_attestation = parent / old["attestation"]
    attestation = {
        "record_type": "USER_APPROVAL_SEVEN_ADDITIONAL_COMPLETE_VISIBLE_REGIONS",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "source": "User Phase 5.5 instructions in this chat",
        "statement": STATEMENT,
        "region_ids": review["candidate_ids"],
        "project": review["project"],
        "review_manifest": str(review_path),
        "review_manifest_sha256": sha256(review_path),
        "artifact_sha256": review["artifact_sha256"],
        "geometry_records": geometry,
        "prior_six_region_attestation": str(prior_attestation),
        "prior_six_region_attestation_sha256": sha256(prior_attestation),
        "future_test_reservations": str(reserved_path),
        "future_test_reservations_sha256": sha256(reserved_path),
        "scope": (
            "Exact seven reviewed footprints and visible observations only; NoData/padding excluded"
        ),
        "unsupported_classes_are_not_background": True,
        "original_six_need_no_repeat_review": True,
        "immutability": (
            "Exclusive creation, read-only file, SHA-256 checked before use; "
            "not a WORM-storage claim"
        ),
    }
    attestation_path = reports / "seven_region_approval.json"
    attestation_hash = write_attestation(attestation_path, attestation)
    regions = []
    old_assignment = {}
    for crop in old["crops"]:
        old_assignment[crop["tile_id"]] = crop["split"]
    for tile in old["tiles"]:
        sample = tile["source_sample"]
        regions.append(
            {
                "tile_id": tile["tile_id"],
                "native_xy_bbox": tile["native_xy_bbox"],
                "source_global_ids": [r["global_id"] for r in sample["instances"]],
                "existing_split": old_assignment.get(tile["tile_id"]),
                "newly_approved": False,
            }
        )
    for tile in spec["tiles"]:
        regions.append(
            {
                "tile_id": tile["candidate_id"],
                "native_xy_bbox": tile["bounds"],
                "source_global_ids": tile["source_global_ids"],
                "newly_approved": True,
            }
        )
    allocation = allocate_regions(regions, reservations)
    crops = []
    for crop in old["crops"]:
        if crop["size"] != 512:
            continue
        entry = dict(crop)
        entry["file"] = str(parent / crop["file"])
        if allocation["assignment"][crop["tile_id"]] != crop["split"]:
            raise ValueError("Rebuilt allocation changed protected evaluation membership")
        crops.append(entry)
    class_by_id = {}
    for uid, vid in zip(frame.global_id, frame.visual_id, strict=True):
        class_by_id[str(uid)] = int(vid)
    uids = {uid: i + 1 for i, uid in enumerate(sorted(class_by_id))}
    counts, new_tiles, observed_ids, cropped_ids = Counter(), [], set(), set()
    for tile in spec["tiles"]:
        subset = frame[frame.candidate_id == tile["candidate_id"]]
        affine = Affine(*tile["affine"])
        arrays = {}
        for name in ("observed", "normalized", "count"):
            with rasterio.open(tile["layers"][name]) as ds:
                if ds.transform != affine or ds.shape != (1024, 1024):
                    raise ValueError("Reviewed image/mask pixel mapping changed")
                arrays[name] = ds.read(1)
        observed = arrays["observed"].astype(bool)
        features = [
            LabelFeature(
                uids[row.global_id], row.global_id, int(row.visual_id), "polygon", row.geometry
            )
            for _, row in subset.iterrows()
            if row.visual_id in (1, 2)
        ]
        labels, _ = reviewed_masks(
            affine,
            (1024, 1024),
            observed,
            np.ones((1024, 1024), bool),
            features,
            completeness=box(*tile["bounds"]),
        )
        unsupported = list(subset[~subset.visual_id.isin([1, 2])].geometry)
        exclusion = (
            rasterize(
                [(g, 1) for g in unsupported],
                out_shape=(1024, 1024),
                transform=affine,
                all_touched=True,
                dtype="uint8",
            ).astype(bool)
            if unsupported
            else np.zeros((1024, 1024), bool)
        )
        if exclusion.any():
            exclusion = binary_dilation(exclusion, iterations=2)
        labels.state[exclusion & observed] = UNKNOWN
        labels.state[~observed] = NO_DATA
        valid = np.isin(labels.state, [BACKGROUND, INSTANCE]) & observed & ~exclusion
        labels.instance[~valid] = 0
        labels.semantic[~valid] = 255
        for record in labels.instances:
            record["pixels"] = (
                int((labels.instance == record["local_id"]).sum()) if record["local_id"] else 0
            )
            if record["pixels"] >= 4:
                observed_ids.add(record["global_id"])
        paths = {}
        for name, array in {
            "instance": labels.instance,
            "semantic": labels.semantic,
            "state": labels.state,
            "validity": valid.astype(np.uint8),
            "unsupported": exclusion.astype(np.uint8),
            "padding": np.zeros((1024, 1024), np.uint8),
        }.items():
            path = output / "tiles" / tile["candidate_id"] / (name + ".tif")
            write_tif(path, array, affine, None)
            paths[name] = str(path.relative_to(output))
        image = np.asarray(Image.open(tile["images"]["grayscale"]))
        rgb = np.repeat(image[..., None], 3, axis=2)
        tile_counts = Counter(
            {
                str(k): int(v)
                for k, v in zip(*np.unique(labels.state, return_counts=True), strict=True)
            }
        )
        counts.update(tile_counts)
        new_tiles.append(
            {
                "region_id": tile["candidate_id"],
                "affine": tile["affine"],
                "bounds": tile["bounds"],
                "observed_reference": tile["layers"]["observed"],
                "normalized_reference": tile["layers"]["normalized"],
                "grayscale_reference": tile["images"]["grayscale"],
                "files": paths,
                "file_sha256": {k: sha256(output / v) for k, v in paths.items()},
                "instances": labels.instances,
                "pixel_states": dict(tile_counts),
                "source_global_ids": tile["source_global_ids"],
            }
        )
        for row in range(0, 513, 128):
            for col in range(0, 513, 128):
                window = np.s_[row : row + 512, col : col + 512]
                validity = valid[window].copy()
                if validity.mean() < 0.1:
                    continue
                targets, classes, identities = [], [], []
                for record in labels.instances:
                    if not record["local_id"]:
                        continue
                    mask = (labels.instance[window] == record["local_id"]) & validity
                    if 0 < mask.sum() < 4:
                        validity[mask] = False
                        continue
                    if mask.any():
                        targets.append(mask)
                        classes.append(record["visual_id"] - 1)
                        identities.append(record["global_id"])
                masks = (
                    np.stack(targets).astype(np.uint8)
                    if targets
                    else np.empty((0, 512, 512), np.uint8)
                )
                masks[:, ~validity] = 0
                cropped_ids.update(identities)
                name = f"{tile['candidate_id']}_s512_r{row}_c{col}"
                path = output / "crops" / (name + ".npz")
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    path,
                    rgb=rgb[window],
                    masks=masks,
                    labels=np.asarray(classes, np.int64),
                    validity=validity,
                )
                transform = affine * Affine.translation(col, row)
                crops.append(
                    {
                        "crop_id": name,
                        "tile_id": tile["candidate_id"],
                        "file": str(path.relative_to(output)),
                        "sha256": sha256(path),
                        "size": 512,
                        "window": [row, col, 512, 512],
                        "affine": list(transform)[:6],
                        "native_xy_bbox": [
                            transform.c,
                            transform.f - 25.6,
                            transform.c + 25.6,
                            transform.f,
                        ],
                        "global_instance_ids": identities,
                        "labels": classes,
                        "split": allocation["assignment"][tile["candidate_id"]],
                        "complete_visible_region": True,
                        "valid_pixels": int(validity.sum()),
                        "foreground_pixels": int(masks.any(0).sum()),
                        "background_only": not len(targets),
                    }
                )
    verify_hashes(review["artifact_sha256"])
    unique_class = {"train": {}, "val": {}}
    for crop in crops:
        for uid, label in zip(crop["global_instance_ids"], crop["labels"], strict=True):
            unique_class[crop["split"]][uid] = label
    stats = {
        "new_approved_feature_ids": len(uids),
        "new_supported_ids_with_observed_pixels": len(observed_ids),
        "new_supported_ids_retained_in_crops": len(cropped_ids),
        "unique_source_instance_ids_by_split_class": {
            split: dict(Counter(str(v) for v in ids.values()))
            for split, ids in unique_class.items()
        },
        "crop_counts_by_split": dict(Counter(c["split"] for c in crops)),
        "hard_negative_training_crops": sum(
            not c["labels"] for c in crops if c["split"] == "train"
        ),
        "new_pixel_state_appearances": dict(counts),
        "unique_region_footprint_native_square_units_by_split": {
            split: float(
                unary_union(
                    [
                        box(*r["native_xy_bbox"])
                        for r in regions
                        if allocation["assignment"][r["tile_id"]] == split
                    ]
                ).area
            )
            for split in ("train", "val")
        },
        "overlapping_crop_counts_are_not_independent_regions": True,
    }
    manifest = {
        "schema": "phase5_5-approved-expanded-v1",
        "dataset_version": output.name,
        "parent_dataset": str(parent),
        "parent_manifest_sha256": sha256(parent / "metadata/dataset_manifest.json"),
        "approval": str(attestation_path),
        "approval_sha256": attestation_hash,
        "region_allocation": allocation,
        "regions": regions,
        "new_tiles": new_tiles,
        "crops": crops,
        "statistics": stats,
        "training_classes": {"0": "longitudinal_solid", "1": "longitudinal_dash"},
        "unsupported_class_policy": (
            "All footprints outside visual 1/2 excluded plus two-pixel margin"
        ),
        "future_test_reservations": str(reserved_path),
        "future_test_reservations_sha256": sha256(reserved_path),
    }
    dump(output / "metadata/dataset_manifest.json", manifest)
    dump(reports / "dataset_statistics.json", stats)
    dump(reports / "geographic_allocation.json", allocation)
    return manifest


def check_dataset(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "metadata/dataset_manifest.json").read_text())
    if manifest.get("schema") != "phase5_5-approved-expanded-v1":
        raise ValueError("Unexpected expanded dataset schema")
    approval = Path(manifest["approval"])
    if sha256(approval) != manifest["approval_sha256"]:
        raise ValueError("Approval changed")
    attestation = json.loads(approval.read_text())
    verify_hashes(attestation["artifact_sha256"])
    parent = Path(manifest["parent_dataset"])
    if sha256(parent / "metadata/dataset_manifest.json") != manifest["parent_manifest_sha256"]:
        raise ValueError("Prior dataset changed")
    check_complete_dataset(parent)
    reserved_path = Path(manifest["future_test_reservations"])
    if sha256(reserved_path) != manifest["future_test_reservations_sha256"]:
        raise ValueError("Protected future-test designation changed")
    allocation = allocate_regions(
        manifest["regions"], json.loads(reserved_path.read_text())["regions"]
    )
    if allocation != manifest["region_allocation"]:
        raise ValueError("Geographic allocation changed")
    valid_tiles = {}
    for tile in manifest["new_tiles"]:
        arrays = {}
        for name, path in tile["files"].items():
            if sha256(root / path) != tile["file_sha256"][name]:
                raise ValueError("Supervision raster changed")
            with rasterio.open(root / path) as ds:
                if ds.transform != Affine(*tile["affine"]):
                    raise ValueError("Pixel mapping changed")
                arrays[name] = ds.read(1)
        with rasterio.open(tile["observed_reference"]) as ds:
            observed = ds.read(1).astype(bool)
        valid = arrays["validity"].astype(bool)
        if (
            valid
            & (~observed | arrays["unsupported"].astype(bool) | arrays["padding"].astype(bool))
        ).any():
            raise ValueError("Excluded area acquired supervision")
        if not np.array_equal(arrays["instance"] > 0, (arrays["state"] == INSTANCE) & valid):
            raise ValueError("Instance/state mismatch")
        valid_tiles[tile["region_id"]] = valid
    for crop in manifest["crops"]:
        if sha256(root / crop["file"]) != crop["sha256"]:
            raise ValueError("Crop changed")
        if crop["split"] != allocation["assignment"][crop["tile_id"]]:
            raise ValueError("Crop split differs from region allocation")
        with np.load(root / crop["file"]) as data:
            if data["rgb"].shape != (512, 512, 3) or data["masks"][:, ~data["validity"]].any():
                raise ValueError("Unsafe crop targets or sampling")
            if not np.isin(data["labels"], [0, 1]).all():
                raise ValueError("Unsupported training class")
            if crop["tile_id"] in valid_tiles:
                row, col, height, width = crop["window"]
                if (
                    data["validity"]
                    & ~valid_tiles[crop["tile_id"]][row : row + height, col : col + width]
                ).any():
                    raise ValueError("Crop expanded supervision scope")
    return {
        "training_ready": True,
        "status": "DATASET_READY",
        "statistics": manifest["statistics"],
        "geographic_components": manifest["region_allocation"]["components"],
        "future_test_regions_used": False,
    }
