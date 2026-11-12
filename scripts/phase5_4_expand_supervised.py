"""Increase native positive/negative crops strictly inside the six approved regions."""

import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from shapely.geometry import box
from shapely.ops import unary_union

from roadmarking.dataset.complete_regions import check_complete_dataset
from roadmarking.rasterization.pipeline import dump, sha256


def main():
    root = Path(__file__).resolve().parents[1]
    context = json.loads((root / "outputs/reports/phase5_4/latest.json").read_text())
    reports = Path(context["report_dir"])
    baseline = json.loads((Path(context["baseline_run"]) / "result.json").read_text())
    source = Path(baseline["training_config"]["dataset"])
    check_complete_dataset(source)
    manifest = json.loads((source / "metadata/dataset_manifest.json").read_text())
    positive_source = Path(manifest["source_dataset"])
    output = root / "data/processed/phase5_training" / ("phase5_4_" + context["run_id"])
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source / "metadata", output / "metadata")
    shutil.copytree(source / "tiles", output / "tiles")
    old_crops = manifest["crops"]
    crops = []
    # Hold development validation fixed, byte-for-byte, at the baseline protocol.
    for crop in old_crops:
        if crop["split"] == "val" and crop["size"] == 512:
            crop = dict(crop)
            crop["file"] = str(source / crop["file"])
            crops.append(crop)
    for tile in manifest["tiles"]:
        if tile["group"] not in manifest["splits"]["train_groups"]:
            continue
        sample = tile["source_sample"]
        with rasterio.open(positive_source / sample["files"]["instance"]) as ds:
            instances = ds.read(1)
        with rasterio.open(source / tile["files"]["validity"]) as ds:
            valid = ds.read(1).astype(bool)
        rgb = np.asarray(Image.open(positive_source / sample["rgb_file"]))
        affine = Affine(*sample["affine"])
        for row in range(0, 513, 128):
            for col in range(0, 513, 128):
                window = np.s_[row : row + 512, col : col + 512]
                validity = valid[window].copy()
                if validity.mean() < 0.10:
                    continue
                local = instances[window]
                targets, labels, ids = [], [], []
                for instance in sample["instances"]:
                    if instance["visual_id"] not in (1, 2) or not instance["local_id"]:
                        continue
                    mask = (local == instance["local_id"]) & validity
                    if 0 < mask.sum() < 4:
                        validity[mask] = False
                        continue
                    if mask.any():
                        targets.append(mask)
                        labels.append(instance["visual_id"] - 1)
                        ids.append(instance["global_id"])
                masks = (
                    np.stack(targets).astype(np.uint8)
                    if targets
                    else np.empty((0, 512, 512), np.uint8)
                )
                masks[:, ~validity] = 0
                name = f"{sample['tile_id']}_s512_r{row}_c{col}"
                path = output / "crops" / (name + ".npz")
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    path,
                    rgb=rgb[window],
                    masks=masks,
                    labels=np.asarray(labels, np.int64),
                    validity=validity,
                )
                transform = affine * Affine.translation(col, row)
                background = validity & ~masks.any(0)
                bright = (
                    float(np.quantile(rgb[window][background, 0], 0.99))
                    if background.any()
                    else None
                )
                crops.append(
                    {
                        "crop_id": name,
                        "file": str(path.relative_to(output)),
                        "sha256": sha256(path),
                        "tile_id": sample["tile_id"],
                        "group": tile["group"],
                        "size": 512,
                        "window": [row, col, 512, 512],
                        "affine": list(transform)[:6],
                        "native_xy_bbox": [
                            transform.c,
                            transform.f - 25.6,
                            transform.c + 25.6,
                            transform.f,
                        ],
                        "global_instance_ids": ids,
                        "labels": labels,
                        "complete_visible_region": True,
                        "valid_pixels": int(validity.sum()),
                        "foreground_pixels": int(masks.any(0).sum()),
                        "split": "train",
                        "background_only": not len(targets),
                        "background_intensity_p99": bright,
                        "sampling_stratum": (
                            "bright_verified_negative"
                            if not targets and bright >= 180
                            else "verified_negative" if not targets else "positive"
                        ),
                    }
                )
    manifest.update(
        dataset_version=output.name,
        crops=crops,
        expansion_scope="DENSER_CROPS_AND_TRUE_NEGATIVES_WITHIN_EXISTING_APPROVAL_ONLY",
        parent_supervised_dataset=str(source),
        criterion=baseline["training_config"]["criterion"],
    )
    dump(output / "metadata/dataset_manifest.json", manifest)
    check = check_complete_dataset(output)
    old_train = [c for c in old_crops if c["split"] == "train" and c["size"] == 512]
    train = [c for c in crops if c["split"] == "train"]

    def unique_ids(items):
        return set(i for c in items for i in c["global_instance_ids"])

    def area(items):
        return float(unary_union([box(*c["native_xy_bbox"]) for c in items]).area)

    audit = {
        "dataset": str(output),
        "parent_dataset": str(source),
        "readiness": check,
        "old_native512_training_crops": len(old_train),
        "new_native512_training_crops": len(train),
        "verified_background_only_crops": sum(c["background_only"] for c in train),
        "bright_verified_negative_crops": sum(
            c["sampling_stratum"] == "bright_verified_negative" for c in train
        ),
        "unique_training_instance_ids_before": len(unique_ids(old_train)),
        "unique_training_instance_ids_after": len(unique_ids(train)),
        "unique_training_footprint_native_square_units_before": area(old_train),
        "unique_training_footprint_native_square_units_after": area(train),
        "new_independent_geographic_regions_approved": 0,
        "validation_crops_unchanged": 5,
        "counts_are_not_new_independent_regions": True,
        "instance_class_appearances": dict(Counter(str(i) for c in train for i in c["labels"])),
    }
    dump(reports / "supervised_dataset_expansion.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
