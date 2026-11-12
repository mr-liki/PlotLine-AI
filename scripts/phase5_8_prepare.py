"""Audit real development support and export synthetic training augmentation."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from shapely.geometry import box

from roadmarking.dataset.labels import LabelFeature
from roadmarking.dataset.synthetic_adapter import adapt
from roadmarking.dataset.verified_masks import reviewed_masks
from roadmarking.rasterization.pipeline import dump, sha256

root = Path(__file__).resolve().parents[1]
run = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
report = root / "outputs/reports/phase5_8" / run
output = root / "data/processed/phase5_8_synthetic_adapter" / run
report.mkdir(parents=True)
output.mkdir(parents=True)
prior = json.loads((root / "outputs/reports/phase5_7_expanded/latest.json").read_text())
real = Path(prior["dataset"])
manifest = json.loads((real / "metadata/dataset_manifest.json").read_text())
review = json.loads((root / "outputs/reports/phase5_7_new_review/latest.json").read_text())
spec = json.loads((Path(review["report_dir"]) / "new_region_candidates.json").read_text())
tile = next(t for t in spec["tiles"] if t["candidate_id"] == "NEW_DEV_REVIEW")
with rasterio.open(tile["layers"]["observed"]) as ds:
    observed, affine = ds.read(1).astype(bool), ds.transform
frame = gpd.read_file(spec["candidate_store"], layer="candidate_paint_footprints")
rows = frame[(frame.candidate_id == "NEW_DEV_REVIEW") & frame.visual_id.isin([1, 2])]
features = [
    LabelFeature(i + 1, r.global_id, int(r.visual_id), "polygon", r.geometry)
    for i, (_, r) in enumerate(rows.iterrows())
]
labels, _ = reviewed_masks(
    affine,
    observed.shape,
    observed,
    np.ones_like(observed),
    features,
    completeness=box(*tile["bounds"]),
)
paint = (labels.instance > 0) & np.isin(labels.state, [2, 3]) & observed
yy, xx = np.where(paint)
coverage = {
    "observed_pixels": int(observed.sum()),
    "painted_observed_pixels": int(paint.sum()),
    "supported_source_ids": len(features),
    "positive_crop_proposals": [],
    "existing_approval_scope_preserved": True,
}
if len(yy):
    coverage["paint_pixel_bounds"] = [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]
    for size in (512, 384, 256):
        row = int(np.clip(np.median(yy) - size // 2, 0, 1024 - size))
        col = int(np.clip(np.median(xx) - size // 2, 0, 1024 - size))
        v = observed[row : row + size, col : col + size]
        p = paint[row : row + size, col : col + size]
        coverage["positive_crop_proposals"].append(
            {
                "size": size,
                "window": [row, col, size, size],
                "observed_pixels": int(v.sum()),
                "observed_paint_pixels": int(p.sum()),
                "affine": list(affine * rasterio.Affine.translation(col, row))[:6],
                "meaningful_positive_evidence": bool(p.sum() >= 32 and v.mean() >= 0.1),
            }
        )
coverage["decision"] = "Do not lower coverage threshold to manufacture positive geography"
dump(report / "real_positive_coverage_audit.json", coverage)
sources = [
    root / "road_markings_synthetic_500",
    root / "road_markings_synthetic_1000_v2/road_markings_synthetic_1000_v2",
]
crops = [{**c, "domain": "real"} for c in manifest["crops"]]
seen, hashes, stats, examples = {}, {}, {}, []
start = time.perf_counter()
synthetic_hist = {"foreground": np.zeros(256, np.int64), "background": np.zeros(256, np.int64)}
for volume, source in enumerate(sources):
    meta = json.loads((source / "annotations.json").read_text())
    counts = {
        "images": 0,
        "converted_train": 0,
        "converted_instances": 0,
        "empty_original_images": 0,
        "input_instance_classes": {},
        "duplicates": [],
        "ignored_pixels": 0,
        "original_dash_groups": 0,
        "adapted_dash_components": 0,
    }
    for item in meta["images"]:
        paths = [
            source / k / item["split"] / item["image"] for k in ("images", "instances", "masks")
        ]
        image, instance, semantic = [np.asarray(Image.open(p)) for p in paths]
        assert image.shape == instance.shape == semantic.shape == (512, 512)
        digests = [sha256(p) for p in paths]
        hashes.update({str(p): h for p, h in zip(paths, digests, strict=True)})
        image_hash = digests[0]
        duplicate = image_hash in seen
        if duplicate:
            counts["duplicates"].append({"image": str(paths[0]), "first": seen[image_hash]})
        else:
            seen[image_hash] = str(paths[0])
        masks, classes, valid, conversion = adapt(image, instance, semantic, item["instances"])
        assert not masks[:, ~valid].any()
        counts["images"] += 1
        counts["empty_original_images"] += int(not item["instances"])
        counts["converted_instances"] += len(classes)
        counts["ignored_pixels"] += int((~valid).sum())
        counts["original_dash_groups"] += sum(o["class_id"] == 0 for o in item["instances"])
        counts["adapted_dash_components"] += int((classes == 1).sum())
        for obj in item["instances"]:
            key = str(obj["class_id"])
            counts["input_instance_classes"][key] = counts["input_instance_classes"].get(key, 0) + 1
        paint = masks.any(0)
        for key, pixels in (
            ("foreground", image[paint & valid]),
            ("background", image[~paint & valid]),
        ):
            synthetic_hist[key] += np.bincount(pixels, minlength=256)
        if item["split"] != "train" or duplicate or not valid.any():
            continue
        target = output / f"synth_v{volume}_{Path(item['image']).stem}.npz"
        rgb = np.repeat(image[:, :, None], 3, axis=2)
        np.savez_compressed(target, rgb=rgb, masks=masks, labels=classes, validity=valid)
        ids = [
            f"synthetic:{volume}:{item['image']}:{m['original_group_id']}:{m['component']}"
            for m in conversion
        ]
        crops.append(
            {
                "crop_id": target.stem,
                "file": str(target),
                "sha256": sha256(target),
                "size": 512,
                "split": "train",
                "tile_id": "SYNTHETIC_TRAIN",
                "domain": "synthetic",
                "labels": classes.tolist(),
                "global_instance_ids": ids,
                "original_source": str(paths[0]),
                "conversion": conversion,
                "physical_gsd": None,
                "affine": None,
            }
        )
        counts["converted_train"] += 1
        if len(examples) < 8:
            examples.append(
                {"file": str(target), "source": str(paths[0]), "conversion": conversion}
            )
    stats[str(source)] = counts
    dump(report / "synthetic_audit_progress.json", stats)
real_hist = {"foreground": np.zeros(256, np.int64), "background": np.zeros(256, np.int64)}
real_valid, real_pixels = 0, 0
for crop in manifest["crops"]:
    with np.load(crop["file"]) as data:
        image = data["rgb"][:, :, 0]
        valid = data["validity"]
        paint = data["masks"].any(0)
        real_valid += int(valid.sum())
        real_pixels += valid.size
        for key, pixels in (
            ("foreground", image[paint & valid]),
            ("background", image[~paint & valid]),
        ):
            real_hist[key] += np.bincount(pixels, minlength=256)


def summarize(hist):
    return {
        key: {
            "count": int(h.sum()),
            "mean": float(np.dot(np.arange(256), h) / max(1, h.sum())),
            "quantiles_10_50_90": np.searchsorted(
                np.cumsum(h), np.asarray([0.1, 0.5, 0.9]) * h.sum()
            ).tolist(),
        }
        for key, h in hist.items()
    }


dump(
    report / "domain_gap.json",
    {
        "real": summarize(real_hist),
        "synthetic": summarize(synthetic_hist),
        "real_valid_pixel_fraction": real_valid / max(1, real_pixels),
        "scope": "Real crop appearance histograms, not unique geography or pavement-only labels",
        "physical_scale": "Synthetic GSD undefined; no metric width/length comparison",
        "limits": (
            "Pavement-only, curvature, spacing and hard-negative categories "
            "not inferred from global histograms"
        ),
        "augmentation": "None added; preserve procedural speckle/dropout and identity",
    },
)
dump(report / "source_hashes.json", hashes)
dump(
    report / "synthetic_audit.json",
    {
        "collections": stats,
        "adapter_seconds": time.perf_counter() - start,
        "mapping": {"synthetic_0_dash": "model_1", "synthetic_1_solid": "model_0"},
        "policy": (
            "8-connected components, minimum12 pixels/span6; "
            "ignore close (<4px) ambiguous dash groups; unsupported margin2; "
            "zero-intensity void"
        ),
        "originals_modified": False,
        "converted_examples": examples,
    },
)
dump(
    output / "metadata/dataset_manifest.json",
    {
        "parent_real": str(real),
        "crops": crops,
        "synthetic_is_augmentation_only": True,
        "protected_final_test_included": False,
    },
)
context = {
    "run_id": run,
    "report_dir": str(report),
    "dataset": str(output),
    "real_parent": str(real),
    "initialization": prior["selected_checkpoint"],
    "status": "SYNTHETIC_ADAPTER_READY",
}
dump(report.parent / "latest.json", context)
print(json.dumps(context, indent=2))
