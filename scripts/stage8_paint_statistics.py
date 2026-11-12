"""Observed real paint in approved training/dev crops; never load test crops."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import shapely
from affine import Affine
from scipy.ndimage import binary_dilation

from phase5_11_common import ROOT, client_features, load
from roadmarking.rasterization.pipeline import dump

OUT = ROOT / "outputs/reports/stage8/8_4_paint_statistics.json"


def measure(crops, features):
    points, contrast = defaultdict(list), defaultdict(list)
    for c in crops:
        if c["split"] not in ("train", "val"):
            raise ValueError("Protected crop rejected")
        with np.load(c["file"]) as d:
            valid = d["validity"].astype(bool)
            masks = d["masks"].astype(bool) & valid
            labels = d["labels"]
            image = d["rgb"].astype(float).mean(-1)
        transform = Affine(*c["affine"][:6])
        for uid, cls, mask in zip(c["global_instance_ids"], labels, masks, strict=True):
            r, col = np.nonzero(mask)
            x, y = transform * (col + 0.5, r + 0.5)
            key = (c["split"], int(cls), uid)
            points[key].append(np.column_stack((x, y)))
            ring = (
                binary_dilation(mask, iterations=10)
                & ~binary_dilation(mask, iterations=2)
                & valid
                & ~masks.any(0)
            )
            if mask.any() and ring.any():
                contrast[key].append(float(np.median(image[mask]) - np.median(image[ring])))
    records = []
    for (split, cls, uid), parts in points.items():
        # Deduplicate repeated windows in native coordinates at 0.05 pixel precision.
        xy = np.unique(np.round(np.concatenate(parts), 6), axis=0)
        f = features.get(uid)
        supported = None
        if f is not None and f["centerline"] is not None and len(xy):
            along = shapely.line_locate_point(f["centerline"], shapely.points(xy))
            bins = np.bincount(np.floor(along).astype(int))
            supported = float(np.count_nonzero(bins >= 3))
        records.append(
            {
                "split": split,
                "class": "solid" if cls == 0 else "dash",
                "global_id": uid,
                "observed_paint_pixels": len(xy),
                "supported_length_native": supported,
                "contrast_rgb_mean_over_crops": (
                    float(np.mean(contrast[(split, cls, uid)]))
                    if contrast[(split, cls, uid)]
                    else None
                ),
            }
        )
    return records


def main():
    features = {f["global_id"]: f for f in client_features(check_network=False)[0]}
    result = {}
    for phase, manifest in (
        (
            "phase5_12",
            Path(load(ROOT / "outputs/reports/phase5_12/latest.json")["dataset"])
            / "metadata/dataset_manifest.json",
        ),
        (
            "stage6",
            ROOT
            / (
                "data/processed/phase5_training/stage6_20261106T185832535587Z/metadata/"
                "dataset_manifest.json"
            ),
        ),
    ):
        data = load(manifest)
        crops = [c for c in data["crops"] if c["split"] in ("train", "val")]
        records = measure(crops, features)
        summary = []
        for split in ("train", "val"):
            for cls in ("solid", "dash"):
                group = [r for r in records if r["split"] == split and r["class"] == cls]
                values = [
                    r["contrast_rgb_mean_over_crops"]
                    for r in group
                    if r["contrast_rgb_mean_over_crops"] is not None
                ]
                summary.append(
                    {
                        "split": split,
                        "class": cls,
                        "ids": len(group),
                        "observed_paint_pixels": sum(r["observed_paint_pixels"] for r in group),
                        "supported_length_native": sum(
                            r["supported_length_native"] or 0 for r in group
                        ),
                        "contrast_rgb_mean_over_ids": float(np.mean(values)) if values else None,
                    }
                )
        result[phase] = {
            "manifest": str(manifest),
            "summary": summary,
            "per_id": records,
            "region_statistics": data.get("statistics", {}),
        }
        print(f"{phase}: {len(records)} identities measured", flush=True)
        dump(OUT, result)
    result["method"] = (
        "Valid observed target pixels, native XY union per identity/split; "
        "1-native-unit centreline bins supported by >=3 pixels. Contrast uses "
        "rendered RGB scale, not raw intensity. Acquisition attribution comes "
        "from region metadata in 8_4_paint_metadata.json; missing mappings "
        "remain unresolved. No test crop loaded."
    )
    dump(OUT, result)


if __name__ == "__main__":
    main()
