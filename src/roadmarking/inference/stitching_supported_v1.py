"""Experimental Stage 9G support-constrained stitching, version 1.

Inputs are geographic masks and per-crop support arrays on a verified common native
grid. Use only after affine and overlap checks. Disable identity-only gap links
pending a policy that can verify connections and preserve recognition quality.
"""

import numpy as np
from scipy.ndimage import label

from roadmarking.inference.stitching import stitch


def stitch_supported_v1(masks, scores, classes, crop_ids, windows, crop_support, thresholds):
    """Clip and split before overlap matching. Audit splits and retain source query IDs."""
    valid = np.logical_or.reduce([crop_support[crop] for crop in windows])
    fragments, fragment_scores, fragment_classes, fragment_crops, sources = [], [], [], [], []
    removed_pixels = 0
    split_queries = 0
    for i, (mask, score, cls, crop) in enumerate(
        zip(masks, scores, classes, crop_ids, strict=True)
    ):
        if mask.shape != valid.shape or crop_support[crop].shape != valid.shape:
            raise ValueError("Prediction and crop support must share the verified native grid")
        clipped = mask & crop_support[crop]
        removed_pixels += int((mask & ~clipped).sum())
        components, count = label(clipped, structure=np.ones((3, 3), bool))
        split_queries += count > 1
        for component in range(1, count + 1):
            fragments.append(components == component)
            fragment_scores.append(score)
            fragment_classes.append(cls)
            fragment_crops.append(crop)
            sources.append(i)
    outputs, audit = stitch(
        fragments,
        fragment_scores,
        fragment_classes,
        fragment_crops,
        windows,
        valid,
        thresholds,
        link_px=0,
    )
    for entry in outputs:
        entry["members"] = sorted({sources[j] for j in entry["members"]})
        # Overlap merges of connected inputs stay connected, and links add no pixels.
        if (entry["mask"] & ~valid).any() or label(entry["mask"], np.ones((3, 3), bool))[1] != 1:
            raise RuntimeError("Unsupported or disconnected stitched output")
    return outputs, dict(audit, removed_pixels=removed_pixels, split_queries=split_queries)
