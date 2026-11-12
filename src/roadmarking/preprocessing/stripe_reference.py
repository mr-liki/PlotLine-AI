"""Independent local-contrast stripe proposals; no proprietary tracing code.

Detection runs on all observations, never on the old road mask. Short local PCA
segments follow curvature; only direction-compatible, observed gaps may join.
Every result is a hypothesis, including bright roofs and fences.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from roadmarking.rasterization.rasterizer import pixel_to_world


@dataclass(frozen=True)
class StripeSettings:
    neighborhood_m: float = 2.0
    contrast_floor: float = 0.035
    contrast_sigma: float = 0.8
    segment_window_m: float = 12.0
    min_length_m: float = 1.5
    max_width_m: float = 0.8
    min_elongation: float = 4.0
    max_gap_m: float = 5.0
    max_turn_degrees: float = 20.0
    min_gap_observed_fraction: float = 0.6


def local_contrast(image, observed, resolution, settings=None):
    cfg = settings or StripeSettings()
    sigma = max(1, cfg.neighborhood_m / resolution)
    weight = ndimage.gaussian_filter(observed.astype(float), sigma)
    mean = ndimage.gaussian_filter(np.where(observed, image, 0.0), sigma) / np.maximum(weight, 1e-8)
    second = ndimage.gaussian_filter(np.where(observed, image**2, 0.0), sigma)
    std = np.sqrt(np.maximum(0, second / np.maximum(weight, 1e-8) - mean**2))
    contrast = np.where(observed, image - mean, 0)
    bright = observed & (contrast > np.maximum(cfg.contrast_floor, cfg.contrast_sigma * std))
    return contrast.astype(np.float32), bright


def _segment(coords, cfg):
    if len(coords) < 5:
        return None
    center = np.median(coords, axis=0)
    _, _, axes = np.linalg.svd(coords - center, full_matrices=False)
    direction = axes[0]
    if direction[np.argmax(np.abs(direction))] < 0:
        direction = -direction
    along = (coords - center) @ direction
    cross = (coords - center) @ np.array([-direction[1], direction[0]])
    lo, hi = np.percentile(along, [2, 98])
    width = float(np.percentile(cross, 95) - np.percentile(cross, 5))
    length = float(hi - lo)
    if (
        length < cfg.min_length_m
        or width > cfg.max_width_m
        or length / max(width, 0.1) < cfg.min_elongation
    ):
        return None
    return {
        "coords": [list(center + lo * direction), list(center + hi * direction)],
        "support_pixels": len(coords),
        "width_m": width,
        "length_m": length,
        "review_status": "PROPOSED_UNREVIEWED",
        "method": "local_contrast_local_PCA",
        "uncertainty": ["bright_linear_object_not_verified_paint", "native_units_unconfirmed"],
        "supported_gaps": [],
    }


def detect_references(image, observed, transform, settings=None):
    cfg = settings or StripeSettings()
    resolution = transform.a
    contrast, bright = local_contrast(image, observed, resolution, cfg)
    # One-cell dilation groups sparse samples; fitting still uses ORIGINAL bright cells.
    labels, _ = ndimage.label(ndimage.binary_dilation(bright), np.ones((3, 3)))
    step = max(4, round(cfg.segment_window_m / resolution))
    proposals = []
    for r0 in range(0, image.shape[0], step):
        for c0 in range(0, image.shape[1], step):
            region = labels[r0 : r0 + step, c0 : c0 + step]
            selected = bright[r0 : r0 + step, c0 : c0 + step]
            for label in np.unique(region[selected]):
                if label == 0:
                    continue
                rows, cols = np.nonzero((region == label) & selected)
                x, y = pixel_to_world(transform, cols + c0, rows + r0)
                proposal = _segment(np.column_stack((x, y)), cfg)
                if proposal:
                    proposals.append(proposal)
    # Greedy endpoint graph with explicit gap and support limits. Parallel offset
    # markings remain alternatives, rather than being collapsed to one centerline.
    cosine = np.cos(np.deg2rad(cfg.max_turn_degrees))
    changed = True
    while changed:
        changed = False
        best = None
        for i, a in enumerate(proposals):
            pa = np.asarray(a["coords"])
            da = pa[-1] - pa[-2]
            da /= max(np.linalg.norm(da), 1e-12)
            for j, b in enumerate(proposals):
                if i == j:
                    continue
                pb = np.asarray(b["coords"])
                db = pb[1] - pb[0]
                db /= max(np.linalg.norm(db), 1e-12)
                delta = pb[0] - pa[-1]
                gap = np.linalg.norm(delta)
                if gap > cfg.max_gap_m or np.dot(da, db) < cosine:
                    continue
                if gap > resolution and np.dot(da, delta / gap) < cosine:
                    continue
                samples = (
                    pa[-1] + np.linspace(0, 1, max(2, int(gap / resolution) + 1))[:, None] * delta
                )
                col = np.floor((samples[:, 0] - transform.c) / resolution).astype(int)
                row = np.floor((samples[:, 1] - transform.f) / transform.e).astype(int)
                inside = (
                    (row >= 0) & (col >= 0) & (row < observed.shape[0]) & (col < observed.shape[1])
                )
                support = float(observed[row[inside], col[inside]].sum()) / len(row)
                if support < cfg.min_gap_observed_fraction:
                    continue
                if best is None or gap < best[0]:
                    best = (gap, i, j, support)
        if best:
            gap, i, j, support = best
            a, b = proposals[i], proposals[j]
            a["coords"] += b["coords"]
            a["length_m"] += b["length_m"] + float(gap)
            a["support_pixels"] += b["support_pixels"]
            a["supported_gaps"] += b["supported_gaps"] + [
                {"length_m": float(gap), "observed_fraction": support, "paint_confirmed": False}
            ]
            proposals.pop(j)
            changed = True
    for i, proposal in enumerate(proposals):
        proposal["id"] = f"stripe_{i:04d}"
        proposal["reference_supported"] = proposal["length_m"] >= 8
    return proposals, {"contrast": contrast, "candidate_paint": bright.astype(np.uint8)}
