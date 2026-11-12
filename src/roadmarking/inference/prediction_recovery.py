"""Saved-prediction recovery: exact cached scoring, two-stage geographic filtering and
conservative observed-overlap solid reconciliation.

No function in this module accepts annotation geometry for repair. Ground truth only
enters ``packed_iou``/``cached_class_ap`` for evaluation. Reconciliation never adds a
pixel that was not already inside an existing predicted mask, never bridges unobserved
(NoData) gaps and never unions dashed markings.
"""

import numpy as np
from scipy.ndimage import binary_dilation, label

RECONCILIATION_RULES = {
    "classes": [0],
    "requires_different_crop_provenance": True,
    "minimum_observed_intersection_pixels": 8,
    "minimum_shared_support_agreement": 0.7,
    "minimum_local_pca_linearity": 0.9,
    "maximum_tangent_difference_degrees": 10.0,
    "maximum_lateral_offset_native_pixels": 1.5,
    "new_observed_components_must_be_anchored": True,
    "intensity_ring_dilations": [2, 5],
    "minimum_ring_observed_pixels": 16,
    "minimum_median_contrast_over_ring": 8.0,
    "output": "union of existing predicted masks only; no added, filled or extrapolated pixels",
}


def _popcount(packed):
    return int(np.bitwise_count(packed).sum())


def packed_iou(masks, targets, valid):
    """Exact IoU restricted to ``valid`` using packed bits (identical to ``iou_matrix``)."""
    valid = np.asarray(valid, bool)

    def pack(values):
        return [np.packbits(np.asarray(m, bool) & valid) for m in values]

    a, b = pack(masks), pack(targets)
    areas_a = np.array([_popcount(x) for x in a], dtype=np.int64)
    areas_b = np.array([_popcount(x) for x in b], dtype=np.int64)
    result = np.zeros((len(a), len(b)), float)
    for i, x in enumerate(a):
        if not areas_a[i]:
            continue
        for j, y in enumerate(b):
            if not areas_b[j]:
                continue
            intersection = _popcount(np.bitwise_and(x, y))
            if intersection:
                result[i, j] = intersection / max(1, areas_a[i] + areas_b[j] - intersection)
    return result


def filter_geographic(scores, classes, qualities, overlaps, crop_ids, config):
    """Per-crop thresholds + same-class NMS, then global same-class NMS.

    The two-stage order reproduces the Phase 5.9 frozen protocol; NMS is not associative.
    Threshold comparisons keep ``score >= threshold`` and ``quality >= mask_quality``.
    """

    def nms(indices):
        selected = []
        for i in sorted(indices, key=lambda j: (-scores[j], j)):
            if not any(
                classes[i] == classes[j] and overlaps[i, j] > config["nms_iou"] for j in selected
            ):
                selected.append(i)
        return selected

    local = []
    for crop in dict.fromkeys(crop_ids):
        eligible = [
            i
            for i in range(len(scores))
            if crop_ids[i] == crop
            and scores[i] >= config["confidence"][str(int(classes[i]))]
            and qualities[i] >= config["mask_quality"]
        ]
        local.extend(nms(eligible))
    return nms(local)


def cached_class_ap(iou, scores, classes, target_classes):
    """101-point per-class AP from a cached proposal x target IoU matrix.

    Matches ``roadmarking.training.pilot.instance_metrics`` (AP averaged over IoU
    .50:.05:.95, AP50, AP75). It is confidence-truncated: only supplied proposals count.
    """
    iou = np.asarray(iou, float).reshape(len(scores), len(target_classes))
    result = {}
    for cls, name in ((0, "longitudinal_solid"), (1, "longitudinal_dash")):
        truth = np.where(np.asarray(target_classes) == cls)[0]
        if not len(truth):
            result[name] = {"status": "NOT_EVALUATED", "ground_truth_instances": 0}
            continue
        proposed = sorted(np.where(np.asarray(classes) == cls)[0], key=lambda i: -scores[i])
        aps = []
        for threshold in np.arange(0.5, 0.951, 0.05):
            matched, hits = set(), []
            for index in proposed:
                available = [(iou[index, t], int(t)) for t in truth if t not in matched]
                value, best = max(available, default=(0.0, -1))
                positive = value >= threshold
                if positive:
                    matched.add(best)
                hits.append(int(positive))
            tp = np.cumsum(hits) if hits else np.zeros(0)
            fp = np.cumsum([1 - h for h in hits]) if hits else np.zeros(0)
            recall = tp / len(truth)
            precision = tp / np.maximum(tp + fp, 1)
            aps.append(
                float(
                    np.mean(
                        [max(precision[recall >= r], default=0.0) for r in np.linspace(0, 1, 101)]
                    )
                )
            )
        result[name] = {
            "status": "EVALUATED_ON_VALID_PIXELS_CONFIDENCE_TRUNCATED",
            "ground_truth_instances": len(truth),
            "AP": float(np.mean(aps)),
            "AP50": aps[0],
            "AP75": aps[5],
        }
    return result


def _axis(points):
    values, vectors = np.linalg.eigh(np.cov(points.T))
    return vectors[:, -1], float(values[-1] / max(1e-9, values.sum()))


def compatible(a, b, observed, image):
    """Decide whether two solid proposals from different crops describe the same paint.

    Uses only raw-proposal evidence, raw source observation support and real intensity.
    """
    if (
        a["class"] != 0
        or b["class"] != 0
        or not (a["crops"] - b["crops"] or b["crops"] - a["crops"])
    ):
        return False, "class_or_same_crop"
    am, bm = a["evidence"] & observed, b["evidence"] & observed
    common = a["support"] & b["support"] & observed
    intersection = int((am & bm & common).sum())
    if intersection < 8:
        return False, "insufficient_observed_intersection"
    agreement = intersection / max(1, min((am & common).sum(), (bm & common).sum()))
    if agreement < 0.7:
        return False, "overlap_disagreement"
    pa, pb = np.column_stack(np.where(am & common)), np.column_stack(np.where(bm & common))
    axis_a, linear_a = _axis(pa)
    axis_b, linear_b = _axis(pb)
    angle = np.degrees(np.arccos(np.clip(abs(np.dot(axis_a, axis_b)), 0, 1)))
    if min(linear_a, linear_b) < 0.9 or angle > 10:
        return False, "nonlinear_or_tangent_mismatch"
    normal = np.array([-axis_a[1], axis_a[0]])
    lateral = abs(np.median(pa @ normal) - np.median(pb @ normal))
    if lateral > 1.5:
        return False, "parallel_lateral_offset"
    union = am | bm
    components, _ = label(union, structure=np.ones((3, 3), int))
    # Every newly introduced observed island must connect to the existing mask.
    anchored = set(np.unique(components[am])) - {0}
    incoming = set(np.unique(components[bm])) - {0}
    if not incoming.issubset(anchored):
        return False, "unanchored_disconnected_fragment"
    ring = binary_dilation(union, iterations=5) & ~binary_dilation(union, iterations=2) & observed
    if ring.sum() < 16 or np.median(image[union]) < np.median(image[ring]) + 8:
        return False, "insufficient_local_intensity_contrast"
    return True, "observed_overlap_tangent_intensity_supported"


def reconcile_supported(proposals, observed, image):
    """Greedy score-ordered union of compatible solid fragments; returns (items, events)."""
    selected, events = [], []
    for original in sorted(proposals, key=lambda p: -p["score"]):
        item = {
            **original,
            "mask": original["mask"].copy(),
            "evidence": original["evidence"].copy(),
            "support": original["support"].copy(),
            "crops": set(original["crops"]),
            "members": list(original["members"]),
        }
        merged = False
        for other in selected:
            ok, reason = compatible(other, item, observed, image)
            if ok:
                other["mask"] = other["mask"] | item["mask"]
                other["evidence"] |= item["evidence"]
                other["support"] |= item["support"]
                other["crops"] |= item["crops"]
                other["members"] += item["members"]
                events.append(
                    {
                        "event": "solid_union",
                        "absorbed": list(item["members"]),
                        "members": other["members"].copy(),
                        "reason": reason,
                    }
                )
                merged = True
                break
        if not merged:
            selected.append(item)
    return selected, events
