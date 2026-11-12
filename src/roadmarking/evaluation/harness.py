"""Stage-0 evaluation harness.

All metrics are computed from additive per-block records so that a paired spatial block
bootstrap can resample non-overlapping geographic blocks (not correlated instances).
Ground truth is used only for evaluation. Nothing here reads protected-test data.
"""

from collections import defaultdict

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

CLASSES = {0: "solid", 1: "dash"}


# --------------------------------------------------------------------------- folds


def region_groups(regions, guard=500.0, link_shared_ids=True):
    """Union regions closer than ``guard`` (and, optionally, sharing any source ID).

    Long client edge-line features run for kilometres, so ``link_shared_ids=True`` can chain
    nearly all regions together. With ``False`` only proximity links regions; callers must then
    report source IDs that span folds (distinct paint sections of one long feature).
    """
    parent = list(range(len(regions)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            a, b = regions[i], regions[j]
            shared = link_shared_ids and bool(set(a["ids"]) & set(b["ids"]))
            if a["box"].distance(b["box"]) < guard or shared:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(len(regions)):
        groups[find(i)].append(i)
    return [sorted(g) for g in groups.values()]


def assign_folds(regions, groups, k=5):
    """Greedy balance of linked groups into ``k`` folds by unique instance count."""
    sizes = [len({u for i in g for u in regions[i]["ids"]}) for g in groups]
    order = sorted(range(len(groups)), key=lambda g: (-sizes[g], min(groups[g])))
    folds = [[] for _ in range(k)]
    load = [0] * k
    for g in order:
        f = min(range(k), key=lambda f: (load[f], f))
        folds[f].extend(groups[g])
        load[f] += sizes[g]
    return [sorted(f) for f in folds]


# --------------------------------------------------------------------------- metrics


def ap_from_flags(scores, flags, n_gt):
    """101-point interpolated AP from score-ranked TP flags (IoU fixed upstream)."""
    if n_gt == 0:
        return None
    order = np.argsort(-np.asarray(scores, float), kind="stable")
    hits = np.asarray(flags, int)[order]
    if not len(hits):
        return 0.0
    tp = np.cumsum(hits)
    fp = np.cumsum(1 - hits)
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1)
    return float(
        np.mean([max(precision[recall >= r], default=0.0) for r in np.linspace(0, 1, 101)])
    )


def greedy_flags(iou, scores, classes, target_classes, threshold=0.5):
    """Score-ordered same-class greedy matching; returns (flags, matched_gt per detection)."""
    iou = np.asarray(iou).reshape(len(scores), len(target_classes))
    flags = np.zeros(len(scores), int)
    match = np.full(len(scores), -1)
    used = set()
    for i in sorted(range(len(scores)), key=lambda i: (-scores[i], i)):
        cands = [
            (iou[i, j], j)
            for j in range(len(target_classes))
            if target_classes[j] == classes[i] and j not in used
        ]
        value, best = max(cands, default=(0.0, -1))
        if value >= threshold:
            flags[i], match[i] = 1, best
            used.add(best)
    return flags, match


def buffer_counts(pred, gt, valid, buffer_px=3):
    """Wiedemann-style completeness/correctness counts on skeletons inside valid support."""
    gt_skel = skeletonize(gt) & valid
    pred_skel = skeletonize(pred & valid) & valid
    near_pred = distance_transform_edt(~pred) <= buffer_px if pred.any() else np.zeros_like(pred)
    near_gt = distance_transform_edt(~gt) <= buffer_px if gt.any() else np.zeros_like(gt)
    return {
        "completeness_hit": gt_skel & near_pred,
        "completeness_all": gt_skel,
        "correctness_hit": pred_skel & near_gt,
        "correctness_all": pred_skel,
    }


def block_of(rows, cols, block):
    return (np.asarray(rows) // block) * 100000 + (np.asarray(cols) // block)


def block_records(component, masks, scores, classes, keep, iou, block=256, buffer_px=3):
    """Additive per-block statistics for one model on one component.

    ``masks/scores/classes/iou`` are ALL saved proposals (for threshold-free AP); ``keep``
    indexes the filtered operating set (for F1, pixel, buffer, fragmentation metrics).
    """
    targets, tl, valid = component["targets"], component["target_labels"], component["valid"]
    records = defaultdict(lambda: defaultdict(float))
    detections = defaultdict(list)

    def centroid_block(mask):
        ys, xs = np.nonzero(mask)
        if not len(ys):
            return None
        return int(block_of(int(np.median(ys)), int(np.median(xs)), block))

    gt_blocks = [centroid_block(t) for t in targets]
    # Threshold-free AP inputs: every saved proposal, IoU50 greedy flags over all proposals.
    flags_all, _ = greedy_flags(iou, scores, classes, tl)
    for i in range(len(scores)):
        b = centroid_block(masks[i] & valid)
        if b is not None:
            detections[b].append((CLASSES[int(classes[i])], float(scores[i]), int(flags_all[i])))
    for j, b in enumerate(gt_blocks):
        if b is not None:
            records[b][f"{CLASSES[int(tl[j])]}_gt"] += 1
    # Operating point (filtered) instance counts.
    keep = list(keep)
    flags_op, match_op = greedy_flags(
        iou[keep] if len(keep) else np.zeros((0, len(tl))), scores[keep], classes[keep], tl
    )
    matched = set()
    for k, i in enumerate(keep):
        cls = CLASSES[int(classes[i])]
        if flags_op[k]:
            records[gt_blocks[match_op[k]]][f"{cls}_tp"] += 1
            matched.add(int(match_op[k]))
        else:
            b = centroid_block(masks[i] & valid)
            if b is not None:
                records[b][f"{cls}_fp"] += 1
    for j, b in enumerate(gt_blocks):
        if b is not None and j not in matched:
            records[b][f"{CLASSES[int(tl[j])]}_fn"] += 1
    # Pixel and buffer counts, split by the block each pixel lies in.
    shape = valid.shape
    rr, cc = np.indices(shape)
    pix_block = block_of(rr, cc, block)
    kept_masks = masks[keep] if keep else np.zeros((0, *shape), bool)
    pred_union = kept_masks.any(0) & valid if len(kept_masks) else np.zeros(shape, bool)
    gt_union = targets.any(0) if len(targets) else np.zeros(shape, bool)
    arrays = {
        "pixel_tp": pred_union & gt_union,
        "pixel_fp": pred_union & ~gt_union,
        "pixel_fn": gt_union & ~pred_union,
        "valid_px": valid,
    }
    solid_pred = (
        kept_masks[classes[keep] == 0].any(0)
        if len(kept_masks) and (classes[keep] == 0).any()
        else np.zeros(shape, bool)
    )
    solid_gt = targets[tl == 0].any(0) if (tl == 0).any() else np.zeros(shape, bool)
    for key, arr in buffer_counts(solid_pred, solid_gt, valid, buffer_px).items():
        arrays["solid_" + key] = arr
    for key, arr in arrays.items():
        ids, counts = np.unique(pix_block[arr], return_counts=True)
        for b, n in zip(ids.tolist(), counts.tolist(), strict=True):
            records[b][key] += n
    # Fragmentation: filtered solid masks overlapping each solid GT (>= 4 px), at GT block.
    for j in np.where(tl == 0)[0]:
        pieces = sum(
            int((kept_masks[k] & targets[j]).sum() >= 4)
            for k in range(len(kept_masks))
            if classes[keep[k]] == 0
        )
        if pieces and gt_blocks[j] is not None:
            records[gt_blocks[j]]["solid_covered_gt"] += 1
            records[gt_blocks[j]]["solid_pieces"] += pieces
    for b in detections:
        records[b]  # ensure every detection block exists
    return {b: dict(v) for b, v in records.items()}, dict(detections)


def summarize(records, detections, area_km2=None):
    """Metric suite from (possibly resampled) block records and detections."""
    tot = defaultdict(float)
    for r in records:
        for k, v in r.items():
            tot[k] += v
    dets = [d for block in detections for d in block]
    out = {}
    for cls in ("solid", "dash"):
        tp, fp, fn = tot[f"{cls}_tp"], tot[f"{cls}_fp"], tot[f"{cls}_fn"]
        p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
        out[f"{cls}_precision"], out[f"{cls}_recall"] = p, r
        out[f"{cls}_f1"] = 2 * p * r / max(1e-12, p + r)
        sel = [(s, f) for c, s, f in dets if c == cls]
        out[f"{cls}_ap50"] = ap_from_flags(
            [s for s, _ in sel], [f for _, f in sel], int(tot[f"{cls}_gt"])
        )
        out[f"{cls}_gt"] = int(tot[f"{cls}_gt"])
        out[f"{cls}_fp"] = float(fp)
    out["macro_f1"] = (out["solid_f1"] + out["dash_f1"]) / 2
    tp, fp, fn = tot["pixel_tp"], tot["pixel_fp"], tot["pixel_fn"]
    out["pixel_precision"] = tp / max(1, tp + fp)
    out["pixel_recall"] = tp / max(1, tp + fn)
    out["foreground_iou"] = tp / max(1, tp + fp + fn)
    out["solid_buffer_completeness"] = tot["solid_completeness_hit"] / max(
        1, tot["solid_completeness_all"]
    )
    out["solid_buffer_correctness"] = tot["solid_correctness_hit"] / max(
        1, tot["solid_correctness_all"]
    )
    out["solid_pieces_per_covered_gt"] = (
        tot["solid_pieces"] / tot["solid_covered_gt"] if tot["solid_covered_gt"] else None
    )
    if area_km2:
        out["fp_per_km2"] = (tot["solid_fp"] + tot["dash_fp"]) / area_km2
    return out


def paired_block_bootstrap(rec_a, det_a, rec_b, det_b, n=2000, seed=0, area_km2=None):
    """Percentile CIs for model A, model B and B - A, resampling shared spatial blocks."""
    blocks = sorted(set(rec_a) | set(rec_b) | set(det_a) | set(det_b))
    rng = np.random.default_rng(seed)
    base_a = summarize(
        [rec_a.get(b, {}) for b in blocks], [det_a.get(b, []) for b in blocks], area_km2
    )
    base_b = summarize(
        [rec_b.get(b, {}) for b in blocks], [det_b.get(b, []) for b in blocks], area_km2
    )
    draws = defaultdict(lambda: {"a": [], "b": [], "d": []})
    for _ in range(n):
        pick = rng.integers(0, len(blocks), len(blocks))
        sb = [blocks[i] for i in pick]
        ma = summarize([rec_a.get(b, {}) for b in sb], [det_a.get(b, []) for b in sb], area_km2)
        mb = summarize([rec_b.get(b, {}) for b in sb], [det_b.get(b, []) for b in sb], area_km2)
        for k in base_a:
            if isinstance(base_a[k], float) and ma[k] is not None and mb[k] is not None:
                draws[k]["a"].append(ma[k])
                draws[k]["b"].append(mb[k])
                draws[k]["d"].append(mb[k] - ma[k])
    out = {}
    for k, d in draws.items():

        def ci(x):
            return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

        out[k] = {
            "a": base_a[k],
            "b": base_b[k],
            "difference": (
                base_b[k] - base_a[k] if base_a[k] is not None and base_b[k] is not None else None
            ),
            "ci_a": ci(d["a"]),
            "ci_b": ci(d["b"]),
            "ci_difference": ci(d["d"]),
            "difference_excludes_zero": bool(ci(d["d"])[0] > 0 or ci(d["d"])[1] < 0),
        }
    return {"blocks": len(blocks), "resamples": n, "metrics": out}
