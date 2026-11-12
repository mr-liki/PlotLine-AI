"""Stage 7G metric suite on geographic (tile-level) components.

Every physical marking is one record keyed by its global ID, so a line seen by several
crops counts once. Instances are matched greedily by score within class. Unmatched kept
predictions are split into *duplicates* (their best same-class target at the IoU threshold
was already matched by another prediction) and *false positives* (everything else).
Strata are descriptive; small strata are reported with their counts, never as precise CIs.
"""

from collections import defaultdict

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize

from roadmarking.evaluation.harness import ap_from_flags

NARROW_PX = 4.0  # mean width <= 0.20 m
LONG_PX = 512  # skeleton length > 25.6 m (spans more than one crop)


def match(iou, scores, classes, keep, target_classes, threshold):
    """Greedy score-ordered one-to-one matching of kept predictions within class."""
    order = sorted(keep, key=lambda i: -scores[i])
    taken, pred_flag, dup = set(), {}, {}
    for i in order:
        cand = [
            j
            for j in range(iou.shape[1])
            if target_classes[j] == classes[i] and iou[i, j] >= threshold
        ]
        free = [j for j in cand if j not in taken]
        if free:
            j = max(free, key=lambda j: iou[i, j])
            taken.add(j)
            pred_flag[i], dup[i] = True, False
        else:
            pred_flag[i], dup[i] = False, bool(cand)
    return taken, pred_flag, dup


def class_ap(iou, scores, classes, candidates, target_classes, threshold):
    """Threshold-free AP per class from all post-NMS candidates."""
    _, flags, _ = match(iou, scores, classes, list(candidates), target_classes, threshold)
    out = {}
    for c, name in ((0, "solid"), (1, "dash")):
        sel = [i for i in candidates if classes[i] == c]
        out[name] = ap_from_flags(
            [scores[i] for i in sel],
            [flags[i] for i in sel],
            int((np.asarray(target_classes) == c).sum()),
        )
    return out


def target_features(comp):
    """Per-target width, length, contrast and a roadside-structure proximity proxy."""
    valid, image = comp["valid"], comp["image"].astype(float)
    targets = comp["targets"]
    gt_union = targets.any(0) if len(targets) else np.zeros(valid.shape, bool)
    road = valid & ~binary_dilation(gt_union, iterations=3)
    bright = np.percentile(image[road], 95) if road.any() else np.inf
    feats = []
    for t in targets:
        skel = skeletonize(t)
        length = max(1, int(skel.sum()))
        ring = binary_dilation(t, iterations=10) & ~binary_dilation(t, iterations=3) & road
        near = binary_dilation(t, iterations=40) & ~binary_dilation(t, iterations=3) & road
        contrast = float(image[t].mean() - np.median(image[ring])) if ring.any() else np.nan
        feats.append(
            {
                "area_px": int(t.sum()),
                "length_px": length,
                "mean_width_px": float(t.sum() / length),
                "contrast": contrast,
                # Proxy: bright non-target structure (kerb, barrier, vegetation return) nearby.
                "near_bright_structure": bool((image[near] >= bright).sum() >= 50),
                "skeleton": skel,
            }
        )
    return feats


def component_metrics(comp, masks, scores, classes, keep, candidates, iou, stitch_audit=None):
    """Per-target and per-prediction records plus pixel counts for one component."""
    valid, targets = comp["valid"], comp["targets"]
    tcls = np.asarray(comp["target_labels"], int)
    keep = list(keep)
    union = masks[keep].any(0) & valid if keep else np.zeros(valid.shape, bool)
    gt_union = targets.any(0) if len(targets) else np.zeros(valid.shape, bool)
    pixel = {
        "tp": int((union & gt_union).sum()),
        "fp": int((union & ~gt_union).sum()),
        "fn": int((~union & gt_union & valid).sum()),
    }
    near_union = binary_dilation(union, iterations=1)
    matched = {}
    pred_rec = {}
    for thr, key in ((0.5, "50"), (0.75, "75")):
        taken, flags, dup = match(iou, scores, classes, keep, tcls, thr)
        matched[key] = taken
        for i in keep:
            rec = pred_rec.setdefault(
                i,
                {
                    "class": int(classes[i]),
                    "score": float(scores[i]),
                    "area_px": int((masks[i] & valid).sum()),
                },
            )
            rec[f"tp{key}"], rec[f"duplicate{key}"] = flags[i], dup[i]
    feats = target_features(comp)
    per_target = []
    for j, gid in enumerate(comp["ids"]):
        t, f = targets[j], feats[j]
        pieces = sum(
            1
            for i in keep
            if classes[i] == tcls[j]
            and (masks[i] & t).sum() >= 0.5 * max(1, (masks[i] & valid).sum())
        )
        per_target.append(
            {
                "id": gid,
                "component": comp["name"],
                "class": int(tcls[j]),
                "matched50": j in matched["50"],
                "matched75": j in matched["75"],
                "pixel_recall": float((union & t).sum() / max(1, t.sum())),
                "centerline_coverage": float(
                    (f["skeleton"] & near_union).sum() / max(1, f["skeleton"].sum())
                ),
                "pieces": pieces,
                **{k: v for k, v in f.items() if k != "skeleton"},
            }
        )
    ap = {
        "ap50": class_ap(iou, scores, classes, candidates, tcls, 0.5),
        "ap75": class_ap(iou, scores, classes, candidates, tcls, 0.75),
        "candidates": len(candidates),
    }
    return {
        "component": comp["name"],
        "pixel": pixel,
        "per_target": per_target,
        "per_pred": list(pred_rec.values()),
        "ap": ap,
        "stitch_audit": stitch_audit or {},
        "valid_km2": float(valid.sum()) * 0.05 * 0.05 / 1e6,
    }


def dedupe_targets(records):
    """One record per physical ID; an ID split across components keeps its largest part."""
    best = {}
    for r in records:
        if r["id"] not in best or r["area_px"] > best[r["id"]]["area_px"]:
            best[r["id"]] = r
    return list(best.values())


def _prf(tp, fp, fn):
    p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return p, r, 2 * p * r / max(1e-12, p + r)


def aggregate(components):
    """Pooled metric list over components (targets deduplicated by physical ID)."""
    px = defaultdict(int)
    for c in components:
        for k, v in c["pixel"].items():
            px[k] += v
    p, r, f = _prf(px["tp"], px["fp"], px["fn"])
    out = {
        "pixel_precision": p,
        "pixel_recall": r,
        "pixel_f1": f,
        "pixel_iou": px["tp"] / max(1, px["tp"] + px["fp"] + px["fn"]),
    }
    targets = dedupe_targets([t for c in components for t in c["per_target"]])
    preds = [p for c in components for p in c["per_pred"]]
    narrow = [t for t in targets if t["mean_width_px"] <= NARROW_PX]
    out["narrow_marking_pixel_recall"] = (
        float(np.mean([t["pixel_recall"] for t in narrow])) if narrow else None
    )
    out["narrow_marking_count"] = len(narrow)
    out["centerline_coverage"] = (
        float(np.mean([t["centerline_coverage"] for t in targets])) if targets else None
    )
    for cls, name in ((0, "solid"), (1, "dash"), (None, "all")):
        ts = [t for t in targets if cls is None or t["class"] == cls]
        ps = [p for p in preds if cls is None or p["class"] == cls]
        for key in ("50", "75"):
            tp = sum(t[f"matched{key}"] for t in ts)
            fp = sum(not p[f"tp{key}"] and not p[f"duplicate{key}"] for p in ps)
            dup = sum(p[f"duplicate{key}"] for p in ps)
            prec, rec, f1 = _prf(tp, fp + dup, len(ts) - tp)
            out[f"{name}_instance_precision{key}"] = prec
            out[f"{name}_instance_recall{key}"] = rec
            out[f"{name}_instance_f1{key}"] = f1
            out[f"{name}_misses{key}"] = len(ts) - tp
            out[f"{name}_false_positives{key}"] = fp
            out[f"{name}_duplicates{key}"] = dup
        out[f"{name}_gt"] = len(ts)
        covered = [t for t in ts if t["pieces"]]
        out[f"{name}_fragmented"] = sum(t["pieces"] > 1 for t in covered)
        out[f"{name}_pieces_per_covered"] = (
            float(np.mean([t["pieces"] for t in covered])) if covered else None
        )
    for metric in ("ap50", "ap75"):
        for name in ("solid", "dash"):
            vals = [c["ap"][metric][name] for c in components if c["ap"][metric][name] is not None]
            out[f"{name}_{metric}_mean_over_components"] = float(np.mean(vals)) if vals else None
    linked = [
        c["stitch_audit"]["false_links"]
        for c in components
        if "false_links" in c.get("stitch_audit", {})
    ]
    out["false_stitch_links"] = sum(linked) if linked else None  # set by the 7F audit
    km2 = sum(c["valid_km2"] for c in components)
    out["fp_per_km2"] = out["all_false_positives50"] / km2 if km2 else None
    out["strata"] = strata(targets)
    return out


def strata(targets):
    """Instance recall (IoU50) and pixel recall by descriptive strata, with counts."""
    contrasts = [t["contrast"] for t in targets if np.isfinite(t["contrast"])]
    cut = float(np.median(contrasts)) if contrasts else 0.0
    groups = {
        "solid": lambda t: t["class"] == 0,
        "dash": lambda t: t["class"] == 1,
        "long": lambda t: t["length_px"] > LONG_PX,
        "short": lambda t: t["length_px"] <= LONG_PX,
        "narrow": lambda t: t["mean_width_px"] <= NARROW_PX,
        "faint": lambda t: np.isfinite(t["contrast"]) and t["contrast"] < cut,
        "clear": lambda t: np.isfinite(t["contrast"]) and t["contrast"] >= cut,
        "near_bright_structure": lambda t: t["near_bright_structure"],
        "away_from_bright_structure": lambda t: not t["near_bright_structure"],
    }
    by_area = defaultdict(list)
    for t in targets:
        by_area[t["component"]].append(t)
    out = {}
    for name, rule in groups.items():
        sel = [t for t in targets if rule(t)]
        out[name] = _stratum(sel)
    out["by_area"] = {k: _stratum(v) for k, v in sorted(by_area.items())}
    out["contrast_median_cut"] = cut
    return out


def _stratum(sel):
    return {
        "n": len(sel),
        "instance_recall50": float(np.mean([t["matched50"] for t in sel])) if sel else None,
        "pixel_recall": float(np.mean([t["pixel_recall"] for t in sel])) if sel else None,
        "centerline_coverage": (
            float(np.mean([t["centerline_coverage"] for t in sel])) if sel else None
        ),
    }


def group_bootstrap(components, n=2000, seed=0, keys=None):
    """Percentile intervals resampling whole geographic components (regions)."""
    rng = np.random.default_rng(seed)
    keys = keys or ["pixel_iou", "pixel_recall", "all_instance_recall50", "all_instance_f150"]
    draws = defaultdict(list)
    for _ in range(n):
        pick = rng.integers(0, len(components), len(components))
        sample = []
        for k, i in enumerate(pick):
            comp = dict(components[i])
            # Re-key IDs per draw slot so a resampled region counts as a distinct copy.
            comp["per_target"] = [{**t, "id": f"{t['id']}#{k}"} for t in comp["per_target"]]
            sample.append(comp)
        agg = aggregate(sample)
        for key in keys:
            if agg[key] is not None:
                draws[key].append(agg[key])
    return {
        "groups": len(components),
        "caution": (
            "Fewer than ~10 groups: intervals are indicative only."
            if len(components) < 10
            else None
        ),
        "intervals": {
            k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in draws.items()
        },
    }
