"""Phase 5.10: development-only saved-prediction recovery under the sealed protocol.

Reads only saved predictions, saved crops, approved development ground truth and raw
observation/intensity rasters for the three development components. No neural inference,
no weight change, no protected-test access.
"""

import json
import math
import time
from pathlib import Path

import matplotlib
import numpy as np
import rasterio
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.morphology import skeletonize

from roadmarking.inference.calibration import aggregate_counts, operating_counts
from roadmarking.inference.prediction_recovery import (
    cached_class_ap,
    filter_geographic,
    packed_iou,
    reconcile_supported,
)
from roadmarking.rasterization.pipeline import dump, sha256

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models/mask2former_swin_s/phase5_7_expanded_20261106T111240383727Z"
SAVED = CHECKPOINT / "validation_step0050"
NATIVE = 0.05
CLASS_NAMES = {0: "solid", 1: "dash"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_band(path):
    with rasterio.open(path) as ds:
        return ds.read(1), tuple(ds.transform)[:6]


# --------------------------------------------------------------------------- components


def _attach_proposals(comp, crops, prediction_folder):
    shape = comp["shape"]
    masks, scores, classes, qualities, crop_ids, local = [], [], [], [], [], []
    for crop in crops:
        with np.load(Path(prediction_folder) / (crop["crop_id"] + ".npz")) as data:
            raw = data["masks"].astype(bool)
            v = data["validity"].astype(bool)
            crop["raw"] = raw
            crop["saved_validity"] = v
            crop["saved_selected"] = data["selected"].astype(int).tolist()
            r, c = crop["row"], crop["col"]
            h, w = v.shape
            for k in range(len(raw)):
                m = np.zeros(shape, bool)
                m[r : r + h, c : c + w] = raw[k] & v
                masks.append(m)
                scores.append(float(data["scores"][k]))
                classes.append(int(data["labels"][k]))
                qualities.append(float(data["qualities"][k]))
                crop_ids.append(crop["crop_id"])
                local.append(k)
            support = np.zeros(shape, bool)
            support[r : r + h, c : c + w] = crop["observed_local"]
            crop["support"] = support
            # Saved validity must lie on actual source observations.
            assert not (v & ~crop["observed_local"]).any(), crop["crop_id"]
    comp["masks"] = np.stack(masks) if masks else np.zeros((0, *shape), bool)
    comp["scores"] = np.asarray(scores)
    comp["classes"] = np.asarray(classes, int)
    comp["qualities"] = np.asarray(qualities)
    comp["crop_ids"] = crop_ids
    comp["local_index"] = local
    comp["crops"] = crops


def legacy_component(manifest, p54, crop_ids, prediction_folder=SAVED):
    tiles = {t["tile_id"]: t for t in p54["tiles"]}
    entries = [c for c in manifest["crops"] if c["crop_id"] in crop_ids]
    x0 = min(c["affine"][2] for c in entries)
    y1 = max(c["affine"][5] for c in entries)
    crops = []
    for entry in entries:
        row = round((y1 - entry["affine"][5]) / NATIVE)
        col = round((entry["affine"][2] - x0) / NATIVE)
        crops.append({**entry, "row": row, "col": col})
    shape = (max(c["row"] for c in crops) + 512, max(c["col"] for c in crops) + 512)
    valid = np.zeros(shape, bool)
    observed = np.zeros(shape, bool)
    obs_seen = np.zeros(shape, bool)
    image = np.zeros(shape, np.uint8)
    img_seen = np.zeros(shape, bool)
    truths, labels = {}, {}
    conflicts = {"observed": 0, "intensity": 0, "target_overlap": 0}
    for crop in crops:
        r, c = crop["row"], crop["col"]
        with np.load(crop["file"]) as data:
            v = data["validity"].astype(bool)
            targets = data["masks"].astype(bool)
            tl = data["labels"].astype(int)
            gray = data["rgb"][..., 0]
        assert len(crop["global_instance_ids"]) == len(targets) == len(tl)
        obs_tile, transform = read_band(
            tiles[crop["tile_id"]]["source_sample"]["phase3_layers"]["observed"]
        )
        assert np.allclose(transform, tiles[crop["tile_id"]]["affine"])
        wr, wc, h, w = crop["window"]
        obs = obs_tile[wr : wr + h, wc : wc + w].astype(bool)
        crop["observed_local"] = obs
        sl = (slice(r, r + h), slice(c, c + w))
        conflicts["observed"] += int((obs_seen[sl] & (observed[sl] != obs)).sum())
        conflicts["intensity"] += int((img_seen[sl] & obs & (image[sl] != gray)).sum())
        observed[sl] |= obs
        obs_seen[sl] = True
        image[sl] = np.where(img_seen[sl], image[sl], gray)
        img_seen[sl] = True
        valid[sl] |= v
        for mask, cls, uid in zip(targets, tl, crop["global_instance_ids"], strict=True):
            truths.setdefault(uid, np.zeros(shape, bool))[sl] |= mask & v
            labels[uid] = int(cls)
        crop["targets_local"] = targets & v
        crop["target_labels_local"] = tl
        crop["validity_local"] = v
    ids = sorted(truths)
    for uid in ids:
        for other in ids:
            if other > uid:
                conflicts["target_overlap"] += int((truths[uid] & truths[other]).sum())
    targets = np.stack([truths[u] & valid for u in ids])
    comp = {
        "name": "LEGACY_POSITIVE_DEV",
        "shape": shape,
        "valid": valid,
        "observed": observed,
        "image": image,
        "targets": targets,
        "target_labels": np.asarray([labels[u] for u in ids], int),
        "ids": ids,
        "positive": True,
        "consistency": conflicts,
        "origin_native_xy": [x0, y1],
    }
    _attach_proposals(comp, crops, prediction_folder)
    return comp


def tile_component(name, tile, entries, prediction_folder, ground_truth=None, ids=None):
    observed_tile, transform = read_band(tile["layers"]["observed"])
    assert np.allclose(transform, tile["affine"])
    observed = observed_tile.astype(bool)
    image = np.asarray(Image.open(tile["images"]["grayscale"]).convert("L"))
    shape = observed.shape
    crops = []
    for entry in entries:
        r, c, h, w = entry["window"]
        crop = {**entry, "row": r, "col": c}
        crop["observed_local"] = observed[r : r + h, c : c + w]
        with np.load(entry["file"]) as data:
            crop["validity_local"] = data["validity"].astype(bool)
            crop["targets_local"] = data["masks"].astype(bool) & crop["validity_local"]
            crop["target_labels_local"] = data["labels"].astype(int)
        crops.append(crop)
    if ground_truth is not None:
        with np.load(ground_truth) as gt:
            targets = gt["masks"].astype(bool)
            target_labels = gt["labels"].astype(int)
            valid = gt["validity"].astype(bool)
        assert len(ids) == len(targets)
    else:
        valid = np.zeros(shape, bool)
        for crop in crops:
            r, c = crop["row"], crop["col"]
            h, w = crop["validity_local"].shape
            valid[r : r + h, c : c + w] |= crop["validity_local"]
        targets = np.zeros((0, *shape), bool)
        target_labels = np.zeros(0, int)
        ids = []
    support = np.zeros(shape, bool)
    for crop in crops:
        r, c = crop["row"], crop["col"]
        h, w = crop["validity_local"].shape
        support[r : r + h, c : c + w] |= crop["validity_local"]
    assert not (valid & ~support).any(), "scored pixels outside actual crop validity"
    comp = {
        "name": name,
        "shape": shape,
        "valid": valid,
        "observed": observed,
        "image": image,
        "targets": targets & valid,
        "target_labels": target_labels,
        "ids": list(ids),
        "positive": len(targets) > 0,
        "consistency": {"single_tile": True},
    }
    _attach_proposals(comp, crops, prediction_folder)
    return comp


def overlap_matrix(comp):
    """Same-class global IoU on crop-validity-clipped masks; skips non-overlapping windows."""
    n = len(comp["scores"])
    packed = [np.packbits(m) for m in comp["masks"]]
    areas = np.array([int(np.bitwise_count(p).sum()) for p in packed])
    windows = {
        c["crop_id"]: (c["row"], c["col"], c["row"] + 512, c["col"] + 512) for c in comp["crops"]
    }

    def touch(a, b):
        a, b = windows[a], windows[b]
        return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

    result = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            if comp["classes"][i] != comp["classes"][j] or not touch(
                comp["crop_ids"][i], comp["crop_ids"][j]
            ):
                continue
            inter = int(np.bitwise_count(np.bitwise_and(packed[i], packed[j])).sum())
            if inter:
                result[i, j] = result[j, i] = inter / max(1, areas[i] + areas[j] - inter)
    return result


def intersections(masks, targets, valid):
    pm = [np.packbits(m & valid) for m in masks]
    pt = [np.packbits(t & valid) for t in targets]
    out = np.zeros((len(pm), len(pt)), np.int64)
    for i, a in enumerate(pm):
        for j, b in enumerate(pt):
            out[i, j] = int(np.bitwise_count(np.bitwise_and(a, b)).sum())
    return out


# --------------------------------------------------------------------------- metrics


def score(comp, masks, scores, classes, iou=None, inter=None):
    valid, targets, tl = comp["valid"], comp["targets"], comp["target_labels"]
    if iou is None:
        iou = packed_iou(masks, targets, valid)
    if inter is None:
        inter = intersections(masks, targets, valid)
    iou = np.asarray(iou).reshape(len(scores), len(tl))
    inter = np.asarray(inter).reshape(len(scores), len(tl))
    rec = {
        "iou": iou,
        "scores": np.asarray(scores),
        "labels": np.asarray(classes),
        "target_labels": tl,
    }
    idx = list(range(len(scores)))
    ops = operating_counts(rec, idx)
    counts = aggregate_counts([rec], [idx])
    shape = comp["shape"]
    pred_union = masks.any(0) & valid if len(masks) else np.zeros(shape, bool)
    gt_union = targets.any(0) if len(targets) else np.zeros(shape, bool)
    pixel = {}
    for cls, key in ((None, "all_supported_paint"), (0, "solid"), (1, "dash")):
        sel = np.ones(len(classes), bool) if cls is None else np.asarray(classes) == cls
        a = masks[sel].any(0) & valid if sel.any() else np.zeros(shape, bool)
        tsel = np.ones(len(tl), bool) if cls is None else tl == cls
        b = targets[tsel].any(0) if tsel.any() else np.zeros(shape, bool)
        tp, fp, fn = int((a & b).sum()), int((a & ~b).sum()), int((b & ~a).sum())
        p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
        pixel[key] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p,
            "recall": r,
            "f1": 2 * p * r / max(1e-12, p + r),
        }
    fg = (gt_union & pred_union).sum() / max(1, ((gt_union | pred_union) & valid).sum())
    result = {
        "predictions": len(scores),
        "per_class": {
            CLASS_NAMES[c]: {
                k: ops[str(c)][k]
                for k in ("tp", "fp", "fn", "duplicates", "precision", "recall", "f1")
            }
            for c in (0, 1)
        },
        "macro_f1": counts["macro_f1"],
        "pixel": pixel,
        "foreground_iou": float(fg),
    }
    if len(tl):
        result["ap"] = cached_class_ap(iou, scores, classes, tl)
        skeleton = comp["skeleton"]
        near = (
            distance_transform_edt(~pred_union) <= 1 if pred_union.any() else np.zeros(shape, bool)
        )
        result["observed_skeleton_coverage_within_1px"] = float(
            (skeleton & near).sum() / max(1, skeleton.sum())
        )
        frag = []
        for j in np.where(tl == 0)[0]:
            hits = [i for i in idx if classes[i] == 0 and inter[i, j] >= 4]
            frag.append(len(hits))
        covered = [f for f in frag if f]
        result["solid_fragmentation"] = {
            "solid_targets": len(frag),
            "targets_with_any_overlapping_solid_mask": len(covered),
            "targets_with_multiple_overlapping_solid_masks": sum(f > 1 for f in frag),
            "overlapping_solid_masks_total": int(sum(frag)),
            "mean_masks_per_covered_target": float(np.mean(covered)) if covered else 0.0,
            "per_target": frag,
        }
        dash_t = np.where(tl == 1)[0]
        areas = targets.sum((1, 2))
        multi = 0
        for i in idx:
            if classes[i] != 1:
                continue
            hits = sum(inter[i, j] >= 4 and inter[i, j] / max(1, areas[j]) >= 0.1 for j in dash_t)
            multi += hits > 1
        result["dash_separation"] = {
            "dash_targets": len(dash_t),
            "potential_multi_dash_prediction_masks": int(multi),
            "dash_targets_touched_by_dash_prediction": int(
                sum(any(classes[i] == 1 and inter[i, j] >= 4 for i in idx) for j in dash_t)
            ),
        }
    result["_ops"] = ops
    return result


def public(metrics):
    return {k: v for k, v in metrics.items() if not k.startswith("_")}


def crop_appearance(comp, config, overlaps):
    records, selections = [], []
    for crop in comp["crops"]:
        idx = [i for i, cid in enumerate(comp["crop_ids"]) if cid == crop["crop_id"]]
        local = filter_geographic(
            comp["scores"][idx],
            comp["classes"][idx],
            comp["qualities"][idx],
            overlaps[np.ix_(idx, idx)],
            [crop["crop_id"]] * len(idx),
            config,
        )
        raw = crop["raw"]
        v = crop["validity_local"]
        records.append(
            {
                "iou": packed_iou(raw, crop["targets_local"], v).reshape(
                    len(raw), len(crop["targets_local"])
                ),
                "scores": comp["scores"][idx],
                "labels": comp["classes"][idx],
                "target_labels": crop["target_labels_local"],
            }
        )
        selections.append(local)
    return aggregate_counts(records, selections)


# --------------------------------------------------------------------------- selection


def feasible(cand, base, neg_cand, neg_base):
    reasons = []
    for comp in base:
        b, c = base[comp], cand[comp]
        fp_b = sum(b["per_class"][k]["fp"] for k in ("solid", "dash"))
        fp_c = sum(c["per_class"][k]["fp"] for k in ("solid", "dash"))
        if fp_c > max(fp_b + 2, math.ceil(1.25 * fp_b)):
            reasons.append(f"{comp}:C4_total_fp_budget")
        for k in ("solid", "dash"):
            bk, ck = b["per_class"][k], c["per_class"][k]
            if ck["tp"] < bk["tp"]:
                reasons.append(f"{comp}:{k}:C1_tp_floor")
            if bk["tp"] > 0 and ck["precision"] < bk["precision"] - 0.05 - 1e-12:
                reasons.append(f"{comp}:{k}:C2_precision_floor")
            if bk["tp"] == 0 and ck["tp"] > 0 and ck["precision"] < 0.10 - 1e-12:
                reasons.append(f"{comp}:{k}:C3_precision_floor")
            if ck["fp"] > max(bk["fp"] + 2, math.ceil(1.25 * bk["fp"])):
                reasons.append(f"{comp}:{k}:C4_class_fp_budget")
    fn_c = sum(neg_cand["per_class"][k]["fp"] for k in ("solid", "dash"))
    fn_b = sum(neg_base["per_class"][k]["fp"] for k in ("solid", "dash"))
    if fn_c > fn_b:
        reasons.append("SPARSE_NEGATIVE_CONTROL:C5_negative_fp")
    return not reasons, reasons


def objective(cand, neg):
    f1 = [cand[c]["macro_f1"] for c in cand]
    fp = sum(cand[c]["per_class"][k]["fp"] for c in cand for k in ("solid", "dash"))
    fp += sum(neg["per_class"][k]["fp"] for k in ("solid", "dash"))
    return min(f1), float(np.mean(f1)), fp


# --------------------------------------------------------------------------- outputs

COLORS = {0: (1.0, 0.25, 0.1), 1: (0.1, 0.6, 1.0)}


def overlay(gray, layers, alpha=0.65):
    base = np.repeat((gray.astype(float) / 255.0)[..., None], 3, -1)
    for mask, color in layers:
        if mask is not None and mask.any():
            base[mask] = (1 - alpha) * base[mask] + alpha * np.asarray(color)
    return base


def gallery(comp, stages, path):
    rows = 1 + len(stages)
    fig, axes = plt.subplots(
        rows, 3, figsize=(15, 4.2 * rows * comp["shape"][0] / comp["shape"][1] + 1)
    )
    gray, valid, tl = comp["image"], comp["valid"], comp["target_labels"]
    gt_layers = [
        (comp["targets"][tl == c].any(0) if (tl == c).any() else None, COLORS[c]) for c in (0, 1)
    ]
    first = [
        ("Actual intensity (established rendering)", overlay(gray, [])),
        ("Approved GT on scored support (red solid, blue dash)", overlay(gray, gt_layers)),
        ("Not scored (outside valid support)", overlay(gray, [(~valid, (0.5, 0.0, 0.5))], 0.35)),
    ]
    for ax, (title, img) in zip(axes[0], first, strict=True):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
    for row, (name, masks, classes, metrics) in enumerate(stages, start=1):
        ops = metrics["_ops"]
        fps = [i for c in ("0", "1") for i in ops[c]["false_indices"]]
        miss = [j for c in ("0", "1") for j in ops[c]["missed_indices"]]
        pred_layers = [
            (masks[classes == c].any(0) if (classes == c).any() else None, COLORS[c])
            for c in (0, 1)
        ]
        fp_layer = [(masks[fps].any(0) if fps else None, (1.0, 0.9, 0.0))]
        fn_layer = [(comp["targets"][miss].any(0) if miss else None, (1.0, 0.0, 1.0))]
        pc = metrics["per_class"]
        panels = [
            (
                f"{name}: predictions ({len(masks)}) macroF1 {metrics['macro_f1']:.3f}",
                overlay(gray, pred_layers),
            ),
            (
                f"{name}: unmatched preds (FP s{pc['solid']['fp']}/d{pc['dash']['fp']})",
                overlay(gray, fp_layer),
            ),
            (
                f"{name}: missed GT (FN s{pc['solid']['fn']}/d{pc['dash']['fn']})",
                overlay(gray, fn_layer),
            ),
        ]
        for ax, (title, img) in zip(axes[row], panels, strict=True):
            ax.imshow(img)
            ax.set_title(title, fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"{comp['name']} — saved-prediction recovery (development only; observed support)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def event_gallery(comp, events, path):
    events = [e for e in events if e["stage_masks"] is not None][:12]
    if not events:
        return False
    cols = 3
    rows = math.ceil(len(events) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    palette = [(1, 0.2, 0.1), (0.1, 0.9, 0.2), (0.2, 0.4, 1), (1, 0.8, 0), (0.9, 0.2, 0.9)]
    for ax, event in zip(axes.flat, events, strict=False):
        members = event["stage_masks"]
        union = np.any(members, 0)
        ys, xs = np.where(union)
        y0, y1 = max(0, ys.min() - 25), min(union.shape[0], ys.max() + 25)
        x0, x1 = max(0, xs.min() - 25), min(union.shape[1], xs.max() + 25)
        outlines = [(binary_dilation(m) & ~m, palette[k % 5]) for k, m in enumerate(members)]
        img = overlay(comp["image"], [(m, palette[k % 5]) for k, m in enumerate(members)], 0.35)
        img = overlay((img.mean(-1) * 255).astype(np.uint8), outlines, 0.9)
        ax.imshow(img[y0:y1, x0:x1])
        ax.imshow(
            np.ma.masked_where(comp["observed"][y0:y1, x0:x1], np.ones((y1 - y0, x1 - x0))),
            cmap="Greys",
            alpha=0.35,
            vmin=0,
            vmax=1,
        )
        ax.set_title("\n".join(event["members"])[:160], fontsize=7)
        ax.axis("off")
    for ax in axes.flat[len(events) :]:
        ax.axis("off")
    fig.suptitle(f"{comp['name']} — solid unions (members coloured; grey = unobserved)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def save_stage(path, comp, masks, scores, classes, sources, members):
    np.savez_compressed(
        path,
        masks=masks,
        scores=np.asarray(scores),
        labels=np.asarray(classes),
        validity=comp["valid"],
        source_crop_ids=(
            np.asarray(sources, dtype=object).astype(str) if sources else np.zeros(0, str)
        ),
        members=np.asarray([json.dumps(m) for m in members], dtype=str),
    )


# --------------------------------------------------------------------------- main


def main():
    began = time.perf_counter()
    timing = {}
    latest = load(ROOT / "outputs/reports/phase5_10/latest.json")
    out = Path(latest["report_dir"])
    assert sha256(out / "protocol.json") == latest["protocol_sha256"]
    protocol = load(out / "protocol.json")
    hashes = load(out / "input_hash_manifest.json")["sha256"]
    assert sha256(out / "input_hash_manifest.json") == protocol["input_hash_manifest_sha256"]
    tick = time.perf_counter()
    changed = [p for p, h in hashes.items() if sha256(p) != h]
    assert not changed, changed
    timing["input_hash_verification_seconds"] = time.perf_counter() - tick
    (out / "artifacts").mkdir(exist_ok=True)
    (out / "galleries").mkdir(exist_ok=True)

    tick = time.perf_counter()
    p57 = load(ROOT / "outputs/reports/phase5_7_expanded/latest.json")
    p59 = load(ROOT / "outputs/reports/phase5_9/latest.json")
    p55 = load(ROOT / "outputs/reports/phase5_5/latest.json")
    manifest = load(Path(p57["dataset"]) / "metadata/dataset_manifest.json")
    p54 = load(
        Path(load(Path(p55["dataset"]) / "metadata/dataset_manifest.json")["parent_dataset"])
        / "metadata/dataset_manifest.json"
    )
    roles = protocol["data_roles"]
    legacy = legacy_component(manifest, p54, roles["LEGACY_POSITIVE_DEV"]["crops"])
    report59, dataset59 = Path(p59["report_dir"]), Path(p59["dataset"])
    pos_spec = load(
        ROOT
        / "outputs/reports/phase5_8_actual_review/20261106T115446528985Z/new_region_candidates.json"
    )
    pos_tile = next(
        t for t in pos_spec["tiles"] if t["candidate_id"] == "PHASE58_POSITIVE_DEV_REVIEW"
    )
    eligible_ids = load(report59 / "native_crop_and_visibility_audit.json")["eligible_ids"]
    new = tile_component(
        "NEW_POSITIVE_DEV",
        pos_tile,
        load(dataset59 / "manifest.json")["crops"],
        report59 / "inference",
        dataset59 / "PHASE58_POSITIVE_DEV_REVIEW_ground_truth.npz",
        eligible_ids,
    )
    neg_spec = load(
        ROOT
        / "outputs/reports/phase5_7_new_review/20261106T105319835838Z/new_region_candidates.json"
    )
    neg_tile = next(t for t in neg_spec["tiles"] if t["candidate_id"] == "NEW_DEV_REVIEW")
    neg_entries = [
        c for c in manifest["crops"] if c["crop_id"] in roles["SPARSE_NEGATIVE_CONTROL"]["crops"]
    ]
    sparse = tile_component("SPARSE_NEGATIVE_CONTROL", neg_tile, neg_entries, SAVED)
    comps = [legacy, new, sparse]
    for comp in comps:
        comp["skeleton"] = (
            skeletonize(comp["targets"].any(0)) & comp["valid"] if len(comp["targets"]) else None
        )
    timing["component_build_seconds"] = time.perf_counter() - tick

    tick = time.perf_counter()
    for comp in comps:
        comp["overlaps"] = overlap_matrix(comp)
        comp["iou"] = packed_iou(comp["masks"], comp["targets"], comp["valid"]).reshape(
            len(comp["scores"]), len(comp["target_labels"])
        )
        comp["inter"] = intersections(comp["masks"], comp["targets"], comp["valid"])
    timing["cached_iou_seconds"] = time.perf_counter() - tick

    def run_config(comp, config):
        sel = filter_geographic(
            comp["scores"],
            comp["classes"],
            comp["qualities"],
            comp["overlaps"],
            comp["crop_ids"],
            config,
        )
        local = []
        for crop in comp["crops"]:
            idx = [i for i, c in enumerate(comp["crop_ids"]) if c == crop["crop_id"]]
            part = filter_geographic(
                comp["scores"][idx],
                comp["classes"][idx],
                comp["qualities"][idx],
                comp["overlaps"][np.ix_(idx, idx)],
                [crop["crop_id"]] * len(idx),
                config,
            )
            local.append((crop, [idx[k] for k in part]))
        metrics = score(
            comp,
            comp["masks"][sel],
            comp["scores"][sel],
            comp["classes"][sel],
            comp["iou"][sel],
            comp["inter"][sel],
        )
        metrics["global_nms_suppressed"] = sum(len(v) for _, v in local) - len(sel)
        return sel, local, metrics

    # ---- baseline reproduction (stop on failure)
    frozen = protocol["baseline_reproduction"]["configuration"]
    reproduction = {}
    baseline = {}
    for comp in comps:
        sel, local, metrics = run_config(comp, frozen)
        saved_match = all(
            sorted(comp["local_index"][i] for i in idx) == sorted(crop["saved_selected"])
            for crop, idx in local
        )
        baseline[comp["name"]] = (sel, metrics)
        reproduction[comp["name"]] = {
            "per_crop_selection_matches_saved_selected": saved_match,
            "geographic": public(metrics),
            "crop_appearance": (
                crop_appearance(comp, frozen, comp["overlaps"]) if comp["positive"] else None
            ),
        }
    req = protocol["baseline_reproduction"]["required_NEW_POSITIVE_DEV"]
    nm = baseline["NEW_POSITIVE_DEV"][1]
    checks = {
        "new_counts": all(
            nm["per_class"][CLASS_NAMES[int(c)]][k] == req[c][k]
            for c in ("0", "1")
            for k in ("tp", "fp", "fn")
        ),
        "new_suppressed": nm["global_nms_suppressed"] == req["global_nms_suppressed"],
        "new_post_nms": nm["predictions"] == req["post_nms_predictions"],
        "new_foreground_iou": abs(nm["foreground_iou"] - req["foreground_iou"]) < 1e-9,
        "saved_per_crop_selection": all(
            r["per_crop_selection_matches_saved_selected"] for r in reproduction.values()
        ),
    }
    for name, expected in protocol["baseline_reproduction"][
        "required_crop_appearance_macro_f1"
    ].items():
        checks[f"{name}_crop_appearance_macro_f1"] = (
            abs(reproduction[name]["crop_appearance"]["macro_f1"] - expected) < 1e-6
        )
    # Independent cross-check against the Phase 5.9 saved global prediction masks.
    with np.load(report59 / "PHASE58_POSITIVE_DEV_REVIEW_global_predictions.npz") as old:
        old_masks = old["masks"].astype(bool)
    sel = baseline["NEW_POSITIVE_DEV"][0]
    checks["new_masks_identical_to_phase5_9"] = len(old_masks) == len(sel) and all(
        np.array_equal(a, b) for a, b in zip(old_masks, new["masks"][sel], strict=True)
    )
    dump(
        out / "baseline_reproduction.json",
        {
            "checks": checks,
            "components": reproduction,
            "consistency": {c["name"]: c["consistency"] for c in comps},
        },
    )
    if not all(checks.values()):
        raise SystemExit(f"BASELINE REPRODUCTION FAILED: {checks}")

    # ---- predeclared search
    tick = time.perf_counter()
    positive = [c for c in comps if c["positive"]]
    base_pos = {c["name"]: baseline[c["name"]][1] for c in positive}
    base_neg = baseline["SPARSE_NEGATIVE_CONTROL"][1]
    base_obj = objective(base_pos, base_neg)
    rows = []
    for k, config in enumerate(protocol["search"]["grid"]):
        cand = {c["name"]: run_config(c, config)[2] for c in positive}
        neg = run_config(sparse, config)[2]
        ok, reasons = feasible(cand, base_pos, neg, base_neg)
        obj = objective(cand, neg)
        rows.append(
            {
                "index": k,
                "config": config,
                "is_frozen": config == frozen,
                "feasible": ok,
                "violations": reasons,
                "objective": {
                    "worst_macro_f1": obj[0],
                    "mean_macro_f1": obj[1],
                    "total_fp": obj[2],
                },
                "components": {
                    n: {
                        "per_class": m["per_class"],
                        "macro_f1": m["macro_f1"],
                        "foreground_iou": m["foreground_iou"],
                        "pixel_all": m["pixel"]["all_supported_paint"],
                        "ap": m.get("ap"),
                        "predictions": m["predictions"],
                    }
                    for n, m in {**cand, "SPARSE_NEGATIVE_CONTROL": neg}.items()
                },
            }
        )
    timing["search_seconds"] = time.perf_counter() - tick

    def rank(row):
        c = row["config"]
        o = row["objective"]
        return (
            -o["worst_macro_f1"],
            -o["mean_macro_f1"],
            o["total_fp"],
            not row["is_frozen"],
            -c["confidence"]["0"],
            -c["confidence"]["1"],
            -c["mask_quality"],
            c["nms_iou"],
        )

    candidates = sorted([r for r in rows if r["feasible"]], key=rank)
    best = candidates[0] if candidates else None
    improved = (
        best is not None
        and (best["objective"]["worst_macro_f1"], best["objective"]["mean_macro_f1"]) > base_obj[:2]
    )
    chosen = best["config"] if improved else frozen
    selection = {
        "baseline_objective": {
            "worst_macro_f1": base_obj[0],
            "mean_macro_f1": base_obj[1],
            "total_fp": base_obj[2],
        },
        "feasible_count": len(candidates),
        "best_feasible_index": best["index"] if best else None,
        "strict_improvement": bool(improved),
        "selected_config": chosen,
        "selected_is_frozen": chosen == frozen,
        "ranking_feasible": [r["index"] for r in candidates],
    }
    dump(out / "search_results.json", {"rows": rows, "selection": selection})

    # ---- stage comparison, reconciliation, artifacts
    results, events_all = {}, {}
    tick_recon = 0.0
    for comp in comps:
        name = comp["name"]
        n = len(comp["scores"])
        stages = {}
        raw_idx = list(range(n))
        stages["raw"] = (
            raw_idx,
            score(comp, comp["masks"], comp["scores"], comp["classes"], comp["iou"], comp["inter"]),
        )
        f_sel, _, f_m = run_config(comp, frozen)
        s_sel, _, s_m = run_config(comp, chosen)
        stages["frozen"] = (f_sel, f_m)
        stages["selected"] = (s_sel, s_m)
        stage_arrays = {}
        for key in ("raw", "frozen", "selected"):
            idx = stages[key][0]
            stage_arrays[key] = (
                comp["masks"][idx],
                comp["scores"][idx],
                comp["classes"][idx],
                [comp["crop_ids"][i] for i in idx],
                [[f"{comp['crop_ids'][i]}#{comp['local_index'][i]}"] for i in idx],
            )
        crop_lookup = {c["crop_id"]: c for c in comp["crops"]}
        for key, base_key in (("selected_reconciled", "selected"), ("frozen_reconciled", "frozen")):
            t0 = time.perf_counter()
            props = []
            for i in stages[base_key][0]:
                crop = crop_lookup[comp["crop_ids"][i]]
                r, c = crop["row"], crop["col"]
                ev = np.zeros(comp["shape"], bool)
                h, w = crop["observed_local"].shape
                ev[r : r + h, c : c + w] = (
                    crop["raw"][comp["local_index"][i]] & crop["observed_local"]
                )
                props.append(
                    {
                        "mask": comp["masks"][i],
                        "evidence": ev,
                        "support": crop["support"],
                        "class": int(comp["classes"][i]),
                        "score": float(comp["scores"][i]),
                        "crops": {crop["crop_id"]},
                        "members": [f"{crop['crop_id']}#{comp['local_index'][i]}"],
                    }
                )
            refined, events = reconcile_supported(props, comp["observed"], comp["image"])
            tick_recon += time.perf_counter() - t0
            masks = (
                np.stack([p["mask"] for p in refined])
                if refined
                else np.zeros((0, *comp["shape"]), bool)
            )
            scores = np.asarray([p["score"] for p in refined])
            classes = np.asarray([p["class"] for p in refined], int)
            before = stage_arrays[base_key][0]
            assert np.array_equal(
                masks.any(0) if len(masks) else np.zeros(comp["shape"], bool),
                before.any(0) if len(before) else np.zeros(comp["shape"], bool),
            ), "pixels changed"
            assert not any(
                p["class"] == 1 and len(p["members"]) > 1 for p in refined
            ), "dash unioned"
            metrics = score(comp, masks, scores, classes)
            metrics["reconciliation_events"] = len(events)
            stages[key] = (None, metrics)
            stage_arrays[key] = (
                masks,
                scores,
                classes,
                [sorted(p["crops"]) for p in refined],
                [p["members"] for p in refined],
            )
            member_mask = {pp["members"][0]: pp["mask"] for pp in props}
            events_all.setdefault(name, {})[key] = [
                {
                    "members": e["members"],
                    "absorbed": e["absorbed"],
                    "reason": e["reason"],
                    "stage_masks": np.stack([member_mask[m] for m in e["members"]]),
                }
                for e in events
            ]
        for key, (masks, scores, classes, sources, members) in stage_arrays.items():
            save_stage(
                out / "artifacts" / f"{name}_{key}.npz",
                comp,
                masks,
                scores,
                classes,
                [";".join(s) if isinstance(s, list) else s for s in sources],
                members,
            )
        results[name] = {k: public(v[1]) for k, v in stages.items()}
        if comp["positive"]:
            results[name]["crop_appearance"] = {
                "frozen": crop_appearance(comp, frozen, comp["overlaps"]),
                "selected": crop_appearance(comp, chosen, comp["overlaps"]),
            }
            gallery(
                comp,
                [
                    (k, stage_arrays[k][0], stage_arrays[k][2], stages[k][1])
                    for k in ("raw", "frozen", "selected", "selected_reconciled")
                ],
                out / "galleries" / f"{name}_raw_frozen_selected_refined.png",
            )
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax, key in zip(axes, ("raw", "frozen", "selected"), strict=True):
                masks, _, classes, _, _ = stage_arrays[key]
                ax.imshow(
                    overlay(
                        comp["image"],
                        [
                            (
                                masks[classes == c].any(0) if (classes == c).any() else None,
                                COLORS[c],
                            )
                            for c in (0, 1)
                        ]
                        + [(~comp["valid"], (0.4, 0, 0.4))],
                        0.5,
                    )
                )
                ax.set_title(f"{name} {key}: {len(masks)} predictions (all are FP)", fontsize=9)
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(out / "galleries" / f"{name}_negative_control.png", dpi=110)
            plt.close(fig)
        for key in ("selected_reconciled", "frozen_reconciled"):
            event_gallery(
                comp, events_all[name][key], out / "galleries" / f"{name}_{key}_events.png"
            )
        results[name]["inputs"] = {
            "crops": [c["crop_id"] for c in comp["crops"]],
            "saved_raw_proposals": n,
            "unique_source_ids": len(comp["ids"]),
            "scored_valid_pixels": int(comp["valid"].sum()),
            "raw_observed_pixels": int(comp["observed"].sum()),
            "canvas_shape": list(comp["shape"]),
            "consistency": comp["consistency"],
        }
    timing["reconciliation_seconds"] = tick_recon
    events_json = {
        n: {
            k: [{kk: vv for kk, vv in e.items() if kk != "stage_masks"} for e in v]
            for k, v in d.items()
        }
        for n, d in events_all.items()
    }
    dump(out / "reconciliation_events.json", events_json)

    # ---- reconciliation adoption (predeclared criterion)
    def adoption(base_key, recon_key):
        strict = False
        for comp in comps:
            b, r = results[comp["name"]][base_key], results[comp["name"]][recon_key]
            for k in ("solid", "dash"):
                if (
                    r["per_class"][k]["tp"] < b["per_class"][k]["tp"]
                    or r["per_class"][k]["fp"] > b["per_class"][k]["fp"]
                ):
                    return False, "violates TP non-decrease / FP non-increase"
                strict |= (
                    r["per_class"][k]["tp"] > b["per_class"][k]["tp"]
                    or r["per_class"][k]["fp"] < b["per_class"][k]["fp"]
                )
            if comp["positive"]:
                strict |= (
                    r["solid_fragmentation"]["overlapping_solid_masks_total"]
                    < b["solid_fragmentation"]["overlapping_solid_masks_total"]
                )
        return strict, "strict improvement" if strict else "no strict improvement"

    adopt_sel = adoption("selected", "selected_reconciled")
    adopt_frozen = adoption("frozen", "frozen_reconciled")
    for path, expected in hashes.items():
        assert sha256(path) == expected, path
    timing["total_seconds"] = time.perf_counter() - began
    final = {
        "phase": "5.10",
        "run_id": protocol["run_id"],
        "protocol_sha256": latest["protocol_sha256"],
        "development_only": True,
        "protected_final_test_used": False,
        "neural_inference_performed": False,
        "weights_modified": False,
        "baseline_reproduction": checks,
        "selection": selection,
        "selected_postprocessing": chosen,
        "reconciliation_adoption": {
            "selected": {"adopt": adopt_sel[0], "reason": adopt_sel[1]},
            "frozen_diagnostic": {"adopt": adopt_frozen[0], "reason": adopt_frozen[1]},
        },
        "components": results,
        "runtime_seconds": timing,
        "inputs_reverified_after_run": True,
    }
    dump(out / "results.json", final)
    dump(ROOT / "outputs/reports/phase5_10/latest.json", {**latest, "status": "EVALUATED"})
    print(
        json.dumps(
            {
                "checks": checks,
                "selection": selection,
                "adoption": final["reconciliation_adoption"],
                "timing": timing,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
