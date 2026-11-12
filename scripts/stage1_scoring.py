"""Stage 1: score/selection study (1a) and weight averaging (1b). No training.

1a  Score variants recomputed from saved per-query logits of the DEFAULT model, fitted and
    evaluated with cross-fitting over the 26 P511/P512 regions the default never trained on.
1b  SWA and WiSE-FT weight averages, evaluated on the development areas only (descriptive:
    those models trained on P511/P512 regions).
The protocol is written once, before any variant is scored, and re-verified on every run.
"""

import gc
import json
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from scipy.special import logit
from skimage.morphology import skeletonize
from sklearn.linear_model import LogisticRegression
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

import phase5_10_recovery as rec
from phase5_11_common import ROOT, load
from phase5_11_evaluate import components
from roadmarking.dataset.phase55 import write_attestation
from roadmarking.evaluation.harness import (
    block_records,
    greedy_flags,
    paired_block_bootstrap,
    summarize,
)
from roadmarking.inference.prediction_recovery import filter_geographic, packed_iou
from roadmarking.rasterization.pipeline import dump, sha256
from roadmarking.training.expanded_continuous import evaluate as infer

OUT = ROOT / "outputs/reports/stage1_scoring"
DEFAULT = ROOT / "models/mask2former_swin_s/phase5_7_expanded_20261106T111240383727Z"
FROZEN = {"confidence": {"0": 0.85, "1": 0.55}, "mask_quality": 0.8, "nms_iou": 0.3}
LOST = "4b28eb52fa431a19bf87cd0a9314c37a7967d31fa3ed5f4e49f7454bfdec084b"
VARIANTS = [
    "V0_frozen",
    "V1_refit_thresholds",
    "V2_top50_quality",
    "V3_skeleton_quality",
    "V4_class_only",
    "V5_platt_calibrated",
]
SIMPLICITY = [
    "V1_refit_thresholds",
    "V3_skeleton_quality",
    "V2_top50_quality",
    "V4_class_only",
    "V5_platt_calibrated",
]
P_FLOOR = 0.25


def protocol():
    path = OUT / "protocol.json"
    if path.exists():
        return load(path), sha256(path)
    body = {
        "stage": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "PREDECLARED_BEFORE_ANY_VARIANT_SCORED",
        "pool": "26 P511/P512 training regions unseen by the default step-50 model; 5-fold "
        "cross-fitting using stage0 folds (fit on 4 folds, score the 5th)",
        "variants": {
            "V0_frozen": "class prob x mean sigmoid over logit>0; quality>=.8 gate; solid .85 / "
            "dash .55; proposals require score>=.25 (current pipeline)",
            "V1_refit_thresholds": "V0 score, no quality gate, per-class threshold fitted",
            "V2_top50_quality": "class prob x mean of the top 50% mask sigmoids; fitted",
            "V3_skeleton_quality": "class prob x mean sigmoid on the mask skeleton; fitted",
            "V4_class_only": "class prob only; fitted",
            "V5_platt_calibrated": "per-class logistic on logit(p), logit(q_mean), "
            "logit(q_top50), logit(q_skeleton), log(area); fitted",
        },
        "proposal_pool_for_V1_V5": f"non-no-object queries, >=4 valid px, class prob >= {P_FLOOR}",
        "threshold_rule": "per class, the score maximising crop-appearance F1 (IoU>=.5, after "
        "per-crop same-class NMS .3) on the 4 fitting folds; ties -> higher",
        "dev_areas": "thresholds/calibrators refit on ALL 26 pool regions, then applied once to "
        "LEGACY/PHASE58/negative control (never fitted on dev)",
        "bootstrap": "paired, 2000 resamples; units = regions (pool) or 256-px blocks (dev)",
        "gates_variant_vs_V0": {
            "G1_primary": "pooled crop-appearance solid AP50 difference CI lower bound > 0",
            "G2": "pooled solid buffer completeness difference CI lower bound > -0.02",
            "G3": "pooled crop-appearance dash AP50 difference CI lower bound > -0.03",
            "G4": "pooled geographic foreground IoU difference CI lower bound > -0.02",
            "G5": "negative-control FP/km2 <= V0",
            "G6": "legacy lost-edge paint coverage >= 0.60",
        },
        "choice": f"among variants passing all gates, the first in simplicity order {SIMPLICITY}",
        "part_1b": {
            "candidates": [
                "P512_step400",
                "SWA_P512_steps300-600",
                "WiSE_FT_alpha0.3",
                "WiSE_FT_alpha0.5",
                "WiSE_FT_alpha0.7",
            ],
            "wise_ft": "theta = (1-alpha) * default + alpha * P512_step400 (same COCO lineage)",
            "scoring": "V0 and the chosen 1a variant (V1 if none passes), thresholds fitted on "
            "the default's pool (not refit per candidate)",
            "evaluation": "development areas only; descriptive because candidates trained on "
            "the pool regions",
            "advance_to_stage2_if": "pooled-dev FG IoU diff CI lower > -0.02 AND new-area FG "
            "IoU diff CI lower > 0 AND lost-edge coverage >= .60 AND "
            "negative FP/km2 <= default",
        },
        "protected_final_test": "sealed (stage0_harness/protected_test_freeze.json)",
    }
    write_attestation(path, body)
    return body, sha256(path)


# --------------------------------------------------------------------------- features


def crop_features(npz_path, validity):
    """All candidate queries of one crop with score factors (masks at native 512)."""
    with np.load(npz_path) as d:
        cl = torch.from_numpy(d["class_logits"])[0].softmax(-1).numpy()
        logits = torch.nn.functional.interpolate(
            torch.from_numpy(d["mask_logits"]),
            size=validity.shape,
            mode="bilinear",
            align_corners=False,
        )[0].numpy()
    prob = 1 / (1 + np.exp(-logits))
    rows = []
    for q in range(len(cl)):
        c = int(cl[q].argmax())
        if c == 2 or cl[q, c] < P_FLOOR:
            continue
        m = (logits[q] > 0) & validity
        if m.sum() < 4:
            continue
        vals = prob[q][m]
        top = np.sort(vals)[len(vals) // 2 :]
        sk = skeletonize(m)
        rows.append(
            {
                "mask": m,
                "class": c,
                "p": float(cl[q, c]),
                "q_mean": float(vals.mean()),
                "q_top": float(top.mean()),
                "q_skel": float(prob[q][sk].mean()) if sk.any() else float(vals.mean()),
                "area": int(m.sum()),
            }
        )
    return rows


def variant_score(rows, variant, calibrator=None):
    if not rows:
        return np.zeros(0)
    p = np.array([r["p"] for r in rows])
    if variant in ("V0_frozen", "V1_refit_thresholds"):
        return p * np.array([r["q_mean"] for r in rows])
    if variant == "V2_top50_quality":
        return p * np.array([r["q_top"] for r in rows])
    if variant == "V3_skeleton_quality":
        return p * np.array([r["q_skel"] for r in rows])
    if variant == "V4_class_only":
        return p
    x = features_matrix(rows)
    cls = np.array([r["class"] for r in rows])
    out = np.zeros(len(rows))
    for c in (0, 1):
        sel = cls == c
        if sel.any() and calibrator.get(c) is not None:
            out[sel] = calibrator[c].predict_proba(x[sel])[:, 1]
    return out


def features_matrix(rows):
    eps = 1e-4

    def lg(v):
        return logit(np.clip(np.asarray(v), eps, 1 - eps))

    return np.column_stack(
        [
            lg([r["p"] for r in rows]),
            lg([r["q_mean"] for r in rows]),
            lg([r["q_top"] for r in rows]),
            lg([r["q_skel"] for r in rows]),
            np.log([r["area"] for r in rows]),
        ]
    )


def crop_nms(rows, scores, iou_thr=0.3):
    order = sorted(range(len(rows)), key=lambda i: (-scores[i], i))
    kept = []
    packs = {i: np.packbits(rows[i]["mask"]) for i in order}
    area = {i: rows[i]["area"] for i in order}
    for i in order:
        ok = True
        for j in kept:
            if rows[j]["class"] != rows[i]["class"]:
                continue
            inter = int(np.bitwise_count(np.bitwise_and(packs[i], packs[j])).sum())
            if inter / max(1, area[i] + area[j] - inter) > iou_thr:
                ok = False
                break
        if ok:
            kept.append(i)
    return kept


# --------------------------------------------------------------------------- data


def load_pool(manifest_crops, folder):
    """Per crop: features + crop-level IoU vs crop targets."""
    crops = {}
    for c in manifest_crops:
        with np.load(c["file"]) as d:
            v = d["validity"].astype(bool)
            t = d["masks"].astype(bool) & v
            tl = d["labels"].astype(int)
        rows = crop_features(Path(folder) / (c["crop_id"] + ".npz"), v)
        iou = (
            packed_iou([r["mask"] for r in rows], t, v).reshape(len(rows), len(t))
            if rows
            else np.zeros((0, len(t)))
        )
        crops[c["crop_id"]] = {
            "crop": c,
            "rows": rows,
            "iou": iou,
            "tl": tl,
            "valid": v,
            "targets": t,
        }
    return crops


def crop_level(crop, scores, kept):
    """Return per-class lists (score, greedy TP flag) over NMS-kept proposals, and GT count."""
    out = {}
    classes = np.array([r["class"] for r in crop["rows"]], int)
    kept = np.array(kept, int)
    for c in (0, 1):
        sel = kept[classes[kept] == c] if len(kept) else kept
        flags, _ = greedy_flags(
            crop["iou"][sel] if len(sel) else np.zeros((0, len(crop["tl"]))),
            scores[sel],
            classes[sel],
            crop["tl"],
        )
        out[c] = (
            list(zip(scores[sel].tolist(), flags.tolist(), strict=True)),
            int((crop["tl"] == c).sum()),
        )
    return out


def fit_thresholds(crops, variant, calibrator=None):
    """Per-class threshold maximising pooled crop-appearance F1 (prefix property of NMS)."""
    pooled = {0: [], 1: []}
    gts = {0: 0, 1: 0}
    for crop in crops:
        s = variant_score(crop["rows"], variant, calibrator)
        kept = crop_nms(crop["rows"], s)
        for c, (lst, n) in crop_level(crop, s, kept).items():
            pooled[c] += lst
            gts[c] += n
    thr = {}
    for c in (0, 1):
        lst = sorted(pooled[c], key=lambda x: -x[0])
        best = (-1, 1.0)
        tp = 0
        for k, (score, flag) in enumerate(lst):
            tp += flag
            prec, recl = tp / (k + 1), tp / max(1, gts[c])
            f1 = 2 * prec * recl / max(1e-12, prec + recl)
            if f1 > best[0] + 1e-12:
                best = (f1, score)
        thr[str(c)] = float(best[1])
    return thr


def fit_calibrator(crops):
    rows, labels = [], []
    for crop in crops:
        for i, r in enumerate(crop["rows"]):
            same = crop["tl"] == r["class"]
            rows.append(r)
            labels.append(int(same.any() and crop["iou"][i, same].max() >= 0.5))
    x = features_matrix(rows)
    cls = np.array([r["class"] for r in rows])
    y = np.array(labels)
    cal = {}
    for c in (0, 1):
        sel = cls == c
        cal[c] = (
            LogisticRegression(C=1.0, max_iter=1000).fit(x[sel], y[sel])
            if sel.sum() > 10 and len(set(y[sel])) == 2
            else None
        )
    return cal


def config_for(variant, thr):
    if variant == "V0_frozen":
        return FROZEN
    return {"confidence": thr, "mask_quality": 0.0, "nms_iou": 0.3}


def region_component(name, crops, scores_by_crop):
    """Geographic component from in-memory proposals (variant scores) of a region's crops."""
    cs = [c["crop"] for c in crops]
    x0 = min(c["affine"][2] for c in cs)
    y1 = max(c["affine"][5] for c in cs)
    pos = {
        c["crop_id"]: (round((y1 - c["affine"][5]) / 0.05), round((c["affine"][2] - x0) / 0.05))
        for c in cs
    }
    shape = (max(r for r, _ in pos.values()) + 512, max(c for _, c in pos.values()) + 512)
    valid = np.zeros(shape, bool)
    truths, labels = {}, {}
    masks, scores, classes, quals, crop_ids = [], [], [], [], []
    for crop in crops:
        c = crop["crop"]
        r0, c0 = pos[c["crop_id"]]
        sl = (slice(r0, r0 + 512), slice(c0, c0 + 512))
        valid[sl] |= crop["valid"]
        for m, lab, uid in zip(crop["targets"], crop["tl"], c["global_instance_ids"], strict=True):
            truths.setdefault(uid, np.zeros(shape, bool))[sl] |= m
            labels[uid] = int(lab)
        for k, row in enumerate(crop["rows"]):
            g = np.zeros(shape, bool)
            g[sl] = row["mask"]
            masks.append(g)
            scores.append(float(scores_by_crop[c["crop_id"]][k]))
            classes.append(row["class"])
            quals.append(row["q_mean"])
            crop_ids.append(c["crop_id"])
    ids = sorted(truths)
    comp = {
        "name": name,
        "shape": shape,
        "valid": valid,
        "ids": ids,
        "crops": [{**c, "row": pos[c["crop_id"]][0], "col": pos[c["crop_id"]][1]} for c in cs],
        "targets": np.stack([truths[u] for u in ids]) if ids else np.zeros((0, *shape), bool),
        "target_labels": np.asarray([labels[u] for u in ids], int),
        "masks": np.stack(masks) if masks else np.zeros((0, *shape), bool),
        "scores": np.asarray(scores),
        "classes": np.asarray(classes, int),
        "qualities": np.asarray(quals),
        "crop_ids": crop_ids,
    }
    comp["overlaps"] = rec.overlap_matrix(comp)
    comp["iou"] = packed_iou(comp["masks"], comp["targets"], comp["valid"]).reshape(
        len(scores), len(ids)
    )
    return comp


def v0_mask(crop_rows):
    """V0 proposal set = the current pipeline's (score >= .25 floor)."""
    return [r["p"] * r["q_mean"] >= 0.25 for r in crop_rows]


def evaluate_region(name, crops, variant, thr, calibrator):
    scores_by_crop, region_crops = {}, []
    crop_records = defaultdict(float)
    crop_dets = []
    for crop in crops:
        rows = crop["rows"]
        if variant == "V0_frozen":
            keep_rows = [r for r, k in zip(rows, v0_mask(rows), strict=True) if k]
            idx = [i for i, k in enumerate(v0_mask(rows)) if k]
            crop = {
                **crop,
                "rows": keep_rows,
                "iou": crop["iou"][idx] if idx else np.zeros((0, len(crop["tl"]))),
            }
        s = variant_score(crop["rows"], variant, calibrator)
        kept = crop_nms(crop["rows"], s)
        if variant == "V0_frozen":
            kept_op = [
                i
                for i in kept
                if s[i] >= FROZEN["confidence"][str(crop["rows"][i]["class"])]
                and crop["rows"][i]["q_mean"] >= FROZEN["mask_quality"]
            ]
        else:
            kept_op = [i for i in kept if s[i] >= thr[str(crop["rows"][i]["class"])]]
        cl = crop_level(crop, s, kept)
        for c, (lst, n) in cl.items():
            name_c = rec.CLASS_NAMES[c]
            crop_records[f"{name_c}_gt"] += n
            crop_dets += [(name_c, sc, f) for sc, f in lst]
        op = crop_level(crop, s, kept_op)
        for c, (lst, n) in op.items():
            name_c = rec.CLASS_NAMES[c]
            tp = sum(f for _, f in lst)
            crop_records[f"{name_c}_tp"] += tp
            crop_records[f"{name_c}_fp"] += len(lst) - tp
            crop_records[f"{name_c}_fn"] += n - tp
        scores_by_crop[crop["crop"]["crop_id"]] = s
        region_crops.append(crop)
    comp = region_component(name, region_crops, scores_by_crop)
    cfg = config_for(variant, thr)
    keep = filter_geographic(
        comp["scores"], comp["classes"], comp["qualities"], comp["overlaps"], comp["crop_ids"], cfg
    )
    geo_recs, geo_dets = block_records(
        comp, comp["masks"], comp["scores"], comp["classes"], keep, comp["iou"], block=256
    )
    geo = defaultdict(float)
    for r in geo_recs.values():
        for k, v in r.items():
            geo[k] += v
    return dict(crop_records), crop_dets, dict(geo), [d for v in geo_dets.values() for d in v]


# --------------------------------------------------------------------------- 1b models


def averaged_model(name, weights):
    """Write an averaged checkpoint (weights: list of (folder, coefficient))."""
    target = ROOT / "models/mask2former_swin_s/stage1_averaged" / name
    if (target / "model/model.safetensors").exists():
        return target
    state = None
    for folder, w in weights:
        sd = load_file(str(Path(folder) / "model/model.safetensors"))
        if state is None:
            state = {k: v.float() * w for k, v in sd.items()}
        else:
            for k, v in sd.items():
                state[k] += v.float() * w
    base = Path(weights[0][0])
    (target / "model").mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.contiguous() for k, v in state.items()}, str(target / "model/model.safetensors")
    )
    for f in (base / "model").glob("*.json"):
        (target / "model" / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    import shutil

    shutil.copytree(base / "processor", target / "processor", dirs_exist_ok=True)
    dump(
        target / "bundle.json",
        {
            "weight_sha256": sha256(target / "model/model.safetensors"),
            "sources": [(str(f), w) for f, w in weights],
            "status": "STAGE1_DIAGNOSTIC_AVERAGE_NOT_PRODUCTION",
        },
    )
    return target


def run_inference(model_dir, dataset, crops, target):
    if (Path(target) / "metrics.json").exists():
        return
    model = (
        Mask2FormerForUniversalSegmentation.from_pretrained(
            Path(model_dir) / "model", local_files_only=True
        )
        .to("cuda")
        .eval()
    )
    processor = Mask2FormerImageProcessor.from_pretrained(
        Path(model_dir) / "processor", local_files_only=True
    )
    infer(model, processor, dataset, crops, FROZEN, target)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def write_variant_folder(src_folder, crops_list, variant, thr, calibrator, dest):
    """Prediction folder for the dev-area component builders (variant scores, no q gate)."""
    dest.mkdir(parents=True, exist_ok=True)
    for c in crops_list:
        with np.load(Path(src_folder) / (c["crop_id"] + ".npz")) as d:
            v = d["validity"].astype(bool)
        rows = crop_features(Path(src_folder) / (c["crop_id"] + ".npz"), v)
        if variant == "V0_frozen":
            rows = [r for r, k in zip(rows, v0_mask(rows), strict=True) if k]
        s = variant_score(rows, variant, calibrator)
        np.savez_compressed(
            dest / (c["crop_id"] + ".npz"),
            masks=np.stack([r["mask"] for r in rows]) if rows else np.zeros((0, *v.shape), bool),
            scores=s,
            labels=np.array([r["class"] for r in rows], int),
            qualities=np.array([r["q_mean"] if variant == "V0_frozen" else 1.0 for r in rows]),
            selected=np.zeros(0, int),
            validity=v,
        )


def dev_eval(leg_folder, new_folder, variant, thr, calibrator, tag):
    p57 = load(ROOT / "outputs/reports/phase5_7_expanded/latest.json")
    m57 = load(Path(p57["dataset"]) / "metadata/dataset_manifest.json")
    p59 = load(ROOT / "outputs/reports/phase5_9/latest.json")
    m59 = load(Path(p59["dataset"]) / "manifest.json")
    base = OUT / "dev_variants" / tag / variant
    write_variant_folder(
        leg_folder,
        [c for c in m57["crops"] if c["split"] == "val"],
        variant,
        thr,
        calibrator,
        base / "legacy",
    )
    write_variant_folder(new_folder, m59["crops"], variant, thr, calibrator, base / "new")
    comps = components(base / "legacy", base / "new")
    cfg = config_for(variant, thr)
    res = {}
    for comp in comps:
        keep = filter_geographic(
            comp["scores"],
            comp["classes"],
            comp["qualities"],
            comp["overlaps"],
            comp["crop_ids"],
            cfg,
        )
        recs, dets = block_records(
            comp, comp["masks"], comp["scores"], comp["classes"], keep, comp["iou"], block=256
        )
        area = float(comp["valid"].sum()) * 0.0025 / 1e6
        entry = {
            "records": recs,
            "detections": dets,
            "area_km2": area,
            "summary": summarize(recs.values(), dets.values(), area),
        }
        if LOST in comp["ids"]:
            j = comp["ids"].index(LOST)
            solid = comp["masks"][keep][comp["classes"][keep] == 0] if keep else []
            union = solid.any(0) if len(solid) else np.zeros(comp["shape"], bool)
            entry["lost_edge_coverage"] = float(
                (union & comp["targets"][j]).sum() / comp["targets"][j].sum()
            )
        res[comp["name"]] = entry
    return res


def pooled_dev(res):
    recs, dets = {}, {}
    for area in ("LEGACY_POSITIVE_DEV", "NEW_POSITIVE_DEV"):
        for b, v in res[area]["records"].items():
            recs[f"{area}:{b}"] = v
        for b, v in res[area]["detections"].items():
            dets[f"{area}:{b}"] = v
    return recs, dets


# --------------------------------------------------------------------------- main


def main():
    began = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    _, proto_hash = protocol()
    folds = load(ROOT / "outputs/reports/stage0_harness/folds.json")["folds"]
    l12 = load(ROOT / "outputs/reports/phase5_12/latest.json")
    dataset = Path(l12["dataset"])
    manifest = load(dataset / "metadata/dataset_manifest.json")
    pool_regions = {r: f["fold"] for f in folds for r in f["regions"] if r.startswith("P51")}
    pool_crops = [
        c for c in manifest["crops"] if c["split"] == "train" and c["tile_id"] in pool_regions
    ]
    pred_folder = OUT / "default_pool_inference"
    tick = time.perf_counter()
    run_inference(DEFAULT / "checkpoint_step0050", dataset, pool_crops, pred_folder)
    infer_s = time.perf_counter() - tick
    crops = load_pool(pool_crops, pred_folder)
    by_region = defaultdict(list)
    for crop in crops.values():
        by_region[crop["crop"]["tile_id"]].append(crop)

    # ---- 1a cross-fitted evaluation on the unseen pool
    results = {
        v: {
            "crop_records": {},
            "crop_dets": {},
            "geo_records": {},
            "geo_dets": {},
            "thresholds": {},
        }
        for v in VARIANTS
    }
    for f in range(5):
        fit = [c for r, k in pool_regions.items() if k != f for c in by_region[r]]
        held = [r for r, k in pool_regions.items() if k == f]
        cal = fit_calibrator(fit)
        for v in VARIANTS:
            thr = (
                FROZEN["confidence"]
                if v == "V0_frozen"
                else fit_thresholds(fit, v, cal if v == "V5_platt_calibrated" else None)
            )
            results[v]["thresholds"][f] = thr
            for r in held:
                cr, cd, gr, gd = evaluate_region(
                    r, by_region[r], v, thr, cal if v == "V5_platt_calibrated" else None
                )
                results[v]["crop_records"][r] = cr
                results[v]["crop_dets"][r] = cd
                results[v]["geo_records"][r] = gr
                results[v]["geo_dets"][r] = gd
        print(f"fold {f} done", flush=True)

    pool_summary, pool_boot = {}, {}
    for v in VARIANTS:
        pool_summary[v] = {
            "crop_appearance": summarize(
                results[v]["crop_records"].values(), results[v]["crop_dets"].values()
            ),
            "geographic": summarize(
                results[v]["geo_records"].values(), results[v]["geo_dets"].values()
            ),
            "thresholds_by_fold": results[v]["thresholds"],
        }
    for v in VARIANTS[1:]:
        pool_boot[v] = {
            "crop_appearance": paired_block_bootstrap(
                results["V0_frozen"]["crop_records"],
                results["V0_frozen"]["crop_dets"],
                results[v]["crop_records"],
                results[v]["crop_dets"],
                n=2000,
                seed=0,
            ),
            "geographic": paired_block_bootstrap(
                results["V0_frozen"]["geo_records"],
                results["V0_frozen"]["geo_dets"],
                results[v]["geo_records"],
                results[v]["geo_dets"],
                n=2000,
                seed=0,
            ),
        }

    # ---- dev areas: refit on ALL pool regions, apply once
    all_pool = [c for r in pool_regions for c in by_region[r]]
    cal_all = fit_calibrator(all_pool)
    thr_all = {
        v: (
            FROZEN["confidence"]
            if v == "V0_frozen"
            else fit_thresholds(all_pool, v, cal_all if v == "V5_platt_calibrated" else None)
        )
        for v in VARIANTS
    }
    p59 = load(ROOT / "outputs/reports/phase5_9/latest.json")
    dev = {}
    for v in VARIANTS:
        dev[v] = dev_eval(
            DEFAULT / "validation_step0050",
            Path(p59["report_dir"]) / "inference",
            v,
            thr_all[v],
            cal_all if v == "V5_platt_calibrated" else None,
            "default",
        )
    neg_v0 = dev["V0_frozen"]["SPARSE_NEGATIVE_CONTROL"]["summary"].get("fp_per_km2", 0.0)

    # ---- gates
    gates = {}
    for v in VARIANTS[1:]:
        ca = pool_boot[v]["crop_appearance"]["metrics"]
        ge = pool_boot[v]["geographic"]["metrics"]
        g = {
            "G1_solid_ap50": ca["solid_ap50"]["ci_difference"][0] > 0,
            "G2_solid_buffer_completeness": ge["solid_buffer_completeness"]["ci_difference"][0]
            > -0.02,
            "G3_dash_ap50": ca["dash_ap50"]["ci_difference"][0] > -0.03,
            "G4_geo_fg_iou": ge["foreground_iou"]["ci_difference"][0] > -0.02,
            "G5_negative_fp": dev[v]["SPARSE_NEGATIVE_CONTROL"]["summary"].get("fp_per_km2", 0.0)
            <= neg_v0,
            "G6_lost_edge": dev[v]["LEGACY_POSITIVE_DEV"].get("lost_edge_coverage", 0) >= 0.60,
        }
        gates[v] = {"checks": g, "passes": all(g.values())}
    passing = [v for v in SIMPLICITY if gates[v]["passes"]]
    chosen = passing[0] if passing else None
    scoring_1b = chosen or "V1_refit_thresholds"

    # ---- 1b weight averaging on dev areas
    r12 = Path(l12["training_run"])
    cand_models = {
        "P512_step400": r12 / "checkpoint_step0400",
        "SWA_P512_steps300-600": averaged_model(
            "swa_p512_300_600",
            [(r12 / f"checkpoint_step{s:04d}", 0.25) for s in (300, 400, 500, 600)],
        ),
    }
    for a in (0.3, 0.5, 0.7):
        cand_models[f"WiSE_FT_alpha{a}"] = averaged_model(
            f"wise_ft_alpha{a}",
            [(DEFAULT / "checkpoint_step0050", 1 - a), (r12 / "checkpoint_step0400", a)],
        )
    p57 = load(ROOT / "outputs/reports/phase5_7_expanded/latest.json")
    m57 = load(Path(p57["dataset"]) / "metadata/dataset_manifest.json")
    m59 = load(Path(p59["dataset"]) / "manifest.json")
    dev_1b = {"default": {"V0_frozen": dev["V0_frozen"], scoring_1b: dev[scoring_1b]}}
    for name, folder in cand_models.items():
        leg = OUT / "candidate_inference" / name / "legacy"
        new = OUT / "candidate_inference" / name / "new"
        run_inference(
            folder, Path(p57["dataset"]), [c for c in m57["crops"] if c["split"] == "val"], leg
        )
        run_inference(folder, Path(p59["dataset"]), m59["crops"], new)
        dev_1b[name] = {}
        for v in ("V0_frozen", scoring_1b):
            dev_1b[name][v] = dev_eval(
                leg, new, v, thr_all[v], cal_all if v == "V5_platt_calibrated" else None, name
            )
    advance, cand_boot = {}, {}
    for name in cand_models:
        for v in ("V0_frozen", scoring_1b):
            a_rec, a_det = pooled_dev(dev_1b["default"][v])
            b_rec, b_det = pooled_dev(dev_1b[name][v])
            pooled = paired_block_bootstrap(a_rec, a_det, b_rec, b_det, n=2000, seed=0)
            newarea = paired_block_bootstrap(
                dev_1b["default"][v]["NEW_POSITIVE_DEV"]["records"],
                dev_1b["default"][v]["NEW_POSITIVE_DEV"]["detections"],
                dev_1b[name][v]["NEW_POSITIVE_DEV"]["records"],
                dev_1b[name][v]["NEW_POSITIVE_DEV"]["detections"],
                n=2000,
                seed=0,
            )
            key = f"{name}|{v}"
            cand_boot[key] = {"pooled": pooled, "new_area": newarea}
            neg = dev_1b[name][v]["SPARSE_NEGATIVE_CONTROL"]["summary"].get("fp_per_km2", 0.0)
            advance[key] = {
                "pooled_fg_iou_noninferior": pooled["metrics"]["foreground_iou"]["ci_difference"][0]
                > -0.02,
                "new_area_fg_iou_gain": newarea["metrics"]["foreground_iou"]["ci_difference"][0]
                > 0,
                "lost_edge_ge_0.60": dev_1b[name][v]["LEGACY_POSITIVE_DEV"].get(
                    "lost_edge_coverage", 0
                )
                >= 0.60,
                "negative_fp_not_worse": neg
                <= dev_1b["default"][v]["SPARSE_NEGATIVE_CONTROL"]["summary"].get(
                    "fp_per_km2", 0.0
                ),
            }
            advance[key]["advance"] = all(advance[key].values())

    def strip(d):
        return {
            a: {k: v for k, v in e.items() if k not in ("records", "detections")}
            for a, e in d.items()
        }

    out = {
        "stage": 1,
        "protocol_sha256": proto_hash,
        "pool_regions": len(pool_regions),
        "pool_crops": len(pool_crops),
        "pool_inference_seconds": infer_s,
        "pool_summary": pool_summary,
        "pool_bootstrap_vs_V0": pool_boot,
        "thresholds_all_pool": thr_all,
        "dev_default_variants": {v: strip(d) for v, d in dev.items()},
        "gates": gates,
        "chosen_variant": chosen,
        "scoring_used_for_1b": scoring_1b,
        "dev_1b": {m: {v: strip(d) for v, d in x.items()} for m, x in dev_1b.items()},
        "candidate_bootstrap_vs_default": cand_boot,
        "advance_to_stage2": advance,
        "runtime_seconds": time.perf_counter() - began,
        "training_performed": False,
        "protected_final_test_used": False,
    }
    dump(OUT / "stage1_results.json", out)
    print(
        json.dumps(
            {"gates": gates, "chosen": chosen, "advance": advance, "thresholds": thr_all}, indent=1
        )
    )


if __name__ == "__main__":
    main()
