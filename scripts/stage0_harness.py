"""Stage 0: evaluation harness — folds, frozen metric suite, block bootstrap, factor logs.

No training. Inference only for a leave-one-region-out plumbing check on TRAINING crops.
The protected final test is sealed by metadata only; its imagery/labels are never opened.
"""

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import shapely
import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

import phase5_10_recovery as rec
from phase5_11_common import EDGE, ROOT, client_features, load, reserved_geography
from phase5_11_evaluate import FROZEN, components
from roadmarking.dataset.phase55 import write_attestation
from roadmarking.evaluation.harness import (
    assign_folds,
    block_records,
    greedy_flags,
    paired_block_bootstrap,
    region_groups,
    summarize,
)
from roadmarking.inference.prediction_recovery import filter_geographic, packed_iou
from roadmarking.rasterization.pipeline import dump, sha256
from roadmarking.training.expanded_continuous import evaluate as infer

OUT = ROOT / "outputs/reports/stage0_harness"
DEFAULT = ROOT / "models/mask2former_swin_s/phase5_7_expanded_20261106T111240383727Z"
SUITE = [
    "solid_ap50",
    "dash_ap50",
    "solid_buffer_completeness",
    "solid_buffer_correctness",
    "solid_f1",
    "dash_f1",
    "macro_f1",
    "foreground_iou",
    "pixel_precision",
    "pixel_recall",
    "solid_pieces_per_covered_gt",
]
BLOCK = 256  # 12.8 m non-overlapping bootstrap blocks


def evaluate_component(comp):
    keep = filter_geographic(
        comp["scores"],
        comp["classes"],
        comp["qualities"],
        comp["overlaps"],
        comp["crop_ids"],
        FROZEN,
    )
    recs, dets = block_records(
        comp, comp["masks"], comp["scores"], comp["classes"], keep, comp["iou"], block=BLOCK
    )
    return keep, recs, dets


def factor_log(model, comp, keep, designations):
    rows = []
    flags_all, _ = greedy_flags(comp["iou"], comp["scores"], comp["classes"], comp["target_labels"])
    kept = set(keep)
    flags_op, match_op = greedy_flags(
        comp["iou"][keep] if keep else np.zeros((0, len(comp["target_labels"]))),
        comp["scores"][keep],
        comp["classes"][keep],
        comp["target_labels"],
    )
    op = {i: (int(f), int(m)) for i, f, m in zip(keep, flags_op, match_op, strict=True)}
    gt_union = comp["targets"].any(0) if len(comp["targets"]) else np.zeros(comp["shape"], bool)
    for i in range(len(comp["scores"])):
        q = float(comp["qualities"][i])
        j = int(np.argmax(comp["iou"][i])) if comp["iou"].shape[1] else -1
        rows.append(
            {
                "model": model,
                "area": comp["name"],
                "crop": comp["crop_ids"][i],
                "class": rec.CLASS_NAMES[int(comp["classes"][i])],
                "class_prob": float(comp["scores"][i]) / max(1e-9, q),
                "mask_quality": q,
                "score": float(comp["scores"][i]),
                "kept_by_frozen_filter": i in kept,
                "tp_iou50_operating": op.get(i, (0, -1))[0] if i in kept else None,
                "tp_iou50_all_proposals": int(flags_all[i]),
                "best_iou": float(comp["iou"][i, j]) if j >= 0 else 0.0,
                "on_paint_fraction": float(
                    (comp["masks"][i] & gt_union & comp["valid"]).sum()
                    / max(1, (comp["masks"][i] & comp["valid"]).sum())
                ),
                "best_gt_designation": designations.get(comp["ids"][j]) if j >= 0 else None,
            }
        )
    return rows


def factor_summary(rows):
    out = {}
    for model in sorted({r["model"] for r in rows}):
        for cls in ("solid", "dash"):
            sel = [r for r in rows if r["model"] == model and r["class"] == cls]
            for label, cond in (
                ("tp", lambda r: r["tp_iou50_all_proposals"] == 1),
                ("fp", lambda r: r["tp_iou50_all_proposals"] == 0),
                (
                    "fp_fragment_on_paint",
                    lambda r: r["tp_iou50_all_proposals"] == 0 and r["on_paint_fraction"] >= 0.5,
                ),
                (
                    "fp_off_paint",
                    lambda r: r["tp_iou50_all_proposals"] == 0 and r["on_paint_fraction"] < 0.5,
                ),
            ):
                s = [r for r in sel if cond(r)]
                if s:
                    out[f"{model}|{cls}|{label}"] = {
                        "n": len(s),
                        "median_class_prob": float(np.median([r["class_prob"] for r in s])),
                        "median_mask_quality": float(np.median([r["mask_quality"] for r in s])),
                        "median_score": float(np.median([r["score"] for r in s])),
                    }
    return out


def crop_component(name, crops, folder):
    """Generic geographic component from training crops + saved predictions (any region)."""
    x0 = min(c["affine"][2] for c in crops)
    y1 = max(c["affine"][5] for c in crops)
    for c in crops:
        c["row"] = round((y1 - c["affine"][5]) / 0.05)
        c["col"] = round((c["affine"][2] - x0) / 0.05)
    shape = (max(c["row"] for c in crops) + 512, max(c["col"] for c in crops) + 512)
    valid = np.zeros(shape, bool)
    truths, labels = {}, {}
    masks, scores, classes, qualities, crop_ids, local = [], [], [], [], [], []
    for c in crops:
        sl = (slice(c["row"], c["row"] + 512), slice(c["col"], c["col"] + 512))
        with np.load(c["file"]) as d:
            v = d["validity"].astype(bool)
            for m, lab, uid in zip(
                d["masks"].astype(bool), d["labels"], c["global_instance_ids"], strict=True
            ):
                truths.setdefault(uid, np.zeros(shape, bool))[sl] |= m & v
                labels[uid] = int(lab)
        valid[sl] |= v
        with np.load(Path(folder) / (c["crop_id"] + ".npz")) as p:
            for k in range(len(p["scores"])):
                m = np.zeros(shape, bool)
                m[sl] = p["masks"][k] & p["validity"]
                masks.append(m)
                scores.append(float(p["scores"][k]))
                classes.append(int(p["labels"][k]))
                qualities.append(float(p["qualities"][k]))
                crop_ids.append(c["crop_id"])
                local.append(k)
    ids = sorted(truths)
    comp = {
        "name": name,
        "shape": shape,
        "valid": valid,
        "ids": ids,
        "crops": crops,
        "targets": np.stack([truths[u] for u in ids]) if ids else np.zeros((0, *shape), bool),
        "target_labels": np.asarray([labels[u] for u in ids], int),
        "masks": np.stack(masks) if masks else np.zeros((0, *shape), bool),
        "scores": np.asarray(scores),
        "classes": np.asarray(classes, int),
        "qualities": np.asarray(qualities),
        "crop_ids": crop_ids,
        "local_index": local,
    }
    comp["overlaps"] = rec.overlap_matrix(comp)
    comp["iou"] = packed_iou(comp["masks"], comp["targets"], comp["valid"]).reshape(
        len(comp["scores"]), len(ids)
    )
    return comp


def main():
    began = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    designations = {
        f["global_id"]: f["designation"] for f in client_features(check_network=False)[0]
    }
    regions_reserved, _, _ = reserved_geography()
    l12 = load(ROOT / "outputs/reports/phase5_12/latest.json")
    manifest = load(Path(l12["dataset"]) / "metadata/dataset_manifest.json")

    # ---- 1. Seal the protected final test (metadata only).
    protected = next(r for r in regions_reserved if r["role"] == "protected_final_test")
    train = [c for c in manifest["crops"] if c["split"] == "train"]
    tiles = sorted({c["tile_id"] for c in train})
    regions = []
    for t in tiles:
        cs = [c for c in train if c["tile_id"] == t]
        boxes = [
            shapely.box(
                c["affine"][2],
                c["affine"][5] - c["size"] * 0.05,
                c["affine"][2] + c["size"] * 0.05,
                c["affine"][5],
            )
            for c in cs
        ]
        ids = {u for c in cs for u in c["global_instance_ids"]}
        lab = {
            u: lab for c in cs for u, lab in zip(c["global_instance_ids"], c["labels"], strict=True)
        }
        regions.append(
            {
                "id": t,
                "box": shapely.union_all(boxes),
                "ids": sorted(ids),
                "solid": sum(v == 0 for v in lab.values()),
                "dash": sum(v == 1 for v in lab.values()),
                "crops": len(cs),
            }
        )
    min_dist = min(r["box"].distance(protected["box"]) for r in regions)
    shared = set(protected["ids"]) & {u for r in regions for u in r["ids"]}
    assert min_dist >= 500 and not shared
    freeze = OUT / "protected_test_freeze.json"
    if not freeze.exists():
        write_attestation(
            freeze,
            {
                "region": protected["id"],
                "bounds": protected["bounds"],
                "bounds_sha256": sha256_text(json.dumps(protected["bounds"])),
                "source_ids_count": len(protected["ids"]),
                "min_distance_to_any_fold_region_native": min_dist,
                "shared_ids_with_folds": 0,
                "policy": "Opened exactly once, for the frozen final candidate, at the end of the "
                "roadmap. Never used for selection, thresholds or calibration.",
                "sealed_utc": datetime.now(UTC).isoformat(),
            },
        )

    # ---- 2. Leakage-safe leave-one-region-group-out folds.
    folds_path = OUT / "folds.json"
    if not folds_path.exists():
        groups = region_groups(regions, guard=500.0, link_shared_ids=False)
        folds = assign_folds(regions, groups, k=5)
        fold_ids = [{u for i in f for u in regions[i]["ids"]} for f in folds]
        spanning = sorted(
            {u for a in range(5) for b in range(a + 1, 5) for u in fold_ids[a] & fold_ids[b]}
        )
        assert all(folds), "empty fold"
        labels = {
            u: lab
            for c in train
            for u, lab in zip(c["global_instance_ids"], c["labels"], strict=True)
        }
        assert all(labels[u] == 0 for u in spanning), "a dash spans folds"
        cross = len(spanning)
        write_attestation(
            folds_path,
            {
                "k": 5,
                "guard_native": 500.0,
                "linked_groups": len(groups),
                "rule": "regions closer than 500 native units are kept in the same fold. Long "
                "client edge-line features span many regions; linking by shared ID collapsed "
                "the data into 4 groups (folds_rejected_shared_id_rule.json), so solid IDs "
                "spanning folds are listed for a strict metric variant.",
                "spanning_solid_ids": spanning,
                "folds": [
                    {
                        "fold": k,
                        "regions": [regions[i]["id"] for i in f],
                        "unique_ids": len(fold_ids[k]),
                        "solid": sum(regions[i]["solid"] for i in f),
                        "dash": sum(regions[i]["dash"] for i in f),
                        "crops": sum(regions[i]["crops"] for i in f),
                    }
                    for k, f in enumerate(folds)
                ],
                "cross_fold_shared_ids": cross,
                "cross_fold_shared_ids_are_all_solid": True,
                "fixed_development_folds": [
                    "LEGACY_POSITIVE_DEV",
                    "NEW_POSITIVE_DEV",
                    "SPARSE_NEGATIVE_CONTROL",
                ],
                "dataset_manifest_sha256": sha256(
                    Path(l12["dataset"]) / "metadata/dataset_manifest.json"
                ),
            },
        )
    folds = load(folds_path)

    # ---- 3. Dev areas through the harness; reproduce existing numbers.
    p59 = load(ROOT / "outputs/reports/phase5_9/latest.json")
    sources = {
        "default_step50": (DEFAULT / "validation_step0050", Path(p59["report_dir"]) / "inference"),
        "p512_step400": (
            Path(l12["training_run"]) / "validation_step0400",
            Path(l12["report_dir"]) / "evaluation/step0400_phase58",
        ),
    }
    reference = load(Path(l12["report_dir"]) / "checkpoint_evaluation.json")["evaluations"]
    ref_step = {"default_step50": "0", "p512_step400": "400"}
    harness, factors, reproduction = {}, [], {}
    for model, (leg, new) in sources.items():
        comps = components(leg, new)
        harness[model] = {}
        for comp in comps:
            keep, recs, dets = evaluate_component(comp)
            area_km2 = float(comp["valid"].sum()) * 0.0025 / 1e6
            summary = summarize(recs.values(), dets.values(), area_km2)
            harness[model][comp["name"]] = {
                "records": recs,
                "detections": dets,
                "area_km2": area_km2,
                "summary": summary,
            }
            factors += factor_log(model, comp, keep, designations)
            r = reference[ref_step[model]][comp["name"]]["filtered"]
            reproduction[f"{model}|{comp['name']}"] = {
                "per_class_counts_match": all(
                    summary[f"{c}_fp"] == r["per_class"][c]["fp"]
                    and abs(summary[f"{c}_f1"] - r["per_class"][c]["f1"]) < 1e-9
                    for c in ("solid", "dash")
                ),
                "macro_f1_match": abs(summary["macro_f1"] - r["macro_f1"]) < 1e-9,
                "foreground_iou_match": abs(summary["foreground_iou"] - r["foreground_iou"]) < 1e-9,
            }
        del comps
    assert all(all(v.values()) for v in reproduction.values()), reproduction

    # ---- 4. Paired block bootstrap: default vs Phase 5.12 candidate.
    boots = {}
    for area in ("LEGACY_POSITIVE_DEV", "NEW_POSITIVE_DEV"):
        a, b = harness["default_step50"][area], harness["p512_step400"][area]
        boots[area] = paired_block_bootstrap(
            a["records"], a["detections"], b["records"], b["detections"], n=2000, seed=0
        )

    def pooled(model):
        recs, dets = {}, {}
        for area in ("LEGACY_POSITIVE_DEV", "NEW_POSITIVE_DEV"):
            for blk, v in harness[model][area]["records"].items():
                recs[f"{area}:{blk}"] = v
            for blk, v in harness[model][area]["detections"].items():
                dets[f"{area}:{blk}"] = v
        return recs, dets

    ra, da = pooled("default_step50")
    rb, db = pooled("p512_step400")
    boots["POOLED_DEV"] = paired_block_bootstrap(ra, da, rb, db, n=2000, seed=0)
    neg = {m: harness[m]["SPARSE_NEGATIVE_CONTROL"]["summary"].get("fp_per_km2") for m in harness}

    # ---- 5. LORO plumbing check on one fold (training crops; default model).
    fold = min(folds["folds"], key=lambda f: f["crops"])
    fold_crops = [c for c in train if c["tile_id"] in set(fold["regions"])]
    target = OUT / f"loro_plumbing_fold{fold['fold']}_default"
    tick = time.perf_counter()
    if not (target / "metrics.json").exists():
        model = (
            Mask2FormerForUniversalSegmentation.from_pretrained(
                DEFAULT / "checkpoint_step0050/model", local_files_only=True
            )
            .to("cuda")
            .eval()
        )
        processor = Mask2FormerImageProcessor.from_pretrained(
            DEFAULT / "checkpoint_step0050/processor", local_files_only=True
        )
        infer(model, processor, Path(l12["dataset"]), fold_crops, FROZEN, target)
        del model
        torch.cuda.empty_cache()
    infer_seconds = time.perf_counter() - tick
    plumbing = {}
    for region in fold["regions"]:
        comp = crop_component(
            region, [dict(c) for c in fold_crops if c["tile_id"] == region], target
        )
        _, recs, dets = evaluate_component(comp)
        plumbing[region] = summarize(recs.values(), dets.values())
    dump(OUT / "factor_log.json", factors)
    report = {
        "stage": 0,
        "created_utc": datetime.now(UTC).isoformat(),
        "protected_test_freeze_sha256": sha256(freeze),
        "folds_sha256": sha256(folds_path),
        "folds": folds["folds"],
        "metric_suite": SUITE,
        "bootstrap": {
            "unit": f"{BLOCK} px ({BLOCK * 0.05} m) blocks",
            "resamples": 2000,
            "seed": 0,
        },
        "reproduction_of_phase5_12_numbers": reproduction,
        "dev_summaries": {m: {a: v["summary"] for a, v in h.items()} for m, h in harness.items()},
        "default_vs_p512_bootstrap": boots,
        "negative_control_fp_per_km2": neg,
        "score_factor_summary": factor_summary(factors),
        "loro_plumbing": {
            "fold": fold["fold"],
            "regions": fold["regions"],
            "crops": len(fold_crops),
            "inference_seconds": infer_seconds,
            "seconds_per_crop": infer_seconds / max(1, len(fold_crops)),
            "metrics": plumbing,
            "caveat": "Plumbing only: the default model was trained on these "
            "regions, so these numbers are NOT generalization.",
        },
        "edge_designations": sorted(EDGE),
        "runtime_seconds": time.perf_counter() - began,
        "training_performed": False,
        "protected_final_test_used": False,
    }
    dump(OUT / "stage0_results.json", report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "folds",
                    "reproduction_of_phase5_12_numbers",
                    "negative_control_fp_per_km2",
                    "score_factor_summary",
                )
            },
            indent=1,
        )
    )
    for area, b in boots.items():
        print(
            area,
            b["blocks"],
            {
                k: (round(v["a"], 3), round(v["b"], 3), [round(x, 3) for x in v["ci_difference"]])
                for k, v in b["metrics"].items()
                if k in SUITE
            },
        )


def sha256_text(text):
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


if __name__ == "__main__":
    main()
