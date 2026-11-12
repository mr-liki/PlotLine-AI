"""Phase 5.11 Stage 8-9: evaluate every checkpoint on both development geographies.

Definitions are exactly Phase 5.10's (geographic complete-instance IoU>=.5, frozen
postprocessing, per-crop filter + global NMS). Raw-model and filtered results are kept
separate. The protected final test is never read.
"""

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

import phase5_10_recovery as rec
from phase5_11_common import ROOT, load
from roadmarking.inference.prediction_recovery import filter_geographic, packed_iou
from roadmarking.rasterization.pipeline import dump, sha256
from roadmarking.training.expanded_continuous import evaluate as infer

FROZEN = {"confidence": {"0": 0.85, "1": 0.55}, "mask_quality": 0.8, "nms_iou": 0.3}
STEPS = [0, 100, 200, 300, 400, 500, 600]


def components(prediction_legacy, prediction_new):
    p57 = load(ROOT / "outputs/reports/phase5_7_expanded/latest.json")
    p59 = load(ROOT / "outputs/reports/phase5_9/latest.json")
    p55 = load(ROOT / "outputs/reports/phase5_5/latest.json")
    manifest = load(Path(p57["dataset"]) / "metadata/dataset_manifest.json")
    p54 = load(
        Path(load(Path(p55["dataset"]) / "metadata/dataset_manifest.json")["parent_dataset"])
        / "metadata/dataset_manifest.json"
    )
    protocol = load(ROOT / "outputs/reports/phase5_10/20261106T132745062304Z/protocol.json")
    roles = protocol["data_roles"]
    legacy = rec.legacy_component(
        manifest, p54, roles["LEGACY_POSITIVE_DEV"]["crops"], prediction_legacy
    )
    report59, dataset59 = Path(p59["report_dir"]), Path(p59["dataset"])
    spec = load(
        ROOT / "outputs/reports/phase5_8_actual_review/20261106T115446528985Z/"
        "new_region_candidates.json"
    )
    tile = next(t for t in spec["tiles"] if t["candidate_id"] == "PHASE58_POSITIVE_DEV_REVIEW")
    new = rec.tile_component(
        "NEW_POSITIVE_DEV",
        tile,
        load(dataset59 / "manifest.json")["crops"],
        prediction_new,
        dataset59 / "PHASE58_POSITIVE_DEV_REVIEW_ground_truth.npz",
        load(report59 / "native_crop_and_visibility_audit.json")["eligible_ids"],
    )
    neg_spec = load(
        ROOT / "outputs/reports/phase5_7_new_review/20261106T105319835838Z/"
        "new_region_candidates.json"
    )
    neg_tile = next(t for t in neg_spec["tiles"] if t["candidate_id"] == "NEW_DEV_REVIEW")
    entries = [
        c for c in manifest["crops"] if c["crop_id"] in roles["SPARSE_NEGATIVE_CONTROL"]["crops"]
    ]
    sparse = rec.tile_component("SPARSE_NEGATIVE_CONTROL", neg_tile, entries, prediction_legacy)
    for comp in (legacy, new, sparse):
        comp["skeleton"] = (
            rec.skeletonize(comp["targets"].any(0)) & comp["valid"]
            if len(comp["targets"])
            else None
        )
        comp["overlaps"] = rec.overlap_matrix(comp)
        comp["iou"] = packed_iou(comp["masks"], comp["targets"], comp["valid"]).reshape(
            len(comp["scores"]), len(comp["target_labels"])
        )
        comp["inter"] = rec.intersections(comp["masks"], comp["targets"], comp["valid"])
    return legacy, new, sparse


def boundary(comp, idx, metrics):
    rows = []
    for cls in ("0", "1"):
        for pred, target in metrics["_ops"][cls]["assignments"].items():
            a, b = comp["masks"][idx[pred]] & comp["valid"], comp["targets"][target]
            ba, bb = a & ~binary_erosion(a), b & ~binary_erosion(b)
            d = np.r_[distance_transform_edt(~bb)[ba], distance_transform_edt(~ba)[bb]] * 0.05
            rows.append(
                {
                    "class": rec.CLASS_NAMES[int(cls)],
                    "mean": float(d.mean()),
                    "p95": float(np.percentile(d, 95)),
                }
            )
    if not rows:
        return {"matched_instances": 0, "status": "NO_MATCHES"}
    return {
        "matched_instances": len(rows),
        "mean_symmetric_observed_boundary_distance_native": float(
            np.mean([r["mean"] for r in rows])
        ),
        "p95_native": float(np.mean([r["p95"] for r in rows])),
        "caveat": "Observed-support boundary of matched instances only; not full-paint "
        "lateral accuracy",
    }


def cross_tile(comp, config):
    local = []
    for crop in comp["crops"]:
        ids = [i for i, c in enumerate(comp["crop_ids"]) if c == crop["crop_id"]]
        part = filter_geographic(
            comp["scores"][ids],
            comp["classes"][ids],
            comp["qualities"][ids],
            comp["overlaps"][np.ix_(ids, ids)],
            [crop["crop_id"]] * len(ids),
            config,
        )
        local += [ids[k] for k in part]
    agree = conflict = 0
    for a_i, a in enumerate(local):
        for b in local[a_i + 1 :]:
            if comp["crop_ids"][a] == comp["crop_ids"][b]:
                continue
            inter = (comp["masks"][a] & comp["masks"][b]).sum()
            if not inter:
                continue
            iou = inter / max(1, (comp["masks"][a] | comp["masks"][b]).sum())
            if iou < 0.3:
                continue
            if comp["classes"][a] == comp["classes"][b]:
                agree += 1
            else:
                conflict += 1
    return {
        "overlapping_cross_crop_prediction_pairs_iou30": agree + conflict,
        "same_class": agree,
        "class_conflicts": conflict,
        "class_agreement": agree / max(1, agree + conflict),
    }


def negatives(comp, idx, metrics):
    """Heuristic false-positive context categories (GT-free context; reviewer should verify)."""
    paint = comp["targets"].any(0) if len(comp["targets"]) else np.zeros(comp["shape"], bool)
    near = binary_dilation(paint, iterations=20)
    out = {
        "near_approved_marking_fragment_or_boundary": 0,
        "sparse_return_context": 0,
        "dense_bright_structure_context": 0,
    }
    for cls in ("0", "1"):
        for k in metrics["_ops"][cls]["false_indices"]:
            m = comp["masks"][idx[k]] & comp["valid"]
            if (m & near).sum() >= 0.5 * max(1, m.sum()):
                out["near_approved_marking_fragment_or_boundary"] += 1
                continue
            ring = binary_dilation(m, iterations=10) & ~binary_dilation(m, iterations=2)
            density = comp["observed"][ring].mean() if ring.any() else 0
            key = "sparse_return_context" if density < 0.5 else "dense_bright_structure_context"
            out[key] += 1
    out["definition"] = (
        "near: >=50% of the mask within 1 m of approved paint; sparse: <50% "
        "observed returns around it (vegetation/edge-of-scan); dense: observed "
        "surroundings (kerbs, guardrails, bright asphalt or unlabeled structure)"
    )
    return out


def score_step(comps):
    result = {}
    for comp in comps:
        sel = filter_geographic(
            comp["scores"],
            comp["classes"],
            comp["qualities"],
            comp["overlaps"],
            comp["crop_ids"],
            FROZEN,
        )
        filt = rec.score(
            comp,
            comp["masks"][sel],
            comp["scores"][sel],
            comp["classes"][sel],
            comp["iou"][sel],
            comp["inter"][sel],
        )
        raw = rec.score(
            comp, comp["masks"], comp["scores"], comp["classes"], comp["iou"], comp["inter"]
        )
        entry = {
            "filtered": rec.public(filt),
            "raw_model": rec.public(raw),
            "cross_tile": cross_tile(comp, FROZEN),
        }
        if comp["positive"]:
            entry["boundary"] = boundary(comp, sel, filt)
            entry["false_positive_context"] = negatives(comp, sel, filt)
            entry["crop_appearance"] = rec.crop_appearance(comp, FROZEN, comp["overlaps"])
        result[comp["name"]] = entry
    return result


def main(phase="phase5_11"):
    latest = load(ROOT / "outputs/reports" / phase / "latest.json")
    out = Path(latest["report_dir"])
    run = Path(latest["training_run"])
    protocol = load(out / "training_protocol.json")
    p59 = load(ROOT / "outputs/reports/phase5_9/latest.json")
    dataset59 = Path(p59["dataset"])
    crops59 = load(dataset59 / "manifest.json")["crops"]
    evaluations, speeds = {}, {}
    for step in STEPS:
        folder = run / f"checkpoint_step{step:04d}"
        bundle = load(folder / "bundle.json")
        assert sha256(folder / "model/model.safetensors") == bundle["weight_sha256"]
        target = out / "evaluation" / f"step{step:04d}_phase58"
        if not (target / "metrics.json").exists():
            model = (
                Mask2FormerForUniversalSegmentation.from_pretrained(
                    folder / "model", local_files_only=True
                )
                .to("cuda")
                .eval()
            )
            processor = Mask2FormerImageProcessor.from_pretrained(
                folder / "processor", local_files_only=True
            )
            torch.cuda.reset_peak_memory_stats()
            tick = time.perf_counter()
            metrics = infer(model, processor, dataset59, crops59, FROZEN, target)
            speeds[step] = {
                "model_only_seconds_per_512_crop": metrics["model_only_seconds_total"]
                / len(crops59),
                "end_to_end_seconds_per_crop": (time.perf_counter() - tick) / len(crops59),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            del model, processor
            gc.collect()
            torch.cuda.empty_cache()
        comps = components(run / f"validation_step{step:04d}", target)
        evaluations[step] = score_step(comps)
        print(
            step,
            {
                n: (v["filtered"]["macro_f1"], v["raw_model"]["macro_f1"])
                for n, v in evaluations[step].items()
            },
            flush=True,
        )
        del comps
        gc.collect()
    # Sanity: step 0 must reproduce the Phase 5.10 frozen baseline.
    base = load(ROOT / "outputs/reports/phase5_10/20261106T132745062304Z/results.json")
    reproduction = {
        name: {
            c: [evaluations[0][name]["filtered"]["per_class"][c][k] for k in ("tp", "fp", "fn")]
            == [base["components"][name]["frozen"]["per_class"][c][k] for k in ("tp", "fp", "fn")]
            for c in ("solid", "dash")
        }
        for name in ("LEGACY_POSITIVE_DEV", "NEW_POSITIVE_DEV", "SPARSE_NEGATIVE_CONTROL")
    }

    def objective(step):
        e = evaluations[step]
        f1 = [e[n]["filtered"]["macro_f1"] for n in ("LEGACY_POSITIVE_DEV", "NEW_POSITIVE_DEV")]
        fp = sum(e[n]["filtered"]["per_class"][c]["fp"] for n in e for c in ("solid", "dash"))
        return min(f1), float(np.mean(f1)), fp

    neg0 = sum(
        evaluations[0]["SPARSE_NEGATIVE_CONTROL"]["filtered"]["per_class"][c]["fp"]
        for c in ("solid", "dash")
    )
    feasible = [
        s
        for s in STEPS
        if sum(
            evaluations[s]["SPARSE_NEGATIVE_CONTROL"]["filtered"]["per_class"][c]["fp"]
            for c in ("solid", "dash")
        )
        <= neg0 + 2
    ]
    best = sorted(feasible, key=lambda s: (-objective(s)[0], -objective(s)[1], objective(s)[2], s))[
        0
    ]
    improved = objective(best)[:2] > objective(0)[:2]
    selected = best if improved else 0
    summary = {
        "protocol_sha256": sha256(out / "training_protocol.json"),
        "step0_reproduces_phase5_10": reproduction,
        "objective_by_step": {
            s: dict(zip(("worst_macro_f1", "mean_macro_f1", "total_fp"), objective(s), strict=True))
            for s in STEPS
        },
        "feasible_steps": feasible,
        "best_step": best,
        "strict_improvement_over_step0": bool(improved),
        "selected_step": selected,
        "selected_checkpoint": str(run / f"checkpoint_step{selected:04d}"),
        "inference_speed": speeds,
        "selection_rule": protocol["selection"],
    }
    dump(out / "checkpoint_evaluation.json", {"summary": summary, "evaluations": evaluations})
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "phase5_11")
