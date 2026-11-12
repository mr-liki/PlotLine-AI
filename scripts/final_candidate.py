"""Final stage (pre-test): build, evaluate on development areas, and FREEZE one candidate.

Candidate = adopted training recipe (Stage 4, or Stage 5 if adopted) trained once on ALL
approved training data (seed 895, 600 steps, default warm start), V1 thresholds fitted on
that recipe's leave-one-region-out held-out predictions (both seeds), and Stage-3 stitching.
Evaluated against the current default pipeline on LEGACY / PHASE58 / negative control.
The protected final test is NOT touched: this script stops at a frozen attestation.
"""

import json
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np

from phase5_11_common import ROOT, load
from phase5_11_evaluate import components
from roadmarking.dataset.phase55 import write_attestation
from roadmarking.evaluation.harness import block_records, paired_block_bootstrap, summarize
from roadmarking.inference.prediction_recovery import filter_geographic, packed_iou
from roadmarking.inference.stitching import stitch
from roadmarking.rasterization.pipeline import dump, sha256
from roadmarking.training.expanded_continuous import run_training
from stage1_scoring import DEFAULT, FROZEN, fit_thresholds, load_pool, run_inference
from stage4_loro import bright_background_fraction
from stage5_selftrain import recipe_of

OUT = ROOT / "outputs/reports/final_candidate"
LOST = "4b28eb52fa431a19bf87cd0a9314c37a7967d31fa3ed5f4e49f7454bfdec084b"


def adopted_recipe():
    s4 = load(ROOT / "outputs/reports/stage4_loro/stage4_results.json")["adopted"]
    s5p = ROOT / "outputs/reports/stage5_selftrain/stage5_results.json"
    s5 = load(s5p) if s5p.exists() else {}
    s2 = load(ROOT / "outputs/reports/stage2_loro/stage2_results.json")["adopted_recipe"]
    folder = {"A0": ("stage2_loro", s2)}.get(s4, ("stage4_loro", s4))
    return s4, bool(s5.get("adopt_self_training")), folder


def pipeline(comp, thr, use_stitch):
    """Per-crop NMS then either two-stage NMS (filter) or stitching; thresholds per rule."""
    if not use_stitch:
        keep = filter_geographic(
            comp["scores"],
            comp["classes"],
            comp["qualities"],
            comp["overlaps"],
            comp["crop_ids"],
            thr,
        )
        return comp["masks"], comp["scores"], comp["classes"], keep, {}
    local = []
    for crop in comp["crops"]:
        idx = [i for i, c in enumerate(comp["crop_ids"]) if c == crop["crop_id"]]
        part = filter_geographic(
            comp["scores"][idx],
            comp["classes"][idx],
            comp["qualities"][idx],
            comp["overlaps"][np.ix_(idx, idx)],
            [crop["crop_id"]] * len(idx),
            {"confidence": {"0": 0.0, "1": 0.0}, "mask_quality": 0.0, "nms_iou": 0.3},
        )
        local += [idx[k] for k in part]
    windows = {c["crop_id"]: (c["row"], c["col"]) for c in comp["crops"]}
    inst, audit = stitch(
        [comp["masks"][i] for i in local],
        [float(comp["scores"][i]) for i in local],
        [int(comp["classes"][i]) for i in local],
        [comp["crop_ids"][i] for i in local],
        windows,
        comp["valid"],
        thr["confidence"],
    )
    shape = comp["shape"]
    masks = np.stack([i["mask"] for i in inst]) if inst else np.zeros((0, *shape), bool)
    return (
        masks,
        np.array([i["score"] for i in inst]),
        np.array([i["class"] for i in inst], int),
        list(range(len(inst))),
        audit,
    )


def evaluate(leg, new, thr, use_stitch):
    res = {}
    for comp in components(leg, new):
        masks, scores, classes, keep, audit = pipeline(comp, thr, use_stitch)
        iou = packed_iou(masks, comp["targets"], comp["valid"]).reshape(
            len(scores), len(comp["ids"])
        )
        recs, dets = block_records(comp, masks, scores, classes, keep, iou)
        area = float(comp["valid"].sum()) * 0.0025 / 1e6
        e = {
            "records": recs,
            "detections": dets,
            "audit": audit,
            "summary": summarize(recs.values(), dets.values(), area),
        }
        if LOST in comp["ids"]:
            j = comp["ids"].index(LOST)
            sel = [k for k in keep if classes[k] == 0]
            union = masks[sel].any(0) if sel else np.zeros(comp["shape"], bool)
            e["lost_edge_coverage"] = float(
                (union & comp["targets"][j]).sum() / comp["targets"][j].sum()
            )
        res[comp["name"]] = e
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s4, s5_adopted, folder = adopted_recipe()
    kwargs, aug, hard_flag = recipe_of(s4)
    l12 = load(ROOT / "outputs/reports/phase5_12/latest.json")
    previous = load(Path(l12["report_dir"]) / "training_protocol.json")
    root = Path(l12["dataset"])
    manifest = load(root / "metadata/dataset_manifest.json")
    folds = load(ROOT / "outputs/reports/stage0_harness/folds.json")["folds"]
    train = [c for c in manifest["crops"] if c["split"] == "train"]
    # Thresholds from this recipe's held-out predictions (both seeds), never from dev.
    pool = []
    for seed in (895, 896):
        for f in folds:
            crops = [
                c
                for c in train
                if c["tile_id"] in set(f["regions"]) and c["tile_id"].startswith("P51")
            ]
            pred = (
                ROOT
                / "outputs/reports"
                / folder[0]
                / "inference"
                / f"{folder[1]}_s{seed}_f{f['fold']}"
            )
            pool += list(load_pool(crops, pred).values())
    thr = {
        "confidence": fit_thresholds(pool, "V1_refit_thresholds"),
        "mask_quality": 0.0,
        "nms_iou": 0.3,
    }
    cycle = list(previous["sampling_cycle"])
    hard = set()
    if hard_flag:
        cycle = ["hard_negative" if s == "verified_negative" else s for s in cycle]
        bright = {c["crop_id"]: bright_background_fraction(c) for c in train}
        cut = np.percentile(list(bright.values()), 75)
        hard = {c for c, v in bright.items() if v >= cut}
    model_dir = ROOT / "models/mask2former_swin_s/final_candidate"
    final = model_dir / "checkpoint_step0600"
    if not (final / "bundle.json").exists():
        from phase5_11_export import check_phase511

        run_training(
            root,
            DEFAULT / "checkpoint_step0050/model",
            DEFAULT / "checkpoint_step0050/processor",
            model_dir,
            FROZEN,
            max_steps=600,
            stop_after=600,
            learning_rate=previous["optimizer"]["AdamW_lr"],
            sampling_cycle=cycle,
            dataset_check=check_phase511,
            evaluation_interval=600,
            initial_loss_scale=previous["optimizer"]["fp16_initial_scale"],
            expected_baseline_counts={"fp": 132, "fn": 11},
            extra_pools={
                "p511_solid": lambda c: c.get("phase") in ("5.11", "5.12") and 0 in c["labels"],
                "p511_dash": lambda c: c.get("phase") in ("5.11", "5.12") and 1 in c["labels"],
                "hard_negative": lambda c: c["crop_id"] in hard,
            },
            seed=895,
            criterion_kwargs=kwargs,
            augment_fn=partial(aug) if aug else None,
        )
    p57 = load(ROOT / "outputs/reports/phase5_7_expanded/latest.json")
    m57 = load(Path(p57["dataset"]) / "metadata/dataset_manifest.json")
    p59 = load(ROOT / "outputs/reports/phase5_9/latest.json")
    m59 = load(Path(p59["dataset"]) / "manifest.json")
    leg = OUT / "inference/legacy"
    new = OUT / "inference/new"
    run_inference(
        final, Path(p57["dataset"]), [c for c in m57["crops"] if c["split"] == "val"], leg
    )
    run_inference(final, Path(p59["dataset"]), m59["crops"], new)
    d_leg, d_new = DEFAULT / "validation_step0050", Path(p59["report_dir"]) / "inference"
    default_thr = {
        "confidence": FROZEN["confidence"],
        "mask_quality": FROZEN["mask_quality"],
        "nms_iou": 0.3,
    }
    results = {
        "default_pipeline": evaluate(d_leg, d_new, default_thr, False),
        "default_model_V1_stitch": evaluate(d_leg, d_new, {**thr}, True),
        "candidate_V1_filter": evaluate(leg, new, thr, False),
        "candidate_V1_stitch": evaluate(leg, new, thr, True),
    }

    def pooled(r):
        recs, dets = {}, {}
        for area in ("LEGACY_POSITIVE_DEV", "NEW_POSITIVE_DEV"):
            for b, v in r[area]["records"].items():
                recs[f"{area}:{b}"] = v
            for b, v in r[area]["detections"].items():
                dets[f"{area}:{b}"] = v
        return recs, dets

    boot = {}
    base = results["default_pipeline"]
    for name in ("default_model_V1_stitch", "candidate_V1_filter", "candidate_V1_stitch"):
        boot[name] = {
            "pooled": paired_block_bootstrap(*pooled(base), *pooled(results[name]), n=2000, seed=0)[
                "metrics"
            ]
        }
        for area in ("LEGACY_POSITIVE_DEV", "NEW_POSITIVE_DEV"):
            boot[name][area] = paired_block_bootstrap(
                base[area]["records"],
                base[area]["detections"],
                results[name][area]["records"],
                results[name][area]["detections"],
                n=2000,
                seed=0,
            )["metrics"]
    neg = {k: v["SPARSE_NEGATIVE_CONTROL"]["summary"].get("fp_per_km2") for k, v in results.items()}
    g5 = neg["candidate_V1_stitch"] <= neg["default_pipeline"]
    strip = {
        k: {
            a: {kk: vv for kk, vv in e.items() if kk not in ("records", "detections")}
            for a, e in v.items()
        }
        for k, v in results.items()
    }
    freeze = {
        "status": "FROZEN_CANDIDATE_AWAITING_USER_AUTHORIZATION_FOR_ONE_TIME_FINAL_TEST",
        "frozen_utc": datetime.now(UTC).isoformat(),
        "recipe": {
            "stage4": s4,
            "stage5_self_training": s5_adopted,
            "criterion_kwargs": kwargs,
            "augmentation": bool(aug),
            "hard_negative_stratum": hard_flag,
            "seed": 895,
            "steps": 600,
        },
        "weights": str(final / "model/model.safetensors"),
        "weights_sha256": sha256(final / "model/model.safetensors"),
        "processor": str(final / "processor"),
        "dataset_manifest_sha256": sha256(root / "metadata/dataset_manifest.json"),
        "thresholds_V1": thr["confidence"],
        "postprocessing": "per-crop NMS .3, then stitching (roadmarking.inference.stitching, "
        "default parameters), thresholds after merging",
        "stitching_code_sha256": sha256(ROOT / "src/roadmarking/inference/stitching.py"),
        "negative_control_G5_stitching": g5,
        "protected_final_test_used": False,
    }
    dump(
        OUT / "final_candidate_results.json",
        {
            "results": strip,
            "bootstrap_vs_default": boot,
            "negative_control_fp_per_km2": neg,
            "freeze": freeze,
        },
    )
    path = OUT / "frozen_candidate.json"
    if not path.exists():
        write_attestation(path, freeze)
    print(
        json.dumps(
            {
                "neg": neg,
                "g5": g5,
                "summary": {
                    k: {
                        a: (
                            round(e["summary"]["foreground_iou"], 3),
                            round(e["summary"]["solid_f1"], 3),
                            round(e["summary"]["dash_f1"], 3),
                            round(e["summary"]["solid_buffer_completeness"], 3),
                            e.get("lost_edge_coverage"),
                        )
                        for a, e in v.items()
                        if a != "SPARSE_NEGATIVE_CONTROL"
                    }
                    for k, v in strip.items()
                },
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
