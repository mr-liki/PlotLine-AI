"""Shared Stage 7 evaluation: fixed pipelines, full metric list, physical-identity link audit."""

from pathlib import Path

import numpy as np

import phase5_10_recovery as rec
from final_candidate import pipeline
from phase5_11_common import ROOT, load
from roadmarking.evaluation.full_metrics import aggregate, component_metrics
from roadmarking.evaluation.harness import block_records, summarize
from roadmarking.evaluation.link_audit import classify_outputs, duplicates, physical_lines
from roadmarking.inference.prediction_recovery import packed_iou
from stage1_scoring import DEFAULT, FROZEN

OUT = ROOT / "outputs/reports/stage7"
FROZEN_CANDIDATE = ROOT / "outputs/reports/final_candidate/frozen_candidate.json"
S6_DEV = ROOT / "data/processed/stage6_fresh_dev/20261106T185832535587Z/manifest.json"
SHIFTS = (-0.05, 0.0, 0.05)  # descriptive confidence sensitivity, never used for selection


def settings():
    """The two fixed reference pipelines (no retuning anywhere in Stage 7)."""
    frozen = load(FROZEN_CANDIDATE)
    return {
        "default_V0": {
            "checkpoint": str(DEFAULT / "checkpoint_step0050"),
            "thresholds": {
                "confidence": FROZEN["confidence"],
                "mask_quality": FROZEN["mask_quality"],
                "nms_iou": 0.3,
            },
            "stitch": False,
        },
        "frozen_candidate_V1_stitch": {
            "checkpoint": str(Path(frozen["weights"]).parents[1]),
            "thresholds": {
                "confidence": frozen["thresholds_V1"],
                "mask_quality": 0.0,
                "nms_iou": 0.3,
            },
            "stitch": True,
        },
    }


def s6_tile():
    latest = load(ROOT / "outputs/reports/stage6_review/latest.json")
    spec = load(Path(latest["report_dir"]) / "new_region_candidates.json")
    return next(t for t in spec["tiles"] if t["candidate_id"] == "S6_01")


def shifted(thr, delta):
    conf = {k: float(np.clip(v + delta, 0.0, 0.999)) for k, v in thr["confidence"].items()}
    return {**thr, "confidence": conf}


def zero(thr):
    return {**thr, "confidence": {"0": 0.0, "1": 0.0}, "mask_quality": 0.0}


def evaluate_component(comp, thr, use_stitch):
    """Full Stage-7 metrics for one geographic component and one fixed pipeline."""
    masks, scores, classes, keep, audit = pipeline(comp, thr, use_stitch)
    iou = packed_iou(masks, comp["targets"], comp["valid"]).reshape(len(scores), len(comp["ids"]))
    # Threshold-free AP: the same pipeline with zero confidence/quality thresholds.
    am, ascore, acls, akeep, _ = pipeline(comp, zero(thr), use_stitch)
    aiou = packed_iou(am, comp["targets"], comp["valid"]).reshape(len(ascore), len(comp["ids"]))
    ap = component_metrics(comp, am, ascore, acls, akeep, akeep, aiou)["ap"]
    tcls = np.asarray(comp["target_labels"], int)
    lines = physical_lines(comp["targets"], tcls)
    links, link_counts = classify_outputs(
        masks, classes, keep, comp["targets"], tcls, comp["valid"], lines
    )
    link_counts["output_solids"] = int(sum(classes[k] == 0 for k in keep))
    link_counts["output_instances"] = len(keep)
    link_counts["false_links"] = link_counts["true_false_links"]
    link_counts["duplicate_covered_targets"] = duplicates(
        masks, classes, keep, comp["targets"], tcls
    )
    link_counts["stitch_merges"] = audit
    metrics = component_metrics(comp, masks, scores, classes, keep, keep, iou, link_counts)
    metrics["ap"] = ap
    metrics["link_records"] = links
    metrics["physical_lines"] = {
        "client_ids": len(lines),
        "physical_lines": len(set(lines)),
    }
    recs, dets = block_records(comp, masks, scores, classes, keep, iou)
    area = float(comp["valid"].sum()) * 0.0025 / 1e6
    return {
        "full": metrics,
        "harness_summary": summarize(recs.values(), dets.values(), area),
        "records": recs,
        "detections": dets,
    }


def s6_component(prediction_folder):
    manifest = load(S6_DEV)
    with np.load(manifest["ground_truth"]) as gt:
        n = len(gt["labels"])
    assert n == len(manifest["eligible_ids"])
    comp = rec.tile_component(
        "S6_01",
        s6_tile(),
        manifest["crops"],
        prediction_folder,
        manifest["ground_truth"],
        manifest["eligible_ids"],
    )
    comp["overlaps"] = rec.overlap_matrix(comp)
    return comp


def headline(result):
    agg = aggregate([result["full"]])
    keys = [
        "pixel_precision",
        "pixel_recall",
        "pixel_iou",
        "all_instance_precision50",
        "all_instance_recall50",
        "all_instance_f150",
        "solid_instance_recall50",
        "dash_instance_recall50",
        "centerline_coverage",
        "narrow_marking_pixel_recall",
        "false_stitch_links",
    ]
    return {k: agg[k] for k in keys} | {
        "solid_ap50": result["full"]["ap"]["ap50"]["solid"],
        "dash_ap50": result["full"]["ap"]["ap50"]["dash"],
    }
