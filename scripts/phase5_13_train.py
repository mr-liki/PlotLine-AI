"""Phase 5.13: one predeclared corrective experiment — solid-exposure rebalance only.

Identical to Phase 5.12 (same data, initialization, budget, optimizer, criterion) except the
sampling cycle, which restores solid-paint exposure. Acceptance criteria and tolerances are
fixed here, from measured checkpoint variability, before any candidate is scored.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from phase5_11_common import ROOT, load
from phase5_11_export import check_phase511
from phase5_11_train import FROZEN, SELECTED, WEIGHT_SHA
from roadmarking.dataset.phase55 import write_attestation
from roadmarking.rasterization.pipeline import dump, sha256
from roadmarking.training.expanded_continuous import run_training

CYCLE = ["p511_solid", "solid", "p511_solid", "solid", "dash", "verified_negative"]


def main():
    out = ROOT / "outputs/reports/phase5_13"
    l12 = load(ROOT / "outputs/reports/phase5_12/latest.json")
    previous = load(Path(l12["report_dir"]) / "training_protocol.json")
    dataset = Path(l12["dataset"])
    assert sha256(SELECTED / "model/model.safetensors") == WEIGHT_SHA
    variability = load(out / "stage4_threshold_sensitivity.json")["checkpoint_variability"]
    leg_std = variability["phase5_12|LEGACY_POSITIVE_DEV|foreground_iou"]["std"]
    new_std = variability["phase5_12|NEW_POSITIVE_DEV|foreground_iou"]["std"]
    baseline = load(Path(l12["report_dir"]) / "checkpoint_evaluation.json")["evaluations"]["0"]
    b_leg = baseline["LEGACY_POSITIVE_DEV"]["filtered"]["foreground_iou"]
    b_new = baseline["NEW_POSITIVE_DEV"]["filtered"]["foreground_iou"]
    b_fp = sum(
        baseline[a]["filtered"]["per_class"][c]["fp"] for a in baseline for c in ("solid", "dash")
    )
    b_multi = baseline["NEW_POSITIVE_DEV"]["filtered"]["dash_separation"][
        "potential_multi_dash_prediction_masks"
    ]
    criteria = {
        "postprocessing": FROZEN,
        "C1_lost_edge_coverage_min": 0.60,
        "C2_legacy_foreground_iou_min": b_leg - 3 * leg_std,
        "C3_new_foreground_iou_min": b_new + new_std,
        "C3b_new_macro_f1_gt": 0.0,
        "C4_total_fp_max": max(b_fp + 2, int(-(-1.25 * b_fp // 1))),
        "C5_sparse_negative_fp_max": 2,
        "C5b_new_multi_dash_masks_max": b_multi + 1,
        "derivation": {
            "legacy_fg_iou_baseline": b_leg,
            "legacy_fg_iou_checkpoint_std": leg_std,
            "new_fg_iou_baseline": b_new,
            "new_fg_iou_checkpoint_std": new_std,
            "baseline_total_fp": b_fp,
            "lost_edge_default_coverage": 0.70,
            "lost_edge_id": "4b28eb52fa431a19bf87cd0a9314c37a7967d31fa3ed5f4e49f7454bfdec084b",
        },
        "selection_among_passing_checkpoints": [
            "max new foreground IoU",
            "max legacy foreground IoU",
            "earlier step",
        ],
        "if_none_pass": "retain the default step-50 model",
        "checkpoints_considered": [100, 200, 300, 400, 500, 600],
        "macro_f1_note": "New-area macro F1 varies by std ~.05 across checkpoints, so it is a "
        "gate (>0), not a ranking key.",
    }
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "models/mask2former_swin_s" / ("phase5_13_" + run_id)
    protocol = {
        **{
            k: previous[k]
            for k in (
                "architecture",
                "initialization",
                "initialization_weight_sha256",
                "budget",
                "optimizer",
                "unchanged_components",
                "checkpoints",
                "protected_final_test",
            )
        },
        "phase": "5.13",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "PREDECLARED_BEFORE_TRAINING",
        "dataset": str(dataset),
        "dataset_manifest_sha256": sha256(dataset / "metadata/dataset_manifest.json"),
        "single_controlled_change": {
            "sampling_cycle_phase5_12": previous["sampling_cycle"],
            "sampling_cycle_phase5_13": CYCLE,
        },
        "justification": "Stage 3: solid paint exposure per step fell 14% and solid:dash pixel "
        "ratio 3.8 -> 2.5. Stage 4: the lost edge is a confidence-margin "
        "failure (.818 vs .85) with labels and forgetting ruled out.",
        "postprocessing_for_selection": FROZEN,
        "selection": criteria,
        "no_teacher_regularization": True,
        "synthetic_data": False,
    }
    write_attestation(out / "training_protocol.json", protocol)
    dump(
        out / "latest.json",
        {
            "run_id": run_id,
            "report_dir": str(out),
            "dataset": str(dataset),
            "training_run": str(output),
            "status": "TRAINING",
        },
    )
    result = run_training(
        dataset,
        SELECTED / "model",
        SELECTED / "processor",
        output,
        FROZEN,
        max_steps=previous["budget"]["max_optimizer_steps"],
        stop_after=previous["budget"]["max_optimizer_steps"],
        learning_rate=previous["optimizer"]["AdamW_lr"],
        sampling_cycle=CYCLE,
        dataset_check=check_phase511,
        evaluation_interval=100,
        initial_loss_scale=previous["optimizer"]["fp16_initial_scale"],
        expected_baseline_counts={"fp": 132, "fn": 11},
        extra_pools={
            "p511_solid": lambda c: c.get("phase") in ("5.11", "5.12") and 0 in c["labels"],
            "p511_dash": lambda c: c.get("phase") in ("5.11", "5.12") and 1 in c["labels"],
        },
    )
    dump(out / "training_result.json", result)
    dump(
        out / "latest.json",
        {
            "run_id": run_id,
            "report_dir": str(out),
            "dataset": str(dataset),
            "training_run": str(output),
            "status": "TRAINED",
        },
    )
    print(json.dumps({"criteria": criteria, "result": result}, indent=1))


if __name__ == "__main__":
    main()
