"""Stage 7C: write-once 2x2 protocol (data size x thin-line supervision fix).

Written BEFORE any Stage-7 training run. No post-hoc extensions: the step budget,
checkpoints, seeds, evaluation pipelines, gates and decision rules below are final.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from phase5_11_common import ROOT, load
from roadmarking.dataset.phase55 import write_attestation
from roadmarking.rasterization.pipeline import sha256
from stage1_scoring import DEFAULT
from stage7_common import FROZEN_CANDIDATE, OUT, S6_DEV, settings

PROTOCOL = OUT / "7C_protocol.json"
D456 = ROOT / "data/processed/phase5_training/phase5_12_20261106T165054344613Z"
D500 = ROOT / "data/processed/phase5_training/stage6_20261106T185832535587Z"
ARMS = {
    "A": {"dataset": str(D456), "markings": 456, "bilinear_fix": False},
    "B": {"dataset": str(D456), "markings": 456, "bilinear_fix": True},
    "C": {"dataset": str(D500), "markings": 500, "bilinear_fix": False},
    "D": {"dataset": str(D500), "markings": 500, "bilinear_fix": True},
}
SEEDS = [895, 896, 897]
STEPS = 600
CHECKPOINTS = [200, 400, 600]
CODE = [
    "src/roadmarking/training/expanded_continuous.py",
    "src/roadmarking/training/valid_criterion.py",
    "src/roadmarking/inference/stitching.py",
    "src/roadmarking/evaluation/full_metrics.py",
    "src/roadmarking/evaluation/link_audit.py",
    "src/roadmarking/evaluation/harness.py",
    "scripts/stage7_common.py",
    "scripts/stage7e_train.py",
    "scripts/stage7h_evaluate.py",
]


def main():
    if PROTOCOL.exists():
        raise SystemExit("Protocol already written; it is write-once.")
    l12 = load(ROOT / "outputs/reports/phase5_12/latest.json")
    previous = load(Path(l12["report_dir"]) / "training_protocol.json")
    baseline = OUT / "7D_S6_01_baseline/baseline_results.json"
    assert baseline.exists(), "S6_01 baseline must be recorded first (7D)"
    protocol = {
        "stage": "7C",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "PREDECLARED_BEFORE_TRAINING",
        "design": "2x2 factorial: data size (456 vs 500 approved markings) x thin-line "
        "supervision fix (nearest vs bilinear native-pixel logit sampling)",
        "arms": ARMS,
        "datasets_sha256": {
            k: sha256(Path(v["dataset"]) / "metadata/dataset_manifest.json")
            for k, v in ARMS.items()
        },
        "shared": {
            "architecture": "Mask2Former Swin-S (unchanged)",
            "initialization": str(DEFAULT / "checkpoint_step0050"),
            "initialization_weight_sha256": sha256(
                DEFAULT / "checkpoint_step0050/model/model.safetensors"
            ),
            "optimizer": previous["optimizer"],
            "steps": STEPS,
            "sampling_cycle": previous["sampling_cycle"],
            "sampling_pools": "existing rule + p511_solid / p511_dash (phase 5.11/5.12 "
            "crops, which include the Stage-6 crops in C/D)",
            "augmentation": "none (A0)",
            "skeleton_recall": "off",
            "fix_on": {"bilinear": True},
            "fix_off": {},
            "seeds": SEEDS,
            "checkpoints_saved": [0, *CHECKPOINTS],
            "resumable_state": "model + processor + optimizer/scheduler/scaler state per "
            "checkpoint (training_state.pt)",
        },
        "repeated_seed_check": "3 seeds per arm (12 runs). Seed 895 of arm A re-runs the "
        "frozen-candidate recipe; weight-hash equality is reported as a determinism check.",
        "checkpoint_rule": "PRIMARY = step 600 for every run, fixed in advance. Steps 200/400 "
        "are reported only as trajectory/stability evidence and are never selected. Not "
        "lowest loss, not best validation.",
        "evaluation": {
            "development_areas": {
                "S6_01": str(S6_DEV),
                "PHASE58": "NEW_POSITIVE_DEV (Phase 5.9 export)",
                "LEGACY": "LEGACY_POSITIVE_DEV crops",
                "NEGATIVE_CONTROL": "SPARSE_NEGATIVE_CONTROL",
            },
            "reported_separately": True,
            "primary_pipeline": "frozen-candidate pipeline: V1 thresholds "
            f"{load(FROZEN_CANDIDATE)['thresholds_V1']} + per-crop NMS 0.3 + stitching",
            "secondary_pipeline": "default V0 thresholds, two-stage NMS, no stitching",
            "threshold_free": "AP50 / AP75 from the same pipeline at zero thresholds",
            "references": settings(),
            "no_threshold_tuning": True,
            "excluded": "NEW_FINAL_TEST_RESERVED (consumed, historical) is never loaded; "
            "LORO pool regions are training data for every arm",
        },
        "gates_7H": {
            "G1_improvement": "pooled dev (S6_01 + PHASE58 + LEGACY) pixel IoU OR all-instance "
            "recall@IoU50: paired block-bootstrap 95% CI lower bound > 0 versus the reference "
            "in >= 2 of 3 seeds",
            "G2_no_significant_harm": "no seed has a pooled pixel-IoU or all-instance-recall50 "
            "CI upper bound < 0 versus the reference",
            "G3_solid_completeness": "seed-median pooled solid buffer completeness difference "
            ">= -0.05",
            "G4_negative_control": "seed-median negative-control FPs <= reference + 1",
            "G5_stitching_safety": "every seed: true false links (outputs joining different "
            "physical lines, dashes or classes; client splits of one painted line excluded) "
            "<= 5% of output instances on the dev pool",
            "G6_legacy": "seed-median LEGACY pixel IoU >= reference - 0.03",
            "references": ["default_V0", "frozen_candidate_V1_stitch"],
            "selection": "Among arms passing G1-G6 against BOTH references, the highest "
            "seed-mean pooled all-instance F1@50 at step 600. If none passes, the default and "
            "the frozen candidate stay unchanged.",
        },
        "fix_verification": {
            "rule": "TRAINING_OBJECTIVE_IMPROVEMENT_VERIFIED if, over the 6 seed-paired "
            "comparisons B-A and D-C, >= 5 are positive for pooled threshold-free solid AP50 "
            "AND >= 5 are positive for pooled pixel IoU, AND B and D pass G5",
            "data_effect": "C-A and D-B reported descriptively with the same pairing",
        },
        "statuses": {
            "RECOGNITION_IMPROVED_PROVISIONAL": "an arm passes G1-G6 against both references",
            "STITCHING_SAFETY_BLOCKED": "an arm passes G1-G4,G6 but fails G5",
            "NO_IMPROVEMENT": "no arm passes G1 against the frozen candidate",
            "MORE_REAL_DATA_REQUIRED": "reliability targets unmet and data effect (C/D vs A/B) "
            "positive or dash recall remains near zero",
            "NEW_INDEPENDENT_TEST_REQUIRED": "always, until new approved test regions exist",
        },
        "reliability_targets_provisional": {
            "pixel_precision": 0.90,
            "pixel_recall": 0.95,
            "instance_recall_iou50": 0.90,
        },
        "resource_limits": "RTX 3050 6 GB; one run at a time; segments bounded at 1200 s",
        "code_sha256": {p: sha256(ROOT / p) for p in CODE},
        "s6_01_baseline_sha256": sha256(baseline),
        "frozen_candidate_sha256": sha256(FROZEN_CANDIDATE),
    }
    write_attestation(PROTOCOL, protocol)
    print(json.dumps({"written": str(PROTOCOL), "runs": len(ARMS) * len(SEEDS)}, indent=1))


if __name__ == "__main__":
    main()
