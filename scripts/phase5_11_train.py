"""Phase 5.11 Stage 7: one bounded, predeclared warm-start run on the expanded real dataset."""

import json
from datetime import UTC, datetime
from pathlib import Path

import torch

from phase5_11_common import ROOT, load
from phase5_11_export import check_phase511
from roadmarking.dataset.phase55 import write_attestation
from roadmarking.rasterization.pipeline import dump, sha256
from roadmarking.training.expanded_continuous import run_training

SELECTED = (
    ROOT / "models/mask2former_swin_s/phase5_7_expanded_20261106T111240383727Z/checkpoint_step0050"
)
WEIGHT_SHA = "98ab4462a1d23a4b7a3ee7b08b1480eb3f959e4b80968792107c7bb496a77786"
FROZEN = {"confidence": {"0": 0.85, "1": 0.55}, "mask_quality": 0.8, "nms_iou": 0.3}


def main():
    latest = load(ROOT / "outputs/reports/phase5_11/latest.json")
    out = Path(latest["report_dir"])
    dataset = Path(latest["dataset"])
    assert sha256(SELECTED / "model/model.safetensors") == WEIGHT_SHA
    readiness = check_phase511(dataset)
    output = ROOT / "models/mask2former_swin_s" / ("phase5_11_" + latest["run_id"])
    protocol = {
        "phase": "5.11",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "PREDECLARED_BEFORE_TRAINING",
        "architecture": "Mask2Former Swin-S (unchanged)",
        "initialization": str(SELECTED),
        "initialization_weight_sha256": WEIGHT_SHA,
        "dataset": str(dataset),
        "dataset_manifest_sha256": sha256(dataset / "metadata/dataset_manifest.json"),
        "budget": {"max_optimizer_steps": 600, "single_run": True, "no_extension_on_failure": True},
        "optimizer": {
            "AdamW_lr": 2e-6,
            "cosine_to": 4e-7,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
            "micro_batch": 1,
            "fp16_initial_scale": 16,
            "seed": 895,
        },
        "sampling_cycle": [
            "p511_solid",
            "dash",
            "p511_dash",
            "solid",
            "p511_solid",
            "verified_negative",
        ],
        "sampling_pools": {
            "p511_solid": "new Phase 5.11 crops containing a solid instance",
            "p511_dash": "new Phase 5.11 crops containing a dash instance",
            "solid / dash / verified_negative": "all approved training crops (existing rule)",
        },
        "unchanged_components": [
            "native 512 crops",
            "shared grayscale normalization, gamma .7",
            "NoData-aware Hungarian matching",
            "valid-only BCE/Dice",
            "auxiliary losses",
            "class map 0 solid / 1 dash / 2 no-object",
        ],
        "checkpoints": "complete resumable bundle every 100 steps (0..600)",
        "postprocessing_for_selection": FROZEN,
        "selection": {
            "data": "LEGACY_POSITIVE_DEV and PHASE58_POSITIVE_DEV_REVIEW geographic "
            "complete-instance metrics (Phase 5.10 definitions), plus sparse negative control",
            "constraint": "sparse negative control FP <= step-0 FP + 2",
            "order": [
                "max worst-geography macro F1",
                "max mean macro F1",
                "min total FP",
                "earlier step",
            ],
            "improvement_required": "strictly better (worst, mean) than step 0, else retain "
            "the existing checkpoint",
            "caveat": "Selection on development data is optimistic; not a final test.",
        },
        "protected_final_test": "sealed; never read",
        "readiness": readiness,
    }
    write_attestation(out / "training_protocol.json", protocol)
    torch.cuda.reset_peak_memory_stats()
    result = run_training(
        dataset,
        SELECTED / "model",
        SELECTED / "processor",
        output,
        FROZEN,
        max_steps=600,
        stop_after=600,
        learning_rate=2e-6,
        sampling_cycle=protocol["sampling_cycle"],
        dataset_check=check_phase511,
        evaluation_interval=100,
        initial_loss_scale=16,
        expected_baseline_counts={"fp": 132, "fn": 11},
        extra_pools={
            "p511_solid": lambda c: c.get("phase") == "5.11" and 0 in c["labels"],
            "p511_dash": lambda c: c.get("phase") == "5.11" and 1 in c["labels"],
        },
    )
    dump(out / "training_result.json", result)
    dump(
        ROOT / "outputs/reports/phase5_11/latest.json",
        {**latest, "training_run": str(output), "status": "TRAINED_PROVISIONAL"},
    )
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
