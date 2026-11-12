"""Phase 5.12: the same bounded warm-start protocol as Phase 5.11 on the 26-region dataset.

Controlled comparison: initialization, budget, optimizer, loss, sampling cycle and selection
rule are identical to Phase 5.11; only the approved training data changes.
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

NEW_PHASES = ("5.11", "5.12")


def main():
    latest = load(ROOT / "outputs/reports/phase5_12/latest.json")
    out = Path(latest["report_dir"])
    dataset = Path(latest["dataset"])
    assert sha256(SELECTED / "model/model.safetensors") == WEIGHT_SHA
    previous = load(
        Path(load(ROOT / "outputs/reports/phase5_11/latest.json")["report_dir"])
        / "training_protocol.json"
    )
    readiness = check_phase511(dataset)
    output = ROOT / "models/mask2former_swin_s" / ("phase5_12_" + latest["run_id"])
    protocol = {
        **{
            k: previous[k]
            for k in (
                "architecture",
                "initialization",
                "initialization_weight_sha256",
                "budget",
                "optimizer",
                "sampling_cycle",
                "unchanged_components",
                "checkpoints",
                "postprocessing_for_selection",
                "selection",
                "protected_final_test",
            )
        },
        "phase": "5.12",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "PREDECLARED_BEFORE_TRAINING",
        "dataset": str(dataset),
        "dataset_manifest_sha256": sha256(dataset / "metadata/dataset_manifest.json"),
        "controlled_change": "training data only: +10 approved Phase 5.12 regions",
        "sampling_pools": {
            "p511_solid": "approved Phase 5.11 + 5.12 crops containing a solid instance",
            "p511_dash": "approved Phase 5.11 + 5.12 crops containing a dash instance",
            "solid / dash / verified_negative": "all approved training crops (existing rule)",
        },
        "comparison": "step 0 (default), Phase 5.11 selected step 400, and this run's "
        "selected step, with identical geographic definitions",
        "readiness": readiness,
    }
    write_attestation(out / "training_protocol.json", protocol)
    opt = protocol["optimizer"]
    result = run_training(
        dataset,
        SELECTED / "model",
        SELECTED / "processor",
        output,
        FROZEN,
        max_steps=protocol["budget"]["max_optimizer_steps"],
        stop_after=protocol["budget"]["max_optimizer_steps"],
        learning_rate=opt["AdamW_lr"],
        sampling_cycle=protocol["sampling_cycle"],
        dataset_check=check_phase511,
        evaluation_interval=100,
        initial_loss_scale=opt["fp16_initial_scale"],
        expected_baseline_counts={"fp": 132, "fn": 11},
        extra_pools={
            "p511_solid": lambda c: c.get("phase") in NEW_PHASES and 0 in c["labels"],
            "p511_dash": lambda c: c.get("phase") in NEW_PHASES and 1 in c["labels"],
        },
    )
    dump(out / "training_result.json", result)
    dump(
        ROOT / "outputs/reports/phase5_12/latest.json",
        {**latest, "training_run": str(output), "status": "TRAINED_PROVISIONAL"},
    )
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
