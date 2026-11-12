"""Diagnose saved benchmark failures once; start a fixed development-only experiment."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from roadmarking.dataset.phase55 import check_dataset
from roadmarking.inference.calibration import iou_matrix
from roadmarking.rasterization.pipeline import dump, sha256
from roadmarking.training.phase57_continuous import run_training

root = Path(__file__).resolve().parents[1]
run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
report = root / "outputs/reports/phase5_7" / run_id
report.mkdir(parents=True)
prior = json.loads((root / "outputs/reports/phase5_5/latest.json").read_text())
test = json.loads((root / "outputs/reports/phase5_6_independent/latest.json").read_text())
protected = Path(test["report_dir"])
test_dataset = Path(test["dataset"])
test_manifest = json.loads((test_dataset / "manifest.json").read_text())
evidence, protected_ids = [], set()
for crop in test_manifest["crops"]:
    protected_ids.update(crop["global_instance_ids"])
    with np.load(crop["file"]) as data:
        targets, classes, valid = data["masks"], data["labels"], data["validity"]
    with np.load(protected / "inference" / (crop["crop_id"] + ".npz")) as data:
        masks, labels, selected = data["masks"], data["labels"], data["selected"].astype(int)
        scores = data["scores"]
    ious = iou_matrix(masks, targets, valid)
    for j in np.where(classes == 0)[0]:
        truth = targets[j].astype(bool)
        item = {
            "crop": crop["crop_id"],
            "source_id": crop["global_instance_ids"][j],
            "valid_gt_pixels": int(truth.sum()),
            "touches_crop_boundary": bool(truth[[0, -1]].any() or truth[:, [0, -1]].any()),
        }
        for cls, name in ((0, "solid"), (1, "dash")):
            raw = np.where(labels == cls)[0]
            kept = [i for i in selected if labels[i] == cls]
            item[name] = {
                "best_raw_iou": float(max(ious[raw, j], default=0)),
                "best_post_iou": float(max(ious[kept, j], default=0)),
                "raw_union_gt_pixel_coverage": float(
                    (masks[raw].any(0) & truth & valid).sum() / max(1, (truth & valid).sum())
                ),
                "best_raw_score": float(scores[raw[np.argmax(ious[raw, j])]]) if len(raw) else None,
            }
        evidence.append(item)
dump(report / "saved_benchmark_solid_diagnosis.json", evidence)
dump(
    report / "protected_benchmark.json",
    {
        "role": "CONSULTED_DIAGNOSTIC_BENCHMARK_NOT_UNTOUCHED_FINAL_TEST",
        "source_report": str(protected),
        "source_report_sha256": sha256(protected / "report.json"),
        "approval_sha256": sha256(protected / "three_region_approval.json"),
        "inference_rerun": False,
        "used_for_training_or_threshold_selection": False,
        "known_evaluation_issue": "Phase 5.6 used NMS without fragment union; full solid target "
        "may span several crops. Original metrics retained, not replaced.",
    },
)
dataset = Path(prior["dataset"])
check_dataset(dataset)
manifest = json.loads((dataset / "metadata/dataset_manifest.json").read_text())
assert not (protected_ids & {uid for c in manifest["crops"] for uid in c["global_instance_ids"]})
settings = json.loads((Path(prior["report_dir"]) / "selected_postprocessing.json").read_text())
output = root / "models/mask2former_swin_s" / ("phase5_7_" + run_id)
context = {
    "run_id": run_id,
    "report_dir": str(report),
    "dataset": str(dataset),
    "training_run": str(output),
    "protected_test": str(protected),
    "budget": "One 400-step experiment; snapshots every100; no test reevaluation",
}
dump(report.parent / "latest.json", context)
print(json.dumps(context), flush=True)
selected = Path(prior["selected_checkpoint"])
baseline_metrics = json.loads((selected / "bundle.json").read_text())["metrics"]
expected_counts = {k: sum(baseline_metrics["raw"][str(i)][k] for i in (0, 1)) for k in ("fp", "fn")}
result = run_training(
    dataset,
    selected / "model",
    selected / "processor",
    output,
    settings,
    max_steps=400,
    stop_after=400,
    learning_rate=2e-6,
    sampling_cycle=["solid", "solid", "dash", "verified_negative"],
    dataset_check=check_dataset,
    evaluation_interval=100,
    initial_loss_scale=16,
    expected_baseline_counts=expected_counts,
)
dump(report / "training_result.json", result)
print(json.dumps(result, indent=2), flush=True)
