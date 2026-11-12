"""Report new real-geography approval, synthetic audit and fixed pilot outcomes."""

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from roadmarking.inference.calibration import aggregate_counts, iou_matrix
from roadmarking.rasterization.pipeline import dump, sha256

root = Path(__file__).resolve().parents[1]
context_path = root / "outputs/reports/phase5_7_expanded/latest.json"
context = json.loads(context_path.read_text())
report, dataset, run = (Path(context[k]) for k in ("report_dir", "dataset", "training_run"))
manifest = json.loads((dataset / "metadata/dataset_manifest.json").read_text())
best = json.loads((run / "best.json").read_text())
history = json.loads((run / "validation_history.json").read_text())
logs = json.loads((run / "training_log.json").read_text())
results = {}
for entry in history:
    step, metrics = entry["step"], entry["metrics"]
    by_group = {}
    for crop in metrics["crops"]:
        with np.load(run / f"validation_step{step:04d}" / (crop["crop_id"] + ".npz")) as p:
            masks, scores, labels = p["masks"], p["scores"], p["labels"]
            valid, selected = p["validity"], p["selected"].astype(int).tolist()
        with np.load(crop["file"]) as t:
            targets, target_labels = t["masks"], t["labels"]
        rec = {
            "scores": scores,
            "labels": labels,
            "target_labels": target_labels,
            "iou": iou_matrix(masks, targets, valid),
        }
        key = (
            "new_sparse_development"
            if crop["tile_id"] == "NEW_DEV_REVIEW"
            else "legacy_development"
        )
        by_group.setdefault(key, []).append((rec, selected))
    results[str(step)] = {
        "combined": metrics["postprocessed"],
        "groups": {
            k: aggregate_counts([r for r, _ in v], [s for _, s in v]) for k, v in by_group.items()
        },
        "foreground_iou": metrics["postprocessed_mean_foreground_iou"],
        "per_crop_ap": metrics["postprocessed_per_crop_ap"],
        "model_only_seconds": metrics["model_only_seconds_total"],
    }
baseline, selected = results["0"]["combined"], results[str(best["step"])]["combined"]
stats = {}
for split in ("train", "val"):
    crops = [c for c in manifest["crops"] if c["split"] == split]
    stats[split] = {
        "crop_appearances": len(crops),
        "unique_ids": len({uid for c in crops for uid in c["global_instance_ids"]}),
    }
suite = ET.parse(root / "outputs/reports/approved_expansion_pytest.xml").getroot().find("testsuite")
assert suite.attrib["failures"] == "0" and suite.attrib["errors"] == "0"
frozen = json.loads(
    (root / "outputs/reports/phase5_6a/20261106T093757339115Z/frozen_model.json").read_text()
)
for path, expected in frozen["artifact_sha256"].items():
    assert sha256(path) == expected
result = {
    "status": "REAL_APPROVED_DATA_EXPANDED",
    "selected_checkpoint": best["checkpoint"],
    "selected_step": best["step"],
    "combined_development_baseline": baseline,
    "combined_development_selected": selected,
    "comparisons": results,
    "statistics": stats,
    "new_observed_instances": context["new_statistics"],
    "synthetic_used": False,
    "fresh_final_test_approved_but_not_used": True,
    "old_diagnostic_test_not_reevaluated": True,
    "new_optimizer_steps": len(logs),
    "peak_gpu_allocated_bytes": max(x["peak_allocated_bytes"] for x in logs),
    "optimizer_step_seconds": sum(x["seconds"] for x in logs),
    "pytest": dict(suite.attrib),
    "production_ready": False,
}
dump(report / "report.json", result)
dump(
    report / "selected_inference.json",
    {
        "checkpoint": best["checkpoint"],
        "postprocessing": history[0]["metrics"]["postprocessing"],
        "production_ready": False,
        "selected_using": "Combined approved development crops only",
    },
)
for filename in (
    "approved_expansion_tests.log",
    "approved_expansion_ruff.log",
    "approved_expansion_black.log",
    "approved_expansion_pytest.xml",
):
    shutil.copy2(root / "outputs/reports" / filename, report / filename)
lines = [
    "# Approved real-data expansion and synthetic assessment",
    "",
    "User approval is recorded in new_three_region_approval.json, with exact project, "
    "geometry and source-raster hashes. The three geographic roles remain train, "
    "development and protected final test. No repeat review is required.",
    "",
    "New training support: 30 observable supported IDs (16 solid,14 dash). New "
    "development support: one observable solid, no observed dashes, only7723 valid "
    "pixels before cropping. Unobserved annotations are not counted as examples. "
    "Sparse validation crops permit >=1% valid support; training requires >=10%. "
    "NoData, boundaries, conflicts, unsupported symbols and tiny targets remain ignored.",
    "",
    f"Retained dataset counts: `{stats}`. Crop counts are overlapping appearances.",
    "",
    "Synthetic dataset: all500 image/instance/semantic triplets audited. It is "
    "procedural data, not measured LiDAR. Synthetic class0 dashed maps to model1, "
    "synthetic1 solid maps to model0; classes2..15 would require exclusion. Multiple "
    "painted dashes share instance IDs. A native physical GSD is not established. "
    "It was not mixed into this pilot: an adapter and separate real-development "
    "ablation are required to assess benefit without confounding geography expansion. "
    "See synthetic_full_audit.json and synthetic_source_hashes.json.",
    "",
    "Pilot: one bounded200-step Swin-S experiment, native512, batch1, lr1e-6, "
    "scale16, unchanged valid-only losses. Sampling cycle solid,new_region,dash,"
    "new_region,verified_negative deliberately exposes new geography. Full optimizer, "
    "scheduler, scaler, RNG and model snapshots saved every50 steps. The expanded "
    "development baseline was measured then reproduced before updates. Thresholds "
    "remained fixed. Neither protected test contributed to training or selection.",
    "",
    "| Step | Combined development macro F1 |",
    "|---|---:|",
]
for step, e in results.items():
    lines.append(f"| {step} | {e['combined']['macro_f1']:.4f} |")
lines += [
    "",
    f"Selected bundle: `{best['checkpoint']}`.",
    "This combined development set changed, so these numbers are not directly "
    "comparable with the earlier five-crop-only F1. report.json separates legacy "
    "and new sparse geography, and includes precision/recall/F1, AP, IoU and timings. "
    "Overlapping appearances and one sparse solid do not establish independent "
    "generalization. The newly approved final test remains unconsumed.",
    "",
    f"QA: {int(suite.attrib['tests'])-int(suite.attrib['skipped'])} passed, "
    f"{suite.attrib['skipped']} skipped. Geometry/approval hashes, split IDs, native "
    "targets and NoData checks passed. Ruff/Black results accompany this report. "
    f"Peak allocated GPU memory {result['peak_gpu_allocated_bytes']/2**30:.3f} GiB; "
    f"summed optimizer-step time {result['optimizer_step_seconds']:.1f}s.",
    "",
    "No production readiness or final 3D digitization is claimed.",
]
(report / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
context.update(
    report=str(report / "report.md"),
    selected_checkpoint=best["checkpoint"],
    status=result["status"],
)
dump(context_path, context)
print(
    json.dumps(
        {
            "selected_step": best["step"],
            "baseline": baseline["macro_f1"],
            "selected": selected["macro_f1"],
            "statistics": stats,
        },
        indent=2,
    )
)
