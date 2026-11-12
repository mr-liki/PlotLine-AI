"""Publish measured development results and outstanding new-geography approval gates."""

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from roadmarking.rasterization.pipeline import dump, sha256

root = Path(__file__).resolve().parents[1]
context = json.loads((root / "outputs/reports/phase5_7/latest.json").read_text())
report, run = Path(context["report_dir"]), Path(context["training_run"])
comparisons = json.loads((report / "development_comparisons.json").read_text())
best = json.loads((run / "best.json").read_text())
logs = json.loads((run / "training_log.json").read_text())
audit = json.loads((report / "saved_benchmark_solid_diagnosis.json").read_text())
summary = {
    "solid_crop_appearances": len(audit),
    "crop_boundary_intersections": sum(x["touches_crop_boundary"] for x in audit),
    "raw_dash_best_iou_exceeds_solid": sum(
        x["dash"]["best_raw_iou"] > x["solid"]["best_raw_iou"] for x in audit
    ),
    "raw_solid_iou50_lost_after_filter": sum(
        x["solid"]["best_raw_iou"] >= 0.5 > x["solid"]["best_post_iou"] for x in audit
    ),
    "raw_solid_union_coverage50_but_no_single_iou50": sum(
        x["solid"]["raw_union_gt_pixel_coverage"] >= 0.5 > x["solid"]["best_raw_iou"] for x in audit
    ),
}
dump(report / "failure_cause_summary.json", summary)
new_context = json.loads((root / "outputs/reports/phase5_7_new_review/latest.json").read_text())
new_report = Path(new_context["report_dir"])
new_spec = json.loads((new_report / "new_region_candidates.json").read_text())
reservation = (
    root / "outputs/reports/phase5_7/20261106T104924782729Z/new_geography_reservations.json"
)
shutil.copy2(reservation, report / "new_geography_reservations.json")
for filename in (
    "phase5_7_train.log",
    "phase5_7_tests.log",
    "phase5_7_ruff.log",
    "phase5_7_black.log",
    "phase5_7_healthcheck.log",
    "phase5_7_healthcheck.json",
    "phase5_7_pytest.xml",
):
    shutil.copy2(root / "outputs/reports" / filename, report / filename)
suite = ET.parse(report / "phase5_7_pytest.xml").getroot().find("testsuite")
assert suite.attrib["failures"] == "0" and suite.attrib["errors"] == "0"
assert "All checks passed" in (report / "phase5_7_ruff.log").read_text()
assert "would be left unchanged" in (report / "phase5_7_black.log").read_text()
frozen = json.loads(
    (root / "outputs/reports/phase5_6a/20261106T093757339115Z/frozen_model.json").read_text()
)
for filename, expected in frozen["artifact_sha256"].items():
    assert sha256(filename) == expected, filename
baseline = comparisons["0"]["crop_metrics"]["postprocessed"]
selected = comparisons[str(best["step"])]["crop_metrics"]["postprocessed"]
status = (
    "RECOGNITION_IMPROVED_PROVISIONAL"
    if selected["macro_f1"] > baseline["macro_f1"]
    else "MORE_TRAINING_DATA_REQUIRED"
)
result = {
    "status": status,
    "selected_checkpoint": best["checkpoint"],
    "selected_step": best["step"],
    "baseline_development": baseline,
    "selected_development": selected,
    "optimizer_steps": len(logs),
    "peak_allocated_bytes": max(x["peak_allocated_bytes"] for x in logs),
    "summed_optimizer_step_seconds": sum(x["seconds"] for x in logs),
    "protected_benchmark_reevaluated": False,
    "new_training_samples_used": 0,
    "new_review_project": str(Path(new_spec["output"]) / "independent_test_review.qgs"),
    "new_final_test_approval": "PENDING_NEW_REGION_ONLY",
    "failure_cause_summary": summary,
    "pytest": dict(suite.attrib),
    "vectorization_pilot_ready": False,
}
dump(report / "report.json", result)
dump(
    report / "code_hashes.json",
    {
        str(p.relative_to(root)): sha256(p)
        for p in [
            *root.glob("scripts/phase5_7*.py"),
            root / "src/roadmarking/inference/continuity.py",
            root / "src/roadmarking/training/phase57_continuous.py",
        ]
    },
)
lines = [
    "# Phase 5.7 — " + status,
    "",
    "The consulted Phase 5.6 benchmark, its ground truth, original metrics and frozen "
    "model are preserved. No further benchmark inference, training, augmentation or "
    "threshold selection used those regions. They are not an untouched final test.",
    "",
    "## Failure audit",
    "",
    f"Saved-output diagnostic counts: `{summary}`.",
    "Counts are overlapping crop appearances, not mutually exclusive root causes. "
    "saved_benchmark_solid_diagnosis.json contains native-grid raw/postprocessed "
    "IoU, class alternatives, boundary contact and painted-pixel coverage by source ID. "
    "The saved proposal floor .25 means lower-confidence raw queries are not assessed "
    "by this mask-level audit. Missing source observations remain excluded.",
    "Phase 5.6 reconciled duplicates but never united solid fragments. Consequently "
    "a long client solid spanning crops could receive several partial detections "
    "without one IoU50 match. This convention and actual model errors both matter. "
    "The old test metrics are retained unchanged; no corrected test score was selected.",
    "",
    "## Controlled development experiment",
    "",
    "Warm start: selected Phase 5.5 Swin-S. One fixed 400-update AdamW experiment, "
    "lr 2e-6 cosine decay, 512 native pixels, batch1, loss scale16, no resizing. "
    "Sampling cycle: solid, solid, dash, verified negative. The approved 214 crops "
    "and geographically separate five-crop development component are unchanged. "
    "No new loss or architecture was introduced. Existing valid-only matching/BCE/"
    "Dice/auxiliary behavior is retained; no silent skipped updates. Complete model, "
    "optimizer, scheduler, scaler, RNG, dataset and config saved every100 steps.",
    "The first launch stopped at the legacy Phase 5.3 baseline-count guard before "
    "updates. The new isolated trainer verifies the actual Phase 5.5 warm-start "
    "counts instead. The hash-frozen legacy inference/trainer source was restored "
    "exactly, and frozen artifacts reverified. No unbounded retries/search.",
    "",
    "| Checkpoint | Development macro F1 (same frozen settings) |",
    "|---|---:|",
]
history = json.loads((run / "validation_history.json").read_text())
for entry in history:
    lines.append(f"| step {entry['step']} | {entry['metrics']['postprocessed']['macro_f1']:.4f} |")
lines += [
    "",
    f"Selected bundle: `{best['checkpoint']}`.",
    f"Peak allocated memory: {result['peak_allocated_bytes']/2**30:.3f} GiB. "
    f"Summed optimizer-step time: {result['summed_optimizer_step_seconds']:.1f}s. "
    "Snapshot/evaluation/startup overhead is excluded from this training-only timing.",
    "",
    "## Evaluation geometry and continuity ablations",
    "",
    "development_comparisons.json compares identical native-grid development data "
    "for baseline, selected and last checkpoint. It includes per-class P/R/F1, "
    "AP50/AP75, IoU, pixel P/R, observed skeleton coverage, matched boundary/width "
    "proxies, component counts and model-only inference timing. Saved masks and "
    "development panels permit reproduction without new inference.",
    "The geographic protocol unions GT pixels by stable source ID and reconciles "
    "predictions on the same valid support. Optional solid continuity merges only "
    "overlapping observed masks with >=4 intersecting pixels and >=.5 agreement in "
    "their shared observed crop support. It adds no pixels outside the raw mask union, "
    "does not bridge missing observations and never unions dash instances. This is "
    "separate from network predictions and is not silently enabled in the selected model.",
    "Observed width/skeleton/boundary proxies are affected by LiDAR holes. They do "
    "not establish complete-paint lateral accuracy, physical solid continuity or "
    "vertical accuracy. False-connection risk remains for overlapping parallel "
    "predictions; no whole-survey connectivity or final vectorization is attempted.",
    "",
    "## Taxonomy feasibility",
    "",
    "A short solid fragment and one dash can share local appearance. Shared-stripe "
    "class-agnostic development scoring is included as an explicit diagnostic "
    "ablation. It does not retrain or replace the two-class head and does not satisfy "
    "the client solid/dash requirement by itself. A contextual designation stage "
    "would need separately approved diverse data and validation before adoption.",
    "",
    "## New geographic data and fresh evaluation reservation",
    "",
    f"New QGIS review package: `{result['new_review_project']}`.",
    "Bookmarks NEW_TRAIN_REVIEW, NEW_DEV_REVIEW, NEW_FINAL_TEST_RESERVED have "
    "disjoint source IDs and at least500 native units of separation from used "
    "regions and earlier candidates. Three matching sources passed full-point "
    "checks before rendering. Actual painted-candidate IDs were rechecked for "
    "leakage. Native .05 imagery, observations, NoData, original references, "
    "candidate outlines, exact affines and source identities are preserved.",
    "These are review candidates, not automatically approved paint masks. No claim "
    "is made that every desired condition (faded paint, curves, merges) exists "
    "without inspecting the imagery. Additional overlapping crops are not counted "
    "as independent diversity. Zero new samples entered this experiment. "
    "NEW_FINAL_TEST_RESERVED remains protected from training, tuning and inference. "
    "Fresh final generalization cannot be reported until its labels are approved.",
    "",
    "## QA and decision",
    "",
    f"Pytest: `{suite.attrib}`. Ruff, Black and local healthcheck results accompany "
    "this report. New tests verify union-pixel conservation, validity, separation "
    "of real dashes and refusal to bridge unobserved gaps. Existing NoData-aware "
    "loss/matching/gradient/resume tests were rerun. All sixteen existing approvals "
    "are preserved; only genuinely new windows require first review.",
    "More independent training geography is the principal unresolved limitation: "
    "the current training set still has one connected component and only77 unique "
    "marking IDs, with just10 solid IDs. Development improvements alone cannot "
    "justify a digitization pilot. No production readiness is claimed.",
]
(report / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
context.update(
    status=status, selected_checkpoint=best["checkpoint"], report=str(report / "report.md")
)
dump(report.parent / "latest.json", context)
print(json.dumps(result, indent=2))
