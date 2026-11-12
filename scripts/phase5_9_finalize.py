"""Audit stored frozen outputs and publish geographically distinct development findings."""

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, label

from roadmarking.inference.calibration import iou_matrix
from roadmarking.rasterization.pipeline import dump, sha256

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
context = json.loads((root / "outputs/reports/phase5_9/latest.json").read_text())
report, dataset = Path(context["report_dir"]), Path(context["dataset"])
result = json.loads((report / "report.json").read_text())
manifest = json.loads((dataset / "manifest.json").read_text())
approval = json.loads((report / "approved_positive_development_attestation.json").read_text())
assert sha256(approval["candidate_store"]) == approval["candidate_sha256"]
assert sha256(approval["project"]) == approval["project_sha256"]
for path, expected in approval["rasters"].items():
    assert sha256(path) == expected
for crop in manifest["crops"]:
    assert sha256(crop["file"]) == crop["sha256"]
raw_audit = []
actual_support = np.zeros((1024, 1024), bool)
for crop in manifest["crops"]:
    with np.load(crop["file"]) as d:
        targets, classes, valid = d["masks"].astype(bool), d["labels"], d["validity"]
        assert not targets[:, ~valid].any()
    row, col, h, w = crop["window"]
    actual_support[row : row + h, col : col + w] |= valid
    with np.load(report / "inference" / (crop["crop_id"] + ".npz")) as p:
        masks, labels, scores = p["masks"], p["labels"], p["scores"]
        selected = p["selected"].astype(int)
    ious = iou_matrix(masks, targets, valid)
    for j, uid in enumerate(crop["global_instance_ids"]):
        cls = int(classes[j])
        correct = np.where(labels == cls)[0]
        other = np.where(labels != cls)[0]
        kept = [i for i in selected if labels[i] == cls]
        truth = targets[j] & valid
        raw_audit.append(
            {
                "crop": crop["crop_id"],
                "source_id": uid,
                "class": cls,
                "best_raw_same_class_iou": float(max(ious[correct, j], default=0)),
                "best_raw_other_class_iou": float(max(ious[other, j], default=0)),
                "best_post_same_class_iou": float(max(ious[kept, j], default=0)),
                "best_raw_same_class_confidence": (
                    float(scores[correct[np.argmax(ious[correct, j])]]) if len(correct) else None
                ),
                "raw_same_class_union_gt_coverage": float(
                    (masks[correct].any(0) & truth).sum() / max(1, truth.sum())
                ),
                "gt_touches_crop_boundary": bool(truth[[0, -1]].any() or truth[:, [0, -1]].any()),
            }
        )
dump(report / "raw_filtered_crop_error_audit.json", raw_audit)
summary = {
    "target_crop_appearances": len(raw_audit),
    "same_class_raw_iou50_opportunities": sum(
        r["best_raw_same_class_iou"] >= 0.5 for r in raw_audit
    ),
    "raw_iou50_opportunities_lost_after_filter": sum(
        r["best_raw_same_class_iou"] >= 0.5 > r["best_post_same_class_iou"] for r in raw_audit
    ),
    "other_class_iou50_opportunities": sum(r["best_raw_other_class_iou"] >= 0.5 for r in raw_audit),
    "raw_union_coverage50_without_single_iou50": sum(
        r["raw_same_class_union_gt_coverage"] >= 0.5 > r["best_raw_same_class_iou"]
        for r in raw_audit
    ),
    "boundary_intersecting_targets": sum(r["gt_touches_crop_boundary"] for r in raw_audit),
    "raw_scope": "Saved proposals above fixed .25 proposal floor, not all neural queries",
}
name = next(iter(result["per_region"]))
truth_file = np.load(dataset / (name + "_ground_truth.npz"))
pred_file = np.load(report / (name + "_global_predictions.npz"))
gt, gt_classes, valid = (
    truth_file["masks"].astype(bool),
    truth_file["labels"],
    truth_file["validity"],
)
pred, pred_classes = pred_file["masks"].astype(bool), pred_file["labels"]
assert not (valid & ~actual_support).any()
pixel = {}
for cls, key in ((None, "all_supported_paint"), (0, "solid"), (1, "dash")):
    a = (pred if cls is None else pred[pred_classes == cls]).any(0) & valid
    b = (gt if cls is None else gt[gt_classes == cls]).any(0) & valid
    tp, fp, fn = int((a & b).sum()), int((a & ~b).sum()), int((b & ~a).sum())
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    pixel[key] = {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
gt_union, pred_union = gt.any(0) & valid, pred.any(0) & valid
safe = binary_erosion(valid, structure=np.ones((3, 3), bool))
gb = (gt_union & ~binary_erosion(gt_union)) & safe
pb = (pred_union & ~binary_erosion(pred_union)) & safe
boundary = {
    "reliable_observed_neighborhood_pixels": int(safe.sum()),
    "units": "native coordinate units; source CRS records meters",
    "accuracy_status": "NOT_A_DEFENSIBLE_FULL_PAINT_BOUNDARY_OR_LATERAL_ACCURACY_MEASURE",
    "warning": (
        "Ambiguous true-boundary pixels are excluded and no global instance matches exist. "
        "Safe-core distances are diagnostics, not physical marking lateral error."
    ),
    "note": (
        "Class-agnostic union boundary diagnostic; excludes invalid 3x3 "
        "neighborhoods. Not a matched-instance lateral offset."
    ),
}
if gb.any() and pb.any():
    pd, gd = distance_transform_edt(~gb)[pb] * 0.05, distance_transform_edt(~pb)[gb] * 0.05
    boundary.update(
        predicted_boundary_precision_within_one_pixel=float((pd <= 0.05).mean()),
        gt_boundary_recall_within_one_pixel=float((gd <= 0.05).mean()),
        symmetric_mean_distance_native=float(np.r_[pd, gd].mean()),
        symmetric_p95_distance_native=float(np.percentile(np.r_[pd, gd], 95)),
    )
fragmentation = []
ids = result["pooled_unique_gt_ids"]
# Single-region evaluation target IDs preserve export order, not pooled sorted order.
crop_ids = json.loads((report / "native_crop_and_visibility_audit.json").read_text())[
    "eligible_ids"
]
for mask, cls, uid in zip(gt, gt_classes, crop_ids, strict=True):
    eligible = pred[pred_classes == cls]
    intersect = np.array([(p & mask & valid).sum() for p in eligible])
    overlapping = eligible[intersect >= 4]
    union = overlapping.any(0) if len(overlapping) else np.zeros_like(valid)
    fragmentation.append(
        {
            "source_id": uid,
            "class": int(cls),
            "observed_gt_pixels": int(mask.sum()),
            "overlapping_same_class_prediction_masks": int((intersect >= 4).sum()),
            "same_class_union_gt_pixel_coverage": float((union & mask).sum() / max(1, mask.sum())),
            "observed_gt_components": int(label(mask)[1]),
            "intersecting_prediction_components": int(label(union & mask)[1]),
            "caveat": (
                "Components are affected by missing LiDAR returns "
                "and do not alone measure physical breaks"
            ),
        }
    )
dash_targets = gt[gt_classes == 1]
multi_dash = sum(
    sum((p & t).sum() >= 4 and (p & t).sum() / max(1, t.sum()) >= 0.1 for t in dash_targets) > 1
    for p in pred[pred_classes == 1]
)
dump(
    report / "pixel_boundary_fragmentation_diagnostics.json",
    {
        "pixel": pixel,
        "boundary": boundary,
        "per_source_fragmentation": fragmentation,
        "potential_multi_dash_prediction_masks": int(multi_dash),
        "multi_dash_definition": ">=4 pixels and >=10% of each of two approved dash masks",
        "matched_instance_boundary_accuracy": "NOT_ESTABLISHED_NO_IOU50_MATCHES",
    },
)
dump(report / "error_summary.json", summary)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, title, mask in zip(
    axes,
    [
        "Approved boundary (safe support)",
        "Predicted boundary (safe support)",
        "Paint pixel disagreement",
    ],
    [gb, pb, (gt_union ^ pred_union) & valid],
    strict=True,
):
    ax.imshow(np.ma.masked_where(~valid, np.zeros_like(valid)), cmap="gray", vmin=0, vmax=1)
    ax.imshow(np.ma.masked_where(~mask, mask), cmap="autumn", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
fig.tight_layout()
fig.savefig(report / "boundary_fragmentation_errors.png", dpi=130)
plt.close(fig)
suite = ET.parse(root / "outputs/reports/phase5_9_pytest.xml").getroot().find("testsuite")
assert suite.attrib["failures"] == "0" and suite.attrib["errors"] == "0"
assert "All checks passed!" in (root / "outputs/reports/phase5_9_ruff.log").read_text()
assert "would be left unchanged" in (root / "outputs/reports/phase5_9_black.log").read_text()
for filename in (
    "phase5_9_pytest.xml",
    "phase5_9_tests.log",
    "phase5_9_ruff.log",
    "phase5_9_black.log",
    "phase5_9_healthcheck.log",
    "phase5_9_healthcheck.json",
):
    shutil.copy2(root / "outputs/reports" / filename, report / filename)
frozen = json.loads((report / "frozen_model.json").read_text())
for path, expected in frozen["artifact_sha256"].items():
    assert sha256(path) == expected
result.update(
    development_only=True,
    protected_final_test_used=False,
    retraining_performed=False,
    historical_legacy_appearance_macro_f1=0.5927272727272728,
    raw_error_summary=summary,
    pixel_metrics=pixel,
    pytest=dict(suite.attrib),
)
dump(report / "report.json", result)
counts = result["pooled_geographic"]
lines = [
    "# Phase 5.9: approved positive geographic development evaluation",
    "",
    "Decision: recognition improvement is required. No training or recalibration was "
    "performed. The protected final-test region remains sealed.",
    "",
    "## Approval, data and separation",
    "",
    "approved_positive_development_attestation.json records the explicit Phase5.9 "
    "approval for positions, painted-area geometry, classes and all visible targets. "
    "It links the exact QGIS project, source geometry hashes, raster hashes, footprint, "
    "affines, source request and frozen model. The file is exclusively created and "
    "read-only with SHA-256 checks; no external WORM-storage guarantee is implied.",
    "All8 observed solid IDs and11 dash IDs produce eligible model targets. Nine "
    "native512 crops meet the unchanged >=10% valid-support rule. Native384/256 "
    "eligibility was checked (22/39 context windows) but smaller crops were not "
    "needed. There was no downsampling or lowered coverage threshold. Approved "
    "region masks and separate evaluation masks, IDs, states and affine mappings "
    "are exported; no source outlines were altered.",
    "734073 unobserved pixels are excluded;305735 pixels are reliably supervised "
    "after boundary, conflict, unsupported-symbol and tiny-target exclusions. All "
    "scored pixels are supported by actual crop validity. Source IDs and >=500 "
    "native-unit geographic separation from all nineteen prior regions were "
    "verified, including training and the protected final test. This window stays "
    "development-only, never training. Existing approvals remain intact.",
    "",
    "## Frozen recognition result",
    "",
    "The selected Phase5.7 expanded step50 model and saved processor were loaded. "
    "Class map, shared raster normalization, gamma.7, confidence(.85 solid/.55 dash), "
    "mask-quality.8 and same-class NMS.3 were frozen and hash-checked before/after.",
    "| Class | Precision | Recall | F1 | TP | FP | FN | AP50 | AP75 |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for cls, name in (("0", "Solid"), ("1", "Dash")):
    v = counts[cls]
    lines.append(
        f"| {name} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} | "
        f"{v['tp']} | {v['fp']} | {v['fn']} | 0.0000 | 0.0000 |"
    )
lines += [
    "",
    "Macro F1:0.0000. Foreground IoU:0.2059. Zero complete-instance IoU50 "
    "matches does not mean zero painted pixels recognized. Pixel metrics are "
    "reported separately in pixel_boundary_fragmentation_diagnostics.json. "
    "AP is measured on frozen filtered predictions above the fixed proposal "
    "floor, using101 recall points, not an unrestricted low-threshold COCO claim.",
    "Fifteen overlapping crop predictions were suppressed in geographic "
    "coordinates, leaving23 predictions. All19 observable source identities "
    "were missed under complete-instance IoU50 matching. Zero matched-target "
    "duplicate counts does not prove all false predictions are physically "
    "distinct; unmatched fragments can still describe pieces of the same paint. "
    "Individual dashes were not joined and no geometry was extrapolated.",
    "",
    f"Saved raw/postprocessed error audit: `{summary}`.",
    "Crop appearances are distinct from unique geographic instances. Raw "
    "opportunities use the saved proposal floor.25, not every unfiltered query. "
    "No threshold search or new neural inference was used for error analysis.",
    "Boundary and fragmentation images, class-agnostic safe-neighborhood "
    "boundary distances, per-source query overlap and observed-mask component "
    "counts accompany the report. No matched-instance boundary accuracy can "
    "be established with zero correct matches. Scan holes affect component "
    "counts and skeletons; no full-paint width or vertical accuracy is claimed.",
    "",
    "## Legacy comparison and next development decision",
    "",
    "Legacy macro F1.5927 is from different geography and overlapping crop "
    "appearances. It is retained as historical context, not a directly "
    "comparable improvement baseline. This new development region demonstrates "
    "a geographic failure of the frozen operating point; one region alone "
    "does not establish population-wide performance.",
    "Do not start another long run automatically. Keep this approved window "
    "as development. Next, prepare separate real training sections with "
    "curves/merges, narrow paint, symbols as ignored regions and bright "
    "roadside negatives; verify actual point support and obtain first approval "
    "only for new geometry. A predeclared small real-only experiment can then "
    "test class-confusion/fragmentation fixes against both legacy and this "
    "new development geography. Do not repeat synthetic exposure runs without "
    "new evidence, retire this validation window only under an explicit "
    "reassignment plan, and keep final-test data out of all selection.",
    "",
    f"Measured runtime: `{result['runtime']}`. Model-only time is not full "
    "throughput; total includes startup/export, inference, reconciliation, "
    "metrics and previews. Per-region and per-crop metrics are saved separately.",
    "",
    f"Full regression: {int(suite.attrib['tests'])-int(suite.attrib['skipped'])} "
    f"passed, {suite.attrib['skipped']} optional network test skipped. Ruff, Black "
    "and local healthcheck passed. Original model/source/approval hashes and "
    "native crop validity were checked. No final GIS digitization was begun.",
]
(report / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(
    json.dumps(
        {
            "report": str(report / "report.md"),
            "counts": counts,
            "pixel": pixel,
            "raw_audit": summary,
        },
        indent=2,
    )
)
