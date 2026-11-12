"""Publish fixed-budget ablations, data-review gate and saved-output galleries."""

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
import numpy as np
import rasterio
from scipy.ndimage import binary_erosion, distance_transform_edt, label
from skimage.morphology import skeletonize

from roadmarking.inference.calibration import iou_matrix, operating_counts
from roadmarking.rasterization.pipeline import dump, sha256

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
context_path = root / "outputs/reports/phase5_8/latest.json"
context = json.loads(context_path.read_text())
report, dataset = Path(context["report_dir"]), Path(context["dataset"])
experiments = json.loads((report / "experiments.json").read_text())
comparisons, geometry = [], []
for experiment in experiments:
    folder = Path(experiment["output"])
    history = json.loads((folder / "validation_history.json").read_text())
    logs = json.loads((folder / "training_log.json").read_text())
    assert (
        len(logs) == 100
        and sum(x["sampling_stratum"] == "synthetic" for x in logs) == experiment["synthetic_steps"]
    )
    for entry in history:
        m = entry["metrics"]
        ap = {}
        for cls in ("longitudinal_solid", "longitudinal_dash"):
            ap[cls] = {
                k: float(
                    np.mean([v[cls][k] for v in m["postprocessed_per_crop_ap"] if k in v[cls]])
                )
                for k in ("AP50", "AP75")
            }
        comparisons.append(
            {
                "experiment": experiment["name"],
                "step": entry["step"],
                "counts": m["postprocessed"],
                "foreground_iou": m["postprocessed_mean_foreground_iou"],
                "mean_per_crop_confidence_truncated_ap": ap,
                "model_only_seconds": m["model_only_seconds_total"],
            }
        )
    m = history[-1]["metrics"]
    for crop in m["crops"]:
        with np.load(crop["file"]) as d:
            targets, classes, valid = d["masks"].astype(bool), d["labels"], d["validity"]
        with np.load(folder / "validation_step0100" / (crop["crop_id"] + ".npz")) as p:
            masks, scores, kinds, selected = (
                p["masks"],
                p["scores"],
                p["labels"],
                p["selected"].astype(int).tolist(),
            )
        rec = {
            "masks": masks,
            "scores": scores,
            "labels": kinds,
            "targets": targets,
            "target_labels": classes,
            "valid": valid,
            "iou": iou_matrix(masks, targets, valid),
        }
        ops = operating_counts(rec, selected)
        predicted = masks[selected].any(0) & valid
        truth = targets.any(0) & valid
        skel = skeletonize(truth)
        item = {
            "experiment": experiment["name"],
            "crop": crop["crop_id"],
            "geographic_role": "REAL_DEVELOPMENT_APPEARANCE_NOT_UNIQUE_OBJECT",
            "observed_centerline_pixel_coverage_within_one_pixel": float(
                (skel & (distance_transform_edt(~predicted) <= 1)).sum() / max(1, skel.sum())
            ),
            "matched_instances": [],
        }
        for cls in ("0", "1"):
            for pi, ti in ops[cls]["assignments"].items():
                a, b = masks[pi] & valid, targets[ti] & valid
                ba, bb = a & ~binary_erosion(a), b & ~binary_erosion(b)
                distances = (
                    np.r_[distance_transform_edt(~bb)[ba], distance_transform_edt(~ba)[bb]] * 0.05
                )
                item["matched_instances"].append(
                    {
                        "class": cls,
                        "source_id": crop["global_instance_ids"][ti],
                        "mean_observed_boundary_distance_native": float(distances.mean()),
                        "observed_prediction_components": int(label(a)[1]),
                        "observed_target_components": int(label(b)[1]),
                        "boundary_pixels_within_one_pixel_fraction": float(
                            (distances <= 0.05).mean()
                        ),
                    }
                )
        geometry.append(item)
    # Actual final network outputs, one example per experiment, compared to common baseline.
    crop = next(c for c in m["crops"] if c["tile_id"] != "NEW_DEV_REVIEW")
    d = np.load(crop["file"])
    before = np.load(folder / "validation_step0000" / (crop["crop_id"] + ".npz"))
    after = np.load(folder / "validation_step0100" / (crop["crop_id"] + ".npz"))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    overlays = [
        np.zeros(d["validity"].shape, bool),
        d["masks"].any(0),
        before["masks"][before["selected"].astype(int)].any(0),
        after["masks"][after["selected"].astype(int)].any(0),
    ]
    for ax, title, mask in zip(
        axes,
        [
            "Intensity",
            "Approved real targets",
            "Common starting model",
            experiment["name"] + " step100",
        ],
        overlays,
        strict=True,
    ):
        ax.imshow(d["rgb"])
        mask = mask.astype(bool) & d["validity"]
        ax.imshow(np.ma.masked_where(~mask, mask), cmap="winter", alpha=0.7, vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(report / (experiment["name"] + "_real_comparison.png"), dpi=130)
    plt.close(fig)
dump(report / "development_comparisons.json", comparisons)
dump(report / "observed_geometry_diagnostics.json", geometry)
adapter = json.loads((report / "synthetic_audit.json").read_text())
examples = adapter["converted_examples"]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, item in zip(axes.flat, examples, strict=False):
    d = np.load(item["file"])
    ax.imshow(d["rgb"])
    mask = d["masks"].any(0)
    ax.imshow(np.ma.masked_where(~mask, mask), cmap="autumn", alpha=0.6, vmin=0, vmax=1)
    ax.set_title(Path(item["source"]).stem)
    ax.axis("off")
fig.tight_layout()
fig.savefig(report / "synthetic_adapter_gallery.png", dpi=120)
plt.close(fig)
new = json.loads((report / "new_development_search.json").read_text())
review = json.loads((root / "outputs/reports/phase5_8_actual_review/latest.json").read_text())
review_output = Path(review["output"])
dump(
    report / "new_review_artifact_hashes.json",
    {str(p): sha256(p) for p in review_output.rglob("*") if p.is_file()},
)
with rasterio.open(new["support"]["layers"]["observed"]) as ds:
    assert ds.transform.a == 0.05 and ds.transform.e == -0.05 and ds.read(1).sum() > 0
suite = ET.parse(root / "outputs/reports/phase5_8_pytest.xml").getroot().find("testsuite")
assert suite.attrib["failures"] == "0" and suite.attrib["errors"] == "0"
frozen = json.loads(
    (root / "outputs/reports/phase5_6a/20261106T093757339115Z/frozen_model.json").read_text()
)
for filename, expected in frozen["artifact_sha256"].items():
    assert sha256(filename) == expected
result = {
    "status": "REAL_DATA_REVIEW_REQUIRED",
    "starting_checkpoint_retained": context["initialization"],
    "new_review_project": str(review_output / "independent_test_review.qgs"),
    "new_real_support": new["support"],
    "comparisons": comparisons,
    "final_test_used": False,
    "historical_diagnostic_benchmark_rerun": False,
    "synthetic_policy": "Augmentation only; no synthetic GIS scale or real accuracy claims",
    "experiments": experiments,
    "pytest": dict(suite.attrib),
    "production_ready": False,
}
dump(report / "report.json", result)
for name in (
    "phase5_8_ruff.log",
    "phase5_8_black.log",
    "phase5_8_tests.log",
    "phase5_8_pytest.xml",
    "phase5_8_healthcheck.json",
    "phase5_8_healthcheck.log",
):
    shutil.copy2(root / "outputs/reports" / name, report / name)
lines = [
    "# Phase 5.8 — REAL_DATA_REVIEW_REQUIRED",
    "",
    "The Phase 5.7 expanded checkpoint is retained. No equal-budget ablation beat "
    "its real development macro F1. Protected final-test images/labels were not "
    "used for training, threshold optimization or evaluation. Historical diagnostic "
    "regions were not rerun or relabeled as fresh tests.",
    "",
    "## Real geographic coverage",
    "",
    "The approved development addition has7732 observed pixels and only17 observed "
    "supported paint pixels across a wide area. Native paint-centred512/384/256 "
    "crops contain16/14/13 paint pixels respectively, still without meaningful "
    "observed context. No lower eligibility threshold or fabricated fill was used "
    "to manufacture positive validation. Existing approvals remain untouched.",
    "The first header-selected candidate had zero actual observed points and was "
    "rejected. Actual-point search in two previously validated acquisitions found "
    "no suitable independent positive candidate. A bounded search of two additional "
    "authoritative acquisitions then recovered the new review window below. "
    "Any failed source was rejected as a whole; partial decodes were not admitted.",
    "",
    f"New first-review project: `{result['new_review_project']}`.",
    "Bookmark PHASE58_POSITIVE_DEV_REVIEW contains8 observed solid IDs,11 dash IDs "
    "and7180 observed paint pixels, with29.99% observed support. It is separated "
    "from all nineteen reviewed regions by >=500 native units and shares no client "
    "marking IDs. Ground truth and completeness approval are pending for this "
    "genuinely new observed window only. No new candidate was used to select models.",
    "",
    "## Synthetic conversion and domain gap",
    "",
    "Both user-supplied collections were inspected:500 Volume1 and1000 Volume2 "
    "image/instance/semantic triplets. All original inputs remain unchanged. "
    "Source hashes, metadata alignment, duplicates, empty scenes, classes and "
    "conversion totals are recorded in synthetic_audit.json and source_hashes.json.",
    "Synthetic0 dash maps to model1; synthetic1 solid maps to model0. Classes2–15 "
    "and a two-pixel margin are excluded, never ordinary background. Zero-valued "
    "generator dropout is ignored. Dashes split by8-connected visible components, "
    "with minimum12 pixels and span6; small artifacts are ignored. Groups with "
    "components closer than4 pixels are conservatively ignored because they may "
    "be dropout fragments. Original group IDs and component IDs remain in metadata. "
    "No arbitrary gap bridging or fabricated native GSD is introduced. Full synthetic "
    "validation/test splits remain out of training and real evaluation.",
    "Measured intensity means differ: real paint217.05/background81.07; synthetic "
    "paint147.01/background60.59. These are observed-mask/crop distributions, not "
    "pavement-only estimates or independent geographic counts. Procedural speckle "
    "and dropout remain; no cleaning/rescaling is added. Physical width, dash spacing "
    "and curvature equivalence are unestablished; no synthetic metric GIS claim.",
    "",
    "## Controlled real-development ablations",
    "",
    "Each run uses the same selected checkpoint, real train/development data, "
    "native512 resolution, fixed confidence/NMS settings, seed895, AdamW lr1e-6, "
    "FP16 scale16, gradient clip1 and100 optimizer steps. Synthetic exposure is "
    "exactly0/10/25 steps. Real sampling is held in the same relative proportions. "
    "No new loss: valid-only Hungarian/BCE/Dice/auxiliary and partial-observation "
    "classification behavior are preserved. Full resumable states saved at0/50/100. "
    "These are small pilots, not complete epochs over1200 synthetic train assets.",
    "",
    "| Run | Step | Real development macro F1 | Mean foreground IoU |",
    "|---|---:|---:|---:|",
]
for c in comparisons:
    lines.append(
        f"| {c['experiment']} | {c['step']} | "
        f"{c['counts']['macro_f1']:.4f} | {c['foreground_iou']:.4f} |"
    )
lines += [
    "",
    "The common starting score is0.5927; no experiment improved it. These remain "
    "overlapping development appearances. Real geographic generalization remains "
    "unverified until the new positive development window is reviewed and used "
    "under a documented selection procedure. The protected final test stays sealed.",
    "",
    "## Diagnostics, runtime and QA",
    "",
    "development_comparisons.json includes per-class P/R/F1, AP50/AP75 and IoU. "
    "AP is confidence-truncated mean per-crop101-point AP, not pooled COCO AP. "
    "Observed skeleton coverage, boundary alignment and component counts are "
    "conditional sparse-support diagnostics; holes prevent claims of complete "
    "physical width/continuity or vertical accuracy. Before/after real galleries "
    "use stored network masks, not synthetic examples. Postprocessing was unchanged.",
    "Runtime records include optimizer-step times, segment elapsed time, peak "
    "allocation/reservation, adapter time and model-only inference time. Training "
    "loads/preprocessing and inference mask processing are not separately isolated "
    "by the existing trainer; no model-only speed is presented as full throughput.",
    f"Full pytest: {int(suite.attrib['tests'])-int(suite.attrib['skipped'])} passed, "
    f"{suite.attrib['skipped']} optional network test skipped. Ruff, Black and local "
    "health results accompany the report. Adapter tests cover class reversal, "
    "group splitting, unsupported/NoData exclusion, tiny artifacts and ambiguous "
    "dropout gaps. Existing loss/gradient/split/native-affine checks were rerun.",
    "",
    "No protected final evaluation or production3D digitization was performed. "
    "Next gate is first geometry/completeness approval of the single new "
    "PHASE58_POSITIVE_DEV_REVIEW window; no prior reviews need repeating.",
]
(report / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
context.update(
    status=result["status"],
    report=str(report / "report.md"),
    new_review_project=result["new_review_project"],
)
dump(context_path, context)
print(
    json.dumps(
        {
            "status": result["status"],
            "report": str(report / "report.md"),
            "review": result["new_review_project"],
        },
        indent=2,
    )
)
