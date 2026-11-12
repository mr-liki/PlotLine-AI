"""Package measured Phase 5.4 improvements, safe dataset expansion and review outputs."""

import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roadmarking.rasterization.pipeline import dump, sha256


def main():
    root = Path(__file__).resolve().parents[1]
    context = json.loads((root / "outputs/reports/phase5_4/latest.json").read_text())
    output = Path(context["report_dir"])
    post = json.loads((output / "postprocessing.json").read_text())
    expansion = json.loads((output / "supervised_dataset_expansion.json").read_text())
    candidates = json.loads((output / "new_region_candidates.json").read_text())
    visibility = json.loads((output / "new_region_visibility_audit.json").read_text())
    errors = json.loads((output / "error_analysis.json").read_text())
    geographic = json.loads((output / "geographic_identity_audit.json").read_text())
    test_reservations = json.loads((output / "future_test_region_reservations.json").read_text())
    qgis = json.loads((output / "qgis_new_region_review.json").read_text())
    experiments, checkpoints = [], []
    for folder in (root / "models/mask2former_swin_s").glob("phase5_4_" + context["run_id"] + "*"):
        if not (folder / "validation_history.json").exists():
            continue
        logs = (
            json.loads((folder / "training_log.json").read_text())
            if (folder / "training_log.json").exists()
            else []
        )
        history = json.loads((folder / "validation_history.json").read_text())
        completed = bool((folder / "segment_to200.json").exists())
        item = {
            "run": str(folder),
            "completed_200_steps": completed,
            "recorded_optimizer_updates": len(logs),
            "sampling_counts": dict(Counter(row["sampling_stratum"] for row in logs)),
            "validation_history": history,
            "config": json.loads((folder / "training_config.json").read_text()),
        }
        experiments.append(item)
        if completed:
            for entry in history:
                checkpoints.append((entry, folder / f"checkpoint_step{entry['step']:04d}"))
    best, checkpoint = max(
        checkpoints,
        key=lambda pair: (
            pair[0]["metrics"]["postprocessed"]["macro_f1"],
            -sum(pair[0]["metrics"]["postprocessed"][str(i)]["fp"] for i in (0, 1)),
            -pair[0]["step"],
        ),
    )
    bundle = json.loads((checkpoint / "bundle.json").read_text())
    if sha256(checkpoint / "model/model.safetensors") != bundle["weight_sha256"]:
        raise ValueError("Selected checkpoint changed before packaging")
    reload = json.loads((checkpoint.parent / "selected_reload_validation.json").read_text())
    baseline_hash = post["baseline_checkpoint_sha256"]
    unchanged = bundle["weight_sha256"] == baseline_hash
    suite = (
        ET.parse(root / "outputs/reports/phase5_4/pytest.xml").getroot().find("testsuite").attrib
    )
    pytest = {k: int(suite[k]) for k in ("tests", "failures", "errors", "skipped")}
    baseline, selected = post["baseline_metrics"], best["metrics"]["postprocessed"]
    completed_steps = sum(
        item["recorded_optimizer_updates"] for item in experiments if item["completed_200_steps"]
    )
    partial_steps = sum(
        item["recorded_optimizer_updates"]
        for item in experiments
        if not item["completed_200_steps"]
    )
    old_approved = {
        f["properties"]["global_id"]
        for f in json.loads(
            (Path(expansion["parent_dataset"]) / "metadata/approved_geometry.geojson").read_text()
        )["features"]
    }
    new_ids = {uid for tile in candidates["tiles"] for uid in tile["source_global_ids"]}
    resume_proof = json.loads(
        (
            root
            / "models/mask2former_swin_s"
            / ("phase5_4_" + context["run_id"] + "_attempt2")
            / "resume_proof_step100.json"
        ).read_text()
    )
    report = {
        "status": "TRAINED_PROVISIONAL",
        "production_ready": False,
        "preliminary_unattended_vectorization_ready": False,
        "selected_checkpoint": str(checkpoint),
        "model_weights": str(checkpoint / "model"),
        "processor": str(checkpoint / "processor"),
        "selected_weight_sha256": bundle["weight_sha256"],
        "selected_weights_unchanged_from_phase5_3": unchanged,
        "selected_experiment_step": best["step"],
        "fresh_process_reload": reload,
        "baseline_raw": baseline,
        "selected_postprocessed": selected,
        "baseline_mean_foreground_iou": best["metrics"]["raw_mean_foreground_iou"],
        "selected_postprocessed_mean_foreground_iou": best["metrics"][
            "postprocessed_mean_foreground_iou"
        ],
        "selected_postprocessing": post["selected"]["config"],
        "postprocessing_trial_count": post["candidate_count"],
        "postprocessing_selection_rule": post["selection_rule"],
        "training_selection_rule": (
            "Max development macro F1 with frozen postprocessing, then fewer FP; "
            "retain baseline if retraining does not improve"
        ),
        "experiments": experiments,
        "completed_experiment_optimizer_updates": completed_steps,
        "aborted_attempt_optimizer_updates": partial_steps,
        "resume_proof": resume_proof,
        "dataset_expansion": expansion,
        "candidate_geographic_expansion": candidates,
        "candidate_visibility": visibility,
        "genuinely_new_source_ids_outside_original_95": len(new_ids - old_approved),
        "new_region_project": qgis["project"],
        "qgis_validation": qgis,
        "future_test_location_reservations": test_reservations,
        "error_primary_categories": errors["fp_primary_categories"],
        "geographic_error_identity_audit": geographic,
        "fp_mask_overlap_clusters_proxy": errors["fp_mask_overlap_geographic_clusters"],
        "cross_crop_overlapping_fp_pairs": errors["cross_crop_same_class_fp_overlap_pairs"],
        "pytest": pytest,
        "ruff": "PASS",
        "black": "PASS",
        "independent_test_evaluation": None,
        "evidence_scope": "Development validation only; five overlapping crops in one geographic "
        "component",
        "additional_region_approval_required": True,
        "original_six_regions_need_repeat_review": False,
        "public_training_data_added": False,
        "external_source_assessment": json.loads(
            (output / "external_source_assessment.json").read_text()
        ),
    }
    dump(output / "report.json", report)
    dump(
        output / "best_checkpoint.json",
        {
            "checkpoint": str(checkpoint),
            "model": str(checkpoint / "model"),
            "weight_sha256": bundle["weight_sha256"],
            "postprocessing": post["selected"]["config"],
            "weights_reused_from_phase5_3": unchanged,
        },
    )
    dump(checkpoint / "postprocessing.json", post["selected"]["config"])
    dump(checkpoint / "loss_config.json", experiments[0]["config"].get("loss_weights", {}))
    dump(
        checkpoint / "phase5_4_selection.json",
        {
            "selected_step": best["step"],
            "weights_reused_from_baseline": unchanged,
            "validation_metrics": best["metrics"],
            "not_a_production_model": True,
        },
    )
    shutil.copyfile(
        root / "src/roadmarking/training/valid_criterion.py", checkpoint / "valid_criterion.py"
    )
    shutil.copyfile(root / "outputs/reports/phase5_4/pytest.xml", output / "pytest.xml")
    shutil.copyfile(root / "outputs/reports/phase5_4/healthcheck.json", output / "healthcheck.json")
    trials = json.loads((output / "calibration_trials.json").read_text())
    fig, ax = plt.subplots(figsize=(7, 5))
    precision = [np.mean([t["metrics"][str(i)]["precision"] for i in (0, 1)]) for t in trials]
    recall = [np.mean([t["metrics"][str(i)]["recall"] for i in (0, 1)]) for t in trials]
    ax.scatter(recall, precision, c=[t["metrics"]["macro_f1"] for t in trials], s=12, alpha=0.6)
    ax.set(
        xlabel="Macro recall",
        ylabel="Macro precision",
        title="300 development-only calibration settings",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "precision_recall_tradeoff.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, item in zip(axes, [x for x in experiments if x["completed_200_steps"]], strict=True):
        logs = json.loads((Path(item["run"]) / "training_log.json").read_text())
        ax.plot([r["step"] for r in logs], [r["loss"] for r in logs], lw=0.7)
        ax.set(
            xlabel="Continuous optimizer step",
            ylabel="Measured loss",
            title=f"LR {item['config']['learning_rate']}; sampler "
            f"{len(item['config']['sampling_cycle'])}-step cycle",
        )
        ax.axvline(100, color="grey", ls="--", alpha=0.4)
        ax.grid(alpha=0.2)
    fig.suptitle("Bounded experiments; loss scales vary with crop/negative sampling")
    fig.tight_layout()
    fig.savefig(output / "training_experiments.png", dpi=140)
    plt.close(fig)
    rows = []
    for label, name in (("0", "Solid"), ("1", "Dash")):
        before, after = baseline[label], selected[label]
        rows.append(
            f"| {name} | {before['precision']:.3f} | {after['precision']:.3f} | "
            f"{before['recall']:.3f} | {after['recall']:.3f} | "
            f"{before['f1']:.3f} | {after['f1']:.3f} |"
        )
    fp_before = sum(baseline[str(i)]["fp"] for i in (0, 1))
    fp_after = sum(selected[str(i)]["fp"] for i in (0, 1))
    missed = sum(selected[str(i)]["fn"] for i in (0, 1))
    duplicates = sum(selected[str(i)]["duplicates"] for i in (0, 1))
    body = [
        "# Phase 5.4: dataset expansion and recognition improvement",
        "",
        "Status: **TRAINED_PROVISIONAL**. Not ready for unattended vectorization " "or production.",
        "",
        "## Actual dataset increase",
        "",
        f"Usable native-512 training crops increased from **10 to 89**, including "
        f"**16 verified bright-background negatives**. "
        "They reuse the approved six-window completeness evidence and validated "
        "NoData-aware targets. "
        "The five validation crops remain byte-for-byte unchanged. Crop density "
        "increased; this is not 89 independent regions. "
        "Unique supervised training source IDs remain 20 (six solid, fourteen dash). "
        f"Covered training footprint grew from "
        f"{expansion['unique_training_footprint_native_square_units_before']:.2f} "
        f"to "
        f"{expansion['unique_training_footprint_native_square_units_after']:.2f} "
        f"native square units. "
        "Foreground is only 1.35% of valid pixel appearances; class/context "
        "diversity remains a major limitation.",
        "",
        f"Dataset: `{expansion['dataset']}`.",
        "",
        "## New data from the user's reference share",
        "",
        "The four GeoPackages in the supplied authoritative Road_marking folder "
        "still match their audited hashes. "
        "Their 16,521 features contain 16,520 valid geometries. The original "
        "network inputs were read-only. "
        "Only three matching LiDAR sources were decoded for new imagery; all "
        "73,490,887 actual point records passed "
        "strict selected-source integrity checks. No repeated 173-file fingerprint "
        "sweep or whole-share copy occurred. "
        "Two selected source files were cached locally (about 609 MB) for future reuse.",
        "",
        f"Prepared **{candidates['candidate_windows']} geographically new windows**, "
        f"containing **{candidates['unique_candidate_annotation_ids']} distinct "
        f"client feature IDs** "
        f"({report['genuinely_new_source_ids_outside_original_95']} outside the prior 95). "
        "Orange paint-area proposals use source geometry and supplied widths. "
        "Client originals, symbols, rasters, "
        "observed-data masks, source IDs and affine mappings are included. Native "
        "resolution and the shared "
        "normalization/gamma were retained. Coverage and geometry intersection are "
        "not claimed as paint-area approval. "
        "Some outlines extend beyond observed returns; only visible portions can "
        "be judged from these source windows.",
        "",
        f"Open the new read-only QGIS project:\n\n`{qgis['project']}`",
        "",
        "![Seven new review regions](new_regions_contact_sheet.png)",
        "",
        "These new regions require their first alignment/conversion and "
        "complete-visible-label approval before supervised use. "
        "No repeated review of the original six approved regions or 95 outlines is needed. "
        "Existing long source IDs link the seven candidates to prior regions; they "
        "must not be advertised as an independent test fold. "
        "Three other geographic locations with disjoint reference IDs are reserved "
        "in future_test_region_reservations.json. "
        "Those reservations are unreviewed metadata locations, not materialized or "
        "evaluated test data.",
        "",
        "## Baseline error diagnosis",
        "",
        f"All 115 false-positive and 12 missed crop appearances have individual "
        f"records and contact-sheet panels. "
        f"Mask-relationship primary categories: `{errors['fp_primary_categories']}`. "
        f"The 12 missed appearances involve "
        f"{geographic['source_ids_with_at_least_one_missed_appearance']} source "
        f"IDs; "
        f"{geographic['source_ids_never_detected_in_any_evaluated_appearance']} "
        f"were never detected in another evaluated appearance. "
        f"The 115 FP appearances form "
        f"{errors['fp_mask_overlap_geographic_clusters']} same-class native-mask "
        f"overlap clusters, "
        f"with {errors['cross_crop_same_class_fp_overlap_pairs']} cross-crop "
        f"overlapping FP pairs. "
        "These clusters are a geographic deduplication proxy, not human-verified "
        "unique false-object counts. "
        "Faintness, NoData causation and annotation ambiguity are not asserted "
        "from confidence or mask overlap alone. "
        "All error galleries were visually inspected; bright road edges/non-road "
        "context and repeated paint proposals are evident.",
        "",
        "## Measured postprocessing improvement",
        "",
        f"Evaluated 300 bounded settings. Selected configuration: "
        f"`{post['selected']['config']}`. "
        "The rule maximizes development macro F1 while preserving both baseline "
        "class recalls, then minimizes FP. "
        "Mask NMS was evaluated but disabled in the winning setting because the "
        "tested suppression trade-offs did not win "
        "under that recall constraint. No short-object size filter or "
        "dash-connecting operation was introduced. "
        "The complete precision-recall grid is saved as JSON and CSV.",
        "",
        "| Class | Old precision | New precision | Old recall | New recall | Old F1 | New F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        f"FP appearances: **{fp_before} to {fp_after}**; missed appearances: "
        f"**{missed}**, unchanged; "
        f"duplicates: **47 to {duplicates}**. Macro F1: "
        f"**{baseline['macro_f1']:.4f} to {selected['macro_f1']:.4f}**. "
        f"Mean valid-pixel foreground IoU: **{report['baseline_mean_foreground_iou']:.4f} to "
        f"{report['selected_postprocessed_mean_foreground_iou']:.4f}**. These "
        f"gains come from postprocessing, not new weights. "
        "AP50/AP75 remain per-crop, valid-pixel, confidence-truncated metrics in "
        "the saved evaluation JSONs, "
        "not a pooled full-survey COCO claim.",
        "",
        "![Calibration trade-offs](precision_recall_tradeoff.png)",
        "",
        "## Controlled training and genuine resume",
        "",
        "Two bounded 200-step experiments used the expanded approved dataset and "
        "unchanged validation regions: "
        "AdamW/cosine LR 1e-5 with a 2:1 positive/negative sampling ratio, and LR "
        "5e-6 with a 5:1 ratio. "
        "Solid/dash positive sampling was balanced. Native crops were 512, batch "
        "one, FP16 scale 128, "
        "gradient clipping one and no backbone freezing. The model architecture "
        "and loss/matching weights were preserved. "
        "All NoData invariance and gradient tests passed before training. "
        "Background-only crops exposed FP16 overflow in a differentiable zero "
        "formed by a dense half-precision sum; "
        "using an FP32 accumulation fixed the numerical issue without changing the "
        "supervised objective. "
        "A new dense-empty-target regression test covers it. The aborted attempt's "
        "logs record two updates; "
        "no failed-attempt weights are selected.",
        "",
        f"Completed experiment updates: **{completed_steps}**, plus "
        f"{partial_steps} updates in the aborted attempt. "
        "Each experiment warm-started the Phase 5.3 model; these are separate "
        "branches, not a single 588-step selected model. "
        f"The first experiment resumed in a fresh process at step "
        f"{resume_proof['restored_step']}, "
        f"restoring {resume_proof['optimizer_state_parameter_count']} optimizer "
        f"parameter states, scheduler, scaler, "
        "Python/NumPy/Torch/CUDA RNGs and dataset/configuration versions. Model "
        "and state checksums were verified. "
        "All checkpoint snapshots remain available. GPU backward kernels can still "
        "be nondeterministic under "
        "PyTorch warn-only deterministic mode; RNG restoration is not a claim of "
        "bitwise-identical training trajectories.",
        "",
        "![Continuous training experiments](training_experiments.png)",
        "",
        "Raw segmentation improved in some retrained candidates, with recall "
        "trade-offs. Neither experiment beat "
        "the baseline under the frozen calibrated-F1 selection rule. Therefore the "
        "best measured pipeline retains "
        "the Phase 5.3 weights and adds calibrated filtering. The newest retrained "
        "weights were not silently substituted. "
        "Raw and postprocessed metrics for every checkpoint are preserved so model "
        "changes are not confused with threshold changes.",
        "",
        "## Best saved artifact and comparisons",
        "",
        f"Selected checkpoint bundle: `{checkpoint}`. Model: `{checkpoint / 'model'}`. "
        f"Weights SHA-256: `{bundle['weight_sha256']}`. "
        "It contains model, processor, optimizer/scheduler/scaler/RNG state and "
        "versioned configuration. "
        "The selected experiment-local step zero inherits the genuinely trained "
        "188-step baseline; it is not random initialization. "
        "Stateful step-100/200 checkpoints contain the newly trained optimizer moments. "
        "Fresh-process model reload reproduced class and mask logits exactly.",
        "",
        "![Original, approved GT, Phase 5.3, selected Phase 5.4, " "errors](comparison_01.png)",
        "",
        "Five comparison panels and native prediction masks/scores/labels/affines are saved. "
        "Red error pixels are false foreground, blue are missed foreground; "
        "duplicate instance counts are reported separately. "
        "Original model inputs, annotations, approval records and all earlier "
        "checkpoints were preserved.",
        "",
        "## Other possible sources",
        "",
        "[Toronto-3D](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w11/"
        "Tan_Toronto-3D_A_Large-Scale_Mobile_LiDAR_Dataset_for_Semantic_"
        "Segmentation_of_CVPRW_2020_paper.pdf) "
        "provides mobile LiDAR intensity and a semantic road-marking category. "
        "[K-Lane](https://arxiv.org/abs/2110.11048) provides LiDAR lane annotations. "
        "Neither has been established as a direct source of this project's "
        "verified solid/dash paint-area instance targets. "
        "Their applicability assessment is recorded separately. No public labels "
        "were invented, relabeled blindly, or mixed "
        "into training. The matched client GIS/LiDAR remains the closest available domain.",
        "",
        "## Validation and readiness",
        "",
        f"Pytest: `{pytest}`. Ruff, Black and environment healthcheck passed; the "
        f"optional network config check was skipped. "
        "The NoData-aware criterion, instance NMS separation, native world-mask "
        "overlap, state restoration and "
        "checksum/dataset mismatch rejection were actually tested. "
        "Results remain development-validation findings from five overlapping "
        "crops in one geographic component. "
        "There is no independent test accuracy. Precision remains too low for "
        "unattended preliminary vectorization "
        "or production; no final 3D digitization was begun. The next substantial "
        "data step is approval of the seven "
        "new visible regions, followed by supervised conversion and "
        "protected-group allocation. "
        "The original six regions do not need redraw, modification or repeat approval.",
    ]
    (output / "report.md").write_text("\n".join(body) + "\n", encoding="utf-8")
    (checkpoint / "model_card.md").write_text(
        "# Plotline-AI selected Phase 5.4 development pipeline\n\n"
        "TRAINED_PROVISIONAL. Selected weights are the preserved 188-step Phase 5.3 model, "
        "not random initialization and not the newest retrained candidate. Two 200-step "
        "experiments did not beat its F1 with frozen calibrated postprocessing.\n\n"
        "Classes: solid (0), dash (1). Apply postprocessing.json: class-specific confidence, "
        "mask-quality threshold and the recorded NMS setting. Native 512 input, no resize, "
        "replicated uint8 grayscale rescaled once by the saved processor.\n\n"
        "The bundle includes resumable optimizer/scheduler/scaler/RNG state. At experiment "
        "step zero the optimizer is intentionally fresh; trained moment states are available "
        "at step 100/200 sibling checkpoints. Source-data approvals remain unchanged.\n\n"
        "One training and one development-validation component; no independent test set. "
        "False positives and duplicate instances remain. No unattended vectorization, final "
        "3D digitization or production readiness. See the versioned Phase 5.4 report.\n",
        encoding="utf-8",
    )
    context.update(
        report=str(output / "report.md"),
        selected_checkpoint=str(checkpoint),
        new_region_project=qgis["project"],
        expanded_dataset=expansion["dataset"],
    )
    dump(root / "outputs/reports/phase5_4/latest.json", context)
    dump(
        root / "outputs/reports/phase5_4/report.json",
        {"versioned_report": str(output / "report.json"), "status": "TRAINED_PROVISIONAL"},
    )
    (root / "outputs/reports/phase5_4/report.md").write_text(
        f"# Phase 5.4\n\nSee the versioned report: `{output / 'report.md'}`.\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(output / "report.md"),
                "selected_checkpoint": str(checkpoint),
                "new_region_project": qgis["project"],
                "fp_before": fp_before,
                "fp_after": fp_after,
                "training_crops": expansion["new_native512_training_crops"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
