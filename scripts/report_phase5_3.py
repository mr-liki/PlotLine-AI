"""Assemble Phase 5.3 evidence from actual supervised exports and trained runs."""

import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers

from roadmarking.rasterization.pipeline import dump, sha256


def main(run):
    root = Path(__file__).resolve().parents[1]
    reports = root / "outputs/reports/phase5_3"
    run = Path(run).resolve()
    result = json.loads((run / "result.json").read_text())
    evaluation = json.loads((run / "validation_evaluation.json").read_text())
    reload = json.loads((run / "reload_validation.json").read_text())
    dataset = Path(result["training_config"]["dataset"])
    manifest = json.loads((dataset / "metadata/dataset_manifest.json").read_text())
    readiness = json.loads((reports / "fresh_dataset_readiness.json").read_text())
    initialization = json.loads((reports / "model_initialization.json").read_text())
    chain_names = [
        "phase5_3_pilot_native512_fp16scale128",
        "phase5_3_scope_pilot_native512_100steps",
        run.name,
    ]
    chain = []
    for name in dict.fromkeys(chain_names):
        folder = root / "models/mask2former_swin_s" / name
        measured = json.loads((folder / "result.json").read_text())
        logs = json.loads((folder / "training_log.json").read_text())
        chain.append(
            {
                "run": str(folder),
                "optimizer_steps": len(logs),
                "first_loss": logs[0]["loss"],
                "last_loss": logs[-1]["loss"],
                "mean_step_seconds": float(np.mean([row["seconds"] for row in logs])),
                "elapsed_seconds": measured["elapsed_seconds"],
                "checkpoint_sha256": measured["weight_sha256"],
                "logs": logs,
            }
        )
    total_steps = sum(c["optimizer_steps"] for c in chain)
    fig, axes = plt.subplots(1, len(chain), figsize=(14, 4), squeeze=False)
    for ax, entry in zip(axes[0], chain, strict=True):
        logs = entry["logs"]
        ax.plot([r["optimizer_step"] for r in logs], [r["loss"] for r in logs], lw=1)
        ax.set_title(f"{entry['optimizer_steps']} optimizer steps")
        ax.set_xlabel("Step within run")
        ax.set_ylabel("Weighted training loss")
        ax.grid(alpha=0.2)
    fig.suptitle("Measured training losses; changing crops and criterion versions")
    fig.tight_layout()
    fig.savefig(reports / "training_loss.png", dpi=140)
    plt.close(fig)
    pytest = ET.parse(reports / "pytest_final.xml").getroot().find("testsuite").attrib
    tests = {k: int(pytest[k]) for k in ("tests", "failures", "errors", "skipped")}
    crop_counts = Counter((c["size"], c["split"]) for c in manifest["crops"])
    source_hashes = {
        str(root / p): sha256(root / p)
        for p in (
            "src/roadmarking/training/valid_criterion.py",
            "src/roadmarking/training/pilot.py",
            "src/roadmarking/dataset/complete_regions.py",
            "scripts/phase5_3_initialize.py",
        )
    }
    failures = []
    for filename in (root / "models/mask2former_swin_s").glob("phase5_3*/failure.json"):
        failure = json.loads(filename.read_text())
        failures.append(
            {
                "attempt": str(filename.parent),
                "type": failure["type"],
                "error": failure["error"],
                "cuda_peak_allocated_bytes": failure.get("cuda_peak_allocated_bytes"),
            }
        )
    training_metrics = result["training_preview"]["metrics"]
    for name in ("longitudinal_solid", "longitudinal_dash"):
        if training_metrics[name].get("status") != "NOT_EVALUATED":
            training_metrics[name]["status"] = "TRAINING_PREVIEW_ONLY_NOT_INDEPENDENT_ACCURACY"
    report = {
        "status": "TRAINED_PROVISIONAL",
        "production_ready": False,
        "phase6_ready": False,
        "checkpoint": result["checkpoint"],
        "processor": result["processor"],
        "checkpoint_sha256": result["weight_sha256"],
        "checkpoint_reload": reload,
        "source_checkpoint": initialization,
        "dataset": str(dataset),
        "dataset_readiness": readiness,
        "region_completeness_attestation": str(reports / "region_completeness_attestation.json"),
        "preserved_outlines": manifest["preserved_outline_count"],
        "verified_background_pixel_appearances": manifest["verified_background_pixels"],
        "valid_paint_pixel_appearances": manifest["valid_paint_pixels"],
        "ignored_pixel_appearances": manifest["ignored_pixels"],
        "crop_counts_by_size_split": {
            f"{size}_{split}": count for (size, split), count in crop_counts.items()
        },
        "training_native_size": 512,
        "spatial_split": manifest["splits"],
        "trained_classes": manifest["training_classes"],
        "excluded_visual_classes": [3, 6],
        "training_runs": chain,
        "total_successful_optimizer_steps": total_steps,
        "training_only_preview_metrics": training_metrics,
        "independent_validation": evaluation,
        "independent_test_metrics": None,
        "gpu_peak_allocated_bytes": max(
            json.loads((Path(c["run"]) / "result.json").read_text())["cuda_peak_allocated_bytes"]
            for c in chain
        ),
        "gpu_peak_reserved_bytes": max(
            json.loads((Path(c["run"]) / "result.json").read_text())["cuda_peak_reserved_bytes"]
            for c in chain
        ),
        "criterion": result["training_config"]["criterion"],
        "failed_attempts": failures,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
        },
        "source_code_sha256": source_hashes,
        "pytest": tests,
        "healthcheck": json.loads((reports / "healthcheck.json").read_text()),
        "limitations": [
            "Only two trained visual classes; approved symbols remain preserved but excluded",
            "One training and one validation geographic component; no independent test set",
            "Overlapping crop counts are appearances, not independent samples",
            "Validation used during engineering development; no blind test accuracy claimed",
            "Partial-image unmatched queries have valid-pixel null-mask loss but no "
            "global no-object classification loss",
            "False positives, duplicate instances and narrow-marking misses remain",
            "AP is per-crop valid-pixel 101-point AP at a 0.25 confidence floor, not "
            "full-survey COCO AP",
            "Optimizer moments not saved; checkpoint is reloadable for inference, not "
            "an exact optimizer-state resume",
            "No final vector/3D digitization; no production-ready claim",
        ],
    }
    dump(reports / "report.json", report)
    dump(
        reports / "latest.json",
        {
            "run": str(run),
            "checkpoint": result["checkpoint"],
            "report": str(reports / "report.json"),
        },
    )
    dump(
        run / "checkpoint_provenance.json",
        {
            "official_initialization": initialization,
            "training_chain": [{k: v for k, v in entry.items() if k != "logs"} for entry in chain],
            "software": report["software"],
            "source_code_sha256": source_hashes,
        },
    )
    dump(run / "loss_config.json", report["criterion"])
    dump(run / "validation_metrics.json", evaluation)
    dump(
        run / "training_metrics.json",
        {"optimizer_steps_total": total_steps, "training_preview": training_metrics},
    )
    dump(
        run / "test_metrics.json",
        {"status": "NOT_EVALUATED", "reason": "No independent test group"},
    )
    shutil.copyfile(
        reports / "region_completeness_attestation.json",
        run / "region_completeness_attestation.json",
    )
    shutil.copyfile(
        root / "src/roadmarking/training/valid_criterion.py", run / "valid_criterion.py"
    )
    rows = []
    for name, metrics in evaluation["aggregate"].items():
        rows.append(
            f"| {name} | {metrics['precision_iou50']:.3f} | {metrics['recall_iou50']:.3f} | "
            f"{metrics['f1_iou50']:.3f} | {metrics['false_positives']} | "
            f"{metrics['missed_instances']} | "
            f"{metrics['duplicate_detections']} | {metrics['mean_per_crop_AP50']:.3f} |"
        )
    body = [
        "# Phase 5.3 execution report",
        "",
        "Status: **TRAINED_PROVISIONAL**. Phase 6 / production readiness: **NO**.",
        "",
        "## Approved complete regions and export",
        "",
        f"The user's explicit region-level completeness attestation is recorded in "
        f"`{reports / 'region_completeness_attestation.json'}`. "
        "It is linked to the exact six review-window IDs, prior approval, annotation "
        "hashes and raster hashes. "
        "All 95 original outlines and identities are preserved. No repeated QGIS "
        "review, source audit or geometry edit was required.",
        "",
        f"Dataset: `{dataset}`. Verified background: {manifest['verified_background_pixels']:,}; "
        f"valid paint: {manifest['valid_paint_pixels']:,}; ignored: "
        f"{manifest['ignored_pixels']:,} pixel appearances. "
        "Observed non-paint became background only inside the six completed regions. "
        "NoData, padding, ambiguous boundaries "
        "and unsupported symbol footprints (plus a two-pixel exclusion margin) remain "
        "excluded. Only visual 1/2 map to model 0/1. "
        "Arrow/give-way classes are NOT TRAINED / NOT EVALUATED, while their approved "
        "geometry remains archived.",
        "",
        f"Exported crops: `{report['crop_counts_by_size_split']}`. Training used "
        f"native 512 crops without resizing; "
        "384/256 fallback crops were prepared but not needed. Source 1024 grids and "
        "0.05-native-unit affines remain unchanged. "
        "Absolute CRS/units were not newly inferred.",
        "",
        "## NoData-aware matching and loss",
        "",
        "The custom criterion samples only valid native pixels with deterministic "
        "foreground/background stratification and "
        "population weights. Matching costs, BCE and Dice exclude ignored pixels. "
        "Every decoder auxiliary output uses the same "
        "criterion. Native points map directly to the model output grid; no "
        "ignored-neighbor interpolation is used in loss sampling. "
        "Changing ignored logits/targets leaves assignments and supervised losses "
        "invariant; their gradients are zero. "
        "Controlled tests cover missing paint observations, outside-paint missing "
        "data, padding, unsupported regions, multiple/partial "
        "objects and empty regions. In partial images, unmatched-query global class CE "
        "is disabled. Completed-region evidence and "
        "one-to-one matching instead permit null-mask BCE for unmatched queries on "
        "valid pixels only. This avoids labeling an "
        "excluded object as global no-object and supplies spatial negative supervision "
        "for duplicates.",
        "",
        "## Geographic allocation",
        "",
        "Whole connected components, a 10-native-unit guard and shared annotation IDs "
        "keep LEFT/RIGHT in training and INTERCHANGE "
        "in validation. No shared global instances or cross-split footprint "
        "violations. There is one geographic component per split, "
        "five native-512 validation crops with overlaps, and no independent test "
        "group. Crop appearances are not independent replicates.",
        "",
        "## Official initialization and measured GPU execution",
        "",
        f"Official checkpoint: `{initialization['checkpoint']}` at `{initialization['revision']}`. "
        "Swin-S depths [2,2,18,2] were checked. Expected class-head rows were "
        "initialized and the pretrained no-object row retained. "
        "Installed Transformers renamed/recomputed relative-position index buffers; "
        "exact values were checked against safetensors. "
        "The unused final Swin norm was tested by counterfactual forward passes; "
        "actual backbone feature maps are unchanged. "
        "A first serialized initialization had a stale 81-class criterion buffer; a "
        "new version corrected it to three entries. "
        "Prior initialization artifacts and failed attempts were retained.",
        "",
        f"Actual RTX 3050 6 GB training: **{total_steps} successful optimizer steps** "
        f"across bounded runs. "
        "AdamW 2e-5, weight decay .01, FP16, batch 1, accumulation 1, clip norm 1, seed 895. "
        "Full backbone remained trainable. Gradient checkpointing was unsupported by "
        "this model implementation. "
        "The default large FP16 scale caused nonfinite gradients and stopped before an "
        "update; scale 128 passed finite-gradient checks. "
        f"Peak allocation: {report['gpu_peak_allocated_bytes']/2**30:.3f} GiB; "
        f"peak reservation: {report['gpu_peak_reserved_bytes']/2**30:.3f} GiB. "
        "No smaller-crop memory test was needed. Each continuation reset AdamW state "
        "and recorded its warm-start checkpoint.",
        "",
        "![Measured training loss](training_loss.png)",
        "",
        "## Recognition and geographically held-out validation",
        "",
        "All five available 512 validation crops were evaluated using the trained "
        "checkpoint. Original intensity, approved GT, "
        "prediction overlays, comparison panels, pixel errors, masks, confidence "
        "scores, instance IDs and native affines are saved "
        f"under `{run / 'validation_crops'}`. The installed stock postprocessor "
        f"hard-codes a 384 intermediate grid; the final inference "
        "path instead interpolates model logits directly to the native crop size. No "
        "image downsampling or final GIS digitization occurred.",
        "",
        "| Class | Precision @.5 | Recall @.5 | F1 @.5 | FP | Missed | Duplicates | "
        "Mean crop AP50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        f"Mean per-crop foreground IoU on valid pixels: "
        f"**{evaluation['mean_foreground_IoU_valid_pixels']:.4f}**. "
        "Counts include overlapping crop appearances. AP uses 101 recall points, only "
        "valid mask pixels and a fixed .25 confidence "
        "floor; mean crop AP is not standard pooled full-survey COCO AP. Full per-crop "
        "AP/AP50/AP75 and prediction records are in "
        "report.json. Training-preview metrics are stored separately and are not "
        "independent accuracy. This is development "
        "validation, not an untouched test. Results are too limited and false "
        "positives too frequent for production or digitization.",
        "",
        "## Saved trained model and reload",
        "",
        f"Checkpoint: `{result['checkpoint']}`. SHA-256: `{result['weight_sha256']}`. "
        f"Size: {result['checkpoint_bytes']:,} bytes. Processor: `{result['processor']}`. "
        f"Fresh-process reload: `{reload}`. Deterministic runtime flags reproduce logits exactly. "
        "This is the final bounded-pilot checkpoint, not a claimed best "
        "validation-selected production model. Optimizer moments "
        "are not saved; full training reruns use the documented "
        "initialization/seed/configuration chain.",
        "",
        "## Tests and limitations",
        "",
        f"Final pytest: `{tests}`. Ruff and Black results are in "
        f"validation_results.json; environment healthcheck is in healthcheck.json. "
        "NoData criterion gates passed before every criterion version was used for "
        "training. The initial eight-step pilot and "
        "100-step conservative pilot remain available separately from the final "
        "valid-null-mask refinement. No previous source, "
        "QGIS approval, phase output or checkpoint was overwritten or deleted; no Git "
        "commit was made.",
        "",
        "Next work: improve duplicate and false-positive suppression with the "
        "validated spatial loss, broader independent "
        "geographic coverage and a defensible test set; validate unsupported class "
        "mapping before adding symbols. No new review "
        "of the already approved outlines or completed regions is requested. The "
        "original checkpoint licensing provenance remains "
        "recorded as `other`; this run performs no commercial deployment or redistribution.",
    ]
    (reports / "report.md").write_text("\n".join(body) + "\n", encoding="utf-8")
    (run / "model_card.md").write_text(
        "# Plotline-AI Mask2Former Swin-S engineering pilot\n\n"
        f"Status: TRAINED_PROVISIONAL. Official initialization: {initialization['checkpoint']} "
        f"at {initialization['revision']}. Total optimizer steps in warm-start chain: "
        f"{total_steps}.\n\n"
        "Classes: solid (0), dash (1). Approved arrows/give-way outlines are excluded, "
        "not background. "
        "Training scope: one geographic component; development validation: one separate component, "
        "five overlapping native-resolution crops; no independent test.\n\n"
        "The custom NoData-aware criterion and loss config are included beside this file. It masks "
        "matching and all mask/auxiliary losses, uses valid-domain null masks for "
        "unmatched queries, "
        "and suppresses unmatched class loss in partial images. Predicted masks require careful "
        "interpretation around missing observations.\n\n"
        "Not production-ready. False positives, duplicates, narrow/faded-paint misses and limited "
        "geographic/class coverage remain. No automatic 3D digitization. AP is per-crop and "
        "confidence-truncated, not a full-survey COCO result. See validation_metrics.json.\n\n"
        "Reload with the saved processor, do_resize=False and replicated grayscale uint8 RGB "
        "rescaled once. Native masks map back with each crop's affine. Checkpoint "
        "reload is tested; "
        "optimizer moments are not saved. Base checkpoint license provenance: other; no commercial "
        "deployment or redistribution was performed.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "checkpoint": report["checkpoint"],
                "optimizer_steps": total_steps,
                "validation": evaluation["aggregate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    main(parser.parse_args().run)
