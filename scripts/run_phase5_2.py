"""Export approved outlines, audit valid supervision, and fail closed before training."""

import argparse
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from roadmarking.dataset.approved_export import export_approved
from roadmarking.rasterization.pipeline import dump
from roadmarking.training.supervision_audit import audit_stock_criterion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.approval = args.approval.resolve()
    root = Path(__file__).resolve().parents[1]
    reports = root / "outputs/reports/phase5_2"
    if (reports / "report.json").exists():
        raise FileExistsError("Preserve previous Phase 5.2 reports; choose a new report version")
    start = time.perf_counter()
    ident = "phase5_2_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or root / "data/processed/phase5_training" / ident
    manifest = export_approved(args.approval, output, root / "configs/phase5_2_class_mapping.json")
    loss = audit_stock_criterion()
    counts = Counter()
    for sample in manifest["samples"]:
        counts.update(sample["label_state_pixels"])
    report = {
        "status": "SUPERVISION_INCOMPLETE",
        "outlines_approved": True,
        "dataset": str(output),
        "approval": str(args.approval),
        "approval_reconciliation": str(output / "metadata/approval_reconciliation.json"),
        "approved_outlines": manifest["approved_annotation_count"],
        "unique_outline_counts_by_visual_id": manifest["unique_outline_counts_by_visual_id"],
        "exported_image_mask_pairs": len(manifest["samples"]),
        "safe_training_samples": 0,
        "verified_complete_regions": 0,
        "pixel_counts_tile_appearances": dict(counts),
        "class_mapping": manifest["class_mapping"],
        "spatial_groups": manifest["splits"],
        "criterion_audit": loss,
        "preparation_seconds": time.perf_counter() - start,
        "model_checkpoint": "facebook/mask2former-swin-small-coco-instance",
        "checkpoint_initialized": False,
        "gpu_model_memory_measured": False,
        "optimizer_steps": 0,
        "trained_checkpoint": None,
        "evaluation": None,
        "recognition_outputs": [],
        "protected_hash_mismatches": manifest["protected_hash_mismatches"],
        "blockers": [
            "No recorded complete-region or verified-background geometry",
            "No fully supervised native 512/384/256 crop; see sample crop audits",
            "Stock matching/BCE/Dice/auxiliary losses cannot exclude arbitrary unknown pixels",
            "Unknown instances can receive incorrect no-object supervision "
            "even with masked pixel losses",
        ],
        "next_requirement": (
            "Establish complete-label/background scope for selected observed regions "
            "through existing coverage/QC evidence or new region-level attestation. "
            "Approved outlines need no redraw or repeat review. Then implement and "
            "validate NoData-aware matching and all losses, or select genuinely "
            "fully observed safe crops."
        ),
    }
    dump(reports / "report.json", report)
    dump(reports / "loss_audit.json", loss)
    text = [
        "# Phase 5.2 approved outline export",
        "",
        "Status: **SUPERVISION_INCOMPLETE**.",
        "",
        "## Approval preserved",
        "",
        f"Existing user attestation: `{args.approval}`. All seven approved artifact hashes match. "
        "Displayed/source raster equality and historical source hashes were checked locally. "
        "No network source rescan, invented human checklist or transfer to changed geometry. "
        "The symbols-enabled approved project supersedes the older project named in the request.",
        "",
        "## Export and alignment",
        "",
        f"Dataset: `{output}`. {report['approved_outlines']} approved outlines; "
        f"{report['exported_image_mask_pairs']} positive image-mask pairs. "
        "Original geometry, holes, "
        "dash identity, native 0.05 sampling and affine mappings are retained. RGB source bytes "
        "are copied unchanged; normalized float images retain NoData. QA previews are separate. "
        "Approval scope covers only the six reviewed windows, not every original GIS feature.",
        "",
        "## Supervision audit",
        "",
        "```json",
        json.dumps(dict(counts), indent=2),
        "```",
        "",
        "Counts include overlapping tile appearances. Zero verified background/complete regions. "
        "Observed unlabeled pixels remain unknown. NoData, padding, boundary and conflict states "
        "remain distinct. Positive validity masks are not safe full instance-loss masks. "
        "Every native 512/384/256 crop position was tested; "
        "none meets stock supervision requirements.",
        "",
        "## Classes and spatial splits",
        "",
        json.dumps(report["unique_outline_counts_by_visual_id"], indent=2),
        "",
        "Solid/dash retain visual 1/2 and model 0/1. Original arrow/give-way code hints remain "
        "provisional, with no model training ID assigned. Their approved footprints are retained "
        "as positive geometry/mask evidence, not silently promoted to a validated class migration. "
        "Spatial connected components combine the 10-native-unit guard with shared global IDs. "
        "No train/validation/test assignment was forced onto unsafe examples.",
        "",
        "## Installed loss and engineering decision",
        "",
        "The installed criterion was inspected and tested without weights. Changing an assumed "
        "ignored pixel changes stock BCE and Dice. Hungarian matching samples unrestricted pixels. "
        "Mask losses have no validity argument; auxiliary stages recurse through the same losses. "
        "Unmatched queries receive no-object gradients, including possible unannotated markings. "
        "A pixel-only custom loss cannot resolve the missing completeness evidence. A custom "
        "criterion was therefore not presented as a solution to this semantic prerequisite.",
        "",
        "## Model, GPU, training, evaluation and recognition",
        "",
        "Not executed: no safe training samples. Zero optimizer steps, no trained checkpoint, "
        "no measured model GPU memory, no independent accuracy or recognition images. The fixed "
        "Swin-S architecture is preserved. Trainer completion remains dependent work after the "
        "supervision prerequisite; no executable end-to-end engine is claimed.",
        "",
        "## Exact remaining requirement",
        "",
        report["next_requirement"],
        "",
        f"Preparation duration: {report['preparation_seconds']:.3f} seconds. "
        "Original annotations, approved QGIS project, rasters and prior checkpoints "
        "were not modified.",
    ]
    (reports / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "dataset",
                    "approved_outlines",
                    "exported_image_mask_pairs",
                    "pixel_counts_tile_appearances",
                )
            },
            indent=2,
        )
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
