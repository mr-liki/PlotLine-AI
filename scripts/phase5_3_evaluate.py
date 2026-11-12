"""Evaluate every available native-512 validation crop with the trained checkpoint."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from roadmarking.rasterization.pipeline import dump, sha256
from roadmarking.training.pilot import preview_and_evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    result = json.loads((args.run / "result.json").read_text())
    checkpoint = Path(result["checkpoint"])
    if sha256(checkpoint / "model.safetensors") != result["weight_sha256"]:
        raise ValueError("Checkpoint changed")
    dataset = Path(result["training_config"]["dataset"])
    manifest = json.loads((dataset / "metadata/dataset_manifest.json").read_text())
    crops = [c for c in manifest["crops"] if c["size"] == 512 and c["split"] == "val"]
    model = (
        Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint, local_files_only=True)
        .to("cuda")
        .eval()
    )
    processor = Mask2FormerImageProcessor.from_pretrained(
        result["processor"], local_files_only=True
    )
    records = []
    for crop in crops:
        output = args.run / "validation_crops" / crop["crop_id"]
        evaluated = preview_and_evaluate(model, processor, dataset, crop, "cuda", output)
        metadata = json.loads((output / "predictions.json").read_text())
        metadata["model_checkpoint"] = str(checkpoint)
        metadata["model_weight_sha256"] = result["weight_sha256"]
        metadata["dataset_version"] = manifest["dataset_version"]
        metadata["prediction_instances"] = [
            {
                "prediction_id": f"{crop['crop_id']}:{i}",
                "mask_index": i,
                "visual_class": model.config.id2label[label],
                "training_class_id": label,
                "confidence": score,
                "source_tile_id": crop["tile_id"],
                "affine": crop["affine"],
                "world_coordinates": "NATIVE_XY_UNTRANSFORMED",
            }
            for i, (label, score) in enumerate(
                zip(metadata["class_ids"], metadata["scores"], strict=True)
            )
        ]
        dump(output / "predictions.json", metadata)
        records.append(evaluated)
    aggregate = {}
    for name in ("longitudinal_solid", "longitudinal_dash"):
        totals = Counter()
        ap, ap50, ap75 = [], [], []
        for record in records:
            metrics = record["metrics"][name]
            if metrics.get("status") == "NOT_EVALUATED":
                continue
            for key in (
                "ground_truth_instances",
                "true_positives",
                "false_positives",
                "missed_instances",
                "duplicate_detections",
            ):
                totals[key] += metrics[key]
            ap.append(metrics["AP"])
            ap50.append(metrics["AP50"])
            ap75.append(metrics["AP75"])
        tp, fp, fn = totals["true_positives"], totals["false_positives"], totals["missed_instances"]
        p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
        aggregate[name] = {
            **dict(totals),
            "precision_iou50": p,
            "recall_iou50": r,
            "f1_iou50": 2 * p * r / max(1e-12, p + r),
            "mean_per_crop_AP": float(np.mean(ap)) if ap else None,
            "mean_per_crop_AP50": float(np.mean(ap50)) if ap50 else None,
            "mean_per_crop_AP75": float(np.mean(ap75)) if ap75 else None,
        }
    report = {
        "scope": "ALL_FIVE_GEOGRAPHICALLY_HELD_OUT_NATIVE512_VALIDATION_CROPS",
        "crop_count": len(records),
        "per_crop_results": records,
        "aggregate": aggregate,
        "mean_foreground_IoU_valid_pixels": float(
            np.mean([r["metrics"]["foreground_iou_on_valid_pixels"] for r in records])
        ),
        "counts_are_crop_appearances": True,
        "overlapping_crops_not_independent_replicates": True,
        "only_one_independent_validation_geographic_group": True,
        "not_a_full_survey_or_COCO_dataset_AP": True,
        "confidence_threshold": 0.25,
        "mask_threshold": 0.5,
        "prediction_reconciliation": (
            "Raw per-crop instance masks; duplicates reported, no vector digitization"
        ),
        "model_checkpoint": str(checkpoint),
        "model_weight_sha256": result["weight_sha256"],
    }
    dump(args.run / "validation_evaluation.json", report)
    print(
        json.dumps(
            {k: report[k] for k in ("crop_count", "aggregate", "mean_foreground_IoU_valid_pixels")},
            indent=2,
        )
    )
