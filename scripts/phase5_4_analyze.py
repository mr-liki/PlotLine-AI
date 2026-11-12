"""Audit every saved baseline error and calibrate native mask NMS on development data."""

import csv
import itertools
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

from roadmarking.inference.calibration import (
    aggregate_counts,
    filter_instances,
    load_saved_validation,
    native_world_iou,
    operating_counts,
)
from roadmarking.rasterization.pipeline import dump, sha256
from roadmarking.training.pilot import instance_metrics


def main():
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = root / "outputs/reports/phase5_4" / stamp
    out.mkdir(parents=True, exist_ok=False)
    baseline = json.loads((root / "outputs/reports/phase5_3/latest.json").read_text())
    run = Path(baseline["run"])
    dump(
        root / "outputs/reports/phase5_4/latest.json",
        {"report_dir": str(out), "baseline_run": str(run), "run_id": stamp},
    )
    network = root / "outputs/reports/phase5_4/network_reference_check.json"
    if network.exists():
        dump(out / network.name, json.loads(network.read_text()))
    records = load_saved_validation(run)
    baseline_settings = {"confidence": {"0": 0.25, "1": 0.25}, "mask_quality": 0.5, "nms_iou": None}
    selections = [list(range(len(r["scores"]))) for r in records]
    before = aggregate_counts(records, selections)
    if (
        sum(before[str(c)]["fp"] for c in (0, 1)) != 115
        or sum(before[str(c)]["fn"] for c in (0, 1)) != 12
    ):
        raise ValueError("Saved baseline no longer reproduces Phase 5.3 protocol")
    errors, tags, category_counts, missed_ids = [], Counter(), Counter(), set()
    error_masks = []
    for record in records:
        counts = operating_counts(record, range(len(record["scores"])))
        all_assignments = {
            i: j for label in ("0", "1") for i, j in counts[label]["assignments"].items()
        }
        for label in ("0", "1"):
            for index in counts[label]["false_indices"]:
                mask = record["masks"][index]
                same = np.where(record["target_labels"] == int(label))[0]
                other = np.where(record["target_labels"] != int(label))[0]
                same_iou = max((record["iou"][index, j] for j in same), default=0.0)
                other_iou = max((record["iou"][index, j] for j in other), default=0.0)
                assigned_truth = set(all_assignments.values())
                if any(record["iou"][index, j] >= 0.5 and j in assigned_truth for j in same):
                    category = "duplicate_proposal_within_crop"
                elif other_iou >= 0.5:
                    category = "incorrect_visual_class"
                elif same_iou >= 0.1:
                    target = record["targets"][max(same, key=lambda j: record["iou"][index, j])]
                    coverage = (mask & target & record["valid"]).sum() / max(1, target.sum())
                    category = (
                        "marking_fragmentation"
                        if coverage < 0.5
                        else "incorrect_segmentation_boundary"
                    )
                else:
                    category = "false_marking_detection"
                secondary = []
                if record["scores"][index] < 0.5:
                    secondary.append("low_confidence_below_0.5")
                if (mask & record["valid"]).sum() / max(1, mask.sum()) < 0.5:
                    secondary.append("majority_mask_outside_valid_supervision")
                tags.update(secondary)
                category_counts[category] += 1
                row = {
                    "error_id": len(errors),
                    "kind": "FP",
                    "crop_id": record["crop"]["crop_id"],
                    "prediction_index": int(index),
                    "class_id": int(label),
                    "confidence": float(record["scores"][index]),
                    "category": category,
                    "secondary_tags": secondary,
                    "same_class_best_iou": float(same_iou),
                    "other_class_best_iou": float(other_iou),
                }
                errors.append(row)
                error_masks.append((record, mask, row))
            for index in counts[label]["missed_indices"]:
                mask = record["targets"][index]
                ring = (
                    binary_dilation(mask, iterations=5)
                    & ~record["targets"].any(0)
                    & record["valid"]
                )
                contrast = (
                    float(np.median(record["rgb"][mask, 0]) - np.median(record["rgb"][ring, 0]))
                    if ring.any()
                    else None
                )
                uid = record["crop"]["global_instance_ids"][index]
                missed_ids.add(uid)
                row = {
                    "error_id": len(errors),
                    "kind": "FN",
                    "crop_id": record["crop"]["crop_id"],
                    "target_index": int(index),
                    "global_instance_id": uid,
                    "class_id": int(label),
                    "category": "missed_supported_marking",
                    "median_local_intensity_contrast": contrast,
                    "faint_paint_proxy": contrast is not None and contrast < 10,
                    "faintness_not_human_confirmed": True,
                }
                errors.append(row)
                error_masks.append((record, mask, row))
    # Mask-overlap geographic clustering is a measured proxy, not a new ground-truth
    # unique-error count. Full source instance identities remain available for FNs.
    fp_items = [item for item in error_masks if item[2]["kind"] == "FP"]
    parent = list(range(len(fp_items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cross_pairs = 0
    for i, (record, mask, row) in enumerate(fp_items):
        for j, (other, second, second_row) in enumerate(fp_items[:i]):
            if row["class_id"] != second_row["class_id"]:
                continue
            if (
                native_world_iou(
                    mask & record["valid"], record["crop"], second & other["valid"], other["crop"]
                )
                > 0.5
            ):
                parent[find(i)] = find(j)
                if row["crop_id"] != second_row["crop_id"]:
                    cross_pairs += 1
    gallery = out / "error_gallery"
    gallery.mkdir()
    for page in range((len(error_masks) + 15) // 16):
        canvas = Image.new("RGB", (800, 880), "white")
        draw = ImageDraw.Draw(canvas)
        for slot, (record, mask, row) in enumerate(error_masks[page * 16 : (page + 1) * 16]):
            ys, xs = np.where(mask & record["valid"])
            y0, y1 = max(0, int(ys.min()) - 16), min(mask.shape[0], int(ys.max()) + 17)
            x0, x1 = max(0, int(xs.min()) - 16), min(mask.shape[1], int(xs.max()) + 17)
            pixels = record["rgb"].copy().astype(float)
            gt = record["targets"].any(0) & record["valid"]
            pixels[gt] = pixels[gt] * 0.4 + np.array([0, 230, 255]) * 0.6
            selected = mask & record["valid"]
            pixels[selected] = pixels[selected] * 0.4 + np.array([255, 50, 50]) * 0.6
            image = Image.fromarray(pixels.astype(np.uint8)[y0:y1, x0:x1])
            image.thumbnail((196, 174))
            x, y = slot % 4 * 200, slot // 4 * 220
            canvas.paste(image, (x, y))
            draw.text(
                (x + 2, y + 176),
                f"{row['error_id']}: {row['kind']} class {row['class_id']}",
                fill="black",
            )
            draw.text((x + 2, y + 192), row["category"][:29], fill="black")
        canvas.save(gallery / f"page_{page + 1:02d}.png")
    dump(
        out / "error_analysis.json",
        {
            "baseline": before,
            "errors": errors,
            "fp_primary_categories": dict(category_counts),
            "secondary_tags": dict(tags),
            "fn_unique_source_instances": len(missed_ids),
            "fp_mask_overlap_geographic_clusters": len({find(i) for i in range(len(parent))}),
            "cross_crop_same_class_fp_overlap_pairs": cross_pairs,
            "semantic_causes_not_proven_by_mask_relationship": [
                "faintness",
                "NoData causation",
                "annotation ambiguity",
                "unsupported class identity",
            ],
            "all_127_error_appearances_have_gallery_panels": len(error_masks) == 127,
        },
    )
    trials = []
    best = None
    for solid, dash, quality, nms in itertools.product(
        [0.25, 0.4, 0.55, 0.7, 0.85],
        [0.25, 0.4, 0.55, 0.7, 0.85],
        [0.5, 0.65, 0.8],
        [None, 0.3, 0.5, 0.7],
    ):
        config = {"confidence": {"0": solid, "1": dash}, "mask_quality": quality, "nms_iou": nms}
        selected = [
            filter_instances(
                r["scores"], r["labels"], r["prediction_overlap"], r["qualities"], config
            )
            for r in records
        ]
        metrics = aggregate_counts(records, selected)
        safe_recall = all(metrics[str(i)]["recall"] >= before[str(i)]["recall"] for i in (0, 1))
        trial = {"config": config, "metrics": metrics, "retains_baseline_recall": safe_recall}
        trials.append(trial)
        key = (metrics["macro_f1"], -sum(metrics[str(i)]["fp"] for i in (0, 1)))
        if safe_recall and (best is None or key > best[0]):
            best = (key, trial)
    if best is None:
        raise ValueError("Calibration grid lacks a recall-preserving option")
    winner = best[1]
    settings = winner["config"]
    full_metrics = []
    for record in records:
        selected = filter_instances(
            record["scores"],
            record["labels"],
            record["prediction_overlap"],
            record["qualities"],
            settings,
        )
        full_metrics.append(
            instance_metrics(
                record["masks"][selected],
                record["scores"][selected],
                record["labels"][selected],
                record["targets"],
                record["target_labels"],
                record["valid"],
            )
        )
    dump(out / "calibration_trials.json", trials)
    dump(
        out / "postprocessing.json",
        {
            "baseline_settings": baseline_settings,
            "baseline_metrics": before,
            "selected": winner,
            "selected_per_crop_metrics": full_metrics,
            "candidate_count": len(trials),
            "selection_rule": (
                "Max macro F1 subject to preserving each baseline class recall, "
                "then fewer false positives"
            ),
            "development_only": True,
            "score_floor": 0.25,
            "mask_nms_never_merges_paint": True,
            "baseline_checkpoint_sha256": sha256(run / "pilot_checkpoint/model.safetensors"),
        },
    )
    with (out / "precision_recall_tradeoff.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "solid_threshold",
                "dash_threshold",
                "mask_quality",
                "nms_iou",
                "solid_precision",
                "solid_recall",
                "dash_precision",
                "dash_recall",
                "macro_f1",
            ]
        )
        for trial in trials:
            c, m = trial["config"], trial["metrics"]
            writer.writerow(
                [
                    c["confidence"]["0"],
                    c["confidence"]["1"],
                    c["mask_quality"],
                    c["nms_iou"],
                    m["0"]["precision"],
                    m["0"]["recall"],
                    m["1"]["precision"],
                    m["1"]["recall"],
                    m["macro_f1"],
                ]
            )
    print(json.dumps({"report_dir": str(out), "postprocessing": winner}, indent=2), flush=True)


if __name__ == "__main__":
    main()
