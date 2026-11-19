"""Native instance-mask filtering and development-only calibration; no geometry merging."""

import json
from pathlib import Path

import numpy as np


def iou_matrix(a, b, valid):
    """IoU of every mask pair inside ``valid``. Pixel counts are computed as one matrix
    product; counts stay exact integers (float32 is exact below 2**24 pixels per mask, float64
    is used beyond), so the values equal the pixel-by-pixel definition."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    result = np.zeros((len(a), len(b)), np.float64)
    if not len(a) or not len(b):
        return result
    valid = np.asarray(valid, bool)
    av = (a & valid).reshape(len(a), -1)
    bv = (b & valid).reshape(len(b), -1)
    dtype = np.float32 if av.shape[1] < 2**24 else np.float64
    fa, fb = av.astype(dtype), bv.astype(dtype)
    inter = np.rint(fa @ fb.T).astype(np.int64)
    counts_a = av.sum(1).astype(np.int64)
    counts_b = bv.sum(1).astype(np.int64)
    union = np.maximum(1, counts_a[:, None] + counts_b[None, :] - inter)
    return inter / union


def filter_instances(scores, labels, overlaps, qualities, config):
    """Class-wise mask NMS suppresses whole duplicates, never joins separate dashes."""
    selected = []
    for index in sorted(range(len(scores)), key=lambda i: (-scores[i], i)):
        threshold = config["confidence"][str(int(labels[index]))]
        if scores[index] < threshold or qualities[index] < config["mask_quality"]:
            continue
        if config["nms_iou"] is not None and any(
            labels[other] == labels[index] and overlaps[index, other] > config["nms_iou"]
            for other in selected
        ):
            continue
        selected.append(index)
    return selected


def operating_counts(record, selected, threshold=0.5):
    result = {}
    for label in (0, 1):
        targets = np.where(record["target_labels"] == label)[0]
        proposed = sorted(
            [i for i in selected if record["labels"][i] == label],
            key=lambda i: -record["scores"][i],
        )
        matched, tp, fp, duplicates = set(), 0, 0, 0
        assignments, false_indices = {}, []
        for index in proposed:
            possible = [
                (record["iou"][index, tid], int(tid)) for tid in targets if tid not in matched
            ]
            value, best = max(possible, default=(0.0, -1))
            if value >= threshold:
                tp += 1
                matched.add(best)
                assignments[index] = best
            else:
                fp += 1
                false_indices.append(index)
                if any(
                    record["iou"][index, tid] >= threshold and tid in matched for tid in targets
                ):
                    duplicates += 1
        precision, recall = tp / max(1, tp + fp), tp / max(1, len(targets))
        result[str(label)] = {
            "tp": tp,
            "fp": fp,
            "fn": len(targets) - tp,
            "duplicates": duplicates,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(1e-12, precision + recall),
            "assignments": assignments,
            "false_indices": false_indices,
            "missed_indices": [int(i) for i in targets if i not in matched],
        }
    return result


def aggregate_counts(records, selections):
    totals = {str(i): {k: 0 for k in ("tp", "fp", "fn", "duplicates")} for i in (0, 1)}
    for record, selected in zip(records, selections, strict=True):
        current = operating_counts(record, selected)
        for label in totals:
            for key in totals[label]:
                totals[label][key] += current[label][key]
    for value in totals.values():
        p = value["tp"] / max(1, value["tp"] + value["fp"])
        r = value["tp"] / max(1, value["tp"] + value["fn"])
        value.update(precision=p, recall=r, f1=2 * p * r / max(1e-12, p + r))
    totals["macro_f1"] = float(np.mean([totals[str(i)]["f1"] for i in (0, 1)]))
    return totals


def load_saved_validation(run):
    """Use exact saved masks, preserving the Phase 5.3 before/after protocol."""
    run = Path(run)
    evaluated = json.loads((run / "validation_evaluation.json").read_text())
    dataset = Path(json.loads((run / "result.json").read_text())["training_config"]["dataset"])
    records = []
    for item in evaluated["per_crop_results"]:
        crop = item["crop"]
        folder = Path(item["output"])
        with np.load(folder / "prediction_masks.npz") as data:
            masks, scores, labels, valid = (
                data[k].copy() for k in ("masks", "scores", "labels", "validity")
            )
        with np.load(dataset / crop["file"]) as data:
            targets, target_labels, rgb = (data[k].copy() for k in ("masks", "labels", "rgb"))
        # Recover mask-quality evidence from the saved raw logits. Scores remain
        # the exact originally recorded GPU values used for the baseline metrics.
        import torch

        reference = np.load(folder / "reload_reference.npz")
        native = torch.nn.functional.interpolate(
            torch.from_numpy(reference["mask_logits"]),
            size=valid.shape,
            mode="bilinear",
            align_corners=False,
        )[0]
        probabilities = native.sigmoid().numpy()
        proposals = native.numpy() > 0
        class_labels = torch.from_numpy(reference["class_logits"])[0].argmax(-1).numpy()
        qualities, query_indices = [], []
        for mask, label in zip(masks, labels, strict=True):
            candidates = np.where(class_labels == label)[0]
            best = max(
                candidates,
                key=lambda j: (mask & proposals[j] & valid).sum()
                / max(1, ((mask | proposals[j]) & valid).sum()),
            )
            query_indices.append(int(best))
            qualities.append(float(probabilities[best][mask & valid].mean()))
        records.append(
            {
                "crop": crop,
                "masks": masks.astype(bool),
                "scores": scores,
                "labels": labels,
                "valid": valid,
                "targets": targets.astype(bool),
                "target_labels": target_labels,
                "rgb": rgb,
                "qualities": np.asarray(qualities),
                "query_indices": query_indices,
                "iou": iou_matrix(masks, targets, valid),
                "prediction_overlap": iou_matrix(masks, masks, valid),
            }
        )
    return records


def native_world_iou(first_mask, first_crop, second_mask, second_crop):
    """Compare overlapping crop masks on the original 0.05 native grid."""

    def origin(crop):
        affine = crop["affine"]
        return round(-affine[5] / affine[0]), round(affine[2] / affine[0])

    r1, c1 = origin(first_crop)
    r2, c2 = origin(second_crop)
    y0, x0 = max(r1, r2), max(c1, c2)
    y1, x1 = min(r1 + first_mask.shape[0], r2 + second_mask.shape[0]), min(
        c1 + first_mask.shape[1], c2 + second_mask.shape[1]
    )
    if y1 <= y0 or x1 <= x0:
        return 0.0
    intersection = (
        first_mask[y0 - r1 : y1 - r1, x0 - c1 : x1 - c1]
        & second_mask[y0 - r2 : y1 - r2, x0 - c2 : x1 - c2]
    ).sum()
    return float(intersection / max(1, first_mask.sum() + second_mask.sum() - intersection))
