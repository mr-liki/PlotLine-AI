"""Bounded pretrained GPU pilot using the tested valid-supervision criterion."""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from roadmarking.dataset.complete_regions import check_complete_dataset
from roadmarking.rasterization.pipeline import dump, sha256
from roadmarking.training.valid_criterion import ValidMask2FormerCriterion


def load_crop(root, record, processor, device):
    path = Path(root) / record["file"]
    if sha256(path) != record["sha256"]:
        raise ValueError("Training crop hash mismatch")
    with np.load(path) as data:
        rgb = data["rgb"].copy()
        targets = torch.from_numpy(data["masks"].astype(np.float32)).to(device)
        labels = torch.from_numpy(data["labels"].astype(np.int64)).to(device)
        validity = torch.from_numpy(data["validity"].copy()).to(device)
    encoded = processor(images=rgb, return_tensors="pt")
    if encoded["pixel_values"].shape[-2:] != (record["size"], record["size"]):
        raise ValueError("Processor resized a native-resolution crop")
    return encoded["pixel_values"].to(device), [targets], [labels], [validity], rgb


def compute(criterion, outputs, targets, labels, validity):
    with torch.autocast("cuda", enabled=False):
        losses, matches = criterion(
            outputs.masks_queries_logits,
            outputs.class_queries_logits,
            targets,
            labels,
            validity,
            [True],
            outputs.auxiliary_logits,
        )
        total = criterion.total(losses)
    return total, losses, matches


def instance_metrics(predictions, scores, prediction_classes, targets, target_classes, valid):
    """Per-crop valid-pixel AP with 101 recall points; not a full-survey COCO claim."""
    predictions = np.asarray(predictions, bool)
    targets = np.asarray(targets, bool)
    metrics = {}
    for label, name in ((0, "longitudinal_solid"), (1, "longitudinal_dash")):
        truth_ids = np.where(np.asarray(target_classes) == label)[0]
        proposed = sorted(
            np.where(np.asarray(prediction_classes) == label)[0], key=lambda i: -scores[i]
        )
        if not len(truth_ids):
            metrics[name] = {"status": "NOT_EVALUATED", "ground_truth_instances": 0}
            continue
        aps, operating = [], None
        for threshold in np.arange(0.5, 0.951, 0.05):
            matched, tp, fp, duplicates = set(), [], [], 0
            for index in proposed:
                candidate = predictions[index] & valid
                ious = []
                for tid in truth_ids:
                    mask = targets[tid] & valid
                    union = (candidate | mask).sum()
                    ious.append((float((candidate & mask).sum() / max(1, union)), int(tid)))
                available = [(iou, tid) for iou, tid in ious if tid not in matched]
                iou, best = max(available, default=(0.0, -1))
                positive = iou >= threshold and best not in matched
                if not positive and any(
                    value >= threshold and tid in matched for value, tid in ious
                ):
                    duplicates += 1
                if positive:
                    matched.add(best)
                tp.append(int(positive))
                fp.append(int(not positive))
            cumulative_tp, cumulative_fp = np.cumsum(tp), np.cumsum(fp)
            recall = cumulative_tp / len(truth_ids)
            precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
            ap = np.mean([max(precision[recall >= r], default=0.0) for r in np.linspace(0, 1, 101)])
            aps.append(float(ap))
            if operating is None:
                true, false, missed = sum(tp), sum(fp), len(truth_ids) - sum(tp)
                p, r = true / max(1, true + false), true / len(truth_ids)
                operating = {
                    "precision_at_iou50": p,
                    "recall_at_iou50": r,
                    "f1_at_iou50": 2 * p * r / max(1e-12, p + r),
                    "true_positives": true,
                    "false_positives": false,
                    "missed_instances": missed,
                    "duplicate_detections": duplicates,
                }
        metrics[name] = {
            "status": "EVALUATED_PER_CROP_ON_VALID_PIXELS",
            "ground_truth_instances": len(truth_ids),
            "AP": float(np.mean(aps)),
            "AP50": aps[0],
            "AP75": aps[5],
            **operating,
        }
    return metrics


def preview_and_evaluate(model, processor, dataset, crop, device, output):
    pixels, targets, labels, validity, rgb = load_crop(dataset, crop, processor, device)
    model.eval()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        outputs = model(pixel_values=pixels, output_auxiliary_logits=False)
    torch.cuda.synchronize()
    latency = time.perf_counter() - start
    predictions, scores, classes = [], [], []
    valid = validity[0].cpu().numpy()
    # The installed stock instance postprocessor hard-codes a 384 intermediate
    # grid. Interpolate logits directly to the original native crop instead.
    native_logits = torch.nn.functional.interpolate(
        outputs.masks_queries_logits.float(),
        size=(crop["size"], crop["size"]),
        mode="bilinear",
        align_corners=False,
    )[0]
    query_scores, query_labels = outputs.class_queries_logits[0].float().softmax(-1).max(-1)
    for index in range(len(native_logits)):
        label = int(query_labels[index])
        if label == model.config.num_labels:
            continue
        mask = (native_logits[index] > 0).cpu().numpy()
        selected = mask & valid
        if selected.sum() < 4:
            continue
        mask_quality = float(native_logits[index].sigmoid().cpu().numpy()[selected].mean())
        score = float(query_scores[index]) * mask_quality
        if score < 0.25:
            continue
        predictions.append(mask)
        scores.append(score)
        classes.append(label)
    shape = (0, crop["size"], crop["size"])
    predictions = np.stack(predictions) if predictions else np.empty(shape, bool)
    truth = targets[0].cpu().numpy().astype(bool)
    target_classes = labels[0].cpu().numpy()
    metrics = instance_metrics(predictions, scores, classes, truth, target_classes, valid)
    target_union = truth.any(0)
    prediction_union = predictions.any(0)
    metrics["foreground_iou_on_valid_pixels"] = float(
        (target_union & prediction_union & valid).sum()
        / max(1, ((target_union | prediction_union) & valid).sum())
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    Image.fromarray(rgb).save(output / "original_intensity.png")

    def overlay(masks):
        image = rgb.copy().astype(float)
        rng = np.random.default_rng(895)
        for mask in masks:
            color = rng.uniform(60, 255, 3)
            image[mask & valid] = image[mask & valid] * 0.45 + color * 0.55
        return Image.fromarray(image.clip(0, 255).astype(np.uint8))

    gt_image, predicted_image = overlay(truth), overlay(predictions)
    gt_image.save(output / "approved_ground_truth.png")
    draw = ImageDraw.Draw(predicted_image)
    for mask, score, label in zip(predictions, scores, classes, strict=True):
        ys, xs = np.where(mask & valid)
        if len(xs):
            draw.text((int(xs.min()), int(ys.min())), f"{label}:{score:.2f}", fill="yellow")
    predicted_image.save(output / "recognition.png")
    comparison = Image.new("RGB", (rgb.shape[1] * 2, rgb.shape[0]))
    comparison.paste(gt_image, (0, 0))
    comparison.paste(predicted_image, (rgb.shape[1], 0))
    comparison.save(output / "comparison.png")
    error = rgb.copy()
    error[valid & prediction_union & ~target_union] = [255, 40, 40]
    error[valid & target_union & ~prediction_union] = [40, 100, 255]
    error[~valid] = [80, 0, 80]
    Image.fromarray(error).save(output / "errors.png")
    np.savez_compressed(
        output / "prediction_masks.npz",
        masks=predictions,
        scores=np.asarray(scores),
        labels=np.asarray(classes),
        validity=valid,
    )
    dump(
        output / "predictions.json",
        {
            "crop": crop,
            "class_ids": classes,
            "scores": scores,
            "model_only_seconds": latency,
            "metrics": metrics,
            "evaluation_scope": (
                "ONE_CROP_FROM_GEOGRAPHICALLY_HELD_OUT_GROUP"
                if crop["split"] == "val"
                else "TRAINING_ENGINEERING_PREVIEW"
            ),
        },
    )
    np.savez_compressed(
        output / "reload_reference.npz",
        class_logits=outputs.class_queries_logits.float().cpu().numpy(),
        mask_logits=outputs.masks_queries_logits.float().cpu().numpy(),
    )
    return {
        "metrics": metrics,
        "model_only_seconds": latency,
        "prediction_instances": len(predictions),
        "output": str(output),
        "crop": crop,
    }


def run_pilot(dataset, initialization, output, size=512, steps=8, freeze_backbone=False):
    torch.set_num_threads(8)
    torch.manual_seed(895)
    np.random.seed(895)
    random.seed(895)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    check_complete_dataset(dataset)
    manifest = json.loads((Path(dataset) / "metadata/dataset_manifest.json").read_text())
    training = sorted(
        [c for c in manifest["crops"] if c["size"] == size and c["split"] == "train"],
        key=lambda c: (-c["foreground_pixels"], c["crop_id"]),
    )
    validation = sorted(
        [c for c in manifest["crops"] if c["size"] == size and c["split"] == "val"],
        key=lambda c: (-c["foreground_pixels"], c["crop_id"]),
    )
    if not training:
        raise ValueError("No training crop for requested native size")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise ValueError("CUDA required for this pilot")
    processor_path = Path(initialization) / "processor"
    if not processor_path.exists():
        processor_path = Path(initialization).parent / "processor"
    processor = Mask2FormerImageProcessor.from_pretrained(processor_path, local_files_only=True)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        initialization, local_files_only=True
    )
    checkpointing = "UNSUPPORTED"
    try:
        model.gradient_checkpointing_enable()
        checkpointing = "ENABLED"
    except (ValueError, AttributeError):
        pass
    if freeze_backbone:
        for parameter in model.model.pixel_level_module.encoder.parameters():
            parameter.requires_grad_(False)
    model.to(device)
    criterion = ValidMask2FormerCriterion(max_points=4096).to(device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=2e-5,
        weight_decay=0.01,
        foreach=False,
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0)
    torch.cuda.reset_peak_memory_stats()
    memory_before = torch.cuda.mem_get_info()
    training_log = []
    start = time.perf_counter()
    for step in range(steps):
        if time.perf_counter() - start > 900:
            raise TimeoutError("Bounded pilot runtime exceeded 900 seconds")
        # Four repetitions provide a tiny-sample learning check; subsequent steps
        # visit other crops in the approved training group. No held-out tuning.
        crop = training[0] if step < 4 else training[(step - 4) % len(training)]
        pixels, targets, labels, validity, _ = load_crop(dataset, crop, processor, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        tick = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = model(pixel_values=pixels, output_auxiliary_logits=True)
        total, losses, _ = compute(criterion, outputs, targets, labels, validity)
        if not torch.isfinite(total):
            raise ValueError("Nonfinite supervised loss")
        print(
            json.dumps(
                {
                    "stage": "before_backward",
                    "step": step + 1,
                    "loss": float(total.detach()),
                    "fp16_scale": scaler.get_scale(),
                }
            ),
            flush=True,
        )
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < old_scale:
            raise ValueError("FP16 scaler skipped optimizer step")
        torch.cuda.synchronize()
        row = {
            "optimizer_step": step + 1,
            "crop_id": crop["crop_id"],
            "loss": float(total.detach()),
            "loss_components": {k: float(v.detach()) for k, v in losses.items()},
            "gradient_norm_before_clip": float(gradient_norm),
            "seconds": time.perf_counter() - tick,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        training_log.append(row)
        dump(output / "training_log.json", training_log)
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in ("optimizer_step", "loss", "seconds", "cuda_peak_allocated_bytes")
                }
            ),
            flush=True,
        )
        del outputs, losses, total
    model.eval()
    checkpoint = output / "pilot_checkpoint"
    model.save_pretrained(checkpoint, safe_serialization=True)
    processor.save_pretrained(output / "processor")
    evaluation_crop = validation[0] if validation else training[0]
    evaluation = preview_and_evaluate(
        model, processor, dataset, evaluation_crop, device, output / "recognition"
    )
    # Also show the training example, separately from independent held-out metrics.
    sanity = preview_and_evaluate(
        model, processor, dataset, training[0], device, output / "training_preview"
    )
    config = {
        "dataset": str(dataset),
        "dataset_version": manifest["dataset_version"],
        "initialization": str(initialization),
        "native_crop_size": size,
        "optimizer": "AdamW",
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "micro_batch": 1,
        "gradient_accumulation": 1,
        "optimizer_steps": len(training_log),
        "seed": 895,
        "mixed_precision": "fp16",
        "gradient_clip_norm": 1,
        "gradient_checkpointing": checkpointing,
        "backbone_frozen": freeze_backbone,
        "criterion": {
            **manifest["criterion"],
            "weights": criterion.weights,
            "version": "phase5_3_v2_valid_null_queries",
            "implementation_sha256": sha256(Path(__file__).with_name("valid_criterion.py")),
        },
        "independent_groups": manifest["splits"],
        "no_object_limitation": (
            "Unmatched-query class loss suppressed in every partial-validity image"
        ),
    }
    dump(output / "training_config.json", config)
    dump(output / "dataset_manifest.json", manifest)
    dump(output / "class_mapping.json", manifest["training_classes"])
    record = {
        "status": "TRAINED_PROVISIONAL",
        "checkpoint": str(checkpoint),
        "processor": str(output / "processor"),
        "weight_sha256": sha256(checkpoint / "model.safetensors"),
        "checkpoint_bytes": (checkpoint / "model.safetensors").stat().st_size,
        "optimizer_steps": len(training_log),
        "elapsed_seconds": time.perf_counter() - start,
        "cuda_device": torch.cuda.get_device_name(),
        "cuda_free_total_before": memory_before,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "training_config": config,
        "held_out_evaluation": evaluation if validation else None,
        "training_preview": sanity,
        "independent_test_evaluation": None,
        "production_ready": False,
        "best_checkpoint_selection": "Final bounded-pilot checkpoint; no best-model search",
    }
    dump(output / "result.json", record)
    return record
