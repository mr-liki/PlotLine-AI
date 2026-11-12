"""Bounded, resumable Phase 5.4 training on approved positive and hard-negative crops."""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from roadmarking.dataset.complete_regions import check_complete_dataset
from roadmarking.inference.calibration import aggregate_counts, filter_instances, iou_matrix
from roadmarking.rasterization.pipeline import dump, sha256
from roadmarking.training.checkpoint_manager import load_training_state, save_training_state
from roadmarking.training.pilot import compute, instance_metrics, load_crop
from roadmarking.training.valid_criterion import ValidMask2FormerCriterion


def evaluate(model, processor, dataset, crops, config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model.eval()
    records, selections, elapsed = [], [], 0.0
    for crop in crops:
        pixels, targets, labels, validity, _ = load_crop(dataset, crop, processor, "cuda")
        torch.cuda.synchronize()
        tick = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            predictions = model(pixel_values=pixels, output_auxiliary_logits=False)
        torch.cuda.synchronize()
        elapsed += time.perf_counter() - tick
        logits = torch.nn.functional.interpolate(
            predictions.masks_queries_logits.float(),
            size=validity[0].shape,
            mode="bilinear",
            align_corners=False,
        )[0]
        scores, classes = predictions.class_queries_logits[0].float().softmax(-1).max(-1)
        native = logits.cpu().numpy()
        probabilities = logits.sigmoid().cpu().numpy()
        valid = validity[0].cpu().numpy()
        masks, confidences, kinds, qualities = [], [], [], []
        for index in range(len(logits)):
            label = int(classes[index])
            if label == 2:
                continue
            mask = native[index] > 0
            selected = mask & valid
            if selected.sum() < 4:
                continue
            quality = float(probabilities[index][selected].mean())
            confidence = float(scores[index]) * quality
            if confidence < 0.25:
                continue
            masks.append(mask)
            confidences.append(confidence)
            kinds.append(label)
            qualities.append(quality)
        masks = np.stack(masks) if masks else np.empty((0, *valid.shape), bool)
        confidence = np.asarray(confidences)
        kinds, qualities = np.asarray(kinds), np.asarray(qualities)
        truth, target_labels = targets[0].cpu().numpy().astype(bool), labels[0].cpu().numpy()
        record = {
            "crop": crop,
            "masks": masks,
            "scores": confidence,
            "labels": kinds,
            "qualities": qualities,
            "valid": valid,
            "targets": truth,
            "target_labels": target_labels,
            "iou": iou_matrix(masks, truth, valid),
            "prediction_overlap": iou_matrix(masks, masks, valid),
        }
        selected = filter_instances(
            confidence, kinds, record["prediction_overlap"], qualities, config
        )
        selections.append(selected)
        records.append(record)
        np.savez_compressed(
            output / (crop["crop_id"] + ".npz"),
            masks=masks,
            scores=confidence,
            labels=kinds,
            qualities=qualities,
            selected=np.asarray(selected),
            validity=valid,
            class_logits=predictions.class_queries_logits.float().cpu().numpy(),
            mask_logits=predictions.masks_queries_logits.float().cpu().numpy(),
        )
    raw = aggregate_counts(records, [list(range(len(r["scores"]))) for r in records])
    filtered = aggregate_counts(records, selections)
    raw_iou, filtered_iou, ap = [], [], []
    for r, selected in zip(records, selections, strict=True):
        target = r["targets"].any(0)
        for selection, values in (
            (list(range(len(r["scores"]))), raw_iou),
            (selected, filtered_iou),
        ):
            predicted = r["masks"][selection].any(0)
            values.append(
                float(
                    (predicted & target & r["valid"]).sum()
                    / max(1, ((predicted | target) & r["valid"]).sum())
                )
            )
        ap.append(
            instance_metrics(
                r["masks"][selected],
                r["scores"][selected],
                r["labels"][selected],
                r["targets"],
                r["target_labels"],
                r["valid"],
            )
        )
    summary = {
        "raw": raw,
        "postprocessed": filtered,
        "raw_mean_foreground_iou": float(np.mean(raw_iou)),
        "postprocessed_mean_foreground_iou": float(np.mean(filtered_iou)),
        "postprocessed_per_crop_ap": ap,
        "postprocessing": config,
        "crops": crops,
        "model_only_seconds_total": elapsed,
        "development_validation_only": True,
        "crop_counts_are_appearances": True,
    }
    dump(output / "metrics.json", summary)
    return summary


def load_augmented(dataset, record, processor, rng, augment_fn, device="cuda"):
    """Load a native crop, apply a joint augmentation, then the unchanged processor."""
    path = Path(dataset) / record["file"]
    if sha256(path) != record["sha256"]:
        raise ValueError("Training crop hash mismatch")
    with np.load(path) as data:
        rgb, masks, labels, valid = augment_fn(
            data["rgb"].copy(),
            data["masks"].astype(bool),
            data["labels"].astype(np.int64),
            data["validity"].astype(bool),
            rng,
        )
    encoded = processor(images=rgb, return_tensors="pt")
    if encoded["pixel_values"].shape[-2:] != (record["size"], record["size"]):
        raise ValueError("Processor resized a native-resolution crop")
    return (
        encoded["pixel_values"].to(device),
        [torch.from_numpy(masks.astype(np.float32)).to(device)],
        [torch.from_numpy(labels.astype(np.int64)).to(device)],
        [torch.from_numpy(valid).to(device)],
    )


def run_training(
    dataset,
    initialization,
    processor_path,
    output,
    postprocessing,
    stop_after=100,
    max_steps=200,
    resume=None,
    learning_rate=1e-5,
    sampling_cycle=None,
    dataset_check=None,
    evaluation_interval=50,
    resume_loss_scale=None,
    initial_loss_scale=128.0,
    expected_baseline_counts=None,
    extra_pools=None,
    seed=895,
    criterion_kwargs=None,
    augment_fn=None,
):
    torch.set_num_threads(8)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    augment_rng = np.random.default_rng(seed + 10_000)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    readiness = (dataset_check or check_complete_dataset)(dataset)
    manifest = json.loads((dataset / "metadata/dataset_manifest.json").read_text())
    dataset_hash = sha256(dataset / "metadata/dataset_manifest.json")
    train = [c for c in manifest["crops"] if c["split"] == "train" and c["size"] == 512]
    validation = [c for c in manifest["crops"] if c["split"] == "val" and c["size"] == 512]
    pools = {
        "solid": [c for c in train if 0 in c["labels"]],
        "dash": [c for c in train if 1 in c["labels"]],
        "verified_negative": [c for c in train if not c["labels"]],
    }
    pools["new_region"] = [c for c in train if c["tile_id"] == "NEW_TRAIN_REVIEW"]
    # Optional named predicates add sampling strata (e.g. newly approved geography).
    for name, predicate in (extra_pools or {}).items():
        pools[name] = [c for c in train if predicate(c)]
    if any(not pools[name] for name in (sampling_cycle or list(pools))):
        raise ValueError("Expected approved class-balanced and verified-negative pools")
    if resume:
        saved = Path(resume)
        bundle = json.loads((saved.parent / "bundle.json").read_text())
        if sha256(saved.parent / "model/model.safetensors") != bundle["weight_sha256"]:
            raise ValueError("Resume model weights changed")
        initialization = saved.parent / "model"
        processor_path = saved.parent / "processor"
    elif output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=bool(resume))
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        initialization, local_files_only=True
    ).to("cuda")
    processor = Mask2FormerImageProcessor.from_pretrained(processor_path, local_files_only=True)
    criterion = ValidMask2FormerCriterion(**(criterion_kwargs or {})).to("cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01, foreach=False
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps, eta_min=learning_rate * 0.2
    )
    if initial_loss_scale <= 0:
        raise ValueError("Positive numerical loss scale required")
    scaler = torch.amp.GradScaler("cuda", init_scale=initial_loss_scale)
    config = {
        "dataset": str(dataset),
        "dataset_sha256": dataset_hash,
        "initialization": str(initialization),
        "max_optimizer_steps": max_steps,
        "learning_rate": learning_rate,
        "minimum_lr": learning_rate * 0.2,
        "scheduler": "CosineAnnealingLR",
        "weight_decay": 0.01,
        "seed": seed,
        "criterion_kwargs": criterion_kwargs or {},
        "augmentation": getattr(augment_fn, "__name__", None) if augment_fn else None,
        "fp16_initial_scale": initial_loss_scale,
        "micro_batch": 1,
        "gradient_accumulation": 1,
        "gradient_clip_norm": 1,
        "native_crop": 512,
        "do_resize": False,
        "sampling_cycle": sampling_cycle or list(pools),
        "pool_sizes": {k: len(v) for k, v in pools.items()},
        "evaluation_interval": evaluation_interval,
        "loss_weights": criterion.weights,
        "criterion_sha256": sha256(Path(__file__).with_name("valid_criterion.py")),
        "postprocessing_frozen_before_training": postprocessing,
        "selection_rule": "Highest development macro F1 with frozen postprocessing, then fewer FP",
    }
    start_step, logs, history = 0, [], []
    if resume:
        state = load_training_state(resume, optimizer, scheduler, scaler, dataset_hash)
        start_step = state["step"]
        if state["config"]["max_optimizer_steps"] != max_steps:
            raise ValueError("Resume scheduler budget changed")
        config = state["config"]
        logs = [
            x
            for x in json.loads((output / "training_log.json").read_text())
            if x["step"] <= start_step
        ]
        history = [
            x
            for x in json.loads((output / "validation_history.json").read_text())
            if x["step"] <= start_step
        ]
        if resume_loss_scale is not None:
            if not 0 < resume_loss_scale <= scaler.get_scale():
                raise ValueError("Recovery scale must be positive and no larger than saved scale")
            previous_scale = scaler.get_scale()
            scale_state = scaler.state_dict()
            scale_state["scale"] = float(resume_loss_scale)
            scaler.load_state_dict(scale_state)
            config["numerical_recovery"] = {
                "checkpoint_step": start_step,
                "previous_scale": previous_scale,
                "recovery_scale": resume_loss_scale,
                "finite_gradient_guard_preserved": True,
            }
            dump(output / "training_config.json", config)
        dump(
            output / f"resume_proof_step{start_step}.json",
            {
                "restored_step": start_step,
                "optimizer_state_parameter_count": len(optimizer.state),
                "scheduler_last_epoch": scheduler.last_epoch,
                "scaler_scale": scaler.get_scale(),
                "rngs_restored": True,
                "dataset_hash_verified": True,
                "current_lr": optimizer.param_groups[0]["lr"],
            },
        )
    else:
        dump(output / "training_config.json", config)
        initial = evaluate(
            model, processor, dataset, validation, postprocessing, output / "validation_step0000"
        )
        expected = expected_baseline_counts or {"fp": 115, "fn": 12}
        if any(
            sum(initial["raw"][str(i)][key] for i in (0, 1)) != expected[key]
            for key in ("fp", "fn")
        ):
            raise ValueError("Before/after evaluation does not reproduce the declared warm start")
        history.append({"step": 0, "metrics": initial})
    best = max(
        history,
        key=lambda h: (
            h["metrics"]["postprocessed"]["macro_f1"],
            -sum(h["metrics"]["postprocessed"][str(i)]["fp"] for i in (0, 1)),
        ),
    )

    def checkpoint(step, metrics):
        folder = output / f"checkpoint_step{step:04d}"
        if folder.exists():
            raise FileExistsError(folder)
        model.save_pretrained(folder / "model", safe_serialization=True)
        processor.save_pretrained(folder / "processor")
        save_training_state(
            folder / "training_state.pt", optimizer, scheduler, scaler, step, config, dataset_hash
        )
        dump(
            folder / "bundle.json",
            {
                "step": step,
                "weight_sha256": sha256(folder / "model/model.safetensors"),
                "dataset_sha256": dataset_hash,
                "metrics": metrics,
            },
        )
        return folder

    if not resume:
        checkpoint(0, history[0]["metrics"])
        dump(output / "validation_history.json", history)
    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    pool_names = config["sampling_cycle"]
    for step in range(start_step, min(stop_after, max_steps)):
        if time.perf_counter() - began > 1200:
            raise TimeoutError("Bounded training segment exceeded 1200 seconds")
        stratum = pool_names[step % len(pool_names)]
        crop = random.choice(pools[stratum])
        if augment_fn is None:
            pixels, targets, labels, validity, _ = load_crop(dataset, crop, processor, "cuda")
        else:
            pixels, targets, labels, validity = load_augmented(
                dataset, crop, processor, augment_rng, augment_fn
            )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        tick = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            predictions = model(pixel_values=pixels, output_auxiliary_logits=True)
        total, losses, _ = compute(criterion, predictions, targets, labels, validity)
        if not torch.isfinite(total):
            raise ValueError("Nonfinite loss")
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < previous_scale:
            raise ValueError("Gradient scaler skipped optimizer update")
        scheduler.step()
        torch.cuda.synchronize()
        logs.append(
            {
                "step": step + 1,
                "crop_id": crop["crop_id"],
                "sampling_stratum": stratum,
                "loss": float(total.detach()),
                "gradient_norm": float(gradient),
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter() - tick,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
        )
        dump(output / "training_log.json", logs)
        print(json.dumps(logs[-1]), flush=True)
        del predictions, total, losses
        if (step + 1) % config.get("evaluation_interval", 50) == 0 or step + 1 == stop_after:
            metrics = evaluate(
                model,
                processor,
                dataset,
                validation,
                postprocessing,
                output / f"validation_step{step+1:04d}",
            )
            checkpoint(step + 1, metrics)
            entry = {"step": step + 1, "metrics": metrics}
            history.append(entry)
            dump(output / "validation_history.json", history)

            def key(h):
                return (
                    h["metrics"]["postprocessed"]["macro_f1"],
                    -sum(h["metrics"]["postprocessed"][str(i)]["fp"] for i in (0, 1)),
                )

            if key(entry) > key(best):
                best = entry
            dump(
                output / "best.json",
                {
                    "step": best["step"],
                    "checkpoint": str(output / f"checkpoint_step{best['step']:04d}"),
                    "metrics": best["metrics"],
                },
            )
    record = {
        "status": "TRAINED_PROVISIONAL",
        "current_step": len(logs),
        "best_step": best["step"],
        "best_checkpoint": str(output / f"checkpoint_step{best['step']:04d}"),
        "segment_elapsed_seconds": time.perf_counter() - began,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "resumed_from": str(resume) if resume else None,
        "readiness": readiness,
    }
    dump(output / f"segment_to{len(logs)}.json", record)
    return record
