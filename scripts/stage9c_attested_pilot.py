"""Bounded, classifier-only Stage 9C pilot on verified data and provisional development."""

import gc
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyogrio
import rasterio
import torch
from affine import Affine
from PIL import Image
from rasterio.features import rasterize
from shapely.geometry import box
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from roadmarking.training.pilot import load_crop
from roadmarking.training.road_text_head import CLASS_NAMES, make_three_class_head
from roadmarking.training.synthetic_text_v1 import load_synthetic_text_sample
from roadmarking.training.valid_criterion import ValidMask2FormerCriterion
from stage9c_audit import BASE, ROOT, dump, sha256


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def crop_box(record):
    if "native_xy_bbox" in record:
        return box(*record["native_xy_bbox"])
    affine = Affine(*record["affine"])
    size = record["size"]
    corners = [affine * (x, y) for x, y in ((0, 0), (0, size), (size, 0), (size, size))]
    xs, ys = zip(*corners, strict=True)
    return box(min(xs), min(ys), max(xs), max(ys))


def source_context():
    reports = ROOT / "outputs/reports"
    prior = read(reports / "stage9c_1/latest.json")
    lineage = read(Path(prior["run"]) / "provenance_audit.json")
    audit_dir = Path(read(BASE / "latest.json")["run"])
    audit = read(audit_dir / "synthetic_text_dataset_audit.json")
    source = Path(audit["source"])
    # The old synthetic text set was retired (deleted by the user, 2026-11-12). Its hashes are
    # still enforced whenever the folder exists; real-data protections below are unaffected.
    if source.exists():
        assert all(sha256(source / p) == h for p, h in audit["source_file_sha256"].items())
    assert not lineage["approved_n9_01_exact_archive_matches"]
    assert not any(
        r["exact_archive_match"] for r in lineage["known_n9_01_clipboard_reference_checks"]
    )
    state = read(reports / "stage6_review/latest.json")
    dataset = Path(state["dataset"])
    manifest_path = dataset / "metadata/dataset_manifest.json"
    manifest = read(manifest_path)
    stage9b = reports / "stage9b"
    final = stage9b / "stroke_gap_review"
    approval = read(final / "n9_01_text_finalization_manifest.json")
    assert all(sha256(Path(r["path"])) == r["sha256"] for r in approval["files"])
    spec = read(Path(read(stage9b / "latest.json")["report_dir"]) / "review_spec.json")
    tile = next(t for t in spec["tiles"] if t["candidate_id"] == "N9_01")
    protected = []
    for stage in ("stage7_new_test", "stage8_new_solid_review"):
        folder = reports / stage
        pointer = folder / "latest.json"
        directory = (
            Path(read(pointer)["report_dir"])
            if pointer.exists()
            else next(p for p in folder.iterdir() if (p / "new_region_candidates.json").is_file())
        )
        protected.extend(read(directory / "new_region_candidates.json")["tiles"])
    bounds = [box(*t["bounds"]) for t in protected]
    ids = set(manifest["protected_test_ids"])
    ids.update(uid for t in protected for uid in t.get("source_global_ids", []))
    train = [c for c in manifest["crops"] if c["split"] == "train" and c["size"] == 512]
    val = [c for c in manifest["crops"] if c["split"] == "val" and c["size"] == 512]
    assert not any(box(*tile["bounds"]).intersects(crop_box(c)) for c in train)
    assert not any(box(*tile["bounds"]).intersects(b) for b in bounds)
    train = [
        c
        for c in train
        if not ids.intersection(c["global_instance_ids"])
        and not any(crop_box(c).intersects(b) for b in bounds)
    ]
    val = [
        c
        for c in val
        if not ids.intersection(c["global_instance_ids"])
        and not any(crop_box(c).intersects(b) for b in bounds)
    ]
    assert not (
        {uid for c in train for uid in c["global_instance_ids"]}
        & {uid for c in val for uid in c["global_instance_ids"]}
    )
    fresh_path = Path(state["fresh_dev"]) / "manifest.json"
    fresh_manifest = read(fresh_path)
    assert fresh_manifest["role"].startswith("FRESH_DEVELOPMENT")
    fresh = [{**c, "split": "val"} for c in fresh_manifest["crops"]]
    fresh = [
        c
        for c in fresh
        if c["split"] == "val"
        and c["size"] == 512
        and not ids.intersection(c["global_instance_ids"])
        and not any(crop_box(c).intersects(b) for b in bounds)
    ]
    assert not (
        {uid for c in train for uid in c["global_instance_ids"]}
        & {uid for c in fresh for uid in c["global_instance_ids"]}
    )
    return (
        reports,
        audit,
        source,
        state,
        dataset,
        manifest_path,
        train,
        val,
        fresh,
        tile,
        spec,
        approval,
    )


def text_arrays(reports, tile, spec):
    stage = reports / "stage9b"
    characters = pyogrio.read_dataframe(
        spec["text_candidates"], layer="character_candidates", where="candidate_id='N9_01'"
    )
    groups = pyogrio.read_dataframe(
        spec["text_candidates"], layer="group_and_unknown_fragments", where="candidate_id='N9_01'"
    )
    supports = [*characters.geometry, *groups.geometry]
    for folder in ("coverage_followup", "text_coverage_refinement"):
        supports.extend(
            pyogrio.read_dataframe(
                stage / folder / "coverage_followup.gpkg",
                layer="additional_bright_support",
                where="region='N9_01'",
            ).geometry
        )
    for folder in (
        "final_text_review",
        "lowest_group_review",
        "bottom_stroke_review",
        "stroke_gap_review",
    ):
        supports.extend(
            pyogrio.read_dataframe(
                stage / folder / "coverage_followup.gpkg", layer="stroke_candidates"
            ).geometry
        )
    lines = pyogrio.read_dataframe(
        spec["base_stage9_candidates"],
        layer="candidate_paint_footprints",
        where="candidate_id='N9_01'",
    )
    with rasterio.open(tile["layers"]["observed"]) as ds:
        observed = ds.read(1).astype(bool)[:512, :512]
        transform = ds.transform

    def raster(shapes):
        return rasterize(
            [(g, 1) for g in shapes], out_shape=observed.shape, transform=transform
        ).astype(bool)

    text = raster(supports) & observed
    non_text = raster(lines.geometry) & observed & ~text
    rgb = np.array(Image.open(tile["images"]["grayscale"]).convert("RGB"))[:512, :512]
    return rgb, text, non_text, observed


def predict(model, pixels, observed):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        outputs = model(pixel_values=pixels, output_auxiliary_logits=False)
    native = torch.nn.functional.interpolate(
        outputs.masks_queries_logits.float(),
        size=observed.shape,
        mode="bilinear",
        align_corners=False,
    )[0]
    classes = outputs.class_queries_logits[0].float().softmax(-1)
    scores, labels = classes.max(-1)
    masks = (native > 0) & torch.as_tensor(observed, device=native.device)
    pixels_per_mask = masks.flatten(1).sum(1)
    quality = (native.sigmoid() * masks).flatten(1).sum(1) / pixels_per_mask.clamp(min=1)
    scores = scores * quality
    masks = masks.cpu().numpy()
    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()
    keep = (scores >= 0.25) & (pixels_per_mask.cpu().numpy() >= 4)
    result = {}
    for i in range(model.config.num_labels):
        result[i] = masks[keep & (labels == i)].any(axis=0)
    return result


def counts(pred, target, valid):
    return np.array(
        [
            (pred & target & valid).sum(),
            (pred & ~target & valid).sum(),
            (~pred & target & valid).sum(),
        ],
        dtype=np.int64,
    )


def metrics(values):
    tp, fp, fn = map(int, values)
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        iou=tp / (tp + fp + fn) if tp + fp + fn else None,
    )


def evaluate(model, processor, development, text):
    results = {i: np.zeros(3, dtype=np.int64) for i in (0, 1)}
    for root, crop in development:
        pixel, masks, labels, valid, _ = load_crop(root, crop, processor, "cuda")
        keep = valid[0].cpu().numpy().astype(bool)
        preds = predict(model, pixel, keep)
        for label in results:
            truth = masks[0][labels[0] == label].bool().any(0).cpu().numpy()
            results[label] += counts(preds[label], truth, keep)
    image, positive, non_text, observed = text
    pixel = processor(images=image, return_tensors="pt").pixel_values.cuda()
    predictions = predict(model, pixel, observed)
    text_pred = predictions.get(2, np.zeros_like(positive))
    return dict(
        solid_dash={CLASS_NAMES[i]: metrics(values) for i, values in results.items()},
        text=metrics(counts(text_pred, positive, positive | non_text)),
        text_scored_positive_pixels=int(positive.sum()),
        text_scored_lane_negative_pixels=int(non_text.sum()),
        text_predicted_unscored_observed_pixels=int(
            (text_pred & observed & ~(positive | non_text)).sum()
        ),
    )


def main():
    (
        reports,
        audit,
        source,
        state,
        dataset,
        manifest_path,
        train,
        val,
        fresh,
        tile,
        spec,
        approval,
    ) = source_context()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = BASE / "attested_pilot" / run_id
    out.mkdir(parents=True)
    prior = read(reports / "stage9c_1/latest.json")
    attestation = {
        "status": "USER_ATTESTED_DIFFERENT_ROAD_IMAGE",
        "statement": "The screenshot used to generate road_text_lidar_synthetic_500 came from a different road image, not N9_01.",  # noqa: E501
        "previous_audit": str(Path(prior["run"]) / "provenance_audit.json"),
        "original_reference_hash_verified": False,
        "exact_generator_lineage_verified": False,
        "known_material_conflicts_found": False,
        "verified_synthetic_files": len(audit["source_file_sha256"]),
        "N9_01_approval_hashes_verified": len(approval["files"]),
        "N9_01_role": "provisional real development only; no independent claim",
    }
    dump(out / "provenance_update.json", attestation)
    dump(BASE / "latest_provenance.json", {"run": str(out), **attestation})
    checks = read(reports / "stage8b/ai_preservation.json")["checks"]
    assert all(sha256(Path(r["path"])) == r["expected_sha256"] for r in checks)
    model_path = Path(checks[0]["path"]).parent
    processor_path = model_path.parent / "processor"
    rng = np.random.default_rng(895)
    pools = {label: [c for c in train if label in c["labels"]] for label in (0, 1)}
    synth = [r for r in audit["sample_audit"] if r["split"] == "train" and r["text"]]
    schedule = []
    for step in range(40):
        if step % 4 == 3:
            item = synth[int(rng.integers(len(synth)))]
            schedule.append(("synthetic", item))
        else:
            label = step % 2
            pool = pools[label]
            schedule.append(("real", pool[int(rng.integers(len(pool)))]))
    development = [(dataset, c) for c in sorted(val, key=lambda c: c["crop_id"])[:12]]
    development.extend((Path(state["fresh_dev"]), c) for c in fresh)
    assert development and all(c["split"] == "val" for _, c in development)
    text = text_arrays(reports, tile, spec)
    dump(
        out / "training_protocol.json",
        dict(
            steps=40,
            checkpoints=[20, 40],
            real_synthetic_counts=dict(Counter(k for k, _ in schedule)),
            real_dataset_manifest_sha256=sha256(manifest_path),
            init_sha256=checks[0]["expected_sha256"],
            synthetic_source_hashes=audit["source_file_sha256"],
            class_mapping=dict(enumerate(CLASS_NAMES)),
            seed=895,
            native_crop=512,
            microbatch=1,
            trainable="classification head only",
            optimizer="AdamW",
            learning_rate=0.0002,
            weight_decay=0.01,
            loss="valid-only matching and classification; frozen mask branch",
            schedules=schedule,
            development=[c for _, c in development],
            operating_point="fixed query score times valid mask quality >=0.25 and logit >0",
            scoring="crop pixel metrics at fixed raw operating point; no stitching or promotion",
            selection="none, report both predeclared checkpoints",
            regression_gate="report solid/dash F1 drops over 0.03 absolute",
            n9_01_scope="Reviewed text and observed lane paint; remaining pixels ignored.",
            protected_test_inference=False,
        ),
    )
    torch.manual_seed(895)
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    processor = Mask2FormerImageProcessor.from_pretrained(processor_path, local_files_only=True)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path, local_files_only=True)
    model.requires_grad_(False).cuda().eval()
    print("Evaluating frozen baseline", flush=True)
    baseline = evaluate(model, processor, development, text)
    dump(out / "baseline.json", baseline)
    old = model.class_predictor
    model.class_predictor = make_three_class_head(old, seed=895, source_labels=CLASS_NAMES[:2])
    model.config.num_labels = 3
    model.config.id2label = dict(enumerate(CLASS_NAMES))
    model.config.label2id = {v: k for k, v in model.config.id2label.items()}
    model.criterion = type(model.criterion)(model.config, model.weight_dict).cuda()
    criterion = ValidMask2FormerCriterion(num_classes=3).cuda()
    optimizer = torch.optim.AdamW(model.class_predictor.parameters(), lr=0.0002, weight_decay=0.01)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    logs, evaluations = [], {}
    model_root = ROOT / "models/mask2former_swin_s" / f"stage9c_attested_{run_id}"
    for step, (kind, record) in enumerate(schedule, 1):
        if kind == "real":
            pixels, masks, labels, validity, _ = load_crop(dataset, record, processor, "cuda")
        else:
            sample = load_synthetic_text_sample(source / "train", Path(record["name"]).stem)
            image = sample["intensity"].numpy()
            rgb = np.repeat((image * 255).round().astype("uint8")[..., None], 3, axis=-1)
            pixels = processor(images=rgb, return_tensors="pt").pixel_values.cuda()
            masks, labels, validity = [
                [sample[key].cuda()] for key in ("mask_labels", "class_labels", "valid_mask")
            ]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model(pixel_values=pixels, output_auxiliary_logits=False)
        losses, _ = criterion.one_image(
            pred.masks_queries_logits[0].float(),
            pred.class_queries_logits[0].float(),
            masks[0],
            labels[0],
            validity[0],
            True,
        )
        loss = losses["loss_class"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.class_predictor.parameters(), 1, error_if_nonfinite=True
        )
        optimizer.step()
        logs.append(
            dict(
                step=step,
                role=kind,
                class_loss=float(loss.detach()),
                labels=labels[0].tolist(),
                foreground_pixels=int(masks[0].sum()),
            )
        )
        if step in (20, 40):
            print(f"Evaluating and saving step {step}", flush=True)
            checkpoint = model_root / f"step{step:04}"
            model.save_pretrained(checkpoint / "model")
            processor.save_pretrained(checkpoint / "processor")
            torch.save(
                dict(
                    optimizer=optimizer.state_dict(),
                    step=step,
                    cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(),
                ),
                checkpoint / "optimizer.pt",
            )
            evaluations[str(step)] = evaluate(model, processor, development, text)
            dump(out / "checkpoint_evaluations.json", evaluations)
        dump(out / "training_log.json", logs)
    torch.cuda.synchronize()
    result = dict(
        baseline=baseline,
        checkpoints=evaluations,
        elapsed_seconds=time.perf_counter() - start,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        models=str(model_root),
        production_promoted=False,
        protected_test_inference=False,
    )
    del model
    gc.collect()
    assert all(sha256(Path(r["path"])) == r["expected_sha256"] for r in checks)
    dump(out / "training_results.json", result)
    print(out, flush=True)


if __name__ == "__main__":
    main()
