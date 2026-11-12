"""Stage 9K protected qualitative evaluation of the supplied unannotated synthetic cohort."""

import hashlib
import html
import subprocess
import threading
import time
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import psutil
import torch
from PIL import Image
from scipy.ndimage import label
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from roadmarking.inference.calibration import iou_matrix
from roadmarking.inference.prediction_recovery import filter_geographic
from stage9d_common import ROOT, dump, frozen_checks, read, sha256, verify_frozen

SOURCE = ROOT / "roadway_mixed_raw_test_500"
BASE = ROOT / "outputs/stage9k_mixed_500_test"
REPORTS = ROOT / "outputs/reports"
FROZEN = REPORTS / "final_candidate/frozen_candidate.json"
CLASSES = {0: "longitudinal_solid", 1: "longitudinal_dash"}
SEED = 895
SHAPE = (512, 512)


def signature(pixels):
    thumb = np.asarray(Image.fromarray(pixels).resize((32, 32), Image.Resampling.BILINEAR)).astype(
        np.int16
    )
    small = np.asarray(Image.fromarray(pixels).resize((9, 9), Image.Resampling.BILINEAR))
    bits = np.r_[small[:, 1:] > small[:, :-1], small[1:, :].T > small[:-1, :].T]
    return np.packbits(bits), thumb


def similar(test, refs):
    matches = []
    if not refs:
        return matches
    fingerprints = np.asarray([f for _, f, _ in refs])
    thumbnails = np.asarray([a for _, _, a in refs])
    popcount = np.array([i.bit_count() for i in range(256)])
    for name, bits, thumb in test:
        distance = popcount[fingerprints ^ bits].sum(axis=1)
        for j in np.flatnonzero(distance <= 8):
            target, _, _ = refs[j]
            if target == name:
                continue
            mae = float(np.abs(thumbnails[j] - thumb).mean())
            if mae <= 6:
                matches.append(
                    dict(image=name, reference=target, hamming=int(distance[j]), mae=mae)
                )
    return matches


def inventory(out):
    files, tests, signatures = [], [], []
    seen, decoded = defaultdict(list), defaultdict(list)
    for path in sorted(SOURCE.rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(SOURCE).as_posix()
        record = dict(file=name, bytes=path.stat().st_size, sha256=sha256(path))
        seen[record["sha256"]].append(name)
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
            try:
                with Image.open(path) as image:
                    image.load()
                    pixels = np.asarray(image)
                    record.update(
                        format=image.format,
                        size=list(image.size),
                        mode=image.mode,
                        dtype=str(pixels.dtype),
                        channels=len(image.getbands()),
                        bit_depth=pixels.dtype.itemsize * 8,
                        intensity_range=[int(pixels.min()), int(pixels.max())],
                        intensity_p01_p50_p99=np.percentile(pixels, [1, 50, 99]).tolist(),
                        metadata_keys=sorted(image.info),
                    )
                    if path.name == "PREVIEW_20_RAW.jpg":
                        record["excluded"] = "README identifies it as a preview contact sheet"
                    elif image.mode != "L" or image.size != SHAPE or pixels.dtype != np.uint8:
                        record["excluded"] = (
                            "Input is outside this sealed native grayscale contract"
                        )
                    else:
                        tests.append(record)
                        signatures.append((name, *signature(pixels)))
                        decoded[hashlib.sha256(pixels.tobytes()).hexdigest()].append(name)
            except (OSError, ValueError) as exc:
                record["excluded"] = f"Unreadable image: {exc}"
        files.append(record)
    self_matches = [m for m in similar(signatures, signatures) if m["image"] < m["reference"]]
    archive = SOURCE.with_suffix(".zip")
    archive_record = None
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            contents = {m.filename: m.file_size for m in z.infolist() if not m.is_dir()}
            archived = {
                name.removeprefix(f"{SOURCE.name}/"): size for name, size in contents.items()
            }
            matches = sum(
                record["file"] in archived and archived[record["file"]] == record["bytes"]
                for record in files
            )
            archive_record = dict(
                path=str(archive),
                sha256=sha256(archive),
                members=len(contents),
                matching_relative_name_and_size=matches,
            )
    description = dict(
        source=str(SOURCE),
        files=files,
        file_count=len(files),
        images_discovered=sum("format" in r or "excluded" in r for r in files),
        images_eligible=len(tests),
        images_excluded=[r for r in files if "excluded" in r],
        file_formats=dict(Counter(Path(r["file"]).suffix.lower() for r in files)),
        exact_file_duplicate_groups=[v for v in seen.values() if len(v) > 1],
        exact_decoded_duplicate_groups=[v for v in decoded.values() if len(v) > 1],
        suspected_near_duplicates=self_matches,
        companion_archive=archive_record,
        provenance="README states procedurally synthesized LiDAR-style images, not real LiDAR",
        annotation_files=[],
        observation_masks=[],
        source_las_laz=[],
        split_manifests=[],
        geographic_metadata=None,
        acquisition_ids=None,
        class_coverage="README only; not verified",
        annotation_status="GROUND_TRUTH_REQUIRED",
        evaluation_mode="MODE_C_QUALITATIVE_ONLY",
    )
    dump(out / "inventory/dataset_inventory.json", description)
    return tests, signatures, description


def independence(out, tests, signatures):
    references = []
    paths = set()
    for folder in ("road_markings_synthetic_500", "road_text_lidar_synthetic_500"):
        paths.update(
            p
            for p in (ROOT / folder).rglob("*.png")
            if p.parent.name
            in {
                "images",
                "intensity",
            }
        )
    stage6 = read(REPORTS / "stage6_review/latest.json")
    datasets = [
        Path(stage6["dataset"]) / "metadata/dataset_manifest.json",
        Path(stage6["fresh_dev"]) / "manifest.json",
    ]
    crop_paths = set()
    for dataset in datasets:
        for crop in read(dataset)["crops"]:
            if crop.get("split") in {"train", "val"} or dataset.name == "manifest.json":
                crop_paths.add(
                    (dataset.parent.parent if dataset.parent.name == "metadata" else dataset.parent)
                    / crop["file"]
                )
    decoded_refs = defaultdict(list)
    for path in sorted(paths | crop_paths):
        if path.suffix == ".npz":
            with np.load(path) as data:
                rgb = data["rgb"]
                pixels = np.array(Image.fromarray(rgb).convert("L"))
        else:
            with Image.open(path) as image:
                pixels = np.array(image.convert("L"))
        relative = path.relative_to(ROOT).as_posix()
        references.append((relative, *signature(pixels)))
        if pixels.shape == SHAPE:
            decoded_refs[hashlib.sha256(pixels.tobytes()).hexdigest()].append(relative)
    near = similar(signatures, references)
    by_image = defaultdict(list)
    for m in near:
        by_image[m["image"]].append(m)
    rows = []
    for t in tests:
        name = t["file"]
        with Image.open(SOURCE / name) as image:
            pixel_hash = hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
        exact = decoded_refs.get(pixel_hash, [])
        status = (
            "CONFIRMED_OVERLAP"
            if exact
            else "POTENTIAL_OVERLAP" if by_image[name] else "PROVENANCE_INSUFFICIENT"
        )
        rows.append(
            dict(
                image=name,
                classification=status,
                decoded_sha256=pixel_hash,
                exact_comparator_pixels=exact,
                near_comparator_images=by_image[name],
            )
        )
    metadata_sources = [
        "stage9/20261107T100706188083Z/physical_line_audit.json",
        "stage8b/evaluation_eligibility.json",
        "stage9b/latest.json",
        "stage7/7A_protected_test_provenance.json",
        "stage8_new_solid_review/latest.json",
        "stage9i/latest.json",
    ]
    hashes = {p: sha256(REPORTS / p) for p in metadata_sources}
    dump(
        out / "inventory/independence_audit.json",
        dict(
            status="DATASET_INDEPENDENCE_UNVERIFIED",
            images=rows,
            counts=dict(Counter(r["classification"] for r in rows)),
            comparison_images=len(references),
            approved_crop_comparators=len(crop_paths),
            synthetic_comparators=len(paths),
            reference_metadata_hashes=hashes,
            comparison="Exact decoded grayscale pixels, 144-bit difference hash <=8 plus thumbnail "
            "mean absolute error <=6 for approximate appearance. No protected pixels were decoded.",
            limits="Historical physical/geographic comparisons and the 500-unit guard "
            "are inapplicable because input has no real-world coordinates, road IDs, "
            "source windows or physical IDs.",
            known_real_roads=0,
            known_real_acquisitions=0,
            distinct_synthetic_scene_sources=None,
        ),
    )
    return rows


def freeze(out):
    checks = frozen_checks()
    verify_frozen(checks)
    contract = read(FROZEN)
    weights = Path(contract["weights"])
    assert sha256(weights) == contract["weights_sha256"]
    processor_path = Path(contract["processor"])
    processor = Mask2FormerImageProcessor.from_pretrained(processor_path, local_files_only=True)
    model, loading = Mask2FormerForUniversalSegmentation.from_pretrained(
        weights.parent,
        local_files_only=True,
        output_loading_info=True,
    )
    assert not any(loading.values()), loading
    assert model.config.id2label == CLASSES
    assert model.config.label2id == {v: k for k, v in CLASSES.items()}
    assert model.config.num_labels == 2 and model.class_predictor.out_features == 3
    assert not processor.do_resize
    stitch_path = ROOT / "src/roadmarking/inference/stitching.py"
    assert sha256(stitch_path) == contract["stitching_code_sha256"]
    paths = [
        FROZEN,
        weights,
        weights.parent / "config.json",
        processor_path / "preprocessor_config.json",
        stitch_path,
        ROOT / "scripts/final_candidate.py",
        Path(__file__).resolve(),
        ROOT / "src/roadmarking/training/expanded_continuous.py",
        ROOT / "src/roadmarking/inference/prediction_recovery.py",
        ROOT / "src/roadmarking/inference/calibration.py",
    ]
    import importlib.metadata as metadata

    dump(
        out / "inventory/frozen_model.json",
        dict(
            contract=contract,
            hashes={str(p): sha256(p) for p in paths},
            class_mapping=CLASSES,
            label2id=model.config.label2id,
            no_object_index=2,
            architecture=model.config.architectures,
            backbone=model.config.backbone_config.model_type,
            processor=read(processor_path / "preprocessor_config.json"),
            strict_reload=True,
            crop_size=list(SHAPE),
            tile_overlap=None,
            mask_logit_threshold=0.0,
            min_mask_area=4,
            raw_score_threshold=0.25,
            mask_quality_threshold=0.0,
            v1_nms_iou=0.3,
            stitching="Not executed: no verified common native spatial continuity",
            absent_observation_support=(
                "Full image used as scoring domain only, no NoData assumptions"
            ),
            inference_dtype="fp16 CUDA autocast, float32 interpolation and scoring",
            seed=SEED,
            cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(0),
            dependencies={
                p: metadata.version(p)
                for p in ("torch", "transformers", "numpy", "Pillow", "scipy", "psutil")
            },
        ),
    )
    return model.cuda().eval(), processor, contract["thresholds_V1"], checks


@torch.inference_mode()
def infer(model, processor, image, thresholds):
    tick = time.perf_counter()
    rgb = np.repeat(image[..., None], 3, axis=2)
    x = processor(images=rgb, return_tensors="pt").pixel_values.cuda()
    assert x.shape == (1, 3, *SHAPE)
    torch.cuda.synchronize()
    pre = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.float16):
        pred = model(pixel_values=x, output_auxiliary_logits=False)
    torch.cuda.synchronize()
    net = time.perf_counter()
    native = torch.nn.functional.interpolate(
        pred.masks_queries_logits.float(),
        size=SHAPE,
        mode="bilinear",
        align_corners=False,
    )[0]
    probs = native.sigmoid().cpu().numpy()
    logits = native.cpu().numpy()
    confidence, labels = pred.class_queries_logits[0].float().softmax(-1).max(-1)
    assert np.isfinite(logits).all() and torch.isfinite(confidence).all()
    masks, kinds, scores, quality, query, class_probs = [], [], [], [], [], []
    for i, (cls, p) in enumerate(zip(labels.tolist(), confidence.tolist(), strict=True)):
        if cls == 2:
            continue
        mask = logits[i] > 0
        if mask.sum() < 4:
            continue
        q = float(probs[i][mask].mean())
        s = p * q
        if s >= 0.25:
            masks.append(mask)
            kinds.append(cls)
            scores.append(s)
            quality.append(q)
            class_probs.append(p)
            query.append(i)
    masks = np.asarray(masks, bool).reshape(len(masks), *SHAPE)
    classes, scores, quality = np.asarray(kinds, int), np.asarray(scores), np.asarray(quality)
    selected = (
        filter_geographic(
            scores,
            classes,
            quality,
            iou_matrix(masks, masks, np.ones(SHAPE, bool)),
            ["image"] * len(scores),
            dict(confidence={"0": 0.0, "1": 0.0}, mask_quality=0.0, nms_iou=0.3),
        )
        if len(scores)
        else []
    )
    kept = [j for j in selected if scores[j] >= thresholds[str(classes[j])]]
    arrays = dict(
        masks=np.packbits(masks.reshape(len(masks), SHAPE[0] * SHAPE[1]), axis=1),
        shape=np.asarray(SHAPE),
        classes=classes,
        scores=scores,
        qualities=quality,
        class_probabilities=np.asarray(class_probs),
        query_indices=np.asarray(query, int),
        accepted=np.asarray(kept, int),
    )
    return (
        masks,
        arrays,
        dict(
            preprocessing_seconds=pre - tick,
            inference_seconds=net - pre,
            postprocessing_seconds=time.perf_counter() - net,
        ),
    )


def gpu_monitor(stop, samples):
    while not stop.is_set():
        try:
            query = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            usage, mem = (int(x.strip()) for x in query.stdout.splitlines()[0].split(","))
            samples.append(dict(gpu_utilization_percent=usage, total_gpu_memory_mib=mem))
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            pass
        stop.wait(2)


def outputs(out, image, sid, masks, arrays):
    assert masks.shape[1:] == image.shape == SHAPE
    accepted = arrays["accepted"]
    planes = np.zeros((2, *SHAPE), bool)
    counts = Counter()
    disconnected = []
    for i in accepted:
        cls = arrays["classes"][i]
        counts[int(cls)] += 1
        planes[cls] |= masks[i]
        fragments = label(masks[i], structure=np.ones((3, 3), bool))[1]
        if fragments > 1:
            disconnected.append(dict(query=int(arrays["query_indices"][i]), fragments=fragments))
    code = planes[0].astype(np.uint8) + 2 * planes[1].astype(np.uint8)
    Image.fromarray(code).save(out / "prediction_masks" / f"{sid}.png")
    with Image.open(out / "prediction_masks" / f"{sid}.png") as saved:
        assert np.array_equal(np.asarray(saved), code)
    canvas = np.repeat(image[..., None], 3, axis=2)
    for value, color in ((1, (17, 230, 185)), (2, (42, 121, 255)), (3, (107, 166, 253))):
        select = code == value
        canvas[select] = (0.25 * canvas[select] + 0.75 * np.array(color)).astype(np.uint8)
    Image.fromarray(canvas).save(out / "overlays" / f"{sid}.png")
    np.savez_compressed(out / "inference_predictions" / f"{sid}.npz", **arrays)
    return dict(
        raw_candidates=len(masks),
        accepted_solid=counts[0],
        accepted_dash=counts[1],
        accepted_pixels=int((code > 0).sum()),
        cross_class_pixels=int((code == 3).sum()),
        disconnected_accepted_queries=disconnected,
        raw_nodata=None,
        unclipped_accepted_nodata=None,
        clipped_accepted_nodata=None,
        predictions=[
            dict(
                query=int(arrays["query_indices"][i]),
                class_name=CLASSES[int(arrays["classes"][i])],
                score=float(arrays["scores"][i]),
                mask_quality=float(arrays["qualities"][i]),
                class_probability=float(arrays["class_probabilities"][i]),
            )
            for i in accepted
        ],
    )


def galleries(out, rows):
    from urllib.parse import quote

    header = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Stage 9K image review</title><style>
body{font:16px system-ui;margin:2rem;background:#111827;color:#f3f4f6}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem}
article{background:#1f2937;padding:1rem} img{width:100%;height:auto}
a{color:#93c5fd} p{line-height:1.45}
</style>"""
    cards, blind = [], []
    for r in rows:
        sid = html.escape(r["id"])
        original = quote((SOURCE / r["source"]).as_uri(), safe=":/%")
        cards.append(
            f'<article><h2>{sid}</h2><img loading="lazy" src="overlays/{sid}.png" '
            f'alt="Predictions for {sid}"><p>Solid {r["accepted_solid"]} | Dash '
            f'{r["accepted_dash"]} | <a href="{original}">Original</a> | '
            f'<a href="prediction_masks/{sid}.png">Mask codes</a></p></article>'
        )
        blind.append(
            f'<article><h2>{sid}</h2><a href="{original}">'
            f'<img loading="lazy" src="{original}" alt="{sid} original"></a></article>'
        )
    (out / "gallery.html").write_text(
        header + "<h1>Frozen recognition overlays</h1><p>Teal solid, blue dash. "
        "Counts are accepted queries. Synthetic and unscored.</p><main>"
        + "".join(cards)
        + "</main>",
        encoding="utf-8",
    )
    (out / "blind_annotation/inputs.html").write_text(
        header + "<h1>Independent annotation inputs</h1><main>" + "".join(blind) + "</main>",
        encoding="utf-8",
    )


def main(validated_intake=None):
    start = time.perf_counter()
    out = BASE / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    for folder in (
        "inventory",
        "inference_predictions",
        "original_references",
        "overlays",
        "prediction_masks",
        "metrics",
        "failure_cases",
        "reports",
        "qa",
        "blind_annotation",
    ):
        (out / folder).mkdir(parents=True)
    print(out, flush=True)
    if validated_intake is None:
        tests, signatures, description = inventory(out)
        independence(out, tests, signatures)
    else:
        description = read(validated_intake)
        assert Path(description["source"]).resolve() == SOURCE.resolve()
        files = {r["file"]: r for r in description["files"]}
        assert set(files) == {
            p.relative_to(SOURCE).as_posix() for p in SOURCE.rglob("*") if p.is_file()
        }
        for name, record in files.items():
            assert sha256(SOURCE / name) == record["sha256"], name
        tests = [record for record in files.values() if record.get("eligible_for_intake")]
        assert len(tests) == description["eligible_native_png"]
        dump(out / "inventory/dataset_inventory.json", description)
    assert tests, "No eligible input images"
    dump(
        out / "inventory/preprocessing.json",
        dict(
            grayscale="8-bit native samples repeated into RGB without changing pixel values",
            normalization="Frozen processor rescale 1/255 and channel mean/std, no resize",
            crop_shape=list(SHAPE),
            border_or_domain="Synthetic styling may differ from LiDAR",
            support=(
                "No authoritative observed/NoData masks or physical scale. Dark pixels retained."
            ),
            image_statistics="See per-file dtype, intensity range and percentiles in inventory",
        ),
    )
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    model, processor, thresholds, checks = freeze(out)
    fixed = [tests[i]["file"] for i in sorted({0, len(tests) // 2, len(tests) - 1})]
    repeat = {}
    for name in fixed:
        with Image.open(SOURCE / name) as f:
            pixels = np.array(f)
        _, before, _ = infer(model, processor, pixels, thresholds)
        _, after, _ = infer(model, processor, pixels, thresholds)
        result = {key: np.array_equal(before[key], after[key]) for key in before}
        assert all(result.values()), (name, result)
        repeat[name] = result
    torch.cuda.reset_peak_memory_stats()
    stop, utilization = threading.Event(), []
    sampler = threading.Thread(target=gpu_monitor, args=(stop, utilization), daemon=True)
    sampler.start()
    rows, failures = [], []
    tick = time.perf_counter()
    try:
        for n, rec in enumerate(tests, 1):
            path = SOURCE / rec["file"]
            image_start = time.perf_counter()
            try:
                with Image.open(path) as f:
                    pixels = np.array(f)
                masks, arrays, timing = infer(model, processor, pixels, thresholds)
                findings = outputs(out, pixels, path.stem, masks, arrays)
                rows.append(
                    dict(
                        id=path.stem,
                        source=rec["file"],
                        **timing,
                        **findings,
                        end_to_end_seconds=time.perf_counter() - image_start,
                    )
                )
            except (OSError, ValueError, RuntimeError) as exc:
                failures.append(dict(file=rec["file"], error=str(exc)))
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
            if n % 50 == 0:
                print(f"{n}/{len(tests)} checked, {len(failures)} errors", flush=True)
    finally:
        stop.set()
        sampler.join(timeout=4)
    elapsed = time.perf_counter() - tick
    memory = psutil.Process().memory_info()
    perf = dict(
        loop_seconds=elapsed,
        total_net_seconds=sum(r["inference_seconds"] for r in rows),
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        process_peak_working_set_bytes=getattr(memory, "peak_wset", None),
        gpu_samples=utilization,
        phase_latency_seconds=(
            {
                key: dict(
                    zip(
                        ("p50", "p95"),
                        np.percentile([r[key] for r in rows], [50, 95]).tolist(),
                        strict=True,
                    )
                )
                for key in (
                    "preprocessing_seconds",
                    "inference_seconds",
                    "postprocessing_seconds",
                    "end_to_end_seconds",
                )
            }
            if rows
            else {}
        ),
    )
    dump(
        out / "metrics/prediction_summary.json",
        dict(
            eligible=len(tests),
            processed=len(rows),
            failures=failures,
            performance=perf,
            rows=rows,
        ),
    )
    dump(
        out / "failure_cases/review_candidates.json",
        dict(
            status="Unscored review candidates",
            errors=failures,
            no_accepted=[r["id"] for r in rows if not r["predictions"]],
            disconnected=[r["id"] for r in rows if r["disconnected_accepted_queries"]],
            cross_class_overlap=[r["id"] for r in rows if r["cross_class_pixels"]],
        ),
    )
    dump(
        out / "original_references/manifest.json",
        [dict(source=str(SOURCE / r["file"]), sha256=r["sha256"]) for r in tests],
    )
    dump(
        out / "blind_annotation/inputs.json",
        [
            dict(
                id=Path(r["file"]).stem,
                source=str(SOURCE / r["file"]),
                sha256=r["sha256"],
                size=r["size"],
            )
            for r in tests
        ],
    )
    galleries(out, rows)
    verify_frozen(checks)
    for r in description["files"]:
        assert sha256(SOURCE / r["file"]) == r["sha256"], r["file"]
    dump(
        out / "qa/inference_checks.json",
        dict(
            repeatability=repeat,
            frozen_hashes_verified=True,
            input_hashes_verified=len(description["files"]),
            output_masks_verified=len(rows),
            class_mapping_verified=True,
            no_data_alignment="Unavailable",
            total_pipeline_seconds=time.perf_counter() - start,
        ),
    )
    print(out, len(rows), len(failures), flush=True)


if __name__ == "__main__":
    main()
