"""One frozen inference pass after explicit three-region geometry/completeness approval."""

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import rasterio
import torch
from affine import Affine
from PIL import Image
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, label
from shapely.geometry import box
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from roadmarking.dataset.labels import LabelFeature
from roadmarking.dataset.phase55 import write_attestation
from roadmarking.dataset.verified_masks import reviewed_masks
from roadmarking.inference.calibration import aggregate_counts, iou_matrix, operating_counts
from roadmarking.rasterization.pipeline import dump, sha256
from roadmarking.training.continuous import evaluate
from roadmarking.training.pilot import instance_metrics

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def record(masks, scores, labels, targets, target_labels, valid):
    return {
        "masks": masks,
        "scores": scores,
        "labels": labels,
        "targets": targets,
        "target_labels": target_labels,
        "valid": valid,
        "iou": iou_matrix(masks, targets, valid),
    }


def main():
    began = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    context = json.loads((root / "outputs/reports/phase5_6b/latest.json").read_text())
    spec = json.loads(Path(context["spec"]).read_text())
    prior = Path(context["spec"]).parent
    frozen = json.loads((prior / "frozen_model.json").read_text())
    for filename, expected in frozen["artifact_sha256"].items():
        assert sha256(filename) == expected, filename
    frame = gpd.read_file(context["candidate_store"], layer="corrections")
    assert frame.geometry.is_valid.all() and frame.correction_id.is_unique
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    report = root / "outputs/reports/phase5_6_independent" / run_id
    dataset = root / "data/processed/phase5_6_approved_test" / run_id
    report.mkdir(parents=True)
    dataset.mkdir(parents=True)
    approval = {
        "source": "User explicitly answered yes to geometry and complete visible-target "
        "coverage in all three test regions, excluding NoData and unsupported symbols",
        "region_ids": [t["candidate_id"] for t in spec["tiles"]],
        "candidate_store": context["candidate_store"],
        "candidate_sha256": sha256(context["candidate_store"]),
        "project": context["project"],
        "project_sha256": sha256(context["project"]),
        "scope": "Observed pixels within the exact three reviewed test footprints only",
        "geometry": [
            {
                "id": r.correction_id,
                "source_id": r.global_id,
                "action": r.edit_action,
                "class": int(r.visual_id),
                "sha256": hashlib.sha256(r.geometry.wkb).hexdigest(),
            }
            for _, r in frame.iterrows()
        ],
        "rasters": {p: sha256(p) for t in spec["tiles"] for p in t["layers"].values()},
        "frozen_model_sha256": sha256(prior / "frozen_model.json"),
    }
    approval_hash = write_attestation(report / "three_region_approval.json", approval)
    train_context = json.loads((root / "outputs/reports/phase5_5/latest.json").read_text())
    train_manifest = json.loads(
        (Path(train_context["dataset"]) / "metadata/dataset_manifest.json").read_text()
    )
    used_ids = set().union(*(set(r["source_global_ids"]) for r in train_manifest["regions"]))
    assert not (set(frame.global_id.dropna()) & used_ids)
    for tile in spec["tiles"]:
        assert (
            min(
                box(*tile["bounds"]).distance(box(*r["native_xy_bbox"]))
                for r in train_manifest["regions"]
            )
            > 10
        )
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    checkpoint = Path(frozen["checkpoint"])
    model = (
        Mask2FormerForUniversalSegmentation.from_pretrained(
            checkpoint / "model", local_files_only=True
        )
        .to("cuda")
        .eval()
    )
    processor = Mask2FormerImageProcessor.from_pretrained(
        checkpoint / "processor", local_files_only=True
    )
    torch.cuda.reset_peak_memory_stats()
    crops, regions = [], []
    tick = time.perf_counter()
    for tile in spec["tiles"]:
        name = tile["candidate_id"]
        affine = Affine(*tile["affine"])
        with rasterio.open(tile["layers"]["observed"]) as ds:
            observed = ds.read(1).astype(bool)
        rows = frame[(frame.candidate_id == name) & (frame.edit_action != "REMOVE")]
        features = [
            LabelFeature(
                i + 1, r.global_id or r.correction_id, int(r.visual_id), "polygon", r.geometry
            )
            for i, (_, r) in enumerate(rows.iterrows())
            if int(r.visual_id) in (1, 2)
        ]
        unsupported = [r.geometry for _, r in rows.iterrows() if int(r.visual_id) not in (1, 2)]
        excluded = np.zeros(observed.shape, bool)
        if unsupported:
            excluded = binary_dilation(
                rasterize([(g, 1) for g in unsupported], observed.shape, transform=affine).astype(
                    bool
                ),
                iterations=2,
            )
        labels, _ = reviewed_masks(
            affine,
            observed.shape,
            observed,
            np.ones_like(observed),
            features,
            completeness=box(*tile["bounds"]),
        )
        valid = np.isin(labels.state, [2, 3]) & observed & ~excluded
        targets, classes, ids = [], [], []
        for entry in labels.instances:
            mask = (labels.instance == entry["local_id"]) & valid
            if entry["local_id"] and mask.sum() >= 4:
                targets.append(mask)
                classes.append(entry["visual_id"] - 1)
                ids.append(entry["global_id"])
        targets = np.stack(targets) if targets else np.empty((0, 1024, 1024), bool)
        rgb = np.asarray(Image.open(tile["images"]["grayscale"]).convert("RGB"))
        region_crops = []
        for row in (0, 256, 512):
            for col in (0, 256, 512):
                v = valid[row : row + 512, col : col + 512]
                if not v.any():
                    continue
                masks = targets[:, row : row + 512, col : col + 512].copy()
                keep = masks.sum((1, 2)) >= 4
                # Tiny partial targets remain ignored, never turn into background.
                if (~keep).any():
                    v = v & ~masks[~keep].any(0)
                masks = masks[keep] & v
                path = dataset / f"{name}_r{row}_c{col}.npz"
                np.savez_compressed(
                    path,
                    rgb=rgb[row : row + 512, col : col + 512],
                    masks=masks,
                    labels=np.asarray(classes)[keep],
                    validity=v,
                )
                crop = {
                    "crop_id": path.stem,
                    "file": str(path),
                    "sha256": sha256(path),
                    "size": 512,
                    "tile_id": name,
                    "window": [row, col, 512, 512],
                    "affine": list(affine * Affine.translation(col, row))[:6],
                    "global_instance_ids": np.asarray(ids)[keep].tolist(),
                }
                crops.append(crop)
                region_crops.append(crop)
        regions.append(
            {
                "name": name,
                "rgb": rgb,
                "targets": targets,
                "target_labels": np.asarray(classes),
                "ids": ids,
                "valid": valid,
                "observed": observed,
                "crops": region_crops,
            }
        )
        np.savez_compressed(
            dataset / (name + "_ground_truth.npz"),
            masks=targets,
            labels=classes,
            validity=valid,
            observed=observed,
        )
    export_seconds = time.perf_counter() - tick
    dump(
        dataset / "manifest.json",
        {
            "approval_sha256": approval_hash,
            "crops": crops,
            "role": "INDEPENDENT_TEST_ONLY_NEVER_TRAIN",
        },
    )
    # Exactly one network forward per crop. Stored logits/masks permit later audit without rerun.
    metrics = evaluate(
        model, processor, dataset, crops, frozen["postprocessing"], report / "inference"
    )
    del model
    reconcile_tick = time.perf_counter()
    region_records, per_region, geometry = [], {}, []
    for region in regions:
        proposals = []
        for crop in region["crops"]:
            data = np.load(report / "inference" / (crop["crop_id"] + ".npz"))
            row, col, _, _ = crop["window"]
            for index in data["selected"].astype(int):
                mask = np.zeros((1024, 1024), bool)
                mask[row : row + 512, col : col + 512] = data["masks"][index] & data["validity"]
                proposals.append(
                    (
                        float(data["scores"][index]),
                        int(data["labels"][index]),
                        mask,
                        crop["crop_id"],
                    )
                )
        selected, suppressed = [], 0
        for proposal in sorted(proposals, key=lambda p: -p[0]):
            _score, cls, mask, _ = proposal
            if any(
                cls == p[1] and (mask & p[2]).sum() / max(1, (mask | p[2]).sum()) > 0.3
                for p in selected
            ):
                suppressed += 1
            else:
                selected.append(proposal)
        masks = np.stack([p[2] for p in selected]) if selected else np.empty((0, 1024, 1024), bool)
        rec = record(
            masks,
            np.asarray([p[0] for p in selected]),
            np.asarray([p[1] for p in selected]),
            region["targets"],
            region["target_labels"],
            region["valid"],
        )
        rec["ids"], rec["region"] = region["ids"], region["name"]
        region_records.append(rec)
        chosen = list(range(len(masks)))
        counts = aggregate_counts([rec], [chosen])
        gt_union, pr_union = region["targets"].any(0), masks.any(0)
        fg_iou = (gt_union & pr_union).sum() / max(
            1, ((gt_union | pr_union) & region["valid"]).sum()
        )
        per_region[region["name"]] = {
            "counts": counts,
            "foreground_iou": float(fg_iou),
            "ap": instance_metrics(
                masks,
                rec["scores"],
                rec["labels"],
                rec["targets"],
                rec["target_labels"],
                rec["valid"],
            ),
            "cross_crop_suppressed_duplicates": suppressed,
            "unobserved_pixels_excluded": int((~region["observed"]).sum()),
            "observable_unique_targets": len(rec["ids"]),
        }
        ops = operating_counts(rec, chosen)
        false, missed = [], []
        for cls in ("0", "1"):
            false.extend(ops[cls]["false_indices"])
            missed.extend(ops[cls]["missed_indices"])
            for pred, target in ops[cls]["assignments"].items():
                a, b = masks[pred], rec["targets"][target]
                ba, bb = a & ~binary_erosion(a), b & ~binary_erosion(b)
                distances = (
                    np.r_[distance_transform_edt(~bb)[ba], distance_transform_edt(~ba)[bb]] * 0.05
                )
                geometry.append(
                    {
                        "region": region["name"],
                        "source_id": rec["ids"][target],
                        "mean_symmetric_observed_boundary_distance_m": float(distances.mean()),
                        "boundary_distance_p95_m": float(np.percentile(distances, 95)),
                        "prediction_components": int(label(a)[1]),
                        "truth_components": int(label(b)[1]),
                        "note": (
                            "Observed support only; scan holes affect boundaries/components; "
                            "not full-paint lateral error"
                        ),
                    }
                )
        np.savez_compressed(
            report / (region["name"] + "_global_predictions.npz"),
            masks=masks,
            scores=rec["scores"],
            labels=rec["labels"],
            validity=rec["valid"],
            source_crop_ids=[p[3] for p in selected],
        )
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        overlays = [
            [],
            list(rec["targets"]),
            list(masks),
            [masks[i] for i in false],
            [rec["targets"][i] for i in missed],
            [~region["valid"]],
        ]
        for ax, title, overlay in zip(
            axes.flat,
            [
                "Intensity",
                "Approved ground truth",
                "Frozen predictions",
                "False positives",
                "Missed markings",
                "Excluded pixels",
            ],
            overlays,
            strict=True,
        ):
            ax.imshow(region["rgb"])
            for mask in overlay:
                ax.imshow(
                    np.ma.masked_where(~mask.astype(bool), mask),
                    alpha=0.6,
                    cmap="autumn",
                    vmin=0,
                    vmax=1,
                )
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(region["name"] + " — independent observed-support evaluation")
        fig.tight_layout()
        fig.savefig(report / (region["name"] + "_comparison.png"), dpi=120)
        plt.close(fig)
    # Pool by physical client identity: all disjoint observed fragments contribute to GT area.
    global_ids = sorted({uid for rec in region_records for uid in rec["ids"]})
    area = {
        uid: sum(
            int(rec["targets"][j].sum())
            for rec in region_records
            for j, value in enumerate(rec["ids"])
            if value == uid
        )
        for uid in global_ids
    }
    labels = {
        uid: int(rec["target_labels"][j])
        for rec in region_records
        for j, uid in enumerate(rec["ids"])
    }
    pooled_iou, pooled_scores, pooled_labels = [], [], []
    for rec in region_records:
        for i, mask in enumerate(rec["masks"]):
            row = np.zeros(len(global_ids))
            for j, uid in enumerate(rec["ids"]):
                intersection = int((mask & rec["targets"][j]).sum())
                row[global_ids.index(uid)] = intersection / max(
                    1, int(mask.sum()) + area[uid] - intersection
                )
            pooled_iou.append(row)
            pooled_scores.append(rec["scores"][i])
            pooled_labels.append(rec["labels"][i])
    pooled = {
        "iou": np.asarray(pooled_iou).reshape(-1, len(global_ids)),
        "scores": np.asarray(pooled_scores),
        "labels": np.asarray(pooled_labels),
        "target_labels": np.asarray([labels[uid] for uid in global_ids]),
    }
    pooled_counts = aggregate_counts([pooled], [list(range(len(pooled_scores)))])
    dump(report / "geometry_observed_support.json", geometry)
    result = {
        "status": "INDEPENDENT_TEST_COMPLETE",
        "readiness": "RECOGNITION_IMPROVEMENT_REQUIRED",
        "per_region": per_region,
        "pooled_geographic": pooled_counts,
        "pooled_unique_gt_ids": global_ids,
        "protocol": (
            "Frozen thresholds; global same-class NMS .3; pooled IoU uses "
            "all observed fragments per source ID"
        ),
        "inference_count": len(crops),
        "approval_sha256": approval_hash,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "runtime": {
            "export_image_loading_seconds": export_seconds,
            "model_only_seconds": metrics["model_only_seconds_total"],
            "reconciliation_metrics_and_previews_seconds": time.perf_counter() - reconcile_tick,
            "total_end_to_end_seconds": time.perf_counter() - began,
        },
        "limitations": [
            "Sparse observed support, not full-paint geometry",
            "Shared long source features pooled; regions are not three independent samples",
            "Full centerline lateral deviation, width, continuity and XYZ recovery not established",
            "AP is mean per-region confidence-truncated 101-point AP; pooled COCO AP not claimed",
        ],
    }
    dump(report / "report.json", result)
    dump(
        report.parent / "latest.json",
        {"report_dir": str(report), "dataset": str(dataset), "status": result["status"]},
    )
    metrics["development_validation_only"] = False
    metrics["role"] = "INDEPENDENT_TEST_APPROVED"
    dump(report / "per_crop_metrics.json", metrics)
    (report / "report.md").write_text(
        "# Independent frozen-model evaluation\n\n"
        "Decision: RECOGNITION_IMPROVEMENT_REQUIRED; not approved for production vectorization.\n\n"
        "Three-region geometry and complete visible-target approval recorded with immutable "
        "hash-bound scope. Only observed supported solid/dash pixels supervised; NoData, "
        "unsupported symbols, ambiguous boundaries and conflicts excluded.\n\n"
        f"Pooled geographic recognition: `{pooled_counts}`.\n\n"
        "Per-region precision/recall/F1, AP50/AP75, IoU, suppression counts and missing "
        "observations are in report.json; per-crop appearances are in per_crop_metrics.json. "
        "Shared client IDs are pooled once using their total observed area across regions. "
        "Do not compare these pooled counts directly with overlapping development crops.\n\n"
        "Actual image/GT/prediction/FP/FN/exclusion panels accompany each region. "
        "Observed boundary distances and fragmentation counts are conditional on correct "
        "matches; sparse scan holes limit full painted-geometry interpretation. Full width, "
        "lateral centerline, continuity and elevation accuracy remain unestablished.\n\n"
        f"Measured timing: `{result['runtime']}`; peak allocated GPU memory "
        f"{result['gpu_peak_allocated_bytes']/2**30:.3f} GiB. Total includes startup, export, "
        "inference, filtering, reconciliation, metrics and previews; "
        "model-only time is separate.\n\n"
        "No training, threshold tuning, or final GIS vectorization was performed.\n",
        encoding="utf-8",
    )
    for filename, expected in frozen["artifact_sha256"].items():
        assert sha256(filename) == expected
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
