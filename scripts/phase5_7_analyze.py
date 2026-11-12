"""Compare development checkpoints and explicit observed-overlap continuity ablations."""

import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, label
from skimage.morphology import skeletonize

from roadmarking.inference.calibration import aggregate_counts, iou_matrix, operating_counts
from roadmarking.inference.continuity import reconcile
from roadmarking.rasterization.pipeline import dump
from roadmarking.training.pilot import instance_metrics

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def analyze(run, step, dataset, output):
    folder = run / f"validation_step{step:04d}"
    metrics = json.loads((folder / "metrics.json").read_text())
    crops = metrics["crops"]
    x0, y1 = min(c["affine"][2] for c in crops), max(c["affine"][5] for c in crops)
    origins = [
        (round((y1 - c["affine"][5]) / 0.05), round((c["affine"][2] - x0) / 0.05)) for c in crops
    ]
    shape = max(r for r, _ in origins) + 512, max(c for _, c in origins) + 512
    valid, rgb = np.zeros(shape, bool), np.zeros((*shape, 3), np.uint8)
    truths, labels, proposals = {}, {}, []
    for crop, (row, col) in zip(crops, origins, strict=True):
        data = np.load(dataset / crop["file"])
        v = data["validity"].astype(bool)
        valid[row : row + 512, col : col + 512] |= v
        patch = rgb[row : row + 512, col : col + 512]
        patch[v] = data["rgb"][v]
        for mask, cls, uid in zip(
            data["masks"], data["labels"], crop["global_instance_ids"], strict=True
        ):
            truths.setdefault(uid, np.zeros(shape, bool))[row : row + 512, col : col + 512] |= (
                mask.astype(bool) & v
            )
            labels[uid] = int(cls)
        predictions = np.load(folder / (crop["crop_id"] + ".npz"))
        for index in predictions["selected"].astype(int):
            mask, support = np.zeros(shape, bool), np.zeros(shape, bool)
            mask[row : row + 512, col : col + 512] = predictions["masks"][index] & v
            support[row : row + 512, col : col + 512] = v
            proposals.append(
                {
                    "mask": mask,
                    "support": support,
                    "score": float(predictions["scores"][index]),
                    "label": int(predictions["labels"][index]),
                }
            )
    ids = sorted(truths)
    targets = np.stack([truths[uid] for uid in ids])
    target_labels = np.array([labels[uid] for uid in ids])
    result, display = {}, []
    for union, name in ((False, "global_nms_only"), (True, "observed_solid_union")):
        selected, events = reconcile(proposals, valid, union_solids=union)
        masks = np.stack([p["mask"] for p in selected]) if selected else np.empty((0, *shape), bool)
        scores, classes = np.array([p["score"] for p in selected]), np.array(
            [p["label"] for p in selected]
        )
        rec = {
            "masks": masks,
            "scores": scores,
            "labels": classes,
            "targets": targets,
            "target_labels": target_labels,
            "valid": valid,
            "iou": iou_matrix(masks, targets, valid),
        }
        indices = list(range(len(masks)))
        counts = aggregate_counts([rec], [indices])
        truth, predicted = targets.any(0), masks.any(0)
        tp = int((truth & predicted & valid).sum())
        fp, fn = int((~truth & predicted & valid).sum()), int((truth & ~predicted & valid).sum())
        skel = skeletonize(truth) & valid
        centerline = float(
            (skel & (distance_transform_edt(~predicted) <= 1)).sum() / max(1, skel.sum())
        )
        geometry = []
        ops = operating_counts(rec, indices)
        for cls in ("0", "1"):
            for pi, ti in ops[cls]["assignments"].items():
                a, b = masks[pi] & valid, targets[ti] & valid
                ba, bb = a & ~binary_erosion(a), b & ~binary_erosion(b)
                distances = (
                    np.r_[distance_transform_edt(~bb)[ba], distance_transform_edt(~ba)[bb]] * 0.05
                )
                sa, sb = skeletonize(a), skeletonize(b)
                wa, wb = (
                    2 * distance_transform_edt(a)[sa] * 0.05,
                    2 * distance_transform_edt(b)[sb] * 0.05,
                )
                geometry.append(
                    {
                        "source_id": ids[ti],
                        "class": cls,
                        "mean_observed_boundary_distance_native": float(distances.mean()),
                        "median_observed_width_difference_native": float(
                            np.median(wa) - np.median(wb)
                        ),
                        "prediction_components": int(label(a)[1]),
                        "gt_components": int(label(b)[1]),
                        "caveat": (
                            "Observed masks only; LiDAR holes affect " "skeletons/widths/components"
                        ),
                    }
                )
        agnostic = {
            **rec,
            "labels": np.zeros(len(classes), int),
            "target_labels": np.zeros(len(target_labels), int),
        }
        result[name] = {
            "counts": counts,
            "ap": instance_metrics(masks, scores, classes, targets, target_labels, valid),
            "pixel_precision": tp / max(1, tp + fp),
            "pixel_recall": tp / max(1, tp + fn),
            "foreground_iou": tp / max(1, tp + fp + fn),
            "observed_skeleton_coverage_within_one_pixel": centerline,
            "geometry": geometry,
            "events": {e: events.count(e) for e in set(events)},
            "shared_stripe_taxonomy_diagnostic": aggregate_counts([agnostic], [indices])["0"],
            "taxonomy_is_not_migrated_or_trained": True,
        }
        display.append(predicted)
        np.savez_compressed(
            output / f"dev_step{step:04d}_{name}.npz",
            masks=masks,
            scores=scores,
            labels=classes,
            validity=valid,
            targets=targets,
            target_labels=target_labels,
        )
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, title, mask in zip(
        axes,
        ["Actual development intensity", "Approved targets", "NMS only", "Observed solid union"],
        [np.zeros(shape, bool), targets.any(0), *display],
        strict=True,
    ):
        ax.imshow(rgb)
        ax.imshow(np.ma.masked_where(~mask, mask), cmap="winter", alpha=0.6, vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / f"development_step{step:04d}.png", dpi=130)
    plt.close(fig)
    return {"crop_metrics": metrics, "geographic_ablations": result}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    context = json.loads((root / "outputs/reports/phase5_7/latest.json").read_text())
    run, output, dataset = (
        Path(context["training_run"]),
        Path(context["report_dir"]),
        Path(context["dataset"]),
    )
    best = json.loads((run / "best.json").read_text())
    comparisons = {
        str(step): analyze(run, step, dataset, output) for step in sorted({0, best["step"], 400})
    }
    dump(output / "development_comparisons.json", comparisons)
    dump(
        output / "selected_inference.json",
        {
            "checkpoint": best["checkpoint"],
            "postprocessing": comparisons[str(best["step"])]["crop_metrics"]["postprocessing"],
            "continuity_refinement": "EXPERIMENTAL_ABLATION_NOT_AUTOMATICALLY_ENABLED",
            "selection": "Frozen development macro F1 only; protected test not reevaluated",
            "production_ready": False,
        },
    )
    print(
        json.dumps(
            {"best_step": best["step"], "counts": best["metrics"]["postprocessed"]}, indent=2
        )
    )
