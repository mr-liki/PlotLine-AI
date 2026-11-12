"""Compare actual saved predictions with a bounded, recall-constrained development grid."""

import itertools
import json
from pathlib import Path

import matplotlib
import numpy as np
import rasterio

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from roadmarking.inference.calibration import (
    aggregate_counts,
    filter_instances,
    iou_matrix,
    native_world_iou,
    operating_counts,
)
from roadmarking.rasterization.pipeline import dump, sha256
from roadmarking.training.pilot import instance_metrics


def read_records(folder, dataset):
    metrics = json.loads((folder / "metrics.json").read_text())
    records = []
    for crop in metrics["crops"]:
        with np.load(folder / (crop["crop_id"] + ".npz")) as data:
            r = {k: data[k].copy() for k in ("masks", "scores", "labels", "qualities")}
            r["valid"] = data["validity"].copy()
        with np.load(dataset / crop["file"]) as data:
            r.update(targets=data["masks"], target_labels=data["labels"], rgb=data["rgb"])
        r.update(crop=crop, iou=iou_matrix(r["masks"], r["targets"], r["valid"]))
        r["prediction_overlap"] = iou_matrix(r["masks"], r["masks"], r["valid"])
        records.append(r)
    return metrics, records


def selections(records, config):
    return [
        filter_instances(r["scores"], r["labels"], r["prediction_overlap"], r["qualities"], config)
        for r in records
    ]


def calibrate(records):
    trials, best = [], None
    for solid, dash, quality, nms in itertools.product(
        (0.25, 0.4, 0.55, 0.7, 0.85),
        (0.25, 0.4, 0.55, 0.7, 0.85),
        (0.5, 0.65, 0.8),
        (None, 0.3, 0.5, 0.7),
    ):
        config = {"confidence": {"0": solid, "1": dash}, "mask_quality": quality, "nms_iou": nms}
        counts = aggregate_counts(records, selections(records, config))
        eligible = counts["0"]["tp"] >= 3 and counts["1"]["tp"] >= 17
        row = {"config": config, "counts": counts, "preserves_baseline_class_recall": eligible}
        trials.append(row)
        key = (counts["macro_f1"], -sum(counts[str(i)]["fp"] for i in (0, 1)))
        if eligible and (best is None or key > best[0]):
            best = key, row
    return trials, best[1] if best else None


def details(records, chosen, config):
    counts = aggregate_counts(records, chosen)
    aps, ious, false = [], [], []
    missed_ids, seen_ids = set(), set()
    for r, selected in zip(records, chosen, strict=True):
        aps.append(
            instance_metrics(
                r["masks"][selected],
                r["scores"][selected],
                r["labels"][selected],
                r["targets"],
                r["target_labels"],
                r["valid"],
            )
        )
        truth = r["targets"].any(0)
        predicted = r["masks"][selected].any(0)
        ious.append(
            float(
                (predicted & truth & r["valid"]).sum()
                / max(1, ((predicted | truth) & r["valid"]).sum())
            )
        )
        ops = operating_counts(r, selected)
        ids = r["crop"]["global_instance_ids"]
        for cls in ("0", "1"):
            for index in ops[cls]["false_indices"]:
                false.append((r["masks"][index] & r["valid"], r["crop"], int(cls)))
            missed_ids.update(ids[i] for i in ops[cls]["missed_indices"])
            seen_ids.update(ids[i] for i in ops[cls]["assignments"].values())
    # Geographic deduplication is an explicit overlap-based proxy, not extra human truth.
    parent = list(range(len(false)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, (mask, crop, cls) in enumerate(false):
        for j in range(i):
            other, other_crop, other_cls = false[j]
            if cls == other_cls and native_world_iou(mask, crop, other, other_crop) >= 0.5:
                parent[find(i)] = find(j)
    ap_summary = {}
    for name in ("longitudinal_solid", "longitudinal_dash"):
        ap_summary[name] = {
            key: float(np.mean([a[name][key] for a in aps if key in a[name]]))
            for key in ("AP50", "AP75", "AP")
        }
    return {
        "counts": counts,
        "mean_foreground_iou": float(np.mean(ious)),
        "mean_per_crop_confidence_truncated_ap": ap_summary,
        "geographic_false_positive_clusters_iou50_proxy": len({find(i) for i in parent}),
        "unique_gt_ids_never_detected": sorted(missed_ids - seen_ids),
        "config": config,
    }


def panels(records, before, after, output):
    for ri in (0, 2, 4):
        r, new = records[0][ri], records[1][ri]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        titles = [
            "Original intensity",
            "Approved targets",
            "Phase 5.4",
            "Phase 5.5",
            "Phase 5.5 false positives",
            "Phase 5.5 missed targets",
        ]
        masks = [
            [],
            list(r["targets"]),
            list(r["masks"][before[ri]]),
            list(new["masks"][after[ri]]),
            [],
            [],
        ]
        ops = operating_counts(new, after[ri])
        for cls in ("0", "1"):
            masks[4].extend(new["masks"][i] for i in ops[cls]["false_indices"])
            masks[5].extend(new["targets"][i] for i in ops[cls]["missed_indices"])
        for index, (ax, title) in enumerate(zip(axes.flat, titles, strict=True)):
            ax.imshow(r["rgb"])
            for mask in masks[index]:
                mask = np.asarray(mask, dtype=bool)
                masked = np.ma.masked_where(~(mask & r["valid"]), mask)
                ax.imshow(
                    masked, cmap="autumn" if index == 4 else "winter", alpha=0.65, vmin=0, vmax=1
                )
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(r["crop"]["crop_id"] + " | development validation; ignored pixels hidden")
        fig.tight_layout()
        fig.savefig(output / f"comparison_{ri}.png", dpi=140)
        plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    context_path = root / "outputs/reports/phase5_5/latest.json"
    context = json.loads(context_path.read_text())
    report, dataset = Path(context["report_dir"]), Path(context["dataset"])
    run = root / "models/mask2former_swin_s" / ("phase5_5_" + context["run_id"])
    history = json.loads((run / "validation_history.json").read_text())
    frozen = json.loads((run / "training_config.json").read_text())[
        "postprocessing_frozen_before_training"
    ]
    candidate_rows, baseline_records, selected_records = [], None, None
    selected, best_key, baseline_tuned = None, None, None
    for entry in history:
        step = entry["step"]
        folder = run / f"validation_step{step:04d}"
        metrics, records = read_records(folder, dataset)
        trials, best = calibrate(records)
        dump(report / f"calibration_step{step:04d}.json", trials)
        candidate_rows.append({"step": step, "frozen": metrics, "retuned": best})
        if step == 0:
            baseline_records, baseline_tuned = records, best
        if best:
            counts = best["counts"]
            key = counts["macro_f1"], -sum(counts[str(i)]["fp"] for i in (0, 1))
            if best_key is None or key > best_key:
                selected, best_key, selected_records = (step, best), key, records
    if selected is None:
        raise ValueError("No candidate preserves baseline per-class recall")
    step, best = selected
    selected_path = run / f"checkpoint_step{step:04d}"
    before = selections(baseline_records, frozen)
    after = selections(selected_records, best["config"])
    baseline = details(baseline_records, before, frozen)
    result = details(selected_records, after, best["config"])
    bundle = json.loads((selected_path / "bundle.json").read_text())
    logs = json.loads((run / "training_log.json").read_text())
    manifest = json.loads((dataset / "metadata/dataset_manifest.json").read_text())
    unique_pixels = {"observed": [], "valid": [], "foreground": [], "background": []}
    for tile in manifest["new_tiles"]:
        with rasterio.open(tile["observed_reference"]) as source:
            observed = source.read(1).astype(bool)
        with rasterio.open(dataset / tile["files"]["state"]) as source:
            state = source.read(1)
        with rasterio.open(dataset / tile["files"]["validity"]) as source:
            valid = source.read(1).astype(bool)
        affine = tile["affine"]
        row0, col0 = round(-affine[5] / affine[0]), round(affine[2] / affine[0])
        for name, mask in (
            ("observed", observed),
            ("valid", valid),
            ("foreground", state == 3),
            ("background", state == 2),
        ):
            rows, cols = np.where(mask)
            # Global native-grid integer key; bit packing is injective for this dataset.
            keys = ((rows.astype(np.int64) + row0) << 32) | (
                (cols.astype(np.int64) + col0) & 0xFFFFFFFF
            )
            unique_pixels[name].append(keys)
    unique_counts = {
        name: len(np.unique(np.concatenate(parts))) for name, parts in unique_pixels.items()
    }
    evidence = {
        "status": "DEVELOPMENT_ENGINEERING_MODEL_NOT_PRODUCTION_READY",
        "selected_checkpoint": str(selected_path),
        "selected_step": step,
        "weight_sha256": bundle["weight_sha256"],
        "new_weight_updates_selected": step > 0,
        "baseline_frozen": baseline,
        "selected_retuned": result,
        "baseline_retuned_same_300_grid": baseline_tuned,
        "candidate_comparisons": candidate_rows,
        "grid_budget_per_checkpoint": 300,
        "selection": "Highest macro F1 preserving >=3 solid and >=17 dash true appearances",
        "evaluation_scope": "Five overlapping crops; one development geographic component",
        "independent_test": "NOT_EVALUATED; three reservations untouched",
        "ap_protocol": "Mean per-crop 101-point AP, score floor .25; not pooled COCO AP",
        "geographic_fp_protocol": "Same-class world-mask IoU>=.5 connected clusters; proxy only",
        "training_steps": len(logs),
        "numerical_recovery": json.loads((run / "training_config.json").read_text()).get(
            "numerical_recovery"
        ),
        "discarded_finite_updates_before_recovery": 45,
        "training_loss_first_50_mean": float(np.mean([x["loss"] for x in logs[:50]])),
        "training_loss_last_50_mean": float(np.mean([x["loss"] for x in logs[-50:]])),
        "peak_allocated_bytes": max(x["peak_allocated_bytes"] for x in logs),
        "training_step_seconds_total": sum(x["seconds"] for x in logs),
        "dataset_manifest_sha256": sha256(dataset / "metadata/dataset_manifest.json"),
        "dataset_statistics": manifest["statistics"],
        "new_regions_unique_observed_grid_pixel_counts": unique_counts,
        "approval": manifest["approval"],
        "approval_sha256": manifest["approval_sha256"],
    }
    dump(report / "recognition_evaluation.json", evidence)
    dump(report / "selected_postprocessing.json", best["config"])
    dump(
        report / "selected_inference.json",
        {
            "checkpoint": str(selected_path),
            "model": str(selected_path / "model"),
            "processor": str(selected_path / "processor"),
            "postprocessing_override": str(report / "selected_postprocessing.json"),
            "weight_sha256": bundle["weight_sha256"],
            "frozen_evaluation_postprocessing": str(run / "training_config.json"),
            "production_ready": False,
        },
    )
    (report / "model_card.md").write_text(
        "# Phase 5.5 selected development model\n\n"
        "Architecture: facebook/mask2former-swin-small-coco-instance, fixed Swin-S.\n\n"
        f"Weight bundle: `{selected_path}`; SHA-256 `{bundle['weight_sha256']}`.\n\n"
        f"Selected retained updates after the Phase 5.4 warm start: {step}. "
        "The warm start inherits 188 Phase 5.3 updates.\n\n"
        "Use selected_inference.json with selected_postprocessing.json. The training "
        "config preserves the frozen training evaluation settings.\n\n"
        "Input: RGB-replicated normalized LiDAR intensity, native 0.05 coordinate units, "
        "512 pixels, processor normalization, do_resize=False. Labels: 0 solid, 1 dash; "
        "no-object 2. Unsupported arrows/symbols and unobserved pixels excluded.\n\n"
        "Development engineering only. One validation geographic component, five "
        "overlapping crops, no independent test. Not production ready. No 3D vectorization.\n",
        encoding="utf-8",
    )
    dump(
        report / "code_hashes.json",
        {
            str(p.relative_to(root)): sha256(p)
            for p in [
                *root.glob("scripts/phase5_5*.py"),
                root / "src/roadmarking/dataset/phase55.py",
                root / "src/roadmarking/training/continuous.py",
                root / "src/roadmarking/training/valid_criterion.py",
            ]
        },
    )
    panels((baseline_records, selected_records), before, after, report)
    lines = [
        "# Phase 5.5 — approved dataset expansion and bounded recognition experiment",
        "",
        "Decision: development engineering model; NOT production ready.",
        "",
        f"Selected bundle: `{selected_path}`. Selected new optimizer step: {step}.",
        f"Actual 600-step experiment bundle: `{run / 'checkpoint_step0600'}`.",
        "All previous checkpoints, annotations, LiDAR and approvals were preserved.",
        "",
        "## Approval and supervised data",
        "",
        f"Seven-region approval: `{manifest['approval']}`.",
        f"SHA-256: `{manifest['approval_sha256']}`.",
        "The exclusive-created read-only attestation binds exact region IDs, project, "
        "62 reviewed artifact hashes, per-feature geometry hashes, source IDs/widths and "
        "the prior six-region approval. Read-only plus hash checking detects changes; "
        "it does not provide external WORM storage.",
        "Approval covers visible observations within the exact reviewed footprints only. "
        "All 95 prior outlines and individual dash/source identities are retained.",
        "125 reviewed IDs yield 65 supported IDs with observed paint pixels; 63 retained "
        "in native 512 crops, including six identities already represented in training.",
        "Unique training identities: 20 → 77 (10 solid, 67 dash); validation: 3 solid, "
        "22 dash. Training crops: 214; validation crops: 5; hard-negative training crops: 16. "
        "Overlapping crops are appearances, not independent geographic examples.",
        f"Unique new-region native-grid pixels: `{unique_counts}`.",
        "Pixel-state appearance counts (overlaps retained): "
        f"`{manifest['statistics']['new_pixel_state_appearances']}`. "
        "0=NoData, 1=unknown/unsupported, 2=verified background, 3=paint, "
        "4=boundary, 5=conflict. NoData, padding, ambiguous boundaries, conflicts and "
        "unsupported symbols never become supervised solid/dash background.",
        "",
        "## Geography",
        "",
        "Full source IDs and footprint guards produce two components. Eleven regions "
        "share the training component; the two interchange windows form the validation "
        "component. The seven new regions inherit training through measured linkage, "
        "rather than automatic assignment. No shared cross-fold IDs or guard violations.",
        "Unique full-region footprint area: train 27,525.12; validation 4,587.52 native "
        "square units. These areas include unobserved pixels; observed pixel counts above "
        "describe actual support. Sampling stays 0.05 native coordinate units.",
        "Three future-test reservations are unchanged, unused and unevaluated. "
        "Only one development validation component exists; no independent test claim.",
        "",
        "## Training",
        "",
        "Fixed official Mask2Former Swin-S architecture; warm start from Phase 5.4 "
        "selected weights. Native 512 pixels, do_resize=False, processor normalization "
        "preserved, class 0 solid/1 dash. AdamW lr 5e-6, cosine decay to 1e-6 over "
        "600 steps, batch 1, mixed precision, clip 1, seed 895; five positive samples "
        "per verified-negative sample. Valid-only Hungarian BCE/Dice and auxiliary "
        "losses reused. Unmatched mask loss sees only valid pixels; global unmatched "
        "no-object CE remains disabled on partially observed images.",
        "600 retained bounded updates with an actual full-state resume at 300. "
        "The first resumed segment stopped before update 346 on non-finite FP16 gradients. "
        "Its 45 post-checkpoint finite updates were discarded, and failure logs preserved. "
        "One numerical recovery restored step 300, lowered loss scale 128 to 16, "
        "and kept finite-loss/gradient guards enabled. 645 successful updates were executed "
        "across all attempts; the final retained trajectory contains 600. "
        "Snapshots every 100 steps save model, processor, AdamW, scheduler, scaler, "
        "RNG, dataset/config hashes. No infinite parameter search.",
        f"Peak allocated GPU memory: {evidence['peak_allocated_bytes']/2**30:.3f} GiB. "
        f"Summed optimizer-step time: {evidence['training_step_seconds_total']:.1f}s.",
        f"Mean sampled loss first/last 50: {evidence['training_loss_first_50_mean']:.4f} / "
        f"{evidence['training_loss_last_50_mean']:.4f}; "
        "stochastic class mixtures limit interpretation.",
        "",
        "## Comparable development recognition",
        "",
        "| Measure | Phase 5.4 frozen | Phase 5.5 selected + development calibration |",
        "|---|---:|---:|",
    ]
    for label, name in (("0", "Solid"), ("1", "Dash")):
        for key in ("precision", "recall", "f1"):
            lines.append(
                f"| {name} {key} | {baseline['counts'][label][key]:.4f} "
                f"| {result['counts'][label][key]:.4f} |"
            )
    for key, title in (
        ("fp", "False-positive appearances"),
        ("fn", "Missed appearances"),
        ("duplicates", "Duplicate appearances"),
    ):
        values = [sum(x["counts"][str(i)][key] for i in (0, 1)) for x in (baseline, result)]
        lines.append(f"| {title} | {values[0]} | {values[1]} |")
    lines.extend(
        [
            f"| Macro F1 | {baseline['counts']['macro_f1']:.4f} | "
            f"{result['counts']['macro_f1']:.4f} |",
            f"| Mean foreground IoU | {baseline['mean_foreground_iou']:.4f} | "
            f"{result['mean_foreground_iou']:.4f} |",
            "",
            "Full per-class AP50/AP75, mask IoU, inference timings, geographic FP-cluster "
            "proxy, missed source IDs, raw/frozen/retuned results for every checkpoint and "
            "the equally retuned baseline are in recognition_evaluation.json. AP is mean "
            "per-crop 101-point AP above score floor .25, not pooled COCO AP. Geographic FP "
            "clusters use same-class world-mask IoU>=.5 and are a proxy, not independently "
            "verified unique error objects. Duplicates remain false positives.",
            "Calibration is limited to 300 fixed settings per saved checkpoint; requires "
            "at least the original 3 solid and 17 dash true-positive appearances. "
            "Frozen-threshold comparisons distinguish model-weight changes from calibration. "
            "Selection on this same small development set makes these optimistic development "
            "results, not independent generalization estimates.",
            "comparison_0/2/4.png show actual intensity, approved masks, baseline/selected "
            "predictions, false positives and misses. No synthetic recognition examples.",
            "",
            "## Validation and limitations",
            "",
            "17 pretraining supervision/resume tests passed. Full regression results, "
            "Ruff, Black, healthcheck and fresh selected-model reload evidence accompany "
            "this report. Consult verification.json for final check outcomes.",
            "No final 3D vectorization was started. Sparse paint observations, only two "
            "geographic components and residual recognition errors prevent production approval.",
        ]
    )
    (report / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    context.update(selected_checkpoint=str(selected_path), report=str(report / "report.md"))
    dump(context_path, context)
    print(json.dumps({"selected_step": step, "baseline": baseline, "selected": result}, indent=2))


if __name__ == "__main__":
    main()
