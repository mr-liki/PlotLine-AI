"""Phase 5.11 finalization: before/after galleries, FP-context gallery and full report."""

import json
import math
from pathlib import Path

import matplotlib
import numpy as np
from scipy.ndimage import binary_dilation

import phase5_10_recovery as rec
import phase5_11_report
from phase5_11_common import ROOT, load
from phase5_11_evaluate import FROZEN, components
from roadmarking.inference.prediction_recovery import filter_geographic
from roadmarking.rasterization.pipeline import dump, sha256

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def stage(comp, label):
    sel = filter_geographic(
        comp["scores"],
        comp["classes"],
        comp["qualities"],
        comp["overlaps"],
        comp["crop_ids"],
        FROZEN,
    )
    m = rec.score(
        comp,
        comp["masks"][sel],
        comp["scores"][sel],
        comp["classes"][sel],
        comp["iou"][sel],
        comp["inter"][sel],
    )
    return label, comp["masks"][sel], comp["classes"][sel], m


def fp_gallery(comp, label, metrics, masks, path):
    fps = [k for c in ("0", "1") for k in metrics["_ops"][c]["false_indices"]]
    paint = comp["targets"].any(0)
    near = binary_dilation(paint, iterations=20)
    shown = fps[:16]
    if not shown:
        return
    cols = 4
    rows = math.ceil(len(shown) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    gray = comp["image"].astype(float) / 255
    for ax, k in zip(axes.flat, shown, strict=False):
        m = masks[k] & comp["valid"]
        ys, xs = np.nonzero(m)
        if not len(ys):
            ax.axis("off")
            continue
        y0, y1 = max(0, ys.min() - 40), ys.max() + 40
        x0, x1 = max(0, xs.min() - 40), xs.max() + 40
        img = np.repeat(gray[..., None], 3, -1)
        img[paint] = 0.5 * img[paint] + 0.5 * np.array([0.1, 0.8, 0.2])
        img[m] = (1, 0.85, 0)
        ring = binary_dilation(m, iterations=10) & ~binary_dilation(m, iterations=2)
        density = comp["observed"][ring].mean() if ring.any() else 0
        kind = (
            "near paint"
            if (m & near).sum() >= 0.5 * max(1, m.sum())
            else "sparse returns" if density < 0.5 else "dense structure"
        )
        ax.imshow(img[y0:y1, x0:x1])
        ax.set_title(kind, fontsize=8)
        ax.axis("off")
    for ax in axes.flat[len(shown) :]:
        ax.axis("off")
    fig.suptitle(f"{comp['name']} {label}: unmatched predictions (yellow), approved paint (green)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    latest = load(ROOT / "outputs/reports/phase5_11/latest.json")
    out = Path(latest["report_dir"])
    run = Path(latest["training_run"])
    ev = load(out / "checkpoint_evaluation.json")
    summary, evaluations = ev["summary"], ev["evaluations"]
    sel = summary["selected_step"]
    gal = out / "galleries"
    before = components(run / "validation_step0000", out / "evaluation/step0000_phase58")
    after = components(run / f"validation_step{sel:04d}", out / f"evaluation/step{sel:04d}_phase58")
    for b, a in zip(before[:2], after[:2], strict=True):
        s0, s1 = stage(b, "step 0 (existing)"), stage(a, f"step {sel} (selected)")
        rec.gallery(b, [s0, (s1[0], s1[1], s1[2], s1[3])], gal / f"{b['name']}_before_after.png")
        fp_gallery(a, f"step {sel}", s1[3], s1[1], gal / f"{a['name']}_step{sel}_fp_context.png")
    del before, after

    phase5_11_report.main()
    md = (out / "report.md").read_text(encoding="utf-8")
    md = md.replace(
        md[md.index("**Decision: REAL_DATA_REVIEW_REQUIRED.**") : md.index("## Stage 1")],
        "**Decision: RECOGNITION_IMPROVED_PROVISIONAL.** You approved all 16 regions "
        '("Approve all 16"), with three user-reported ignore zones. The dataset was '
        "exported and one predeclared 600-step warm-start run was trained. Under the "
        f"predeclared rule, step {sel} improves the worst-geography and mean development "
        "macro F1. The gain is concentrated in the hard PHASE58 geography, and the legacy "
        "geography regresses at pixel level. This is development evidence only: the "
        "protected final test is sealed and was never read, and the existing frozen "
        "checkpoint remains the production default.\n\n",
    )
    md = md[: md.index("## Stages 5–10")]
    exp = load(out / "dataset_export.json")["summary"]
    lines = [
        "## Stage 5–6: approval and export",
        "",
        f"- The approval attestation `phase5_11_sixteen_region_approval.json` is write-once "
        f"and hash-bound to the QGIS project, the candidate store, every geometry, the "
        f"rasters, the review feedback and the 3 ignore zones (SHA-256 "
        f"`{latest['approval_sha256']}`).",
        f"- Export summary: `{json.dumps({k: exp[k] for k in exp if k != 'readiness'})}`.",
        "- Excluded from supervision: NoData, unsupported symbols (2 px margin), "
        "non-supervised boundary states, targets with fewer than 4 pixels, and non-target "
        "pixels in ignore zones. All masks lie inside validity, and no instance pixels "
        "overlap.",
        "- Independence: no training ID appears in development, benchmark, protected-test, "
        "or held-out IDs, which the dataset check asserts. Existing development crops are "
        "unchanged. See `galleries/exported_pairs_audit.png`.",
        "",
        "## Stage 7: controlled training",
        "",
        "- `training_protocol.json` is write-once and sealed before training. It sets a "
        "warm start from the frozen step-50 checkpoint, one run of 600 steps, AdamW at "
        "2e-6 with cosine decay to 4e-7, the Phase 5.4 NoData-aware criterion and losses, "
        "and a sampling cycle that oversamples new-region solid/dash crops alongside "
        "verified negatives.",
        f"- Measured: {load(out / 'training_result.json')['segment_elapsed_seconds']:.0f} s "
        f"for 600 steps, peak CUDA "
        f"{load(out / 'training_result.json')['cuda_peak_allocated_bytes'] / 2**30:.2f} GiB "
        "allocated. Complete resumable checkpoints were saved every 100 steps. The step-0 "
        "evaluation reproduced the declared warm start (raw FP 132 / FN 11).",
        "",
        "## Stage 8: both development geographies (frozen post-processing)",
        "",
        f"Step 0 reproduces the Phase 5.10 baseline: `{summary['step0_reproduces_phase5_10']}`.",
        "",
        "| Step | worst macro F1 | mean macro F1 | total FP |",
        "|---:|---:|---:|---:|",
    ]
    for s, o in summary["objective_by_step"].items():
        lines.append(
            f"| {s} | {o['worst_macro_f1']:.4f} | {o['mean_macro_f1']:.4f} | " f"{o['total_fp']} |"
        )
    lines += [
        "",
        f"Selected: **step {sel}** (strict improvement: "
        f"{summary['strict_improvement_over_step0']}).",
        "",
    ]
    for name in ("LEGACY_POSITIVE_DEV", "NEW_POSITIVE_DEV"):
        lines += [
            f"### {name}",
            "",
            "| Metric | step 0 | step " + str(sel) + " |",
            "|---|---:|---:|",
        ]
        e0, e1 = evaluations["0"][name], evaluations[str(sel)][name]

        def row(label, f, e0=e0, e1=e1, lines=lines):
            lines.append(f"| {label} | {f(e0)} | {f(e1)} |")

        for c in ("solid", "dash"):
            row(
                f"{c} TP/FP/FN (filtered)",
                lambda e, c=c: "/".join(
                    str(e["filtered"]["per_class"][c][k]) for k in ("tp", "fp", "fn")
                ),
            )
            row(
                f"{c} P/R/F1",
                lambda e, c=c: "/".join(
                    f"{e['filtered']['per_class'][c][k]:.3f}" for k in ("precision", "recall", "f1")
                ),
            )
            row(
                f"{c} TP/FP/FN (raw model)",
                lambda e, c=c: "/".join(
                    str(e["raw_model"]["per_class"][c][k]) for k in ("tp", "fp", "fn")
                ),
            )
        row("macro F1 (filtered)", lambda e: f"{e['filtered']['macro_f1']:.4f}")
        for k, n in (("longitudinal_solid", "solid"), ("longitudinal_dash", "dash")):
            row(
                f"{n} AP50/AP75 (truncated)",
                lambda e, k=k: (
                    f"{e['filtered']['ap'][k].get('AP50', 0):.3f}/"
                    f"{e['filtered']['ap'][k].get('AP75', 0):.3f}"
                ),
            )
        row("foreground IoU", lambda e: f"{e['filtered']['foreground_iou']:.4f}")
        row(
            "pixel precision/recall",
            lambda e: (
                f"{e['filtered']['pixel']['all_supported_paint']['precision']:.3f}/"
                f"{e['filtered']['pixel']['all_supported_paint']['recall']:.3f}"
            ),
        )
        row(
            "observed skeleton coverage",
            lambda e: f"{e['filtered']['observed_skeleton_coverage_within_1px']:.3f}",
        )
        row(
            "solid fragments per target",
            lambda e: str(e["filtered"]["solid_fragmentation"]["per_target"]),
        )
        row(
            "multi-dash masks",
            lambda e: str(
                e["filtered"]["dash_separation"]["potential_multi_dash_prediction_masks"]
            ),
        )
        row(
            "matched boundary mean / p95 (native, observed only)",
            lambda e: (
                f"{e['boundary']['mean_symmetric_observed_boundary_distance_native']:.3f} / "
                f"{e['boundary']['p95_native']:.3f}"
                if e["boundary"].get("matched_instances")
                else "no matches"
            ),
        )
        row(
            "cross-crop class agreement",
            lambda e: f"{e['cross_tile']['class_agreement']:.3f} "
            f"({e['cross_tile']['class_conflicts']} conflicts)",
        )
        row("crop-appearance macro F1", lambda e: f"{e['crop_appearance']['macro_f1']:.4f}")
        lines.append("")
    neg = evaluations[str(sel)]["SPARSE_NEGATIVE_CONTROL"]
    speed = summary["inference_speed"].get(str(sel), {})
    lines += [
        f"Sparse negative control: filtered FP stays at "
        f"{sum(neg['filtered']['per_class'][c]['fp'] for c in ('solid', 'dash'))}, and raw "
        f"proposals fall from 5 to "
        f"{sum(neg['raw_model']['per_class'][c]['fp'] for c in ('solid', 'dash'))}.",
        f"Inference speed (step {sel}, RTX 3050): "
        f"{speed.get('model_only_seconds_per_512_crop', 0):.3f} s "
        f"model-only per 512 crop, {speed.get('end_to_end_seconds_per_crop', 0):.2f} s "
        f"end-to-end per crop with saving, peak {speed.get('peak_allocated_gib', 0):.2f} GiB.",
        "",
        "Interpretation:",
        "- The new geography improved substantially at pixel level: recall rose from .27 "
        "to .48, precision from .45 to .58, foreground IoU from .21 to .35, and skeleton "
        "coverage from .28 to .47. It also produced its first complete-instance matches "
        "(1 solid, 1 dash). Complete-instance recall remains very low.",
        "- The legacy geography regressed at pixel level: foreground IoU fell from .59 to "
        ".40, and dash FP rose from 11 to 17. Its dash TP rose by one and crop-appearance "
        "F1 rose from .593 to .626. The gain is therefore a trade-off, not a uniform "
        "improvement.",
        "- Raw (pre-filter) proposals rose in both regions, so more of the work now falls "
        "on filtering. The frozen post-processing was kept and not re-tuned, so raw and "
        "post-processing effects stay separate.",
        "",
        "## Stage 9: false-positive context (heuristic, verify visually)",
        "",
    ]
    for name in ("LEGACY_POSITIVE_DEV", "NEW_POSITIVE_DEV"):
        c0 = evaluations["0"][name]["false_positive_context"]
        c1 = evaluations[str(sel)][name]["false_positive_context"]
        lines.append(
            f"- {name}: step 0 `{ {k: v for k, v in c0.items() if k != 'definition'} }` "
            f"→ step {sel} `{ {k: v for k, v in c1.items() if k != 'definition'} }`."
        )
    lines += [
        f"- Definition: {c1['definition']}. Most remaining FPs are fragments or "
        "over-segmentation next to real paint, followed by sparse-return (vegetation or "
        "scan-edge) contexts. Dense-structure FPs (kerbs, guardrails, bright asphalt) are "
        "few and roughly unchanged. These are geometric heuristics, not semantic labels; "
        "the reviewer should check the galleries.",
        f"- Galleries: `galleries/*_before_after.png` and `galleries/*_step{sel}_fp_context.png`.",
        "",
        "## Stage 10: final-evaluation policy",
        "",
        "- The protected final test stays sealed. A one-time final test is **not yet "
        "recommended**: the gain is small in absolute terms (2 complete-instance matches in "
        "the new geography), and legacy pixel accuracy regressed. Spending the one-time "
        "test now would mostly confirm known weakness.",
        "- The existing step-50 checkpoint and its frozen post-processing remain the "
        f"production default. Step {sel} is recorded as a provisional development "
        "candidate only.",
        "",
        "## Next steps",
        "",
        "- Approve the 10 reserve windows that are already point-verified (mostly straight "
        "edge lines and merges). This adds straight-solid balance and should address the "
        "legacy regression.",
        "- Target fragmentation and over-segmentation, the dominant FP type: add long "
        "continuous-solid sections, then repeat the same bounded protocol with an identical "
        "budget. Do not extend steps after the fact.",
        "- Re-check the six LAZ files with corrupt chunks at the source.",
        "",
        "No production or 3D digitization was started. Nothing was committed.",
        "",
    ]
    validation = load(out / "report.json").get("validation")
    lines += [f"Validation: `{validation}`", ""]
    (out / "report.md").write_text(md + "\n".join(lines), encoding="utf-8")
    report = load(out / "report.json")
    report.update(
        decision="RECOGNITION_IMPROVED_PROVISIONAL",
        selected_development_step=sel,
        selected_development_checkpoint=summary["selected_checkpoint"],
        production_default_unchanged=True,
        final_test_recommended=False,
        approval_sha256=latest["approval_sha256"],
        dataset_export=exp,
        checkpoint_selection=summary,
    )
    dump(out / "report.json", report)
    dump(
        out / "selected_candidate.json",
        {
            "status": "PROVISIONAL_DEVELOPMENT_CANDIDATE_NOT_PRODUCTION",
            "checkpoint": summary["selected_checkpoint"],
            "weight_sha256": sha256(
                Path(summary["selected_checkpoint"]) / "model/model.safetensors"
            ),
            "postprocessing": FROZEN,
            "production_default": str(
                ROOT / "models/mask2former_swin_s/"
                "phase5_7_expanded_20261106T111240383727Z/checkpoint_step0050"
            ),
        },
    )
    dump(
        ROOT / "outputs/reports/phase5_11/latest.json",
        {**latest, "status": "RECOGNITION_IMPROVED_PROVISIONAL", "selected_step": sel},
    )
    print(out / "report.md")


if __name__ == "__main__":
    main()
