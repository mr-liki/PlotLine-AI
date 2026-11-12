"""Phase 5.13: apply predeclared acceptance criteria, render edge before/after, write report."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
import numpy as np

from phase5_11_common import ROOT, load
from phase5_11_evaluate import FROZEN, components
from roadmarking.inference.prediction_recovery import filter_geographic
from roadmarking.rasterization.pipeline import dump, sha256

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOST = "4b28eb52fa431a19bf87cd0a9314c37a7967d31fa3ed5f4e49f7454bfdec084b"
STEPS = [100, 200, 300, 400, 500, 600]


def edge_coverage(comps):
    comp = comps[0]
    sel = filter_geographic(
        comp["scores"],
        comp["classes"],
        comp["qualities"],
        comp["overlaps"],
        comp["crop_ids"],
        FROZEN,
    )
    solid = comp["masks"][sel][comp["classes"][sel] == 0]
    union = solid.any(0) if len(solid) else np.zeros(comp["shape"], bool)
    j = comp["ids"].index(LOST)
    return float((union & comp["targets"][j]).sum() / comp["targets"][j].sum()), union, comp


def main():
    out = ROOT / "outputs/reports/phase5_13"
    latest = load(out / "latest.json")
    run = Path(latest["training_run"])
    protocol = load(out / "training_protocol.json")
    crit = protocol["selection"]
    ev = load(out / "checkpoint_evaluation.json")["evaluations"]
    rows, unions = {}, {}
    for step in STEPS:
        e = ev[str(step)]
        cov, union, _legacy_comp = edge_coverage(
            components(
                run / f"validation_step{step:04d}", out / f"evaluation/step{step:04d}_phase58"
            )
        )
        unions[step] = union
        fp = sum(e[a]["filtered"]["per_class"][c]["fp"] for a in e for c in ("solid", "dash"))
        checks = {
            "C1_lost_edge": cov >= crit["C1_lost_edge_coverage_min"],
            "C2_legacy_fg": e["LEGACY_POSITIVE_DEV"]["filtered"]["foreground_iou"]
            >= crit["C2_legacy_foreground_iou_min"],
            "C3_new_fg": e["NEW_POSITIVE_DEV"]["filtered"]["foreground_iou"]
            >= crit["C3_new_foreground_iou_min"],
            "C3b_new_f1": e["NEW_POSITIVE_DEV"]["filtered"]["macro_f1"]
            > crit["C3b_new_macro_f1_gt"],
            "C4_fp": fp <= crit["C4_total_fp_max"],
            "C5_negative": sum(
                e["SPARSE_NEGATIVE_CONTROL"]["filtered"]["per_class"][c]["fp"]
                for c in ("solid", "dash")
            )
            <= crit["C5_sparse_negative_fp_max"],
            "C5b_multi_dash": e["NEW_POSITIVE_DEV"]["filtered"]["dash_separation"][
                "potential_multi_dash_prediction_masks"
            ]
            <= crit["C5b_new_multi_dash_masks_max"],
        }
        rows[step] = {
            "lost_edge_coverage": cov,
            "total_fp": fp,
            "checks": checks,
            "passes": all(checks.values()),
            "legacy_fg": e["LEGACY_POSITIVE_DEV"]["filtered"]["foreground_iou"],
            "new_fg": e["NEW_POSITIVE_DEV"]["filtered"]["foreground_iou"],
            "legacy_f1": e["LEGACY_POSITIVE_DEV"]["filtered"]["macro_f1"],
            "new_f1": e["NEW_POSITIVE_DEV"]["filtered"]["macro_f1"],
        }
    passing = [s for s in STEPS if rows[s]["passes"]]
    selected = (
        sorted(passing, key=lambda s: (-rows[s]["new_fg"], -rows[s]["legacy_fg"], s))[0]
        if passing
        else None
    )
    decision = (
        "CANDIDATE_MEETS_PREDECLARED_CRITERIA_PROVISIONAL"
        if selected
        else "RETAIN_DEFAULT_NO_CANDIDATE_MEETS_CRITERIA"
    )
    # Before/after image of the long road-edge line (default, 5.12 s400, 5.13 best/selected).
    show = selected or max(STEPS, key=lambda s: rows[s]["lost_edge_coverage"])
    p59 = load(ROOT / "outputs/reports/phase5_9/latest.json")
    l12 = load(ROOT / "outputs/reports/phase5_12/latest.json")
    panels = {
        "default step 50": components(
            ROOT
            / "models/mask2former_swin_s/phase5_7_expanded_20261106T111240383727Z"
            / "validation_step0050",
            Path(p59["report_dir"]) / "inference",
        ),
        "5.12 step 400": components(
            Path(l12["training_run"]) / "validation_step0400",
            Path(l12["report_dir"]) / "evaluation/step0400_phase58",
        ),
    }
    fig, axes = plt.subplots(3, 1, figsize=(16, 13))
    items = [(k, *edge_coverage(v)) for k, v in panels.items()]
    items.append(
        (
            f"5.13 step {show}" + (" (selected)" if selected else " (best edge, not " "accepted)"),
            rows[show]["lost_edge_coverage"],
            unions[show],
            items[0][3],
        )
    )
    comp = items[0][3]
    j = comp["ids"].index(LOST)
    target = comp["targets"][j]
    ys, xs = np.nonzero(target)
    y0, y1, x0, x1 = max(0, ys.min() - 30), ys.max() + 30, max(0, xs.min() - 30), xs.max() + 30
    for ax, (label, cov, union, _) in zip(axes, items, strict=True):
        img = np.repeat((comp["image"][y0:y1, x0:x1] / 255.0)[..., None], 3, -1)
        t, u = target[y0:y1, x0:x1], union[y0:y1, x0:x1]
        img[t & u] = (0.1, 0.9, 0.2)
        img[t & ~u] = (1, 0, 1)
        img[u & ~t] = (1, 0.6, 0)
        ax.imshow(img)
        ax.set_title(
            f"{label}: lost-edge paint covered {cov:.0%} (green covered, magenta "
            "missed, orange solid mask off-paint)",
            fontsize=10,
        )
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "lost_edge_before_after.png", dpi=90)
    plt.close(fig)

    trace = load(out / "stage1_lost_edge_trace.json")
    audit = load(out / "stage2_label_audit.json")
    exposure = load(out / "stage3_training_exposure.json")
    probe = load(out / "stage4_training_edge_calibration_probe.json")["results"]
    sens = load(out / "stage4_threshold_sensitivity.json")
    validation = {}
    xml = ROOT / "outputs/reports/phase5_13_pytest.xml"
    if xml.exists():
        s = ET.parse(xml).getroot()
        s = s.find("testsuite") if s.tag == "testsuites" else s
        validation["pytest"] = {k: s.attrib[k] for k in ("tests", "failures", "errors", "skipped")}
        for n in ("phase5_13_ruff.log", "phase5_13_black.log", "phase5_13_healthcheck.json"):
            if (ROOT / "outputs/reports" / n).exists():
                validation[n] = sha256(ROOT / "outputs/reports" / n)
    hyp = {
        "A_class_confidence_suppression": "PARTIAL: 5.11 solid prob fell to .54/.76 on 2 of 3 "
        "crops (solid/dash confusion); 5.12 class prob unchanged (.95-.99). On training edge "
        "lines, candidate class probs are not lower than default.",
        "B_mask_quality_degradation": "PARTIAL (5.12): location retained (raw crop IoU .74-.81 "
        "except one .54), but mask quality fell .87-.90 -> .82-.84 and centreline fg-prob p10 "
        ".88 -> .27; enough to cross .85.",
        "C_background_confusion": "NOT SUPPORTED: edge-line neighbourhood statistics of new "
        "crops (ignored, background, bright-background fractions at 1-4 px) match original "
        "training crops; no footprint pixel labelled background.",
        "D_sampling_imbalance": "Exposure shift is real (solid px/step -14%, solid:dash pixel "
        "ratio 3.8 -> 2.5), but restoring solid exposure did NOT fix it: in the predeclared "
        "rebalance run, lost-edge coverage stayed at "
        + "/".join(f"{rows[s]['lost_edge_coverage']:.2f}" for s in STEPS)
        + " (steps 100-600) and legacy FG IoU fell further. Not the cause.",
        "E_catastrophic_forgetting": "NOT SUPPORTED: on ORIGINAL training edge lines the 5.11 "
        "candidate is more confident than default (.860 vs .825) and 5.12 is similar (.811).",
        "F_annotation_or_conversion_inconsistency": "REJECTED: 100% of valid footprint "
        "supervised, 0 footprint pixels as background, 0 cross-crop conflicts in 17M "
        "overlapping pixels, class IDs consistent with client designations.",
        "G_threshold_margin_and_selection_bias (found)": "SUPPORTED: even the default scores a "
        "median .825 on its own training edge lines (56% below .85). The lost edge falls from "
        ".855-.895 to .818. At solid .80 (diagnostic only), 5.12 legacy FG IoU returns to .592 "
        "(default .588) and edge coverage to .57. The default checkpoint and .85 threshold were "
        "both selected on this legacy development set, which biases comparisons toward the "
        "default.",
    }
    report = {
        "phase": "5.13",
        "decision": decision,
        "selected_step": selected,
        "lost_marking": trace["identity"],
        "raw_trace": trace["models"],
        "label_audit": {
            k: audit[k]
            for k in (
                "footprint_vs_target_contradictions_new_crops",
                "cross_tile_consistency_new_crops",
                "class_ids_consistent_with_client_designation",
            )
        },
        "kerb_background_proximity": audit["neighbourhood_by_group_class_edge"],
        "training_exposure": exposure,
        "training_edge_calibration_probe": probe,
        "threshold_sensitivity_diagnostic": sens["results"],
        "checkpoint_variability": sens["checkpoint_variability"],
        "hypotheses": hyp,
        "protocol": protocol,
        "candidates": rows,
        "default_model_unchanged": selected is None,
        "protected_final_test_used": False,
        "validation": validation,
    }
    dump(out / "report.json", report)
    md = [
        "# Phase 5.13: road-edge regression diagnosis and corrective experiment",
        "",
        f"**Decision: {decision}.** "
        + (
            f"Step {selected} of the predeclared rebalance run meets every acceptance "
            "criterion. It is a provisional development candidate; the default is not "
            "replaced automatically and the protected final test was not read."
            if selected
            else "No candidate of the single predeclared corrective run met all acceptance "
            "criteria, so the original step-50 model remains the default. The protected "
            "final test was not read."
        ),
        "",
        "## 1. The lost marking",
        "",
        f"- `{LOST}`: client fid 34908, **Guia direita** (right road edge), 0.15 m wide, "
        f"{trace['identity']['clipped_centerline_length_native']:.1f} m inside the legacy "
        f"component, with {trace['identity']['observed_target_pixels']} observed pixels "
        "across 3 overlapping crops.",
        "- **It disappears after the raw model, at filtering.** Every model still produces a "
        "solid-class mask over it in every crop. Confidence is solid probability × mask "
        "quality, and the frozen solid threshold is .85:",
        "",
        "| Model | Crop | P(solid/dash/none) | mask quality | confidence | raw IoU in crop "
        "| fate |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for m, r in trace["models"].items():
        for p in r["per_crop"]:
            md.append(
                f"| {m} | {p['crop'][-22:]} | "
                + "/".join(f"{x:.3f}" for x in p["class_probs_solid_dash_noobj"])
                + f" | {p['mask_quality']:.3f} | {p['confidence_class_x_quality']:.3f} | "
                f"{p['raw_mask_iou_vs_paint_in_crop']:.3f} | {p['fate']} |"
            )
    md += [
        "",
        "| Model | final coverage of paint | centreline fg-prob mean / p10 | final best "
        "single-mask IoU (complete line) |",
        "|---|---:|---:|---:|",
    ]
    for m, r in trace["models"].items():
        g = r["max_solid_query_fg_prob_along_centreline"]
        md.append(
            f"| {m} | {r['final_solid_union_coverage_of_paint']:.2f} | {g['mean']:.2f} / "
            f"{g['p10']:.2f} | {r['final_best_single_mask_iou_complete_identity']:.3f} |"
        )
    md += [
        "",
        "Diagnostic image: `stage1_lost_edge_side_by_side.png`. Cross-tile: default "
        "kept the line in one crop, and the overlapping duplicates were correctly removed "
        "by NMS. In both candidates two or three crops fall just under the threshold.",
        "",
        "## 2. Training-label audit (Phase 5.11 + 5.12 crops)",
        "",
        f"- Footprint vs target: `{json.dumps(report['label_audit'])}`.",
        "- Neighbourhood statistics around solid edge lines (signed px from the target): "
        "see `stage2_label_audit.json`. At 1 px outside the paint, 87% (new) vs 89% "
        "(original) of pixels are ignored boundary. Bright-background share is .46–.48 "
        "(new) vs .48 (original) at 1 px, and .16–.18 vs .20 at 2 px. The new labels do "
        "not add more paint-adjacent bright background than the original training labels.",
        "- No converter defect, so no converter change. No annotation correction package is "
        "needed.",
        "",
        "## 3. Training exposure (actual sampled steps)",
        "",
        "| Run | solid-pos steps | "
        "dash-pos steps | negative steps | original-region steps | solid px | dash px | "
        "solid:dash px | edge px |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, t in exposure.items():
        md.append(
            f"| {name} | {t['solid_positive_steps']} | {t['dash_positive_steps']} | "
            f"{t['hard_negative_steps']} | {t.get('steps_original', 0)} | {t['solid_px']} | "
            f"{t['dash_px']} | {t['solid_to_dash_pixel_ratio']:.2f} | {t['edge_px']} |"
        )
    md += (
        ["", "## 4. Hypotheses", ""]
        + [f"- **{k}**: {v}" for k, v in hyp.items()]
        + [
            "",
            "Training-crop edge calibration probe (training data only):",
            "",
            "| Model | original median conf | original ≥.85 | new median conf | new ≥.85 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for m, r in probe.items():
        md.append(
            f"| {m} | {r['original']['median_confidence']:.3f} | "
            f"{r['original']['fraction_conf_ge_0.85']:.2f} | "
            f"{r['new_5_11_5_12']['median_confidence']:.3f} | "
            f"{r['new_5_11_5_12']['fraction_conf_ge_0.85']:.2f} |"
        )
    md += [
        "",
        "Threshold-margin sensitivity (saved predictions; diagnostic only, not selection):",
        "",
        "| Model | setting | legacy FG IoU | lost-edge coverage | legacy solid FP | new FG "
        "IoU | new solid TP/FP |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for m, r in sens["results"].items():
        for lab, e in r.items():
            L, N = e["LEGACY_POSITIVE_DEV"], e["NEW_POSITIVE_DEV"]  # noqa: N806
            md.append(
                f"| {m} | {lab} | {L['foreground_iou']:.3f} | "
                f"{L['lost_edge_paint_coverage']:.2f} | {L['per_class']['solid']['fp']} | "
                f"{N['foreground_iou']:.3f} | {N['per_class']['solid']['tp']}/"
                f"{N['per_class']['solid']['fp']} |"
            )
    md += [
        "",
        "## 5. Predeclared experiment",
        "",
        f"- Single change: sampling cycle `{protocol['single_controlled_change']}`. "
        "Data, initialization, budget, optimizer and criterion are identical to Phase 5.12. "
        "There was no teacher regularization and no synthetic data.",
        f"- Acceptance criteria, fixed from measured checkpoint variability before scoring: "
        f"`{json.dumps({k: v for k, v in crit.items() if k.startswith('C')})}`.",
        "",
        "## 6. Results (frozen filtering)",
        "",
        "| Step | lost-edge cov | legacy FG IoU | new FG IoU | legacy F1 | new F1 | total FP "
        "| failed checks |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for s in STEPS:
        r = rows[s]
        md.append(
            f"| {s} | {r['lost_edge_coverage']:.2f} | {r['legacy_fg']:.3f} | "
            f"{r['new_fg']:.3f} | {r['legacy_f1']:.3f} | {r['new_f1']:.3f} | "
            f"{r['total_fp']} | {', '.join(k for k, v in r['checks'].items() if not v) or '-'} |"
        )
    md += [
        "",
        "Before/after image of the road-edge line: `lost_edge_before_after.png`. Full "
        "per-checkpoint metrics (P/R/F1, AP50/75, fragmentation, cross-tile, boundary) are "
        "in `checkpoint_evaluation.json`.",
        "",
        "## 7. Limitations",
        "",
        "- Both development areas are small. New-area macro F1 varies by about ±.05 between "
        "adjacent checkpoints, so single-step comparisons are noisy.",
        "- The default model and the .85 threshold were selected on the legacy development "
        "set, which favours the default in legacy comparisons.",
        "- Thin edge lines sit at the confidence threshold for every model. A calibration- or "
        "threshold-robust evaluation, predeclared on development data, is the natural next "
        "step. Any threshold change must be validated without the protected test.",
        "",
        f"Validation: `{validation}`",
        "",
    ]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": decision,
                "selected": selected,
                "rows": {
                    s: {k: v for k, v in r.items() if k != "checks"}
                    | {"failed": [k for k, v in r["checks"].items() if not v]}
                    for s, r in rows.items()
                },
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
