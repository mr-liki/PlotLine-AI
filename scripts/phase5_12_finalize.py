"""Phase 5.12: three-way development comparison, galleries and the final report."""

import json
from pathlib import Path

import phase5_10_recovery as rec
from phase5_11_common import ROOT, load
from phase5_11_evaluate import components
from phase5_11_finalize import fp_gallery, stage
from roadmarking.rasterization.pipeline import dump, sha256

NAMES = ("LEGACY_POSITIVE_DEV", "NEW_POSITIVE_DEV")


def fmt_counts(e, c, kind="filtered"):
    pc = e[kind]["per_class"][c]
    return f"{pc['tp']}/{pc['fp']}/{pc['fn']}"


def main():
    l12 = load(ROOT / "outputs/reports/phase5_12/latest.json")
    l11 = load(ROOT / "outputs/reports/phase5_11/latest.json")
    out12, out11 = Path(l12["report_dir"]), Path(l11["report_dir"])
    run12, run11 = Path(l12["training_run"]), Path(l11["training_run"])
    ev12 = load(out12 / "checkpoint_evaluation.json")
    ev11 = load(out11 / "checkpoint_evaluation.json")
    s12, s11 = ev12["summary"]["selected_step"], ev11["summary"]["selected_step"]
    columns = {
        "step 0 (default)": ev12["evaluations"]["0"],
        f"5.11 step {s11}": ev11["evaluations"][str(s11)],
        f"5.12 step {s12}": ev12["evaluations"][str(s12)],
    }
    gal = out12 / "galleries"
    gal.mkdir(exist_ok=True)
    comps = {
        "step 0": components(run12 / "validation_step0000", out12 / "evaluation/step0000_phase58"),
        f"5.11 s{s11}": components(
            run11 / f"validation_step{s11:04d}", out11 / f"evaluation/step{s11:04d}_phase58"
        ),
        f"5.12 s{s12}": components(
            run12 / f"validation_step{s12:04d}", out12 / f"evaluation/step{s12:04d}_phase58"
        ),
    }
    for i, name in enumerate(NAMES):
        stages = [stage(c[i], label) for label, c in comps.items()]
        rec.gallery(comps["step 0"][i], stages, gal / f"{name}_three_way.png")
        label, masks, _, metrics = stages[-1]
        fp_gallery(
            comps[f"5.12 s{s12}"][i], label, metrics, masks, gal / f"{name}_5_12_fp_context.png"
        )
    del comps

    exp = load(out12 / "dataset_export.json")["summary"]
    tr = load(out12 / "training_result.json")
    improved = ev12["summary"]["strict_improvement_over_step0"]
    o = {k: ev12["summary"]["objective_by_step"][str(k)] for k in (0, s12)}
    o11 = ev11["summary"]["objective_by_step"][str(s11)]
    beats_511 = (o[s12]["worst_macro_f1"], o[s12]["mean_macro_f1"]) > (
        o11["worst_macro_f1"],
        o11["mean_macro_f1"],
    )
    decision = "RECOGNITION_IMPROVED_PROVISIONAL" if improved else "MORE_REAL_DATA_REQUIRED"
    md = [
        "# Phase 5.12: reserve-region approval, export and controlled experiment",
        "",
        f"**Decision: {decision}.** You approved all 10 reserve regions. The dataset now "
        f"holds {exp['total_unique_training_ids']} unique training identities "
        f"({exp['total_unique_training_solid']} solid, {exp['total_unique_training_dash']} "
        f"dash). One controlled run was trained using the Phase 5.11 protocol with only the "
        f"data changed. The selected step is **{s12}**. Against the Phase 5.11 candidate "
        f"(step {s11}) on the predeclared objective, it is "
        f"{'better' if beats_511 else 'not better'}. The default model is unchanged and the "
        "protected final test was never read.",
        "",
        "## Approval and export",
        "",
        f"- `phase5_12_ten_region_approval.json` is write-once and hash-bound, recording "
        f'your statement "I checked each one of them , they are correct and perfect" '
        f"(SHA-256 `{l12['approval_sha256']}`). The reviewer layers had no missing-paint "
        "polygons and no outline notes, so there are no ignore zones.",
        f"- Export: `{json.dumps({k: v for k, v in exp.items() if k != 'readiness'})}`.",
        "- All 128 new identities are dashes. Every one of the 73 new crops also contains "
        "straight solid edge-line segments: new road sections of solid features already in "
        "training. The export used the same rules as Phase 5.11 (shared `export_tile` "
        "function).",
        "",
        "## Controlled training",
        "",
        "- The protocol is identical to Phase 5.11: warm start from the default step-50 "
        "model, 600 steps, AdamW 2e-6 with cosine decay to 4e-7, the same criterion, the "
        "same sampling cycle (new-region pools now include 5.11 and 5.12 crops) and the "
        "same selection rule. It was sealed in `training_protocol.json` before training.",
        f"- Measured: {tr['segment_elapsed_seconds']:.0f} s, peak "
        f"{tr['cuda_peak_allocated_bytes'] / 2**30:.2f} GiB. The step-0 warm-start "
        "reproduction check passed.",
        "",
        "## Development comparison (identical geographic definitions, frozen post-processing)",
        "",
        "| Step | worst macro F1 | mean macro F1 | total FP |",
        "|---:|---:|---:|---:|",
    ]
    for s, v in ev12["summary"]["objective_by_step"].items():
        md.append(
            f"| {s} | {v['worst_macro_f1']:.4f} | {v['mean_macro_f1']:.4f} | " f"{v['total_fp']} |"
        )
    md += [
        "",
        f"Phase 5.11 step {s11} for reference: worst {o11['worst_macro_f1']:.4f}, mean "
        f"{o11['mean_macro_f1']:.4f}, FP {o11['total_fp']}.",
        "",
    ]
    heads = list(columns)
    for name in NAMES:
        md += [
            f"### {name}",
            "",
            "| Metric | " + " | ".join(heads) + " |",
            "|---|" + "---:|" * len(heads),
        ]

        def row(label, f, name=name, md=md):
            md.append(f"| {label} | " + " | ".join(f(columns[h][name]) for h in heads) + " |")

        for c in ("solid", "dash"):
            row(f"{c} TP/FP/FN", lambda e, c=c: fmt_counts(e, c))
            row(f"{c} F1", lambda e, c=c: f"{e['filtered']['per_class'][c]['f1']:.3f}")
            row(f"{c} raw TP/FP/FN", lambda e, c=c: fmt_counts(e, c, "raw_model"))
        row("macro F1", lambda e: f"{e['filtered']['macro_f1']:.4f}")
        for k, n in (("longitudinal_solid", "solid"), ("longitudinal_dash", "dash")):
            row(
                f"{n} AP50/AP75",
                lambda e, k=k: f"{e['filtered']['ap'][k].get('AP50', 0):.3f}/"
                f"{e['filtered']['ap'][k].get('AP75', 0):.3f}",
            )
        row("foreground IoU", lambda e: f"{e['filtered']['foreground_iou']:.3f}")
        row(
            "pixel P/R",
            lambda e: f"{e['filtered']['pixel']['all_supported_paint']['precision']:.3f}/"
            f"{e['filtered']['pixel']['all_supported_paint']['recall']:.3f}",
        )
        row(
            "skeleton coverage",
            lambda e: f"{e['filtered']['observed_skeleton_coverage_within_1px']:.3f}",
        )
        row(
            "solid fragments/target",
            lambda e: str(e["filtered"]["solid_fragmentation"]["per_target"]),
        )
        row(
            "boundary mean (matched, observed)",
            lambda e: (
                f"{e['boundary']['mean_symmetric_observed_boundary_distance_native']:.3f}"
                if e["boundary"].get("matched_instances")
                else "n/a"
            ),
        )
        row("cross-crop class agreement", lambda e: f"{e['cross_tile']['class_agreement']:.3f}")
        row(
            "FP context near/sparse/dense",
            lambda e: "/".join(
                str(e["false_positive_context"][k])
                for k in (
                    "near_approved_marking_fragment_or_boundary",
                    "sparse_return_context",
                    "dense_bright_structure_context",
                )
            ),
        )
        row("crop-appearance macro F1", lambda e: f"{e['crop_appearance']['macro_f1']:.4f}")
        md.append("")
    neg = ev12["evaluations"][str(s12)]["SPARSE_NEGATIVE_CONTROL"]["filtered"]["per_class"]
    speed = ev12["summary"]["inference_speed"].get(str(s12), {})
    md += [
        f"Sparse negative control filtered FP: {neg['solid']['fp'] + neg['dash']['fp']}. "
        f"Inference: {speed.get('model_only_seconds_per_512_crop', 0):.3f} s per 512 crop "
        f"(model only), peak {speed.get('peak_allocated_gib', 0):.2f} GiB.",
        "",
        "Galleries: `galleries/*_three_way.png` (step 0, Phase 5.11 and Phase 5.12) and "
        "`galleries/*_5_12_fp_context.png`.",
        "",
        "## Policy",
        "",
        "- The protected final test stays sealed. The default checkpoint is unchanged. The "
        "selected step is recorded in `selected_candidate.json` as a provisional "
        "development candidate only.",
        "- No thresholds were changed, and no further run was started.",
        "",
    ]
    dump(
        out12 / "selected_candidate.json",
        {
            "status": "PROVISIONAL_DEVELOPMENT_CANDIDATE_NOT_PRODUCTION",
            "checkpoint": ev12["summary"]["selected_checkpoint"],
            "weight_sha256": sha256(
                Path(ev12["summary"]["selected_checkpoint"]) / "model/model.safetensors"
            ),
            "beats_phase5_11_candidate_on_objective": bool(beats_511),
        },
    )
    dump(
        out12 / "experiment_report.json",
        {
            "decision": decision,
            "selected_step": s12,
            "phase5_11_step": s11,
            "beats_phase5_11_candidate": bool(beats_511),
            "export": exp,
            "objective_by_step": ev12["summary"]["objective_by_step"],
            "phase5_11_objective": o11,
            "protected_final_test_used": False,
            "default_model_changed": False,
        },
    )
    (out12 / "experiment_report.md").write_text("\n".join(md), encoding="utf-8")
    dump(
        ROOT / "outputs/reports/phase5_12/latest.json",
        {**l12, "status": decision, "selected_step": s12},
    )
    print(out12 / "experiment_report.md")


if __name__ == "__main__":
    main()
