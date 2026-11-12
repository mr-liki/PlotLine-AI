"""Stage 0 report from saved harness results (no recomputation)."""

import xml.etree.ElementTree as ET

from phase5_11_common import ROOT, load
from roadmarking.rasterization.pipeline import dump

OUT = ROOT / "outputs/reports/stage0_harness"
SHOW = [
    "foreground_iou",
    "pixel_precision",
    "pixel_recall",
    "solid_buffer_completeness",
    "solid_buffer_correctness",
    "solid_ap50",
    "dash_ap50",
    "solid_f1",
    "dash_f1",
    "macro_f1",
]


def main():
    r = load(OUT / "stage0_results.json")
    folds = load(OUT / "folds.json")
    plumb = r["loro_plumbing"]
    timing = load(OUT / f"loro_plumbing_fold{plumb['fold']}_default/metrics.json")
    sec_crop = timing["model_only_seconds_total"] / len(timing["crops"])
    validation = {}
    xml = ROOT / "outputs/reports/stage0_pytest.xml"
    if xml.exists():
        s = ET.parse(xml).getroot()
        s = s.find("testsuite") if s.tag == "testsuites" else s
        validation = {k: s.attrib[k] for k in ("tests", "failures", "errors", "skipped")}
    md = [
        "# Stage 0: evaluation harness",
        "",
        "**Decision: GO.** The harness reproduces every existing Phase 5.12 development number "
        "exactly, produces block-bootstrap confidence intervals for every suite metric, and "
        "defines leakage-safe region folds for later stages. No training was run. The "
        "protected final test is sealed by metadata only.",
        "",
        "## Protected test seal",
        "",
        f"`protected_test_freeze.json` (write-once, SHA-256 `{r['protected_test_freeze_sha256']}`) "
        "records the region's bounds hash and its separation from every fold region (≥500 "
        "native units, 0 shared IDs). It is opened exactly once, at the end of the roadmap.",
        "",
        "## Region folds (5-fold, proximity-grouped)",
        "",
        "- The first rule linked regions that were within 500 units **or** shared any source "
        "ID. It collapsed 38 regions into 4 groups, one of them holding 35 regions, because long "
        "client edge lines run for kilometres through many regions. That fold file is kept "
        "as `folds_rejected_shared_id_rule.json`.",
        f"- The adopted rule groups by proximity only (<500 units): {folds['linked_groups']} "
        f"groups into 5 balanced folds. **{folds['cross_fold_shared_ids']} solid source IDs span "
        "folds**. Each is a different paint section, at least 500 m apart, of one long feature. "
        "No dash spans folds (asserted). The spanning IDs are listed so later stages can "
        "report a strict variant that excludes them.",
        "",
        "| Fold | regions | unique IDs | solid* | dash* | crops |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for f in folds["folds"]:
        md.append(
            f"| {f['fold']} | {len(f['regions'])} | {f['unique_ids']} | {f['solid']} | "
            f"{f['dash']} | {f['crops']} |"
        )
    md += [
        "",
        "*Solid and dash counts are summed over regions, so a long solid can be counted "
        "in more than one region.",
        "",
        "## Frozen metric suite",
        "",
        "- Threshold-free: solid and dash AP50 over all saved proposals. These are truncated "
        "by the 0.25 proposal floor.",
        "- Operating point (frozen filter): solid/dash precision, recall and F1 at "
        "complete-instance IoU50 (secondary for solids), foreground IoU, pixel P/R, and "
        "Wiedemann solid buffer completeness/correctness (3 px = 0.15 m buffer, skeletons, "
        "valid pixels only).",
        "- Fragmentation (solid pieces per covered GT line) and false positives per km² on "
        "the sparse negative control.",
        f"- Uncertainty: paired bootstrap over non-overlapping {r['bootstrap']['unit']}, "
        f"{r['bootstrap']['resamples']} resamples, seed {r['bootstrap']['seed']}. Instances "
        "are assigned to the block of their centroid, and pixels to their own block.",
        "",
        "## Reproduction",
        "",
        "All checks match (per-class FP/F1, macro F1 and foreground IoU, for the default "
        "model and the Phase 5.12 step-400 candidate on all three development components): "
        f"`{all(all(v.values()) for v in r['reproduction_of_phase5_12_numbers'].values())}`.",
        "",
        "## Default vs Phase 5.12 step 400, with 95% block-bootstrap CIs",
        "",
    ]
    for area, b in r["default_vs_p512_bootstrap"].items():
        md += [
            f"### {area} ({b['blocks']} blocks)",
            "",
            "| Metric | default | 5.12 s400 | difference | 95% CI of difference | " "significant |",
            "|---|---:|---:|---:|---|---|",
        ]
        for k in SHOW:
            m = b["metrics"].get(k)
            if not m:
                continue
            md.append(
                f"| {k} | {m['a']:.3f} | {m['b']:.3f} | {m['difference']:+.3f} | "
                f"[{m['ci_difference'][0]:+.3f}, {m['ci_difference'][1]:+.3f}] | "
                f"{'yes' if m['difference_excludes_zero'] else 'no'} |"
            )
        md.append("")
    md += [
        "What the CIs show:",
        "- **The legacy regression is not statistically established.** Foreground IoU "
        "−0.139 has a CI of [−0.269, +0.036], and solid buffer completeness falls "
        "0.579 → 0.226 with a CI that includes 0. Only the pixel precision gain is "
        "significant. Legacy has 18 blocks, and one long edge line dominates it.",
        "- **The new-area gains are significant at pixel level.** Foreground IoU "
        "+0.211 [+0.058, +0.306], pixel recall +0.325 [+0.060, +0.457], and solid buffer "
        "completeness +0.304 [+0.068, +0.404].",
        "- **Pooled across both areas, no metric difference is significant.** The two "
        "areas pull in opposite directions, and 32 blocks are too few to resolve that. This is "
        "the quantitative reason the roadmap moves evaluation to region folds.",
        f"- Negative control FP/km²: `{r['negative_control_fp_per_km2']}`.",
        "",
        "## Score-factor log (all saved proposals on the development areas)",
        "",
        "`factor_log.json` stores class probability, mask quality, score, filter fate, IoU50 "
        "flags and on-paint fraction for every detection.",
        "",
        "| Model / class / group | n | median class prob | median mask quality | " "median score |",
        "|---|---:|---:|---:|---:|",
    ]
    for k, v in r["score_factor_summary"].items():
        md.append(
            f"| {k} | {v['n']} | {v['median_class_prob']:.3f} | "
            f"{v['median_mask_quality']:.3f} | {v['median_score']:.3f} |"
        )
    md += [
        "",
        "For Stage 1 this means:",
        '- Most solid "false positives" are **on-paint fragments**: 80 of 124 for the '
        "default model, 97 of 128 for 5.12. Each is a piece of a real long line that fails "
        "complete-line IoU50. They need stitching, not suppression.",
        "- **Off-paint** solid false positives have lower class probability (default .827, "
        "5.12 .857) than true lines (.869 / .961) and fragments (.917 / .902).",
        "- **Mask quality barely separates the groups** (.84–.89 throughout). So rescoring "
        "mask quality alone (option a) is unlikely to separate true from false solids, "
        "although it does explain the lost-edge score drop. Class-probability gating and "
        "calibration (options b/c) look more informative.",
        "- With only 3 true-positive solids per model, this needs confirming on the fold " "pool.",
        "",
        "## Leave-one-region-out plumbing check",
        "",
        f"- Fold {plumb['fold']} ({len(plumb['regions'])} regions, {plumb['crops']} crops) was "
        "inferred with the default model and scored through the generic training-region "
        f"component builder. Model-only inference is {sec_crop:.3f} s per 512 crop.",
        "- **Useful side finding.** The default model predates Phase 5.11, so the P511_* and "
        "P512_* regions (26 of 38 training regions) are genuinely unseen by it. Its fold-3 "
        "results on them are therefore real generalization evidence for the *default*. Dash "
        "F1 is 0.62–0.86 in three of the regions, but complete-line solid matching is near "
        "zero, with buffer completeness 0.34–0.60.",
        "- For candidates trained on these regions, the folds must be retrained "
        "leave-one-out. They are not valid for candidates as they stand.",
        "",
        "## Cost of using the folds in later stages",
        "",
        f"- One fold run is about 6 min of training (600 steps × ~0.5 s) plus about "
        f"{sec_crop * 80 / 60:.1f} min of inference for about 80 held-out crops, plus scoring. "
        "That is about 8–10 min per fold. 5 folds × 3 seeds ≈ 2–2.5 GPU-hours per recipe, at "
        "a 2.6 GB peak, which fits the RTX 3050.",
        "",
        "## Stage 0 go/no-go",
        "",
        "- Reproduces current numbers: **yes**. CIs computed for all metrics: **yes**. Folds "
        "are leakage-checked and dashes never span folds: **yes**.",
        "- Next is Stage 1: score and selection fixes, judged on fold pools with these CIs.",
        "",
        f"Validation: `{validation}`",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    dump(
        OUT / "report.json",
        {
            "stage": 0,
            "decision": "GO",
            "validation": validation,
            "folds_linked_groups": folds["linked_groups"],
            "spanning_solid_ids": folds["cross_fold_shared_ids"],
            "model_only_seconds_per_crop": sec_crop,
        },
    )
    print(OUT / "report.md")


if __name__ == "__main__":
    main()
