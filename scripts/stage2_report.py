"""Stage 2 report from saved results."""

from phase5_11_common import ROOT, load

OUT = ROOT / "outputs/reports/stage2_loro"


def ci(m):
    return f"{m['difference']:+.3f} [{m['ci_difference'][0]:+.3f}, {m['ci_difference'][1]:+.3f}]"


def main():
    r = load(OUT / "stage2_results.json")
    md = [
        "# Stage 2: thin-line supervision (leave-one-region-out, 5 folds × 2 seeds)",
        "",
        f"Protocol SHA-256 `{r['protocol_sha256']}`, sealed before training. Each run used "
        f"the final step-600 weights; median training time was "
        f"{r['train_seconds_per_run_median'] / 60:.1f} min per run. Results are scored on the "
        "26 held-out P511/P512 regions, which the warm start never saw.",
        "",
        "| Recipe | scoring | crop solid AP50 | crop dash AP50 | solid buffer compl. | "
        "buffer corr. | geo FG IoU | geo dash F1 | solid FP | dash FP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec, s in r["summary"].items():
        for v, d in s.items():
            c, g = d["crop"], d["geo"]
            md.append(
                f"| {rec} | {v} | {c['solid_ap50']:.3f} | {c['dash_ap50']:.3f} | "
                f"{g['solid_buffer_completeness']:.3f} | {g['solid_buffer_correctness']:.3f} "
                f"| {g['foreground_iou']:.3f} | {g['dash_f1']:.3f} | {g['solid_fp']:.0f} | "
                f"{g['dash_fp']:.0f} |"
            )
    md += [
        "",
        "Paired differences (V1 thresholds), 95% region-bootstrap CI:",
        "",
        "| Comparison | solid AP50 | dash AP50 | solid buffer completeness | FG IoU | gates |",
        "|---|---|---|---|---|---|",
    ]
    for k, c in r["comparisons"].items():
        v = c["V1_refit_thresholds"]
        md.append(
            f"| {k} | {ci(v['crop']['solid_ap50'])} | {ci(v['crop']['dash_ap50'])} | "
            f"{ci(v['geo']['solid_buffer_completeness'])} | {ci(v['geo']['foreground_iou'])} "
            f"| {r['gates'][k]} |"
        )
    conf = r["median_tp_solid_confidence"]
    md += [
        "",
        "Median V0 confidence of correct (crop IoU50) solid detections: "
        + ", ".join(f"{k} {v:.3f}" for k, v in conf.items())
        + ".",
        "",
        f"**Adopted under the sealed rule: {r['adopted_recipe']}.** R1 and R2 fail only G2: "
        "the solid buffer completeness CI includes 0.",
        "",
        "What the run shows:",
        "- The bilinear logit sampling fix (R1) is strongly supported on held-out roads. "
        "Solid AP50 is +0.19 with a CI that excludes 0, foreground IoU rises "
        "significantly, and the median confidence of correct solids rises from 0.866 to "
        "0.932. That removes the thin-line threshold-margin problem diagnosed in Phase 5.13. "
        "No metric gets significantly worse.",
        "- The skeleton-recall term adds a further small, significant solid-AP gain over "
        "R1, +0.007 to +0.046.",
        "- Process note: the roadmap's no-go action was to keep the sampling change if "
        "neutral-positive, but the Stage 2 adoption rule did not include that clause. "
        "The sealed rule is honoured and R0 stays the base. Stage 4 predeclares an extra "
        "recipe, A3 = R1 + augmentation, to test the sampling change again under its own "
        "gates.",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
