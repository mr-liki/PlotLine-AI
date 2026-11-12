"""Stage 4 report from saved results."""

from phase5_11_common import ROOT, load

OUT = ROOT / "outputs/reports/stage4_loro"


def ci(m):
    return f"{m['difference']:+.3f} [{m['ci_difference'][0]:+.3f}, {m['ci_difference'][1]:+.3f}]"


def main():
    r = load(OUT / "stage4_results.json")
    md = [
        "# Stage 4: appearance augmentation and hard negatives (LORO, 5 folds × 2 seeds)",
        "",
        f"Protocol SHA-256 `{r['protocol_sha256']}`. The base recipe is {r['base_recipe']}. "
        "A1 adds 8-way dihedral, intensity and NoData augmentation, A2 adds a hard-negative "
        "stratum on top of A1, and A3 is R1 bilinear sampling plus the A1 augmentation.",
        "",
        "| Recipe (V1) | crop solid AP50 | crop dash AP50 | solid buffer compl. | corr. | FG IoU"
        " | dash F1 | dash precision | solid FP | dash FP | median TP solid conf. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n, s in r["summary"].items():
        c, g = s["V1_refit_thresholds"]["crop"], s["V1_refit_thresholds"]["geo"]
        md.append(
            f"| {n} | {c['solid_ap50']:.3f} | {c['dash_ap50']:.3f} | "
            f"{g['solid_buffer_completeness']:.3f} | {g['solid_buffer_correctness']:.3f} | "
            f"{g['foreground_iou']:.3f} | {g['dash_f1']:.3f} | {g['dash_precision']:.3f} | "
            f"{g['solid_fp']:.0f} | {g['dash_fp']:.0f} | "
            f"{r['median_tp_solid_confidence'][n]:.3f} |"
        )
    md += [
        "",
        "| Comparison | dash AP50 | dash FP | solid AP50 | solid buffer compl. | gates |",
        "|---|---|---|---|---|---|",
    ]
    for k, c in r["comparisons"].items():
        v = c["V1_refit_thresholds"]
        md.append(
            f"| {k} | {ci(v['crop']['dash_ap50'])} | {ci(v['geo']['dash_fp'])} | "
            f"{ci(v['crop']['solid_ap50'])} | {ci(v['geo']['solid_buffer_completeness'])} | "
            f"{r['gates'][k]} |"
        )
    md += [
        "",
        f"**Adopted: {r['adopted']}** (no recipe passes every gate).",
        "",
        "What the run shows:",
        "- **A1 (augmentation)** gives a small significant dash-AP gain. Its solid AP and "
        "completeness CIs dip just below the −0.02 non-inferiority bounds.",
        "- **A2 (hard-negative oversampling) is harmful**. Dash FP rises by 55–139, and dash "
        "and solid AP fall. Oversampling bright-background crops did not teach the model to "
        "reject vegetation or kerbs; it simply cost positive exposure.",
        "- **A3 (bilinear sampling + augmentation)** reproduces Stage 2's strong solid-AP gain "
        "(+0.16 to +0.24) and adds dash AP. It fails only G4: completeness CI lower bound "
        "−0.044 under the plain filter.",
        "- Across Stages 2 and 4, the bilinear fix improves ranking and confidence "
        "every time but never quite shows non-inferior buffer completeness under the plain "
        "filter. Stage 3 stitching recovers completeness for every model, so the two "
        "interact. That combination was not predeclared as a joint test.",
        "- Limitation: no real-dash copy-paste (no reliable road-surface mask).",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
