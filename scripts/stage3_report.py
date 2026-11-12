"""Stage 3 report from saved results (default, R0 and R1 held-out predictions)."""

from phase5_11_common import ROOT, load

OUT = ROOT / "outputs/reports/stage3_stitching"
KEYS = [
    "solid_f1",
    "solid_ap50",
    "solid_buffer_completeness",
    "solid_buffer_correctness",
    "solid_buffer_quality",
    "solid_pieces_per_covered_gt",
    "dash_f1",
    "dash_precision",
    "foreground_iou",
]


def main():
    md = [
        "# Stage 3: evidence-constrained stitching (post-processing only)",
        "",
        "Protocol `protocol.json` was sealed before scoring. It compares the current "
        "two-stage NMS filter with stitching (in-strip IoS merging, observed-only gap "
        "links, central-copy dash dedupe, thresholds after merging) on held-out "
        "P511/P512 predictions. The parameters come from the literature and were not "
        "tuned.",
        "",
    ]
    for src in ("default", "R0_baseline", "R1_bilinear"):
        r = load(OUT / f"stage3_results_{src}.json")
        s, b = r["summary"], r["bootstrap_stitch_vs_filter"]
        md += [
            f"## Predictions from: {src}",
            "",
            "| Metric | filter | stitch | difference [95% CI] |",
            "|---|---:|---:|---|",
        ]
        for k in KEYS:
            ci = (
                (
                    f"{b[k]['difference']:+.3f} [{b[k]['ci_difference'][0]:+.3f}, "
                    f"{b[k]['ci_difference'][1]:+.3f}]"
                )
                if k in b
                else "-"
            )
            md.append(f"| {k} | {s['filter'][k] or 0:.3f} | {s['stitch'][k] or 0:.3f} | {ci} |")
        a = r["audit_totals"]
        md += [
            "",
            f"Audit: {a['solid_overlap_merges']:.0f} merges, {a['solid_gap_links']:.0f} "
            f"observed-only gap links, {a['dash_duplicates_removed']:.0f} dash duplicates "
            f"removed, {a['false_links']:.0f}/{a['output_solids']:.0f} false links. "
            f"Gates: `{r['gates']}`, adopt: **{r['adopt_stitching']}**.",
            "",
        ]
    md += [
        "## Decision",
        "",
        "Stitching is **adopted** for the default and the Stage-2 base recipe (R0), where "
        "every gate passes. It raises complete-line solid F1 by about 0.25–0.28, solid "
        "buffer completeness by about 0.14–0.16 and dash precision by about 0.17–0.19, "
        "with correctness unchanged. On R1 it gives the best absolute numbers but 7.2% "
        "false links, which fails G3. The false-link rate depends on the model and must be "
        "re-checked for any new model. The negative-control check (G5) is part of the "
        "final-candidate evaluation.",
        "The training-side instance redefinition (random crop offsets) was not run. "
        "Stage 3's go criterion was met by post-processing alone, which is the roadmap's "
        "lower-cost path.",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
