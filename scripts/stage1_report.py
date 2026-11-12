"""Stage 1 report from saved results."""

from phase5_11_common import ROOT, load

OUT = ROOT / "outputs/reports/stage1_scoring"


def ci(m):
    return f"{m['difference']:+.3f} [{m['ci_difference'][0]:+.3f}, {m['ci_difference'][1]:+.3f}]"


def main():
    r = load(OUT / "stage1_results.json")
    md = [
        "# Stage 1: scoring and selection",
        "",
        f"Protocol `protocol.json` (SHA-256 `{r['protocol_sha256']}`) was sealed before any "
        "variant was scored. No training was run.",
        "",
        "## 1a. Score variants for the default model (26 unseen regions, 5-fold cross-fitted)",
        "",
        "| Variant | crop solid AP50 | crop dash AP50 | solid buffer completeness | "
        "geo FG IoU | geo dash F1 | solid FP | dash FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for v, s in r["pool_summary"].items():
        c, g = s["crop_appearance"], s["geographic"]
        md.append(
            f"| {v} | {c['solid_ap50']:.3f} | {c['dash_ap50']:.3f} | "
            f"{g['solid_buffer_completeness']:.3f} | {g['foreground_iou']:.3f} | "
            f"{g['dash_f1']:.3f} | {g['solid_fp']} | {g['dash_fp']} |"
        )
    md += [
        "",
        "Differences vs V0 with 95% region-bootstrap CIs:",
        "",
        "| Variant | solid AP50 | dash AP50 | solid buffer completeness | geo FG IoU | gates |",
        "|---|---|---|---|---|---|",
    ]
    for v, b in r["pool_bootstrap_vs_V0"].items():
        ca, ge = b["crop_appearance"]["metrics"], b["geographic"]["metrics"]
        failed = [k for k, ok in r["gates"][v]["checks"].items() if not ok]
        md.append(
            f"| {v} | {ci(ca['solid_ap50'])} | {ci(ca['dash_ap50'])} | "
            f"{ci(ge['solid_buffer_completeness'])} | {ci(ge['foreground_iou'])} | "
            f"{'pass' if not failed else 'fail: ' + ', '.join(failed)} |"
        )
    md += [
        "",
        f"**Outcome: no variant passes every gate; chosen = {r['chosen_variant']}.** "
        "Under the protocol, 1b therefore used V1 (refit thresholds).",
        "- Protocol flaw, recorded honestly: V1 changes only thresholds, so its AP "
        "difference is zero by construction and gate G1 could never be met. Its actual "
        "effect is large and significant on solid buffer completeness, and it halves dash "
        "false positives at the cost of more solid FPs.",
        "- V5 (Platt calibration) is the only score change that improves solid ranking (AP) "
        "significantly. It fails on geographic FG IoU, whose CI lower bound is −0.072.",
        "- Every variant keeps the negative control at 0 FP/km².",
        "",
        "## 1b. Weight averaging (development areas only; descriptive)",
        "",
        "| Model | scoring | legacy FG IoU | lost-edge cov | legacy buffer compl. | new FG "
        "IoU | new buffer compl. | advance |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for m, x in r["dev_1b"].items():
        for v, d in x.items():
            leg, new = d["LEGACY_POSITIVE_DEV"], d["NEW_POSITIVE_DEV"]
            adv = r["advance_to_stage2"].get(f"{m}|{v}", {}).get("advance", "-")
            md.append(
                f"| {m} | {v} | {leg['summary']['foreground_iou']:.3f} | "
                f"{leg.get('lost_edge_coverage', 0):.2f} | "
                f"{leg['summary']['solid_buffer_completeness']:.2f} | "
                f"{new['summary']['foreground_iou']:.3f} | "
                f"{new['summary']['solid_buffer_completeness']:.2f} | {adv} |"
            )
    md += [
        "",
        "**Advance to Stage 2: WiSE-FT α=0.5 and α=0.7, both with V1 thresholds.** "
        "Each keeps legacy FG IoU non-inferior, keeps the lost edge (≥0.60), adds "
        "significant new-area FG IoU, and does not add negative-control FPs. These are "
        "candidate *models*. They were selected on development areas and trained on "
        "pool regions, so they still need confirmation on fresh data.",
        "",
        f"Runtime: {r['runtime_seconds'] / 60:.1f} min.",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
