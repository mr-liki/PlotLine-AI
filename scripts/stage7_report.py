# ruff: noqa: E501, N806
"""Stage 7 report package: report.md + versioned artifact manifest (SHA-256 per file)."""

import json
from datetime import UTC, datetime
from pathlib import Path

from phase5_11_common import ROOT, load
from roadmarking.rasterization.pipeline import dump, sha256
from stage7_common import OUT


def f(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main():
    prov = load(OUT / "7A_protected_test_provenance.json")
    fit = load(OUT / "7B_thin_line_fit_table.json")["rows"]
    base = load(OUT / "7D_S6_01_baseline/baseline_results.json")
    proto = load(OUT / "7C_protocol.json")
    links = load(OUT / "7F_stage3_link_breakdown.json")
    res = load(OUT / "7H_results.json")
    nt = load(ROOT / "outputs/reports/stage7_new_test/latest.json")
    nts = load(Path(nt["report_dir"]) / "new_region_candidates.json")
    runs = {p.stem: load(p) for p in sorted((OUT / "runs").glob("*.json"))}
    L = ["# Stage 7: controlled 2×2, stitching safety and test provenance", ""]
    L += [f"**Final status:** {', '.join(f'`{s}`' for s in res['statuses'])}", ""]
    L += [
        "## Decision",
        "",
        "- **Primary: `NO_IMPROVEMENT`.** No arm passes G1 against the frozen candidate in "
        ">= 2 of 3 seeds. The default and the frozen candidate stay unchanged; no checkpoint "
        "is promoted.",
        "- `STITCHING_SAFETY_BLOCKED` applies only to arm C versus the *default* model (it "
        "passes every other gate there but its true link-error rate exceeds 5% in a seed); C "
        "does not beat the frozen candidate either.",
        "- The thin-line fix is **not verified** by the sealed rule (solid AP50 up in 4/6 "
        "pairs, pixel IoU up in 3/6; 5/6 required). It raised instance recall@50 in 5/6 "
        "pairs (+0.02 to +0.08, descriptive), but lowered LEGACY pixel IoU (G6 fails for B "
        "and D) and raised link errors under stitching.",
        "- The +44 Stage-6 markings (500 vs 456) gave no consistent effect (C-A, D-B mixed "
        "signs, magnitudes <= 0.025 IoU).",
        "- Seed spread at step 600 is small (pixel IoU SD ~0.01), but trajectories are "
        "non-monotonic (all three arm-A runs score higher at step 200 than 600); the predeclared "
        "step-600 rule was kept.",
        "- The A_s895 rerun of the frozen-candidate recipe is not bit-identical (different "
        "weight hash) but lands close (pooled pixel IoU 0.407 vs 0.413).",
        "- Reliability targets remain far away (pixel precision ~0.57 vs 0.90, recall ~0.60 "
        "vs 0.95, instance recall ~0.27 vs 0.90): `MORE_REAL_DATA_REQUIRED` and "
        "`NEW_INDEPENDENT_TEST_REQUIRED`.",
        "",
    ]
    L += ["## 7A Protected-test provenance", ""]
    L += [f"- {k.replace('_', ' ')}: **{v}**" for k, v in prov["answers"].items()]
    L += ["", prov["discrepancy_explanation"], ""]
    for k, v in prov["split_roles"].items():
        L.append(f"- *{k}*: {', '.join(v) if v else 'none (see 7J)'}")
    L += ["", "## 7B Thin-line supervision (synthetic software test, not training data)", ""]
    L += ["Best achievable fit of 1/4-grid logits to a native line; line recall at p > 0.5.", ""]
    L += ["| Width | Offset in cell | Nearest | Bilinear |", "|---|---|---:|---:|"]
    for r in fit:
        L.append(
            f"| {r['width_px']} px ({r['width_m']:.2f} m) | {r['row_offset_in_cell']} | "
            f"{f(r['nearest']['line_recall_at_0_5'], 2)} | {f(r['bilinear']['line_recall_at_0_5'], 2)} |"
        )
    L += [
        "",
        "Nearest sampling cannot represent a 3 px (0.15 m) edge line that straddles an output "
        "cell boundary (0% recoverable); bilinear recovers ≥ 96% at every offset for ≥ 3 px. "
        "1 px lines are below what a 1/4 grid can represent with either sampler. Coordinate "
        "convention, align_corners=False, border padding, NoData invariance, gradient "
        "locality, finiteness (FP16/extreme logits) and auxiliary-loss identity are tested in "
        "`tests/test_criterion_supervision_math.py`.",
        "",
        "## 7D S6_01 first-use baseline (write-once, fixed settings)",
        "",
        "| Metric | Default V0 | Frozen candidate V1 + stitch |",
        "|---|---:|---:|",
    ]
    hd, hc = base["default_V0"]["headline"], base["frozen_candidate_V1_stitch"]["headline"]
    for k in hd:
        L.append(f"| {k} | {f(hd[k])} | {f(hc[k])} |")
    L += ["", base["caution"], "S6_01 is now development data.", ""]
    L += ["## 7F Stitching safety: Stage-3 false links re-broken-down by physical identity", ""]
    L += [
        "| Source | Stage-3 rule | same line (harmless) | parallel joined | solid+dash | "
        "dashes joined | true errors / outputs | solid join errors / solids | NoData-bridged |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k, v in links.items():
        b = v["breakdown"]
        L.append(
            f"| {k} | {b.get('stage3_rule_false_links', 0)}/{b['output_solids']} "
            f"({b['stage3_rule_rate']:.1%}) | {b.get('same_physical_line', 0)} | "
            f"{b.get('parallel_lines_joined', 0)} | {b.get('solid_dash_joined', 0)} | "
            f"{b.get('dashes_joined', 0)} | {b.get('true_false_links', 0)}/{b['output_instances']} "
            f"({b['true_false_link_rate_of_outputs']:.1%}) | {b['solid_join_error_rate']:.1%} | "
            f"{b.get('nodata_bridges', 0)} |"
        )
    L += ["", "## 7C/7E 2×2 runs (step 600 primary, fixed in advance)", ""]
    L += [
        "| Run | Pixel IoU | Inst. recall@50 | Inst. F1@50 | Dash recall@50 | True link err. | s | Peak GB | ID coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    final = str(proto["shared"]["steps"])
    for n, r in res["runs"].items():
        s = r[final]["summary"]
        ru = runs.get(n, {})
        L.append(
            f"| {n} | {f(s['pixel_iou'])} | {f(s['all_instance_recall50'])} | "
            f"{f(s['all_instance_f150'])} | {f(s['dash_instance_recall50'])} | "
            f"{s['true_false_link_rate']:.1%} | {f(ru.get('segment_elapsed_seconds'), 0)} | "
            f"{f(ru.get('cuda_peak_allocated_bytes', 0) / 1e9, 2)} | "
            f"{f(ru.get('exposure', {}).get('id_coverage'), 2)} |"
        )
    L += [
        "",
        "| Reference | Pixel IoU | Inst. recall@50 | Inst. F1@50 | Dash recall@50 |",
        "|---|---:|---:|---:|---:|",
    ]
    for n, r in res["references"].items():
        s = r["pooled"]
        L.append(
            f"| {n} | {f(s['pixel_iou'])} | {f(s['all_instance_recall50'])} | {f(s['all_instance_f150'])} | {f(s['dash_instance_recall50'])} |"
        )
    L += ["", "## 7G Strata (instance recall@50 / n, IDs deduplicated; descriptive only)", ""]
    keys = ["solid", "dash", "long", "short", "narrow", "faint", "clear", "near_bright_structure"]
    L.append("| Model | " + " | ".join(keys) + " |")
    L.append("|---|" + "---:|" * len(keys))
    rows = [(n, r["pooled"]["strata"]) for n, r in res["references"].items()]
    rows += [(n, r[final]["summary"]["strata"]) for n, r in res["runs"].items()]
    for n, st in rows:
        L.append(
            f"| {n} | "
            + " | ".join(f"{f(st[k]['instance_recall50'], 2)} / {st[k]['n']}" for k in keys)
            + " |"
        )
    L += ["", "Strata with n < 10 are indicative only; no CI is claimed for them.", ""]
    L += ["", "## 7H Predeclared gates (each arm vs both references)", ""]
    L += [
        "| Arm | Reference | G1 | G2 | G3 | G4 | G5 | G6 | All |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for a, v in res["arms"].items():
        for r, g in v["versus"].items():
            gs = g["gates"]
            L.append(
                f"| {a} | {r} | "
                + " | ".join("✓" if gs[k] else "✗" for k in ("G1", "G2", "G3", "G4", "G5", "G6"))
                + f" | {'✓' if g['passes_all'] else '✗'} |"
            )
    L += [
        "",
        f"Selected arm: **{res['selected_arm'] or 'none — default and frozen candidate stay unchanged'}**",
        "",
    ]
    L += [
        "### Fix effect (B−A, D−C, paired by seed)",
        "",
        "| Pair | Solid AP50 Δ | Pixel IoU Δ | Inst. recall Δ | Link err. (fix on) |",
        "|---|---:|---:|---:|---:|",
    ]
    for p in res["fix_pairs"]:
        L.append(
            f"| {p['pair']} | {p['solid_ap50_diff']:+.3f} | {p['pixel_iou_diff']:+.3f} | {p['instance_recall50_diff']:+.3f} | {p['true_false_link_rate_on']:.1%} |"
        )
    L += ["", f"Fix verified by the sealed rule: **{res['fix_verified']}**", ""]
    L += [
        "### Data effect (C−A, D−B)",
        "",
        "| Pair | Pixel IoU Δ | Inst. recall Δ | Dash recall Δ |",
        "|---|---:|---:|---:|",
    ]
    for p in res["data_pairs"]:
        L.append(
            f"| {p['pair']} | {p['pixel_iou_diff']:+.3f} | {p['instance_recall50_diff']:+.3f} | {p['dash_recall50_diff']:+.3f} |"
        )
    L += [
        "",
        "## 7I Reliability gate (provisional targets)",
        "",
        "| Metric | Value | Target |",
        "|---|---:|---:|",
    ]
    for k, (v, t) in res["reliability_gate"]["values_vs_targets"].items():
        L.append(f"| {k} | {f(v)} | {t} |")
    L += [
        "",
        f"Met: **{res['reliability_gate']['met']}** (dev areas, optimistic; no untouched test)",
        "",
    ]
    L += ["## 7J New independent final-test candidates (human approval required)", ""]
    L += [
        f"{len(nts['tiles'])} regions verified from {nts['candidates_considered']} guarded windows; "
        f"{nts['new_unique_observed_identities']['dash']} new dash / "
        f"{nts['new_unique_observed_identities']['solid']} new solid IDs. Each is ≥ 500 units "
        "from every used footprint and shares no dev/benchmark/test ID. Full ID "
        "disjointness from training was infeasible (0 of 998 windows), because long "
        "client edge lines run the whole corridor; shared IDs are listed per region. "
        f"QGIS package: `{nt['data_dir']}`. Not to be opened before approval and freezing.",
        "",
        "| Region | Acquisition | Solid / dash IDs | Shared training IDs | Distance to training |",
        "|---|---|---:|---:|---:|",
    ]
    for t in nts["tiles"]:
        L.append(
            f"| {t['candidate_id']} | {t['acquisition']} | {t['observed_solid_ids']} / {t['observed_dash_ids']} | {len(t['shares_ids_with_existing_training'])} | {t['min_distance_to_training_native']:.0f} |"
        )
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    files = [
        *sorted(p for p in OUT.rglob("*.json") if "inference" not in p.parts),
        OUT / "report.md",
    ]
    dump(
        OUT / "package_manifest.json",
        {
            "version": "stage7-v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "files": {str(p.relative_to(ROOT)): sha256(p) for p in files},
        },
    )
    print(json.dumps({"statuses": res["statuses"], "selected": res["selected_arm"]}))


if __name__ == "__main__":
    main()
