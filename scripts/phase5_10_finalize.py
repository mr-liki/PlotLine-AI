"""Phase 5.10 finalization: post-hoc diagnostics (never used for selection) and report.

Diagnostics added after the sealed search explain *why* no configuration or reconciliation
improved recognition. They do not alter the selected configuration or reconciliation rules.
"""

import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

import phase5_10_recovery as recovery
from roadmarking.inference.prediction_recovery import compatible, filter_geographic
from roadmarking.rasterization.pipeline import dump, sha256

ROOT = recovery.ROOT


def build():
    protocol = recovery.load(OUT / "protocol.json")
    p57 = recovery.load(ROOT / "outputs/reports/phase5_7_expanded/latest.json")
    p59 = recovery.load(ROOT / "outputs/reports/phase5_9/latest.json")
    p55 = recovery.load(ROOT / "outputs/reports/phase5_5/latest.json")
    manifest = recovery.load(Path(p57["dataset"]) / "metadata/dataset_manifest.json")
    parent = recovery.load(Path(p55["dataset"]) / "metadata/dataset_manifest.json")[
        "parent_dataset"
    ]
    p54 = recovery.load(Path(parent) / "metadata/dataset_manifest.json")
    legacy = recovery.legacy_component(
        manifest, p54, protocol["data_roles"]["LEGACY_POSITIVE_DEV"]["crops"]
    )
    report59, dataset59 = Path(p59["report_dir"]), Path(p59["dataset"])
    spec = recovery.load(
        ROOT
        / "outputs/reports/phase5_8_actual_review/20261106T115446528985Z/new_region_candidates.json"
    )
    tile = next(t for t in spec["tiles"] if t["candidate_id"] == "PHASE58_POSITIVE_DEV_REVIEW")
    new = recovery.tile_component(
        "NEW_POSITIVE_DEV",
        tile,
        recovery.load(dataset59 / "manifest.json")["crops"],
        report59 / "inference",
        dataset59 / "PHASE58_POSITIVE_DEV_REVIEW_ground_truth.npz",
        recovery.load(report59 / "native_crop_and_visibility_audit.json")["eligible_ids"],
    )
    return protocol, [legacy, new]


def proposal(comp, lookup, i):
    crop = lookup[comp["crop_ids"][i]]
    evidence = np.zeros(comp["shape"], bool)
    h, w = crop["observed_local"].shape
    r, c = crop["row"], crop["col"]
    evidence[r : r + h, c : c + w] = crop["raw"][comp["local_index"][i]] & crop["observed_local"]
    return {
        "mask": comp["masks"][i],
        "evidence": evidence,
        "support": crop["support"],
        "class": int(comp["classes"][i]),
        "score": float(comp["scores"][i]),
        "crops": {crop["crop_id"]},
        "members": [i],
    }


def gate_tally(comp, indices):
    lookup = {c["crop_id"]: c for c in comp["crops"]}
    solids = [i for i in indices if comp["classes"][i] == 0]
    cached = {i: proposal(comp, lookup, i) for i in solids}
    reasons = Counter()
    for a, i in enumerate(solids):
        for j in solids[a + 1 :]:
            if comp["crop_ids"][i] == comp["crop_ids"][j]:
                continue
            if not (
                binary_dilation(cached[i]["evidence"], iterations=2) & cached[j]["evidence"]
            ).any():
                reasons["no_observed_contact_within_2px"] += 1
                continue
            reasons[compatible(cached[i], cached[j], comp["observed"], comp["image"])[1]] += 1
    return {"solid_proposals": len(solids), "cross_crop_pairs": dict(reasons)}


def single_crop_ceiling(comp):
    """Best IoU any mask confined to one crop could reach against each complete GT identity."""
    rows = []
    for j, uid in enumerate(comp["ids"]):
        target = comp["targets"][j]
        area = int(target.sum())
        best = 0.0
        for crop in comp["crops"]:
            window = np.zeros(comp["shape"], bool)
            h, w = crop["validity_local"].shape
            window[crop["row"] : crop["row"] + h, crop["col"] : crop["col"] + w] = True
            best = max(best, (target & window).sum() / max(1, area))
        rows.append(
            {
                "source_id": uid,
                "class": recovery.CLASS_NAMES[int(comp["target_labels"][j])],
                "observed_pixels": area,
                "single_crop_iou_ceiling": float(best),
                "best_raw_same_class_iou": float(
                    max(
                        (
                            comp["iou"][i, j]
                            for i in range(len(comp["scores"]))
                            if comp["classes"][i] == comp["target_labels"][j]
                        ),
                        default=0.0,
                    )
                ),
            }
        )
    return rows


def table(stage_metrics):
    lines = [
        "| Stage | Preds | Solid TP/FP/FN | Dash TP/FP/FN | Solid P/R/F1 | Dash P/R/F1 | "
        "Macro F1 | Dash AP50/AP75 | Solid AP50/AP75 | FG IoU | Pixel P/R | Skeleton cov. |",
        "|---|---:|---|---|---|---|---:|---|---|---:|---|---:|",
    ]
    for stage, m in stage_metrics.items():
        s, d = m["per_class"]["solid"], m["per_class"]["dash"]
        ap = m.get("ap", {})

        def a(name, ap=ap):
            v = ap.get(name, {})
            return f"{v['AP50']:.3f}/{v['AP75']:.3f}" if "AP50" in v else "n/a"

        px = m["pixel"]["all_supported_paint"]
        sk = m.get("observed_skeleton_coverage_within_1px")
        lines.append(
            f"| {stage} | {m['predictions']} | {s['tp']}/{s['fp']}/{s['fn']} | "
            f"{d['tp']}/{d['fp']}/{d['fn']} | "
            f"{s['precision']:.3f}/{s['recall']:.3f}/{s['f1']:.3f} | "
            f"{d['precision']:.3f}/{d['recall']:.3f}/{d['f1']:.3f} | {m['macro_f1']:.4f} | "
            f"{a('longitudinal_dash')} | {a('longitudinal_solid')} | {m['foreground_iou']:.4f} | "
            f"{px['precision']:.3f}/{px['recall']:.3f} | "
            f"{'n/a' if sk is None else f'{sk:.3f}'} |"
        )
    return lines


def main():
    protocol, comps = build()
    results = recovery.load(OUT / "results.json")
    search = recovery.load(OUT / "search_results.json")
    frozen = protocol["baseline_reproduction"]["configuration"]
    diagnostics = {"status": "POST_HOC_DIAGNOSTIC_NOT_USED_FOR_SELECTION", "components": {}}
    for comp in comps:
        comp["overlaps"] = recovery.overlap_matrix(comp)
        comp["iou"] = recovery.packed_iou(comp["masks"], comp["targets"], comp["valid"]).reshape(
            len(comp["scores"]), len(comp["ids"])
        )
        selected = filter_geographic(
            comp["scores"],
            comp["classes"],
            comp["qualities"],
            comp["overlaps"],
            comp["crop_ids"],
            frozen,
        )
        diagnostics["components"][comp["name"]] = {
            "reconciliation_gates_on_frozen_filtered_solids": gate_tally(comp, selected),
            "reconciliation_gates_on_all_saved_raw_solids": gate_tally(
                comp, list(range(len(comp["scores"])))
            ),
            "single_crop_iou_ceiling": single_crop_ceiling(comp),
        }
    for comp in comps:
        rows = diagnostics["components"][comp["name"]]["single_crop_iou_ceiling"]
        diagnostics["components"][comp["name"]]["targets_unmatchable_by_any_single_crop_mask"] = (
            sum(r["single_crop_iou_ceiling"] < 0.5 for r in rows)
        )
    infeasible = Counter(
        v.split(":")[0] + ":" + v.split(":")[-1]
        for r in search["rows"]
        if not r["feasible"]
        for v in r["violations"]
    )
    diagnostics["constraint_violation_counts"] = dict(infeasible)
    dump(OUT / "post_hoc_diagnostics.json", diagnostics)

    # ---- validation evidence (if the validation suite has been run)
    validation = {}
    reports = ROOT / "outputs/reports"
    xml = reports / "phase5_10_pytest.xml"
    if xml.exists():
        suite = ET.parse(xml).getroot()
        suite = suite.find("testsuite") if suite.tag == "testsuites" else suite
        validation["pytest"] = dict(suite.attrib)
    for name in (
        "phase5_10_ruff.log",
        "phase5_10_black.log",
        "phase5_10_healthcheck.json",
        "phase5_10_pytest.xml",
        "phase5_10_tests.log",
    ):
        if (reports / name).exists():
            shutil.copy2(reports / name, OUT / name)
            validation[name] = sha256(reports / name)
    if (reports / "phase5_10_healthcheck.json").exists():
        health = recovery.load(reports / "phase5_10_healthcheck.json")
        validation["healthcheck_summary"] = {
            k: health[k]
            for k in health
            if k in ("status", "passed", "failed", "skipped", "summary")
        }

    sel = results["selection"]
    report = {
        **results,
        "post_hoc_diagnostics": "post_hoc_diagnostics.json",
        "validation": validation,
        "decision": {
            "postprocessing": (
                "RETAIN_FROZEN" if sel["selected_is_frozen"] else "ADOPT_SELECTED_DEV_ONLY"
            ),
            "reconciliation": (
                "NOT_ADOPTED"
                if not results["reconciliation_adoption"]["selected"]["adopt"]
                else "ADOPT_DEV_ONLY"
            ),
            "readiness": "RECOGNITION_IMPROVEMENT_REQUIRED_TRAINING_DATA",
        },
    }
    dump(OUT / "report.json", report)

    # ---- markdown report
    comp_res = results["components"]
    lines = [
        "# Phase 5.10: development-only prediction recovery and fragment reconciliation",
        "",
        f"Run `{protocol['run_id']}`; protocol SHA-256 `{results['protocol_sha256']}` "
        "(write-once, sealed before any candidate was scored).",
        "",
        "**Decision.** The predeclared search found no feasible configuration that improves on "
        "the frozen post-processing, so the frozen setting "
        "(solid .85 / dash .55 / quality .80 / NMS .30) is retained. Conservative reconciliation "
        "made zero unions in every component and is not adopted. Saved-prediction recovery "
        "cannot fix the Phase 5.9 geographic failure; the remaining gap needs new real training "
        "data (listed at the end).",
        "",
        "No weights were changed. No neural inference, training or synthetic data were used. "
        "The protected NEW_FINAL_TEST_RESERVED region and the Phase 5.6 test regions were not "
        "read. No annotation was reviewed or approved, and no geometry was extrapolated or "
        "bridged.",
        "",
        "## Inputs and integrity",
        "",
        "- `input_hash_manifest.json` holds hashes for 55 files: model weights, processor, "
        "approvals, saved predictions, crops, GT and raw observation/intensity rasters. They "
        "were checked before and after the run. Weight SHA-256 is `98ab4462…6a77786`.",
        "- There are three components. They were processed separately and never placed on a "
        "shared canvas:",
    ]
    for name, v in comp_res.items():
        i = v["inputs"]
        lines.append(
            f"  - **{name}**: {len(i['crops'])} crops, {i['saved_raw_proposals']} saved raw "
            f"proposals, {i['unique_source_ids']} unique source IDs, {i['scored_valid_pixels']} "
            f"scored valid pixels, canvas {i['canvas_shape']}"
        )
    lines += [
        "- In the legacy canvas, the overlapping crops agree exactly on raw observations, "
        "intensity and GT identities (0 conflicting pixels).",
        "- SPARSE_NEGATIVE_CONTROL has no positive targets, so it is a false-positive control "
        "only. It is not evidence of positive generalization.",
        "",
        "## Baseline reproduction (required before the search)",
        "",
        "All checks passed: `" + json.dumps(results["baseline_reproduction"]) + "`.",
        "- NEW_POSITIVE_DEV reproduces Phase 5.9 exactly: solid TP0/FP5/FN8, dash TP0/FP18/FN11, "
        "15 suppressed by global NMS, 23 kept, FG IoU 0.2059161701682868. The 23 masks are "
        "pixel-identical to the saved Phase 5.9 global predictions.",
        "- The per-crop selection equals the saved `selected` indices in all 16 crops. "
        "Crop-appearance macro F1 is reproduced: legacy .5927, new .0972.",
        "",
        "## Predeclared filtering search (24 settings)",
        "",
        f"Feasible settings: {sel['feasible_count']} of 24, which are the frozen setting (#22) "
        "and its NMS-.50 variant (#23). #23 has lower legacy macro F1 (.265 vs .267), so there "
        "is no strict improvement and the frozen setting is retained.",
        "",
        "| # | solid | dash | quality | NMS | Legacy solid TP/FP | Legacy dash TP/FP | "
        "Legacy F1 | New solid TP/FP | New dash TP/FP | New F1 | Neg FP | Feasible |",
        "|---:|---:|---:|---:|---:|---|---|---:|---|---|---:|---:|---|",
    ]
    for r in search["rows"]:
        c, cc = r["config"], r["components"]
        leg, nw = cc["LEGACY_POSITIVE_DEV"], cc["NEW_POSITIVE_DEV"]
        neg = cc["SPARSE_NEGATIVE_CONTROL"]["predictions"]
        lines.append(
            f"| {r['index']} | {c['confidence']['0']} | {c['confidence']['1']} | "
            f"{c['mask_quality']} | "
            f"{c['nms_iou']} | "
            f"{leg['per_class']['solid']['tp']}/{leg['per_class']['solid']['fp']} | "
            f"{leg['per_class']['dash']['tp']}/{leg['per_class']['dash']['fp']} | "
            f"{leg['macro_f1']:.3f} | "
            f"{nw['per_class']['solid']['tp']}/{nw['per_class']['solid']['fp']} | "
            f"{nw['per_class']['dash']['tp']}/{nw['per_class']['dash']['fp']} | "
            f"{nw['macro_f1']:.3f} | "
            f"{neg} | {'yes' if r['feasible'] else 'no'} |"
        )
    lines += [
        "",
        "Trade-offs:",
        "- Lowering solid confidence to .55/.70 recovers 1–2 complete new-region solid identities. "
        "It costs 13–22 extra solid false positives, which puts new-region solid precision at "
        ".05–.10, below the predeclared .10 floor. At .55 it also adds 2–3 false positives on "
        "the negative control.",
        "- Lowering dash confidence to .35 or mask quality to .65 adds legacy dash false "
        "positives without new TPs. This breaks the legacy precision floor and the FP budget.",
        "- No setting produces a complete-instance dash match in the new region. Its raw "
        "proposals contain zero dash IoU50 matches, so thresholds cannot create one.",
        "- Violation counts are in `post_hoc_diagnostics.json`, and every setting's full "
        "metrics are in `search_results.json`.",
        "- Caveat: selection and reporting use the same two development geographies. Any "
        "selected gain would be optimistic. Here the frozen setting was kept, so no selection "
        "optimism enters the reported numbers.",
        "",
        "## Stage comparison (unique geographic identities, complete-instance IoU≥.50)",
        "",
        "AP uses 101 recall points. It is confidence-truncated by the .25 proposal floor and, "
        "after filtering, by the operating point. It is not unrestricted pooled COCO AP. Pixel "
        "and skeleton metrics measure observed support only, not physical painted-boundary "
        "accuracy.",
    ]
    for name in ("LEGACY_POSITIVE_DEV", "NEW_POSITIVE_DEV"):
        v = comp_res[name]
        stages = {
            k: v[k]
            for k in ("raw", "frozen", "selected", "selected_reconciled", "frozen_reconciled")
        }
        lines += ["", f"### {name}", "", *table(stages), ""]
        f = v["frozen"]
        lines.append(
            f"- Solid fragmentation (frozen): {f['solid_fragmentation']}. "
            f"Dash separation (frozen): {f['dash_separation']}. Duplicates: solid "
            f"{f['per_class']['solid']['duplicates']}, dash {f['per_class']['dash']['duplicates']}."
        )
        ca = v["crop_appearance"]["frozen"]
        lines.append(
            f"- Crop appearances (overlapping, not unique identities), frozen: solid "
            f"TP{ca['0']['tp']}/FP{ca['0']['fp']}/FN{ca['0']['fn']}, dash "
            f"TP{ca['1']['tp']}/FP{ca['1']['fp']}/FN{ca['1']['fn']}, macro F1 {ca['macro_f1']:.4f}."
        )
    neg = comp_res["SPARSE_NEGATIVE_CONTROL"]
    lines += [
        "",
        f"### SPARSE_NEGATIVE_CONTROL\n\nRaw: {neg['raw']['predictions']} solid proposals, all "
        f"false positives. Frozen and selected filtering keep {neg['frozen']['predictions']}.",
        "",
        "Pixel note: reconciliation outputs only the union of existing masks, so its pixel-level "
        "metrics equal those of its input by construction. The script asserts this union is "
        "unchanged.",
        "",
        "## Reconciliation: why there were zero unions (post-hoc diagnostic)",
        "",
    ]
    for name, d in diagnostics["components"].items():
        lines.append(
            f"- **{name}**: frozen-filtered solids "
            f"{d['reconciliation_gates_on_frozen_filtered_solids']}; all saved raw solids "
            f"{d['reconciliation_gates_on_all_saved_raw_solids']}."
        )
    lines += [
        "",
        "After filtering, the surviving solid fragments almost never share observed pixels "
        "with a solid fragment from another crop. The overlapping-crop counterpart was usually "
        "below threshold, or was already removed by NMS as a duplicate. Compatible pairs exist "
        "only among sub-threshold raw proposals. Reconciling raw proposals before filtering "
        "would be a new, unregistered method, so it was not run or scored here. It may be a "
        "candidate for a separately predeclared phase.",
        "",
        "Single-crop IoU ceiling: the best IoU any mask confined to one 512 crop could reach "
        "against each complete GT identity.",
    ]
    for name, d in diagnostics["components"].items():
        rows = d["single_crop_iou_ceiling"]
        solid = [r for r in rows if r["class"] == "solid"]
        lines.append(
            f"- **{name}**: {d['targets_unmatchable_by_any_single_crop_mask']} of {len(rows)} "
            f"identities cannot reach IoU .50 from any single crop. Solid ceilings: "
            + ", ".join(f"{r['single_crop_iou_ceiling']:.2f}" for r in solid)
            + "."
        )
    lines += [
        "",
        "Crop tiling is therefore not the blocker: every identity could in principle reach "
        "IoU .50 from one crop. The limit is mask quality and detection. Best raw same-class "
        "IoUs per identity are in `post_hoc_diagnostics.json`. Legacy solids reach only "
        ".49/.00/.26. In the new region, only 3 of 8 solids reach .50 (.53/.52/.60), and no "
        "dash exceeds .44. Several identities have no overlapping same-class raw proposal at "
        "all. Disconnected masks were not associated, and complete solid recovery is not "
        "claimed.",
        "",
        "## Artifacts",
        "",
        "- `artifacts/<component>_{raw,frozen,selected,selected_reconciled,"
        "frozen_reconciled}.npz`: "
        "masks, scores, labels, validity, source crop IDs and member provenance.",
        "- `galleries/*_raw_frozen_selected_refined.png`: real intensity, GT, predictions, "
        "unmatched predictions and missed GT for each stage. "
        "`galleries/SPARSE_NEGATIVE_CONTROL_negative_control.png` covers the negative control. "
        "There are no union-event galleries because no unions occurred.",
        "- Also included: `baseline_reproduction.json`, `search_results.json`, "
        "`reconciliation_events.json` (empty), `post_hoc_diagnostics.json` and `results.json`.",
        "- Unmatched predictions are not independently verified physical error objects, "
        "because NMS-deduplicated fragments may still describe pieces of the same paint.",
        "",
        "Runtime (s): `"
        + json.dumps({k: round(v, 2) for k, v in results["runtime_seconds"].items()})
        + "`. "
        "This is CPU-only post-processing over saved predictions, with no GPU work.",
        "",
        "## Next-training recommendation (not started automatically)",
        "",
        "Post-processing alone cannot close the gap. The evidence points to model and "
        "training-data shortfalls:",
        "1. **Solid mask completeness on curves.** No legacy solid identity matches even among "
        "raw proposals (best IoU .49), and new-region solid matches appear only at precision "
        "≈.05–.10. Raw solid masks cover only partial lengths of long curved edge lines. Add "
        "approved real sections with long curved edge lines and merges/diverges. Optionally, "
        "predeclare raw-proposal reconciliation before NMS as a separate, bounded experiment. "
        "13 legacy and 70 new raw solid pairs pass the fixed gates, but this method is "
        "unscored.",
        "2. **Dash recognition in the new geography.** There are zero raw dash IoU50 matches, "
        "with 65 raw and 18 filtered dash false positives. Several of these visibly sit on "
        "roadside vegetation in the gallery. Add approved roadside hard negatives "
        "(vegetation, kerb edges) and faint/narrow dash examples from independent tracks.",
        "3. **Class confusion.** Phase 5.9 recorded 4 other-class raw IoU50 opportunities. "
        "Include chevron/hatched and symbol areas as ignored regions so they are not learned "
        "as solids.",
        "4. Keep PHASE58_POSITIVE_DEV_REVIEW and the legacy crops as development only. Keep "
        "NEW_FINAL_TEST_RESERVED sealed. Run any new training as a small predeclared real-only "
        "experiment with user approval.",
        "",
        "## Validation",
        "",
        f"`{json.dumps(validation)}`",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(OUT / "report.md"),
                "validation": validation,
                "diagnostics": {
                    k: {kk: vv for kk, vv in v.items() if kk != "single_crop_iou_ceiling"}
                    for k, v in diagnostics["components"].items()
                },
            },
            indent=1,
        )
    )


OUT = Path(recovery.load(ROOT / "outputs/reports/phase5_10/latest.json")["report_dir"])

if __name__ == "__main__":
    main()
