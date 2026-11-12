"""Stage 7F-7I: score every 2x2 run with the sealed pipelines, apply the predeclared gates.

Nothing here is tuned: thresholds, pipelines, checkpoints (step 600 primary) and gates come
from the write-once 7C protocol.
"""

import json
from pathlib import Path

import numpy as np

from phase5_11_common import ROOT, load
from phase5_11_evaluate import components
from roadmarking.evaluation.full_metrics import aggregate
from roadmarking.evaluation.harness import paired_block_bootstrap
from roadmarking.rasterization.pipeline import dump, sha256
from stage1_scoring import DEFAULT
from stage7_common import OUT, evaluate_component, s6_component, settings

INFER = OUT / "inference"
POOL = ("S6_01", "NEW_POSITIVE_DEV")  # + the legacy component (named at runtime)
NEG = "SPARSE_NEGATIVE_CONTROL"


def build(legacy_folder, new_folder, s6_folder):
    leg, new, neg = components(legacy_folder, new_folder)
    s6 = s6_component(s6_folder)
    return {c["name"]: c for c in (leg, new, neg, s6)}, leg["name"]


def score(comps, thr, use_stitch):
    return {name: evaluate_component(c, thr, use_stitch) for name, c in comps.items()}


def pooled(results, names):
    recs, dets = {}, {}
    for n in names:
        recs.update({f"{n}|{k}": v for k, v in results[n]["records"].items()})
        dets.update({f"{n}|{k}": v for k, v in results[n]["detections"].items()})
    return recs, dets


def summary(results, names):
    agg = aggregate([results[n]["full"] for n in names])
    links = [results[n]["full"]["stitch_audit"] for n in names]
    solids = sum(x["output_solids"] for x in links)
    outputs = sum(x["output_instances"] for x in links)
    agg["true_false_link_rate"] = sum(x["true_false_links"] for x in links) / max(1, outputs)
    agg["solid_join_error_rate"] = sum(
        x.get(k, 0)
        for x in links
        for k in ("parallel_lines_joined", "different_lines_joined", "solid_dash_joined")
    ) / max(1, solids)
    agg["output_instances"] = outputs
    agg["stage3_rule_false_link_rate"] = sum(x["stage3_rule_false_links"] for x in links) / max(
        1, solids
    )
    agg["output_solids"] = solids
    agg["link_kinds"] = {
        k: sum(x.get(k, 0) for x in links)
        for k in (
            "same_physical_line",
            "dashes_joined",
            "solid_dash_joined",
            "parallel_lines_joined",
            "different_lines_joined",
            "nodata_bridges",
            "duplicate_covered_targets",
        )
    }
    hs = [results[n]["harness_summary"] for n in names]
    agg["solid_buffer_completeness_mean_over_areas"] = float(
        np.mean([h["solid_buffer_completeness"] for h in hs])
    )
    return agg


def strip(results):
    return {
        n: {
            "metrics": aggregate([r["full"]]),
            "ap": r["full"]["ap"],
            "links": r["full"]["stitch_audit"],
            "link_records": r["full"]["link_records"],
            "per_target": r["full"]["per_target"],
            "harness_summary": r["harness_summary"],
        }
        for n, r in results.items()
    }


def reference_results(cfg):
    p59 = load(ROOT / "outputs/reports/phase5_9/latest.json")
    base = OUT / "7D_S6_01_baseline/inference"
    folders = {
        "default_V0": (
            DEFAULT / "validation_step0050",
            Path(p59["report_dir"]) / "inference",
            base / "default_V0",
        ),
        "frozen_candidate_V1_stitch": (
            ROOT / "outputs/reports/final_candidate/inference/legacy",
            ROOT / "outputs/reports/final_candidate/inference/new",
            base / "frozen_candidate_V1_stitch",
        ),
    }
    out = {}
    for name, (leg, new, s6) in folders.items():
        comps, legacy_name = build(leg, new, s6)
        c = cfg[name]
        out[name] = score(comps, c["thresholds"], c["stitch"])
    return out, legacy_name


def gate_versus(run_results, ref, names, legacy):
    """Per-seed comparisons of one arm against one reference (G1-G6 of the protocol)."""
    per_seed = []
    for res in run_results:
        boot = paired_block_bootstrap(*pooled(ref, names), *pooled(res, names), n=2000, seed=0)
        m = boot["metrics"]
        a, b = summary(ref, names), summary(res, names)
        per_seed.append(
            {
                "fg_iou_diff_ci": m["foreground_iou"]["ci_difference"],
                "fg_iou_diff": m["foreground_iou"]["difference"],
                "instance_recall50_diff": b["all_instance_recall50"] - a["all_instance_recall50"],
                "instance_recall50": b["all_instance_recall50"],
                "solid_completeness_diff": m["solid_buffer_completeness"]["difference"],
                "negative_fps": aggregate([res[NEG]["full"]])["all_false_positives50"],
                "negative_fps_ref": aggregate([ref[NEG]["full"]])["all_false_positives50"],
                "true_false_link_rate": b["true_false_link_rate"],
                "legacy_iou": aggregate([res[legacy]["full"]])["pixel_iou"],
                "legacy_iou_ref": aggregate([ref[legacy]["full"]])["pixel_iou"],
                "recall_by_id": {
                    t["id"]: t["matched50"] for n in names for t in res[n]["full"]["per_target"]
                },
                "recall_by_id_ref": {
                    t["id"]: t["matched50"] for n in names for t in ref[n]["full"]["per_target"]
                },
            }
        )
    # Instance recall CI: paired ID bootstrap (deduplicated physical IDs).
    for s in per_seed:
        ids = sorted(set(s["recall_by_id"]) & set(s["recall_by_id_ref"]))
        d = np.array([float(s["recall_by_id"][i]) - float(s["recall_by_id_ref"][i]) for i in ids])
        rng = np.random.default_rng(0)
        draws = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)] if len(d) else [0]
        s["instance_recall50_diff_ci"] = [
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ]
        s["paired_ids"] = len(ids)
        del s["recall_by_id"], s["recall_by_id_ref"]
    g1 = (
        sum(s["fg_iou_diff_ci"][0] > 0 or s["instance_recall50_diff_ci"][0] > 0 for s in per_seed)
        >= 2
    )
    g2 = not any(
        s["fg_iou_diff_ci"][1] < 0 or s["instance_recall50_diff_ci"][1] < 0 for s in per_seed
    )
    g3 = float(np.median([s["solid_completeness_diff"] for s in per_seed])) >= -0.05
    g4 = (
        float(np.median([s["negative_fps"] for s in per_seed]))
        <= per_seed[0]["negative_fps_ref"] + 1
    )
    g5 = all(s["true_false_link_rate"] <= 0.05 for s in per_seed)
    g6 = (
        float(np.median([s["legacy_iou"] for s in per_seed]))
        >= per_seed[0]["legacy_iou_ref"] - 0.03
    )
    return {
        "per_seed": per_seed,
        "gates": {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5, "G6": g6},
    }


def main():
    protocol = load(OUT / "7C_protocol.json")
    for path, digest in protocol["code_sha256"].items():
        assert sha256(ROOT / path) == digest, f"code changed since protocol: {path}"
    cfg = settings()
    primary = cfg["frozen_candidate_V1_stitch"]
    secondary = cfg["default_V0"]
    refs, legacy = reference_results(cfg)
    names = [*POOL, legacy]
    runs = {}
    for arm in protocol["arms"]:
        for seed in protocol["shared"]["seeds"]:
            name = f"{arm}_s{seed}"
            runs[name] = {}
            for step in protocol["shared"]["checkpoints_saved"][1:]:
                base = INFER / name / f"step{step:04d}"
                comps, _ = build(base / "legacy", base / "new", base / "s6")
                prim = score(comps, primary["thresholds"], primary["stitch"])
                entry = {"primary": prim, "summary": summary(prim, names)}
                if step == protocol["shared"]["steps"]:
                    sec = score(comps, secondary["thresholds"], secondary["stitch"])
                    entry["secondary_summary"] = summary(sec, names)
                runs[name][step] = entry
    final = protocol["shared"]["steps"]
    arms = {}
    for arm in protocol["arms"]:
        seeds = [f"{arm}_s{s}" for s in protocol["shared"]["seeds"]]
        res = [runs[s][final]["primary"] for s in seeds]
        arms[arm] = {
            "versus": {r: gate_versus(res, refs[r], names, legacy) for r in refs},
            "seed_mean": {
                k: float(np.mean([runs[s][final]["summary"][k] for s in seeds]))
                for k in (
                    "pixel_precision",
                    "pixel_recall",
                    "pixel_iou",
                    "all_instance_precision50",
                    "all_instance_recall50",
                    "all_instance_f150",
                    "solid_instance_recall50",
                    "dash_instance_recall50",
                    "all_instance_f175",
                    "centerline_coverage",
                    "true_false_link_rate",
                    "stage3_rule_false_link_rate",
                )
            },
            "seed_sd_pixel_iou": float(
                np.std([runs[s][final]["summary"]["pixel_iou"] for s in seeds])
            ),
            "trajectory_pixel_iou": {
                s: {st: runs[s][st]["summary"]["pixel_iou"] for st in runs[s]} for s in seeds
            },
            "trajectory_instance_recall50": {
                s: {st: runs[s][st]["summary"]["all_instance_recall50"] for st in runs[s]}
                for s in seeds
            },
        }
        for r in refs:
            arms[arm]["versus"][r]["passes_all"] = all(arms[arm]["versus"][r]["gates"].values())
        arms[arm]["passes_both"] = all(arms[arm]["versus"][r]["passes_all"] for r in refs)

    def solid_ap(res):
        vals = [res[n]["full"]["ap"]["ap50"]["solid"] for n in names]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else 0.0

    pairs = []
    for on, off in (("B", "A"), ("D", "C")):
        for s in protocol["shared"]["seeds"]:
            a, b = runs[f"{off}_s{s}"][final], runs[f"{on}_s{s}"][final]
            pairs.append(
                {
                    "pair": f"{on}-{off} seed {s}",
                    "solid_ap50_diff": solid_ap(b["primary"]) - solid_ap(a["primary"]),
                    "pixel_iou_diff": b["summary"]["pixel_iou"] - a["summary"]["pixel_iou"],
                    "instance_recall50_diff": b["summary"]["all_instance_recall50"]
                    - a["summary"]["all_instance_recall50"],
                    "true_false_link_rate_on": b["summary"]["true_false_link_rate"],
                }
            )
    data_pairs = []
    for more, less in (("C", "A"), ("D", "B")):
        for s in protocol["shared"]["seeds"]:
            a, b = runs[f"{less}_s{s}"][final], runs[f"{more}_s{s}"][final]
            data_pairs.append(
                {
                    "pair": f"{more}-{less} seed {s}",
                    "pixel_iou_diff": b["summary"]["pixel_iou"] - a["summary"]["pixel_iou"],
                    "instance_recall50_diff": b["summary"]["all_instance_recall50"]
                    - a["summary"]["all_instance_recall50"],
                    "dash_recall50_diff": b["summary"]["dash_instance_recall50"]
                    - a["summary"]["dash_instance_recall50"],
                }
            )
    fix_verified = (
        sum(p["solid_ap50_diff"] > 0 for p in pairs) >= 5
        and sum(p["pixel_iou_diff"] > 0 for p in pairs) >= 5
        and arms["B"]["versus"]["frozen_candidate_V1_stitch"]["gates"]["G5"]
        and arms["D"]["versus"]["frozen_candidate_V1_stitch"]["gates"]["G5"]
    )
    passing = [a for a in arms if arms[a]["passes_both"]]
    selected = (
        max(passing, key=lambda a: arms[a]["seed_mean"]["all_instance_f150"]) if passing else None
    )
    blocked = [
        a
        for a in arms
        for r in refs
        if all(v for k, v in arms[a]["versus"][r]["gates"].items() if k != "G5")
        and not arms[a]["versus"][r]["gates"]["G5"]
    ]
    any_g1 = any(arms[a]["versus"]["frozen_candidate_V1_stitch"]["gates"]["G1"] for a in arms)
    targets = protocol["reliability_targets_provisional"]
    best = (
        arms[selected]["seed_mean"]
        if selected
        else summary(refs["frozen_candidate_V1_stitch"], names)
    )
    reliability = {
        "pixel_precision": [best["pixel_precision"], targets["pixel_precision"]],
        "pixel_recall": [best["pixel_recall"], targets["pixel_recall"]],
        "instance_recall_iou50": [best["all_instance_recall50"], targets["instance_recall_iou50"]],
    }
    reliability_met = all(v >= t for v, t in reliability.values())
    statuses = []
    if selected:
        statuses.append("RECOGNITION_IMPROVED_PROVISIONAL")
    if fix_verified:
        statuses.append("TRAINING_OBJECTIVE_IMPROVEMENT_VERIFIED")
    if blocked and not selected:
        statuses.append("STITCHING_SAFETY_BLOCKED")
    if not any_g1:
        statuses.append("NO_IMPROVEMENT")
    data_positive = np.mean([p["pixel_iou_diff"] for p in data_pairs]) > 0
    if not reliability_met and (data_positive or best["dash_instance_recall50"] < 0.5):
        statuses.append("MORE_REAL_DATA_REQUIRED")
    statuses.append("NEW_INDEPENDENT_TEST_REQUIRED")
    out = {
        "protocol_sha256": sha256(OUT / "7C_protocol.json"),
        "references": {
            r: {"pooled": summary(v, names), "areas": strip(v)} for r, v in refs.items()
        },
        "runs": {
            n: {
                st: {
                    "summary": e["summary"],
                    **(
                        {"secondary_summary": e["secondary_summary"]}
                        if "secondary_summary" in e
                        else {}
                    ),
                    **({"areas": strip(e["primary"])} if st == final else {}),
                }
                for st, e in v.items()
            }
            for n, v in runs.items()
        },
        "arms": arms,
        "fix_pairs": pairs,
        "data_pairs": data_pairs,
        "fix_verified": fix_verified,
        "selected_arm": selected,
        "stitching_blocked_arms": sorted(set(blocked)),
        "reliability_gate": {"values_vs_targets": reliability, "met": reliability_met},
        "statuses": statuses,
        "pool": names,
    }
    dump(OUT / "7H_results.json", out)
    print(
        json.dumps(
            {
                "statuses": statuses,
                "selected": selected,
                "fix_verified": fix_verified,
                "arms": {
                    a: {
                        "seed_mean": {k: round(v, 3) for k, v in arms[a]["seed_mean"].items()},
                        "gates": {r: arms[a]["versus"][r]["gates"] for r in refs},
                    }
                    for a in arms
                },
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
