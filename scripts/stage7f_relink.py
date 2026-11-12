"""Stage 7F: re-break-down the Stage-3 false-link rates by physical identity.

Rebuilds exactly the Stage-3 P_stitch outputs (same held-out pools, same cross-fitted V1
thresholds, same stitching parameters) and classifies every output touching several
client IDs with roadmarking.evaluation.link_audit. Nothing is re-tuned.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from phase5_11_common import ROOT, load
from roadmarking.evaluation.link_audit import classify_outputs, physical_lines
from roadmarking.inference.stitching import stitch
from roadmarking.rasterization.pipeline import dump
from stage1_scoring import crop_nms, fit_thresholds, load_pool, variant_score

OUT = ROOT / "outputs/reports/stage7"
SOURCES = ("default", "R0_baseline", "R1_bilinear")
KINDS = (
    "same_physical_line",
    "dashes_joined",
    "solid_dash_joined",
    "parallel_lines_joined",
    "different_lines_joined",
)


def stitched(crops, thr):
    """Stage-3 P_stitch output for one region (identical construction)."""
    cs = [c["crop"] for c in crops]
    x0 = min(c["affine"][2] for c in cs)
    y1 = max(c["affine"][5] for c in cs)
    pos = {
        c["crop_id"]: (round((y1 - c["affine"][5]) / 0.05), round((c["affine"][2] - x0) / 0.05))
        for c in cs
    }
    shape = (max(r for r, _ in pos.values()) + 512, max(c for _, c in pos.values()) + 512)
    valid = np.zeros(shape, bool)
    truths, labels = {}, {}
    masks, scores, classes, crop_ids = [], [], [], []
    for crop in crops:
        c = crop["crop"]
        r0, c0 = pos[c["crop_id"]]
        sl = (slice(r0, r0 + 512), slice(c0, c0 + 512))
        valid[sl] |= crop["valid"]
        for m, lab, uid in zip(crop["targets"], crop["tl"], c["global_instance_ids"], strict=True):
            truths.setdefault(uid, np.zeros(shape, bool))[sl] |= m
            labels[uid] = int(lab)
        s = variant_score(crop["rows"], "V1_refit_thresholds")
        for k in crop_nms(crop["rows"], s):
            g = np.zeros(shape, bool)
            g[sl] = crop["rows"][k]["mask"]
            masks.append(g)
            scores.append(float(s[k]))
            classes.append(crop["rows"][k]["class"])
            crop_ids.append(c["crop_id"])
    ids = sorted(truths)
    targets = np.stack([truths[u] & valid for u in ids]) if ids else np.zeros((0, *shape), bool)
    tl = np.asarray([labels[u] for u in ids], int)
    inst, _ = stitch(
        masks,
        scores,
        classes,
        crop_ids,
        {c["crop_id"]: pos[c["crop_id"]] for c in cs},
        valid,
        {"0": -1.0, "1": -1.0},
    )
    keep = [k for k, i in enumerate(inst) if i["score"] >= thr[str(i["class"])]]
    sm = np.stack([i["mask"] for i in inst]) if inst else np.zeros((0, *shape), bool)
    sc = np.asarray([i["class"] for i in inst], int)
    merged = [k for k in keep if len(inst[k]["members"]) > 1]
    return sm, sc, keep, targets, tl, valid, ids, merged


def run(source, folds, manifest, pool_regions):
    if source == "default":
        sets = [
            (
                "default",
                [
                    c
                    for c in manifest["crops"]
                    if c["split"] == "train" and c["tile_id"] in pool_regions
                ],
                ROOT / "outputs/reports/stage1_scoring/default_pool_inference",
            )
        ]
    else:
        sets = [
            (
                f"s{seed}",
                [
                    c
                    for c in manifest["crops"]
                    if c["split"] == "train"
                    and c["tile_id"] in set(f["regions"])
                    and c["tile_id"].startswith("P51")
                ],
                ROOT / f"outputs/reports/stage2_loro/inference/{source}_s{seed}_f{f['fold']}",
            )
            for seed in (895, 896)
            for f in folds
        ]
    by_tag = defaultdict(lambda: defaultdict(list))
    for tag, pool, folder in sets:
        for crop in load_pool(pool, folder).values():
            by_tag[tag][crop["crop"]["tile_id"]].append(crop)
    totals = defaultdict(int)
    examples = []
    for tag, regions in by_tag.items():
        for f in range(5):
            fit = [c for r, k in pool_regions.items() if k != f for c in regions.get(r, [])]
            thr = fit_thresholds(fit, "V1_refit_thresholds")
            for r in [r for r, k in pool_regions.items() if k == f and r in regions]:
                sm, sc, keep, targets, tl, valid, ids, merged = stitched(regions[r], thr)
                lines = physical_lines(targets, tl)
                recs, counts = classify_outputs(sm, sc, keep, targets, tl, valid, lines)
                for k, v in counts.items():
                    totals[k] += v
                totals["output_solids"] += int(sum(sc[k] == 0 for k in keep))
                totals["output_instances"] += len(keep)
                totals["stitched_outputs"] += len(merged)
                totals["client_ids"] += len(ids)
                totals["physical_lines"] += len(set(lines))
                for x in recs:
                    if x["kind"] != "same_physical_line":
                        examples.append({"run": tag, "region": r, **x})
        print(f"{source} {tag}", flush=True)
    t = dict(totals)
    solids = max(1, t["output_solids"])
    t["stage3_rule_rate"] = t.get("stage3_rule_false_links", 0) / solids
    t["true_false_link_rate_of_outputs"] = t.get("true_false_links", 0) / max(
        1, t["output_instances"]
    )
    t["solid_join_error_rate"] = (
        sum(
            t.get(k, 0)
            for k in ("parallel_lines_joined", "different_lines_joined", "solid_dash_joined")
        )
        / solids
    )
    return t, examples


def main():
    folds = load(ROOT / "outputs/reports/stage0_harness/folds.json")["folds"]
    l12 = load(ROOT / "outputs/reports/phase5_12/latest.json")
    manifest = load(Path(l12["dataset"]) / "metadata/dataset_manifest.json")
    pool_regions = {r: f["fold"] for f in folds for r in f["regions"] if r.startswith("P51")}
    out = {}
    for source in SOURCES:
        totals, examples = run(source, folds, manifest, pool_regions)
        stage3 = load(OUT.parent / f"stage3_stitching/stage3_results_{source}.json")
        out[source] = {
            "breakdown": totals,
            "errors": examples,
            "stage3_reported": stage3.get("audit_totals"),
        }
    dump(OUT / "7F_stage3_link_breakdown.json", out)
    print(
        json.dumps(
            {
                k: {
                    kk: v["breakdown"].get(kk)
                    for kk in (
                        *KINDS,
                        "stage3_rule_false_links",
                        "stage3_rule_rate",
                        "true_false_links",
                        "true_false_link_rate_of_outputs",
                        "solid_join_error_rate",
                        "nodata_bridges",
                        "output_solids",
                    )
                }
                for k, v in out.items()
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
