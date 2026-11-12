"""Stage 3: evidence-constrained stitching vs current filtering, on held-out pool predictions.

Usage: python stage3_stitching.py default            (default model, Stage 1 pool inference)
       python stage3_stitching.py <stage2 recipe>    (LORO predictions, both seeds)
Thresholds: V1 (class prob x mean quality) cross-fitted by fold, applied AFTER merging.
"""

import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from phase5_11_common import ROOT, load
from roadmarking.dataset.phase55 import write_attestation
from roadmarking.evaluation.harness import block_records, paired_block_bootstrap, summarize
from roadmarking.inference.prediction_recovery import filter_geographic, packed_iou
from roadmarking.inference.stitching import stitch
from roadmarking.rasterization.pipeline import dump, sha256
from stage1_scoring import crop_nms, fit_thresholds, load_pool, variant_score

OUT = ROOT / "outputs/reports/stage3_stitching"


def protocol():
    path = OUT / "protocol.json"
    if path.exists():
        return sha256(path)
    write_attestation(
        path,
        {
            "stage": 3,
            "created_utc": datetime.now(UTC).isoformat(),
            "status": "PREDECLARED_BEFORE_SCORING",
            "compare": "P_filter (per-crop NMS + global same-class NMS .3, thresholds before "
            "merging) vs P_stitch (per-crop NMS, then in-strip IoS>=.3 mutual-best "
            "solid merging with <=15 deg agreement, observed-only gap links <=20 px / "
            "<=10 deg / <=3 px lateral, dash central-copy dedupe IoS>=.5, thresholds "
            "AFTER merging)",
            "score": "V1 (class prob x mean sigmoid), per-class thresholds cross-fitted by fold",
            "parameters_fixed_from_literature_not_tuned": True,
            "gates": {
                "G1": "solid pieces per covered GT line diff CI upper < 0",
                "G2": "solid buffer completeness diff CI lower > 0",
                "G3": "false links <= 5% of output solid instances",
                "G4": "geographic dash F1 diff CI lower > -0.02",
                "G5": "negative-control FP unchanged or lower (checked in final stage)",
            },
            "no_go": "keep P_filter",
        },
    )
    return sha256(path)


def region_eval(crops, thr):
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
    targets = np.stack([truths[u] for u in ids]) if ids else np.zeros((0, *shape), bool)
    tl = np.asarray([labels[u] for u in ids], int)
    comp = {"targets": targets, "target_labels": tl, "valid": valid}
    masks_a = np.stack(masks) if masks else np.zeros((0, *shape), bool)
    scores_a, classes_a = np.asarray(scores), np.asarray(classes, int)
    iou = packed_iou(masks_a, targets, valid).reshape(len(scores), len(ids))
    # P_filter: existing two-stage NMS with thresholds before merging.
    overlaps = (
        packed_iou(masks_a, masks_a, np.ones(shape, bool)) if len(masks_a) else np.zeros((0, 0))
    )
    keep = filter_geographic(
        scores_a,
        classes_a,
        np.ones(len(scores)),
        overlaps,
        crop_ids,
        {"confidence": thr, "mask_quality": 0.0, "nms_iou": 0.3},
    )
    out = {"filter": block_records(comp, masks_a, scores_a, classes_a, keep, iou)}
    # P_stitch: merge first (no threshold), then threshold.
    windows = {c["crop_id"]: pos[c["crop_id"]] for c in cs}
    inst, audit = stitch(
        list(masks_a), scores, classes, crop_ids, windows, valid, {"0": -1.0, "1": -1.0}
    )
    if inst:
        sm = np.stack([i["mask"] for i in inst])
        ss = np.asarray([i["score"] for i in inst])
        sc = np.asarray([i["class"] for i in inst], int)
    else:
        sm, ss, sc = np.zeros((0, *shape), bool), np.zeros(0), np.zeros(0, int)
    siou = packed_iou(sm, targets, valid).reshape(len(ss), len(ids))
    skeep = [k for k in range(len(ss)) if ss[k] >= thr[str(sc[k])]]
    out["stitch"] = block_records(comp, sm, ss, sc, skeep, siou)
    # False links: an output solid whose pixels overlap two solid GT IDs (>=10% each).
    solid_gt = np.where(tl == 0)[0]
    false_links = 0
    for k in skeep:
        if sc[k] != 0:
            continue
        px = max(1, int((sm[k] & valid).sum()))
        hits = sum(int((sm[k] & targets[j]).sum()) >= 0.1 * px for j in solid_gt)
        false_links += hits >= 2
    audit.update(false_links=false_links, output_solids=int(sum(sc[k] == 0 for k in skeep)))
    return out, audit


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "default"
    OUT.mkdir(parents=True, exist_ok=True)
    proto_hash = protocol()
    folds = load(ROOT / "outputs/reports/stage0_harness/folds.json")["folds"]
    l12 = load(ROOT / "outputs/reports/phase5_12/latest.json")
    manifest = load(Path(l12["dataset"]) / "metadata/dataset_manifest.json")
    pool_regions = {r: f["fold"] for f in folds for r in f["regions"] if r.startswith("P51")}
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
        sets = []
        for seed in (895, 896):
            for f in folds:
                pool = [
                    c
                    for c in manifest["crops"]
                    if c["split"] == "train"
                    and c["tile_id"] in set(f["regions"])
                    and c["tile_id"].startswith("P51")
                ]
                sets.append(
                    (
                        f"s{seed}",
                        pool,
                        ROOT
                        / f"outputs/reports/stage2_loro/inference/{source}_s{seed}_f{f['fold']}",
                    )
                )
    records = {
        "filter": (defaultdict(lambda: defaultdict(float)), defaultdict(list)),
        "stitch": (defaultdict(lambda: defaultdict(float)), defaultdict(list)),
    }
    audits = defaultdict(float)
    by_tag = defaultdict(lambda: defaultdict(list))
    for tag, pool, folder in sets:
        for crop in load_pool(pool, folder).values():
            by_tag[tag][crop["crop"]["tile_id"]].append(crop)
    for tag, regions in by_tag.items():
        for f in range(5):
            fit = [c for r, k in pool_regions.items() if k != f for c in regions.get(r, [])]
            thr = fit_thresholds(fit, "V1_refit_thresholds")
            for r in [r for r, k in pool_regions.items() if k == f and r in regions]:
                res, audit = region_eval(regions[r], thr)
                for mode, (recs, dets) in res.items():
                    for vals in recs.values():
                        for key, val in vals.items():
                            records[mode][0][r][key] += val
                    for blk_dets in dets.values():
                        records[mode][1][r] += blk_dets
                for k, v in audit.items():
                    audits[k] += v
        print(f"scored {tag}", flush=True)
    fr, fd = records["filter"]
    sr, sd = records["stitch"]
    fr = {k: dict(v) for k, v in fr.items()}
    sr = {k: dict(v) for k, v in sr.items()}
    boot = paired_block_bootstrap(fr, dict(fd), sr, dict(sd), n=2000, seed=0)["metrics"]
    summ = {
        "filter": summarize(fr.values(), fd.values()),
        "stitch": summarize(sr.values(), sd.values()),
    }
    for m in summ.values():
        c, k = m["solid_buffer_completeness"], m["solid_buffer_correctness"]
        m["solid_buffer_quality"] = c * k / max(1e-12, c + k - c * k)
    gates = {
        "G1": (
            boot["solid_pieces_per_covered_gt"]["ci_difference"][1] < 0
            if "solid_pieces_per_covered_gt" in boot
            else False
        ),
        "G2": boot["solid_buffer_completeness"]["ci_difference"][0] > 0,
        "G3": audits["false_links"] <= 0.05 * max(1, audits["output_solids"]),
        "G4": boot["dash_f1"]["ci_difference"][0] > -0.02,
    }
    result = {
        "stage": 3,
        "source": source,
        "protocol_sha256": proto_hash,
        "summary": summ,
        "bootstrap_stitch_vs_filter": boot,
        "audit_totals": dict(audits),
        "gates": gates,
        "adopt_stitching": all(gates.values()),
        "protected_final_test_used": False,
    }
    dump(OUT / f"stage3_results_{source}.json", result)
    print(json.dumps({"summary": summ, "gates": gates, "audit": dict(audits)}, indent=1))


if __name__ == "__main__":
    main()
