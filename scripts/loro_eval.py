"""Shared leave-one-region-out scoring (identical definitions to Stage 2)."""

from collections import defaultdict

import numpy as np

from phase5_11_common import ROOT, load
from roadmarking.evaluation.harness import paired_block_bootstrap, summarize
from stage1_scoring import (
    FROZEN,
    crop_level,
    crop_nms,
    evaluate_region,
    fit_thresholds,
    load_pool,
    variant_score,
)

VARIANTS = ("V0_frozen", "V1_refit_thresholds")


def score_recipe(runs):
    """runs: list of (seed, fold, pool_crops, prediction_folder) for one recipe."""
    folds = load(ROOT / "outputs/reports/stage0_harness/folds.json")["folds"]
    pool_regions = {r: f["fold"] for f in folds for r in f["regions"] if r.startswith("P51")}
    agg = {
        v: {
            "crop_records": defaultdict(lambda: defaultdict(float)),
            "crop_dets": defaultdict(list),
            "geo_records": defaultdict(lambda: defaultdict(float)),
            "geo_dets": defaultdict(list),
        }
        for v in VARIANTS
    }
    tp_conf = []
    for seed in sorted({r[0] for r in runs}):
        by_region = defaultdict(list)
        for s, _k, pool, pred in runs:
            if s == seed:
                for crop in load_pool(pool, pred).values():
                    by_region[crop["crop"]["tile_id"]].append(crop)
        for crops in by_region.values():
            for crop in crops:
                s_ = variant_score(crop["rows"], "V0_frozen")
                lst, _ = crop_level(crop, s_, crop_nms(crop["rows"], s_))[0]
                tp_conf += [sc for sc, flag in lst if flag]
        for f in range(5):
            fit = [c for r, k in pool_regions.items() if k != f for c in by_region[r]]
            thr = {
                "V0_frozen": FROZEN["confidence"],
                "V1_refit_thresholds": fit_thresholds(fit, "V1_refit_thresholds"),
            }
            for r in [r for r, k in pool_regions.items() if k == f]:
                for v in VARIANTS:
                    cr, cd, gr, gd = evaluate_region(r, by_region[r], v, thr[v], None)
                    for key, val in cr.items():
                        agg[v]["crop_records"][r][key] += val
                    agg[v]["crop_dets"][r] += cd
                    for key, val in gr.items():
                        agg[v]["geo_records"][r][key] += val
                    agg[v]["geo_dets"][r] += gd
    result = {
        v: {
            k: {r: (dict(x) if isinstance(x, defaultdict) else x) for r, x in d.items()}
            for k, d in a.items()
        }
        for v, a in agg.items()
    }
    result["median_tp_solid_confidence"] = float(np.median(tp_conf)) if tp_conf else None
    return result


def compare(a, b):
    out = {}
    for v in VARIANTS:
        ra, rb = a[v], b[v]
        out[v] = {
            "crop": paired_block_bootstrap(
                ra["crop_records"],
                ra["crop_dets"],
                rb["crop_records"],
                rb["crop_dets"],
                n=2000,
                seed=0,
            )["metrics"],
            "geo": paired_block_bootstrap(
                ra["geo_records"], ra["geo_dets"], rb["geo_records"], rb["geo_dets"], n=2000, seed=0
            )["metrics"],
        }
    return out


def summary(result):
    return {
        v: {
            "crop": summarize(result[v]["crop_records"].values(), result[v]["crop_dets"].values()),
            "geo": summarize(result[v]["geo_records"].values(), result[v]["geo_dets"].values()),
        }
        for v in VARIANTS
    }
