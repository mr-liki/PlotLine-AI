"""Stage 2: thin-line supervision recipes under leave-one-region-group-out evaluation.

R0 baseline criterion; R1 bilinear logit sampling (train = inference upsampling);
R2 R1 + skeleton-recall loss (w=0.5). Each recipe: 5 folds x 2 seeds, warm start from the
default step-50 model, Phase 5.12 budget/optimizer/sampling, FINAL step used (no checkpoint
selection). Scored on the held-out fold's P511/P512 regions (unseen by the warm start).
"""

import json
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from phase5_11_common import ROOT, load
from phase5_11_export import check_phase511
from roadmarking.dataset.phase55 import write_attestation
from roadmarking.evaluation.harness import paired_block_bootstrap, summarize
from roadmarking.rasterization.pipeline import dump, sha256
from roadmarking.training.expanded_continuous import run_training
from stage1_scoring import (
    DEFAULT,
    FROZEN,
    crop_level,
    crop_nms,
    evaluate_region,
    fit_thresholds,
    load_pool,
    run_inference,
    variant_score,
)

OUT = ROOT / "outputs/reports/stage2_loro"
MODELS = ROOT / "models/mask2former_swin_s/stage2_loro"
DATA = ROOT / "data/processed/stage2_loro"
RECIPES = {
    "R0_baseline": {},
    "R1_bilinear": {"bilinear": True},
    "R2_bilinear_skeleton": {"bilinear": True, "skeleton_recall_weight": 0.5},
}
SEEDS = [895, 896]


def protocol(previous):
    path = OUT / "protocol.json"
    if path.exists():
        return load(path), sha256(path)
    body = {
        "stage": 2,
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "PREDECLARED_BEFORE_TRAINING",
        "recipes": RECIPES,
        "seeds": SEEDS,
        "folds": "stage0_harness/folds.json (5)",
        "training": {
            **{k: previous[k] for k in ("budget", "optimizer", "sampling_cycle")},
            "initialization": str(DEFAULT / "checkpoint_step0050"),
            "data": "Phase 5.12 dataset minus the held-out fold's regions",
            "checkpoint_used": "final step 600 (no selection)",
        },
        "evaluation": "held-out fold P511/P512 regions only (26 total, unseen by warm start); "
        "V0 frozen and V1 thresholds cross-fitted across folds within each "
        "recipe+seed; records summed over seeds; region-bootstrap 2000",
        "gates": {
            "G1": "crop-appearance solid AP50 diff CI lower > 0",
            "G2": "solid buffer completeness (V1) diff CI lower > 0",
            "G3": "crop-appearance dash AP50 diff CI lower > -0.02",
        },
        "adopt": "R1 if it passes G1-G3 vs R0. R2 if it passes vs R0 AND, vs R1, (G1 or G2) "
        "and G3. Otherwise keep R0.",
        "also_reported": "median V0 score of crop-level TP solids; geographic FG IoU; dash F1",
    }
    write_attestation(path, body)
    return body, sha256(path)


def fold_dataset(manifest, fold_regions, k, base):
    path = DATA / f"fold{k}"
    mpath = path / "metadata/dataset_manifest.json"
    if not mpath.exists():
        m = dict(manifest)
        m["crops"] = [
            c
            for c in manifest["crops"]
            if not (c["split"] == "train" and c["tile_id"] in fold_regions)
        ]
        m["stage2_held_out_regions"] = sorted(fold_regions)
        m["stage2_source_manifest_sha256"] = sha256(base / "metadata/dataset_manifest.json")
        dump(mpath, m)
    return path


def main():
    began = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    l12 = load(ROOT / "outputs/reports/phase5_12/latest.json")
    previous = load(Path(l12["report_dir"]) / "training_protocol.json")
    _, proto_hash = protocol(previous)
    base = Path(l12["dataset"])
    manifest = load(base / "metadata/dataset_manifest.json")
    folds = load(ROOT / "outputs/reports/stage0_harness/folds.json")["folds"]
    only_train = "train" in sys.argv
    runs = []
    for recipe, kwargs in RECIPES.items():
        for seed in SEEDS:
            for f in folds:
                k = f["fold"]
                held = set(f["regions"])
                dataset = fold_dataset(manifest, held, k, base)
                out = MODELS / f"{recipe}_s{seed}_f{k}"
                final = out / "checkpoint_step0600"
                if not (final / "bundle.json").exists():
                    tick = time.perf_counter()

                    def check(d, held=held):
                        m = load(Path(d) / "metadata/dataset_manifest.json")
                        assert not any(
                            c["split"] == "train" and c["tile_id"] in held for c in m["crops"]
                        )
                        return check_phase511(d)

                    run_training(
                        dataset,
                        DEFAULT / "checkpoint_step0050/model",
                        DEFAULT / "checkpoint_step0050/processor",
                        out,
                        FROZEN,
                        max_steps=600,
                        stop_after=600,
                        learning_rate=previous["optimizer"]["AdamW_lr"],
                        sampling_cycle=previous["sampling_cycle"],
                        dataset_check=check,
                        evaluation_interval=600,
                        initial_loss_scale=previous["optimizer"]["fp16_initial_scale"],
                        expected_baseline_counts={"fp": 132, "fn": 11},
                        extra_pools={
                            "p511_solid": lambda c: c.get("phase") in ("5.11", "5.12")
                            and 0 in c["labels"],
                            "p511_dash": lambda c: c.get("phase") in ("5.11", "5.12")
                            and 1 in c["labels"],
                        },
                        seed=seed,
                        criterion_kwargs=kwargs,
                    )
                    dump(
                        out / "stage2_run.json",
                        {
                            "seconds": time.perf_counter() - tick,
                            "recipe": recipe,
                            "seed": seed,
                            "fold": k,
                        },
                    )
                pool = [
                    c
                    for c in manifest["crops"]
                    if c["split"] == "train"
                    and c["tile_id"] in held
                    and c["tile_id"].startswith("P51")
                ]
                pred = OUT / "inference" / f"{recipe}_s{seed}_f{k}"
                run_inference(final, base, pool, pred)
                runs.append((recipe, seed, k, pool, pred))
                print(
                    f"done {recipe} seed {seed} fold {k} "
                    f"({(time.perf_counter() - began) / 60:.0f} min)",
                    flush=True,
                )
    if only_train:
        return
    # ---- evaluation
    pool_regions = {r: f["fold"] for f in folds for r in f["regions"] if r.startswith("P51")}
    results = {}
    tp_conf = defaultdict(list)
    for recipe in RECIPES:
        agg = {
            v: {
                "crop_records": defaultdict(lambda: defaultdict(float)),
                "crop_dets": defaultdict(list),
                "geo_records": defaultdict(lambda: defaultdict(float)),
                "geo_dets": defaultdict(list),
            }
            for v in ("V0_frozen", "V1_refit_thresholds")
        }
        for seed in SEEDS:
            by_region = defaultdict(list)
            for r_, s_, _k, pool, pred in runs:
                if r_ == recipe and s_ == seed:
                    for crop in load_pool(pool, pred).values():
                        by_region[crop["crop"]["tile_id"]].append(crop)
            for crops in by_region.values():
                for crop in crops:
                    s = variant_score(crop["rows"], "V0_frozen")
                    kept = crop_nms(crop["rows"], s)
                    lst, _ = crop_level(crop, s, kept)[0]
                    tp_conf[recipe] += [sc for sc, flag in lst if flag]
            for f in range(5):
                fit = [c for r, k in pool_regions.items() if k != f for c in by_region[r]]
                thr = {
                    "V0_frozen": FROZEN["confidence"],
                    "V1_refit_thresholds": fit_thresholds(fit, "V1_refit_thresholds"),
                }
                for r in [r for r, k in pool_regions.items() if k == f]:
                    for v in ("V0_frozen", "V1_refit_thresholds"):
                        cr, cd, gr, gd = evaluate_region(r, by_region[r], v, thr[v], None)
                        for key, val in cr.items():
                            agg[v]["crop_records"][r][key] += val
                        agg[v]["crop_dets"][r] += cd
                        for key, val in gr.items():
                            agg[v]["geo_records"][r][key] += val
                        agg[v]["geo_dets"][r] += gd
        results[recipe] = {
            v: {
                k: {r: (dict(x) if isinstance(x, defaultdict) else x) for r, x in d.items()}
                for k, d in a.items()
            }
            for v, a in agg.items()
        }
        print(f"scored {recipe}", flush=True)

    def boot(a, b):
        out = {}
        for v in ("V0_frozen", "V1_refit_thresholds"):
            ra, rb = results[a][v], results[b][v]
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
                    ra["geo_records"],
                    ra["geo_dets"],
                    rb["geo_records"],
                    rb["geo_dets"],
                    n=2000,
                    seed=0,
                )["metrics"],
            }
        return out

    comparisons = {
        "R1_vs_R0": boot("R0_baseline", "R1_bilinear"),
        "R2_vs_R0": boot("R0_baseline", "R2_bilinear_skeleton"),
        "R2_vs_R1": boot("R1_bilinear", "R2_bilinear_skeleton"),
    }

    def gates(c):
        return {
            "G1": c["V1_refit_thresholds"]["crop"]["solid_ap50"]["ci_difference"][0] > 0,
            "G2": c["V1_refit_thresholds"]["geo"]["solid_buffer_completeness"]["ci_difference"][0]
            > 0,
            "G3": c["V1_refit_thresholds"]["crop"]["dash_ap50"]["ci_difference"][0] > -0.02,
        }

    g = {k: gates(c) for k, c in comparisons.items()}
    adopt = "R0_baseline"
    if all(g["R1_vs_R0"].values()):
        adopt = "R1_bilinear"
    if (
        all(g["R2_vs_R0"].values())
        and (g["R2_vs_R1"]["G1"] or g["R2_vs_R1"]["G2"])
        and g["R2_vs_R1"]["G3"]
    ):
        adopt = "R2_bilinear_skeleton"
    summary = {
        recipe: {
            v: {
                "crop": summarize(
                    results[recipe][v]["crop_records"].values(),
                    results[recipe][v]["crop_dets"].values(),
                ),
                "geo": summarize(
                    results[recipe][v]["geo_records"].values(),
                    results[recipe][v]["geo_dets"].values(),
                ),
            }
            for v in ("V0_frozen", "V1_refit_thresholds")
        }
        for recipe in RECIPES
    }
    times = [
        load(MODELS / f"{r}_s{s}_f{k}/stage2_run.json")["seconds"]
        for r in RECIPES
        for s in SEEDS
        for k in range(5)
        if (MODELS / f"{r}_s{s}_f{k}/stage2_run.json").exists()
    ]
    out = {
        "stage": 2,
        "protocol_sha256": proto_hash,
        "summary": summary,
        "comparisons": comparisons,
        "gates": g,
        "adopted_recipe": adopt,
        "median_tp_solid_confidence": {
            r: float(np.median(v)) if v else None for r, v in tp_conf.items()
        },
        "train_seconds_per_run_median": float(np.median(times)) if times else None,
        "runtime_seconds": time.perf_counter() - began,
        "protected_final_test_used": False,
    }
    dump(OUT / "stage2_results.json", out)
    print(
        json.dumps(
            {
                "gates": g,
                "adopt": adopt,
                "median_tp_solid_confidence": out["median_tp_solid_confidence"],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
