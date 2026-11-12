"""Stage 4: appearance augmentation and hard negatives under leave-one-region-out.

A0 = the Stage-2 adopted recipe (its LORO runs are reused, not retrained).
A1 = A0 + joint augmentation (8 dihedral orientations, intensity gain/gamma/gradient on
     valid pixels, simulated NoData blobs and one-sided density loss).
A3 = R1 bilinear sampling + A1 augmentation (Stage 2 neutral-positive fix, retested here).
A2 = A1 + hard-negative oversampling (the verified-negative stratum draws instead from the
     top-quartile bright-background training crops of the fold).
Real-dash copy-paste is NOT included (no reliable road-surface mask; risk of pasting onto
vegetation) — recorded as a limitation.
"""

import json
import sys
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np

import loro_eval
from phase5_11_common import ROOT, load
from phase5_11_export import check_phase511
from roadmarking.dataset.phase55 import write_attestation
from roadmarking.rasterization.pipeline import dump, sha256
from roadmarking.training.augment import augment
from roadmarking.training.expanded_continuous import run_training
from stage1_scoring import DEFAULT, FROZEN, run_inference
from stage2_loro import RECIPES, SEEDS, fold_dataset

OUT = ROOT / "outputs/reports/stage4_loro"
MODELS = ROOT / "models/mask2former_swin_s/stage4_loro"


def bright_background_fraction(crop):
    with np.load(crop["file"]) as d:
        gray = d["rgb"][..., 0]
        valid = d["validity"].astype(bool)
        fg = d["masks"].astype(bool).any(0) if len(d["masks"]) else np.zeros_like(valid)
    bg = valid & ~fg
    return float((gray[bg] >= 160).mean()) if bg.any() else 0.0


def protocol(base_recipe):
    path = OUT / "protocol.json"
    if path.exists():
        return sha256(path)
    write_attestation(
        path,
        {
            "stage": 4,
            "created_utc": datetime.now(UTC).isoformat(),
            "status": "PREDECLARED_BEFORE_TRAINING",
            "base_recipe": base_recipe,
            "recipes": {
                "A0": "Stage-2 adopted recipe (reused runs)",
                "A1": "A0 + augment(dihedral, intensity, nodata)",
                "A2": "A1 + hard-negative stratum (top-quartile bright-background crops)",
                "A3": "R1 bilinear sampling + A1 augmentation (Stage-2 neutral-positive fix)",
            },
            "seeds": SEEDS,
            "training": "identical to Stage 2 otherwise",
            "gates": {
                "G1": "crop dash AP50 diff CI lower > 0",
                "G2": "geographic dash FP (V1) diff CI upper < 0",
                "G3": "crop solid AP50 diff CI lower > -0.02",
                "G4": "solid buffer completeness (V1) diff CI lower > -0.02",
            },
            "adopt": "Candidates must pass (G1 or G2) and G3 and G4 vs A0. Simplicity order "
            "A1, A3, A2: a later recipe replaces an earlier passing one only if it also passes "
            "vs that recipe. If none passes, keep A0.",
            "limitations": ["no real-dash copy-paste"],
        },
    )
    return sha256(path)


def gates(c):
    v = c["V1_refit_thresholds"]
    return {
        "G1": v["crop"]["dash_ap50"]["ci_difference"][0] > 0,
        "G2": v["geo"]["dash_fp"]["ci_difference"][1] < 0 if "dash_fp" in v["geo"] else False,
        "G3": v["crop"]["solid_ap50"]["ci_difference"][0] > -0.02,
        "G4": v["geo"]["solid_buffer_completeness"]["ci_difference"][0] > -0.02,
    }


def passes(g):
    return (g["G1"] or g["G2"]) and g["G3"] and g["G4"]


def main():
    began = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    s2 = load(ROOT / "outputs/reports/stage2_loro/stage2_results.json")
    base = s2["adopted_recipe"]
    proto_hash = protocol(base)
    l12 = load(ROOT / "outputs/reports/phase5_12/latest.json")
    previous = load(Path(l12["report_dir"]) / "training_protocol.json")
    dataset_root = Path(l12["dataset"])
    manifest = load(dataset_root / "metadata/dataset_manifest.json")
    folds = load(ROOT / "outputs/reports/stage0_harness/folds.json")["folds"]
    train = [c for c in manifest["crops"] if c["split"] == "train"]
    bright = {c["crop_id"]: bright_background_fraction(c) for c in train}
    runs = {"A0": [], "A1": [], "A2": [], "A3": []}
    for seed in SEEDS:
        for f in folds:
            k, held = f["fold"], set(f["regions"])
            pool = [c for c in train if c["tile_id"] in held and c["tile_id"].startswith("P51")]
            runs["A0"].append(
                (
                    seed,
                    k,
                    pool,
                    ROOT / "outputs/reports/stage2_loro/inference" / f"{base}_s{seed}_f{k}",
                )
            )
            dataset = fold_dataset(manifest, held, k, dataset_root)
            fold_train = [c for c in train if c["tile_id"] not in held]
            cut = np.percentile([bright[c["crop_id"]] for c in fold_train], 75)
            hard = {c["crop_id"] for c in fold_train if bright[c["crop_id"]] >= cut}
            for name in ("A1", "A2", "A3"):
                out = MODELS / f"{name}_s{seed}_f{k}"
                final = out / "checkpoint_step0600"
                cycle = list(previous["sampling_cycle"])
                if name == "A2":
                    cycle = ["hard_negative" if s == "verified_negative" else s for s in cycle]
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
                        sampling_cycle=cycle,
                        dataset_check=check,
                        evaluation_interval=600,
                        initial_loss_scale=previous["optimizer"]["fp16_initial_scale"],
                        expected_baseline_counts={"fp": 132, "fn": 11},
                        extra_pools={
                            "p511_solid": lambda c: c.get("phase") in ("5.11", "5.12")
                            and 0 in c["labels"],
                            "p511_dash": lambda c: c.get("phase") in ("5.11", "5.12")
                            and 1 in c["labels"],
                            "hard_negative": lambda c, hard=hard: c["crop_id"] in hard,
                        },
                        seed=seed,
                        criterion_kwargs=RECIPES["R1_bilinear"] if name == "A3" else RECIPES[base],
                        augment_fn=partial(augment),
                    )
                    dump(out / "stage4_run.json", {"seconds": time.perf_counter() - tick})
                pred = OUT / "inference" / f"{name}_s{seed}_f{k}"
                run_inference(final, dataset_root, pool, pred)
                runs[name].append((seed, k, pool, pred))
                print(
                    f"done {name} seed {seed} fold {k} "
                    f"({(time.perf_counter() - began) / 60:.0f} min)",
                    flush=True,
                )
    if "train" in sys.argv:
        return
    scored = {name: loro_eval.score_recipe(r) for name, r in runs.items()}
    pairs = [("A0", "A1"), ("A0", "A2"), ("A0", "A3"), ("A1", "A3"), ("A1", "A2"), ("A3", "A2")]
    comparisons = {f"{b}_vs_{a}": loro_eval.compare(scored[a], scored[b]) for a, b in pairs}
    g = {k: gates(c) for k, c in comparisons.items()}
    adopt = "A0"
    for cand in ("A1", "A3", "A2"):
        if passes(g[f"{cand}_vs_A0"]) and (adopt == "A0" or passes(g[f"{cand}_vs_{adopt}"])):
            adopt = cand
    out = {
        "stage": 4,
        "protocol_sha256": proto_hash,
        "base_recipe": base,
        "summary": {n: loro_eval.summary(s) for n, s in scored.items()},
        "median_tp_solid_confidence": {
            n: s["median_tp_solid_confidence"] for n, s in scored.items()
        },
        "comparisons": comparisons,
        "gates": g,
        "adopted": adopt,
        "runtime_seconds": time.perf_counter() - began,
        "protected_final_test_used": False,
    }
    dump(OUT / "stage4_results.json", out)
    print(json.dumps({"gates": g, "adopted": adopt}, indent=1))


if __name__ == "__main__":
    main()
