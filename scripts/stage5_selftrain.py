"""Stage 5: one gated round of conservative self-training (LORO, 2 seeds).

Teacher  = each fold's Stage-4-adopted model (final step).
Unlabeled = Stage-6 verified regions S6_02..S6_10 (rasters only; S6_01 is excluded as the
            proposed fresh development region). No human labels are used for training.
Pseudo-labels: V1 score >= (cross-fitted V1 threshold + 0.05) after per-crop NMS; supervision
            limited to observed pixels within 10 px of accepted masks; other proposals with
            score >= .25 become ignore. Precision is MEASURED against the (unapproved) client
            outlines and must reach 0.80 before any student is trained.
Student  = default warm start, adopted Stage-4 recipe, the plain "dash" sampling slot replaced
            by a "pseudo" stratum. Compared with the Stage-4 adopted recipe on held-out folds.
"""

import json
import sys
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation

import loro_eval
from phase5_11_common import ROOT, load
from phase5_11_export import check_phase511
from roadmarking.dataset.phase55 import write_attestation
from roadmarking.inference.prediction_recovery import packed_iou
from roadmarking.rasterization.pipeline import dump, sha256
from roadmarking.training.augment import augment
from roadmarking.training.expanded_continuous import run_training
from stage1_scoring import (
    DEFAULT,
    FROZEN,
    crop_features,
    crop_nms,
    fit_thresholds,
    load_pool,
    run_inference,
    variant_score,
)
from stage2_loro import RECIPES, SEEDS, fold_dataset

OUT = ROOT / "outputs/reports/stage5_selftrain"
MODELS = ROOT / "models/mask2former_swin_s/stage5_selftrain"
DATA = ROOT / "data/processed/stage5_selftrain"
MARGIN = 0.05
PRECISION_TARGET = 0.80


def recipe_of(adopted):
    """(criterion kwargs, augmentation, hard-negative stratum) for a Stage-4 recipe name."""
    s2 = load(ROOT / "outputs/reports/stage2_loro/stage2_results.json")["adopted_recipe"]
    return {
        "A0": (RECIPES[s2], None, False),
        "A1": (RECIPES[s2], augment, False),
        "A2": (RECIPES[s2], augment, True),
        "A3": (RECIPES["R1_bilinear"], augment, False),
    }[adopted]


def protocol(adopted):
    path = OUT / "protocol.json"
    if path.exists():
        return sha256(path)
    write_attestation(
        path,
        {
            "stage": 5,
            "created_utc": datetime.now(UTC).isoformat(),
            "status": "PREDECLARED_BEFORE_PSEUDO_LABELLING",
            "teacher_recipe": adopted,
            "unlabeled_regions": "Stage-6 S6_02..S6_10 (S6_01 held out as fresh-dev candidate)",
            "acceptance": f"V1 score >= cross-fitted V1 threshold + {MARGIN}; observed pixels only;"
            " background only within 10 px of accepted masks; other proposals >= .25 "
            "ignored",
            "precision_gate": f"pseudo-instance precision vs client outlines >= {PRECISION_TARGET}"
            " (IoU>=.5, same class) or Stage 5 stops before training",
            "student": "same as teacher recipe; 'dash' slot -> 'pseudo' stratum; 600 steps",
            "gates_vs_teacher_recipe": {
                "G1": "crop dash AP50 or solid AP50 diff CI lower > 0",
                "G2": "the other AP50 diff CI lower > -0.02",
                "G3": "solid buffer completeness (V1) CI lower > -0.02",
            },
            "rounds": "one; a second round only if round one passes (roadmap: stop otherwise)",
        },
    )
    return sha256(path)


def unlabeled_crops():
    s6 = load(
        Path(load(ROOT / "outputs/reports/stage6_review/latest.json")["report_dir"])
        / "new_region_candidates.json"
    )
    store = s6["candidate_store"]
    paint = gpd.read_file(store, layer="candidate_paint_footprints")
    crops = []
    for t in s6["tiles"]:
        if t["proposed_role"] != "TRAINING_CANDIDATE":
            continue
        with rasterio.open(t["layers"]["observed"]) as ds:
            observed, transform = ds.read(1).astype(bool), ds.transform
        rgb = np.asarray(Image.open(t["images"]["grayscale"]).convert("RGB"))
        rows = paint[paint.candidate_id == t["candidate_id"]]
        client = [
            (
                rasterize([(g, 1)], observed.shape, transform=transform).astype(bool) & observed,
                int(v) - 1,
            )
            for g, v in zip(rows.geometry, rows.visual_id, strict=True)
        ]
        for r in (0, 256, 512):
            for c in (0, 256, 512):
                v = observed[r : r + 512, c : c + 512]
                if v.mean() < 0.1:
                    continue
                path = DATA / "unlabeled" / f"{t['candidate_id']}_r{r}_c{c}.npz"
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        path,
                        rgb=rgb[r : r + 512, c : c + 512],
                        masks=np.zeros((0, 512, 512), bool),
                        labels=np.zeros(0, int),
                        validity=v,
                    )
                crops.append(
                    {
                        "crop_id": path.stem,
                        "file": str(path),
                        "sha256": sha256(path),
                        "tile_id": t["candidate_id"],
                        "size": 512,
                        "split": "unlabeled",
                        "window": [r, c, 512, 512],
                        "global_instance_ids": [],
                        "labels": [],
                        "client": [
                            (m[r : r + 512, c : c + 512], k)
                            for m, k in client
                            if m[r : r + 512, c : c + 512].sum() >= 4
                        ],
                    }
                )
    return crops


def pseudo_label(crops, pred_folder, thr, tag):
    made, tp, n_acc = [], 0, 0
    for crop in crops:
        with np.load(crop["file"]) as d:
            rgb, valid = d["rgb"], d["validity"].astype(bool)
        rows = crop_features(Path(pred_folder) / (crop["crop_id"] + ".npz"), valid)
        s = variant_score(rows, "V1_refit_thresholds")
        kept = crop_nms(rows, s)
        acc = [k for k in kept if s[k] >= min(0.99, thr[str(rows[k]["class"])] + MARGIN)]
        if not acc:
            continue
        masks = np.stack([rows[k]["mask"] & valid for k in acc])
        labels = np.array([rows[k]["class"] for k in acc])
        if crop["client"]:
            cm = np.stack([m for m, _ in crop["client"]])
            cl = np.array([k for _, k in crop["client"]])
            iou = packed_iou(masks, cm, valid).reshape(len(acc), len(cl))
            for i in range(len(acc)):
                same = cl == labels[i]
                tp += int(same.any() and iou[i, same].max() >= 0.5)
        n_acc += len(acc)
        support = binary_dilation(masks.any(0), iterations=10) & valid
        uncertain = np.zeros_like(valid)
        for k in kept:
            if k not in acc and s[k] >= 0.25:
                uncertain |= rows[k]["mask"]
        v2 = support & ~(uncertain & ~masks.any(0))
        path = DATA / tag / (crop["crop_id"] + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, rgb=rgb, masks=masks & v2, labels=labels, validity=v2)
        made.append(
            {
                "crop_id": f"{tag}_{crop['crop_id']}",
                "file": str(path),
                "sha256": sha256(path),
                "tile_id": crop["tile_id"],
                "size": 512,
                "split": "train",
                "phase": "pseudo",
                "window": crop["window"],
                "global_instance_ids": [
                    f"pseudo:{tag}:{crop['crop_id']}:{i}" for i in range(len(acc))
                ],
                "labels": labels.tolist(),
            }
        )
    return made, (tp / n_acc if n_acc else 0.0), n_acc


def main():
    began = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    s4 = load(ROOT / "outputs/reports/stage4_loro/stage4_results.json")
    adopted = s4["adopted"]
    proto_hash = protocol(adopted)
    kwargs, aug, hard_flag = recipe_of(adopted)
    s2_recipe = load(ROOT / "outputs/reports/stage2_loro/stage2_results.json")["adopted_recipe"]
    teacher_dir = {"A0": ("stage2_loro", s2_recipe)}.get(adopted, ("stage4_loro", adopted))
    l12 = load(ROOT / "outputs/reports/phase5_12/latest.json")
    previous = load(Path(l12["report_dir"]) / "training_protocol.json")
    root = Path(l12["dataset"])
    manifest = load(root / "metadata/dataset_manifest.json")
    folds = load(ROOT / "outputs/reports/stage0_harness/folds.json")["folds"]
    train = [c for c in manifest["crops"] if c["split"] == "train"]
    unl = unlabeled_crops()
    unl_public = [{k: v for k, v in c.items() if k != "client"} for c in unl]
    gate_log, teacher_runs, student_runs = {}, [], []
    for seed in SEEDS:
        # Teacher thresholds: V1 fitted on all held-out pool predictions of this recipe/seed.
        pool_all = []
        for f in folds:
            pool = [
                c
                for c in train
                if c["tile_id"] in set(f["regions"]) and c["tile_id"].startswith("P51")
            ]
            pred = (
                ROOT
                / "outputs/reports"
                / teacher_dir[0]
                / "inference"
                / f"{teacher_dir[1]}_s{seed}_f{f['fold']}"
            )
            teacher_runs.append((seed, f["fold"], pool, pred))
            pool_all += list(load_pool(pool, pred).values())
        thr = fit_thresholds(pool_all, "V1_refit_thresholds")
        for f in folds:
            k, held = f["fold"], set(f["regions"])
            teacher = (
                ROOT
                / "models/mask2former_swin_s"
                / teacher_dir[0]
                / f"{teacher_dir[1]}_s{seed}_f{k}/checkpoint_step0600"
            )
            upred = OUT / "teacher_unlabeled" / f"s{seed}_f{k}"
            run_inference(teacher, root, unl_public, upred)
            pseudo, precision, n = pseudo_label(unl, upred, thr, f"s{seed}_f{k}")
            gate_log[f"s{seed}_f{k}"] = {
                "pseudo_crops": len(pseudo),
                "accepted_instances": n,
                "precision_vs_client": precision,
            }
            dump(OUT / "pseudo_label_gate.json", gate_log)
            if precision < PRECISION_TARGET:
                print(f"STOP: pseudo precision {precision:.3f} < {PRECISION_TARGET}", flush=True)
                dump(
                    OUT / "stage5_results.json",
                    {
                        "stage": 5,
                        "protocol_sha256": proto_hash,
                        "status": "STOPPED_AT_GATE",
                        "pseudo_label_gate": gate_log,
                        "protected_final_test_used": False,
                    },
                )
                return
            dataset = fold_dataset(manifest, held, k, root)
            sdata = DATA / f"student_s{seed}_f{k}"
            mpath = sdata / "metadata/dataset_manifest.json"
            if not mpath.exists():
                m = load(dataset / "metadata/dataset_manifest.json")
                m["crops"] = m["crops"] + pseudo
                dump(mpath, m)
            fold_train = [c for c in train if c["tile_id"] not in held]
            cycle = ["pseudo" if s == "dash" else s for s in previous["sampling_cycle"]]
            if hard_flag:
                cycle = ["hard_negative" if s == "verified_negative" else s for s in cycle]
            bright = {}
            if hard_flag:
                from stage4_loro import bright_background_fraction

                bright = {c["crop_id"]: bright_background_fraction(c) for c in fold_train}
                cut = np.percentile(list(bright.values()), 75)
                hard = {c for c, v in bright.items() if v >= cut}
            else:
                hard = set()
            out = MODELS / f"student_s{seed}_f{k}"
            final = out / "checkpoint_step0600"
            if not (final / "bundle.json").exists():

                def check(d, held=held):
                    mm = load(Path(d) / "metadata/dataset_manifest.json")
                    assert not any(
                        c["split"] == "train" and c["tile_id"] in held for c in mm["crops"]
                    )
                    return check_phase511(d)

                run_training(
                    sdata,
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
                        "pseudo": lambda c: c.get("phase") == "pseudo",
                        "hard_negative": lambda c, hard=hard: c["crop_id"] in hard,
                    },
                    seed=seed,
                    criterion_kwargs=kwargs,
                    augment_fn=partial(aug) if aug else None,
                )
            pool = [c for c in train if c["tile_id"] in held and c["tile_id"].startswith("P51")]
            pred = OUT / "inference" / f"student_s{seed}_f{k}"
            run_inference(final, root, pool, pred)
            student_runs.append((seed, k, pool, pred))
            print(
                f"done student seed {seed} fold {k} "
                f"({(time.perf_counter() - began) / 60:.0f} min)",
                flush=True,
            )
    if "train" in sys.argv:
        return
    t_scored = loro_eval.score_recipe(teacher_runs)
    s_scored = loro_eval.score_recipe(student_runs)
    comp = loro_eval.compare(t_scored, s_scored)
    v = comp["V1_refit_thresholds"]
    d_ap = v["crop"]["dash_ap50"]["ci_difference"]
    s_ap = v["crop"]["solid_ap50"]["ci_difference"]
    gates = {
        "G1": d_ap[0] > 0 or s_ap[0] > 0,
        "G2": (s_ap[0] > -0.02) if d_ap[0] > 0 else (d_ap[0] > -0.02),
        "G3": v["geo"]["solid_buffer_completeness"]["ci_difference"][0] > -0.02,
    }
    out = {
        "stage": 5,
        "protocol_sha256": proto_hash,
        "teacher_recipe": adopted,
        "pseudo_label_gate": gate_log,
        "summary": {"teacher": loro_eval.summary(t_scored), "student": loro_eval.summary(s_scored)},
        "comparison_student_vs_teacher": comp,
        "gates": gates,
        "adopt_self_training": all(gates.values()),
        "second_round": (
            "not run (one round per protocol)"
            if not all(gates.values())
            else "eligible; not run in this pass"
        ),
        "unlabeled_crops": len(unl),
        "runtime_seconds": time.perf_counter() - began,
        "protected_final_test_used": False,
    }
    dump(OUT / "stage5_results.json", out)
    print(json.dumps({"gates": gates, "gate_log": gate_log}, indent=1))


if __name__ == "__main__":
    main()
