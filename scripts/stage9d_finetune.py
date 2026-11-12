"""Stage 9D-D/E/F/I: one predeclared, bounded three-class fine-tuning experiment.

The protocol and acceptance gates are written to disk before the first optimizer step.
Primary candidate = final step (no checkpoint selection on N9_01). A second identical-seed
run checks reproducibility. Nothing is promoted automatically.
"""

import argparse
import gc
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from roadmarking.training.pilot import load_crop
from roadmarking.training.road_text_head import CLASS_NAMES
from stage9c_attested_pilot import source_context, text_arrays
from stage9d_common import (
    ROOT,
    dump,
    frozen_checks,
    new_run,
    sha256,
    synthetic,
    synthetic_stems,
    verify_frozen,
)
from stage9d_eval import development_crops, full_eval, n9_groups
from stage9d_train_lib import (
    gradient_audit,
    load_three_class,
    make_optimizer,
    new_criterion,
    set_trainable,
    step_loss,
)

SEED = 9042
STEPS = 1200
EVAL_AT = (300, 600, 900, 1200)
WARMUP = 50
SYNTHETIC_EVERY = 3  # 1 in 3 steps synthetic; 2 in 3 real rehearsal
GROUPS = ["query_decoder", "mask_embedder", "class_predictor"]
LRS = dict(query_decoder=5e-5, mask_embedder=5e-5, class_predictor=5e-4)
GATES = dict(
    G1_meaningful_n9_detection="N9_01 text recall >= 0.20 and text TP >= 500 px",
    G2_nontrivial_precision_recall="N9_01 text precision >= 0.30 and recall >= 0.20 (scored px)",
    G3_valid_text_geometry=(
        "synthetic held-out text IoU >= 0.30, zero predicted text on N9_01 NoData, and "
        "at least 2 approved N9_01 text groups with group recall >= 0.30"
    ),
    G4_no_solid_dash_regression=(
        "solid and dash pixel F1 and IoU each drop <= 0.03 abs; instance recall drop <= 0.05; "
        "dash separation drop <= 0.05 vs frozen baseline, same development crops"
    ),
    G5_no_unsafe_nodata=(
        "kept-query raw NoData pixels <= 1.10 x baseline on development crops; stitching "
        "code hash unchanged; text predicted on line-only development crops <= 1% of their "
        "solid+dash target pixels"
    ),
    G6_reproducible="second identical-seed run: N9_01 text IoU and solid/dash F1 within 0.02",
    G7_save_reload="reloaded checkpoint reproduces evaluation counts exactly",
)


def build_schedule(train, synth_stems, rng):
    schedule = []
    real_order, syn_order = [], []
    for step in range(STEPS):
        if step % SYNTHETIC_EVERY == SYNTHETIC_EVERY - 1:
            if not syn_order:
                syn_order = list(rng.permutation(len(synth_stems)))
            schedule.append(("synthetic", synth_stems[syn_order.pop()]))
        else:
            if not real_order:
                real_order = list(rng.permutation(len(train)))
            schedule.append(("real", train[real_order.pop()]["crop_id"]))
    return schedule


def lr_scale(step):
    return min(1.0, step / WARMUP)


def gates(result, baseline, repeat, reload_ok):
    n9 = result["n9_01"]["text"]
    sd, bsd = result["solid_dash"], baseline["solid_dash"]
    groups_hit = sum(g["detected_recall_ge_0_3"] for g in result["n9_01"]["text_groups"].values())
    line_targets = sum(
        sd[k]["tp"] + sd[k]["fn"] for k in ("longitudinal_solid", "longitudinal_dash")
    )
    drops = {
        f"{k}_{m}": (bsd[k][m] or 0) - (sd[k][m] or 0)
        for k in ("longitudinal_solid", "longitudinal_dash")
        for m in ("f1", "iou")
    }
    inst_drop = {k: bsd["instance_recall"][k] - sd["instance_recall"][k] for k in ("solid", "dash")}
    g = dict(
        G1=(n9["recall"] or 0) >= 0.20 and n9["tp"] >= 500,
        G2=(n9["precision"] or 0) >= 0.30 and (n9["recall"] or 0) >= 0.20,
        G3=(result["synthetic_val"]["pixel"]["iou"] or 0) >= 0.30
        and result["n9_01"]["predicted_text_on_nodata"] == 0
        and groups_hit >= 2,
        G4=all(v <= 0.03 for v in drops.values())
        and all(v <= 0.05 for v in inst_drop.values())
        and bsd["dash_separation_rate"] - sd["dash_separation_rate"] <= 0.05,
        G5=sd["nodata_raw_pixels_of_kept_queries"]
        <= 1.10 * max(bsd["nodata_raw_pixels_of_kept_queries"], 1)
        and sd["text_predicted_pixels_on_line_dev_crops"] <= 0.01 * line_targets,
        G6=repeat,
        G7=reload_ok,
    )
    detail = dict(drops=drops, instance_recall_drops=inst_drop, n9_groups_detected=groups_hit)
    return g, detail


def train_once(
    tag, frozen_path, processor, ctx, schedule, eval_args, out, save_root, eval_at=EVAL_AT
):
    torch.manual_seed(SEED)
    model = load_three_class(frozen_path, SEED).cuda()
    trainable = set_trainable(model, GROUPS)
    optimizer, membership = make_optimizer(model, LRS)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    criterion = new_criterion().cuda()
    by_id = {c["crop_id"]: c for c in ctx[6]}
    dataset = ctx[4]
    logs, evals, grad_audits = [], {}, {}
    exposure = Counter()
    model.train()
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step, (kind, key) in enumerate(schedule, 1):
        for g, lr in zip(optimizer.param_groups, base_lrs, strict=True):
            g["lr"] = lr * lr_scale(step)
        if kind == "real":
            pixels, masks, labels, valid, _ = load_crop(dataset, by_id[key], processor, "cuda")
            masks, labels, valid = masks[0], labels[0], valid[0]
        else:
            pixels, masks, labels, valid, _, _ = synthetic("train", key, processor)
        for lab in labels.tolist():
            exposure[f"{kind}_{CLASS_NAMES[lab]}"] += 1
        exposure[f"{kind}_images"] += 1
        optimizer.zero_grad(set_to_none=True)
        total, losses, _ = step_loss(model, criterion, pixels, masks, labels, valid)
        if not torch.isfinite(total):
            raise RuntimeError(f"nonfinite loss at step {step}")
        total.backward()
        if step in (1, 2, 3) or step % 300 == 0:
            grad_audits[step] = dict(kind=kind, groups=gradient_audit(model))
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        logs.append(
            dict(
                step=step,
                kind=kind,
                total=float(total.detach()),
                **{k: float(v.detach()) for k, v in losses.items() if "aux" not in k},
                grad_norm=float(norm),
            )
        )
        if step in eval_at:
            res, pred = full_eval(model, processor, *eval_args)
            evals[step] = res
            print(
                tag,
                step,
                "n9 text",
                res["n9_01"]["text"]["recall"],
                res["n9_01"]["text"]["precision"],
                "synval iou",
                res["synthetic_val"]["pixel"]["iou"],
                "solid f1",
                res["solid_dash"]["longitudinal_solid"]["f1"],
                "dash f1",
                res["solid_dash"]["longitudinal_dash"]["f1"],
                flush=True,
            )
            if save_root is not None:
                ckpt = save_root / f"step{step:04}"
                model.save_pretrained(ckpt / "model")
                processor.save_pretrained(ckpt / "processor")
                np.save(out / f"n9_text_pred_{tag}_step{step:04}.npy", pred)
            model.train()
    elapsed = time.perf_counter() - start
    dump(out / f"training_log_{tag}.json", logs)
    dump(out / f"gradient_audit_{tag}.json", grad_audits)
    info = dict(
        trainable=trainable,
        optimizer=membership,
        exposure=dict(exposure),
        elapsed_seconds=elapsed,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    )
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return evals, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-repeat", action="store_true")
    args = parser.parse_args()
    out = new_run("finetune")
    checks = frozen_checks()
    verify_frozen(checks)
    ctx = source_context()
    reports, audit, _dataset, manifest_path, train, tile, spec = (
        ctx[0],
        ctx[1],
        ctx[4],
        ctx[5],
        ctx[6],
        ctx[9],
        ctx[10],
    )
    frozen_path = Path(checks[0]["path"]).parent
    processor = Mask2FormerImageProcessor.from_pretrained(
        frozen_path.parent / "processor", local_files_only=True
    )
    synth_stems = synthetic_stems("train", positive_only=False)
    train_fit = [s for s in synthetic_stems("train")][:24]
    val_stems = synthetic_stems("val", positive_only=False)
    development = development_crops(ctx)
    text = text_arrays(reports, tile, spec)
    groups = n9_groups(spec, tile, text[1].shape)
    rng = np.random.default_rng(SEED)
    schedule = build_schedule(train, synth_stems, rng)
    run_id = out.name
    save_root = ROOT / "models/mask2former_swin_s" / f"stage9d_{run_id}"
    protocol = dict(
        predeclared_before_training=True,
        warm_start=dict(path=str(frozen_path), sha256=checks[0]["expected_sha256"]),
        head_migration=(
            "make_three_class_head: solid/dash rows copied, no-object 2->3, text row seeded"
        ),
        trainable_groups=GROUPS,
        frozen_groups=["backbone", "pixel_decoder"],
        learning_rates=LRS,
        warmup_steps=WARMUP,
        optimizer="AdamW wd 0.01, grad clip 1.0, bf16 autocast, batch 1, auxiliary losses on",
        criterion="ValidMask2FormerCriterion(num_classes=3, bilinear=True)",
        steps=STEPS,
        intermediate_evaluations=EVAL_AT,
        budget_rationale=(
            "Tiny-subset test reached text IoU 0.60 by step 60 with LR 1e-4; 400 synthetic "
            "images seen once each (1 epoch) with 2:1 real rehearsal and half the LR"
        ),
        schedule_mix=dict(Counter(k for k, _ in schedule)),
        schedule=schedule,
        real_training_crops=len(train),
        real_dataset_manifest_sha256=sha256(manifest_path),
        synthetic_source_hash_count=len(audit["source_file_sha256"]),
        synthetic_train_stems=len(synth_stems),
        synthetic_train_fit_eval=train_fit,
        synthetic_val_eval=val_stems,
        synthetic_test_used=False,
        n9_01_training_use=False,
        n9_01_role="development evaluation only; fixed operating point; no threshold tuning",
        protected_tests_used=False,
        development_crops=[c["crop_id"] for _, c in development],
        operating_point="unchanged Stage 9C: argmax incl. no-object; prob*quality>=0.25; area>=4",
        primary_candidate=f"step {STEPS} (no checkpoint selection on N9_01)",
        acceptance_gates=GATES,
        domain_adaptation="none in this experiment (synthetic used as supplied)",
        seed=SEED,
    )
    dump(out / "training_protocol.json", protocol)
    eval_args = (development, text, groups, train_fit, val_stems)
    # Frozen baseline through the same evaluator
    baseline_model = Mask2FormerForUniversalSegmentation.from_pretrained(
        frozen_path, local_files_only=True
    ).cuda()
    baseline, base_pred = full_eval(baseline_model, processor, *eval_args)
    np.save(out / "n9_text_pred_baseline.npy", base_pred)
    del baseline_model
    torch.cuda.empty_cache()
    dump(out / "baseline_evaluation.json", baseline)
    evals, info = train_once(
        "run1", frozen_path, processor, ctx, schedule, eval_args, out, save_root
    )
    dump(
        out / "checkpoint_evaluations.json",
        {"baseline": baseline, **{str(k): v for k, v in evals.items()}},
    )
    dump(out / "training_info.json", info)
    final = evals[STEPS]
    # Save/reload check
    reloaded = Mask2FormerForUniversalSegmentation.from_pretrained(
        save_root / f"step{STEPS:04}/model", local_files_only=True
    ).cuda()
    re_eval, _ = full_eval(reloaded, processor, *eval_args)
    del reloaded
    torch.cuda.empty_cache()

    def key_counts(r):
        return [
            r["n9_01"]["text"]["tp"],
            r["n9_01"]["text"]["fp"],
            r["solid_dash"]["longitudinal_solid"]["tp"],
            r["solid_dash"]["longitudinal_solid"]["fp"],
            r["solid_dash"]["longitudinal_dash"]["tp"],
            r["solid_dash"]["longitudinal_dash"]["fp"],
            r["synthetic_val"]["pixel"]["tp"],
            r["synthetic_val"]["pixel"]["fp"],
        ]

    reload_ok = key_counts(re_eval) == key_counts(final)
    repeat_detail = None
    if args.skip_repeat:
        repeat_ok = False
    else:
        evals2, _ = train_once(
            "run2_repeat", frozen_path, processor, ctx, schedule, eval_args, out, None
        )
        f2 = evals2[STEPS]
        repeat_detail = dict(
            n9_iou=[final["n9_01"]["text"]["iou"], f2["n9_01"]["text"]["iou"]],
            solid_f1=[
                final["solid_dash"]["longitudinal_solid"]["f1"],
                f2["solid_dash"]["longitudinal_solid"]["f1"],
            ],
            dash_f1=[
                final["solid_dash"]["longitudinal_dash"]["f1"],
                f2["solid_dash"]["longitudinal_dash"]["f1"],
            ],
            synthetic_val_iou=[
                final["synthetic_val"]["pixel"]["iou"],
                f2["synthetic_val"]["pixel"]["iou"],
            ],
        )
        repeat_ok = all(abs((a or 0) - (b or 0)) <= 0.02 for a, b in repeat_detail.values())
        dump(out / "repeat_run_final_evaluation.json", f2)
    gate, detail = gates(final, baseline, repeat_ok, reload_ok)
    verify_frozen(checks)
    stitching_ok = all(sha256(Path(r["path"])) == r["expected_sha256"] for r in checks)
    decision = dict(
        candidate=str(save_root / f"step{STEPS:04}"),
        gates=gate,
        gate_detail=detail,
        repeat=repeat_detail,
        reload_counts_identical=reload_ok,
        preserved_artifact_hashes_ok=stitching_ok,
        all_gates_passed=all(gate.values()),
        promoted=False,
        note="Provisional development gates only; N9_01 is development-used.",
    )
    dump(out / "gate_evaluation.json", decision)
    # Learning-curve panel images
    rgb = text[0]
    tiles = []
    for name in ["baseline", *[f"run1_step{s:04}" for s in EVAL_AT]]:
        p = np.load(
            out
            / ("n9_text_pred_baseline.npy" if name == "baseline" else f"n9_text_pred_{name}.npy")
        )
        ov = rgb.copy()
        ov[text[1] & ~p] = [255, 0, 0]
        ov[p & text[1]] = [0, 255, 0]
        ov[p & ~text[1] & text[3]] = [0, 128, 255]
        tiles.append(ov[:260])
    Image.fromarray(np.concatenate(tiles, 0)).save(out / "n9_01_text_progression.png")
    print(out, decision["all_gates_passed"], gate)


if __name__ == "__main__":
    main()
