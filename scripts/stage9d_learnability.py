"""Stage 9D-C: tiny fixed-subset synthetic text mask learnability (software diagnostic).

Evaluated on the same training samples by design; this is NOT generalization evidence.
Variant A trains query decoder + mask embedder + classifier. Variant B (run only if A
fails the predeclared gate) additionally trains the pixel decoder.
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Mask2FormerImageProcessor

from stage9d_common import (
    dump,
    frozen_checks,
    metrics,
    new_run,
    synthetic,
    synthetic_stems,
    verify_frozen,
)
from stage9d_train_lib import (
    eval_text_sample,
    gradient_audit,
    load_three_class,
    make_optimizer,
    new_criterion,
    set_trainable,
    step_loss,
    summarize,
)

SEED = 9041
SUBSET = 8
STEPS = 240
EVAL_AT = (0, 60, 120, 240)
VARIANTS = {
    "A_decoder_mask_head": ["query_decoder", "mask_embedder", "class_predictor"],
    "B_plus_pixel_decoder": ["query_decoder", "mask_embedder", "class_predictor", "pixel_decoder"],
}
LRS = dict(query_decoder=1e-4, mask_embedder=1e-4, pixel_decoder=5e-5, class_predictor=1e-3)
GATE = dict(
    final_pixel_iou_min=0.5,
    samples_with_text_tp=SUBSET,
    loss_reduction_min=0.5,
    finite_gradients=True,
    trainable_groups_nonzero_gradients=True,
    frozen_groups_no_gradients=True,
)


def evaluate(model, data):
    model.eval()
    rows = [eval_text_sample(model, d[0], d[5], d[6]) for d in data]
    model.train()
    return rows


def panel(data, rows, path):
    tiles = []
    for d, r in zip(data[:4], rows[:4], strict=False):
        rgb = d[4].copy()
        ov = rgb.copy()
        target, valid = d[5], d[6]
        pred = r["post"][2]
        ov[target & ~pred] = [255, 0, 0]
        ov[pred & ~target & valid] = [0, 128, 255]
        ov[pred & target] = [0, 255, 0]
        tiles.append(np.concatenate([rgb, ov], 1)[::2, ::2])
    Image.fromarray(np.concatenate(tiles, 0)).save(path)


def run_variant(name, groups, frozen_path, processor, stems, out):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = load_three_class(frozen_path, SEED).cuda()
    trainable = set_trainable(model, groups)
    optimizer, membership = make_optimizer(model, LRS)
    criterion = new_criterion().cuda()
    data = []
    for stem in stems:
        pixels, masks, labels, valid, rgb, _ = synthetic("train", stem, processor)
        data.append(
            (
                pixels,
                masks,
                labels,
                valid,
                rgb,
                masks[0].bool().cpu().numpy(),
                valid.cpu().numpy().astype(bool),
            )
        )
    model.train()
    # Keep BatchNorm-free Swin in train mode; dropout in decoder is config default.
    evals = {0: summarize(evaluate(model, data))}
    logs, grad_first = [], None
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, STEPS + 1):
        d = data[(step - 1) % len(data)]
        optimizer.zero_grad(set_to_none=True)
        total, losses, (_match,) = step_loss(model, criterion, d[0], d[1], d[2], d[3])
        if not torch.isfinite(total):
            raise RuntimeError(f"nonfinite loss at {step}")
        total.backward()
        if step == 1:
            grad_first = gradient_audit(model)
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        logs.append(
            dict(
                step=step,
                total=float(total.detach()),
                **{k: float(v.detach()) for k, v in losses.items() if "aux" not in k},
                grad_norm=float(norm),
            )
        )
        if step in EVAL_AT:
            rows = evaluate(model, data)
            evals[step] = summarize(rows)
            print(name, step, evals[step]["pixel"]["iou"], flush=True)
    elapsed = time.perf_counter() - start
    rows = evaluate(model, data)
    panel(data, rows, out / f"tiny_subset_{name}.png")
    first = np.mean([r["total"] for r in logs[: len(data)]])
    last = np.mean([r["total"] for r in logs[-len(data) :]])
    final = evals[STEPS]
    frozen_groups = [g for g in grad_first if g not in groups]
    checks = dict(
        final_pixel_iou=final["pixel"]["iou"],
        final_pixel_iou_pass=(final["pixel"]["iou"] or 0) >= GATE["final_pixel_iou_min"],
        samples_with_text_tp_pass=final["samples_with_text_tp"] == SUBSET,
        loss_reduction=float(1 - last / first),
        loss_reduction_pass=float(1 - last / first) >= GATE["loss_reduction_min"],
        finite_gradients=all(r["nonfinite"] == 0 for r in grad_first.values()),
        trainable_groups_nonzero=all(grad_first[g]["norm"] > 0 for g in groups),
        frozen_groups_no_grad=all(grad_first[g]["with_grad"] == 0 for g in frozen_groups),
        optimizer_membership=membership["optimizer_equals_requires_grad"],
    )
    checks["passed"] = all(v for k, v in checks.items() if isinstance(v, bool))
    result = dict(
        variant=name,
        trainable_groups=groups,
        trainable_audit=trainable,
        optimizer=membership,
        first_step_gradients=grad_first,
        evaluations=evals,
        per_sample_final=[
            dict(stem=s, **metrics(r["counts"]), best_query_iou=r["best_query_iou"])
            for s, r in zip(stems, rows, strict=True)
        ],
        loss_first_cycle=float(first),
        loss_last_cycle=float(last),
        gate_checks=checks,
        elapsed_seconds=elapsed,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
    )
    dump(out / f"learnability_{name}.json", result)
    dump(out / f"learnability_{name}_log.json", logs)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-b", action="store_true")
    args = parser.parse_args()
    out = new_run("learnability")
    checks = frozen_checks()
    verify_frozen(checks)
    frozen_path = Path(checks[0]["path"]).parent
    processor = Mask2FormerImageProcessor.from_pretrained(
        frozen_path.parent / "processor", local_files_only=True
    )
    stems = synthetic_stems("train")[:SUBSET]
    dump(
        out / "learnability_protocol.json",
        dict(
            purpose="software/learnability diagnostic only; same-sample evaluation",
            subset=stems,
            seed=SEED,
            steps=STEPS,
            eval_at=EVAL_AT,
            variants=VARIANTS,
            lrs=LRS,
            optimizer="AdamW wd 0.01, grad clip 1.0, bf16 autocast, batch 1, aux losses",
            criterion="ValidMask2FormerCriterion(num_classes=3, bilinear=True)",
            operating_point="unchanged Stage 9C fixed point",
            gate=GATE,
            run_b_only_if_a_fails=True,
        ),
    )
    results = {}
    results["A"] = run_variant(
        "A_decoder_mask_head", VARIANTS["A_decoder_mask_head"], frozen_path, processor, stems, out
    )
    if args.force_b or not results["A"]["gate_checks"]["passed"]:
        results["B"] = run_variant(
            "B_plus_pixel_decoder",
            VARIANTS["B_plus_pixel_decoder"],
            frozen_path,
            processor,
            stems,
            out,
        )
    verify_frozen(checks)
    dump(
        out / "learnability_summary.json",
        {k: dict(gate=v["gate_checks"], final=v["evaluations"][STEPS]) for k, v in results.items()},
    )
    print(out)


if __name__ == "__main__":
    main()
