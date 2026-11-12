"""Stage 9E-A: attribute the real-only dash regression with direct evidence.

Uses the deterministic Stage 9D replica (R0) trained by stage9e_recipes.py and the frozen
model, on the identical 14 development crops:
1. component swaps (class head / mask embedder / query decoder) between frozen and R0;
2. per-decoder-layer raw solid/dash quality from auxiliary predictions;
3. dash false-positive proposal anatomy (confidence, location on verified background);
4. training-probe losses by class and layer (fit vs development);
5. data exposure (crops, tiles, physical groups, foreground pixels);
6. bf16 vs fp32 loss agreement; parameter drift and effective step size.
"""

import copy
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from roadmarking.inference.calibration import iou_matrix
from roadmarking.training.pilot import load_crop
from roadmarking.training.valid_criterion import ValidMask2FormerCriterion
from stage9d_common import ROOT, counts, dump, frozen_checks, metrics, read, verify_frozen
from stage9d_train_lib import group_of
from stage9e_lib import BASE, development_crops, evaluate, new_run, proposals, real_context


def load(path):
    return (
        Mask2FormerForUniversalSegmentation.from_pretrained(path, local_files_only=True)
        .cuda()
        .eval()
    )


def hybrid(base, donor, groups):
    model = copy.deepcopy(base)
    donor_state = donor.state_dict()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if group_of(n) in groups:
                p.copy_(donor_state[n])
    return model


@torch.no_grad()
def per_layer(model, processor, development):
    acc = {}
    for root, crop in development:
        pixel, masks, labels, validity, _ = load_crop(root, crop, processor, "cuda")
        valid = validity[0].cpu().numpy().astype(bool)
        truth = masks[0].bool().cpu().numpy()
        tl = labels[0].cpu().numpy()
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(pixel_values=pixel, output_auxiliary_logits=True)
        layers = [
            *out.auxiliary_logits,
            dict(
                masks_queries_logits=out.masks_queries_logits,
                class_queries_logits=out.class_queries_logits,
            ),
        ]
        for li, layer in enumerate(layers):
            native = torch.nn.functional.interpolate(
                layer["masks_queries_logits"].float(),
                size=valid.shape,
                mode="bilinear",
                align_corners=False,
            )[0]
            probs = layer["class_queries_logits"][0].float().softmax(-1)
            top, lab = probs.max(-1)
            v = torch.as_tensor(valid, device=native.device)
            sel = (native > 0) & v
            area = sel.flatten(1).sum(1)
            q = (native.sigmoid() * sel).flatten(1).sum(1) / area.clamp(min=1)
            keep = (lab < probs.shape[-1] - 1) & (area >= 4) & (top * q >= 0.25)
            for c in (0, 1):
                k = keep & (lab == c)
                pred = sel[k].any(0).cpu().numpy() if k.any() else np.zeros_like(valid)
                t = truth[tl == c].any(0) if (tl == c).any() else np.zeros_like(valid)
                acc.setdefault((li, c), np.zeros(3, np.int64))
                acc[(li, c)] += counts(pred, t, valid)
    return {
        f"layer{li}_{'solid' if c == 0 else 'dash'}": metrics(v)
        for (li, c), v in sorted(acc.items())
    }


@torch.no_grad()
def dash_fp_anatomy(model, processor, development):
    rows = Counter()
    scores = {"tp": [], "fp": []}
    for root, crop in development:
        pixel, masks, labels, validity, _rgb = load_crop(root, crop, processor, "cuda")
        valid = validity[0].cpu().numpy().astype(bool)
        truth = masks[0].bool().cpu().numpy()
        tl = labels[0].cpu().numpy()
        pr = proposals(model, pixel, valid)
        idx = [i for i, lab in enumerate(pr["labels"]) if lab == 1]
        if not idx:
            continue
        pm = pr["masks"][idx]
        dash_t = [truth[k] for k in range(len(tl)) if tl[k] == 1]
        solid_t = [truth[k] for k in range(len(tl)) if tl[k] == 0]
        iou_d = iou_matrix(pm, dash_t, valid).max(1) if dash_t else np.zeros(len(idx))
        any_paint = truth.any(0) if len(truth) else np.zeros_like(valid)
        for j, i in enumerate(idx):
            m = pm[j] & valid
            kind = "tp" if iou_d[j] >= 0.5 else "fp"
            scores[kind].append(float(pr["scores"][i]))
            if kind == "fp":
                frac_paint = (m & any_paint).sum() / max(m.sum(), 1)
                on_solid = (
                    bool(solid_t) and (m & np.any(solid_t, axis=0)).sum() / max(m.sum(), 1) > 0.5
                )
                where = (
                    "on_solid_line"
                    if on_solid
                    else (
                        "partial_dash_or_merge"
                        if frac_paint > 0.3
                        else "verified_background" if (m.sum() > 0) else "ignored_only"
                    )
                )
                rows[where] += 1
                rows["fp_on_invalid_pixels_mostly"] += int((pm[j] & ~valid).sum() > m.sum())
                rows["fp_v1_accepted"] += int(pr["scores"][i] >= 0.8853)
    return dict(
        fp_location=dict(rows),
        tp=len(scores["tp"]),
        fp=len(scores["fp"]),
        fp_score_median=float(np.median(scores["fp"])) if scores["fp"] else None,
        fp_scores_ge_v1_dash_threshold=int(sum(s >= 0.8853 for s in scores["fp"])),
        tp_score_median=float(np.median(scores["tp"])) if scores["tp"] else None,
    )


def probe_losses(model, processor, crops, dataset, autocast_dtype):
    criterion = ValidMask2FormerCriterion(num_classes=2, bilinear=True).cuda()
    totals = Counter()
    by_class = Counter()
    n_class = Counter()
    confident_unmatched = 0
    for crop in crops:
        pixels, masks, labels, valid, _ = load_crop(dataset, crop, processor, "cuda")
        with (
            torch.no_grad(),
            torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None),
        ):
            out = model(pixel_values=pixels, output_auxiliary_logits=True)
        aux = [{k: v.float() for k, v in a.items()} for a in out.auxiliary_logits]
        with torch.no_grad():
            losses, matches = criterion(
                out.masks_queries_logits.float(),
                out.class_queries_logits.float(),
                masks,
                labels,
                valid,
                [True],
                aux,
            )
        for k, v in losses.items():
            totals[k] += float(v)
        qi, ti = matches[0]
        probs = out.class_queries_logits[0].float().softmax(-1)
        for q, t in zip(qi.tolist(), ti.tolist(), strict=True):
            c = int(labels[0][t])
            by_class[c] += float(-torch.log(probs[q, c].clamp(min=1e-8)))
            n_class[c] += 1
        unmatched = torch.ones(len(probs), dtype=torch.bool, device=probs.device)
        unmatched[qi] = False
        confident_unmatched += int(((probs[:, :2].max(-1).values > 0.5) & unmatched).sum())
    n = len(crops)
    return dict(
        mean_losses={k: v / n for k, v in sorted(totals.items())},
        matched_class_nll_by_class={
            ("solid" if c == 0 else "dash"): by_class[c] / max(n_class[c], 1) for c in (0, 1)
        },
        confident_unmatched_queries_per_crop=confident_unmatched / n,
    )


def main():
    out = new_run("attribution")
    checks = frozen_checks()
    verify_frozen(checks)
    frozen_path = Path(checks[0]["path"]).parent
    processor = Mask2FormerImageProcessor.from_pretrained(
        frozen_path.parent / "processor", local_files_only=True
    )
    rc = real_context()
    development = development_crops(rc["ctx"])
    recipes = Path(read(BASE / "recipes/latest.json")["run"])
    r0_path = (
        ROOT
        / "models/mask2former_swin_s"
        / f"stage9e_{recipes.name}"
        / "R0_stage9d_replica_s9061/model"
    )
    frozen_m, r0_m = load(frozen_path), load(r0_path)
    report = {}
    swaps = {
        "frozen": frozen_m,
        "R0": r0_m,
        "frozen+R0_class_head": hybrid(frozen_m, r0_m, {"class_predictor"}),
        "frozen+R0_mask_embedder": hybrid(frozen_m, r0_m, {"mask_embedder"}),
        "frozen+R0_query_decoder": hybrid(frozen_m, r0_m, {"query_decoder"}),
        "frozen+R0_decoder+mask_embedder": hybrid(
            frozen_m, r0_m, {"query_decoder", "mask_embedder"}
        ),
        "R0+frozen_class_head": hybrid(r0_m, frozen_m, {"class_predictor"}),
    }
    report["component_swaps"] = {}
    for name, m in swaps.items():
        e = evaluate(m, processor, development)
        report["component_swaps"][name] = {
            lvl: {c: e[lvl]["pixel"][c] for c in ("solid", "dash")}
            | {"proposals": e[lvl]["proposals"]}
            for lvl in ("raw", "v1")
        }
        print(
            name,
            {
                lvl: (
                    round(e[lvl]["pixel"]["solid"]["f1"] or 0, 3),
                    round(e[lvl]["pixel"]["dash"]["f1"] or 0, 3),
                )
                for lvl in ("raw", "v1")
            },
            flush=True,
        )
        if name not in ("frozen", "R0"):
            del m
            torch.cuda.empty_cache()
    report["per_decoder_layer_raw"] = {
        "frozen": per_layer(frozen_m, processor, development),
        "R0": per_layer(r0_m, processor, development),
    }
    report["dash_false_positive_anatomy"] = {
        "frozen": dash_fp_anatomy(frozen_m, processor, development),
        "R0": dash_fp_anatomy(r0_m, processor, development),
    }
    probe = sorted(rc["stage9d"], key=lambda c: c["crop_id"])[:24]
    report["training_probe_24_of_65"] = {
        "frozen_bf16": probe_losses(frozen_m, processor, probe, rc["dataset"], torch.bfloat16),
        "frozen_fp32": probe_losses(frozen_m, processor, probe, rc["dataset"], None),
        "R0_bf16": probe_losses(r0_m, processor, probe, rc["dataset"], torch.bfloat16),
    }
    held = sorted(
        [c for c in rc["full"] if c["crop_id"] not in {x["crop_id"] for x in rc["stage9d"]}],
        key=lambda c: c["crop_id"],
    )[:24]
    report["approved_crops_not_in_R0_pool_24"] = {
        "frozen_bf16": probe_losses(frozen_m, processor, held, rc["dataset"], torch.bfloat16),
        "R0_bf16": probe_losses(r0_m, processor, held, rc["dataset"], torch.bfloat16),
    }
    # Exposure audit from the R0 training log
    log = read(recipes / "R0_stage9d_replica_s9061_log.json")
    info = read(recipes / "R0_stage9d_replica_s9061_info.json")
    crops = {c["crop_id"]: c for c in rc["full"]}
    audit = read(rc["ctx"][0] / "stage8b/independence_audit.json")
    groups = {u: min(g) for g in audit["physical_groups"] for u in g}
    exposure = Counter(r["crop"] for r in log)
    tiles = Counter(crops[c]["tile_id"] for c in exposure.elements())
    phys = {groups.get(u, u) for c in exposure for u in crops[c]["global_instance_ids"]}
    phys_full = {groups.get(u, u) for c in rc["full"] for u in c["global_instance_ids"]}
    report["exposure"] = dict(
        stage9d_pool_crops=len(rc["stage9d"]),
        approved_full_pool_crops=len(rc["full"]),
        fraction_of_approved=len(rc["stage9d"]) / len(rc["full"]),
        steps=len(log),
        distinct_crops_seen=len(exposure),
        exposures_per_crop_min_max=[min(exposure.values()), max(exposure.values())],
        distinct_tiles=len(tiles),
        approved_tiles=len({c["tile_id"] for c in rc["full"]}),
        distinct_physical_groups=len(phys),
        approved_physical_groups=len(phys_full),
        label_mix_of_pool=dict(
            Counter(tuple(sorted(set(c["labels"]))) for c in rc["stage9d"]).most_common()
        ),
        label_mix_of_approved=dict(
            Counter(tuple(sorted(set(c["labels"]))) for c in rc["full"]).most_common()
        ),
        geography_of_pool=dict(Counter(c["geography"] for c in rc["stage9d"])),
        **{k: v for k, v in info["exposure_summary"].items()},
    )
    report["exposure"] = {str(k): v for k, v in report["exposure"].items()}
    report["exposure"]["label_mix_of_pool"] = {
        str(k): v for k, v in report["exposure"]["label_mix_of_pool"].items()
    }
    report["exposure"]["label_mix_of_approved"] = {
        str(k): v for k, v in report["exposure"]["label_mix_of_approved"].items()
    }
    report["optimization"] = dict(
        stage9d_lr=info["lrs"],
        frozen_model_original_recipe=dict(
            lr=2e-6,
            schedule="cosine to 4e-7",
            trainable="all parameters",
            precision="fp16 + GradScaler",
            criterion="nearest sampling, default",
            steps=600,
            sampling="class-balanced cycle over phase5_12 pools",
        ),
        lr_ratio_decoder_vs_original=info["lrs"]["query_decoder"] / 2e-6,
        relative_parameter_drift=info["relative_parameter_drift"],
        nonfinite_steps=sum(1 for r in log if not np.isfinite(r["total"])),
        loss_class_by_100_steps=[
            float(np.mean([r["loss_class"] for r in log[i : i + 100]]))
            for i in range(0, len(log), 100)
        ],
        loss_unmatched_by_100_steps=[
            float(np.mean([r["loss_unmatched_valid_mask"] for r in log[i : i + 100]]))
            for i in range(0, len(log), 100)
        ],
        grad_norm_median_by_100_steps=[
            float(np.median([r["grad_norm"] for r in log[i : i + 100]]))
            for i in range(0, len(log), 100)
        ],
        clipped_fraction=float(np.mean([r["grad_norm"] > 1.0 for r in log])),
    )
    report["duration"] = {
        k: {lvl: {c: v[lvl]["pixel"][c]["f1"] for c in ("solid", "dash")} for lvl in ("raw", "v1")}
        for k, v in read(recipes / "R0_stage9d_replica_s9061_evals.json").items()
    }
    report["duration"]["0"] = {
        lvl: {
            c: read(recipes / "baseline_frozen.json")[lvl]["pixel"][c]["f1"]
            for c in ("solid", "dash")
        }
        for lvl in ("raw", "v1")
    }
    verify_frozen(checks)
    dump(out / "dash_regression_attribution.json", report)
    print(out)


if __name__ == "__main__":
    main()
