"""Steps 4-8: one bounded corrective warm start from Model B with a paint-protected NoData cap.

Model B has no saved optimizer/scheduler state, so a true continuation is impossible. This is a
documented warm start: Model B weights, fresh AdamW, 20-step warm-up, half the earlier LRs.
Teacher = Model A (frozen, detached). The protocol and gates are sealed before step 1.
"""

import hashlib
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation
from transformers import Mask2FormerForUniversalSegmentation

from correction_common import BASE, MODELS, PRIOR_RUN, dump, model_a_dir, model_b_dir, read
from regsafe_common import frozen_paths, load_model, load_processor, sha256, verify_frozen
from regsafe_eval import CLASSES, GATES, LEVELS, gate_table, run
from regsafe_train import (
    TRAINABLE,
    RealFirstSampler,
    exposure_plan,
    gated_vector,
    reload_identical,
    teacher_grad_free,
)
from roadmarking.inference.calibration import filter_instances, iou_matrix
from roadmarking.training.deterministic_sampling import with_exact_bilinear
from roadmarking.training.gated_criterion import GatedNoObjectCriterion
from roadmarking.training.pilot import load_crop
from roadmarking.training.synthetic_augment import AugmentConfig, augment
from roadmarking.training.teacher_guard import guarded_teacher_loss, protected_nodata_cap_loss
from stage9d_train_lib import gradient_audit, group_of, make_optimizer, set_trainable
from stage9e_lib import V1, development_crops, proposals, real_context, with_root
from stage9e_recipes import LOW

SEED = 9131
STEPS = 150
CHECKPOINTS = (50, 100, 150)
WARMUP = 20
LRS = {k: v * 0.5 for k, v in LOW.items()}
REAL_PER_CYCLE = 7
TEACHER = {"mask": 5.0, "class": 2.0}
BAND_PX = 8
TARGET_RATIO = 0.5
LAMBDA_BOUNDS = (1.0, 50.0)
SYN_AUG = AugmentConfig(scale_probability=1.0, scale_range=(0.3, 0.6))
VS_B = dict(
    pixel_f1_drop=0.02,
    v1_pixel_precision_drop=0.03,
    v1_false_positive_px_ratio=1.25,
    boundary_paint_recall_drop=0.03,
)
EARLY = dict(pixel_f1_drop_vs_b=0.05, raw_nodata_above_b=True, v1_fp_ratio_vs_b=1.5)


def load_step(crop, processor, rng, triples):
    if crop["geography"] != "synthetic":
        pixels, masks, labels, valid, rgb = load_crop(crop["_root"], crop, processor, "cuda")
        v = valid[0].cpu().numpy().astype(bool)
        return pixels, masks[0], labels[0], valid[0], ~v & (rgb[..., 0] == 0), {}
    with np.load(crop["file"]) as d:
        gray, m, lab, v = d["rgb"][..., 0], d["masks"], d["labels"], d["validity"]
    a = augment(gray, m, lab, v, rng, triples, SYN_AUG)
    rgb = np.repeat(a["gray"][..., None], 3, axis=2)
    pixels = processor(images=rgb, return_tensors="pt")["pixel_values"].cuda()
    return (
        pixels,
        torch.from_numpy(a["masks"].astype(np.float32)).cuda(),
        torch.from_numpy(a["labels"]).cuda(),
        torch.from_numpy(a["valid"]).cuda(),
        a["nodata"],
        a["record"],
    )


def nodata_band(valid, nodata):
    return nodata & binary_dilation(valid, iterations=BAND_PX)


def forward_losses(model, teacher, criterion, pixels, masks, labels, valid, nodata, real):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(pixel_values=pixels, output_auxiliary_logits=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        tpred = teacher(pixel_values=pixels, output_auxiliary_logits=False)
    with torch.autocast("cuda", enabled=False):
        sm, sc = pred.masks_queries_logits.float(), pred.class_queries_logits.float()
        tm, tc = tpred.masks_queries_logits.float(), tpred.class_queries_logits.float()
        aux = [{k: v.float() for k, v in a.items()} for a in pred.auxiliary_logits]
        criterion.last_gate.clear()
        losses, matches = criterion(sm, sc, [masks], [labels], [valid], [True], aux)
        criterion.last_gate.clear()
        sup = criterion.total(losses)
        qi, ti = matches[0]
        diag = {}
        kd = sup * 0
        gate = gated_vector(criterion, sm[0], valid, qi)
        diag["gated_queries"] = int(gate.sum())
        if real:
            kd, d = guarded_teacher_loss(
                sm[0], sc[0], tm[0], tc[0], criterion, masks, valid, qi, gate, TEACHER
            )
            diag.update(d)
        band = nodata_band(valid.cpu().numpy().astype(bool), nodata)
        nd, d = protected_nodata_cap_loss(
            sm[0], tm[0], torch.from_numpy(band).cuda(), masks, valid, qi, ti, criterion
        )
        diag.update(d)
    return sup, kd, nd, losses, diag, (sm, qi, ti)


def norm(gs):
    return float(sum(float(g.float().norm() ** 2) for g in gs if g is not None) ** 0.5)


def preflight(model, teacher, criterion, processor, real_train, out):
    rep = dict(trainable=set_trainable(model, TRAINABLE))
    assert rep["trainable"]["backbone"]["trainable_params"] == 0
    assert rep["trainable"]["pixel_decoder"]["trainable_params"] == 0
    opt, membership = make_optimizer(model, LRS)
    assert membership["optimizer_equals_requires_grad"]
    rep["optimizer"] = membership
    rep["continuation"] = (
        "Model B stored weights + processor only (no optimizer, scheduler or step state): "
        "warm start with fresh AdamW, 20-step warm-up, LRs x0.5"
    )
    del opt
    params = [p for p in model.parameters() if p.requires_grad]
    probe = sorted(real_train, key=lambda c: c["crop_id"])[::20][:24]
    rows = []
    for crop in probe:
        pixels, masks, labels, valid, nodata, _ = load_step(crop, processor, None, [])
        model.zero_grad(set_to_none=True)
        sup, kd, nd, _l, diag, (sm, qi, ti) = forward_losses(
            model, teacher, criterion, pixels, masks, labels, valid, nodata, True
        )
        # (1) ignore / NoData invariance of supervised loss.
        altered = masks.clone()
        altered[:, ~valid] = 1 - altered[:, ~valid]
        sup2 = forward_losses(
            model, teacher, criterion, pixels, altered, labels, valid, nodata, True
        )[0]
        # (6) cap never pushes logits of output cells holding the matched query's observed paint.
        g_sm = (
            torch.autograd.grad(nd, sm, retain_graph=True, allow_unused=True)[0]
            if nd.requires_grad
            else None
        )
        paint_cell_push = 0.0
        if g_sm is not None and len(qi):
            from roadmarking.training.teacher_guard import paint_cells

            cells = paint_cells(masks[ti], valid, *sm.shape[-2:])
            paint_cell_push = float(g_sm[0][qi][cells].abs().sum())
        # (7) supervised gradient vs reviewed labels: matched masks pushed up on observed paint.
        contradict = 0.0
        if len(qi):
            from roadmarking.training.teacher_guard import paint_cells

            cells = paint_cells(masks[ti], valid, *sm.shape[-2:])
            g_kdnd = torch.autograd.grad(kd + nd, sm, retain_graph=True, allow_unused=True)[0]
            if g_kdnd is not None:
                # positive gradient = pushes logit down; on paint cells that contradicts labels
                contradict = float(g_kdnd[0][qi][cells].clamp(min=0).sum())
        g_sup = torch.autograd.grad(sup, params, retain_graph=True, allow_unused=True)
        g_nd = (
            torch.autograd.grad(nd, params, retain_graph=True, allow_unused=True)
            if float(nd) > 0
            else [None]
        )
        g_kd = torch.autograd.grad(kd, params, allow_unused=True)
        rows.append(
            dict(
                crop=crop["crop_id"],
                finite=all(bool(torch.isfinite(x)) for x in (sup, kd, nd)),
                finite_grads=all(g is None or bool(torch.isfinite(g).all()) for g in g_sup),
                ignored_gt_invariance=abs(float(sup) - float(sup2))
                < 1e-4 * max(1, abs(float(sup))),
                paint_cell_push_from_cap=paint_cell_push,
                aux_losses_contradicting_paint=contradict,
                grad_sup=norm(g_sup),
                grad_nodata=norm(g_nd),
                grad_teacher=norm(g_kd),
                nodata=float(nd),
                **{k: v for k, v in diag.items() if isinstance(v, (int, float))},
            )
        )
        assert teacher_grad_free(teacher)
    rep["probe"] = rows
    checks = dict(
        nodata_ignore_invariance=all(r["ignored_gt_invariance"] for r in rows),
        finite_losses=all(r["finite"] for r in rows),
        finite_gradients=all(r["finite_grads"] for r in rows),
        cap_never_pushes_own_paint_cells=all(r["paint_cell_push_from_cap"] == 0 for r in rows),
        auxiliary_losses_never_push_down_observed_paint=all(
            r["aux_losses_contradicting_paint"] == 0 for r in rows
        ),
        teacher_frozen=teacher_grad_free(teacher),
    )
    rep["checks"] = checks
    ratios = [r["grad_nodata"] / r["grad_sup"] for r in rows if r["grad_nodata"] > 0]
    rep["nodata_grad_ratios_unit_lambda"] = ratios
    lam = TARGET_RATIO / float(np.median(ratios)) if len(ratios) >= 5 else LAMBDA_BOUNDS[0]
    rep["nodata_lambda"] = float(np.clip(float(f"{lam:.1g}"), *LAMBDA_BOUNDS))
    rep["nodata_lambda_rule"] = (
        f"{TARGET_RATIO} / median(||grad cap|| / ||grad supervised||) on Model B for probe crops "
        f"with violations, 1 significant figure, clipped {LAMBDA_BOUNDS}"
    )
    model.zero_grad(set_to_none=True)
    dump(out / "preflight.json", rep)
    return rep


@torch.no_grad()
def boundary_recall(model, processor, development):
    """Recall of reviewed observed paint pixels within 2 px of NoData (raw and V1)."""
    acc = {lvl: {c: [0, 0] for c in (0, 1)} for lvl in LEVELS}
    for root, crop in development:
        pixel, masks, labels, validity, rgb = load_crop(root, crop, processor, "cuda")
        valid = validity[0].cpu().numpy().astype(bool)
        nod = ~valid & (rgb[..., 0] == 0)
        edge = valid & binary_dilation(nod, iterations=2)
        truth = masks[0].bool().cpu().numpy()
        tl = labels[0].cpu().numpy()
        pr = proposals(model, pixel, valid)
        sel = (
            filter_instances(
                pr["scores"],
                pr["labels"],
                iou_matrix(pr["masks"], pr["masks"], valid),
                pr["qualities"],
                dict(V1, confidence={**V1["confidence"], "2": 1.01}),
            )
            if len(pr["labels"])
            else []
        )
        for c in (0, 1):
            gt = (truth[tl == c].any(0) if (tl == c).any() else np.zeros_like(valid)) & edge
            for lvl, idx in (("raw", range(len(pr["labels"]))), ("v1", sel)):
                u = np.zeros_like(valid)
                for i in idx:
                    if pr["labels"][i] == c:
                        u |= pr["masks"][i]
                acc[lvl][c][0] += int((u & gt).sum())
                acc[lvl][c][1] += int(gt.sum())
    return {
        lvl: {
            CLASSES[c]: dict(covered=a, total=t, recall=a / t if t else None)
            for c, (a, t) in d.items()
        }
        for lvl, d in acc.items()
    }


def stitch_safety(model, processor, dev, out):
    """Production stitching (unchanged) on the development groups: unsupported pixels, gaps."""
    from roadmarking.inference.stitching import stitch
    from stage9c_attested_pilot import source_context
    from stage9g import observed_sources
    from stage9h import prepare, score

    (out / "queries").mkdir(parents=True, exist_ok=True)
    ctx = source_context()
    sources = observed_sources(ctx[3])
    total = Counter()
    for _group, d in prepare(dev, sources, model, processor, out):
        args = (d["masks"], d["scores"], d["classes"], d["ids"], d["windows"])
        outputs = stitch(*args, d["valid"], V1["confidence"])[0]
        s = score(outputs, d)
        total["stitched_unsupported_pixels"] += s["unsupported_pixels"]
        total["stitched_gap_segments_with_unobserved"] += len(s["unsupported_gap_segments"])
        total["stitched_outputs"] += len(outputs)
        for cls in ("0", "1"):
            total[f"stitched_multi_dash_outputs_{cls}"] += s["by_class"][cls]["multi_dash_outputs"]
    return dict(total)


def vs_b_table(res, b, bres_extra, extra):
    checks = {}
    for lvl in LEVELS:
        for c in CLASSES:
            d = (b["real"][lvl]["pixel"][c]["f1"] or 0) - (res["real"][lvl]["pixel"][c]["f1"] or 0)
            checks[f"vsB_real_{lvl}_{c}_pixel_f1_drop<={VS_B['pixel_f1_drop']}"] = (
                d <= VS_B["pixel_f1_drop"],
                round(d, 4),
            )
    for c in CLASSES:
        d = (b["real"]["v1"]["pixel"][c]["precision"] or 0) - (
            res["real"]["v1"]["pixel"][c]["precision"] or 0
        )
        checks[f"vsB_real_v1_{c}_pixel_precision_drop<={VS_B['v1_pixel_precision_drop']}"] = (
            d <= VS_B["v1_pixel_precision_drop"],
            round(d, 4),
        )
    fb = b["real"]["v1"]["cross_class"]["predicted_on_verified_background"]
    fr = res["real"]["v1"]["cross_class"]["predicted_on_verified_background"]
    checks[f"vsB_real_v1_false_positive_px<={VS_B['v1_false_positive_px_ratio']}x"] = (
        fr <= VS_B["v1_false_positive_px_ratio"] * max(fb, 1),
        [fb, fr],
    )
    for lvl in LEVELS:
        for c in CLASSES:
            rb, rr = bres_extra[lvl][c]["recall"], extra[lvl][c]["recall"]
            if rb is None:
                continue
            d = rb - (rr or 0)
            checks[
                f"vsB_real_{lvl}_{c}_boundary_paint_recall_drop<={VS_B['boundary_paint_recall_drop']}"
            ] = (d <= VS_B["boundary_paint_recall_drop"], round(d, 4))
    return checks


def classify(failed):
    if not failed:
        return "CORRECTED_CANDIDATE_IMPROVED_PROVISIONAL"
    if all("nodata" in f for f in failed):
        return "NODATA_SAFETY_BLOCKED"
    return "RECOGNITION_REGRESSION"


def main():
    torch.set_num_threads(4)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = BASE / "training" / stamp
    out.mkdir(parents=True)
    dump(BASE / "training/latest.json", {"run": str(out)})
    print(out, flush=True)
    checks, _, _ = frozen_paths()
    manifest = read(BASE / "preservation/comparison_manifest.json")
    a_dir, b_dir = model_a_dir(), model_b_dir()
    for files in (manifest["model_a"]["files"], manifest["model_b"]["files"]):
        for rel, h in files.items():
            assert sha256(PRIOR_RUN.parents[4] / rel) == h, rel
    processor = load_processor()
    sup_dir = Path(read(BASE.parent / "regression_safe_sprint/supervision/latest.json")["run"])
    syn = read(sup_dir / "synthetic_records.json")
    for r in syn:
        r["_root"] = Path(r["file"]).parent
    syn_train = [r for r in syn if r["role"] == "train"]
    syn_dev = [(r["_root"], r) for r in syn if r["role"] == "development"]
    triples = [
        (r["background_median"], r["background_std"], r["contrast"])
        for r in read(sup_dir / "real_image_statistics.json")
        if r.get("contrast") is not None and r["contrast"] > 0
    ]
    rc = real_context()
    real_train = with_root(rc["full"], rc["dataset"])
    real_dev = development_crops(rc["ctx"])
    assert not {c["crop_id"] for _, c in real_dev} & {c["crop_id"] for c in real_train}
    plan = exposure_plan(real_train, syn_train, SEED, REAL_PER_CYCLE)
    plan["note"] = "plan simulated over 400 steps by the shared helper; this run uses <= 150"
    dump(out / "exposure_plan.json", plan)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    teacher = load_model(a_dir / "model")
    teacher.requires_grad_(False)
    model_b = load_model(b_dir / "model")
    base_a = read(PRIOR_RUN / "baseline.json")
    base_b = read(PRIOR_RUN / "checkpoint_evals.json")["100"]
    # Re-measure both on the identical code path to confirm the archived numbers reproduce.
    re_a = run(teacher, processor, real_dev, syn_dev)
    re_b = run(model_b, processor, real_dev, syn_dev)
    repro = {
        f"{m}_{lvl}_{c}": abs(
            new["real"][lvl]["pixel"][c]["f1"] - old["real"][lvl]["pixel"][c]["f1"]
        )
        for m, new, old in (("A", re_a, base_a), ("B", re_b, base_b))
        for lvl in LEVELS
        for c in CLASSES
    }
    assert max(repro.values()) < 5e-3, repro
    br_a = boundary_recall(teacher, processor, real_dev)
    br_b = boundary_recall(model_b, processor, real_dev)
    dump(
        out / "reference_models.json",
        dict(
            A=re_a,
            B=re_b,
            boundary_A=br_a,
            boundary_B=br_b,
            archived_f1_reproduction_abs_diff=repro,
        ),
    )
    del model_b
    torch.cuda.empty_cache()
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        b_dir / "model", local_files_only=True
    ).cuda()
    criterion = with_exact_bilinear(GatedNoObjectCriterion)(num_classes=2, bilinear=True).cuda()
    pre = preflight(model, teacher, criterion, processor, real_train, out)
    if not all(pre["checks"].values()):
        dump(out / "decision.json", dict(status="PREFLIGHT_FAILED", checks=pre["checks"]))
        raise SystemExit(f"Preflight failed: {pre['checks']}")
    lam = pre["nodata_lambda"]
    protocol = dict(
        warm_start=str(b_dir),
        teacher=str(a_dir),
        continuation=pre["continuation"],
        seed=SEED,
        max_steps=STEPS,
        checkpoints=CHECKPOINTS,
        lrs=LRS,
        warmup=WARMUP,
        trainable=TRAINABLE,
        sampler=f"{REAL_PER_CYCLE} real (Stage 9E cycle) : 1 synthetic (alternating classes)",
        synthetic_augmentation=dict(
            SYN_AUG.__dict__, note="always rescaled to 0.3-0.6: thin strokes"
        ),
        supervised="GatedNoObjectCriterion (evidence-gated no-object), valid-only, exact bilinear",
        teacher_loss="guarded: real only, unmatched queries, verified-valid points, released below "
        "teacher on verified background, class KD only for non-matched non-gated queries",
        correction=f"Paint-protected NoData cap on true unobserved pixels within {BAND_PX} px of "
        "support: cap = clamp(min(teacher,0), -2); (matched query, point) pairs whose bilinear "
        "cells hold that query's observed paint are exempt. Observed paint, verified background "
        "and observed ignored pixels are never constrained by the cap.",
        nodata_lambda=lam,
        historical_gates_vs_A=GATES,
        additional_gates_vs_B=VS_B,
        stitching_gate="production stitch: 0 unsupported pixels and gap segments with unobserved "
        "pixels <= max(A, B)",
        reload_gate="strict clean reload identical outputs",
        early_stop=EARLY,
        selection="latest checkpoint passing every gate; none -> frozen stays selected",
        decision="CORRECTED_CANDIDATE_IMPROVED_PROVISIONAL if all pass; NODATA_SAFETY_BLOCKED if "
        "only NoData gates fail; RECOGNITION_REGRESSION otherwise",
        historical_decisions="unchanged; Model B remains RECOGNITION_REGRESSION",
    )
    dump(out / "protocol.json", protocol)
    psha = hashlib.sha256((out / "protocol.json").read_bytes()).hexdigest()
    print("sealed", psha, "lambda", lam, pre["checks"], flush=True)

    optimizer, membership = make_optimizer(model, LRS)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    sampler = RealFirstSampler(real_train, syn_train, SEED, REAL_PER_CYCLE)
    model_dir = MODELS / f"correction_{stamp}"
    initial = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    logs, evals, gates, grads = [], {}, {}, {}
    exposure = Counter()
    fails, stopped = 0, None
    start = time.perf_counter()
    model.train()
    for step in range(1, STEPS + 1):
        for g, lr in zip(optimizer.param_groups, base_lrs, strict=True):
            g["lr"] = lr * min(1.0, step / WARMUP)
        pool, crop = sampler.draw(step)
        real = not pool.startswith("synthetic")
        rng = np.random.default_rng(SEED * 1000 + step)
        pixels, masks, labels, valid, nodata, aug = load_step(crop, processor, rng, triples)
        dom = "real" if real else "synthetic"
        exposure[f"draws_{dom}"] += 1
        exposure[f"crop::{crop['crop_id']}"] += 1
        for lab in labels.tolist():
            exposure[f"{dom}_instances_class{lab}"] += 1
        optimizer.zero_grad(set_to_none=True)
        sup, kd, nd, losses, diag, _ = forward_losses(
            model, teacher, criterion, pixels, masks, labels, valid, nodata, real
        )
        total = sup + kd + lam * nd
        if not torch.isfinite(total):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        total.backward()
        if step in (1, 2, 3) or step % 50 == 0:
            grads[step] = gradient_audit(model)
        gnorm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        assert teacher_grad_free(teacher)
        logs.append(
            dict(
                step=step,
                pool=pool,
                crop=crop["crop_id"],
                total=float(total.detach()),
                supervised=float(sup.detach()),
                teacher=float(kd.detach()),
                nodata_cap=float(nd.detach()),
                weighted_nodata=lam * float(nd.detach()),
                **{k: float(v.detach()) for k, v in losses.items() if "aux" not in k},
                grad_norm=float(gnorm),
                lr=[g["lr"] for g in optimizer.param_groups],
                aug=aug,
                **diag,
            )
        )
        if step in CHECKPOINTS:
            res = run(model, processor, real_dev, syn_dev)
            extra = boundary_recall(model, processor, real_dev)
            res["boundary_recall"] = extra
            evals[step] = res
            ta, _ = gate_table(res, re_a)
            tb = vs_b_table(res, re_b, br_b, extra)
            table = {**ta, **tb}
            failed = [k for k, v in table.items() if not v[0]]
            reasons = []
            for lvl in LEVELS:
                for c in CLASSES:
                    if (re_b["real"][lvl]["pixel"][c]["f1"] or 0) - (
                        res["real"][lvl]["pixel"][c]["f1"] or 0
                    ) > EARLY["pixel_f1_drop_vs_b"]:
                        reasons.append(f"{lvl}_{c}_f1_drop_vs_B>0.05")
            if (
                res["real"]["nodata"]["raw_mask_pixels_in_nodata"]
                > re_b["real"]["nodata"]["raw_mask_pixels_in_nodata"]
            ):
                reasons.append("raw_nodata_above_model_B")
            if (
                res["real"]["v1"]["cross_class"]["predicted_on_verified_background"]
                > EARLY["v1_fp_ratio_vs_b"]
                * re_b["real"]["v1"]["cross_class"]["predicted_on_verified_background"]
            ):
                reasons.append("v1_fp_>1.5x_model_B")
            ck = model_dir / f"step{step:04d}"
            model.save_pretrained(ck / "model")
            processor.save_pretrained(ck / "processor")
            fails = 0 if not failed else fails + 1
            gates[step] = dict(
                passed=not failed,
                failed=failed,
                checks=table,
                early_stop_reasons=reasons,
                reload_identical=reload_identical(model, ck / "model", processor, real_dev[0]),
                checkpoint_sha256=sha256(ck / "model/model.safetensors"),
            )
            print(step, "PASS" if not failed else f"FAIL {failed}", reasons, flush=True)
            dump(out / "checkpoint_gates.json", {str(s): v for s, v in gates.items()})
            dump(out / "checkpoint_evals.json", {str(s): v for s, v in evals.items()})
            model.train()
            if reasons or fails >= 2:
                stopped = dict(step=step, reasons=reasons or ["two consecutive gate failures"])
                break
    elapsed = time.perf_counter() - start
    drift = {}
    with torch.no_grad():
        acc = {}
        for n, p in model.named_parameters():
            if n in initial:
                r = acc.setdefault(group_of(n), [0.0, 0.0])
                r[0] += float((p - initial[n]).float().norm() ** 2)
                r[1] += float(initial[n].float().norm() ** 2)
        drift = {g: (x**0.5) / max(y**0.5, 1e-12) for g, (x, y) in acc.items()}
    passing = [s for s in sorted(gates) if gates[s]["passed"] and gates[s]["reload_identical"]]
    selected = passing[-1] if passing else None
    stitching = {}
    last = max(gates)
    focus = selected or last
    cand = load_model(model_dir / f"step{focus:04d}" / "model")
    stitching["A"] = stitch_safety(teacher, processor, real_dev, out / "stitch_A")
    mb = load_model(b_dir / "model")
    stitching["B"] = stitch_safety(mb, processor, real_dev, out / "stitch_B")
    del mb
    stitching["candidate"] = stitch_safety(cand, processor, real_dev, out / "stitch_C")
    limit = max(
        stitching["A"]["stitched_gap_segments_with_unobserved"],
        stitching["B"]["stitched_gap_segments_with_unobserved"],
    )
    stitch_ok = (
        stitching["candidate"]["stitched_unsupported_pixels"] == 0
        and stitching["candidate"]["stitched_gap_segments_with_unobserved"] <= limit
    )
    failed_final = list(gates[focus]["failed"]) + ([] if stitch_ok else ["stitching_safety"])
    if not gates[focus]["reload_identical"]:
        failed_final.append("reload_identical")
    if selected and not stitch_ok:
        selected = None
    status = classify(failed_final)
    verify_frozen(checks)
    for files in (manifest["model_a"]["files"], manifest["model_b"]["files"]):
        for rel, h in files.items():
            assert sha256(PRIOR_RUN.parents[4] / rel) == h, rel
    info = dict(
        steps_completed=len(logs),
        stopped=stopped,
        elapsed_seconds=elapsed,
        seconds_per_step=elapsed / max(len(logs), 1),
        gpu_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        exposure={k: v for k, v in exposure.items() if not k.startswith("crop::")},
        distinct_crops=sum(1 for k in exposure if k.startswith("crop::")),
        max_crop_draws=max((v for k, v in exposure.items() if k.startswith("crop::")), default=0),
        relative_parameter_drift_from_B=drift,
        nodata_lambda=lam,
        optimizer=membership,
        candidate_dir=str(model_dir),
        stitching=stitching,
        stitch_ok=stitch_ok,
    )
    dump(out / "training_info.json", info)
    dump(out / "training_log.json", logs)
    dump(out / "gradient_audit.json", {str(s): v for s, v in grads.items()})
    dump(
        out / "decision.json",
        dict(
            status=status,
            selected_checkpoint=selected,
            evaluated_checkpoints=sorted(gates),
            focus_checkpoint=focus,
            failed_gates=failed_final,
            promoted=False,
            frozen_default=True,
            protocol_sha256=psha,
            model_b_historical_status="RECOGNITION_REGRESSION (unchanged)",
        ),
    )
    print(out, status, selected, failed_final, flush=True)


if __name__ == "__main__":
    sys.exit(main())
