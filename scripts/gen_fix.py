"""Targeted fix run (authorised by the user): warm start from realistic_v1 step1500.

Fixes only the failed gates of the previous run:
* weak duplicate/isolated dash proposals -> DuplicateGatedCriterion (duplicates on reviewed
  paint get no-object even when touching NoData);
* raw NoData spill -> one-sided cap: unmatched queries on all unobserved pixels within 8 px of
  support, matched queries only in the deep band (3-8 px) and never on their own paint cells.
Selection (rev2 validation tiles only) additionally requires the raw-dash and NoData criteria.
The real test (67 crops) is evaluated once at the end with the unchanged gates. It was already
used once for the step1500 decision; this is its second use and is reported as such.
"""

import hashlib
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation
from transformers import Mask2FormerForUniversalSegmentation

from correction_train import boundary_recall, stitch_safety
from gen_train import (
    CLASSES,
    GATES,
    LRS,
    TEACHER,
    acceptance,
    augment_real,
    nodata_decomposed,
    param_group,
    set_stage,
)
from regsafe_common import (
    MODELS,
    ROOT,
    dump,
    frozen_paths,
    load_model,
    load_processor,
    sha256,
    verify_frozen,
)
from regsafe_eval import run
from regsafe_train import gated_vector, reload_identical, teacher_grad_free
from roadmarking.training.deterministic_sampling import with_exact_bilinear
from roadmarking.training.duplicate_gate import DuplicateGatedCriterion
from roadmarking.training.pilot import load_crop
from roadmarking.training.teacher_guard import guarded_teacher_loss, protected_nodata_cap_loss
from stage9e_lib import development_crops, real_context, with_root

BASE = ROOT / "outputs/reports/realistic_v1_fix"
SEED = 41011
OPT_STEPS, ACCUM, EVAL_EVERY, WARMUP = 500, 3, 125, 20
LR_SCALE = 0.5
TARGET_RATIO = 0.1  # cap gradient = 10 % of supervised gradient (bounded, calibrated once)
SELECT = dict(v1_pr_drop=0.03, raw_dash_f1_drop=0.03, nodata_ratio=1.10, min_mean_gain=0.03)


def bands(valid, nodata):
    near_far = nodata & binary_dilation(valid, iterations=8)
    deep = near_far & ~binary_dilation(valid, iterations=2)
    return near_far, deep


def losses_for(model, teacher, crit, pixels, masks, labels, valid, nodata, is_real, lam):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(pixel_values=pixels, output_auxiliary_logits=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        tp = teacher(pixel_values=pixels, output_auxiliary_logits=False)
    with torch.autocast("cuda", enabled=False):
        sm, sc = pred.masks_queries_logits.float(), pred.class_queries_logits.float()
        tm, tc = tp.masks_queries_logits.float(), tp.class_queries_logits.float()
        aux = [{k: v.float() for k, v in a.items()} for a in pred.auxiliary_logits]
        crit.last_gate.clear()
        losses, matches = crit(sm, sc, [masks], [labels], [valid], [True], aux)
        crit.last_gate.clear()
        sup = crit.total(losses)
        qi, ti = matches[0]
        kd = sup * 0
        if is_real:
            gate = gated_vector(crit, sm[0], valid, qi)
            kd, _ = guarded_teacher_loss(
                sm[0], sc[0], tm[0], tc[0], crit, masks, valid, qi, gate, TEACHER
            )
        v = valid.cpu().numpy().astype(bool)
        band, deep = bands(v, nodata)
        unmatched = torch.ones(sm.shape[1], dtype=torch.bool, device=sm.device)
        unmatched[qi] = False
        nd_u, _ = protected_nodata_cap_loss(
            sm[0],
            tm[0],
            torch.from_numpy(band).cuda(),
            masks,
            valid,
            qi,
            ti,
            crit,
            apply_to=unmatched,
        )
        nd_m, _ = protected_nodata_cap_loss(
            sm[0],
            tm[0],
            torch.from_numpy(deep).cuda(),
            masks,
            valid,
            qi,
            ti,
            crit,
            apply_to=~unmatched,
        )
        nd = nd_u + nd_m
    return sup, kd, nd, losses


def select_ok(res, base, nd, nd_base):
    gaps, ok = {}, True
    for c in CLASSES:
        for m in ("precision", "recall"):
            d = (base["real"]["v1"]["pixel"][c][m] or 0) - (res["real"]["v1"]["pixel"][c][m] or 0)
            gaps[f"v1_{c}_{m}"] = round(d, 4)
            ok &= d <= SELECT["v1_pr_drop"]
    d = (base["real"]["raw"]["pixel"]["dash"]["f1"] or 0) - (
        res["real"]["raw"]["pixel"]["dash"]["f1"] or 0
    )
    gaps["raw_dash_f1"] = round(d, 4)
    ok &= d <= SELECT["raw_dash_f1_drop"]
    for k in ("fp_query_nodata", "deep_nodata"):
        gaps[k] = [nd_base[k], nd[k]]
        ok &= nd[k] <= SELECT["nodata_ratio"] * max(nd_base[k], 1)
    score = sum(res["real"]["v1"]["pixel"][c]["f1"] or 0 for c in CLASSES) / 2
    fscore = sum(base["real"]["v1"]["pixel"][c]["f1"] or 0 for c in CLASSES) / 2
    ok &= score >= fscore + SELECT["min_mean_gain"]
    return ok, score, gaps


def main():
    torch.set_num_threads(4)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = BASE / stamp
    out.mkdir(parents=True)
    dump(BASE / "latest.json", {"run": str(out)})
    prev = Path(
        json.loads((ROOT / "outputs/reports/realistic_v1_training/latest.json").read_text())["run"]
    )
    rev2 = json.loads((prev / "protocol_rev2.json").read_text())
    start_ck = MODELS / f"realistic_v1_{prev.name}" / "step1500" / "model"
    data = Path(json.loads((ROOT / "data/processed/realistic_v1/latest.json").read_text())["run"])
    man = json.loads((data / "manifest.json").read_text())
    checks, frozen, _ = frozen_paths()
    processor = load_processor()
    rc = real_context()
    ds = Path(rc["dataset"])
    by = {c["crop_id"]: c for c in rc["full"]}
    dev = {c["crop_id"]: (Path(r), c) for r, c in development_crops(rc["ctx"])}
    real_train = with_root([by[i] for i in man["real_splits"]["real_train"]["crops"]], ds)
    s6 = [by[i] for i in man["real_splits"]["real_test"]["crops"] if i in by]
    val = [(ds, c) for c in s6 if c["tile_id"] in rev2["val_tiles"]]
    test = [(ds, c) for c in s6 if c["tile_id"] not in rev2["val_tiles"]] + [
        dev[i] for i in man["real_splits"]["real_test"]["crops"] if i not in by
    ]
    assert len(val) == rev2["val_crops"] and len(test) == rev2["test_crops"]
    gen = json.loads((data / "train" / "records.json").read_text())
    for r in gen:
        r["_root"] = str(Path(r["file"]).parent)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    teacher = load_model(frozen)
    teacher.requires_grad_(False)
    stub = val[:1]
    base_val = run(teacher, processor, val, stub)
    nd_base_val = nodata_decomposed(teacher, processor, val)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        start_ck, local_files_only=True
    ).cuda()
    trainable = set_stage(model, 2)
    crit = with_exact_bilinear(DuplicateGatedCriterion)(num_classes=2, bilinear=True).cuda()
    # Calibrate the cap coefficient once on the warm-start state.
    ratios = []
    params = [p for p in model.parameters() if p.requires_grad]
    for c in sorted(real_train, key=lambda c: c["crop_id"])[::20][:20]:
        pixels, masks, labels, valid, rgb = load_crop(c["_root"], c, processor, "cuda")
        nod = ~valid[0].cpu().numpy().astype(bool) & (rgb[..., 0] == 0)
        sup, kd, nd, _ = losses_for(
            model, teacher, crit, pixels, masks[0], labels[0], valid[0], nod, True, 1.0
        )
        if float(nd.detach()) <= 0:
            continue
        gs = torch.autograd.grad(sup, params, retain_graph=True, allow_unused=True)
        gn = torch.autograd.grad(nd, params, allow_unused=True)

        def n(g):
            return sum(float(x.float().norm() ** 2) for x in g if x is not None) ** 0.5

        ratios.append(n(gn) / max(n(gs), 1e-9))
    lam = (
        float(np.clip(float(f"{TARGET_RATIO / np.median(ratios):.1g}"), 1.0, 50.0))
        if len(ratios) >= 5
        else 5.0
    )
    model.zero_grad(set_to_none=True)
    protocol = dict(
        warm_start=str(start_ck),
        warm_start_sha256=sha256(start_ck / "model.safetensors"),
        fixes=[
            "DuplicateGatedCriterion(min_paint_fraction=0.5)",
            "cap unmatched queries on NoData <= 8 px; "
            "matched queries only 3-8 px, own paint exempt",
        ],
        nodata_lambda=lam,
        calibration_ratios=ratios,
        target_ratio=TARGET_RATIO,
        lrs={k: v * LR_SCALE for k, v in LRS.items()},
        trainable=trainable,
        opt_steps=OPT_STEPS,
        accum=ACCUM,
        eval_every=EVAL_EVERY,
        data=str(data),
        real_fraction=0.5,
        selection=SELECT,
        selection_set=f"rev2 val {rev2['val_tiles']} ({len(val)} crops)",
        acceptance_gates=GATES,
        test=f"{len(test)} crops; second use of this test set",
        frozen_default=True,
    )
    dump(out / "protocol.json", protocol)
    psha = hashlib.sha256((out / "protocol.json").read_bytes()).hexdigest()
    print("sealed", psha, "lambda", lam, flush=True)
    groups = {}
    for n_, p in model.named_parameters():
        if p.requires_grad:
            groups.setdefault(param_group(n_), []).append(p)
    opt = torch.optim.AdamW(
        [dict(params=v, lr=LRS[k] * LR_SCALE, name=k) for k, v in groups.items()], weight_decay=0.01
    )
    rng = np.random.default_rng(SEED)
    order = {"g": [], "r": []}
    exposure = Counter()
    sel, best, logs = {}, None, []
    draw = 0
    t0 = time.perf_counter()
    model.train()
    for step in range(1, OPT_STEPS + 1):
        for g in opt.param_groups:
            g["lr"] = LRS[g["name"]] * LR_SCALE * min(1.0, step / WARMUP)
        opt.zero_grad(set_to_none=True)
        tot = Counter()
        for _ in range(ACCUM):
            draw += 1
            is_real = draw % 2 == 0
            key = "r" if is_real else "g"
            pool = real_train if is_real else gen
            if not order[key]:
                order[key] = list(rng.permutation(len(pool)))
            c = pool[order[key].pop()]
            pixels, masks, labels, valid, rgb = load_crop(c["_root"], c, processor, "cuda")
            if is_real:
                pixels, m0, v0, rgb = augment_real(pixels, masks[0], valid[0], rgb, rng)
            else:
                m0, v0 = masks[0], valid[0]
            nod = ~v0.cpu().numpy().astype(bool) & (rgb[..., 0] == 0)
            try:
                sup, kd, nd, _losses = losses_for(
                    model, teacher, crit, pixels, m0, labels[0], v0, nod, is_real, lam
                )
            except ValueError as err:
                if "sampled valid foreground" not in str(err):
                    raise
                exposure["skipped_unsampleable"] += 1
                continue
            total = (sup + kd + lam * nd) / ACCUM
            if not torch.isfinite(total):
                raise RuntimeError(f"nonfinite at {step}")
            total.backward()
            exposure["real" if is_real else "generated"] += 1
            tot["sup"] += float(sup.detach()) / ACCUM
            tot["kd"] += float(kd.detach()) / ACCUM
            tot["cap"] += lam * float(nd.detach()) / ACCUM
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True
        )
        opt.step()
        assert teacher_grad_free(teacher)
        logs.append(dict(step=step, **tot, duplicates=sum(crit.last_duplicates)))
        crit.last_duplicates.clear()
        if step % 50 == 0:
            print(
                step,
                {k: round(v, 3) for k, v in tot.items()},
                round(time.perf_counter() - t0),
                flush=True,
            )
        if step % EVAL_EVERY == 0:
            res = run(model, processor, val, stub)
            nd_v = nodata_decomposed(model, processor, val)
            ok, score, gaps = select_ok(res, base_val, nd_v, nd_base_val)
            ck = MODELS / f"realistic_v1_fix_{stamp}" / f"step{step:04d}"
            model.save_pretrained(ck / "model")
            processor.save_pretrained(ck / "processor")
            sel[step] = dict(
                eligible=ok, score=score, gaps=gaps, sha256=sha256(ck / "model/model.safetensors")
            )
            if ok and (best is None or score > sel[best]["score"]):
                best = step
            print("eval", step, sel[step], flush=True)
            dump(out / "selection.json", {str(k): v for k, v in sel.items()})
            model.train()
    dump(out / "training_log.json", logs)
    dump(
        out / "training_info.json",
        dict(
            seconds=time.perf_counter() - t0,
            exposure=dict(exposure),
            selected=best,
            gpu_peak_mib=torch.cuda.max_memory_allocated() / 2**20,
        ),
    )
    if best is None:
        dump(
            out / "decision.json",
            dict(
                status="NO_SAFE_IMPROVEMENT",
                reason="no checkpoint met selection",
                frozen_default=True,
                protocol_sha256=psha,
            ),
        )
        print("NO ELIGIBLE CHECKPOINT", flush=True)
        return
    del model
    torch.cuda.empty_cache()
    cpath = MODELS / f"realistic_v1_fix_{stamp}" / f"step{best:04d}" / "model"
    cand = load_model(cpath)
    all_f, all_c = run(teacher, processor, test, stub), run(cand, processor, test, stub)
    nd_f, nd_c = nodata_decomposed(teacher, processor, test), nodata_decomposed(
        cand, processor, test
    )
    dev_test = [x for x in test if x[1]["crop_id"] in dev]
    try:
        stitch = dict(
            frozen=stitch_safety(teacher, processor, dev_test, out / "stitch_f"),
            candidate=stitch_safety(cand, processor, dev_test, out / "stitch_c"),
        )
    except Exception as e:
        stitch = dict(error=repr(e))
    extra = dict(
        stitch=stitch.get("candidate"), reload=reload_identical(cand, cpath, processor, test[0])
    )
    table = acceptance(all_c, all_f, nd_c, nd_f, extra)
    failed = [k for k, v in table.items() if not v[0]]
    status = (
        "IMPROVED_PROVISIONAL"
        if not failed
        else (
            "RECOGNITION_REGRESSION"
            if any(("drop" in f) or ("false_positive" in f) for f in failed)
            else "NO_SAFE_IMPROVEMENT"
        )
    )
    br = dict(
        frozen=boundary_recall(teacher, processor, test),
        candidate=boundary_recall(cand, processor, test),
    )
    verify_frozen(checks)
    dump(
        out / "test_results.json",
        dict(
            frozen=all_f,
            candidate=all_c,
            nodata=dict(frozen=nd_f, candidate=nd_c),
            boundary_recall=br,
            stitching=stitch,
            test_crops=len(test),
        ),
    )
    dump(
        out / "decision.json",
        dict(
            status=status,
            selected=best,
            candidate=str(cpath),
            gates=table,
            failed=failed,
            frozen_default=status != "IMPROVED_PROVISIONAL",
            promoted=False,
            protocol_sha256=psha,
            test_use="second",
        ),
    )
    print("DECISION", status, best, failed, flush=True)


if __name__ == "__main__":
    sys.exit(main())
