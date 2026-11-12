"""One sealed fine-tune on realistic_v1 with staged unfreezing; frozen model stays the default.

Selection uses ONLY the real validation tiles (P512) plus generated validation; the real test
set (77 expanded S6 crops + 14 development crops, >= 500 m from training) is evaluated once at
the end. A candidate is accepted only if it passes every registered gate against the frozen
model on the real test set; otherwise the frozen model remains selected.
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
from transformers import Mask2FormerForUniversalSegmentation

from correction_train import boundary_recall, stitch_safety
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
from regsafe_eval import CLASSES, run
from regsafe_train import gated_vector, reload_identical, teacher_grad_free
from roadmarking.training.deterministic_sampling import with_exact_bilinear
from roadmarking.training.gated_criterion import GatedNoObjectCriterion
from roadmarking.training.pilot import load_crop
from roadmarking.training.teacher_guard import guarded_teacher_loss
from stage9d_train_lib import gradient_audit, group_of
from stage9e_lib import real_context, with_root

BASE = ROOT / "outputs/reports/realistic_v1_training"
SEED = 41001
OPT_STEPS = 1500
ACCUM = 3
UNFREEZE_AT = 750
EVAL_EVERY = 250
REAL_EVERY = 2  # 1 real crop per 2 draws (1 generated : 1 real)
WARMUP = 50
LRS = dict(
    query_decoder=1e-5,
    mask_embedder=1e-5,
    class_predictor=5e-5,
    pixel_decoder=5e-6,
    backbone_stage4=2e-6,
)
STAGE4 = (
    "model.pixel_level_module.encoder.swin.encoder.layers.3.",
    "model.pixel_level_module.encoder.hidden_states_norms.stage4.",
)
TEACHER = {"mask": 5.0, "class": 2.0}
GEN_VAL_N = 150
SELECT_TOL = 0.03  # per-class V1 precision/recall drop allowed vs frozen on REAL VAL
GATES = dict(
    v1_pixel_drop=0.02,  # precision, recall, F1 per class vs frozen (real test)
    raw_pixel_f1_drop=0.03,
    v1_instance_f1_drop=0.05,
    v1_fp_ratio=1.25,
    mean_v1_f1_gain=0.03,
    nodata_fp_query_ratio=1.10,
    nodata_deep_ratio=1.10,
)


def param_group(name):
    if name.startswith(STAGE4):
        return "backbone_stage4"
    g = group_of(name)
    return g


def set_stage(model, stage):
    groups = ["query_decoder", "mask_embedder", "class_predictor", "pixel_decoder"]
    if stage == 2:
        groups.append("backbone_stage4")
    model.requires_grad_(False)
    counts = Counter()
    for n, p in model.named_parameters():
        g = param_group(n)
        if g in groups:
            p.requires_grad_(True)
            counts[g] += p.numel()
    return dict(counts)


def augment_real(pixels, masks, valid, rgb, rng):
    k, flip = int(rng.integers(4)), bool(rng.random() < 0.5)

    def geo(t):
        t = torch.rot90(t, k, dims=(-2, -1))
        return torch.flip(t, dims=(-1,)) if flip else t

    rgb = np.rot90(rgb, k, axes=(0, 1))
    rgb = np.flip(rgb, 1) if flip else rgb
    return geo(pixels), geo(masks), geo(valid), np.ascontiguousarray(rgb)


@torch.no_grad()
def nodata_decomposed(model, processor, dev):
    """Prospective NoData diagnostics: FP-query leakage and deep (3-8 px) leakage."""
    from scipy.ndimage import binary_dilation

    from stage9e_lib import proposals

    out = Counter()
    for root, crop in dev:
        pixel, masks, labels, validity, rgb = load_crop(root, crop, processor, "cuda")
        valid = validity[0].cpu().numpy().astype(bool)
        nod = ~valid & (rgb[..., 0] == 0)
        deep = nod & binary_dilation(valid, iterations=8) & ~binary_dilation(valid, iterations=2)
        truth = masks[0].bool().cpu().numpy()
        tl = labels[0].cpu().numpy()
        pr = proposals(model, pixel, valid)
        fpq = np.zeros_like(valid)
        allq = np.zeros_like(valid)
        for m, lab in zip(pr["masks"], pr["labels"], strict=True):
            hit = any((m & truth[k] & valid).sum() >= 10 for k in range(len(tl)) if tl[k] == lab)
            allq |= m
            if not hit:
                fpq |= m
        out["fp_query_nodata"] += int((fpq & nod).sum())
        out["deep_nodata"] += int((allq & deep).sum())
        out["raw_union_nodata"] += int((allq & nod).sum())
        pred_valid = int((allq & valid).sum())
        out["predicted_valid_px"] += pred_valid
    out["raw_nodata_per_100_valid"] = (
        100 * out["raw_union_nodata"] / max(out["predicted_valid_px"], 1)
    )
    return dict(out)


def selection_ok(res, base):
    ok, gaps = True, {}
    for c in CLASSES:
        for m in ("precision", "recall"):
            d = (base["real"]["v1"]["pixel"][c][m] or 0) - (res["real"]["v1"]["pixel"][c][m] or 0)
            gaps[f"{c}_{m}"] = round(d, 4)
            ok &= d <= SELECT_TOL
    score = sum(res["real"]["v1"]["pixel"][c]["f1"] or 0 for c in CLASSES) / 2
    return ok, score, gaps


def acceptance(res, base, nd, nd_base, extra):
    checks = {}
    r, b = res["real"], base["real"]
    for c in CLASSES:
        for m in ("precision", "recall", "f1"):
            d = (b["v1"]["pixel"][c][m] or 0) - (r["v1"]["pixel"][c][m] or 0)
            checks[f"v1_{c}_{m}_drop<={GATES['v1_pixel_drop']}"] = (
                d <= GATES["v1_pixel_drop"],
                round(d, 4),
            )
        d = (b["raw"]["pixel"][c]["f1"] or 0) - (r["raw"]["pixel"][c]["f1"] or 0)
        checks[f"raw_{c}_f1_drop<={GATES['raw_pixel_f1_drop']}"] = (
            d <= GATES["raw_pixel_f1_drop"],
            round(d, 4),
        )
        d = (b["v1"]["instance_iou50"][c]["f1"] or 0) - (r["v1"]["instance_iou50"][c]["f1"] or 0)
        checks[f"v1_{c}_instance_f1_drop<={GATES['v1_instance_f1_drop']}"] = (
            d <= GATES["v1_instance_f1_drop"],
            round(d, 4),
        )
    fb = b["v1"]["cross_class"]["predicted_on_verified_background"]
    fr = r["v1"]["cross_class"]["predicted_on_verified_background"]
    checks[f"v1_false_positive_px<={GATES['v1_fp_ratio']}x"] = (
        fr <= GATES["v1_fp_ratio"] * max(fb, 1),
        [fb, fr],
    )
    gain = (
        sum((r["v1"]["pixel"][c]["f1"] or 0) - (b["v1"]["pixel"][c]["f1"] or 0) for c in CLASSES)
        / 2
    )
    checks[f"mean_v1_f1_gain>={GATES['mean_v1_f1_gain']}"] = (
        gain >= GATES["mean_v1_f1_gain"],
        round(gain, 4),
    )
    acc = r["nodata"].get("v1_accepted_after_clipping_in_nodata", 0)
    checks["v1_accepted_after_clipping_in_nodata==0"] = (acc == 0, acc)
    for key, ratio in (
        ("fp_query_nodata", GATES["nodata_fp_query_ratio"]),
        ("deep_nodata", GATES["nodata_deep_ratio"]),
    ):
        checks[f"{key}<={ratio}x"] = (
            nd[key] <= ratio * max(nd_base[key], 1),
            [nd_base[key], nd[key]],
        )
    if extra.get("stitch") is not None:
        checks["stitched_unsupported_pixels==0"] = (
            extra["stitch"]["stitched_unsupported_pixels"] == 0,
            extra["stitch"]["stitched_unsupported_pixels"],
        )
    checks["reload_identical"] = (extra["reload"], extra["reload"])
    return checks


def main():
    torch.set_num_threads(4)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = BASE / stamp
    out.mkdir(parents=True)
    dump(BASE / "latest.json", {"run": str(out)})
    checks, frozen, _ = frozen_paths()
    processor = load_processor()
    data = Path(json.loads((ROOT / "data/processed/realistic_v1/latest.json").read_text())["run"])
    man = json.loads((data / "manifest.json").read_text())
    rc = real_context()
    ds = Path(rc["dataset"])
    by_id = {c["crop_id"]: c for c in rc["full"]}
    from stage9e_lib import development_crops

    dev = {c["crop_id"]: (Path(r), c) for r, c in development_crops(rc["ctx"])}
    real_train = with_root([by_id[i] for i in man["real_splits"]["real_train"]["crops"]], ds)
    real_val = [(ds, by_id[i]) for i in man["real_splits"]["real_val"]["crops"]]
    real_test = [
        (ds, by_id[i]) if i in by_id else dev[i] for i in man["real_splits"]["real_test"]["crops"]
    ]
    test_clean = [x for x in real_test if x[1]["crop_id"] in by_id]
    test_dev = [x for x in real_test if x[1]["crop_id"] not in by_id]
    gen = {
        s: json.loads((data / s / "records.json").read_text())
        for s in ("train", "val", "stress_test")
    }
    for s in gen.values():
        for r in s:
            r["_root"] = str(Path(r["file"]).parent)
    gen_val = [(Path(r["_root"]), r) for r in gen["val"][:GEN_VAL_N]]
    stress = [(Path(r["_root"]), r) for r in gen["stress_test"]]
    # Leakage assertions.
    tr_ids = {c["crop_id"] for c in real_train}
    assert not tr_ids & {c["crop_id"] for _, c in real_val + real_test}
    assert not {r["tile_id"] for r in gen["train"]} & {
        c["tile_id"] for _, c in real_val + test_clean
    }

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    teacher = load_model(frozen)
    teacher.requires_grad_(False)
    base_val = run(teacher, processor, real_val, gen_val)
    dump(out / "frozen_val.json", base_val)
    protocol = dict(
        data=str(data),
        data_manifest_sha256=sha256(data / "manifest.json"),
        seed=SEED,
        optimizer_steps=OPT_STEPS,
        accumulation=ACCUM,
        unfreeze_backbone_stage4_at=UNFREEZE_AT,
        lrs=LRS,
        warmup=WARMUP,
        real_fraction=1 / REAL_EVERY,
        loss="GatedNoObjectCriterion (valid-only, exact bilinear) + guarded teacher on real crops",
        teacher_weights=TEACHER,
        selection=f"real VAL (P512) mean V1 F1, eligible only if every per-class V1 "
        f"precision/recall drop vs frozen <= {SELECT_TOL}; evaluated every {EVAL_EVERY} steps",
        acceptance_gates_on_real_test=GATES,
        real_test=dict(clean=len(test_clean), reused_dev=len(test_dev)),
        nodata_policy="production clipping unchanged; accepted NoData must be 0; "
        "prospective decomposed raw diagnostics (FP-query and 3-8 px deep NoData) "
        "gated at 1.10x frozen",
        frozen_default=True,
    )
    dump(out / "protocol.json", protocol)
    psha = hashlib.sha256((out / "protocol.json").read_bytes()).hexdigest()
    print("sealed", psha, flush=True)

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        frozen, local_files_only=True
    ).cuda()
    stage_counts = {1: set_stage(model, 1)}
    assert "backbone_stage4" not in stage_counts[1]

    def make_opt(net):
        groups = {}
        for n, p in net.named_parameters():
            if p.requires_grad:
                groups.setdefault(param_group(n), []).append(p)
        return torch.optim.AdamW(
            [dict(params=v, lr=LRS[k], name=k) for k, v in groups.items()], weight_decay=0.01
        )

    optimizer = make_opt(model)
    criterion = with_exact_bilinear(GatedNoObjectCriterion)(num_classes=2, bilinear=True).cuda()
    rng = np.random.default_rng(SEED)
    order = {
        "gen": list(rng.permutation(len(gen["train"]))),
        "real": list(rng.permutation(len(real_train))),
    }
    logs, evals, sel = [], {}, {}
    exposure = Counter()
    draw = 0
    best = None
    start = time.perf_counter()
    model.train()
    for step in range(1, OPT_STEPS + 1):
        if step == UNFREEZE_AT + 1:
            stage_counts[2] = set_stage(model, 2)
            new = [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and param_group(n) == "backbone_stage4"
            ]
            optimizer.add_param_group(
                dict(params=new, lr=LRS["backbone_stage4"], name="backbone_stage4")
            )
        for g in optimizer.param_groups:
            g["lr"] = LRS[g["name"]] * min(1.0, step / WARMUP)
        optimizer.zero_grad(set_to_none=True)
        totals = Counter()
        for _ in range(ACCUM):
            draw += 1
            is_real = draw % REAL_EVERY == 0
            if is_real:
                if not order["real"]:
                    order["real"] = list(rng.permutation(len(real_train)))
                crop = real_train[order["real"].pop()]
                pixels, masks, labels, valid, rgb = load_crop(
                    crop["_root"], crop, processor, "cuda"
                )
                pixels, m0, v0, rgb = augment_real(pixels, masks[0], valid[0], rgb, rng)
                masks, labels, valid = m0, labels[0], v0
            else:
                if not order["gen"]:
                    order["gen"] = list(rng.permutation(len(gen["train"])))
                crop = gen["train"][order["gen"].pop()]
                pixels, masks, labels, valid, rgb = load_crop(
                    crop["_root"], crop, processor, "cuda"
                )
                masks, labels, valid = masks[0], labels[0], valid[0]
            exposure["real" if is_real else "generated"] += 1
            for lab in labels.tolist():
                exposure[f"{'real' if is_real else 'gen'}_class{lab}"] += 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(pixel_values=pixels, output_auxiliary_logits=True)
            with torch.autocast("cuda", enabled=False):
                sm, sc = pred.masks_queries_logits.float(), pred.class_queries_logits.float()
                aux = [{k: v.float() for k, v in a.items()} for a in pred.auxiliary_logits]
                criterion.last_gate.clear()
                try:
                    losses, matches = criterion(sm, sc, [masks], [labels], [valid], [True], aux)
                except ValueError as err:  # tiny target without a sampled point: skip, count
                    if "sampled valid foreground" not in str(err):
                        raise
                    exposure["skipped_unsampleable"] += 1
                    criterion.last_gate.clear()
                    continue
                criterion.last_gate.clear()
                loss = criterion.total(losses)
                kd = loss * 0
                if is_real:
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        tp = teacher(pixel_values=pixels, output_auxiliary_logits=False)
                    qi = matches[0][0]
                    gate = gated_vector(criterion, sm[0], valid, qi)
                    kd, _ = guarded_teacher_loss(
                        sm[0],
                        sc[0],
                        tp.masks_queries_logits[0].float(),
                        tp.class_queries_logits[0].float(),
                        criterion,
                        masks,
                        valid,
                        qi,
                        gate,
                        TEACHER,
                    )
                total = (loss + kd) / ACCUM
            if not torch.isfinite(total):
                raise RuntimeError(f"nonfinite loss at step {step}")
            total.backward()
            totals["supervised"] += float(loss.detach()) / ACCUM
            totals["teacher"] += float(kd.detach()) / ACCUM
            totals["class"] += float(losses["loss_class"].detach()) / ACCUM
            totals["dice"] += float(losses["loss_dice"].detach()) / ACCUM
        if step in (1, 2, UNFREEZE_AT + 1) or step % 250 == 0:
            g = gradient_audit(model)
            assert all(v["nonfinite"] == 0 for v in g.values())
            dump(out / f"grad_step{step}.json", g)
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        assert teacher_grad_free(teacher)
        logs.append(dict(step=step, grad_norm=float(norm), **totals))
        if step % 50 == 0:
            print(
                step,
                {k: round(v, 3) for k, v in totals.items()},
                round(time.perf_counter() - start),
                flush=True,
            )
        if step % EVAL_EVERY == 0:
            res = run(model, processor, real_val, gen_val)
            ok, score, gaps = selection_ok(res, base_val)
            evals[step] = res
            ck = MODELS / f"realistic_v1_{stamp}" / f"step{step:04d}"
            model.save_pretrained(ck / "model")
            processor.save_pretrained(ck / "processor")
            sel[step] = dict(
                eligible=ok,
                score=score,
                gaps=gaps,
                sha256=sha256(ck / "model/model.safetensors"),
                frozen_score=sum(base_val["real"]["v1"]["pixel"][c]["f1"] or 0 for c in CLASSES)
                / 2,
            )
            if ok and (best is None or score > sel[best]["score"]):
                best = step
            print("eval", step, sel[step], flush=True)
            dump(out / "selection.json", {str(k): v for k, v in sel.items()})
            dump(out / "val_evals.json", {str(k): v for k, v in evals.items()})
            model.train()
    elapsed = time.perf_counter() - start
    dump(out / "training_log.json", logs)
    info = dict(
        elapsed_seconds=elapsed,
        opt_steps=OPT_STEPS,
        forwards=draw,
        exposure=dict(exposure),
        stage_trainable=stage_counts,
        gpu_peak_mib=torch.cuda.max_memory_allocated() / 2**20,
        selected_step=best,
    )
    dump(out / "training_info.json", info)
    if best is None:
        dump(
            out / "decision.json",
            dict(
                status="NO_SAFE_IMPROVEMENT",
                reason="no checkpoint eligible on real val",
                frozen_default=True,
                protocol_sha256=psha,
            ),
        )
        print("NO ELIGIBLE CHECKPOINT", flush=True)
        return
    # Final, single evaluation on the held-out real test set.
    del model
    torch.cuda.empty_cache()
    cpath = MODELS / f"realistic_v1_{stamp}" / f"step{best:04d}" / "model"
    cand = load_model(cpath)
    stress_n = stress[:300]
    res_f = {
        "clean": run(teacher, processor, test_clean, stress_n),
        "dev": run(teacher, processor, test_dev, stress_n[:1]),
    }
    res_c = {
        "clean": run(cand, processor, test_clean, stress_n),
        "dev": run(cand, processor, test_dev, stress_n[:1]),
    }
    all_f = run(teacher, processor, real_test, stress_n[:1])
    all_c = run(cand, processor, real_test, stress_n[:1])
    nd_f, nd_c = nodata_decomposed(teacher, processor, real_test), nodata_decomposed(
        cand, processor, real_test
    )
    br = dict(
        frozen=boundary_recall(teacher, processor, real_test),
        candidate=boundary_recall(cand, processor, real_test),
    )
    stitch = None
    try:
        stitch = dict(
            frozen=stitch_safety(teacher, processor, test_dev, out / "stitch_f"),
            candidate=stitch_safety(cand, processor, test_dev, out / "stitch_c"),
        )
    except Exception as e:
        stitch = dict(error=repr(e))
    extra = dict(
        stitch=stitch.get("candidate") if "candidate" in stitch else None,
        reload=reload_identical(cand, cpath, processor, real_test[0]),
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
    verify_frozen(checks)
    dump(
        out / "test_results.json",
        dict(
            frozen=res_f,
            candidate=res_c,
            frozen_all=all_f,
            candidate_all=all_c,
            nodata=dict(frozen=nd_f, candidate=nd_c),
            boundary_recall=br,
            stitching=stitch,
        ),
    )
    dump(
        out / "decision.json",
        dict(
            status=status,
            selected_step=best,
            candidate=str(cpath),
            gates=table,
            failed=failed,
            frozen_default=status != "IMPROVED_PROVISIONAL",
            promoted=False,
            protocol_sha256=psha,
        ),
    )
    print("DECISION", status, failed, flush=True)


if __name__ == "__main__":
    sys.exit(main())
