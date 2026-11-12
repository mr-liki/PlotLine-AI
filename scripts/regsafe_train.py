"""Steps 3-7: one sealed, real-first, teacher-guarded fine-tune with staged no-regression gates.

Run once. The protocol (sampling, losses, coefficients, gates, early stop, selection) is
written and hashed before the first optimisation step. Evaluation at 100/200/300/400 steps
against the frozen baseline; stop on significant regression or two consecutive gate
failures. Candidates are stored under models/mask2former_swin_s/regsafe_<run>/ and never
replace the frozen checkpoint.
"""

import hashlib
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation
from transformers import Mask2FormerForUniversalSegmentation

from regsafe_common import (
    BASE,
    MODELS,
    dump,
    frozen_paths,
    load_model,
    load_processor,
    read,
    verify_frozen,
)
from regsafe_eval import EARLY_STOP, GATES, early_stop, gate_table, mean_v1_f1_gain, run
from roadmarking.training.deterministic_sampling import with_exact_bilinear
from roadmarking.training.gated_criterion import GatedNoObjectCriterion
from roadmarking.training.pilot import load_crop
from roadmarking.training.synthetic_augment import AugmentConfig, augment
from roadmarking.training.teacher_guard import guarded_teacher_loss, nodata_cap_loss
from stage9d_train_lib import gradient_audit, group_of, make_optimizer, set_trainable
from stage9e_lib import Sampler, development_crops, real_context, with_root
from stage9e_recipes import LOW

SEED = 9101
STEPS = 400
CHECKPOINTS = (100, 200, 300, 400)
REAL_PER_CYCLE, SYN_PER_CYCLE = 4, 1
TEACHER = {"mask": 5.0, "class": 2.0}
NEAR_PX, FAR_PX = 2, 8
NODATA_TARGET_GRAD_RATIO = 0.25
NODATA_LAMBDA_BOUNDS = (0.5, 5.0)
MIN_REAL_CLASS_SHARE = 0.70
TRAINABLE = ["query_decoder", "mask_embedder", "class_predictor"]


class RealFirstSampler:
    """Fixed schedule: per cycle REAL_PER_CYCLE Stage 9E real-cycle draws, then synthetic
    draws alternating solid-positive and dash-positive pools (uniform permutations)."""

    def __init__(self, real, synthetic, seed, real_per_cycle=REAL_PER_CYCLE):
        self.k = real_per_cycle
        self.real = Sampler(real, "cycle", seed)
        self.syn = [
            Sampler([c for c in synthetic if cls in c["labels"]], "uniform", seed + 11 + cls)
            for cls in (0, 1)
        ]
        self.nsyn = 0
        self.nreal = 0

    def draw(self, step):
        offset = (step - 1) % (self.k + SYN_PER_CYCLE)
        if offset < self.k:
            self.nreal += 1
            pool, crop = self.real.draw(self.nreal)
            return f"real_{pool}", crop
        cls = self.nsyn % 2
        self.nsyn += 1
        return f"synthetic_class{cls}", self.syn[cls].draw(self.nsyn)[1]

    def sizes(self):
        return {
            **self.real.sizes(),
            **{f"synthetic_class{c}": s.sizes()["all"] for c, s in enumerate(self.syn)},
        }


def exposure_plan(real, synthetic, seed, k):
    s = RealFirstSampler(real, synthetic, seed, k)
    inst = Counter()
    crops = Counter()
    groups = Counter()
    for step in range(1, STEPS + 1):
        pool, crop = s.draw(step)
        dom = "synthetic" if pool.startswith("synthetic") else "real"
        crops[(dom, crop["crop_id"])] += 1
        for lab in crop["labels"]:
            inst[f"{dom}_class{lab}"] += 1
        for g in crop.get("global_instance_ids", []):
            groups[g] += 1
        inst[f"{dom}_draws"] += 1
    share = {
        c: inst[f"real_class{c}"] / max(inst[f"real_class{c}"] + inst[f"synthetic_class{c}"], 1)
        for c in (0, 1)
    }
    return dict(
        real_per_cycle=k,
        draws=dict(real=inst["real_draws"], synthetic=inst["synthetic_draws"]),
        instance_exposures={k2: v for k2, v in inst.items() if "class" in k2},
        real_share_by_class={"solid": share[0], "dash": share[1]},
        distinct_real_crops=sum(1 for d, _ in crops if d == "real"),
        distinct_synthetic_images=sum(1 for d, _ in crops if d == "synthetic"),
        max_draws_per_real_crop=max((v for (d, _), v in crops.items() if d == "real"), default=0),
        max_draws_per_synthetic_image=max(
            (v for (d, _), v in crops.items() if d == "synthetic"), default=0
        ),
        physical_groups_exposed=len(groups),
        physical_group_exposure_p50=float(np.median(list(groups.values()))) if groups else 0,
        physical_group_exposure_p95=(
            float(np.percentile(list(groups.values()), 95)) if groups else 0
        ),
        physical_group_exposure_max=max(groups.values(), default=0),
        passes=min(share.values()) >= MIN_REAL_CLASS_SHARE,
    )


def bands(valid, nodata):
    near = nodata & binary_dilation(valid, iterations=NEAR_PX)
    far = nodata & binary_dilation(valid, iterations=FAR_PX) & ~near
    return near, far


def load_step(crop, processor, rng, triples):
    """Return pixels, targets, labels, valid (cuda) and numpy nodata, record."""
    if crop["geography"] != "synthetic":
        pixels, masks, labels, valid, rgb = load_crop(crop["_root"], crop, processor, "cuda")
        v = valid[0].cpu().numpy().astype(bool)
        return pixels, masks[0], labels[0], valid[0], ~v & (rgb[..., 0] == 0), {}
    with np.load(crop["file"]) as d:
        gray, m, lab, v = d["rgb"][..., 0], d["masks"], d["labels"], d["validity"]
    a = augment(gray, m, lab, v, rng, triples, AugmentConfig())
    rgb = np.repeat(a["gray"][..., None], 3, axis=2)
    pixels = processor(images=rgb, return_tensors="pt")["pixel_values"].cuda()
    assert pixels.shape[-2:] == (512, 512)
    return (
        pixels,
        torch.from_numpy(a["masks"].astype(np.float32)).cuda(),
        torch.from_numpy(a["labels"]).cuda(),
        torch.from_numpy(a["valid"]).cuda(),
        a["nodata"],
        a["record"],
    )


def gated_vector(criterion, masks, valid, qi):
    unmatched = torch.ones(len(masks), dtype=torch.bool, device=masks.device)
    unmatched[qi] = False
    if bool(valid.all()):
        return unmatched
    frac, _ = criterion.claimed_valid_fraction(masks, valid)
    return unmatched & (frac >= criterion.min_valid_fraction)


def forward_losses(model, teacher, criterion, pixels, masks, labels, valid, nodata, real, lam):
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
        qi = matches[0][0]
        diag = {}
        kd = sup * 0
        if real:
            gate = gated_vector(criterion, sm[0], valid, qi)
            kd, d = guarded_teacher_loss(
                sm[0], sc[0], tm[0], tc[0], criterion, masks, valid, qi, gate, TEACHER
            )
            diag.update(d, gated_queries=int(gate.sum()))
        v = valid.cpu().numpy().astype(bool)
        near, far = bands(v, nodata)
        nd, d = nodata_cap_loss(
            sm[0], tm[0], torch.from_numpy(near).cuda(), torch.from_numpy(far).cuda(), criterion
        )
        diag.update(d)
    return sup, kd, nd, losses, diag


def preflight(model, teacher, criterion, processor, real_train, triples, out):
    """Trainable/optimizer audit, valid-mask gradients, teacher behavior, lambda calibration."""
    report = {}
    report["trainable"] = set_trainable(model, TRAINABLE)
    assert report["trainable"]["backbone"]["trainable_params"] == 0
    assert report["trainable"]["pixel_decoder"]["trainable_params"] == 0
    assert all(report["trainable"][g]["trainable_params"] > 0 for g in TRAINABLE)
    opt, membership = make_optimizer(model, LOW)
    assert membership["optimizer_equals_requires_grad"]
    report["optimizer"] = membership
    del opt
    assert not any(p.requires_grad for p in teacher.parameters())
    probe = sorted(real_train, key=lambda c: c["crop_id"])[::37][:12]
    rows = []
    for crop in probe:
        pixels, masks, labels, valid, nodata, _ = load_step(crop, processor, None, triples)
        if not len(labels):
            continue
        # 1) Ignored ground truth has no effect on supervised loss.
        model.zero_grad(set_to_none=True)
        sup, kd, nd, _losses, diag = forward_losses(
            model, teacher, criterion, pixels, masks, labels, valid, nodata, True, 1.0
        )
        altered = masks.clone()
        altered[:, ~valid] = 1 - altered[:, ~valid]
        sup2, *_ = forward_losses(
            model, teacher, criterion, pixels, altered, labels, valid, nodata, True, 1.0
        )
        # 2) Gradients of each term at the frozen initialisation.
        params = [p for p in model.parameters() if p.requires_grad]
        g_sup = torch.autograd.grad(sup, params, retain_graph=True, allow_unused=True)
        g_kd = torch.autograd.grad(kd, params, retain_graph=True, allow_unused=True)
        g_nd = (
            torch.autograd.grad(nd, params, allow_unused=True)
            if nd.requires_grad and float(nd) > 0
            else [None]
        )

        def norm(gs):
            return float(sum(float(g.float().norm() ** 2) for g in gs if g is not None) ** 0.5)

        rows.append(
            dict(
                crop=crop["crop_id"],
                finite=bool(torch.isfinite(sup) and torch.isfinite(kd) and torch.isfinite(nd)),
                ignored_gt_invariance=abs(float(sup) - float(sup2))
                < 1e-4 * max(1, abs(float(sup))),
                sup=float(sup),
                kd=float(kd),
                nodata=float(nd),
                grad_sup=norm(g_sup),
                grad_kd=norm(g_kd),
                grad_nodata=norm(g_nd),
                **{k: v for k, v in diag.items() if isinstance(v, (int, float))},
            )
        )
        assert teacher_grad_free(teacher)
    report["probe"] = rows
    assert all(r["finite"] for r in rows) and all(r["ignored_gt_invariance"] for r in rows)
    ratios = [r["grad_nodata"] / r["grad_sup"] for r in rows if r["grad_nodata"] > 0]
    kd_rel = [r["grad_kd"] / r["grad_sup"] for r in rows]
    report["kd_grad_ratio_at_init"] = dict(max=max(kd_rel), median=float(np.median(kd_rel)))
    if len(ratios) >= 5:
        lam = NODATA_TARGET_GRAD_RATIO / float(np.median(ratios))
        lam = float(np.clip(float(f"{lam:.1g}"), *NODATA_LAMBDA_BOUNDS))
    else:
        lam = 1.0
    report["nodata_grad_ratios_at_unit_lambda"] = ratios
    report["nodata_lambda"] = lam
    report["nodata_lambda_rule"] = (
        f"lambda = {NODATA_TARGET_GRAD_RATIO} / median(||grad nodata|| / ||grad supervised||) "
        f"on probe crops with a nonzero cap violation, one significant figure, clipped to "
        f"{NODATA_LAMBDA_BOUNDS}; 1.0 if fewer than 5 such crops"
    )
    model.zero_grad(set_to_none=True)
    dump(out / "preflight.json", report)
    return lam


def teacher_grad_free(teacher):
    return all(p.grad is None for p in teacher.parameters())


@torch.no_grad()
def reload_identical(model, path, processor, crop):
    clean = load_model(path)
    pixels = load_crop(crop[0], crop[1], processor, "cuda")[0]
    a = model.eval()(pixel_values=pixels)
    b = clean(pixel_values=pixels)
    same = torch.equal(a.class_queries_logits, b.class_queries_logits) and torch.equal(
        a.masks_queries_logits, b.masks_queries_logits
    )
    ok = clean.config.id2label == {0: "longitudinal_solid", 1: "longitudinal_dash"}
    del clean
    torch.cuda.empty_cache()
    return bool(same and ok)


def main():
    torch.set_num_threads(4)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = BASE / "training" / stamp
    out.mkdir(parents=True)
    dump(BASE / "training" / "latest.json", {"run": str(out)})
    print(out, flush=True)
    checks, frozen, _ = frozen_paths()
    processor = load_processor()
    sup_dir = Path(read(BASE / "supervision/latest.json")["run"])
    syn = read(sup_dir / "synthetic_records.json")
    for r in syn:
        r["_root"] = Path(r["file"]).parent
    syn_train = [r for r in syn if r["role"] == "train"]
    syn_dev = [(r["_root"], r) for r in syn if r["role"] == "development"]
    real_stats = read(sup_dir / "real_image_statistics.json")
    triples = [
        (r["background_median"], r["background_std"], r["contrast"])
        for r in real_stats
        if r.get("contrast") is not None and r["contrast"] > 0
    ]
    rc = real_context()
    real_train = with_root(rc["full"], rc["dataset"])
    real_dev = development_crops(rc["ctx"])
    dev_ids = {c["crop_id"] for _, c in real_dev}
    assert not dev_ids & {c["crop_id"] for c in real_train}
    assert not {c["crop_id"] for _, c in syn_dev} & {c["crop_id"] for c in syn_train}
    plan = exposure_plan(real_train, syn_train, SEED, REAL_PER_CYCLE)
    if not plan["passes"]:
        plan = exposure_plan(real_train, syn_train, SEED, REAL_PER_CYCLE + 1)
        assert plan["passes"], "Class-exposure audit failed at 5:1"
    k = plan["real_per_cycle"]
    dump(out / "exposure_plan.json", plan)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        frozen, local_files_only=True
    ).cuda()
    teacher = load_model(frozen)
    teacher.requires_grad_(False)
    criterion = with_exact_bilinear(GatedNoObjectCriterion)(num_classes=2, bilinear=True).cuda()

    baseline = run(teacher, processor, real_dev, syn_dev)
    dump(out / "baseline.json", baseline)
    lam = preflight(model, teacher, criterion, processor, real_train, triples, out)
    protocol = dict(
        objective="Preserve frozen solid/dash recognition while adapting to LiDAR condition "
        "variation",
        frozen=checks,
        seed=SEED,
        max_steps=STEPS,
        checkpoints=CHECKPOINTS,
        trainable=TRAINABLE,
        frozen_components=["backbone", "pixel_decoder"],
        lrs=LOW,
        warmup_steps=50,
        grad_clip=1.0,
        optimizer="AdamW wd 0.01",
        precision="bf16 autocast forward, fp32 losses",
        deterministic=True,
        sampler=f"{k} Stage 9E real-cycle draws then 1 synthetic draw (alternating solid/dash "
        "positive pools) per cycle",
        exposure_plan=plan,
        supervised="Valid-only exact-bilinear Mask2Former loss with evidence-gated no-object "
        "(GatedNoObjectCriterion, min_valid_fraction 0.95) on real and synthetic",
        teacher="Detached frozen checkpoint. Guarded consistency on REAL crops only: mask BCE on "
        "verified-valid sampled points for unmatched queries, released where student is below "
        "teacher on verified background; class KL only for queries neither matched nor gated.",
        teacher_weights=TEACHER,
        nodata_cap=f"One-sided cap on NoData within {NEAR_PX} px (cap = teacher logit) and "
        f"{NEAR_PX + 1}-{FAR_PX} px (cap = min(teacher, 0)) of valid support, floor -2, "
        "real NoData = invalid & zero intensity; synthetic NoData = augmentation-removed "
        "pixels only",
        nodata_lambda=lam,
        augmentation=dict(AugmentConfig().__dict__, real_triples=len(triples)),
        supervision_run=str(sup_dir),
        gates=GATES,
        early_stop=dict(EARLY_STOP, consecutive_gate_failures=2),
        selection="Latest evaluated checkpoint passing every registered gate. If none passes, "
        "no candidate is accepted. Thresholds, metrics and clipping are fixed.",
        decision="GENERALIZATION_IMPROVED_PROVISIONAL if all gates pass and synthetic mean V1 F1 "
        "gain >= 0.02 with no real V1 F1 decline; INDEPENDENT_REAL_VALIDATION_REQUIRED if all "
        "gates pass without that gain; RECOGNITION_REGRESSION if any real recognition, "
        "false-positive or structure gate fails; NO_SAFE_IMPROVEMENT if only NoData or synthetic "
        "no-regression gates fail.",
        roles="Real dev = 14 reused development crops (not independent). Synthetic dev = 100 "
        "images of 5 held-out themes (previously inspected). 300-image set is qualitative only. "
        "T7/S8 protected, N9_01 development-only and unused.",
    )
    dump(out / "protocol.json", protocol)
    protocol_sha = hashlib.sha256((out / "protocol.json").read_bytes()).hexdigest()
    print("protocol sealed", protocol_sha, "lambda", lam, flush=True)

    optimizer, _ = make_optimizer(model, LOW)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    sampler = RealFirstSampler(real_train, syn_train, SEED, k)
    model_dir = MODELS / f"regsafe_{stamp}"
    initial = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    logs, evals, gates, grads = [], {}, {}, {}
    exposure = Counter()
    group_exposure = Counter()
    fails = 0
    stopped = None
    t_start = time.perf_counter()
    model.train()
    for step in range(1, STEPS + 1):
        for g, lr in zip(optimizer.param_groups, base_lrs, strict=True):
            g["lr"] = lr * min(1.0, step / 50)
        pool, crop = sampler.draw(step)
        real = not pool.startswith("synthetic")
        rng = np.random.default_rng(SEED * 1000 + step)
        pixels, masks, labels, valid, nodata, aug = load_step(crop, processor, rng, triples)
        exposure[f"draws_{'real' if real else 'synthetic'}"] += 1
        exposure[f"crop::{crop['crop_id']}"] += 1
        for lab in labels.tolist():
            exposure[f"{'real' if real else 'synthetic'}_instances_class{lab}"] += 1
        for gid in crop.get("global_instance_ids", []):
            group_exposure[gid] += 1
        optimizer.zero_grad(set_to_none=True)
        sup, kd, nd, losses, diag = forward_losses(
            model, teacher, criterion, pixels, masks, labels, valid, nodata, real, lam
        )
        total = sup + kd + lam * nd
        if not torch.isfinite(total):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        total.backward()
        if step in (1, 2, 3, 5) or step % 100 == 0:
            grads[step] = gradient_audit(model)
        norm = torch.nn.utils.clip_grad_norm_(
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
                **{k2: float(v.detach()) for k2, v in losses.items() if "aux" not in k2},
                grad_norm=float(norm),
                aug=aug,
                **diag,
            )
        )
        if step in CHECKPOINTS:
            res = run(model, processor, real_dev, syn_dev)
            evals[step] = res
            table, ok = gate_table(res, baseline)
            stop = early_stop(res, baseline)
            fails = 0 if ok else fails + 1
            gates[step] = dict(passed=ok, checks=table, early_stop_reasons=stop)
            ck = model_dir / f"step{step:04d}"
            model.save_pretrained(ck / "model")
            processor.save_pretrained(ck / "processor")
            gates[step]["reload_identical"] = reload_identical(
                model, ck / "model", processor, real_dev[0]
            )
            failed = [k2 for k2, v in table.items() if not v[0]]
            print(step, "PASS" if ok else f"FAIL {failed}", "stop" if stop else "", flush=True)
            dump(out / "checkpoint_gates.json", {str(s): v for s, v in gates.items()})
            dump(out / "checkpoint_evals.json", {str(s): v for s, v in evals.items()})
            model.train()
            if stop or fails >= 2:
                stopped = dict(step=step, reasons=stop or ["two consecutive gate failures"])
                break
    elapsed = time.perf_counter() - t_start
    drift = defaultdict(lambda: [0.0, 0.0])
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in initial:
                r = drift[group_of(n)]
                r[0] += float((p - initial[n]).float().norm() ** 2)
                r[1] += float(initial[n].float().norm() ** 2)
    passing = [s for s in sorted(gates) if gates[s]["passed"] and gates[s]["reload_identical"]]
    selected = passing[-1] if passing else None
    last = max(evals)
    status = "NO_SAFE_IMPROVEMENT"
    reference = evals[selected] if selected else evals[last]
    gain_syn = mean_v1_f1_gain(reference, baseline, "synthetic")
    gain_real = mean_v1_f1_gain(reference, baseline, "real")
    if selected:
        if sum(gain_syn.values()) / 2 >= 0.02 and min(gain_real.values()) >= 0:
            status = "GENERALIZATION_IMPROVED_PROVISIONAL"
        else:
            status = "INDEPENDENT_REAL_VALIDATION_REQUIRED"
    else:
        failed = [k2 for k2, v in gates[last]["checks"].items() if not v[0]]
        safety_only = all("nodata" in f or f.startswith("synthetic_") for f in failed)
        status = "NO_SAFE_IMPROVEMENT" if safety_only else "RECOGNITION_REGRESSION"
    verify_frozen(checks)
    info = dict(
        steps_completed=len(logs),
        stopped=stopped,
        elapsed_seconds=elapsed,
        seconds_per_step=elapsed / max(len(logs), 1),
        gpu_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        exposure={k2: v for k2, v in exposure.items() if not k2.startswith("crop::")},
        distinct_crops=sum(1 for k2 in exposure if k2.startswith("crop::")),
        max_crop_draws=max((v for k2, v in exposure.items() if k2.startswith("crop::")), default=0),
        physical_groups_exposed=len(group_exposure),
        physical_group_exposure_max=max(group_exposure.values(), default=0),
        relative_parameter_drift={g: (a**0.5) / max(b**0.5, 1e-12) for g, (a, b) in drift.items()},
        nodata_lambda=lam,
        candidate_dir=str(model_dir),
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
            last_step=last,
            synthetic_v1_f1_gain=gain_syn,
            real_v1_f1_gain=gain_real,
            candidate=str(model_dir / f"step{selected:04d}") if selected else None,
            promoted=False,
            frozen_default=True,
            protocol_sha256=protocol_sha,
            protected_tests_used=False,
        ),
    )
    print(out, status, selected, flush=True)


if __name__ == "__main__":
    sys.exit(main())
