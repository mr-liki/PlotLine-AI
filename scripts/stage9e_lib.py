"""Stage 9E shared library: approved real pools, two-level evaluation, bounded training.

Evaluation reports two prediction levels with one code path for every model:
* raw: established proposal rule (argmax class, score = prob x mean valid sigmoid >= 0.25,
  area >= 4), union clipped to valid pixels (the Stage 9C/9D operating point);
* V1: the frozen candidate's established filter (per-class confidence thresholds from
  outputs/reports/final_candidate/frozen_candidate.json, mask quality 0, class-wise NMS 0.3)
  before cross-crop stitching. Accepted masks are clipped to valid support.
NoData is reported at three levels (raw mask in NoData, accepted mask outside footprint
before clipping, accepted prediction after clipping).
"""

import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import label as connected
from shapely.geometry import box
from transformers import Mask2FormerForUniversalSegmentation

from roadmarking.inference.calibration import filter_instances, iou_matrix
from roadmarking.training.deterministic_sampling import with_exact_bilinear
from roadmarking.training.gated_criterion import GatedNoObjectCriterion
from roadmarking.training.pilot import load_crop
from roadmarking.training.valid_criterion import ValidMask2FormerCriterion
from stage9c_attested_pilot import crop_box, source_context
from stage9d_common import ROOT, counts, dump, metrics, read
from stage9d_train_lib import gradient_audit, group_of, make_optimizer, set_trainable

BASE = ROOT / "outputs/reports/stage9e"
FROZEN_CONTRACT = read(ROOT / "outputs/reports/final_candidate/frozen_candidate.json")
V1 = dict(
    confidence={k: float(v) for k, v in FROZEN_CONTRACT["thresholds_V1"].items()},
    mask_quality=0.0,
    nms_iou=0.3,
)
RAW_SCORE, MIN_AREA = 0.25, 4
LONG_LINE_PX = 200


def new_run(name):
    from datetime import UTC, datetime

    run = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = BASE / name / run
    out.mkdir(parents=True)
    dump(BASE / name / "latest.json", {"run": str(out)})
    return out


def real_context():
    """Approved Stage 6 real pools with explicit protected-population assertions."""
    ctx = source_context()
    reports, state = ctx[0], ctx[3]
    dataset = Path(state["dataset"])
    manifest = read(dataset / "metadata/dataset_manifest.json")
    eligibility = read(reports / "stage8b/evaluation_eligibility.json")
    members = set()
    for pop in eligibility["populations"].values():
        members |= set(pop["member_ids"])
    protected_boxes = []
    for stage in ("stage7_new_test", "stage8_new_solid_review"):
        folder = reports / stage
        pointer = folder / "latest.json"
        directory = (
            Path(read(pointer)["report_dir"])
            if pointer.exists()
            else next(p for p in folder.iterdir() if (p / "new_region_candidates.json").is_file())
        )
        protected_boxes += [
            box(*t["bounds"]) for t in read(directory / "new_region_candidates.json")["tiles"]
        ]
    n9 = box(*ctx[9]["bounds"])
    legacy_ids = {
        c["crop_id"]
        for c in read(
            ROOT
            / "data/processed/phase5_training/phase5_12_20261106T165054344613Z/metadata/dataset_manifest.json"  # noqa: E501
        )["crops"]
        if c["split"] == "train"
    }
    full = [c for c in manifest["crops"] if c["split"] == "train" and c["size"] == 512]
    for c in full:
        if members & set(c["global_instance_ids"]):
            raise RuntimeError(f"T7/S8 population member in training crop {c['crop_id']}")
        if any(crop_box(c).intersects(b) for b in protected_boxes) or crop_box(c).intersects(n9):
            raise RuntimeError(f"Protected or N9_01 footprint overlap: {c['crop_id']}")
    stage9d_ids = {c["crop_id"] for c in ctx[6]}
    for c in full:
        c["geography"] = "legacy" if c["crop_id"] in legacy_ids else "expanded"
    return dict(
        ctx=ctx,
        dataset=dataset,
        full=full,
        stage9d=[c for c in full if c["crop_id"] in stage9d_ids],
        population_members=len(members),
        protected_boxes=len(protected_boxes),
    )


def development_crops(ctx):
    dataset, val, fresh, state = ctx[4], ctx[7], ctx[8], ctx[3]
    dev = [(dataset, c) for c in sorted(val, key=lambda c: c["crop_id"])[:12]]
    dev.extend((Path(state["fresh_dev"]), c) for c in fresh)
    return dev


@torch.no_grad()
def proposals(model, pixels, valid):
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(pixel_values=pixels, output_auxiliary_logits=False)
    native = torch.nn.functional.interpolate(
        out.masks_queries_logits.float(), size=valid.shape, mode="bilinear", align_corners=False
    )[0]
    probs = out.class_queries_logits[0].float().softmax(-1)
    top, labels = probs.max(-1)
    no_object = probs.shape[-1] - 1
    v = torch.as_tensor(valid, device=native.device)
    raw = native > 0
    sel = raw & v
    area = sel.flatten(1).sum(1)
    quality = (native.sigmoid() * sel).flatten(1).sum(1) / area.clamp(min=1)
    score = top * quality
    keep = (labels != no_object) & (area >= MIN_AREA) & (score >= RAW_SCORE)
    idx = torch.nonzero(keep).flatten().tolist()
    return dict(
        masks=raw[idx].cpu().numpy(),
        scores=score[idx].cpu().numpy(),
        labels=labels[idx].cpu().numpy(),
        query_indices=idx,
        qualities=quality[idx].cpu().numpy(),
        sigmoid_max=native[idx].sigmoid().amax(0).cpu().numpy() if idx else None,
    )


def instance_counts(pred_masks, pred_labels, truth, truth_labels, valid, cls):
    p = [m & valid for m, lab in zip(pred_masks, pred_labels, strict=True) if lab == cls]
    t = [
        m & valid
        for m, lab in zip(truth, truth_labels, strict=True)
        if lab == cls and (m & valid).any()
    ]
    if not p or not t:
        return np.array([0, len(p), len(t)])
    iou = iou_matrix(p, t, valid)
    pairs = sorted(
        ((iou[i, j], i, j) for i in range(len(p)) for j in range(len(t)) if iou[i, j] >= 0.5),
        reverse=True,
    )
    used_p, used_t = set(), set()
    for _, i, j in pairs:
        if i not in used_p and j not in used_t:
            used_p.add(i)
            used_t.add(j)
    tp = len(used_p)
    return np.array([tp, len(p) - tp, len(t) - tp])


def evaluate(model, processor, development, classes=(0, 1)):
    model.eval()
    acc = {
        lvl: dict(
            pix={c: np.zeros(3, np.int64) for c in classes},
            inst={c: np.zeros(3, np.int64) for c in classes},
        )
        for lvl in ("raw", "v1")
    }
    complete = {lvl: {c: [0, 0] for c in classes} for lvl in ("raw", "v1")}
    long_cov = {lvl: [] for lvl in ("raw", "v1")}
    merged = {lvl: 0 for lvl in ("raw", "v1")}
    dash_total = 0
    cross = {lvl: Counter() for lvl in ("raw", "v1")}
    nodata = Counter()
    n_props = Counter()
    for root, crop in development:
        pixel, masks, labels, validity, rgb = load_crop(root, crop, processor, "cuda")
        valid = validity[0].cpu().numpy().astype(bool)
        nod = ~valid & (rgb[..., 0] == 0)  # NoData proxy: unobserved zero intensity
        truth = masks[0].bool().cpu().numpy()
        tl = labels[0].cpu().numpy()
        pr = proposals(model, pixel, valid)
        n_props["raw"] += len(pr["labels"])
        if len(pr["labels"]):
            ov = iou_matrix(pr["masks"], pr["masks"], valid)
            sel = filter_instances(
                pr["scores"],
                pr["labels"],
                ov,
                pr["qualities"],
                dict(V1, confidence={**V1["confidence"], "2": 1.01}),
            )
        else:
            sel = []
        n_props["v1"] += len(sel)
        levels = {
            "raw": (pr["masks"], pr["labels"]),
            "v1": (
                pr["masks"][sel] if len(sel) else pr["masks"][:0],
                pr["labels"][sel] if len(sel) else pr["labels"][:0],
            ),
        }
        nodata["raw_mask_pixels_in_nodata"] += (
            int((pr["masks"].any(0) & nod).sum()) if len(pr["labels"]) else 0
        )
        nodata["raw_sigmoid_gt_0.5_in_nodata"] += (
            int(((pr["sigmoid_max"] > 0.5) & nod).sum()) if pr["sigmoid_max"] is not None else 0
        )
        acc_mask = levels["v1"][0]
        nodata["v1_unclipped_outside_footprint"] += (
            int((acc_mask.any(0) & nod).sum()) if len(acc_mask) else 0
        )
        nodata["v1_unclipped_on_any_invalid"] += (
            int((acc_mask.any(0) & ~valid).sum()) if len(acc_mask) else 0
        )
        truth_cls = {
            c: truth[tl == c].any(0) if (tl == c).any() else np.zeros_like(valid) for c in classes
        }
        for lvl, (pm, plab) in levels.items():
            union = {
                c: (pm[plab == c].any(0) if (plab == c).any() else np.zeros_like(valid)) & valid
                for c in (0, 1, 2)
            }
            nodata[f"{lvl}_accepted_after_clipping_in_nodata"] += int(
                sum((union[c] & nod).sum() for c in union)
            )
            for c in classes:
                acc[lvl]["pix"][c] += counts(union[c], truth_cls[c], valid)
                acc[lvl]["inst"][c] += instance_counts(pm, plab, truth, tl, valid, c)
            cross[lvl]["solid_as_dash"] += int((union[1] & truth_cls[0] & ~truth_cls[1]).sum())
            cross[lvl]["dash_as_solid"] += int((union[0] & truth_cls[1] & ~truth_cls[0]).sum())
            cross[lvl]["text_on_lines"] += int((union[2] & (truth_cls[0] | truth_cls[1])).sum())
            cross[lvl]["predicted_on_verified_background"] += int(
                ((union[0] | union[1] | union[2]) & valid & ~truth.any(0)).sum()
                if len(truth)
                else ((union[0] | union[1]) & valid).sum()
            )
            for k in range(len(tl)):
                im = truth[k] & valid
                if not im.any():
                    continue
                cov = (union[tl[k]] & im).sum() / im.sum()
                complete[lvl][tl[k]][1] += 1
                complete[lvl][tl[k]][0] += int(cov >= 0.5)
                ys, xs = np.nonzero(im)
                if tl[k] == 0 and max(np.ptp(ys), np.ptp(xs)) >= LONG_LINE_PX:
                    long_cov[lvl].append(float(cov))
            comps, n = connected(union[1])
            dash_ids = [k for k in range(len(tl)) if tl[k] == 1 and (truth[k] & valid).any()]
            if lvl == "raw":
                dash_total += len(dash_ids)
            hit = set()
            for ci in range(1, n + 1):
                comp = comps == ci
                h = [k for k in dash_ids if (comp & truth[k]).sum() > 10]
                if len(h) >= 2:
                    hit.update(h)
            merged[lvl] += len(hit)
    names = {0: "solid", 1: "dash"}
    result = {}
    for lvl in ("raw", "v1"):
        result[lvl] = dict(
            pixel={names[c]: metrics(acc[lvl]["pix"][c]) for c in classes},
            instance_iou50={names[c]: metrics(acc[lvl]["inst"][c]) for c in classes},
            complete_marking_recall={
                names[c]: complete[lvl][c][0] / max(complete[lvl][c][1], 1) for c in classes
            },
            long_line_coverage=float(np.mean(long_cov[lvl])) if long_cov[lvl] else None,
            dash_separation_rate=1 - merged[lvl] / max(dash_total, 1),
            cross_class=dict(cross[lvl]),
            proposals=n_props[lvl],
        )
    result["nodata"] = dict(nodata)
    result["crops"] = len(development)
    return result


def gate_check(res, base, tol):
    """Predeclared regression tolerances for Stage C."""
    checks = {}
    for lvl, ptol in (("v1", tol["v1_pixel_f1"]), ("raw", tol["raw_pixel_f1"])):
        for c in ("solid", "dash"):
            d = (base[lvl]["pixel"][c]["f1"] or 0) - (res[lvl]["pixel"][c]["f1"] or 0)
            checks[f"{lvl}_{c}_pixel_f1_drop<={ptol}"] = (d <= ptol, round(d, 4))
    for c in ("solid", "dash"):
        d = (base["v1"]["instance_iou50"][c]["f1"] or 0) - (
            res["v1"]["instance_iou50"][c]["f1"] or 0
        )
        checks[f"v1_{c}_instance_f1_drop<={tol['instance_f1']}"] = (
            d <= tol["instance_f1"],
            round(d, 4),
        )
        d = base["v1"]["complete_marking_recall"][c] - res["v1"]["complete_marking_recall"][c]
        checks[f"v1_{c}_complete_recall_drop<={tol['complete_recall']}"] = (
            d <= tol["complete_recall"],
            round(d, 4),
        )
    for key in ("raw_mask_pixels_in_nodata", "v1_unclipped_outside_footprint"):
        b, r = base["nodata"].get(key, 0), res["nodata"].get(key, 0)
        checks[f"{key}<={tol['nodata_ratio']}x"] = (r <= tol["nodata_ratio"] * max(b, 1), [b, r])
    checks["accepted_after_clipping_in_nodata==0"] = (
        res["nodata"].get("v1_accepted_after_clipping_in_nodata", 0) == 0,
        res["nodata"].get("v1_accepted_after_clipping_in_nodata", 0),
    )
    return checks, all(v[0] for v in checks.values())


class Sampler:
    """Deterministic schedules: uniform permutation over a pool, or a class/geography cycle."""

    def __init__(self, crops, mode, seed):
        self.rng = np.random.default_rng(seed)
        self.mode = mode
        if mode == "uniform":
            self.pools = {"all": list(crops)}
            self.cycle = ["all"]
        else:

            def pool(geo, cls):
                return [
                    c
                    for c in crops
                    if c["geography"] == geo
                    and (cls in c["labels"] if cls is not None else not c["labels"])
                ]

            self.pools = {
                "legacy_solid": [
                    c
                    for c in crops
                    if c["geography"] == "legacy" and 0 in c["labels"] and 1 not in c["labels"]
                ],
                "legacy_dash": pool("legacy", 1),
                "expanded_solid": [
                    c for c in crops if c["geography"] == "expanded" and 0 in c["labels"]
                ],
                "expanded_dash": pool("expanded", 1),
                "negative": [c for c in crops if not c["labels"]],
            }
            self.cycle = [
                "legacy_dash",
                "legacy_solid",
                "expanded_dash",
                "legacy_dash",
                "expanded_solid",
                "negative",
            ]
        self.orders = {k: [] for k in self.pools}

    def draw(self, step):
        name = self.cycle[(step - 1) % len(self.cycle)]
        if not self.orders[name]:
            self.orders[name] = list(self.rng.permutation(len(self.pools[name])))
        return name, self.pools[name][self.orders[name].pop()]

    def sizes(self):
        return {k: len(v) for k, v in self.pools.items()}


def teacher_loss(student, teacher, criterion, targets, valid, weights, *, include_invalid=False):
    """Match frozen teacher probabilities on valid and optionally ignored points."""
    idx, w = criterion.sample(targets, valid)
    h, wd = valid.shape
    rows, cols = idx // wd, idx % wd
    s = criterion.logits_at(student.masks_queries_logits[0].float(), rows, cols, h, wd)
    t = criterion.logits_at(teacher.masks_queries_logits[0].float(), rows, cols, h, wd).sigmoid()
    mask_term = (
        (torch.nn.functional.binary_cross_entropy_with_logits(s, t, reduction="none") * w)
        .sum(-1)
        .mean()
    )
    if include_invalid:
        pool = torch.where(~valid.flatten())[0]
        if len(pool):
            count = min(len(pool), criterion.max_points)
            positions = torch.linspace(0, len(pool) - 1, count, device=pool.device).long()
            rows, cols = pool[positions] // wd, pool[positions] % wd
            s = criterion.logits_at(student.masks_queries_logits[0].float(), rows, cols, h, wd)
            t = criterion.logits_at(teacher.masks_queries_logits[0].float(), rows, cols, h, wd)
            mask_term = mask_term + torch.nn.functional.binary_cross_entropy_with_logits(
                s, t.detach().sigmoid()
            )
    sc = student.class_queries_logits[0].float().log_softmax(-1)
    tc = teacher.class_queries_logits[0].float().softmax(-1)
    class_term = torch.nn.functional.kl_div(sc, tc, reduction="batchmean")
    return (
        weights["mask"] * mask_term + weights["class"] * class_term,
        float(mask_term),
        float(class_term),
    )


def train_arm(
    name,
    out,
    frozen_path,
    processor,
    development,
    crops,
    *,
    sampler_mode,
    lrs,
    steps,
    eval_every,
    seed,
    gated=False,
    teacher_weights=None,
    teacher_include_invalid=False,
    before_save=None,
    strict_determinism=False,
    bilinear=True,
    stop_rule=None,
    baseline=None,
    save_dir=None,
    deterministic=False,
    exact_bilinear=True,
    sampler_factory=Sampler,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=not strict_determinism)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        frozen_path, local_files_only=True
    ).cuda()
    trainable = set_trainable(model, ["query_decoder", "mask_embedder", "class_predictor"])
    optimizer, membership = make_optimizer(model, lrs)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    crit_cls = GatedNoObjectCriterion if gated else ValidMask2FormerCriterion
    if exact_bilinear:
        crit_cls = with_exact_bilinear(crit_cls)
    criterion = crit_cls(num_classes=model.config.num_labels, bilinear=bilinear).cuda()
    teacher = None
    if teacher_weights:
        teacher = (
            Mask2FormerForUniversalSegmentation.from_pretrained(frozen_path, local_files_only=True)
            .cuda()
            .eval()
        )
        teacher.requires_grad_(False)
    sampler = sampler_factory(crops, sampler_mode, seed)
    logs, evals, grads = [], {}, {}
    exposure = Counter()
    initial = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    stopped = None
    start = time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        for g, lr in zip(optimizer.param_groups, base_lrs, strict=True):
            g["lr"] = lr * min(1.0, step / 50)
        pool, crop = sampler.draw(step)
        pixels, masks, labels, valid, _ = load_crop(crop["_root"], crop, processor, "cuda")
        masks, labels, valid = masks[0], labels[0], valid[0]
        exposure[f"pool_{pool}"] += 1
        exposure[f"crop_{crop['crop_id']}"] += 1
        for lab in labels.tolist():
            exposure[f"instances_class{lab}"] += 1
        for c in (0, 1):
            sel = labels == c
            exposure[f"fg_pixels_class{c}"] += (
                int((masks[sel].bool().any(0) & valid).sum()) if sel.any() else 0
            )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(pixel_values=pixels, output_auxiliary_logits=True)
        with torch.autocast("cuda", enabled=False):
            aux = [{k: v.float() for k, v in a.items()} for a in pred.auxiliary_logits]
            losses, _matches = criterion(
                pred.masks_queries_logits.float(),
                pred.class_queries_logits.float(),
                [masks],
                [labels],
                [valid],
                [True],
                aux,
            )
            total = criterion.total(losses)
            extra = {}
            if teacher is not None:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    tpred = teacher(pixel_values=pixels, output_auxiliary_logits=False)
                tl, tm, tc = teacher_loss(
                    pred,
                    tpred,
                    criterion,
                    masks,
                    valid,
                    teacher_weights,
                    include_invalid=teacher_include_invalid,
                )
                total = total + tl
                extra = dict(teacher_mask=tm, teacher_class=tc)
        if not torch.isfinite(total):
            raise RuntimeError(f"{name}: nonfinite loss at step {step}")
        total.backward()
        if step in (1, 2, 3) or step % 100 == 0:
            grads[step] = gradient_audit(model)
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        gate = criterion.last_gate.pop() if gated and criterion.last_gate else None
        if gated:
            criterion.last_gate.clear()
        logs.append(
            dict(
                step=step,
                pool=pool,
                crop=crop["crop_id"],
                total=float(total.detach()),
                **{k: float(v.detach()) for k, v in losses.items() if "aux" not in k},
                aux_total={
                    str(i): float(
                        sum(v.detach() for k, v in losses.items() if k.endswith(f"_aux{i}"))
                    )
                    for i in range(len(aux))
                },
                grad_norm=float(norm),
                gate=gate,
                **extra,
            )
        )
        if step % eval_every == 0 or step == steps:
            evals[step] = evaluate(model, processor, development)
            if baseline is not None:
                d = {
                    lvl: {
                        c: round(
                            (baseline[lvl]["pixel"][c]["f1"] or 0)
                            - (evals[step][lvl]["pixel"][c]["f1"] or 0),
                            4,
                        )
                        for c in ("solid", "dash")
                    }
                    for lvl in ("raw", "v1")
                }
                print(name, step, "F1 drops", d, flush=True)
                if stop_rule and stop_rule(evals[step], baseline):
                    stopped = step
                    model.train()
                    break
            model.train()
    drift = {}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in initial:
                g = group_of(n)
                layer = n.split("layers.")[1].split(".")[0] if "decoder.layers." in n else "-"
                key = f"{g}|layer{layer}" if g == "query_decoder" else g
                r = drift.setdefault(key, [0.0, 0.0])
                r[0] += float((p - initial[n]).float().norm() ** 2)
                r[1] += float(initial[n].float().norm() ** 2)
    drift = {k: (a**0.5) / max(b**0.5, 1e-12) for k, (a, b) in drift.items()}
    if save_dir is not None:
        if before_save is not None:
            before_save(model, evals, stopped)
        model.save_pretrained(save_dir / "model")
        processor.save_pretrained(save_dir / "processor")
    info = dict(
        name=name,
        seed=seed,
        sampler=sampler_mode,
        pool_sizes=sampler.sizes(),
        lrs=lrs,
        steps_run=len(logs),
        stopped_early_at=stopped,
        gated=gated,
        teacher=teacher_weights,
        teacher_include_invalid=teacher_include_invalid,
        bilinear=bilinear,
        trainable=trainable,
        optimizer=membership,
        elapsed_seconds=time.perf_counter() - start,
        exposure_summary={k: v for k, v in exposure.items() if not k.startswith("crop_")},
        distinct_crops=sum(1 for k in exposure if k.startswith("crop_")),
        crop_exposure_max=max((v for k, v in exposure.items() if k.startswith("crop_")), default=0),
        relative_parameter_drift=drift,
        deterministic=deterministic,
        exact_bilinear=exact_bilinear,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
    )
    dump(out / f"{name}_info.json", info)
    dump(out / f"{name}_log.json", logs)
    dump(out / f"{name}_evals.json", {str(k): v for k, v in evals.items()})
    dump(out / f"{name}_grads.json", {str(k): v for k, v in grads.items()})
    del model, optimizer, teacher
    torch.cuda.empty_cache()
    return info, evals, logs


def with_root(crops, dataset):
    return [{**c, "_root": dataset} for c in crops]


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
