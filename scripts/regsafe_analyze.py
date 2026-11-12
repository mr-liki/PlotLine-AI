"""Step 1: explain the geometry-sprint regression from archived records plus an inference probe.

Archived: loss/gradient/sampling logs of the geometry sprint (G), Stage 9E C4 and Stage 9F C5.
Probe (inference only): frozen, C5 and G on the identical 14 reused real development crops.
Each raw proposal's NoData pixels are attributed to queries that overlap a same-class
ground-truth instance (line continuing past the observation edge) or to queries that touch
no target (false-positive queries), and to the 8 px NoData boundary band versus deeper NoData.
"""

import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime

import numpy as np

from regsafe_common import (
    BASE,
    C4_RUN,
    C5_MODEL,
    C5_RUN,
    GEOMETRY_MODEL,
    GEOMETRY_RUN,
    boundary_band,
    dump,
    frozen_paths,
    load_model,
    load_processor,
    nodata_mask,
    read,
    torch,
    verify_frozen,
)
from roadmarking.training.pilot import load_crop
from stage9e_lib import development_crops, proposals, real_context


def log_summary(path, info_path):
    log = read(path)
    info = read(info_path)
    by = defaultdict(list)
    for r in log:
        dom = "synthetic" if r["pool"].startswith("synthetic") else "real"
        by[dom].append(r)
    out = dict(
        gated_no_object=info.get("gated"),
        teacher=info.get("teacher"),
        teacher_include_invalid=info.get("teacher_include_invalid"),
        steps=len(log),
        class_predictor_drift=info["relative_parameter_drift"].get("class_predictor"),
        mask_embedder_drift=info["relative_parameter_drift"].get("mask_embedder"),
        exposure=info["exposure_summary"],
        domains={},
    )
    for dom, rows in by.items():
        g = np.array([r["grad_norm"] for r in rows])
        tot = np.array([r["total"] for r in rows])
        teach = np.array(
            [
                (
                    5 * r.get("teacher_mask", 0) + 2 * r.get("teacher_class", 0)
                    if info.get("teacher")
                    else 0.0
                )
                for r in rows
            ]
        )
        out["domains"][dom] = dict(
            steps=len(rows),
            step_share=len(rows) / len(log),
            pre_clip_grad_norm_median=float(np.median(g)),
            clipped_fraction=float((g > 1.0).mean()),
            loss_class_mean=float(np.mean([r["loss_class"] for r in rows])),
            loss_dice_mean=float(np.mean([r["loss_dice"] for r in rows])),
            total_mean=float(tot.mean()),
            teacher_share_of_total_mean=float((teach / np.maximum(tot, 1e-9)).mean()),
        )
    return out


def eval_summary(path, baseline):
    ev = read(path)
    rows = {"frozen": baseline}
    rows.update({f"step{k}": v for k, v in ev.items()})
    return {
        k: dict(
            raw_proposals=v["raw"]["proposals"],
            v1_proposals=v["v1"]["proposals"],
            raw_verified_background_fp=v["raw"]["cross_class"]["predicted_on_verified_background"],
            v1_verified_background_fp=v["v1"]["cross_class"]["predicted_on_verified_background"],
            raw_dash_precision=v["raw"]["pixel"]["dash"]["precision"],
            v1_dash_precision=v["v1"]["pixel"]["dash"]["precision"],
            nodata=v["nodata"],
        )
        for k, v in rows.items()
    }


@torch.no_grad()
def probe(model, teacher, processor, dev):
    acc = Counter()
    per_query = []
    agreement = []
    for root, crop in dev:
        pixel, masks, labels, validity, rgb = load_crop(root, crop, processor, "cuda")
        valid = validity[0].cpu().numpy().astype(bool)
        nod = nodata_mask(valid, rgb)
        band = boundary_band(valid, nod)
        truth = masks[0].bool().cpu().numpy()
        tl = labels[0].cpu().numpy()
        pr = proposals(model, pixel, valid)
        for m, lab, sc in zip(pr["masks"], pr["labels"], pr["scores"], strict=True):
            same = [truth[k] & valid for k in range(len(tl)) if tl[k] == lab]
            hit = any((m & t).sum() >= 10 for t in same)
            fg = truth.any(0) if len(tl) else np.zeros_like(valid)
            kind = "target_query" if hit else "false_positive_query"
            nd = int((m & nod).sum())
            acc[f"{kind}_count"] += 1
            acc[f"{kind}_nodata_px"] += nd
            acc[f"{kind}_nodata_band_px"] += int((m & band).sum())
            acc[f"{kind}_valid_bg_px"] += int((m & valid & ~fg).sum())
            per_query.append(dict(kind=kind, label=int(lab), score=float(sc), nodata=nd))
        acc["union_nodata_px"] += int((pr["masks"].any(0) & nod).sum()) if len(pr["masks"]) else 0
        acc["union_band_px"] += int((pr["masks"].any(0) & band).sum()) if len(pr["masks"]) else 0
        if teacher is not None:
            out_s = model(pixel_values=pixel)
            out_t = teacher(pixel_values=pixel)
            ps = out_s.class_queries_logits[0].float().softmax(-1)
            pt = out_t.class_queries_logits[0].float().softmax(-1)
            ms = out_s.masks_queries_logits[0] > 0
            mt = out_t.masks_queries_logits[0] > 0
            conf = pt[:, :-1].max(-1).values > 0.5
            inter = (ms & mt).flatten(1).sum(1).float()
            union = (ms | mt).flatten(1).sum(1).float().clamp(min=1)
            agreement.append(
                dict(
                    teacher_confident_queries=int(conf.sum()),
                    class_argmax_agree=(
                        float((ps.argmax(-1)[conf] == pt.argmax(-1)[conf]).float().mean())
                        if conf.any()
                        else None
                    ),
                    mask_iou_confident=float((inter / union)[conf].mean()) if conf.any() else None,
                    student_confident_queries=int((ps[:, :-1].max(-1).values > 0.5).sum()),
                )
            )
    out = dict(acc)
    for kind in ("target_query", "false_positive_query"):
        out[f"{kind}_nodata_share"] = acc[f"{kind}_nodata_px"] / max(
            acc["target_query_nodata_px"] + acc["false_positive_query_nodata_px"], 1
        )
    out["band_share_of_union_nodata"] = acc["union_band_px"] / max(acc["union_nodata_px"], 1)
    if agreement:
        out["teacher_agreement"] = dict(
            teacher_confident_queries=sum(a["teacher_confident_queries"] for a in agreement),
            student_confident_queries=sum(a["student_confident_queries"] for a in agreement),
            class_argmax_agree_mean=float(
                np.nanmean([a["class_argmax_agree"] or np.nan for a in agreement])
            ),
            mask_iou_confident_mean=float(
                np.nanmean([a["mask_iou_confident"] or np.nan for a in agreement])
            ),
        )
    return out


def main():
    out = BASE / "analysis" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True)
    checks, frozen, _ = frozen_paths()
    archived = dict(
        geometry_sprint=log_summary(
            GEOMETRY_RUN / "geometry_sprint_log.json", GEOMETRY_RUN / "geometry_sprint_info.json"
        ),
        stage9e_C4=log_summary(
            C4_RUN / "C4_plus_teacher_s9061_log.json", C4_RUN / "C4_plus_teacher_s9061_info.json"
        ),
        stage9f_C5=log_summary(C5_RUN / "C5_s9061_log.json", C5_RUN / "C5_s9061_info.json"),
    )
    base = read(GEOMETRY_RUN / "baseline_real.json")
    evals = dict(
        geometry_sprint=eval_summary(GEOMETRY_RUN / "geometry_sprint_evals.json", base),
        stage9e_C4=eval_summary(C4_RUN / "C4_plus_teacher_s9061_evals.json", base),
        stage9f_C5=eval_summary(C5_RUN / "C5_s9061_evals.json", base),
    )
    processor = load_processor()
    dev = development_crops(real_context()["ctx"])
    teacher = load_model(frozen)
    probes = {"frozen": probe(teacher, None, processor, dev)}
    for name, path in (
        ("stage9f_C5", C5_MODEL / "model"),
        ("geometry_sprint", GEOMETRY_MODEL / "model"),
    ):
        model = load_model(path)
        probes[name] = probe(model, teacher, processor, dev)
        del model
        torch.cuda.empty_cache()
    verify_frozen(checks)
    dump(out / "regression_analysis.json", dict(archived=archived, evals=evals, probe=probes))
    print(out)


if __name__ == "__main__":
    sys.exit(main())
