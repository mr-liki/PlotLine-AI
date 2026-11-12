"""Stage 9D-A/B: diagnose zero text recall of the Stage 9C classifier-only model.

Raw query outputs and the fixed 9C postprocess are reported separately. N9_01 is read
for evaluation only; no threshold is tuned on it.
"""

import gc
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from roadmarking.training.pilot import load_crop
from roadmarking.training.road_text_head import CLASS_NAMES, NO_OBJECT_INDEX
from roadmarking.training.valid_criterion import ValidMask2FormerCriterion
from stage9c_attested_pilot import source_context, text_arrays
from stage9d_common import (
    ROOT,
    TEXT,
    dump,
    frozen_checks,
    new_run,
    postprocess,
    query_table,
    raw_outputs,
    shape_stats,
    synthetic,
    synthetic_stems,
    verify_frozen,
)

STAGE9C_MODEL = sorted((ROOT / "models/mask2former_swin_s").glob("stage9c_attested_*"))[-1]


def summarize_queries(table, target_pixels, post_text, target, scored):
    probs = table["probs"]
    text_p = probs[:, TEXT] if probs.shape[1] > 3 else np.zeros(len(probs))
    argmax = probs.argmax(-1)
    best = int(np.argmax(table["iou"]))
    covered = table["recall"][table["area"] > 0]
    res = dict(
        target_pixels=int(target_pixels),
        max_text_prob=float(text_p.max()),
        queries_text_logit_gt_noobject=int(
            (table["logits"][:, TEXT] > table["logits"][:, -1]).sum() if probs.shape[1] > 3 else 0
        ),
        queries_argmax_text=int((argmax == TEXT).sum()) if probs.shape[1] > 3 else 0,
        queries_positive_text_logit=(
            int((table["logits"][:, TEXT] > 0).sum()) if probs.shape[1] > 3 else 0
        ),
        queries_with_any_overlap=int((table["recall"] > 0).sum()),
        max_single_query_text_recall=float(covered.max()) if len(covered) else 0.0,
        best_iou_query=dict(
            index=best,
            iou=float(table["iou"][best]),
            recall=float(table["recall"][best]),
            precision=float(table["precision"][best]),
            area=int(table["area"][best]),
            quality=float(table["quality"][best]),
            argmax_class=int(argmax[best]),
            text_prob=float(text_p[best]),
            score_if_text=float(text_p[best] * table["quality"][best]),
            no_object_prob=float(probs[best, -1]),
        ),
        best_soft_dice=float(table["soft_dice"].max()),
        postprocessed_text_tp=int((post_text & target & scored).sum()),
        postprocessed_text_pixels=int((post_text & scored).sum()),
    )
    return res


def intensity_profile(gray, target, valid, nodata=None):
    gray = gray.astype(float)
    target = target & valid
    background = valid & ~target
    if nodata is not None:
        background &= ~nodata
    fg = gray[target]
    bg = gray[background]
    edt = distance_transform_edt(target)
    skeleton_width = 2 * np.median(edt[edt > 0]) if (edt > 0).any() else None
    return dict(
        foreground_mean=float(fg.mean()) if fg.size else None,
        foreground_p10_p90=(
            [float(np.percentile(fg, 10)), float(np.percentile(fg, 90))] if fg.size else None
        ),
        background_mean=float(bg.mean()) if bg.size else None,
        background_std=float(bg.std()) if bg.size else None,
        contrast_fg_minus_bg=float(fg.mean() - bg.mean()) if fg.size and bg.size else None,
        cnr=float((fg.mean() - bg.mean()) / max(bg.std(), 1e-6)) if fg.size and bg.size else None,
        zero_fraction=float((gray == 0).mean()),
        stroke_width_proxy_px=float(skeleton_width) if skeleton_width else None,
        foreground_pixels=int(target.sum()),
    )


def main():
    out = new_run("diagnosis")
    checks = frozen_checks()
    verify_frozen(checks)
    ctx = source_context()
    reports, dataset, val, fresh, tile, spec = ctx[0], ctx[4], ctx[7], ctx[8], ctx[9], ctx[10]
    state = ctx[3]
    frozen_path = Path(checks[0]["path"]).parent
    processor = Mask2FormerImageProcessor.from_pretrained(
        frozen_path.parent / "processor", local_files_only=True
    )
    rgb_n9, text_n9, non_text_n9, observed_n9 = text_arrays(reports, tile, spec)
    n9_scored = text_n9 | non_text_n9
    val_stems = synthetic_stems("val")
    train_stems = synthetic_stems("train")[:12]
    report = dict(
        stage9c_model=str(STAGE9C_MODEL / "step0040/model"),
        frozen_model=str(frozen_path),
        operating_point="argmax(class incl. no-object) == text and prob*quality>=0.25, area>=4",
        n9_01_role="evaluation only; thresholds unchanged",
    )
    models = {
        "frozen_two_class": frozen_path,
        "stage9c_step40": STAGE9C_MODEL / "step0040/model",
    }
    for name, path in models.items():
        model = Mask2FormerForUniversalSegmentation.from_pretrained(path, local_files_only=True)
        model.requires_grad_(False).cuda().eval()
        num = model.config.num_labels
        section = {"num_labels": num, "class_rows": int(model.class_predictor.out_features)}
        # N9_01
        pix = processor(images=rgb_n9, return_tensors="pt").pixel_values.cuda()
        native, probs, logits = raw_outputs(model, pix, observed_n9.shape)
        if num == 2:  # pad a text column of -inf so the table is shape compatible
            logits = torch.cat(
                [logits[:, :2], torch.full_like(logits[:, :1], -1e4), logits[:, 2:]], 1
            )
            probs = logits.softmax(-1)
        table = query_table(native, probs, logits, text_n9, observed_n9)
        post = postprocess(native, probs, observed_n9, 3)
        n9 = summarize_queries(table, text_n9.sum(), post[TEXT], text_n9, n9_scored)
        union_raw = ((native > 0).any(0).cpu().numpy()) & observed_n9
        n9["union_all_raw_queries_text_recall"] = float((union_raw & text_n9).sum() / text_n9.sum())
        best = int(n9["best_iou_query"]["index"])
        best_mask = (native[best] > 0).cpu().numpy() & observed_n9
        n9["best_query_shape"] = shape_stats(best_mask)
        n9["text_target_shape"] = shape_stats(text_n9)
        top_masks = np.argsort(-probs[:, :-1].max(-1).values.cpu().numpy())[:10]
        n9["top10_confident_queries"] = [
            dict(
                q=int(q),
                argmax=int(probs[q].argmax()),
                probs=[round(float(p), 4) for p in probs[q]],
                area=int(table["area"][q]),
                text_recall=float(table["recall"][q]),
                lane_overlap=int(((native[q] > 0).cpu().numpy() & non_text_n9).sum()),
                shape=shape_stats((native[q] > 0).cpu().numpy() & observed_n9),
            )
            for q in top_masks
        ]
        # Rejection analysis: which filter rejects the best-overlap query
        b = n9["best_iou_query"]
        n9["best_query_rejection"] = (
            "no_spatial_overlap"
            if b["iou"] == 0
            else (
                "class_argmax_not_text"
                if b["argmax_class"] != TEXT
                else "score_below_threshold" if b["score_if_text"] < 0.25 else "kept"
            )
        )
        section["n9_01"] = n9
        # Synthetic
        syn = []
        for split, stems in (("val", val_stems), ("train", train_stems)):
            for stem in stems:
                pixels, masks, labels, valid, _, _ = synthetic(split, stem, processor)
                target = masks[0].bool().cpu().numpy()
                v = valid.cpu().numpy().astype(bool)
                native, probs, logits = raw_outputs(model, pixels, v.shape)
                if num == 2:
                    logits = torch.cat(
                        [logits[:, :2], torch.full_like(logits[:, :1], -1e4), logits[:, 2:]], 1
                    )
                    probs = logits.softmax(-1)
                table = query_table(native, probs, logits, target, v)
                post = postprocess(native, probs, v, 3)
                row = summarize_queries(table, (target & v).sum(), post[TEXT], target, v)
                row.update(split=split, stem=stem)
                bq = row["best_iou_query"]["index"]
                row["best_query_shape"] = shape_stats((native[bq] > 0).cpu().numpy() & v)
                row["target_shape"] = shape_stats(target & v)
                syn.append(row)
        section["synthetic"] = syn

        def agg(rows, key, sub=None):
            values = [r[key][sub] if sub else r[key] for r in rows]
            return dict(mean=float(np.mean(values)), max=float(np.max(values)))

        vrows = [r for r in syn if r["split"] == "val"]
        section["synthetic_val_summary"] = dict(
            samples=len(vrows),
            best_query_iou=agg(vrows, "best_iou_query", "iou"),
            best_query_text_prob=agg(vrows, "best_iou_query", "text_prob"),
            max_text_prob=agg(vrows, "max_text_prob"),
            samples_with_any_argmax_text=int(sum(r["queries_argmax_text"] > 0 for r in vrows)),
            samples_with_postprocessed_text_tp=int(
                sum(r["postprocessed_text_tp"] > 0 for r in vrows)
            ),
            postprocessed_recall=float(
                sum(r["postprocessed_text_tp"] for r in vrows)
                / sum(r["target_pixels"] for r in vrows)
            ),
            best_query_rejections={
                k: int(
                    sum(
                        1
                        for r in vrows
                        if (
                            "no_spatial_overlap"
                            if r["best_iou_query"]["iou"] == 0
                            else (
                                "class_argmax_not_text"
                                if r["best_iou_query"]["argmax_class"] != TEXT
                                else (
                                    "score_below_threshold"
                                    if r["best_iou_query"]["score_if_text"] < 0.25
                                    else "kept"
                                )
                            )
                        )
                        == k
                    )
                )
                for k in (
                    "no_spatial_overlap",
                    "class_argmax_not_text",
                    "score_below_threshold",
                    "kept",
                )
            },
        )
        # Matching and gradient check for the 9C recipe (classifier-only)
        if num == 3:
            criterion = ValidMask2FormerCriterion(num_classes=3).cuda()
            grads = []
            model.class_predictor.requires_grad_(True)
            for stem in train_stems[:6]:
                pixels, masks, labels, valid, _, _ = synthetic("train", stem, processor)
                model.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    pred = model(pixel_values=pixels, output_auxiliary_logits=False)
                losses, (qi, _ti) = criterion.one_image(
                    pred.masks_queries_logits[0].float(),
                    pred.class_queries_logits[0].float(),
                    masks,
                    labels,
                    valid,
                    True,
                )
                losses["loss_class"].backward()
                g = model.class_predictor.weight.grad
                native = torch.nn.functional.interpolate(
                    pred.masks_queries_logits.float(), size=valid.shape, mode="bilinear"
                )[0].detach()
                t = masks[0].bool() & valid
                q = int(qi[0])
                m = (native[q] > 0) & valid
                grads.append(
                    dict(
                        stem=stem,
                        valid_all=bool(valid.all()),
                        matched_query=q,
                        matched_query_iou=float((m & t).sum() / (m | t).sum().clamp(min=1)),
                        matched_query_area=int(m.sum()),
                        target_pixels=int(t.sum()),
                        loss_class=float(losses["loss_class"]),
                        loss_bce=float(losses["loss_bce"]),
                        loss_dice=float(losses["loss_dice"]),
                        grad_norm_text_row=float(g[TEXT].norm()),
                        grad_norm_noobject_row=float(g[NO_OBJECT_INDEX].norm()),
                        grad_norm_solid_dash_rows=float(g[:2].norm()),
                        mask_branch_requires_grad=any(
                            p.requires_grad
                            for n, p in model.named_parameters()
                            if "class_predictor" not in n
                        ),
                        unmatched_queries_no_object_supervised=bool(valid.all()),
                    )
                )
            section["classifier_only_gradient_check"] = grads
        report[name] = section
        del model
        gc.collect()
        torch.cuda.empty_cache()
    # Intensity / normalization comparison (Stage A item 6, input to Stage G)
    profiles = {"n9_01": intensity_profile(rgb_n9[..., 0], text_n9, observed_n9, ~observed_n9)}
    syn_prof = []
    for stem in val_stems:
        _, masks, _, valid, rgb, _ = synthetic("val", stem, processor, device="cpu")
        syn_prof.append(
            intensity_profile(rgb[..., 0], masks[0].bool().numpy(), valid.numpy().astype(bool))
        )
    profiles["synthetic_val"] = {
        k: float(np.mean([p[k] for p in syn_prof if p[k] is not None]))
        for k in syn_prof[0]
        if not isinstance(syn_prof[0][k], list)
    }
    real = []
    development = [(dataset, c) for c in sorted(val, key=lambda c: c["crop_id"])[:12]]
    development.extend((Path(state["fresh_dev"]), c) for c in fresh)
    for root, crop in development:
        _, masks, labels, valid, rgb = load_crop(root, crop, processor, "cpu")
        t = masks[0].bool().any(0).numpy() if len(masks[0]) else np.zeros(rgb.shape[:2], bool)
        real.append(intensity_profile(rgb[..., 0], t, valid[0].numpy().astype(bool)))
    real = [p for p in real if p["foreground_mean"] is not None]
    profiles["real_line_dev_crops"] = {
        k: float(np.mean([p[k] for p in real if p[k] is not None]))
        for k in real[0]
        if not isinstance(real[0][k], list)
    }
    profiles["processor"] = (
        "identical Mask2FormerImageProcessor (rescale 1/255, ImageNet mean/std) for all"
    )
    profiles["rgb_construction"] = dict(
        synthetic="grayscale PNG replicated to 3 channels",
        n9_01="grayscale PNG converted to RGB",
        real_crops="stored rgb array from npz",
        channels_equal_real=bool(np.all(rgb_n9[..., 0] == rgb_n9[..., 1])),
    )
    report["intensity_profiles"] = profiles
    report["class_names"] = list(CLASS_NAMES)
    verify_frozen(checks)
    dump(out / "zero_recall_diagnosis.json", report)
    print(out)


if __name__ == "__main__":
    main()
