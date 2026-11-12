"""Stage 9D evaluation: unchanged fixed operating point; solid/dash, synthetic, N9_01 text.

Every candidate and the frozen baseline go through the same code path. N9_01 is
development-only and never used to tune thresholds.
"""

from pathlib import Path

import numpy as np
import pyogrio
import rasterio
import torch
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation
from scipy.ndimage import label as connected

from roadmarking.training.pilot import load_crop
from stage9d_common import (
    MIN_PIXELS,
    SCORE_THRESHOLD,
    TEXT,
    counts,
    metrics,
    raw_outputs,
    synthetic,
)

LONG_LINE_MIN_EXTENT_PX = 200


@torch.no_grad()
def predict_full(model, pixels, valid):
    """Fixed 9C operating point plus raw diagnostics (NoData violations, confidences)."""
    native, probs, _ = raw_outputs(model, pixels, valid.shape)
    num = model.config.num_labels
    if num == 2:  # frozen baseline: insert an impossible text column
        probs = torch.cat([probs[:, :2], torch.zeros_like(probs[:, :1]), probs[:, 2:]], 1)
    valid_t = torch.as_tensor(valid, device=native.device)
    raw = native > 0
    masks = raw & valid_t
    area = masks.flatten(1).sum(1)
    quality = (native.sigmoid() * masks).flatten(1).sum(1) / area.clamp(min=1)
    top, labels = probs.max(-1)
    scores = top * quality
    keep = (scores >= SCORE_THRESHOLD) & (area >= MIN_PIXELS)
    keep_np, labels_np = keep.cpu().numpy(), labels.cpu().numpy()
    masks_np = masks.cpu().numpy()
    raw_np = raw.cpu().numpy()
    post = {i: masks_np[keep_np & (labels_np == i)].any(axis=0) for i in range(3)}
    text_queries = keep_np & (labels_np == TEXT)
    return dict(
        post=post,
        nodata_raw_pixels=int((raw_np[keep_np].any(axis=0) & ~valid).sum()),
        kept_queries=int(keep_np.sum()),
        text_queries=int(text_queries.sum()),
        text_query_class_conf=[float(probs[q, TEXT]) for q in np.nonzero(text_queries)[0]],
        text_query_mask_quality=[float(quality[q]) for q in np.nonzero(text_queries)[0]],
        max_text_prob=float(probs[:, TEXT].max()),
    )


def development_crops(ctx):
    dataset, val, fresh, state = ctx[4], ctx[7], ctx[8], ctx[3]
    dev = [(dataset, c) for c in sorted(val, key=lambda c: c["crop_id"])[:12]]
    dev.extend((Path(state["fresh_dev"]), c) for c in fresh)
    assert dev and all(c["split"] == "val" for _, c in dev)
    return dev


def solid_dash(model, processor, development):
    pix = {i: np.zeros(3, np.int64) for i in (0, 1)}
    inst = {i: [0, 0] for i in (0, 1)}
    long_cov, dash_merged, dash_total = [], 0, 0
    cross = dict(solid_as_dash=0, dash_as_solid=0, solid_as_text=0, dash_as_text=0)
    text_fp_on_lines_crops = 0
    nodata = 0
    for root, crop in development:
        pixel, masks, labels, valid, _ = load_crop(root, crop, processor, "cuda")
        keep = valid[0].cpu().numpy().astype(bool)
        res = predict_full(model, pixel, keep)
        post = res["post"]
        nodata += res["nodata_raw_pixels"]
        text_fp_on_lines_crops += int((post[TEXT] & keep).sum())
        m = masks[0].bool().cpu().numpy()
        lab = labels[0].cpu().numpy()
        truth = {i: m[lab == i].any(0) if (lab == i).any() else np.zeros_like(keep) for i in (0, 1)}
        for i in (0, 1):
            pix[i] += counts(post[i], truth[i], keep)
        cross["solid_as_dash"] += int((post[1] & truth[0] & ~truth[1] & keep).sum())
        cross["dash_as_solid"] += int((post[0] & truth[1] & ~truth[0] & keep).sum())
        cross["solid_as_text"] += int((post[TEXT] & truth[0] & keep).sum())
        cross["dash_as_text"] += int((post[TEXT] & truth[1] & keep).sum())
        for k in range(len(lab)):
            inst_mask = m[k] & keep
            if inst_mask.sum() == 0:
                continue
            cov = (post[lab[k]] & inst_mask).sum() / inst_mask.sum()
            inst[lab[k]][1] += 1
            inst[lab[k]][0] += int(cov >= 0.5)
            if lab[k] == 0:
                ys, xs = np.nonzero(inst_mask)
                if max(np.ptp(ys), np.ptp(xs)) >= LONG_LINE_MIN_EXTENT_PX:
                    long_cov.append(float(cov))
        comps, n = connected(post[1])
        dash_ids = [k for k in range(len(lab)) if lab[k] == 1 and (m[k] & keep).any()]
        dash_total += len(dash_ids)
        merged = set()
        for c in range(1, n + 1):
            comp = comps == c
            hits = [k for k in dash_ids if (comp & m[k]).sum() > 10]
            if len(hits) >= 2:
                merged.update(hits)
        dash_merged += len(merged)
    return dict(
        longitudinal_solid=metrics(pix[0]),
        longitudinal_dash=metrics(pix[1]),
        instance_recall=dict(
            solid=inst[0][0] / max(inst[0][1], 1),
            dash=inst[1][0] / max(inst[1][1], 1),
            solid_instances=inst[0][1],
            dash_instances=inst[1][1],
        ),
        long_line_coverage_mean=float(np.mean(long_cov)) if long_cov else None,
        long_lines=len(long_cov),
        dash_instances_merged_with_neighbour=dash_merged,
        dash_separation_rate=1 - dash_merged / max(dash_total, 1),
        cross_class_pixels=cross,
        text_predicted_pixels_on_line_dev_crops=text_fp_on_lines_crops,
        nodata_raw_pixels_of_kept_queries=nodata,
        crops=len(development),
    )


def synthetic_eval(model, processor, split, stems):
    total = np.zeros(3, np.int64)
    neg_fp, per = 0, []
    nodata = 0
    for stem in stems:
        pixels, masks, labels, valid, _, _ = synthetic(split, stem, processor)
        v = valid.cpu().numpy().astype(bool)
        target = masks[0].bool().cpu().numpy() if len(labels) else np.zeros_like(v)
        res = predict_full(model, pixels, v)
        nodata += res["nodata_raw_pixels"]
        c = counts(res["post"][TEXT], target, v)
        if len(labels):
            total += c
            per.append(metrics(c)["iou"] or 0.0)
        else:
            neg_fp += int(c[1])
    return dict(
        pixel=metrics(total),
        per_sample_iou_mean=float(np.mean(per)) if per else None,
        positives=len(per),
        negatives=len(stems) - len(per),
        negative_image_text_fp_pixels=neg_fp,
        ignore_raw_pixels_of_kept_queries=nodata,
    )


def n9_groups(spec, tile, shape):
    """Approved N9_01 text group footprints (group bbox) for phrase-level reporting."""
    frames = [
        pyogrio.read_dataframe(spec["text_candidates"], layer=layer, where="candidate_id='N9_01'")
        for layer in ("character_candidates", "group_and_unknown_fragments")
    ]
    with rasterio.open(tile["layers"]["observed"]) as ds:
        transform = ds.transform
    groups = {}
    for frame in frames:
        status_col = "review_status"
        for _, row in frame.iterrows():
            g = row["text_group_id"]
            groups.setdefault(g, dict(geoms=[], statuses=set()))
            groups[g]["geoms"].append(row.geometry)
            groups[g]["statuses"].add(row[status_col])
    out = {}
    for g, rec in groups.items():
        r = rasterize([(x, 1) for x in rec["geoms"]], out_shape=(512, 512), transform=transform)
        r = r.astype(bool)[: shape[0], : shape[1]]
        ys, xs = np.nonzero(r)
        if not len(ys):
            continue
        box = np.zeros(shape, bool)
        box[max(ys.min() - 3, 0) : ys.max() + 4, max(xs.min() - 3, 0) : xs.max() + 4] = True
        out[g] = dict(box=box, statuses=sorted(rec["statuses"]))
    return out


def n9_eval(model, processor, text, groups):
    image, positive, non_text, observed = text
    pixel = processor(images=image, return_tensors="pt").pixel_values.cuda()
    res = predict_full(model, pixel, observed)
    pred = res["post"][TEXT]
    scored = positive | non_text
    unscored_pred = pred & observed & ~scored
    near_text = binary_dilation(positive, iterations=6)
    fp_regions, n = connected(unscored_pred & ~near_text)
    fp_sizes = [int((fp_regions == i).sum()) for i in range(1, n + 1)]
    strokes, ns = connected(positive)
    missed = sum(1 for i in range(1, ns + 1) if not (pred & (strokes == i)).any())
    group_rows = {}
    for g, rec in groups.items():
        t = positive & rec["box"]
        tp = int((pred & t).sum())
        group_rows[g] = dict(
            statuses=rec["statuses"],
            text_pixels=int(t.sum()),
            tp=tp,
            recall=tp / max(int(t.sum()), 1),
            detected_recall_ge_0_3=tp / max(int(t.sum()), 1) >= 0.3,
        )
    return dict(
        text=metrics(counts(pred, positive, scored)),
        scored_positive_pixels=int(positive.sum()),
        scored_lane_negative_pixels=int(non_text.sum()),
        text_on_lane_paint_pixels=int((pred & non_text).sum()),
        predicted_text_unscored_observed_pixels=int(unscored_pred.sum()),
        false_positive_text_regions_away_from_text=dict(count=len(fp_sizes), pixels=sum(fp_sizes)),
        stroke_components=int(ns),
        missed_stroke_components=int(missed),
        stroke_component_recall=1 - missed / max(ns, 1),
        text_groups=group_rows,
        text_queries_kept=res["text_queries"],
        text_query_class_confidence=res["text_query_class_conf"],
        text_query_mask_quality=res["text_query_mask_quality"],
        max_text_class_prob=res["max_text_prob"],
        nodata_raw_pixels_of_kept_queries=res["nodata_raw_pixels"],
        predicted_text_on_nodata=int((pred & ~observed).sum()),
        recognized_paint_within_valid_observations=int((pred & observed & positive).sum()),
        _pred=pred,
    )


def full_eval(model, processor, development, text, groups, train_fit_stems, val_stems):
    model.eval()
    n9 = n9_eval(model, processor, text, groups)
    pred = n9.pop("_pred")
    result = dict(
        solid_dash=solid_dash(model, processor, development),
        synthetic_train_fit=synthetic_eval(model, processor, "train", train_fit_stems),
        synthetic_val=synthetic_eval(model, processor, "val", val_stems),
        n9_01=n9,
    )
    return result, pred
