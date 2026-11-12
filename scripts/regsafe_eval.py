"""Shared evaluation and registered no-regression gates for the regression-safe sprint.

Recognition metrics reuse the established ``stage9e_lib.evaluate`` code path (raw and V1
operating points, pre-stitching, valid-support clipping, three NoData levels). This module
adds V1 structural diagnostics (disconnected accepted masks, accepted masks spanning several
ground-truth instances) and the gate table registered in the sealed protocol.
"""

import time
from collections import Counter

import numpy as np
import torch
from scipy.ndimage import label as connected

from roadmarking.inference.calibration import filter_instances, iou_matrix
from roadmarking.training.pilot import load_crop
from stage9e_lib import V1, evaluate, proposals

CLASSES = ("solid", "dash")
LEVELS = ("raw", "v1")

# Registered before training. "drop" = frozen - candidate (positive is worse).
GATES = dict(
    real=dict(
        pixel_f1_drop=dict(v1=0.02, raw=0.03),
        pixel_precision_drop=dict(v1=0.03, raw=0.05),
        pixel_recall_drop=dict(v1=0.03, raw=0.05),
        v1_instance_f1_drop=0.05,
        v1_instance_precision_drop=0.10,
        v1_instance_recall_drop=0.10,
        v1_complete_recall_drop=0.05,
        nodata_ratio=dict(raw_mask_pixels_in_nodata=1.10, v1_unclipped_outside_footprint=1.10),
        v1_accepted_after_clipping_in_nodata=0,
        v1_false_positive_px_ratio=1.50,
        v1_wrong_class_px_ratio=1.50,
        v1_disconnected_extra=2,
        v1_multi_instance_extra=1,
    ),
    synthetic=dict(v1_pixel_f1_drop=0.02, v1_pixel_precision_drop=0.03),
)
EARLY_STOP = dict(any_real_pixel_f1_drop=0.05, nodata_ratio=1.50, v1_false_positive_px_ratio=2.0)


@torch.no_grad()
def structure(model, processor, development):
    """V1 accepted-mask structure and per-crop proposal latency."""
    out = Counter()
    latency = []
    for root, crop in development:
        pixel, masks, labels, validity, _ = load_crop(root, crop, processor, "cuda")
        valid = validity[0].cpu().numpy().astype(bool)
        truth = masks[0].bool().cpu().numpy()
        tl = labels[0].cpu().numpy()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pr = proposals(model, pixel, valid)
        torch.cuda.synchronize()
        latency.append(time.perf_counter() - t0)
        # Same V1 filter as evaluate(): overlap computed on valid support.
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
        for i in sel:
            m = pr["masks"][i] & valid
            comps, n = connected(m, structure=np.ones((3, 3), bool))
            big = sum(int((comps == c).sum()) >= 4 for c in range(1, n + 1))
            hits = [k for k in range(len(tl)) if (m & truth[k]).sum() >= 10]
            same = [k for k in hits if tl[k] == pr["labels"][i]]
            out["v1_accepted"] += 1
            out["v1_disconnected"] += int(big > 1)
            out["v1_multi_instance"] += int(len(hits) >= 2)
            out["v1_multi_dash_instance"] += int(pr["labels"][i] == 1 and len(same) >= 2)
            out["v1_disconnected_multi_instance"] += int(big > 1 and len(hits) >= 2)
    return dict(out), latency


def run(model, processor, real_dev, syn_dev):
    model.eval()
    real = evaluate(model, processor, real_dev)
    real["structure"], lat_r = structure(model, processor, real_dev)
    syn = evaluate(model, processor, syn_dev)
    syn.pop("nodata", None)  # synthetic NoData is not defined
    syn["structure"], lat_s = structure(model, processor, syn_dev)
    return dict(
        real=real,
        synthetic=syn,
        latency_ms=dict(
            real_median=1000 * float(np.median(lat_r)),
            synthetic_median=1000 * float(np.median(lat_s)),
        ),
    )


def _f(d, *keys):
    for k in keys:
        d = d[k]
    return d or 0.0


def gate_table(res, base):
    g, checks = GATES["real"], {}
    r, b = res["real"], base["real"]
    for lvl in LEVELS:
        for c in CLASSES:
            for metric, key in (
                ("f1", "pixel_f1_drop"),
                ("precision", "pixel_precision_drop"),
                ("recall", "pixel_recall_drop"),
            ):
                d = _f(b, lvl, "pixel", c, metric) - _f(r, lvl, "pixel", c, metric)
                checks[f"real_{lvl}_{c}_pixel_{metric}_drop<={g[key][lvl]}"] = (
                    d <= g[key][lvl],
                    round(d, 4),
                )
    for c in CLASSES:
        for metric in ("f1", "precision", "recall"):
            tol = g[f"v1_instance_{metric}_drop"]
            d = _f(b, "v1", "instance_iou50", c, metric) - _f(r, "v1", "instance_iou50", c, metric)
            checks[f"real_v1_{c}_instance_{metric}_drop<={tol}"] = (d <= tol, round(d, 4))
        d = b["v1"]["complete_marking_recall"][c] - r["v1"]["complete_marking_recall"][c]
        tol = g["v1_complete_recall_drop"]
        checks[f"real_v1_{c}_complete_recall_drop<={tol}"] = (d <= tol, round(d, 4))
    for key, ratio in g["nodata_ratio"].items():
        bb, rr = b["nodata"].get(key, 0), r["nodata"].get(key, 0)
        checks[f"real_{key}<={ratio}x"] = (rr <= ratio * max(bb, 1), [bb, rr])
    acc = r["nodata"].get("v1_accepted_after_clipping_in_nodata", 0)
    checks["real_v1_accepted_after_clipping_in_nodata==0"] = (acc == 0, acc)
    fp_b = b["v1"]["cross_class"]["predicted_on_verified_background"]
    fp_r = r["v1"]["cross_class"]["predicted_on_verified_background"]
    tol = g["v1_false_positive_px_ratio"]
    checks[f"real_v1_false_positive_px<={tol}x"] = (fp_r <= tol * max(fp_b, 1), [fp_b, fp_r])
    wc_b = sum(b["v1"]["cross_class"][k] for k in ("solid_as_dash", "dash_as_solid"))
    wc_r = sum(r["v1"]["cross_class"][k] for k in ("solid_as_dash", "dash_as_solid"))
    tol = g["v1_wrong_class_px_ratio"]
    checks[f"real_v1_wrong_class_px<={tol}x"] = (wc_r <= tol * max(wc_b, 1), [wc_b, wc_r])
    for key, extra in (
        ("v1_disconnected", g["v1_disconnected_extra"]),
        ("v1_multi_instance", g["v1_multi_instance_extra"]),
    ):
        bb, rr = b["structure"].get(key, 0), r["structure"].get(key, 0)
        checks[f"real_{key}<=frozen+{extra}"] = (rr <= bb + extra, [bb, rr])
    s, sb, gs = res["synthetic"], base["synthetic"], GATES["synthetic"]
    for c in CLASSES:
        d = _f(sb, "v1", "pixel", c, "f1") - _f(s, "v1", "pixel", c, "f1")
        checks[f"synthetic_v1_{c}_pixel_f1_drop<={gs['v1_pixel_f1_drop']}"] = (
            d <= gs["v1_pixel_f1_drop"],
            round(d, 4),
        )
        d = _f(sb, "v1", "pixel", c, "precision") - _f(s, "v1", "pixel", c, "precision")
        checks[f"synthetic_v1_{c}_pixel_precision_drop<={gs['v1_pixel_precision_drop']}"] = (
            d <= gs["v1_pixel_precision_drop"],
            round(d, 4),
        )
    return checks, all(v[0] for v in checks.values())


def early_stop(res, base):
    r, b = res["real"], base["real"]
    reasons = []
    for lvl in LEVELS:
        for c in CLASSES:
            d = _f(b, lvl, "pixel", c, "f1") - _f(r, lvl, "pixel", c, "f1")
            if d > EARLY_STOP["any_real_pixel_f1_drop"]:
                reasons.append(f"{lvl}_{c}_pixel_f1_drop={d:.3f}")
    for key in ("raw_mask_pixels_in_nodata", "v1_unclipped_outside_footprint"):
        if r["nodata"].get(key, 0) > EARLY_STOP["nodata_ratio"] * max(b["nodata"].get(key, 0), 1):
            reasons.append(f"{key}>{EARLY_STOP['nodata_ratio']}x")
    fp_b = b["v1"]["cross_class"]["predicted_on_verified_background"]
    if r["v1"]["cross_class"]["predicted_on_verified_background"] > EARLY_STOP[
        "v1_false_positive_px_ratio"
    ] * max(fp_b, 1):
        reasons.append("v1_false_positive_px>2x")
    return reasons


def mean_v1_f1_gain(res, base, domain):
    return {
        c: _f(res, domain, "v1", "pixel", c, "f1") - _f(base, domain, "v1", "pixel", c, "f1")
        for c in CLASSES
    }


__all__ = ["CLASSES", "EARLY_STOP", "GATES", "early_stop", "gate_table", "mean_v1_f1_gain", "run"]
