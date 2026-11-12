"""Realism audit of realistic_v1: do added markings look like real ones (to measurements and to
the frozen model)? Added and real markings are compared INSIDE the same generated images.

1. Width (px), local paint contrast, dash length/gap: added vs reviewed real markings.
2. Detectability parity: frozen-model raw/V1 pixel recall on added vs real marking pixels,
   per class, on the same images. Similar recall = the model cannot tell them apart.
3. Label safety: no target on unobserved pixels, no overlapping instances.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.morphology import skeletonize

from regsafe_common import ROOT, dump, frozen_paths, load_model, load_processor
from roadmarking.inference.calibration import filter_instances, iou_matrix
from stage9e_lib import V1, proposals

SAMPLES = 400


def n_added(rec):
    return sum(a["pieces"] if a["kind"] == "dash" else 1 for a in rec["added"])


def stats(mask, gray, valid):
    sk = skeletonize(mask)
    dt = distance_transform_edt(mask)
    width = float(np.median(2 * dt[sk])) if sk.any() else None
    ring = (
        binary_dilation(mask, iterations=4)
        & ~binary_dilation(mask, iterations=1)
        & valid
        & (gray > 0)
    )
    paint = gray[mask & (gray > 0)]
    con = (
        float(np.median(paint) - np.median(gray[ring]))
        if ring.sum() > 10 and len(paint) > 5
        else None
    )
    ys, xs = np.nonzero(mask)
    length = None
    if len(ys) >= 6:
        p = np.stack([ys, xs], 1).astype(float)
        p -= p.mean(0)
        _, evec = np.linalg.eigh(np.cov(p.T))
        length = float(np.ptp(p @ evec[:, 1]) + 1) * 0.05
    return width, con, length


def pct(a):
    a = [x for x in a if x is not None]
    return (
        dict(n=len(a), **{f"p{q}": float(np.percentile(a, q)) for q in (5, 25, 50, 75, 95)})
        if a
        else None
    )


@torch.no_grad()
def main():
    run = Path(json.loads((ROOT / "data/processed/realistic_v1/latest.json").read_text())["run"])
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    recs = json.loads((run / split / "records.json").read_text())
    rng = np.random.default_rng(5)
    pick = [recs[i] for i in rng.choice(len(recs), SAMPLES, replace=False)]
    _, frozen, _ = frozen_paths()
    model = load_model(frozen)
    proc = load_processor()
    agg = {k: {c: [] for c in ("width", "contrast", "length")} for k in ("added", "real")}
    agg_by_class = {
        f"{k}_{c}": {"width": [], "contrast": [], "length": []}
        for k in ("added", "real")
        for c in (0, 1)
    }
    hits = {
        f"{lvl}_{k}_{c}": [0, 0] for lvl in ("raw", "v1") for k in ("added", "real") for c in (0, 1)
    }
    safety = dict(target_on_unobserved=0, overlapping_pixels=0, images=0)
    for r in pick:
        d = np.load(r["file"])
        gray, masks, labels, valid = (
            d["rgb"][..., 0],
            d["masks"].astype(bool),
            d["labels"],
            d["validity"],
        )
        k = len(labels) - n_added(r["record"])
        safety["images"] += 1
        safety["overlapping_pixels"] += int((masks.sum(0) > 1).sum()) if len(masks) else 0
        safety["target_on_unobserved"] += (
            int((masks[k:] & (gray == 0)[None]).sum()) if len(masks) > k else 0
        )
        for i, (m, lab) in enumerate(zip(masks, labels, strict=True)):
            kind = "added" if i >= k else "real"
            if m.sum() < 8:
                continue
            w, c, length = stats(m, gray, valid)
            for key, val in (("width", w), ("contrast", c), ("length", length)):
                agg[kind][key].append(val)
                agg_by_class[f"{kind}_{lab}"][key].append(val)
        pixel = proc(images=d["rgb"], return_tensors="pt")["pixel_values"].cuda()
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
        for lvl, idx in (("raw", range(len(pr["labels"]))), ("v1", sel)):
            union = {c: np.zeros_like(valid) for c in (0, 1)}
            for i in idx:
                union[int(pr["labels"][i])] |= pr["masks"][i]
            for i, (m, lab) in enumerate(zip(masks, labels, strict=True)):
                kind = "added" if i >= k else "real"
                t = m & valid
                h = hits[f"{lvl}_{kind}_{lab}"]
                h[0] += int((union[int(lab)] & t).sum())
                h[1] += int(t.sum())
    recall = {k2: (a / b if b else None) for k2, (a, b) in hits.items()}
    parity = {
        f"{lvl}_{'solid' if c == 0 else 'dash'}": (recall[f"{lvl}_added_{c}"] or 0)
        - (recall[f"{lvl}_real_{c}"] or 0)
        for lvl in ("raw", "v1")
        for c in (0, 1)
    }
    result = dict(
        run=str(run),
        samples=SAMPLES,
        measurements={k: {m: pct(v) for m, v in d.items()} for k, d in agg_by_class.items()},
        frozen_pixel_recall=recall,
        detectability_gap_added_minus_real=parity,
        label_safety=safety,
    )
    result["split"] = split
    dump(run / f"realism_audit_{split}.json", result)
    print(
        json.dumps(
            dict(
                measurements={
                    k: {
                        m: (v and {q: round(v[q], 2) for q in ("p5", "p50", "p95", "n")})
                        for m, v in d.items()
                    }
                    for k, d in result["measurements"].items()
                },
                recall={k: (round(v, 3) if v is not None else None) for k, v in recall.items()},
                gap={k: round(v, 3) for k, v in parity.items()},
                safety=safety,
            ),
            indent=1,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
