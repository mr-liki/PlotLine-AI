"""Step 3: targeted causes for Model B's failed gates (inference only, 14 reused real crops).

A. NoData: each raw NoData pixel of the union is attributed to
   * paint-cell leakage: a same-class target query whose bilinear support cells (4 px output
     grid) contain that target's observed paint, so it cannot be lowered without lowering the
     logits that render the observed paint;
   * target-query leakage outside its own paint cells;
   * false-positive-query leakage.
   The paint-cell share is a floor for any correction that must not suppress observed paint.
B. Raw solid false positives: located per crop and classified as edge widening (<= 2 px from
   reviewed paint), wrong class (on dash), or isolated (verified background > 2 px from paint),
   by query kind and score band.
"""

import sys
from collections import Counter, defaultdict

import numpy as np
import torch
from scipy.ndimage import binary_dilation

from correction_common import BASE, dump, model_a_dir, model_b_dir
from regsafe_common import load_model, load_processor, nodata_mask
from roadmarking.training.pilot import load_crop
from stage9e_lib import V1, development_crops, proposals, real_context

STRIDE = 4


def support_cells(shape, stride=STRIDE):
    """For each native pixel, the (y0, y1, x0, x1) output cells bilinear sampling touches."""
    h, w = shape
    ph, pw = h // stride, w // stride
    y = np.clip((np.arange(h) + 0.5) * ph / h - 0.5, 0, ph - 1)
    x = np.clip((np.arange(w) + 0.5) * pw / w - 0.5, 0, pw - 1)
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    return y0, np.minimum(y0 + 1, ph - 1), x0, np.minimum(x0 + 1, pw - 1)


def protected_pixels(paint, cells):
    """Native pixels whose bilinear support touches an output cell containing ``paint``."""
    y0, y1, x0, x1 = cells
    h, w = paint.shape
    grid = np.zeros((h // STRIDE, w // STRIDE), bool)
    ys, xs = np.nonzero(paint)
    grid[ys // STRIDE, xs // STRIDE] = True
    return grid[y0][:, x0] | grid[y0][:, x1] | grid[y1][:, x0] | grid[y1][:, x1]


def band(score):
    t = float(V1["confidence"]["0"])
    return "lt0.5" if score < 0.5 else ("0.5_to_v1" if score < t else "ge_v1")


@torch.no_grad()
def analyze(model, processor, dev):
    nd = Counter()
    fp = Counter()
    per_crop = defaultdict(dict)
    for root, crop in dev:
        pixel, masks, labels, validity, rgb = load_crop(root, crop, processor, "cuda")
        valid = validity[0].cpu().numpy().astype(bool)
        nod = nodata_mask(valid, rgb)
        near = nod & binary_dilation(valid, iterations=2)
        truth = masks[0].bool().cpu().numpy()
        tl = labels[0].cpu().numpy()
        cells = support_cells(valid.shape)
        paint = truth.any(0) & valid if len(tl) else np.zeros_like(valid)
        dash = (truth[tl == 1].any(0) if (tl == 1).any() else np.zeros_like(valid)) & valid
        solid = (truth[tl == 0].any(0) if (tl == 0).any() else np.zeros_like(valid)) & valid
        edge = binary_dilation(paint, iterations=2) & valid & ~paint
        pr = proposals(model, pixel, valid)
        prot = np.zeros_like(valid)
        tq = np.zeros_like(valid)
        fq = np.zeros_like(valid)
        raw_solid = np.zeros_like(valid)
        fp_by = Counter()
        for m, lab, sc in zip(pr["masks"], pr["labels"], pr["scores"], strict=True):
            same = [truth[k] & valid for k in range(len(tl)) if tl[k] == lab]
            hits = [t for t in same if (m & t).sum() >= 10]
            if hits:
                tq |= m
                for t in hits:
                    prot |= m & protected_pixels(t, cells)
            else:
                fq |= m
            if lab == 0:
                raw_solid |= m
                wrong = m & valid & ~solid
                kind = "target_query" if hits else "fp_query"
                fp_by[f"{kind}|{band(sc)}"] += int(wrong.sum())
        u = (tq | fq) & nod
        p = u & prot
        t_only = u & tq & ~prot
        f_only = u & ~tq & ~prot
        for name, region in (
            ("paint_cell", p),
            ("target_outside_paint_cell", t_only),
            ("fp_query", f_only),
        ):
            nd[f"{name}"] += int(region.sum())
            nd[f"{name}_near"] += int((region & near).sum())
        nd["union"] += int(u.sum())
        sfp = raw_solid & valid & ~solid
        f = dict(
            edge_widening=int((sfp & edge & ~dash).sum()),
            wrong_class_on_dash=int((sfp & dash).sum()),
            isolated_background=int((sfp & ~edge & ~dash).sum()),
            tp=int((raw_solid & solid).sum()),
        )
        for k, v in f.items():
            fp[k] += v
        fp.update(fp_by)
        per_crop[crop["crop_id"]] = dict(solid_fp=f, nodata_union=int(u.sum()))
    return dict(nodata=dict(nd), raw_solid=dict(fp), per_crop=dict(per_crop))


def main():
    out = BASE / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    processor = load_processor()
    dev = development_crops(real_context()["ctx"])
    res = {}
    for name, path in (("frozen", model_a_dir() / "model"), ("step100", model_b_dir() / "model")):
        m = load_model(path)
        res[name] = analyze(m, processor, dev)
        del m
        torch.cuda.empty_cache()
    crops = res["frozen"]["per_crop"]
    res["solid_fp_delta_by_crop"] = sorted(
        (
            (
                cid,
                sum(
                    res["step100"]["per_crop"][cid]["solid_fp"][k]
                    for k in ("edge_widening", "wrong_class_on_dash", "isolated_background")
                )
                - sum(
                    crops[cid]["solid_fp"][k]
                    for k in ("edge_widening", "wrong_class_on_dash", "isolated_background")
                ),
                res["step100"]["per_crop"][cid]["solid_fp"],
                crops[cid]["solid_fp"],
            )
            for cid in crops
        ),
        key=lambda r: -r[1],
    )
    dump(out / "targeted_analysis.json", res)
    print(out)


if __name__ == "__main__":
    sys.exit(main())
