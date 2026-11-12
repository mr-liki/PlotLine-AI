"""Stage 9E-E: non-text hard-negative coverage audit and class-specific false positives.

Client reference geometry (no text class exists in it) locates non-line paint: arrows,
detour arrows, diagonal/chevron hatching, give-way, colour bands, safety markings and
unknown symbols. For every approved training crop, development crop and N9_01 we measure
how much of that paint is present and whether supervision ignores it (validity False) or
would wrongly treat it as verified background. Client geometry is a locator, not an
approved label; it never becomes a training target here.

False positives of the rejected Stage 9D three-class candidate (text) and of the frozen
model (solid/dash) are measured per spatial category at the raw and V1 levels.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
import torch
from affine import Affine
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation
from shapely import wkt
from shapely.geometry import box
from shapely.strtree import STRtree
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from phase5_11_common import client_features
from roadmarking.inference.calibration import filter_instances, iou_matrix
from roadmarking.training.pilot import load_crop
from stage9c_attested_pilot import crop_box, text_arrays
from stage9d_common import ROOT, dump, frozen_checks, read, verify_frozen
from stage9e_lib import V1, development_crops, new_run, proposals, real_context

NAMES = {
    3: "arrow",
    4: "detour_arrow",
    5: "diagonal_hatching",
    6: "give_way",
    7: "colour_band",
    8: "safety_marking",
    -1: "unknown_symbol",
}


def geometry(feature):
    g = feature["geometry"]
    return wkt.loads(g) if isinstance(g, str) else g


def symbol_raster(tree, geoms, vids, footprint, transform, shape):
    out = {}
    for i in tree.query(footprint):
        v = vids[i]
        if v in NAMES:
            r = rasterize(
                [(geoms[i], 1)], out_shape=shape, transform=transform, all_touched=True
            ).astype(bool)
            out[v] = out.get(v, np.zeros(shape, bool)) | r
    return out


def accepted(pr, three_class, valid):
    if not len(pr["labels"]):
        return {c: np.zeros(0, int) for c in ("raw", "v1")}
    ov = iou_matrix(pr["masks"], pr["masks"], valid)
    conf = {**V1["confidence"], "2": 0.5 if three_class else 1.01}
    sel = filter_instances(
        pr["scores"], pr["labels"], ov, pr["qualities"], dict(V1, confidence=conf)
    )
    return {"raw": np.arange(len(pr["labels"])), "v1": np.array(sel, int)}


def categorize(pred_by_class, valid, cats):
    rows = {}
    for cname, cmask in cats.items():
        denom = int(cmask.sum())
        rows[cname] = dict(
            pixels=denom,
            **{f"predicted_{k}": int((p & cmask).sum()) for k, p in pred_by_class.items()},
        )
    return rows


def main():
    out = new_run("hard_negatives")
    checks = frozen_checks()
    verify_frozen(checks)
    rc = real_context()
    ctx = rc["ctx"]
    feats, _, _ = client_features(check_network=False)
    geoms = [geometry(f) for f in feats]
    vids = [int(f["visual_id"]) for f in feats]
    tree = STRtree(geoms)
    # --- coverage in approved training crops
    coverage = {name: Counter() for name in NAMES.values()}
    for crop in rc["full"]:
        with np.load(rc["dataset"] / crop["file"]) as z:
            valid = z["validity"].copy()
            rgb = z["rgb"][..., 0]
        transform = Affine(*crop["affine"][:6])
        sym = symbol_raster(tree, geoms, vids, crop_box(crop), transform, valid.shape)
        for v, m in sym.items():
            c = coverage[NAMES[v]]
            observed = m & (rgb > 0)
            c["crops"] += 1
            c["observed_pixels"] += int(observed.sum())
            c["observed_pixels_ignored"] += int((observed & ~valid).sum())
            c["observed_pixels_valid_background"] += int(
                (observed & valid & ~z_any(crop, rc)).sum()
            )
            c[f"geography_{crop['geography']}"] += 1
    coverage = {k: dict(v) for k, v in coverage.items()}
    # --- false positives: frozen and rejected Stage 9D candidate
    frozen_path = Path(checks[0]["path"]).parent
    processor = Mask2FormerImageProcessor.from_pretrained(
        frozen_path.parent / "processor", local_files_only=True
    )
    gate = read(
        Path(read(ROOT / "outputs/reports/stage9d/finetune/latest.json")["run"])
        / "gate_evaluation.json"
    )
    models = {
        "frozen_two_class": frozen_path,
        "stage9d_rejected_three_class": Path(gate["candidate_corrected_copy"]) / "model",
    }
    fp = {}
    for mname, mpath in models.items():
        model = (
            Mask2FormerForUniversalSegmentation.from_pretrained(mpath, local_files_only=True)
            .cuda()
            .eval()
        )
        three = model.config.num_labels == 3
        totals = {}
        # development crops
        for root, crop in development_crops(ctx):
            pixel, masks, labels, validity, rgb = load_crop(root, crop, processor, "cuda")
            valid = validity[0].cpu().numpy().astype(bool)
            observed = rgb[..., 0] > 0
            truth = masks[0].bool().cpu().numpy()
            labels[0].cpu().numpy()
            lines = truth.any(0) if len(truth) else np.zeros_like(valid)
            sym = symbol_raster(
                tree, geoms, vids, crop_box(crop), Affine(*crop["affine"][:6]), valid.shape
            )
            sym_any = np.any(list(sym.values()), axis=0) if sym else np.zeros_like(valid)
            near = binary_dilation(lines | sym_any, iterations=3)
            pav = observed & ~near
            bright_cut = np.percentile(rgb[..., 0][pav], 99) if pav.any() else 255
            cats = dict(
                lane_paint=lines & valid,
                symbols_observed=sym_any & observed,
                pavement_valid=pav & valid & (rgb[..., 0] < bright_cut),
                bright_non_paint=pav & (rgb[..., 0] >= bright_cut),
                nodata=~observed,
            )
            pr = proposals(model, pixel, valid)  # scores per contract; masks returned unclipped
            sel = accepted(pr, three, valid)
            for lvl in ("raw", "v1"):
                pm, plab = pr["masks"][sel[lvl]], pr["labels"][sel[lvl]]
                pred = {
                    ("solid", "dash", "text")[c]: (
                        pm[plab == c].any(0) if (plab == c).any() else np.zeros_like(valid)
                    )
                    for c in range(3)
                }
                for cname, row in categorize(pred, valid, cats).items():
                    t = totals.setdefault(f"dev_{lvl}", {}).setdefault(cname, Counter())
                    t.update(row)
        # N9_01
        image, positive, non_text, observed = text_arrays(ctx[0], ctx[9], ctx[10])
        with rasterio.open(ctx[9]["layers"]["observed"]) as ds:
            transform = ds.transform
        sym = symbol_raster(tree, geoms, vids, box(*ctx[9]["bounds"]), transform, (512, 512))
        sym = {k: v[: positive.shape[0], : positive.shape[1]] for k, v in sym.items()}
        sym_any = np.any(list(sym.values()), axis=0) if sym else np.zeros_like(positive)
        near = binary_dilation(positive | non_text | sym_any, iterations=3)
        pav = observed & ~near
        cut = np.percentile(image[..., 0][pav], 99)
        cats = dict(
            reviewed_text=positive,
            lane_paint=non_text,
            symbols_observed=sym_any & observed & ~positive,
            pavement=pav & (image[..., 0] < cut),
            bright_non_paint=pav & (image[..., 0] >= cut),
            nodata=~observed,
        )
        pixel = processor(images=image, return_tensors="pt").pixel_values.cuda()
        pr = proposals(model, pixel, observed)
        sel = accepted(pr, three, observed)
        for lvl in ("raw", "v1"):
            pm, plab = pr["masks"][sel[lvl]], pr["labels"][sel[lvl]]
            pred = {
                ("solid", "dash", "text")[c]: (
                    pm[plab == c].any(0) if (plab == c).any() else np.zeros_like(observed)
                )
                for c in range(3)
            }
            totals[f"n9_01_{lvl}"] = {
                k: Counter(v) for k, v in categorize(pred, observed, cats).items()
            }
        totals["n9_01_symbol_classes_present"] = {
            NAMES[k]: int((v & observed).sum()) for k, v in sym.items()
        }
        fp[mname] = {
            k: (
                {kk: dict(vv) for kk, vv in v.items()}
                if isinstance(v, dict) and v and isinstance(next(iter(v.values())), Counter)
                else v
            )
            for k, v in totals.items()
        }
        del model
        torch.cuda.empty_cache()
    # rates
    for block in fp.values():
        for key, cats in block.items():
            if not key.startswith(("dev_", "n9_01_r", "n9_01_v")):
                continue
            for row in cats.values():
                for k in ("predicted_solid", "predicted_dash", "predicted_text"):
                    row[k.replace("predicted", "rate")] = row.get(k, 0) / max(row["pixels"], 1)
    policy = dict(
        current=(
            "Unsupported paint (visual_id not 1/2) is dilated 2 px and set to ignore in every "
            "approved crop; it is never background and never a target."
        ),
        proposed_for_three_class=(
            "Keep symbols ignored for solid/dash. Add them as verified non-text context for the "
            "text class only after human confirmation of each symbol footprint in QGIS: a text-"
            "class query whose mask lies on confirmed non-text paint is penalized; solid/dash "
            "supervision is unchanged. Client geometry alone is not sufficient approval."
        ),
        text_class_in_client_reference=0,
    )
    verify_frozen(checks)
    dump(
        out / "hard_negative_audit.json",
        dict(
            training_crop_coverage=coverage, false_positives=fp, policy=policy, symbol_classes=NAMES
        ),
    )
    print(out)


_TARGET_CACHE = {}


def z_any(crop, rc):
    """Union of approved solid/dash targets for a crop (cached)."""
    if crop["crop_id"] not in _TARGET_CACHE:
        with np.load(rc["dataset"] / crop["file"]) as z:
            m = z["masks"]
            _TARGET_CACHE[crop["crop_id"]] = (
                m.any(0) if len(m) else np.zeros(z["validity"].shape, bool)
            )
    return _TARGET_CACHE[crop["crop_id"]]


if __name__ == "__main__":
    main()
