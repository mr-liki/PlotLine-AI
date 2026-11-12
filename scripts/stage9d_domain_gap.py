"""Stage 9D-G: real/synthetic domain-gap measurement and a predeclared factorial probe.

The probe (scale x observation density) is analysis-only. It is never used to select a
checkpoint or tune thresholds, and none of its transformed images are trained on.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_fill_holes, distance_transform_edt, zoom
from scipy.ndimage import label as connected
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from stage9c_attested_pilot import source_context, text_arrays
from stage9d_common import (
    BASE,
    SYNTH,
    TEXT,
    counts,
    dump,
    frozen_checks,
    metrics,
    new_run,
    read,
    synthetic_stems,
    verify_frozen,
)
from stage9d_eval import n9_groups, predict_full

PROBE = dict(
    n9_upscale=[1, 2, 3, 4],
    synthetic_downscale=[1.0, 0.5, 0.33, 0.25],
    synthetic_density=["native", "real_matched"],
    seed=9043,
)


def pca_extent(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) < 10:
        return None
    pts = np.stack([ys, xs], 1).astype(float)
    pts -= pts.mean(0)
    _, vecs = np.linalg.eigh(np.cov(pts.T))
    proj = pts @ vecs
    minor, major = np.ptp(proj[:, 0]), np.ptp(proj[:, 1])
    angle = float(np.degrees(np.arctan2(vecs[0, 1], vecs[1, 1])) % 180)
    return dict(minor_extent_px=float(minor), major_extent_px=float(major), orientation_deg=angle)


def local_contrast(gray, target, valid):
    dil = distance_transform_edt(~target) <= 4
    ring = dil & ~target & valid
    if not ring.any() or not (target & valid).any():
        return None
    return float(gray[target & valid].mean() - gray[ring].mean())


def synthetic_profile(stem):
    split = SYNTH / "val"
    arr = {
        f: np.array(Image.open(split / f / f"{stem}.png"))
        for f in ("images", "masks", "clean_masks")
    }
    img, vis, clean = arr["images"].astype(float), arr["masks"] > 0, arr["clean_masks"] > 0
    valid = ~(clean & ~vis)
    ext = pca_extent(clean)
    edt = distance_transform_edt(clean)
    return dict(
        fg=float(img[vis].mean()),
        bg=float(img[~clean].mean()),
        bg_std=float(img[~clean].std()),
        local_contrast=local_contrast(img, vis, valid),
        stroke_width_intended_px=float(2 * np.median(edt[edt > 0])),
        observation_density_in_text=float(vis.sum() / clean.sum()),
        observation_density_background=float((img[~clean] > 0).mean()),
        components_per_1000_fg=float(connected(vis)[1] / vis.sum() * 1000),
        phrase_minor_extent_px=ext["minor_extent_px"],
        phrase_major_extent_px=ext["major_extent_px"],
        orientation_deg=ext["orientation_deg"],
        text_area_fraction=float(clean.mean()),
        letter_holes_px=int((binary_fill_holes(clean) & ~clean).sum()),
    )


def n9_profile(rgb, text, observed, groups):
    gray = rgb[..., 0].astype(float)
    bg = observed & ~text
    filled = np.zeros_like(text)
    group_rows = {}
    for g, rec in groups.items():
        t = text & rec["box"]
        ext = pca_extent(t)
        # Intended stroke footprint approximated by closing observed text points
        from scipy.ndimage import binary_closing

        closed = binary_closing(t, iterations=2) & rec["box"]
        filled |= closed
        edt = distance_transform_edt(closed)
        group_rows[g] = dict(
            statuses=rec["statuses"],
            text_pixels=int(t.sum()),
            observed_fraction_in_closed_footprint=float(t.sum() / max(closed.sum(), 1)),
            stroke_width_closed_px=float(2 * np.median(edt[edt > 0])) if (edt > 0).any() else None,
            **(ext or {}),
        )
    return dict(
        fg=float(gray[text].mean()),
        bg=float(gray[bg].mean()),
        bg_std=float(gray[bg].std()),
        local_contrast=local_contrast(gray, text, observed),
        observation_density_tile=float(observed.mean()),
        observation_density_in_text_footprint=float(text.sum() / max(filled.sum(), 1)),
        components_per_1000_fg=float(connected(text)[1] / text.sum() * 1000),
        groups=group_rows,
    )


def evaluate_scaled_n9(model, processor, rgb, text, non_text, observed, factor):
    big = lambda a: np.kron(a, np.ones((factor, factor), dtype=a.dtype))  # noqa: E731
    img = np.stack([big(rgb[..., i]) for i in range(3)], -1)
    obs, pos, neg = big(observed), big(text), big(non_text)
    pixels = processor(images=img, return_tensors="pt").pixel_values.cuda()
    res = predict_full(model, pixels, obs)
    pred = res["post"][TEXT]
    return dict(
        factor=factor,
        text=metrics(counts(pred, pos, pos | neg)),
        unscored_observed_text_pixels=int((pred & obs & ~(pos | neg)).sum()),
        max_text_prob=res["max_text_prob"],
    )


def evaluate_scaled_synthetic(model, processor, stems, scale, density, rng, n9_density):
    total = np.zeros(3, np.int64)
    for stem in stems:
        split = SYNTH / "val"
        img = np.array(Image.open(split / "images" / f"{stem}.png")).astype(float)
        vis = np.array(Image.open(split / "masks" / f"{stem}.png")) > 0
        clean = np.array(Image.open(split / "clean_masks" / f"{stem}.png")) > 0
        if not vis.any():
            continue
        if scale != 1.0:
            img = zoom(img, scale, order=1)
            vis = zoom(vis.astype(float), scale, order=0) > 0.5
            clean = zoom(clean.astype(float), scale, order=0) > 0.5
            canvas = np.zeros((512, 512))
            cv, cc = np.zeros((512, 512), bool), np.zeros((512, 512), bool)
            h, w = img.shape
            canvas[:h, :w], cv[:h, :w], cc[:h, :w] = img, vis, clean
            img, vis, clean = canvas, cv, cc
        observed = np.ones_like(vis)
        if density == "real_matched":
            observed = rng.random(vis.shape) < n9_density
            img = img * observed
        valid = ~(clean & ~vis) & observed
        target = vis & observed
        rgb = np.repeat(img.clip(0, 255).round().astype(np.uint8)[..., None], 3, -1)
        pixels = processor(images=rgb, return_tensors="pt").pixel_values.cuda()
        res = predict_full(model, pixels, valid)
        total += counts(res["post"][TEXT], target, valid)
    return dict(scale=scale, density=density, text=metrics(total))


def main():
    out = new_run("domain_gap")
    checks = frozen_checks()
    verify_frozen(checks)
    ctx = source_context()
    reports, tile, spec = ctx[0], ctx[9], ctx[10]
    rgb, text, non_text, observed = text_arrays(reports, tile, spec)
    groups = n9_groups(spec, tile, text.shape)
    stems = synthetic_stems("val")
    syn = [synthetic_profile(s) for s in stems]
    syn_summary = {
        k: dict(
            median=float(np.median([r[k] for r in syn if r[k] is not None])),
            p10=float(np.percentile([r[k] for r in syn if r[k] is not None], 10)),
            p90=float(np.percentile([r[k] for r in syn if r[k] is not None], 90)),
        )
        for k in syn[0]
    }
    n9 = n9_profile(rgb, text, observed, groups)
    finetune = Path(read(BASE / "finetune/latest.json")["run"])
    decision = read(finetune / "gate_evaluation.json")
    candidate = Path(decision["candidate_corrected_copy"]) / "model"
    processor = Mask2FormerImageProcessor.from_pretrained(
        Path(checks[0]["path"]).parent.parent / "processor", local_files_only=True
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(candidate, local_files_only=True)
    model.cuda().eval()
    rng = np.random.default_rng(PROBE["seed"])
    with torch.no_grad():
        probe_n9 = [
            evaluate_scaled_n9(model, processor, rgb, text, non_text, observed, f)
            for f in PROBE["n9_upscale"]
        ]
        probe_syn = [
            evaluate_scaled_synthetic(
                model, processor, stems, s, d, rng, n9["observation_density_tile"]
            )
            for s in PROBE["synthetic_downscale"]
            for d in PROBE["synthetic_density"]
        ]
    verify_frozen(checks)
    report = dict(
        candidate=str(candidate),
        probe_protocol=PROBE,
        probe_role="analysis only; not used for checkpoint selection or thresholds",
        synthetic_val_profile=syn_summary,
        n9_01_profile=n9,
        probe_n9_upscaled=probe_n9,
        probe_synthetic=probe_syn,
        unsupported_claims=dict(
            synthetic_geospatial_coordinates="none (no georeferencing metadata)",
            synthetic_gsd="unknown; scale compared only in pixels",
        ),
    )
    dump(out / "domain_gap_assessment.json", report)
    print(out)


if __name__ == "__main__":
    main()
