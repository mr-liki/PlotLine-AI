"""Label-preserving synthetic LiDAR-style augmentation matched to approved real statistics.

Every geometric operation is applied identically to image, targets, validity and NoData.
No operation creates a positive label: targets can only shrink or move to ignore.

* Exact geometry: 90-degree rotations and flips.
* Appearance: a global monotone tone curve maps synthetic background median / spread and
  paint median onto a sampled real (background, spread, contrast) triple, with optional
  fading of paint contrast. The curve uses image-level statistics only and treats every
  synthetic pixel as an observation (no NoData is inferred from dark pixels).
* Scale: synthetic and real pixel scales are unknown relative to each other, so the image is
  area-resampled by a random factor (thinner strokes) and placed in a 512 canvas. Output pixels
  are targets only when >= 50 % covered by one target and untouched by ignored source pixels;
  they are background only when fully covered by verified background; everything else is
  ignored. The empty canvas is known-unobserved NoData (zero intensity).
* Observation support: an optional straight half-plane cut (partial markings at a corridor
  edge) and iid point dropout (sparse returns). Removed pixels are zeroed and become NoData.
"""

from dataclasses import dataclass

import cv2
import numpy as np

MIN_TARGET_PX = {0: 4, 1: 8}


@dataclass(frozen=True)
class AugmentConfig:
    tone_probability: float = 0.8
    fade_probability: float = 0.3
    fade_range: tuple = (0.4, 1.0)
    scale_probability: float = 0.7
    scale_range: tuple = (0.3, 1.0)
    cut_probability: float = 0.3
    cut_fraction: tuple = (0.15, 0.5)
    dropout_probability: float = 0.6
    dropout_rate: tuple = (0.1, 0.45)


def tone_curve(gray, fg, valid, target, fade):
    """Monotone piecewise-linear intensity map; returns (image, knots_x, knots_y)."""
    bg = gray[valid & ~fg]
    if not len(bg):
        return gray, None, None
    bg_s, sd_s = float(np.median(bg)), float(bg.std())
    paint_s = float(np.median(gray[fg])) if fg.any() else bg_s + 2 * sd_s
    bg_t, sd_t, contrast_t = target
    paint_t = bg_t + contrast_t * fade
    xs = [0.0, bg_s - sd_s, bg_s + sd_s, paint_s, 255.0]
    ys = [1.0, bg_t - sd_t, bg_t + sd_t, paint_t, 255.0]
    kx, ky = [xs[0]], [ys[0]]
    for x, y in zip(xs[1:], ys[1:], strict=True):
        y = float(np.clip(y, 1, 255))
        if x > kx[-1] + 1e-6 and y > ky[-1] + 1e-6:
            kx.append(x)
            ky.append(y)
    if len(kx) < 2:
        return gray, None, None
    out = np.interp(gray.astype(float), kx, ky)
    return np.clip(np.round(out), 1, 255).astype(np.uint8), kx, ky


def _resize(a, n):
    return cv2.resize(a.astype(np.float32), (n, n), interpolation=cv2.INTER_AREA)


def augment(gray, masks, labels, valid, rng, real_triples, config=None):
    """Return dict(gray, masks, labels, valid, nodata, record). Inputs are 512x512 native."""
    config = config or AugmentConfig()
    size = gray.shape[0]
    masks = np.asarray(masks, bool).reshape(-1, size, size)
    labels = labels_in = np.asarray(labels, np.int64)
    valid = valid.astype(bool)
    record = {}
    k, flip = int(rng.integers(4)), bool(rng.random() < 0.5)
    record.update(rot90=k, flip=flip)

    def geo(a):
        a = np.rot90(a, k, axes=(-2, -1))
        return np.flip(a, -1) if flip else a

    gray, valid = geo(gray).copy(), geo(valid).copy()
    masks = geo(masks).copy()
    fg = masks.any(0) if len(masks) else np.zeros_like(valid)
    if rng.random() < config.tone_probability and len(real_triples):
        target = real_triples[int(rng.integers(len(real_triples)))]
        fade = (
            float(rng.uniform(*config.fade_range))
            if rng.random() < config.fade_probability
            else 1.0
        )
        gray, kx, ky = tone_curve(gray, fg, valid, target, fade)
        record.update(tone_target=[float(v) for v in target], fade=fade, knots=[kx, ky])
    nodata = np.zeros_like(valid)
    if rng.random() < config.scale_probability:
        s = float(np.exp(rng.uniform(np.log(config.scale_range[0]), np.log(config.scale_range[1]))))
        n = max(32, round(size * s))
        oy, ox = (int(rng.integers(0, size - n + 1)) for _ in range(2))
        ignore = ~valid
        bg_frac = _resize(valid & ~fg, n)
        ign_frac = _resize(ignore, n)
        new_masks = np.zeros((len(masks), size, size), bool)
        for i, m in enumerate(masks):
            frac = _resize(m, n)
            new_masks[i, oy : oy + n, ox : ox + n] = (frac >= 0.5) & (ign_frac <= 1e-6)
        canvas = np.zeros((size, size), np.uint8)
        canvas[oy : oy + n, ox : ox + n] = np.clip(np.round(_resize(gray, n)), 1, 255).astype(
            np.uint8
        )
        new_valid = np.zeros((size, size), bool)
        new_valid[oy : oy + n, ox : ox + n] = bg_frac >= 1 - 1e-6
        nodata = np.ones((size, size), bool)
        nodata[oy : oy + n, ox : ox + n] = False
        gray, masks = canvas, new_masks
        valid = new_valid | masks.any(0)
        record.update(scale=s, placed=[oy, ox, n])
    observed = ~nodata
    if rng.random() < config.cut_probability:
        ys, xs = np.nonzero(observed)
        cy, cx = ys.mean(), xs.mean()
        theta = float(rng.uniform(0, 2 * np.pi))
        d = (ys - cy) * np.sin(theta) + (xs - cx) * np.cos(theta)
        frac = float(rng.uniform(*config.cut_fraction))
        cut = d > np.quantile(d, 1 - frac)
        removed = np.zeros_like(observed)
        removed[ys[cut], xs[cut]] = True
        nodata |= removed
        record.update(cut_theta=theta, cut_fraction=frac)
    if rng.random() < config.dropout_probability:
        rate = float(rng.uniform(*config.dropout_rate))
        nodata |= (rng.random(gray.shape) < rate) & ~nodata
        record.update(dropout_rate=rate)
    gray = np.where(nodata, 0, gray).astype(np.uint8)
    valid &= ~nodata
    masks &= valid[None]
    keep = []
    for i, (m, lab) in enumerate(zip(masks, labels, strict=True)):
        if m.sum() >= MIN_TARGET_PX[int(lab)]:
            keep.append(i)
        else:
            valid &= ~m  # too little visible evidence: ignore, never background
    masks, labels = masks[keep], labels[keep]
    if len(masks):
        assert not (masks & ~valid[None]).any()
    assert not (valid & nodata).any() and not gray[nodata].any()
    record.update(targets_kept=len(keep), targets_dropped=len(labels_in) - len(keep))
    return dict(gray=gray, masks=masks, labels=labels, valid=valid, nodata=nodata, record=record)


__all__ = ["MIN_TARGET_PX", "AugmentConfig", "augment", "tone_curve"]
