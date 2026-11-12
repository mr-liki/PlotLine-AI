"""Stage-4 training augmentations for native 512 LiDAR-intensity crops.

Every transform acts jointly on (gray, masks, validity) and preserves the supervision rules:
pixels removed by simulated NoData become invalid (never background), and any instance left
with fewer than 4 valid pixels is dropped with its remaining pixels made invalid, exactly as
the dataset export does.
"""

import numpy as np


def dihedral(gray, masks, valid, k, flip):
    def op(a):
        a = np.rot90(a, k, axes=(-2, -1))
        return np.flip(a, -1) if flip else a

    return op(gray).copy(), op(masks).copy(), op(valid).copy()


def intensity(gray, valid, rng, gain=(0.85, 1.15), gamma=(0.8, 1.25), ramp=0.15):
    g = gray.astype(np.float32) / 255.0
    out = np.power(np.clip(g, 0, 1), rng.uniform(*gamma)) * rng.uniform(*gain)
    theta = rng.uniform(0, np.pi)
    yy, xx = np.mgrid[0 : g.shape[0], 0 : g.shape[1]]
    t = (np.cos(theta) * xx + np.sin(theta) * yy) / max(g.shape)
    out *= 1 + rng.uniform(-ramp, ramp) * (t - t.mean()) * 2
    out = np.where(valid, np.clip(out, 0, 1), g)
    return (out * 255).round().astype(np.uint8)


def nodata(valid, rng, blob_p=0.5, side_p=0.3):
    drop = np.zeros_like(valid)
    h, w = valid.shape
    yy, xx = np.mgrid[0:h, 0:w]
    if rng.random() < blob_p:
        for _ in range(rng.integers(1, 4)):
            cy, cx = rng.uniform(0, h), rng.uniform(0, w)
            ry, rx = rng.uniform(0.04, 0.16) * h, rng.uniform(0.04, 0.16) * w
            drop |= ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
    if rng.random() < side_p:
        theta = rng.uniform(0, 2 * np.pi)
        t = (np.cos(theta) * xx + np.sin(theta) * yy) / max(h, w)
        t = (t - t.min()) / max(1e-9, t.max() - t.min())
        drop |= rng.random(valid.shape) < np.clip(t - 0.5, 0, 0.5) * 1.2
    return valid & ~drop


def enforce_min_pixels(masks, labels, valid, minimum=4):
    masks = masks & valid
    keep = masks.sum((1, 2)) >= minimum if len(masks) else np.zeros(0, bool)
    if len(masks) and (~keep).any():
        valid = valid & ~masks[~keep].any(0)
    return masks[keep], labels[keep], valid


def augment(rgb, masks, labels, valid, rng, use_dihedral=True, use_intensity=True, use_nodata=True):
    gray = rgb[..., 0]
    masks = masks.astype(bool)
    valid = valid.astype(bool)
    if use_dihedral:
        gray, masks, valid = dihedral(
            gray, masks, valid, int(rng.integers(0, 4)), bool(rng.integers(0, 2))
        )
    if use_intensity:
        gray = intensity(gray, valid, rng)
    if use_nodata:
        new_valid = nodata(valid, rng)
        # Keep at least 30% of the original supervision; otherwise skip this step.
        if new_valid.sum() >= 0.3 * valid.sum():
            gray = np.where(new_valid | ~valid, gray, 0).astype(np.uint8)
            valid = new_valid
    masks, labels, valid = enforce_min_pixels(masks, labels, valid)
    if not valid.any():
        raise ValueError("augmentation removed all supervision")
    return np.repeat(gray[..., None], 3, -1), masks, labels, valid
