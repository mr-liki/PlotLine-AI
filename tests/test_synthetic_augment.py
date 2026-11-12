"""Label-preserving synthetic augmentation (regression-safe sprint)."""

from itertools import pairwise

import numpy as np

from roadmarking.training.synthetic_augment import AugmentConfig, augment, tone_curve

S = 512
TRIPLES = [(75.0, 34.0, 163.0), (60.0, 25.0, 90.0)]


def scene():
    gray = np.full((S, S), 20, np.uint8)
    masks = np.zeros((2, S, S), bool)
    masks[0, 100:110, 50:450] = True  # solid
    masks[1, 300:310, 60:120] = True  # dash
    gray[masks.any(0)] = 200
    valid = np.ones((S, S), bool)
    valid[400:420, 200:260] = False  # ignored unsupported paint
    gray[400:420, 200:260] = 210
    return gray, masks, np.array([0, 1]), valid


def test_exact_geometry_keeps_image_and_masks_aligned():
    cfg = AugmentConfig(
        tone_probability=0, scale_probability=0, cut_probability=0, dropout_probability=0
    )
    gray, masks, labels, valid = scene()
    for seed in range(8):
        out = augment(gray, masks, labels, valid, np.random.default_rng(seed), TRIPLES, cfg)
        assert (out["gray"][out["masks"].any(0)] == 200).all()
        assert out["masks"].sum() == masks.sum() and not out["nodata"].any()


def test_no_target_or_background_is_created_from_ignored_or_nodata():
    gray, masks, labels, valid = scene()
    for seed in range(20):
        out = augment(
            gray, masks, labels, valid, np.random.default_rng(seed), TRIPLES, AugmentConfig()
        )
        m, v, nd = out["masks"], out["valid"], out["nodata"]
        assert not (m & ~v[None]).any() and not (v & nd).any()
        assert (out["gray"][nd] == 0).all()
        # Bright ignored paint never becomes verified background.
        assert not (v & (out["gray"] >= 150) & ~m.any(0)).sum() > 0.01 * max(v.sum(), 1)
        assert out["masks"].sum() <= masks.sum()


def test_tone_curve_is_monotone_and_hits_targets():
    gray, masks, _labels, valid = scene()
    out, kx, ky = tone_curve(gray, masks.any(0), valid, TRIPLES[0], 1.0)
    assert all(b > a for a, b in pairwise(kx))
    assert all(b > a for a, b in pairwise(ky))
    assert abs(int(np.median(out[masks.any(0)])) - (75 + 163)) <= 1
    assert out.min() >= 1


def test_scaling_makes_strokes_thinner_and_marks_canvas_nodata():
    cfg = AugmentConfig(
        tone_probability=0,
        scale_probability=1,
        scale_range=(0.4, 0.4),
        cut_probability=0,
        dropout_probability=0,
    )
    gray, masks, labels, valid = scene()
    out = augment(gray, masks, labels, valid, np.random.default_rng(0), TRIPLES, cfg)
    assert out["nodata"].mean() > 0.8 and out["masks"][0].sum() < masks[0].sum() * 0.25
