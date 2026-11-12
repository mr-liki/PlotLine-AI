"""Tests for the road-surface assessment heuristics on a synthetic 1 m grid."""

from __future__ import annotations

import numpy as np
import pytest

from roadmarking.analysis.lidar_stats import GridAccumulator, GridSpec
from roadmarking.analysis.surface import SurfaceCriteria, candidate_surface_mask, mask_fractions


def _scene() -> GridAccumulator:
    """60 x 60 m: flat ground at z=10, a 5 m 'building', a 'tree' patch, a sloped bank and
    an 8 m wide road on a 1.5 m embankment (rows 25-32)."""
    acc = GridAccumulator(GridSpec(0.0, 60.0, 1.0, 60, 60))
    rng = np.random.default_rng(0)
    xs, ys, zs, singles = [], [], [], []
    for r in range(60):
        for c in range(60):
            x = c + rng.random(16)
            y = 60 - r - rng.random(16)
            z = np.full(16, 10.0) + rng.normal(0, 0.01, 16)
            single = np.ones(16)
            if 10 <= r < 20 and 10 <= c < 20:  # flat roof 5 m up
                z += 5.0
            if 40 <= r < 50 and 40 <= c < 50:  # vegetation: multi-return, rough
                z += rng.uniform(0, 3, 16)
                single[:10] = 0
            if c >= 55:  # steep bank: 0.5 m rise per metre
                z += 0.5 * (c - 54)
            elif 25 <= r < 33:  # road on an embankment, 1.5 m above the surroundings
                z += 1.5
            xs.append(x)
            ys.append(y)
            zs.append(z)
            singles.append(single)
    acc.add(
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(zs),
        np.ones(60 * 60 * 16),
        np.concatenate(singles),
    )
    return acc


def test_candidate_mask_excludes_roof_vegetation_and_slope() -> None:
    m = candidate_surface_mask(_scene())
    cand = m["candidate"]
    assert cand[5, 30] and cand[45, 5]  # open flat ground
    assert cand[26:32, 2:53].all()  # road on embankment (interior) is kept
    assert not cand[10:20, 10:20].any()  # roof: isolated by its walls, < 200 m²
    assert m["flat_cell"][15, 15] and not m["large_component"][15, 15]
    assert not cand[40:50, 40:50].any()  # vegetation
    assert not cand[:, 56:].any()  # steep bank
    assert not m["single_return"][45, 45]


def test_block_median_background() -> None:
    from roadmarking.analysis.report import _block_median

    img = np.full((6, 6), 10.0)
    img[0, 0] = 1000.0  # one bright "paint" pixel does not move the median
    img[3:, 3:] = 20.0
    img[5, 5] = np.nan  # no-data ignored
    bg = _block_median(img, 3)
    assert bg.shape == img.shape
    assert bg[0, 0] == 10.0 and bg[4, 4] == 20.0
    all_nan = _block_median(np.full((3, 3), np.nan), 3)
    assert np.isnan(all_nan).all()


def test_mask_fractions_and_grid_validation() -> None:
    acc = _scene()
    m = candidate_surface_mask(acc)
    f = mask_fractions(m, acc.occupied)
    assert f["enough_points"] == 100.0
    assert 50.0 < f["candidate"] < 100.0
    coarse = GridAccumulator(GridSpec(0.0, 10.0, 2.0, 5, 5))
    with pytest.raises(ValueError):
        candidate_surface_mask(coarse, SurfaceCriteria())
