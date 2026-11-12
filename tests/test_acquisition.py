"""Tests for acquisition-geometry estimates on a synthetic diagonal corridor."""

from __future__ import annotations

import numpy as np
import pytest

from roadmarking.analysis.acquisition import across_track_density, corridor_geometry
from roadmarking.analysis.lidar_stats import GridAccumulator, GridSpec


def _diagonal_corridor() -> GridAccumulator:
    """1000 m corridor at 45°, 100 m wide; density 400/m² within ±5 m, 10/m² elsewhere."""
    rng = np.random.default_rng(3)
    along = rng.uniform(0, 1000, 600_000)
    across = rng.uniform(-50, 50, 600_000)
    dense_a = rng.uniform(0, 1000, 3_960_000)
    dense_c = rng.uniform(-5, 5, 3_960_000)
    a = np.concatenate([along, dense_a])
    c = np.concatenate([across, dense_c])
    u = np.array([np.sqrt(0.5), np.sqrt(0.5)])
    v = np.array([-np.sqrt(0.5), np.sqrt(0.5)])
    x = 2000 + a * u[0] + c * v[0]
    y = 5000 + a * u[1] + c * v[1]
    spec = GridSpec.covering(np.array([x.min(), y.min(), 0]), np.array([x.max(), y.max(), 0]), 1.0)
    acc = GridAccumulator(spec)
    n = x.size
    acc.add(x, y, np.zeros(n), np.ones(n), np.ones(n))
    return acc


def test_corridor_geometry_and_speed() -> None:
    acc = _diagonal_corridor()
    geo = corridor_geometry(acc, (100.0, 150.0), valid_fraction=1.0)
    assert geo["corridor_length_m"] == pytest.approx(1000, rel=0.03)
    assert geo["swath_width_m_p5_p50_p95"][1] == pytest.approx(100, rel=0.06)
    assert geo["platform_speed_m_s"] == pytest.approx(20, rel=0.03)
    assert geo["point_rate_pts_s"] == pytest.approx(int(acc.count.sum()) / 50, rel=1e-6)


def test_across_track_density_peak() -> None:
    prof = across_track_density(_diagonal_corridor())
    centre = [p for p in prof if p["offset_m"] == [-10.0, 0.0] or p["offset_m"] == [0.0, 10.0]]
    edge = [p for p in prof if p["offset_m"][0] <= -30.0]
    assert centre and edge
    assert min(p["median_pts_per_m2"] for p in centre) > 100  # dense band near axis
    assert max(p["median_pts_per_m2"] for p in edge) < 20  # sparse swath edges
