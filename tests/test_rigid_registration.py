"""Synthetic geometry tests do not establish real survey correspondences."""

import numpy as np
import pytest
from shapely.affinity import affine_transform
from shapely.geometry import LineString, Polygon

from roadmarking.dataset.rigid_registration import (
    apply_xy,
    distance_stats,
    fit_rigid_multistart,
    grid_sample,
    rigid_solution,
    transform_geometry_xy,
)


def test_rigid_correspondences_recover_large_offset_without_scale_or_reflection():
    rng = np.random.default_rng(1701)
    source = rng.normal(size=(100, 2)) * [500, 100] + [10000, 170000]
    theta = np.deg2rad(23)
    expected = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    target = apply_xy(source, expected, [-30000, -260000])
    rotation, translation = rigid_solution(source, target)
    np.testing.assert_allclose(apply_xy(source, rotation, translation), target, atol=1e-8, rtol=0)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(2), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1)


def test_multistart_recovers_irregular_known_shape_and_deterministic_ids():
    rng = np.random.default_rng(33)
    source = rng.normal(size=(80, 2)) * [1000, 100]
    theta = 0.4
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    target = apply_xy(source, rotation, [5000, -7000])
    a = fit_rigid_multistart(source, target, iterations=50)
    b = fit_rigid_multistart(source, target, iterations=50)
    np.testing.assert_allclose(a["rotation"], b["rotation"], atol=0)
    assert distance_stats(apply_xy(source, a["rotation"], a["translation"]), target)["p95"] < 1e-6


def test_shape_width_holes_and_original_z_are_preserved():
    line = LineString([(0, 0, 10), (1, 0, 15), (2, 1, 20)])
    polygon = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)], holes=[[(1, 1), (2, 1), (2, 2), (1, 2)]])
    theta = 0.3
    parameters = [np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta), 10000, -250000]
    moved = affine_transform(line, parameters)
    assert [c[2] for c in moved.coords] == [10, 15, 20]
    assert moved.length == pytest.approx(line.length, abs=1e-9)
    area = affine_transform(polygon, parameters)
    assert len(area.interiors) == 1 and area.area == pytest.approx(polygon.area, abs=1e-8)


def test_duplicate_sample_cells_and_invalid_correspondences():
    source = np.array([[0, 0], [0.1, 0.1], [5, 5], [5.1, 5.1]])
    assert grid_sample(source, 1).shape == (2, 2)
    with pytest.raises(ValueError):
        rigid_solution(source[:2], source[:2])


def test_unknown_z_cannot_contaminate_xy_during_registration():
    source = LineString([(0, 0, float("nan")), (1, 0, 10), (2, 1, float("nan"))])
    theta = 0.4
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    target = transform_geometry_xy(source, rotation, [10000, -250000])
    before, after = np.asarray(source.coords), np.asarray(target.coords)
    assert np.isfinite(after[:, :2]).all()
    np.testing.assert_allclose(after[:, :2], apply_xy(before[:, :2], rotation, [10000, -250000]))
    np.testing.assert_allclose(after[:, 2], before[:, 2], equal_nan=True)
