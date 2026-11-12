import math

import numpy as np
from shapely.geometry import LineString, MultiLineString, box

from roadmarking.dataset.coverage import (
    cluster_sections,
    paint_contrast,
    radius_from_rate,
    solid_shape,
    turning_degrees,
)


def _arc(radius, degrees, n=400):
    t = np.radians(np.linspace(0, degrees, n))
    return LineString(np.column_stack((radius * np.cos(t), radius * np.sin(t))))


def test_turning_matches_arc_and_ignores_straight_jitter():
    arc = _arc(200.0, 30.0)
    assert math.isclose(turning_degrees(arc, step=1.0), 30.0, rel_tol=0.03)
    rng = np.random.default_rng(0)
    x = np.linspace(0, 50, 500)
    jitter = LineString(np.column_stack((x, rng.normal(0, 0.02, x.size))))
    assert solid_shape(jitter)[0] == "straight"


def test_solid_shape_curved_short_and_multipart():
    shape, _length, rate = solid_shape(_arc(150.0, 20.0))
    assert shape == "curved" and rate > 5
    assert math.isclose(radius_from_rate(rate), 150.0, rel_tol=0.05)
    assert solid_shape(LineString([(0, 0), (5, 0)]))[0] == "short"
    parts = MultiLineString([[(0, 0), (30, 0)], [(40, 0), (70, 0)]])
    assert solid_shape(parts)[0] == "straight"


def test_sections_cluster_by_guard_distance():
    boxes = [box(0, 0, 10, 10), box(15, 0, 25, 10), box(1000, 0, 1010, 10)]
    assert cluster_sections(boxes, 10) == [0, 0, 1]
    assert cluster_sections(boxes, 2) == [0, 1, 2]


def test_paint_contrast_uses_observed_ring_and_excludes_neighbours():
    gray = np.full((60, 60), 50.0)
    mask = np.zeros((60, 60), bool)
    mask[28:32, 5:55] = True
    gray[mask] = 200
    neighbour = np.zeros_like(mask)
    neighbour[34:36, 5:55] = True
    gray[neighbour] = 220
    observed = np.ones_like(mask)
    assert paint_contrast(gray, mask, observed, exclude=neighbour) == 150
    observed[:] = False
    assert paint_contrast(gray, mask, observed) is None
