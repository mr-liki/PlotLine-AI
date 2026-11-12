import hashlib

import pytest
from shapely.geometry import LineString, MultiLineString

from roadmarking.dataset.geometry import painted_area_from_centerline


def test_painted_area_uses_recorded_width_with_flat_caps():
    line = LineString([(0, 0), (4.3, 0)])
    paint = painted_area_from_centerline(line, 0.15)
    assert paint.area == pytest.approx(4.3 * 0.15, rel=1e-9)
    minx, miny, maxx, maxy = paint.bounds
    assert (minx, maxx) == pytest.approx((0.0, 4.3))
    assert (maxy - miny) == pytest.approx(0.15)


def test_separate_dash_parts_are_never_bridged():
    dashes = MultiLineString([[(0, 0), (4, 0)], [(10, 0), (14, 0)]])
    paint = painted_area_from_centerline(dashes, 0.15)
    assert len(getattr(paint, "geoms", [paint])) == 2
    assert paint.area == pytest.approx(2 * 4 * 0.15)


def test_invalid_width_is_rejected():
    with pytest.raises(ValueError):
        painted_area_from_centerline(LineString([(0, 0), (1, 0)]), 0)


def test_stable_source_id_scheme_is_deterministic():
    identity = "abc|Highway|fid=12"
    first = hashlib.sha256(identity.encode()).hexdigest()
    assert first == hashlib.sha256(identity.encode()).hexdigest()
    assert first != hashlib.sha256(b"abc|Highway|fid=13").hexdigest()
