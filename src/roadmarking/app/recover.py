"""Recover lane lines the per-block tracing misses, from the paint itself.

``dash_beside_line``: a short dash painted next to a continuous line (a few decimetres apart)
merges with the line's paint and is fitted as an area mark. The paint that is left after taking
away the traced line is a thin strip: it is exported as a dash, not as a polygon. It keeps only
paint evidence and passes the same paint tests as every other line before export.
"""

from __future__ import annotations

import numpy as np


def _feature(geom, class_id, width_m, flags, layer=0):
    return dict(
        class_id=class_id,
        class_name="solid" if class_id == 0 else "dash",
        geometry_2d=geom,
        length_m=geom.length,
        width_m=width_m,
        flags=list(flags),
        layer=layer,
        mark_id=-1,
    )


def dash_beside_line(poly, lines, max_gap_m=0.6, line_half_m=0.15, max_angle_deg=10.0):
    """When ``poly`` is a dash merged with the paint of a parallel traced line, return the dash
    as a straight 2D line (its axis); otherwise None."""
    import shapely
    from shapely.geometry import LineString

    from roadmarking.digitization.polygon_fit import principal_frame

    near = [ln for ln in lines if ln.distance(poly) <= max_gap_m and ln.length >= 10.0]
    if not near:
        return None
    _, axis, length, _ = principal_frame(poly)
    if not 1.0 <= length <= 9.0:
        return None
    rest = poly
    for ln in near:
        d = ln.project(poly.centroid)
        a = np.asarray(ln.interpolate(max(d - 1, 0)).coords[0])[:2]
        b = np.asarray(ln.interpolate(min(d + 1, ln.length)).coords[0])[:2]
        t = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        if np.degrees(np.arccos(min(1.0, abs(float(axis @ t))))) > max_angle_deg:
            return None
        rest = rest.difference(ln.buffer(line_half_m, cap_style="flat"))
    parts = [g for g in getattr(rest, "geoms", [rest]) if g.geom_type == "Polygon"]
    if not parts:
        return None
    strip = max(parts, key=lambda g: g.area)
    c, ax, length_s, width_s = principal_frame(strip)
    if not (0.08 <= width_s <= 0.45 and length_s >= max(1.0, 4 * width_s) and strip.area >= 0.1):
        return None
    # the strip must lie beside the line, not across it
    if min(ln.distance(shapely.Point(c)) for ln in near) < line_half_m:
        return None
    c = np.asarray(c)
    return LineString([c - ax * length_s / 2, c + ax * length_s / 2]), float(width_s)
