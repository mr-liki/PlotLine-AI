"""Join converging child lines to their parent line at junctions (gore noses, merges, splits).

Client rule: linear elements that physically connect share a vertex (snap, XYZ). A solid
line whose end converges onto a longer solid line (the sideways gap shrinks towards the end
and is small at the end) is extended along a tangent-continuous curve to the point where it
meets the parent; that point is inserted as a vertex of the parent so both share it.
Joins longer than ``free_join_m`` must run on visible paint (the converging stripes of a gore
nose are painted to the tip); shorter ones are accepted as gaps of the scan or of the trace.
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import LineString, Point

from .gap_bridging import hermite

RULES = dict(
    max_end_offset_m=0.6,  # child end at most this far from the parent
    min_convergence_m=0.08,  # sideways gap 3 m back minus gap at the end
    max_angle_deg=25.0,
    max_join_m=25.0,
    free_join_m=3.0,
    min_paint_on_join=0.6,
    back_m=3.0,
)


def _tangent(line, d, span=1.0):
    a = np.asarray(line.interpolate(max(d - span, 0)).coords[0])[:2]
    b = np.asarray(line.interpolate(min(d + span, line.length)).coords[0])[:2]
    t = b - a
    return t / max(np.linalg.norm(t), 1e-9)


def _end(line, at_start, back):
    c = np.asarray(line.coords)[:, :2]
    p = c[0] if at_start else c[-1]
    q = np.asarray(
        line.interpolate(min(back, line.length) if at_start else max(line.length - back, 0)).coords[
            0
        ]
    )[:2]
    t = p - q
    return p, q, t / max(np.linalg.norm(t), 1e-9)


def find_joins(lines, paint_at=None, rules=None, children=None, min_parent_m=10.0):
    """Join plans for converging ends. ``lines``: list of LineString (solid lines);
    ``paint_at(xy[N,2]) -> bool[N]``. Returns list of dict(child, at_start, parent, curve,
    parent_station, length, on_paint)."""
    r = dict(RULES, **(rules or {}))
    tree = shapely.STRtree(lines)
    joins = []
    for i, g in enumerate(lines):
        if g.length < 2.0 or (children is not None and i not in children):
            continue
        for at_start in (True, False):
            p, q, t = _end(g, at_start, r["back_m"])
            best = None
            for j in tree.query(Point(p).buffer(r["max_end_offset_m"])):
                if j == i or lines[j].length < max(g.length * 0.5, min_parent_m):
                    continue
                h = lines[j]
                d = h.project(Point(p))
                if d <= 0.05 or d >= h.length - 0.05:
                    continue  # at the parent's own end: a continuation, not a junction
                foot = np.asarray(h.interpolate(d).coords[0])[:2]
                gap_end = float(np.linalg.norm(foot - p))
                if gap_end < 0.02 or gap_end > r["max_end_offset_m"]:
                    continue
                gap_back = float(h.distance(Point(q)))
                if gap_back - gap_end < r["min_convergence_m"]:
                    continue  # parallel neighbour, not converging
                u = _tangent(h, d)
                if u @ t < 0:
                    u = -u  # parent direction along the child's travel
                ang = np.degrees(np.arccos(np.clip(u @ t, -1, 1)))
                if ang > r["max_angle_deg"]:
                    continue
                # meeting point: where the child's approach closes the sideways gap
                closing = np.tan(np.radians(max(ang, 0.5)))
                ahead = min(gap_end / closing, r["max_join_m"])
                sign = 1.0 if u @ _tangent(h, d) > 0 else -1.0
                ds = float(np.clip(d + sign * ahead, 0.0, h.length))
                target = np.asarray(h.interpolate(ds).coords[0])[:2]
                length = float(np.linalg.norm(target - p))
                if length < 0.02:
                    continue
                curve = hermite(p, t, target, u, step=0.1)
                on_paint = 1.0
                if paint_at is not None and len(curve) > 2:
                    on_paint = float(np.mean(paint_at(curve[1:-1])))
                if length > r["free_join_m"] and on_paint < r["min_paint_on_join"]:
                    continue
                if best is None or gap_end < best["gap_end"]:
                    best = dict(
                        child=i,
                        at_start=at_start,
                        parent=j,
                        curve=curve,
                        parent_station=ds,
                        length=length,
                        on_paint=on_paint,
                        gap_end=gap_end,
                    )
            if best is not None:
                joins.append(best)
    return joins


def insert_vertex(line, station, tol=0.01):
    """``line`` with a vertex at ``station`` (the shared junction vertex)."""
    c = np.asarray(line.coords)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c[:, :2], axis=0), axis=1))]
    k = int(np.searchsorted(s, station))
    if (k < len(s) and abs(s[k] - station) <= tol) or (k > 0 and abs(s[k - 1] - station) <= tol):
        return line
    p = np.asarray(line.interpolate(station).coords[0])
    return LineString(np.vstack([c[:k], p[None, : c.shape[1]], c[k:]]))


def apply_joins(features, paint_at=None, rules=None):
    """Join converging solid child lines to their parents in ``features`` (dicts with
    ``geometry_2d`` and ``class_id``). Returns the number of joins; flags each joined child
    ``joined_at_junction`` and records the join as an inferred segment when it is not on paint."""
    # children: solid lines; parents: any long line (a long line can be labelled dash where
    # the paint of two converging stripes looked broken)
    solid = list(range(len(features)))
    lines = [f["geometry_2d"] for f in features]
    children = {k for k, f in enumerate(features) if f["class_id"] == 0}
    joins = find_joins(lines, paint_at, rules, children=children)
    stations = {}
    for jn in joins:
        f = features[solid[jn["child"]]]
        c = np.asarray(f["geometry_2d"].coords)[:, :2]
        curve = jn["curve"]
        c = np.vstack([curve[::-1][:-1], c]) if jn["at_start"] else np.vstack([c, curve[1:]])
        f["geometry_2d"] = LineString(c)
        f["length_m"] = f["geometry_2d"].length
        f["flags"] = sorted(set(f["flags"]) | {"joined_at_junction"})
        if jn["on_paint"] < 0.6 and jn["length"] > 0.3:
            seg = LineString(curve)
            f.setdefault("inferred_segments", []).append(seg)
            f.setdefault("inferred_labels", []).append("INFERRED_JUNCTION_JOIN")
            f["inferred_length_m"] = f.get("inferred_length_m", 0.0) + seg.length
        stations.setdefault(jn["parent"], []).append(curve[-1])
    for j, targets in stations.items():
        f = features[solid[j]]
        g = f["geometry_2d"]
        # project now: the parent may itself have been extended at a junction meanwhile
        for s in sorted((g.project(Point(t)) for t in targets), reverse=True):
            g = insert_vertex(g, s)
        f["geometry_2d"] = g
    return len(joins)


__all__ = ["RULES", "apply_joins", "find_joins", "insert_vertex"]
