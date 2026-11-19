"""Clean closed polygons for painted area markings (client rule: a 3D polygon drawn on the limits
of the painted area, i.e. straight edges and sharp corners, not a pixel outline).

* ``ShapeLibrary``: the client's own polygons of fixed-shape marks (selection and detour arrows)
  normalised to their principal axis. ``fit_template`` places every library shape on a detected
  paint blob (rotation, uniform scale, shift) and keeps the best overlap: the output is the
  client's clean shape at the measured position, and its code identifies the arrow type.
* ``fit_straight``: for free shapes (hatch bars, chevrons, triangles, colour bands) the fewest
  straight edges that stay within tolerance of the paint outline; each edge is then a least-
  squares line through its outline points and the corners are the edge intersections.
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import Polygon

TEMPLATE_CODES = (
    "M15",
    "M15a",
    "M15b",
    "M15c",
    "M15d",
    "M15e",
    "M15f",
    "M16",
    "M16a",
    "M16b",
)


def principal_frame(poly):
    """(centroid, unit long axis, length along it, width across it) of a polygon."""
    xy = np.asarray(poly.exterior.coords)[:-1]
    dense = shapely.get_coordinates(shapely.segmentize(poly.exterior, 0.05))
    c = np.asarray(poly.centroid.coords[0])
    cov = np.cov((dense - c).T)
    _, v = np.linalg.eigh(cov)
    axis = v[:, -1]
    proj = (xy - c) @ axis
    side = (xy - c) @ np.array([-axis[1], axis[0]])
    return c, axis, float(np.ptp(proj)), float(np.ptp(side))


def normalise(poly):
    """Polygon moved to its centroid and rotated so its long axis is the x axis."""
    c, axis, _, _ = principal_frame(poly)
    angle = np.degrees(np.arctan2(axis[1], axis[0]))
    return affinity.rotate(affinity.translate(poly, -c[0], -c[1]), -angle, origin=(0, 0))


def iou(a, b):
    try:
        inter = a.intersection(b).area
        return inter / max(a.union(b).area, 1e-9)
    except shapely.errors.GEOSException:  # rare topology failure on raster outlines
        a, b = shapely.make_valid(a).buffer(0), shapely.make_valid(b).buffer(0)
        inter = a.intersection(b).area
        return inter / max(a.union(b).area, 1e-9)


class ShapeLibrary:
    """Normalised client shapes per code (fixed-shape marks only)."""

    def __init__(self, examples, per_code=12, seed=0):
        rng = np.random.default_rng(seed)
        groups = {}
        for poly, code in examples:
            if code in TEMPLATE_CODES and poly.is_valid and poly.area > 0.05:
                groups.setdefault(code, []).append(poly)
        self.shapes = []
        for code, polys in sorted(groups.items()):
            pick = rng.choice(len(polys), size=min(per_code, len(polys)), replace=False)
            for k in pick:
                n = normalise(polys[k])
                self.shapes.append((code, n, float(np.ptp(np.asarray(n.exterior.coords)[:, 0]))))

    def __len__(self):
        return len(self.shapes)


def fit_template(blob, library, min_iou=0.7, scale_range=(0.85, 1.15)):
    """Best library shape on ``blob``. Returns (polygon, code, iou) or (None, None, best iou).

    Each shape is aligned with the blob's long axis (both directions), scaled to the blob
    length, then rotation (+-4 degrees), scale and the shift along/across the axis are refined
    by a small search that maximises the overlap with the paint.
    """
    if not len(library):
        return None, None, 0.0
    c, axis, length, _ = principal_frame(blob)
    base = np.degrees(np.arctan2(axis[1], axis[0]))
    best = (0.0, None, None)

    def make_place(shape, flip):
        def place(rot, scale, du, dv):
            g = affinity.scale(shape, scale, scale, origin=(0, 0))
            g = affinity.rotate(g, base + flip + rot, origin=(0, 0))
            off = c + du * axis + dv * np.array([-axis[1], axis[0]])
            return affinity.translate(g, off[0], off[1])

        return place

    starts = []
    for code, shape, t_len in library.shapes:
        s0 = length / max(t_len, 1e-6)
        if not scale_range[0] * 0.8 <= s0 <= scale_range[1] * 1.6:
            continue
        s0 = float(np.clip(s0, *scale_range))
        for flip in (0.0, 180.0):
            place = make_place(shape, flip)
            starts.append((iou(place(0.0, s0, 0.0, 0.0), blob), code, place, s0))
    starts.sort(key=lambda item: -item[0])
    for score, code, place, s0 in starts[:3]:  # refine only the most promising placements
        params = [0.0, s0, 0.0, 0.0]
        for step in ((2.0, 0.05, 0.2, 0.06), (1.0, 0.02, 0.08, 0.03), (0.5, 0.01, 0.03, 0.015)):
            improved = True
            while improved:
                improved = False
                for k, delta in enumerate(step):
                    for sign in (-1, 1):
                        trial = list(params)
                        trial[k] += sign * delta
                        if not scale_range[0] <= trial[1] <= scale_range[1] or abs(trial[0]) > 6:
                            continue
                        sc = iou(place(*trial), blob)
                        if sc > score + 1e-4:
                            params, score, improved = trial, sc, True
        if score > best[0]:
            best = (score, place(*params), code)
    if best[0] >= min_iou:
        return best[1], best[2], best[0]
    return None, None, best[0]


def _line_fit(points):
    c = points.mean(0)
    _, _, vt = np.linalg.svd(points - c)
    return c, vt[0]


def _intersect(p1, d1, p2, d2):
    a = np.column_stack([d1, -d2])
    if abs(np.linalg.det(a)) < 1e-6:
        return None
    t = np.linalg.solve(a, p2 - p1)
    return p1 + t[0] * d1


def _reduce(verts):
    """Visvalingam: polygons with n, n-1, ..., 3 corners, each time dropping the corner whose
    triangle with its neighbours is smallest."""
    verts = list(map(tuple, verts))
    out = {len(verts): np.asarray(verts)}
    while len(verts) > 3:
        areas = []
        for k in range(len(verts)):
            p, q, r = (
                np.asarray(verts[k - 1]),
                np.asarray(verts[k]),
                np.asarray(verts[(k + 1) % len(verts)]),
            )
            u, w = q - p, r - p
            areas.append(abs(u[0] * w[1] - u[1] * w[0]) / 2)
        del verts[int(np.argmin(areas))]
        out[len(verts)] = np.asarray(verts)
    return out


def edge_deviation(poly, blob, step=0.025, q=95):
    """How far the outlines disagree (metres): the ``q``-th percentile of the distances from
    each outline to the other, taken in both directions (a corner overshooting the paint and a
    paint corner cut off both count)."""
    a = shapely.points(shapely.get_coordinates(shapely.segmentize(poly.exterior, step)))
    b = shapely.points(shapely.get_coordinates(shapely.segmentize(blob.exterior, step)))
    return float(
        max(
            np.percentile(shapely.distance(a, blob.exterior), q),
            np.percentile(shapely.distance(b, poly.exterior), q),
        )
    )


def fit_straight(blob, max_vertices=12, tolerance=0.10, min_hole_m2=0.05):
    """Fewest straight edges for a free shape: the smallest corner count (3, 4, ...) whose
    polygon, after least-squares edge refinement, follows the paint outline within
    ``tolerance`` (``edge_deviation``). (A minimum rotated rectangle is not used: it encloses
    every outline pixel and so sits outside the paint edge.) If no count reaches it, the
    closest fit wins (a small penalty per corner keeps the shape simple). Returns (polygon,
    corners, iou)."""
    blob = blob.buffer(0)
    if blob.geom_type == "MultiPolygon":
        blob = max(blob.geoms, key=lambda g: g.area)
    holes = [Polygon(r) for r in blob.interiors if Polygon(r).area >= min_hole_m2]
    if holes:
        # A frame (ring) is one polygon with a hole: outer edge and each hole are fitted with
        # straight edges separately (client spec: "the polygon might have holes").
        outer, _, _ = fit_straight(Polygon(blob.exterior), max_vertices, tolerance)
        inner = []
        for h in holes:
            fitted, _, _ = fit_straight(h, max_vertices, tolerance)
            if outer.contains(fitted):
                inner.append(np.asarray(fitted.exterior.coords))
        poly = Polygon(np.asarray(outer.exterior.coords), inner)
        if not poly.is_valid:
            poly = outer.difference(shapely.union_all([Polygon(r) for r in inner]))
            if poly.geom_type != "Polygon":
                poly = max(poly.geoms, key=lambda g: g.area)
        return poly, len(poly.exterior.coords) - 1, iou(poly, blob)
    blob = Polygon(blob.exterior)
    outline = shapely.get_coordinates(shapely.segmentize(blob.exterior, 0.025))[:-1]
    simple = blob.simplify(0.02, preserve_topology=True)
    candidates = _reduce(np.asarray(simple.exterior.coords)[:-1])
    best = None
    for k in sorted(candidates):
        if k > max_vertices:
            break
        options = []
        refined = _refine(candidates[k], outline)
        if refined is not None and refined.is_valid and refined.area > 0:
            options.append(refined)
        raw = Polygon(candidates[k])
        if raw.is_valid and raw.area > 0:
            options.append(raw)
        for poly in options:
            dev = edge_deviation(poly, blob)
            cost = dev + 0.01 * k
            if best is None or cost < best[1]:
                best = (poly, cost, dev)
        if best is not None and best[2] <= tolerance:
            break
    poly = best[0]
    return poly, len(poly.exterior.coords) - 1, iou(poly, blob)


def _refine(verts, outline):
    """Straight least-squares edges between consecutive corners; corners = edge intersections."""
    n = len(verts)
    tree_idx = [int(np.argmin(np.linalg.norm(outline - v, axis=1))) for v in verts]
    order = np.argsort(tree_idx)
    verts, tree_idx = verts[order], [tree_idx[k] for k in order]
    lines = []
    for k in range(n):
        a, b = tree_idx[k], tree_idx[(k + 1) % n]
        seg = outline[a : b + 1] if b > a else np.vstack([outline[a:], outline[: b + 1]])
        if len(seg) >= 4:  # drop the points next to the corners: the corners are rounded
            cut = max(1, len(seg) // 8)
            seg = seg[cut:-cut] if len(seg) > 2 * cut + 2 else seg
        if len(seg) < 2:
            seg = np.vstack([verts[k], verts[(k + 1) % n]])
        lines.append(_line_fit(seg))
    out = []
    for k in range(n):
        p1, d1 = lines[k - 1]
        p2, d2 = lines[k]
        q = _intersect(p1, d1, p2, d2)
        # nearly parallel neighbours or a far intersection: keep the simplified corner
        if q is None or np.linalg.norm(q - verts[k]) > 0.15:
            q = verts[k]
        out.append(q)
    poly = Polygon(out)
    return (
        poly if poly.is_valid else poly.buffer(0) if poly.buffer(0).geom_type == "Polygon" else None
    )


__all__ = [
    "TEMPLATE_CODES",
    "ShapeLibrary",
    "edge_deviation",
    "fit_straight",
    "fit_template",
    "iou",
    "normalise",
    "principal_frame",
]
