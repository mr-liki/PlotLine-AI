"""Centerlines from observed paint fragments (pixel space, sub-pixel output).

1. connectivity mask (paint + bridged sampling holes) -> fill small holes -> skeleton;
2. 8-connected skeleton graph; main path = longest path between endpoints (graph diameter);
3. side branches shorter than ``spur_px`` are spurs (pruned); longer ones flag ``branching``
   (possible crossing/arrow confusion), never joined into the line;
4. sub-pixel refinement: each path point moves to the paint centroid along the local normal;
5. ends extended to the paint extent along the tangent (skeletons stop half a width short);
6. light smoothing + Douglas-Peucker simplification with pixel-scale tolerances.
Only pixel-unit parameters are used; metres enter only via the verified cell size.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
from shapely.geometry import LineString
from skimage.morphology import remove_small_holes, skeletonize

NEIGH = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def skeleton_graph(skel):
    g = nx.Graph()
    ys, xs = np.nonzero(skel)
    pts = set(zip(ys.tolist(), xs.tolist(), strict=True))
    for y, x in pts:
        g.add_node((y, x))
        for dy, dx in NEIGH:
            q = (y + dy, x + dx)
            if q in pts:
                g.add_edge((y, x), q, weight=float(np.hypot(dy, dx)))
    return g


def main_path(g):
    """Longest shortest path over the largest component; returns (path, longest_branch_px)."""
    if g.number_of_nodes() == 0:
        return [], 0.0
    comp = max(nx.connected_components(g), key=len)
    h = g.subgraph(comp)
    start = next(iter(comp))
    d1 = nx.single_source_dijkstra_path_length(h, start)
    a = max(d1, key=d1.get)
    d2, paths = nx.single_source_dijkstra(h, a)
    b = max(d2, key=d2.get)
    path = paths[b]
    on = set(path)
    longest = 0.0
    if len(on) < len(comp):
        dist = nx.multi_source_dijkstra_path_length(h, on)
        longest = max(dist.values())
    return path, float(longest)


def _normals(p):
    t = np.gradient(p, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    return t, np.column_stack([-t[:, 1], t[:, 0]])


def refine(path, field, halfwidth, step=0.25):
    p = np.asarray(path, float)
    if len(p) < 3:
        return p
    k = max(1, min(3, len(p) // 4))
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    sm = np.column_stack(
        [np.convolve(np.pad(p[:, i], k, mode="edge"), ker, "valid") for i in (0, 1)]
    )
    _, nrm = _normals(sm)
    offs = np.arange(-halfwidth - 1, halfwidth + 1 + 1e-9, step)
    out = []
    for q, nv in zip(sm, nrm, strict=True):
        sy = q[0] + offs * nv[0]
        sx = q[1] + offs * nv[1]
        w = map_coordinates(field, [sy, sx], order=1, mode="constant")
        out.append(q + nv * (float((w * offs).sum() / w.sum())) if w.sum() > 1e-6 else q)
    return np.asarray(out)


def extend_ends(p, field, max_px=6.0, step=0.25):
    if len(p) < 2:
        return p
    out = p.copy()
    for end, nb in ((0, 1), (-1, -2)):
        t = out[end] - out[nb]
        n = np.linalg.norm(t)
        if n < 1e-9:
            continue
        t /= n
        s = 0.0
        while s + step <= max_px:
            q = out[end] + t * (s + step)
            if map_coordinates(field, [[q[0]], [q[1]]], order=1, mode="constant")[0] < 0.5:
                break
            s += step
        out[end] = out[end] + t * s
    return out


def smooth_polyline(p, window_px=30.0, step_px=2.0):
    """Local quadratic regression (tricube weights) of x(s), y(s) along arc length ``s``,
    evaluated on a uniform ``step_px`` grid. Removes raster/scan-stripe jitter while following
    curves whose radius is much larger than the window. Short paths get a straight fit."""
    p = np.asarray(p, float)
    if len(p) < 3:
        return p
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    total = s[-1]
    if total < 1e-6:
        return p[[0, -1]]
    targets = np.linspace(0, total, max(2, int(np.ceil(total / step_px)) + 1))
    out = []
    for t in targets:
        h = max(window_px, 1e-6)
        if not (h <= t <= total - h):
            h *= 2  # one-sided support near the ends: widen the window for a stable fit
        u = (s - t) / h
        w = np.where(np.abs(u) < 1, (1 - np.abs(u) ** 3) ** 3, 0.0)
        if (w > 0).sum() < 4:
            w = np.ones_like(s)
        deg = 2 if total > 2 * window_px and (w > 0).sum() >= 6 else 1
        a = np.vander(s - t, deg + 1)
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(a * sw[:, None], p * sw[:, None], rcond=None)
        out.append(coef[-1])
    return np.asarray(out)


def fragment_centerline(
    connect_mask, paint_mask, spur_factor=3.0, simplify_px=0.2, smooth_window_px=30.0
):
    """Return dict(raw=[(r,c)...], refined=[(r,c)...], width_px, flags) in fragment-local pixels."""
    pad = 3
    m = np.pad(connect_mask, pad)
    filled = remove_small_holes(m, max_size=16)
    dt = distance_transform_edt(filled)
    skel = skeletonize(filled)
    g = skeleton_graph(skel)
    path, branch = main_path(g)
    flags = []
    if len(path) < 2:
        return None
    width = float(2 * np.median(dt[tuple(np.array(path).T)]))
    spur_px = max(spur_factor * width / 2, 4.0)
    if branch > spur_px:
        flags.append("branching")
    if g.number_of_nodes() and len(max(nx.connected_components(g), key=len)) < g.number_of_nodes():
        flags.append("multi_component_skeleton")
    field = filled.astype(float)
    centroid = refine(path, field, halfwidth=max(width / 2, 1.0))
    smooth = smooth_polyline(centroid, window_px=smooth_window_px)
    smooth = extend_ends(smooth, field, max_px=max(width, 2.0))
    line = LineString(smooth[:, ::-1]).simplify(simplify_px, preserve_topology=False)
    pts = np.asarray(line.coords)[:, ::-1] - pad
    raw = np.asarray(path, float) - pad
    return dict(
        raw=raw, centroid=centroid - pad, refined=pts, width_px=width, branch_px=branch, flags=flags
    )


__all__ = [
    "extend_ends",
    "fragment_centerline",
    "main_path",
    "refine",
    "skeleton_graph",
    "smooth_polyline",
]
