"""Trace physical longitudinal lines from independent AI pieces and the observed paint.

Each AI fragment used to be digitized on its own, which gave overlapping pieces, solid/dash
conflicts on the same paint, gaps where the model skipped visible paint and one wobbly fit per
piece. Here the pieces of one physical line are assembled and judged together:

1. ``cluster``: pieces lying on the same paint (any class) become one path;
2. ``chain``: collinear paths are linked end to end (a path only, nothing is decided yet);
3. paint following: each path end is extended while paint is visible in the intensity image;
4. profile: every 5 cm along the path is PAINT (bright return or AI mask), HIDDEN (no return on
   the line's own road surface: vehicle, overpass, unscanned) or DARK (visible unpainted road);
5. segments: continuity and class come from that profile. Visible unpainted road is a real
   break; hidden stretches inside a solid line are joined and returned as INFERRED parts;
   paint the model missed is observed paint; a repeating short-piece rhythm is a dash line;
6. one local-quadratic fit per traced line gives the centerline (client rule: the polyline
   lies on the axis of the mark), cut at the paint run ends; each dash is its own element.

The scene may hold several height levels (``Layer``): the lowest road surface and, where a
structure crosses above it, the deck. A line is always read from the level whose height matches
its own, so a road under a bridge and the road on the bridge are traced independently.
No paint is created: inferred parts are returned separately for styling, review and audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import shapely
from scipy.ndimage import binary_closing, binary_dilation, find_objects, label
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from shapely.ops import substring

from .gap_bridging import end_state, hermite
from .qa import smoothness_mm
from .raster_tiles import pixel_to_xy, xy_to_pixel

STEP = 0.05
PAINT, HIDDEN, DARK = 2, 0, 1
OFFSETS = np.arange(-12, 13) * 0.025  # +-0.30 m across the line
BAND = np.abs(OFFSETS) <= 0.1001
CORE = np.abs(OFFSETS) <= 0.1501
RULES = dict(
    max_gap_m=40,
    max_angle_deg=8,
    overlap_m=0.12,
    max_lateral_m=0.25,
    max_dark_chain_m=14.0,
    level_tolerance_m=0.4,
    max_extend_m=80.0,
    max_width_m=0.45,
    fill_m=0.30,
    min_paint_m=0.15,
    break_dark_m=0.5,
    break_dark_fraction=0.2,
    worn_gap_m=2.0,
    dash_max_m=12.5,
    dash_gap_m=(2.5, 14.0),
    dash_visible_gap_m=1.0,
    pattern_tolerance=0.35,
    min_side_m=1.5,
    min_long_side_m=6.0,
    solid_window_m=8.0,
    dash_window_m=20.0,
    fit_tolerance_m=0.01,
    out_step_m=0.25,
    min_feature_m=0.5,
)
# name: (dash m, gap m, width m). "short" is shared by acceleration and deceleration lines.
PATTERNS = {"normal": (4, 10, 0.15), "short": (3, 4, 0.30), "aviso": (10, 4, 0.15)}


@dataclass
class Layer:
    """One height level on the common pixel grid. ``height`` (a ``HeightGrid``) may be None
    when the scene has a single level and no height test is wanted."""

    bright: np.ndarray
    observed: np.ndarray
    model: np.ndarray
    height: object = None
    name: str = "low"

    def __post_init__(self):
        # Sparse scan stripes still show the road: a station is hidden only when no return
        # lies within 3 px (0.15 m).
        self.seen = binary_dilation(self.observed, iterations=3)
        self.paint = (self.bright & self.observed) | self.model


class Scene:
    def __init__(self, grid, layers, tolerance=RULES["level_tolerance_m"]):
        self.grid, self.layers, self.tolerance = grid, list(layers), tolerance
        self.leveled = all(layer.height is not None for layer in self.layers)

    def heights(self, xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        if not self.leveled:
            return np.zeros((len(self.layers), len(xy)))
        return np.stack([layer.height.at(xy[:, 0], xy[:, 1]) for layer in self.layers])

    def level(self, xy, z):
        """(layer index per point or -1, height of that level). ``z`` NaN = lowest known."""
        h = self.heights(xy)
        z = np.broadcast_to(np.asarray(z, float), h.shape[1:])
        if not self.leveled:
            return np.zeros(h.shape[1], int), h[0]
        d = np.abs(h - z[None])
        unknown = ~np.isfinite(z)
        d[:, unknown] = np.where(
            np.isfinite(h[:, unknown]), np.arange(len(h))[:, None] * 1e-3, np.nan
        )
        d = np.where(np.isfinite(d), d, np.inf)
        idx = d.argmin(0)
        best = np.take_along_axis(d, idx[None], 0)[0]
        idx = np.where(best <= self.tolerance, idx, -1)
        found = np.where(idx >= 0, np.take_along_axis(h, np.maximum(idx, 0)[None], 0)[0], np.nan)
        return idx, found

    def take(self, attr, xy, idx):
        """Sample mask ``attr`` of layer ``idx[i]`` at ``xy[i]`` (xy may carry extra axes)."""
        xy = np.asarray(xy, float)
        r, c = xy_to_pixel(self.grid, xy[..., 0], xy[..., 1])
        r, c = np.rint(r).astype(int), np.rint(c).astype(int)
        ok = (r >= 0) & (c >= 0) & (r < self.grid.rows) & (c < self.grid.cols)
        r, c = r.clip(0, self.grid.rows - 1), c.clip(0, self.grid.cols - 1)
        idx = np.broadcast_to(np.asarray(idx).reshape(-1, *([1] * (xy.ndim - 2))), r.shape)
        out = np.zeros(r.shape, bool)
        for k, layer in enumerate(self.layers):
            sel = ok & (idx == k)
            if sel.any():
                out[sel] = getattr(layer, attr)[r[sel], c[sel]]
        return out

    def inside(self, p):
        x0, y0, x1, y1 = self.grid.bounds
        return x0 <= p[0] <= x1 and y0 <= p[1] <= y1


def arclen(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def resample(coords, step):
    p = np.asarray(coords, float)
    s = arclen(p[:, :2])
    if s[-1] < 1e-9:
        return p[:1]
    t = np.linspace(0, s[-1], max(2, int(np.ceil(s[-1] / step)) + 1))
    return np.column_stack([np.interp(t, s, p[:, k]) for k in range(p.shape[1])])


def runs(values):
    edges = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
    return list(pairwise(edges))


def split_reversals(line, window_m=1.0, max_turn_deg=60.0, min_m=0.5):
    """Cut a polyline where it turns sharply (a skeleton that runs out along one stripe of a
    double line and back along the other). Road markings never turn > 60 degrees in 1 m."""
    if line.length < 2 * window_m:
        return [line]
    p = resample(np.asarray(line.coords)[:, :2], 0.25)
    d = np.diff(p, axis=0)
    heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    k = max(1, round(window_m / 0.25))
    turn = np.abs(heading[k:] - heading[:-k])
    bad = np.flatnonzero(turn > np.deg2rad(max_turn_deg))
    if not len(bad):
        return [line]
    cuts, last = [], -10
    for b in bad:  # one cut per sharp turn, at its middle
        if b - last > k:
            cuts.append(b + k // 2 + 1)
        last = b
    pieces = np.split(p, cuts)
    return [LineString(q) for q in pieces if len(q) >= 2 and arclen(q)[-1] >= min_m]


def drop_overlaps(features, buffer_m=RULES["overlap_m"], keep_fraction=0.3, max_angle_deg=20):
    """Draw shared paint once and connect merging lines (client rule: elements that connect
    share a vertex).

    Longer features are kept first. Along each shorter feature, stations lying within
    ``buffer_m`` of a kept feature AND running in the same direction are "shared" (a crossing
    at an angle is not shared). A feature mostly shared is dropped; shared stretches at its
    ends are trimmed and the new end is snapped onto the line it merges into, so two lines that
    converge at a junction become one line with the branch attached to it.
    Returns (features, dropped, trimmed)."""
    order = sorted(range(len(features)), key=lambda k: -features[k]["geometry_2d"].length)
    cosine = np.cos(np.deg2rad(max_angle_deg))
    kept, kept_lines, dropped, trimmed = [], [], 0, 0
    for k in order:
        f = features[k]
        g = f["geometry_2d"]
        near = (
            shapely.STRtree(kept_lines).query(g.buffer(buffer_m), predicate="intersects")
            if kept_lines
            else []
        )
        if len(near):
            p = resample(np.asarray(g.coords)[:, :2], 0.1)
            t = np.gradient(p, axis=0)
            t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
            shared = np.zeros(len(p), bool)
            owner = np.full(len(p), -1)
            pts = shapely.points(p)
            for j in near:
                line = kept_lines[j]
                close = shapely.distance(pts, line) <= buffer_m
                if not close.any():
                    continue
                q = shapely.line_locate_point(line, pts[close])
                a_ = shapely.get_coordinates(
                    shapely.line_interpolate_point(line, np.maximum(q - 0.2, 0))
                )
                b_ = shapely.get_coordinates(
                    shapely.line_interpolate_point(line, np.minimum(q + 0.2, line.length))
                )
                u = b_ - a_
                u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
                aligned = np.abs(np.sum(u * t[close], axis=1)) >= cosine
                idx = np.flatnonzero(close)[aligned]
                shared[idx] = True
                owner[idx] = j
            length = arclen(p)[-1]
            if shared.mean() * length > max(0.0, length - max(1.0, keep_fraction * length)):
                dropped += 1
                continue
            lo, hi = 0, len(p)
            while lo < hi and shared[lo]:
                lo += 1
            while hi > lo and shared[hi - 1]:
                hi -= 1
            if (lo > 0 or hi < len(p)) and arclen(p[lo:hi])[-1] >= 1.0:
                core = p[lo:hi]
                for end, idx in ((0, lo - 1), (1, hi)):
                    if 0 <= idx < len(p) and owner[idx] >= 0:
                        trunk = kept_lines[owner[idx]]
                        oriented = core if end == 1 else core[::-1]  # ends at the junction
                        cut = np.searchsorted(arclen(oriented), arclen(oriented)[-1] - 1.0)
                        body = oriented[: max(cut, 2)]
                        tip = p[idx]
                        d = trunk.project(shapely.Point(tip))
                        join = np.asarray(trunk.interpolate(d).coords[0])[:2]
                        a_ = np.asarray(trunk.interpolate(max(d - 0.5, 0)).coords[0])[:2]
                        b_ = np.asarray(trunk.interpolate(min(d + 0.5, trunk.length)).coords[0])[:2]
                        tt = (b_ - a_) / max(np.linalg.norm(b_ - a_), 1e-9)
                        t0 = body[-1] - body[-2]
                        t0 /= max(np.linalg.norm(t0), 1e-9)
                        if tt @ t0 < 0:
                            tt = -tt
                        oriented = np.vstack([body, hermite(body[-1], t0, join, tt, step=0.1)[1:]])
                        core = oriented if end == 1 else oriented[::-1]
                g2 = LineString(core)
                f = dict(
                    f,
                    geometry_2d=g2,
                    length_m=g2.length,
                    flags=sorted(set(f["flags"]) | {"joined_to_merging_line"}),
                    **smoothness_mm(core),
                )
                trimmed += 1
        kept.append(f)
        kept_lines.append(f["geometry_2d"])
    return kept, dropped, trimmed


# ---------------------------------------------------------------- 1. overlapping pieces
def cluster(lines, heights=None):
    """Merge pieces lying on the same paint. Returns [(LineString path, member indices)].

    Two pieces merge when they agree (<= 0.12 m, <= 10 degrees) over at least 80 % of their
    common longitudinal extent, so a stripe that only converges onto another one stays
    separate. ``heights`` (optional list of f(points) -> z) keeps levels apart.
    """
    parents = list(range(len(lines)))

    def root(i):
        while i != parents[i]:
            i = parents[i]
        return i

    tree = shapely.STRtree(lines)
    for i, line in enumerate(lines):
        for j in tree.query(line, predicate="dwithin", distance=RULES["overlap_m"]):
            if j <= i:
                continue
            (a, short), (b, long) = sorted(((i, line), (j, lines[j])), key=lambda g: g[1].length)
            count = max(1, int(np.ceil(short.length / 0.25)))
            step = short.length / count
            s = (np.arange(count) + 0.5) * step
            p = shapely.line_interpolate_point(short, s)
            t = shapely.get_coordinates(
                shapely.line_interpolate_point(short, s + step / 2)
            ) - shapely.get_coordinates(shapely.line_interpolate_point(short, s - step / 2))
            q = shapely.line_locate_point(long, p)
            u = shapely.get_coordinates(
                shapely.line_interpolate_point(long, np.minimum(long.length, q + 0.125))
            ) - shapely.get_coordinates(
                shapely.line_interpolate_point(long, np.maximum(0, q - 0.125))
            )
            aligned = abs(np.sum(t * u, axis=1)) >= np.cos(np.deg2rad(10)) * np.linalg.norm(
                t, axis=1
            ) * np.linalg.norm(u, axis=1)
            # Compare the common longitudinal extent. A fragment may extend beyond
            # another fragment, but a diverging stripe must agree along the overlap.
            common = (q > 0.001) & (q < long.length - 0.001)
            agreeing = common & aligned & (shapely.distance(p, long) <= RULES["overlap_m"])
            if heights is not None and agreeing.any():
                xy = shapely.get_coordinates(p)
                dz = np.abs(heights[a](xy) - heights[b](xy))
                agreeing &= ~(dz > RULES["level_tolerance_m"])
            shared = np.count_nonzero(agreeing)
            if step * shared >= min(0.5, short.length / 2) and shared >= 0.8 * np.count_nonzero(
                common
            ):
                parents[root(j)] = root(i)
    groups = {}
    for i in range(len(lines)):
        groups.setdefault(root(i), []).append(i)
    paths = []
    for members in groups.values():
        base = max((lines[i] for i in members), key=lambda g: g.length)
        ends = []
        for start in (True, False):
            p, t = end_state(base, start)
            tails = []
            for i in members:
                line = lines[i]
                for reverse in (False, True):
                    xy = np.asarray(line.coords)[:: (-1 if reverse else 1)]
                    progress = (xy - p) @ t
                    if progress[-1] > 0.05 and np.any(progress <= 0):
                        idx = np.flatnonzero(progress <= 0)[-1]
                        fraction = -progress[idx] / (progress[idx + 1] - progress[idx])
                        crossing = xy[idx] + fraction * (xy[idx + 1] - xy[idx])
                        if np.linalg.norm(crossing - p) <= RULES["overlap_m"]:
                            tails.append((progress[-1], xy[idx + 1 :]))
            ends.append(max(tails, key=lambda pair: pair[0])[1] if tails else np.empty((0, 2)))
        paths.append((LineString(np.vstack((ends[0][::-1], base.coords, ends[1]))), members))
    return paths


def trim_shared_approach(line, retained):
    """Trim a terminal overlap while preserving the branch beyond the split. The branch then
    starts ON the retained line (client rule: connected elements share a vertex)."""
    for reverse in (False, True):
        branch = shapely.reverse(line) if reverse else line
        stations = np.arange(0, branch.length, 0.25)
        points = shapely.line_interpolate_point(branch, stations)
        for trunk in retained:
            near = shapely.distance(points, trunk) <= RULES["overlap_m"]
            if not len(near) or not near[0] or near.all():
                continue
            split = np.flatnonzero(~near)[0]
            if split < 4 or np.count_nonzero(~near[split:]) < 4:
                continue
            projections = shapely.line_locate_point(trunk, points[:split])
            if abs(projections[-1] - projections[0]) < 0.9 * stations[split - 1]:
                continue
            join = trunk.interpolate(projections[-1])
            tail = substring(branch, stations[split], branch.length)
            trimmed = LineString([join.coords[0], *tail.coords])
            return shapely.reverse(trimmed) if reverse else trimmed
    return line


# ---------------------------------------------------------------- 2. chaining
def chain(paths, scene=None, zs=None):
    """Link collinear paths end to end: [(LineString, members)] -> (the same shape, audit).

    Endpoints are matched greedily by cost (gap + 20 x lateral); each endpoint joins once.
    The link is a cubic Hermite curve matching both end tangents. With a ``scene`` a link is
    refused when it would cross more visible unpainted road than the longest dash gap, and
    ``zs`` (end heights per path: [(z_start, z_end)]) keeps height levels apart.
    """
    n = len(paths)
    if n < 2:
        return list(paths), []
    cosine = np.cos(np.deg2rad(RULES["max_angle_deg"]))
    lines = [line for line, _ in paths]
    ends = {}
    for i, line in enumerate(lines):
        for start in (True, False):
            ends[(i, start)] = end_state(line, start, span=min(2.0, max(line.length, 0.1)))
    keys = list(ends)
    tree = cKDTree(np.array([ends[k][0] for k in keys]))
    candidates = []
    for ia, ib in tree.query_pairs(RULES["max_gap_m"]):
        (i, a), (j, b) = keys[ia], keys[ib]
        if i == j:
            continue
        (p, t), (q, u) = ends[(i, a)], ends[(j, b)]
        delta = q - p
        gap = float(np.linalg.norm(delta))
        if gap < 1e-3 or -(t @ u) < cosine:
            continue
        lateral = float(max(abs(delta @ np.array([-v[1], v[0]])) for v in (t, u)))
        if t @ delta <= 0 or -(u @ delta) <= 0:
            continue
        if gap >= 1:
            if min(t @ delta, -(u @ delta)) < cosine * gap:
                continue
            limit = RULES["max_lateral_m"] + 0.005 * gap
            if min(lines[i].length, lines[j].length) < 1.5:
                limit = min(limit, 0.15)  # the tangent of a very short piece is unreliable
            if lateral > limit:
                continue
        elif lateral > 0.10:
            continue
        if zs is not None:
            za, zb = zs[i][0 if a else 1], zs[j][0 if b else 1]
            if np.isfinite(za) and np.isfinite(zb) and abs(za - zb) > 0.3 + 0.08 * gap:
                continue
        if scene is not None and gap > 0.5:
            k = max(3, int(np.ceil(gap / STEP)))
            inner = p + np.linspace(0, 1, k + 2)[1:-1][:, None] * delta
            z = np.linspace(za, zb, k + 2)[1:-1] if zs is not None else np.full(k, np.nan)
            idx, _ = scene.level(inner, z)
            dark = scene.take("seen", inner, idx) & ~scene.take("paint", inner, idx)
            if dark.mean() * gap > RULES["max_dark_chain_m"]:
                continue
        candidates.append((gap + 20 * lateral, i, a, j, b, gap))
    candidates.sort(key=lambda c: c[:5])
    parents = list(range(n))

    def root(i):
        while i != parents[i]:
            i = parents[i]
        return i

    used, link, audit, bridges = set(), {}, [], []
    obstacle = shapely.STRtree(lines)
    for _, i, a, j, b, gap in candidates:
        if (i, a) in used or (j, b) in used or root(i) == root(j):
            continue
        (p, t), (q, u) = ends[(i, a)], ends[(j, b)]
        curve = hermite(p, t, q, -u) if gap > 0.1 else np.vstack([p, q])
        bridge = LineString(curve)
        if gap > 0.3 and (
            any(
                k not in (i, j) and bridge.crosses(lines[k])
                for k in obstacle.query(bridge, predicate="intersects")
            )
            or any(bridge.crosses(other) for other in bridges)
        ):
            audit.append(
                dict(
                    a=paths[i][1],
                    b=paths[j][1],
                    gap_m=gap,
                    accepted=False,
                    reason="crosses_other_feature",
                )
            )
            continue
        used |= {(i, a), (j, b)}
        link[(i, a)] = (j, b, curve)
        link[(j, b)] = (i, a, curve[::-1])
        parents[root(i)] = root(j)
        bridges.append(bridge)
        audit.append(
            dict(a=paths[i][1], b=paths[j][1], gap_m=gap, accepted=True, reason="path_chained")
        )
    out, seen = [], set()
    for i in range(n):
        if i in seen:
            continue
        cur, start = i, True  # walk to the free end of the chain
        while (cur, start) in link:
            cur, far, _ = link[(cur, start)]
            start = not far
        parts, members, enter = [], [], start
        while True:
            seen.add(cur)
            xy = np.asarray(lines[cur].coords)
            parts.append(xy if enter else xy[::-1])
            members += paths[cur][1]
            leave = (cur, not enter)
            if leave not in link:
                break
            cur, enter, curve = link[leave]
            if len(curve) > 2:
                parts.append(curve[1:-1])
        xy = np.vstack(parts)
        keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-6]
        out.append((LineString(xy[keep]), members))
    return out, audit


# ---------------------------------------------------------------- 3. paint following
def extend(scene, path, start, z0, obstacles=None):
    """Follow visible paint beyond one end of ``path``. Returns (points Nx2, heights N).

    Stops at visible unpainted road (> 0.5 m), a hidden stretch (> 0.5 m), paint wider than
    a line (another kind of marking), another line, a change of height level or the edge.
    """
    p, direction = end_state(path, start, span=min(1.5, path.length))
    z = z0
    out, heights, trail = [], [], [p.copy()]
    last_paint = hidden = dark = 0
    while len(out) * STEP < RULES["max_extend_m"]:
        q = p + STEP * direction
        if not scene.inside(q):
            break
        idx, found = scene.level(q[None], z)
        if idx[0] < 0:
            hidden += 1
            if hidden * STEP > 0.5:
                break
            out.append(q)
            heights.append(z)
            p = q
            continue
        z = found[0] if np.isfinite(found[0]) else z
        normal = np.array([-direction[1], direction[0]])
        samples = q + OFFSETS[:, None] * normal
        level = np.full(len(samples), idx[0])
        hits = np.flatnonzero(scene.take("paint", samples, level))
        core = hits[np.abs(OFFSETS[hits]) <= 0.15]
        if len(core):
            centre = hits[np.argmin(np.abs(OFFSETS[hits]))]
            group = next(
                g for g in np.split(hits, np.flatnonzero(np.diff(hits) > 1) + 1) if centre in g
            )
            if np.ptp(OFFSETS[group]) + 0.025 > RULES["max_width_m"]:
                break
            q = q + 0.5 * float(np.mean(OFFSETS[group[np.abs(OFFSETS[group]) <= 0.2]])) * normal
            if (
                obstacles is not None
                and len(out) * STEP > 0.3
                and len(obstacles.query(shapely.Point(q), predicate="dwithin", distance=0.12))
            ):
                break
            out.append(q)
            heights.append(z)
            last_paint = len(out)
            hidden = dark = 0
            trail.append(q)
            if len(trail) >= 40 and len(trail) % 5 == 0:
                a, b = np.mean(trail[-40:-25], axis=0), np.mean(trail[-15:], axis=0)
                trend = (b - a) / max(np.linalg.norm(b - a), 1e-9)
                if direction @ trend >= np.cos(np.deg2rad(8)):
                    direction = 0.7 * direction + 0.3 * trend
                    direction /= np.linalg.norm(direction)
        else:
            if scene.take("seen", samples[BAND], level[BAND]).any():
                dark += 1
            else:
                hidden += 1
            if dark * STEP > 0.5 or hidden * STEP > 0.5:
                break
            out.append(q)
            heights.append(z)
        p = q
    return np.asarray(out[:last_paint]).reshape(-1, 2), np.asarray(heights[:last_paint])


# ---------------------------------------------------------------- 4. profile
def profile(scene, xyz, road_support=None):
    """Per-station state along the path ``xyz`` (N x 3, z may be NaN)."""
    pts = resample(xyz, STEP)
    s = arclen(pts[:, :2])
    z = pts[:, 2]
    if np.isfinite(z).any():
        z = np.interp(s, s[np.isfinite(z)], z[np.isfinite(z)])
    xy = pts[:, :2]
    tangent = np.gradient(xy, axis=0) if len(xy) > 1 else np.zeros_like(xy)
    normal = tangent[:, ::-1] * [-1, 1] / np.maximum(1e-9, np.linalg.norm(tangent, axis=1)[:, None])
    idx, found = scene.level(xy, z)
    samples = xy[:, None] + OFFSETS[None, :, None] * normal[:, None]
    visible = scene.take("bright", samples, idx) & scene.take("observed", samples, idx)
    ai = scene.take("model", samples, idx)
    paint = visible | ai
    seen = scene.take("seen", samples, idx)
    state = np.where(paint[:, BAND].any(1), PAINT, np.where(seen[:, BAND].any(1), DARK, HIDDEN))
    state[idx < 0] = HIDDEN
    # The paint group nearest to the path gives the axis offset and the width of the mark.
    centroid, width = np.zeros(len(s)), np.full(len(s), np.nan)
    source = np.where(visible[:, CORE].any(1)[:, None], visible, paint)
    for i in np.flatnonzero(state == PAINT):
        hits = np.flatnonzero(source[i])
        if not len(hits):
            continue
        centre = hits[np.argmin(np.abs(OFFSETS[hits]))]
        group = next(
            g for g in np.split(hits, np.flatnonzero(np.diff(hits) > 1) + 1) if centre in g
        )
        centroid[i] = OFFSETS[group].mean()
        width[i] = len(group) * 0.025
    centroid = np.clip(centroid, -0.2, 0.2)
    if road_support is not None:
        # A return from an obstruction does not establish that the road is visible.
        for a, b in runs(state != PAINT):
            if state[a] == PAINT or a == 0 or b == len(s):
                continue
            dark = np.flatnonzero(state[a:b] == DARK) + a
            if len(dark) and all(road_support(xy[[a - 1, b]])):
                state[dark[~np.asarray(road_support(xy[dark]), bool)]] = HIDDEN
    return dict(
        s=s,
        xy=xy,
        z=np.where(np.isfinite(found), found, z),
        normal=normal,
        state=state,
        centroid=centroid,
        width=width,
        ai=ai[:, BAND].any(1),
        visible=visible[:, BAND].any(1),
    )


def clean_state(state, fill_m=RULES["fill_m"], min_paint_m=RULES["min_paint_m"]):
    """Close short non-paint runs inside paint (sparse scan stripes, hairline wear) and drop
    paint specks shorter than ``min_paint_m``."""
    state = state.copy()
    for a, b in runs(state):
        inner = a > 0 and b < len(state) and state[a - 1] == PAINT and state[b] == PAINT
        if state[a] != PAINT and inner and (b - a) * STEP <= fill_m:
            state[a:b] = PAINT
    for a, b in runs(state):
        if state[a] == PAINT and (b - a) * STEP < min_paint_m:
            left = state[a - 1] if a > 0 else HIDDEN
            right = state[b] if b < len(state) else HIDDEN
            state[a:b] = DARK if DARK in (left, right) else HIDDEN
    return state


# ---------------------------------------------------------------- 5. segments and class
def dash_pattern(length, gap, width):
    """Nearest client standard rhythm, or None when the measured dash/gap matches none (the
    client rule is then to report the measured dimensions)."""
    measures = np.array(
        [length, gap if gap else 1.0, width if width and np.isfinite(width) else 0.15]
    )
    weights = np.array([1.0, 1.0 if gap else 0.0, 0.5])

    def error(key):
        return np.linalg.norm(weights * np.log(measures / np.asarray(PATTERNS[key], float)))

    best = min(PATTERNS, key=error)
    return best if error(best) <= RULES["pattern_tolerance"] * np.sqrt(weights.sum()) else None


def segments(prof):
    """Continuity and class from the paint profile. Returns a list of dict(a, b (station
    range), class_id, inferred=[(a, b, label)], rule, gap_next_m, dash_pattern, ...)."""
    state = clean_state(prof["state"])
    paint = [(a, b) for a, b in runs(state) if state[a] == PAINT]
    if not paint:
        return [], state
    gaps = []
    for (_, b0), (a1, _) in pairwise(paint):
        dark = np.count_nonzero(state[b0:a1] == DARK) * STEP
        length = (a1 - b0) * STEP
        gaps.append(
            dict(
                a=b0,
                b=a1,
                len=length,
                dark=dark,
                is_break=dark >= RULES["break_dark_m"]
                and dark / length > RULES["break_dark_fraction"],
            )
        )
    lens = np.array([(b - a) * STEP for a, b in paint])
    glen = np.array([g["len"] for g in gaps])
    breaks = np.array([g["is_break"] for g in gaps], bool)
    lo, hi = RULES["dash_gap_m"]
    short = lens <= RULES["dash_max_m"]
    # Dash rhythm: repeating short pieces with dash-sized spacing. One visible unpainted gap
    # between two short pieces is enough; gaps that are only hidden need a regular rhythm of
    # at least three pieces (one occluded solid line must not become dashes).
    # A visible unpainted gap of >= 1 m between two short pieces is a dash gap whatever its
    # length (toll-plaza and urban dashes have 1-2 m gaps); worn paint leaves shorter gaps.
    rhythm = False
    visible_gap = breaks & (glen >= RULES["dash_visible_gap_m"]) & (glen <= hi)
    # Fine rhythm (blocks of 0.3-1.5 m with 0.2-1.5 m visible gaps, e.g. warning lines at
    # merges): at least three such gaps between short pieces.
    if len(paint) >= 4:
        dark_len = np.array([g["dark"] for g in gaps])
        tiny = lens <= 1.5
        fine = tiny[:-1] & tiny[1:] & (dark_len >= 0.2) & (dark_len >= 0.6 * glen) & (glen <= 1.5)
        if fine.sum() >= 3:
            visible_gap = visible_gap | fine
            breaks = breaks | fine
    if len(paint) >= 2:
        pair = short[:-1] & short[1:] & (glen >= lo) & (glen <= hi)
        regular = short.sum() >= 3 and np.std(lens[short]) <= 0.4 * np.mean(lens[short])
        rhythm = bool(
            (short[:-1] & short[1:] & visible_gap).any()
            or (pair.sum() >= 2 and short.mean() >= 0.6 and regular)
        )
    out = []
    cur = dict(a=paint[0][0], b=paint[0][1], inferred=[])
    cur_len = lens[0]
    for k, g in enumerate(gaps):
        nxt = lens[k + 1]
        if rhythm and (g["len"] >= lo or visible_gap[k]) and (short[k] or short[k + 1]):
            join = None
        elif g["is_break"]:
            long_sides = cur_len > RULES["dash_max_m"] and nxt > RULES["dash_max_m"]
            join = "INFERRED_WORN_PAINT" if g["len"] <= RULES["worn_gap_m"] and long_sides else None
        elif g["len"] < 1.0 or (
            max(cur_len, nxt) >= RULES["min_long_side_m"]
            and min(cur_len, nxt) >= RULES["min_side_m"]
        ):
            join = "INFERRED_UNDER_OCCLUSION"
        else:
            join = None
        if join:
            if g["len"] >= 0.3:  # shorter hidden stretches are scan gaps, not occlusion
                cur["inferred"].append((g["a"], g["b"], join))
            cur["b"] = paint[k + 1][1]
            cur_len += g["len"] + nxt
        else:
            cur.update(gap_next_m=float(g["len"]), gap_is_break=bool(g["is_break"]))
            out.append(cur)
            cur, cur_len = dict(a=paint[k + 1][0], b=paint[k + 1][1], inferred=[]), nxt
    out.append(cur)
    for k, sg in enumerate(out):
        length = (sg["b"] - sg["a"]) * STEP
        sl = slice(sg["a"], sg["b"])
        widths = prof["width"][sl][np.isfinite(prof["width"][sl])]
        gap = sg.get("gap_next_m", out[k - 1].get("gap_next_m") if k else None)
        dash_like = rhythm and length <= RULES["dash_max_m"]
        sg.update(
            length=length,
            width=float(np.median(widths)) if len(widths) else np.nan,
            class_id=1 if dash_like else (0 if length > RULES["dash_max_m"] else None),
            rule=(
                "dash_rhythm"
                if dash_like
                else ("long_paint" if length > RULES["dash_max_m"] else "members")
            ),
            dash_gap=gap,
        )
    return out, state


# ---------------------------------------------------------------- 6. geometry
def max_windows(xy, tol_m=RULES["fit_tolerance_m"], chord_m=10.0):
    """Largest half windows keeping the fit within ``tol_m`` of a curve of the path's tightest
    radius R = 1/kappa: quadratic (cubic term ~ h^3 kappa^2 / 6) and linear (h^2 kappa / 2)."""
    pts = resample(xy, chord_m)
    if len(pts) < 3:
        return np.inf, np.inf
    d = np.diff(pts, axis=0)
    turn = np.abs(np.diff(np.unwrap(np.arctan2(d[:, 1], d[:, 0]))))
    kappa = float(np.percentile(turn, 90)) / max(float(np.linalg.norm(d, axis=1).mean()), 1e-6)
    if kappa < 1e-5:
        return np.inf, np.inf
    return max(3.0, (6 * tol_m / kappa**2) ** (1 / 3)), max(1.5, (2 * tol_m / kappa) ** 0.5)


def loess(s, xy, w, s_eval, h, h_linear=np.inf):
    """Weighted local regression of xy on the path parameter (tricube weights, half window
    ``h``). Quadratic where the support spans enough of the window, linear otherwise (short
    pieces, sparse ends) with the window limited to ``h_linear``."""
    s, xy, w = np.asarray(s, float), np.asarray(xy, float), np.asarray(w, float)
    out = np.empty((len(s_eval), 2))
    for k, t in enumerate(s_eval):
        hh = h
        for _ in range(8):
            lo, hi = np.searchsorted(s, [t - hh, t + hh])
            ds = s[lo:hi] - t
            ww = w[lo:hi] * np.maximum(0, 1 - np.abs(ds / hh) ** 3) ** 3
            if np.count_nonzero(ww > 0) >= 10 or (lo == 0 and hi == len(s)):
                break
            hh *= 1.5
        on = ww > 0
        if np.count_nonzero(on) < 2:
            out[k] = xy[np.argmin(np.abs(s - t))]
            continue
        span = np.ptp(ds[on])
        degree = 2 if span >= 4.0 and np.count_nonzero(on) >= 30 else 1
        if degree == 1 and hh > h_linear:
            narrow = on & (np.abs(ds) <= max(h_linear, 1.0))
            if np.count_nonzero(narrow) >= 10:
                ww = ww * narrow
        a = np.vander(ds / hh, degree + 1)
        sw = np.sqrt(ww)[:, None]
        out[k] = np.linalg.lstsq(a * sw, xy[lo:hi] * sw, rcond=None)[0][-1]
    return out


def fit(prof, segs, state):
    """One smooth curve per traced line, evaluated on each segment. Returns [(segment,
    coords, heights, [(LineString, label)])]."""
    observed = prof["xy"] + prof["normal"] * prof["centroid"][:, None]
    w = np.where(state == PAINT, 1.0, 0.0)
    w[(state == PAINT) & (prof["state"] != PAINT)] = 0.05  # closed scan gaps: path only
    for sg in segs:  # inside joined stretches the path itself is a weak guide
        for a, b, _ in sg["inferred"]:
            w[a:b] = np.maximum(w[a:b], 0.02)
    use = w > 0
    if np.count_nonzero(use) < 2:
        return []
    s = prof["s"]
    dashes = sum(sg["class_id"] == 1 for sg in segs)
    h_quadratic, h_linear = max_windows(prof["xy"])
    wanted = (
        RULES["dash_window_m"]
        if dashes >= 2 and dashes >= len(segs) / 2
        else RULES["solid_window_m"]
    )
    h = min(wanted, h_quadratic)
    out = []
    for sg in segs:
        s0, s1 = s[sg["a"]], s[sg["b"] - 1]
        if s1 - s0 < RULES["min_feature_m"]:
            continue
        t = np.linspace(s0, s1, max(2, int(np.ceil((s1 - s0) / RULES["out_step_m"])) + 1))
        xy = loess(s[use], observed[use], w[use], t, h, h_linear)
        z = np.interp(t, s, prof["z"])
        parts = []
        for a, b, tag in sg["inferred"]:
            sel = (t >= s[a] - 0.13) & (t <= s[min(b, len(s) - 1)] + 0.13)
            if np.count_nonzero(sel) >= 2:
                parts.append((LineString(xy[sel]), tag))
        out.append((sg, xy, z, parts))
    return out


# ---------------------------------------------------------------- seeds from paint only
def seed_pieces(bright, covered, grid, min_length_m=3.0, max_width_m=0.35):
    """Line-like paint the model did not report: elongated thin bright components away from
    accepted masks. Returns a list of polylines (N x 2, metres); callers must still check that
    a seed lies on the road surface."""
    from .centerline import fragment_centerline

    free = binary_closing(bright, iterations=2) & ~binary_dilation(covered, iterations=4)
    labels, _ = label(free, structure=np.ones((3, 3), bool))
    out = []
    for k, sl in enumerate(find_objects(labels), start=1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if np.hypot(h, w) * grid.cell < min_length_m:
            continue
        mask = labels[sl] == k
        cl = fragment_centerline(mask, mask, smooth_window_px=30.0)
        if cl is None or len(cl["refined"]) < 2 or "branching" in cl["flags"]:
            continue
        if cl["width_px"] * grid.cell > max_width_m:
            continue
        rc = np.asarray(cl["refined"], float) + np.array([sl[0].start, sl[1].start])
        x, y = pixel_to_xy(grid, rc[:, 0], rc[:, 1])
        xy = np.column_stack([x, y])
        if arclen(xy)[-1] >= min_length_m:
            out.append(xy)
    return out


# ---------------------------------------------------------------- driver
def trace_lines(
    features, grid, bright=None, observed=None, model=None, *, road_support=None, layers=None
):
    """Return (traced features, path chaining audit).

    ``features``: seed pieces with ``geometry_2d``, ``class_id`` (0 solid, 1 dash, -1 unknown),
    ``length_m``, ``mark_id``, ``flags``, ``tiles``, ``queries``, ``instance_ids`` and, with
    several ``layers``, ``layer`` (index of the level the piece was recognised on).
    """
    if layers is None:
        layers = [Layer(bright, observed, model)]
    scene = Scene(grid, layers)
    if not features:
        return [], []
    split = []
    for f in features:
        for part in split_reversals(f["geometry_2d"]):
            split.append(
                dict(f, geometry_2d=part, length_m=part.length)
                if part is not f["geometry_2d"]
                else f
            )
    features = split
    source_lines = [f["geometry_2d"] for f in features]
    level_of = [int(f.get("layer", 0)) for f in features]

    def piece_height(i):
        return lambda xy: scene.heights(xy)[level_of[i]]

    heights = [piece_height(i) for i in range(len(features))] if scene.leveled else None

    def height_along(line, members):
        """Height of a path at its vertices, read from the level of the nearest member."""
        xy = np.asarray(line.coords)[:, :2]
        if not scene.leveled:
            return np.zeros(len(xy))
        h = scene.heights(xy)
        if len({level_of[i] for i in members}) == 1:
            return h[level_of[members[0]]]
        tree = shapely.STRtree([source_lines[i] for i in members])
        near = tree.nearest(shapely.points(xy))
        return h[[level_of[members[j]] for j in near], np.arange(len(xy))]

    def end_heights(line, members):
        z = height_along(line, members)
        good = np.flatnonzero(np.isfinite(z))
        return (z[good[0]], z[good[-1]]) if len(good) else (np.nan, np.nan)

    groups = cluster(source_lines, heights)
    zs = [end_heights(line, members) for line, members in groups]
    paths, audit = chain(groups, scene, zs if scene.leveled else None)
    paths = [(line, members) for line, members in paths if line.length >= STEP]
    # Longest first: a long line claims its paint before a short neighbour can run onto it.
    order = sorted(range(len(paths)), key=lambda k: -paths[k][0].length)
    current = [line for line, _ in paths]
    extended = {}
    for k in order:
        line, members = paths[k]
        others = [g for j, g in enumerate(current) if j != k]
        obstacles = shapely.STRtree(others) if others else None
        z0, z1 = end_heights(line, members)
        head, zh = extend(scene, line, True, z0, obstacles)
        tail, zt = extend(scene, line, False, z1, obstacles)
        xy = np.vstack((head[::-1], np.asarray(line.coords)[:, :2], tail))
        z = np.r_[zh[::-1], height_along(line, members), zt]
        current[k] = LineString(xy)
        extended[k] = (current[k], members, z)
    extended = [extended[k] for k in range(len(paths))]
    # Paint following can make separate seeds converge onto one physical line.
    regrouped = cluster([line for line, _, _ in extended]) if extended else []
    regrouped = [(part, ids) for path, ids in regrouped for part in split_reversals(path)]
    regrouped.sort(key=lambda item: -item[0].length)
    output, retained = [], []
    for track, (path, ids) in enumerate(regrouped, 1):
        path = trim_shared_approach(path, retained)
        retained.append(path)
        members = sorted({i for k in ids for i in extended[k][1]})
        xy = np.asarray(path.coords)[:, :2]
        if scene.leveled:
            known = np.vstack(
                [
                    np.column_stack([np.asarray(extended[k][0].coords)[:, :2], extended[k][2]])
                    for k in ids
                ]
            )
            known = known[np.isfinite(known[:, 2])]
            if len(known):
                _, near = cKDTree(known[:, :2]).query(xy, distance_upper_bound=0.5)
                z = np.where(near < len(known), known[np.minimum(near, len(known) - 1), 2], np.nan)
            else:
                z = np.full(len(xy), np.nan)
        else:
            z = np.zeros(len(xy))
        prof = profile(scene, np.column_stack([xy, z]), road_support)
        segs, state = segments(prof)
        mem = [features[i] for i in members]
        for sg, coords, z_hint, inferred in fit(prof, segs, state):
            line = LineString(coords)
            near = [m for m in mem if m["geometry_2d"].distance(line) <= RULES["overlap_m"]]
            seg_members = near or mem
            base = max(seg_members, key=lambda m: m["length_m"])
            flags = {flag for m in seg_members for flag in m["flags"]} - {
                "class_conflict",
                "joined_by_paint_evidence",
                "branching",
                "multi_component_skeleton",
            }
            cls, pattern = sg["class_id"], None
            if cls is None:
                votes = [
                    (
                        m["geometry_2d"].intersection(line.buffer(RULES["overlap_m"])).length,
                        m["class_id"],
                    )
                    for m in near
                    if m["class_id"] in (0, 1)
                ]
                if votes:
                    cls = int(
                        sum(v for v, c in votes if c == 1) > sum(v for v, c in votes if c == 0)
                    )
                else:
                    cls = int(sg["length"] <= 6.5)
                flags.add("isolated_short_segment")
            if cls == 1:
                pattern = dash_pattern(sg["length"], sg["dash_gap"], sg["width"])
            if not near or all("paint_only_detection" in m["flags"] for m in near):
                flags.add("paint_only_detection")
            else:
                flags.discard("paint_only_detection")
            sl = slice(sg["a"], sg["b"])
            painted = prof["state"][sl] == PAINT
            output.append(
                dict(
                    base,
                    mark_id=len(output) + 1,
                    geometry_2d=line,
                    raw_2d=None,
                    length_m=line.length,
                    class_id=cls,
                    class_name="longitudinal_dash" if cls else "longitudinal_solid",
                    width_m=(
                        sg["width"] if np.isfinite(sg["width"]) else base.get("width_m", np.nan)
                    ),
                    flags=sorted(flags),
                    track_id=track,
                    class_rule=sg["rule"],
                    dash_pattern=pattern,
                    gap_next_m=sg.get("gap_next_m"),
                    paint_traced_m=float(
                        np.count_nonzero(prof["visible"][sl] & ~prof["ai"][sl]) * STEP
                    ),
                    paint_fraction=float(painted.mean()),
                    containment=float(painted.mean()),
                    paint_coverage=float(painted.mean()),
                    z_hint=z_hint,
                    inferred_segments=[g for g, _ in inferred],
                    inferred_labels=[lab for _, lab in inferred],
                    inferred_length_m=float(sum(g.length for g, _ in inferred)),
                    merged_marks=[m["mark_id"] for m in seg_members],
                    tiles=sorted({x for m in seg_members for x in m["tiles"]}),
                    queries=sorted({x for m in seg_members for x in m["queries"]}),
                    instance_ids=sorted({x for m in seg_members for x in m["instance_ids"]}),
                    **smoothness_mm(coords),
                )
            )
    output, dropped, trimmed = drop_overlaps(output)
    for k, f in enumerate(output, start=1):
        f["mark_id"] = k
    audit.append(dict(accepted=True, reason="overlap_cleanup", dropped=dropped, trimmed=trimmed))
    return output, audit


__all__ = [
    "PATTERNS",
    "RULES",
    "STEP",
    "Layer",
    "Scene",
    "chain",
    "cluster",
    "dash_pattern",
    "drop_overlaps",
    "extend",
    "fit",
    "loess",
    "profile",
    "runs",
    "seed_pieces",
    "segments",
    "split_reversals",
    "trace_lines",
    "trim_shared_approach",
]
