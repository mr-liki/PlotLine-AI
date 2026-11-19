"""Extract and classify painted area markings using held-out shape references."""

import numpy as np
import shapely
import shapely.ops
from rasterio.features import rasterize, shapes
from scipy.ndimage import binary_closing, binary_opening
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, shape

from .fragments import bright_paint


def descriptor(poly, lines):
    rect = np.asarray(poly.minimum_rotated_rectangle.exterior.coords)
    edges = np.diff(rect, axis=0)
    sides = np.linalg.norm(edges, axis=1)
    width, length = sides.min(), sides.max()
    angle = 0.0
    if lines:
        line = min(lines, key=poly.distance)
        station = line.project(poly.centroid)
        a, b = (line.interpolate(station + d) for d in (-min(station, 0.5), 0.5))
        tangent = np.array([b.x - a.x, b.y - a.y])
        if np.linalg.norm(tangent):
            angle = np.arccos(
                np.clip(
                    abs(edges[sides.argmax()] @ tangent) / (length * np.linalg.norm(tangent)), 0, 1
                )
            )
    return np.array(
        [
            *np.log([poly.area, width, length]),
            poly.area / poly.convex_hull.area,
            len(poly.interiors),
            angle,
            np.log(len(poly.exterior.coords) - 1),  # corner count: bars 4, chevrons ~6
        ]
    )


class ShapeClassifier:
    """k-nearest client shapes. Examples are balanced per code (at most ``per_code``), so a
    very common type (hatch bars) does not swamp rare ones (colour bands, chevrons)."""

    def __init__(self, examples, per_code=150, seed=0):
        rng = np.random.default_rng(seed)
        by_code = {}
        for vec, code in examples:
            by_code.setdefault(code[0], []).append((vec, code))
        examples = []
        for code in sorted(by_code, key=str):
            group = by_code[code]
            keep = rng.choice(len(group), size=min(per_code, len(group)), replace=False)
            examples += [group[k] for k in sorted(keep)]
        self.codes = [code for _, code in examples]
        self.jae = {}
        for rst, jae in self.codes:
            if rst is not None and jae is not None:
                self.jae.setdefault(rst, jae)
        self.tree = None
        if len(examples) >= 2:
            x = np.asarray([vec for vec, _ in examples])
            self.scale = np.maximum(x.std(axis=0), 0.1)
            self.tree = cKDTree(x / self.scale)
            self.limit = np.percentile(self.tree.query(x / self.scale, k=2)[0][:, 1], 95)

    def predict(self, vec):
        if self.tree is None:
            return (None, None), 0.0
        distances, ids = self.tree.query(vec / self.scale, k=min(5, len(self.codes)))
        if distances[0] > 2 * self.limit:
            return (None, None), 0.0
        votes = {}
        for distance, idx in zip(distances, ids, strict=True):
            code = self.codes[idx]
            votes[code] = votes.get(code, 0.0) + 1 / max(0.01, distance)
        code = max(votes, key=votes.get)
        return code, votes[code] / sum(votes.values())


def components(mask, grid, minimum):
    for geom, _ in shapes(mask.astype("uint8"), mask=mask, transform=grid.affine):
        poly = shape(geom).simplify(0.025, preserve_topology=True)
        if poly.is_valid and poly.area >= minimum:
            yield poly


def lift_polygon(poly, surface, step=0.5):
    """Z for the polygon vertices from a plane fitted to the road surface under the mark (a
    painted mark is flat; corners on the paint edge need not have their own surface fit).
    Returns (Polygon Z or None, share of supported samples)."""
    ring = np.asarray(poly.exterior.coords)
    samples = shapely.get_coordinates(shapely.segmentize(poly.exterior, step))
    inner = poly.buffer(-0.1)
    if not inner.is_empty:
        samples = np.vstack([samples, shapely.get_coordinates(inner.representative_point())])
    recs = [surface.z_at(x, y) for x, y in samples]
    ok = np.array([r["ok"] for r in recs])
    if ok.sum() < 3:
        return None, float(ok.mean())
    a = np.column_stack([np.ones(ok.sum()), samples[ok]])
    coef, *_ = np.linalg.lstsq(a, np.array([r["z"] for r in recs])[ok], rcond=None)
    z = coef[0] + ring[:, :2] @ coef[1:]
    holes = [
        np.column_stack(
            [np.asarray(h.coords)[:, :2], coef[0] + np.asarray(h.coords)[:, :2] @ coef[1:]]
        )
        for h in poly.interiors
    ]
    return Polygon(np.column_stack([ring[:, :2], z]), holes), float(ok.mean())


def smoothed_paint(gray, observed, sigma_px=1.5, delta=20, window=81):
    """Paint mask for area marks. Faint marks scanned with sparse points look like speckle, so
    intensity is first averaged over neighbouring returns (normalised Gaussian, NoData
    ignored) and then compared with the local road background."""
    from scipy.ndimage import gaussian_filter

    o = observed.astype(np.float32)
    num = gaussian_filter(gray.astype(np.float32) * o, sigma_px)
    den = gaussian_filter(o, sigma_px)
    support = den > 0.25
    smooth = np.where(support, num / np.maximum(den, 1e-3), 0).astype(np.float32)
    return bright_paint(
        np.clip(smooth, 0, 255).astype(np.uint8), support, delta=delta, window=window
    )


def thin_line_paint(
    mask, lines, grid, step=0.1, half=0.6, max_width=0.4, buffer_m=0.15, min_wide_m=1.2
):
    """Pixels of lane-line paint to remove before finding area marks: only where the paint
    across the line is line-thin. Where a line meets or runs along a wider mark (hatch bar,
    band, chevron) the paint across it is wide, nothing is removed and the mark stays whole."""
    from .raster_tiles import xy_to_pixel

    offsets = np.arange(-half, half + 1e-9, 0.025)
    keep = []
    for g in lines:
        if g.length < step:
            continue
        s = np.arange(0, g.length, step)
        pts = shapely.get_coordinates(shapely.line_interpolate_point(g, s))
        nxt = shapely.get_coordinates(
            shapely.line_interpolate_point(g, np.minimum(s + 0.05, g.length))
        )
        prv = shapely.get_coordinates(shapely.line_interpolate_point(g, np.maximum(s - 0.05, 0)))
        t = nxt - prv
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        n = np.column_stack([-t[:, 1], t[:, 0]])
        xy = pts[:, None, :] + offsets[None, :, None] * n[:, None, :]
        r, c = xy_to_pixel(grid, xy[..., 0], xy[..., 1])
        r, c = np.rint(r).astype(int), np.rint(c).astype(int)
        ok = (r >= 0) & (c >= 0) & (r < mask.shape[0]) & (c < mask.shape[1])
        across = np.zeros(r.shape, bool)
        across[ok] = mask[r[ok], c[ok]]
        centre = len(offsets) // 2
        thin = np.zeros(len(s), bool)
        for i in range(len(s)):
            row = across[i]
            if not row[centre - 4 : centre + 5].any():
                thin[i] = True  # no paint here: nothing to protect
                continue
            # contiguous paint run through the line axis
            lo = hi = centre
            while lo > 0 and row[lo - 1]:
                lo -= 1
            while hi < len(row) - 1 and row[hi + 1]:
                hi += 1
            thin[i] = (hi - lo + 1) * 0.025 <= max_width
        # A mark only touching or crossing the line makes it wide for a short stretch: the line
        # is still removed there (the mark is clipped at the line edge). Only a long wide
        # stretch (the "line" runs along a mark) is kept.
        wide_runs = np.flatnonzero(~thin)
        for run in np.split(wide_runs, np.flatnonzero(np.diff(wide_runs) > 1) + 1):
            if len(run) and len(run) * step < min_wide_m:
                thin[run] = True
        idx = np.flatnonzero(thin)
        if not len(idx):
            continue
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        for run in runs:
            a, b = s[run[0]], min(s[run[-1]] + step, g.length)
            if b - a >= step:
                keep.append(shapely.ops.substring(g, a, b).buffer(buffer_m))
    if not keep:
        return np.zeros(mask.shape, bool)
    return rasterize(keep, out_shape=mask.shape, transform=grid.affine).astype(bool)


def fewest_corners(convex, min_iou=0.93):
    """Convex outer shape with the fewest corners (3, 4, ...) still covering it by
    ``min_iou`` (a give-way triangle has a sharp tip that the edge-distance fit over-splits)."""
    from .polygon_fit import _reduce, iou

    simple = convex.simplify(0.02)
    candidates = _reduce(np.asarray(simple.exterior.coords)[:-1])
    for k in sorted(candidates):
        poly = Polygon(candidates[k])
        if poly.is_valid and iou(poly, convex) >= min_iou:
            return poly
    return simple


def outline_coverage(blob, shape=None, step=0.05, near_m=0.15):
    """Share of the outer shape's boundary (default: the blob's convex hull) lying on paint
    (within ``near_m`` of the blob). High for outline marks such as give-way triangles."""
    shape = blob.convex_hull if shape is None else shape
    pts = shapely.points(shapely.get_coordinates(shapely.segmentize(shape.exterior, step)))
    return float(np.mean(shapely.distance(pts, blob) <= near_m))


def fit_outline(blob, paint_blob=None):
    """Recover sharp triangle corners when the enclosing edges follow the paint."""
    import cv2

    from .polygon_fit import iou

    hull = blob.convex_hull
    coords = np.asarray(fewest_corners(hull).exterior.coords)[:-1]
    # Work in local coordinates to retain centimetre precision in OpenCV's float32 input.
    origin = coords[0]
    _, vertices = cv2.minEnclosingTriangle((coords - origin).astype(np.float32))
    if vertices is not None:
        triangle = Polygon(vertices.reshape(3, 2) + origin)
        if triangle.is_valid and iou(triangle, hull) >= 0.8:
            # Close small scan gaps before finding the unpainted interior.
            closed = (blob if paint_blob is None else paint_blob).buffer(0.10).buffer(-0.10)
            empty = triangle.buffer(-0.12).difference(closed).buffer(-0.15).buffer(0.15)
            parts = list(empty.geoms) if hasattr(empty, "geoms") else [empty]
            parts = [p for p in parts if p.geom_type == "Polygon" and p.area >= 0.1]
            if parts:
                interior = max(parts, key=lambda p: p.area)
                # The inner border follows the same three directions as the outer border.
                outer = np.asarray(triangle.exterior.coords)[:3] - origin
                edges = np.roll(outer, -1, axis=0) - outer
                normals = np.column_stack([-edges[:, 1], edges[:, 0]])
                normals /= np.linalg.norm(normals, axis=1)[:, None]
                center = outer.mean(axis=0)
                normals *= np.where(np.sum(normals * (center - outer), axis=1) > 0, 1, -1)[:, None]
                points = (
                    shapely.get_coordinates(shapely.segmentize(interior.exterior, 0.025)) - origin
                )
                offsets = np.quantile(points @ normals.T, 0.08, axis=0)
                corners = [
                    np.linalg.solve(normals[[k - 1, k]], offsets[[k - 1, k]]) for k in range(3)
                ]
                hole = Polygon(np.asarray(corners) + origin)
                if (
                    hole.is_valid
                    and triangle.contains(hole)
                    and 0.1 < hole.area < 0.85 * triangle.area
                ):
                    ring = Polygon(triangle.exterior.coords, [hole.exterior.coords])
                    if ring.is_valid:
                        return ring
            return triangle
    return None


def split_touching_bars(blob, min_parts=3, steps=(0.05, 0.08, 0.11, 0.14, 0.17), min_area=0.05):
    """Hatch bars or chevrons that touch each other (at a gore tip) or touch an untraced border
    line form one hollow, comb-shaped blob. Shrink the blob until it falls apart into at least
    ``min_parts`` pieces (line-thin border paint disappears first), then grow each piece back
    inside the blob. Returns the separate bars, or None when the blob is one mark."""
    for d in steps:
        core = blob.buffer(-d)
        parts = [g for g in getattr(core, "geoms", [core]) if g.area >= min_area]
        if len(parts) < min_parts:
            continue
        bars, taken = [], None
        for g in sorted(parts, key=lambda g: -g.area):
            bar = g.buffer(d, join_style="mitre").intersection(blob)
            if taken is not None:
                bar = bar.difference(taken)
            bar = max(getattr(bar, "geoms", [bar]), key=lambda h: h.area, default=None)
            if bar is None or bar.is_empty or bar.geom_type != "Polygon":
                continue
            bars.append(bar)
            taken = bar if taken is None else taken.union(bar)
        if len(bars) >= min_parts and _parallel_bars(bars, min_parts):
            return bars
    return None


def _parallel_bars(bars, min_parts, min_area=0.25, min_length=0.8, max_spread_deg=25.0):
    """Hatch bars / chevrons: at least ``min_parts`` sizeable, elongated pieces, all pointing
    the same way. Speckle broken into blobs is not."""
    from .polygon_fit import principal_frame

    angles = []
    for bar in bars:
        _, axis, length, _ = principal_frame(bar)
        if bar.area < min_area or length < min_length:
            return False
        angles.append(np.degrees(np.arctan2(axis[1], axis[0])) % 180.0)
    a = np.radians(np.asarray(angles) * 2.0)  # axial mean (0 and 180 deg are the same)
    mean = np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) / 2.0
    spread = np.abs((np.asarray(angles) - mean + 90.0) % 180.0 - 90.0)
    return len(bars) >= min_parts and float(spread.max()) <= max_spread_deg


def is_true_outline(poly, paint, observed, grid, ring_min=0.6, hole_max=0.25):
    """An outline mark (give-way triangle) is a painted border around an unpainted interior.
    A hatch zone also looks hollow from outside, but its interior holds the painted bars."""
    outer = Polygon(poly.exterior)
    border = paint_density(poly, paint, observed, grid) if poly.interiors else None
    inner = (
        Polygon(poly.interiors[0])
        if poly.interiors
        else outer.buffer(-0.35) if not outer.buffer(-0.35).is_empty else None
    )
    if inner is None or inner.is_empty:
        return True
    if paint_density(inner, paint, observed, grid) > hole_max:
        return False
    return border is None or border >= ring_min


def paint_density(poly, paint, observed, grid):
    """Share of the scanned cells inside ``poly`` that are paint. Real marks are solid paint
    (>= 0.8 on the review areas); blotches of speckle on sparse shoulders are not."""
    inside = rasterize([poly], out_shape=paint.shape, transform=grid.affine).astype(bool)
    seen = inside & observed
    return float(paint[seen].mean()) if seen.any() else 0.0


def extract_other(
    gray,
    observed,
    grid,
    lines,
    surface,
    classifier,
    library=None,
    max_line_distance_m=4.0,
    min_density=0.72,
):
    """Painted area marks as clean closed polygons.

    Pass 1 matches the client's fixed shapes (arrows, ``library``) on the full paint: arrow
    shafts are often read as lane lines, so lines are not removed there, and stronger closing
    makes sparsely scanned paint solid. A good match gives the client's clean shape and code.
    Pass 2 fits the remaining paint (lane lines and fitted arrows removed) with the fewest
    straight edges (``fit_straight``); its type comes from ``classifier``.
    """
    from .polygon_fit import fit_straight, fit_template, principal_frame

    paint = bright_paint(gray, observed, delta=25, window=81)
    rows, claimed = [], []
    # Every road mark lies in a lane: far from all lane lines of its level (verges, roofs,
    # canopies picked up as a deck) nothing is accepted.
    if max_line_distance_m is not None:
        if not lines:
            return []
        near = rasterize(
            [g.buffer(max_line_distance_m) for g in lines],
            out_shape=paint.shape,
            transform=grid.affine,
        ).astype(bool)
        paint &= near
    if library is not None and len(library):
        smooth = smoothed_paint(gray, observed)
        for blob in components(binary_closing(smooth | paint, iterations=3), grid, 0.5):
            _, _, length, width = principal_frame(blob)
            # an arrow has a head: a plain line (0.15-0.3 m wide) is never an arrow
            if not (3.0 <= length <= 10.5 and width >= 0.5):
                continue
            center = blob.representative_point()
            if not surface.z_at(center.x, center.y)["ok"]:
                continue
            poly, code, fit_iou = fit_template(blob, library, min_iou=0.75)
            if poly is not None:
                claimed.append(poly)
                rows.append(_row(poly, code, fit_iou, "client_shape", surface, classifier, lines))
    mask = binary_closing(paint, iterations=1)  # 5 cm: bands 10 cm apart stay separate
    removed = [p.buffer(0.1) for p in claimed]
    if removed:
        mask &= ~rasterize(removed, out_shape=mask.shape, transform=grid.affine).astype(bool)
    mask &= ~thin_line_paint(mask, lines, grid)
    # opening removes 1-2 px spurs where scan speckle touches a mark (spiky corners)
    mask = binary_opening(mask, iterations=1)
    work = list(components(mask, grid, 0.1))
    while work:
        blob = work.pop(0)
        vec = descriptor(blob, lines)
        _, width, length = np.exp(vec[:3])
        # solidity of the filled outline: a frame (ring) is hollow but still a compact mark
        filled = Polygon(blob.exterior)
        if width > 4 or length > 15:
            continue
        outline_mark = False
        if filled.area / filled.convex_hull.area < 0.45 and outline_coverage(blob) >= 0.8:
            # An outline mark (give-way triangle drawn as a painted border, possibly with a small
            # gap) is hollow and open. Recover both boundaries of its painted border.
            # A concave blob whose border is not painted all round (a V-shaped chevron, a bent
            # bar) is a solid mark: it is fitted with straight edges below and must pass the
            # same paint-density test as any other mark.
            outline_mark = True
        if road_share(blob, surface) < 0.7:  # grass verges, structures, gantries
            continue
        if outline_mark:
            inside = rasterize(
                [blob.convex_hull], out_shape=paint.shape, transform=grid.affine
            ).astype(bool)
            full_paint = shapely.union_all(list(components(paint & inside, grid, 0.002)))
            poly = fit_outline(blob, full_paint)
            if poly is not None and not is_true_outline(poly, paint, observed, grid):
                # a hatch zone (bars touching each other / an untraced border): one polygon
                # per bar, never an outline around the zone
                bars = split_touching_bars(blob)
                if bars:
                    work.extend(bars)
                    continue
            if poly is None:
                continue
            fit_iou = outline_coverage(blob, poly)
        else:
            poly, _, fit_iou = fit_straight(blob)
            if paint_density(poly, paint, observed, grid) < min_density:
                # touching bars fitted as one shape enclose unpainted gaps: split into bars
                bars = split_touching_bars(blob)
                work.extend(bars or [])
                continue  # speckle on sparsely scanned shoulders, not solid paint
        row = _row(poly, None, fit_iou, "straight_edges", surface, classifier, lines)
        if row is not None and outline_mark:
            row["shape_method"] = "outline_mark"
            row["review_flags"] = ";".join(x for x in [row["review_flags"], "outline_mark"] if x)
        rows.append(row)
    rows = [r for r in rows if r is not None]
    group_vote(rows)
    return [r for r in rows if not (r["COD_RST"] is None and r["AREA_PINTA"] < 0.5)]


def group_vote(rows, gap_m=2.5, size_ratio=2.0):
    """Marks of one series (a row of chevrons, a hatched area, colour bands) share one type:
    free shapes within ``gap_m`` of each other and of similar size take the confidence-weighted
    majority code of their group. Fitted client shapes (arrows) keep their own code."""
    free = [k for k, r in enumerate(rows) if r["shape_method"] == "straight_edges"]
    if len(free) < 2:
        return
    geoms = [rows[k]["geometry"] for k in free]
    tree = shapely.STRtree(geoms)
    parent = list(range(len(free)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, g in enumerate(geoms):
        for j in tree.query(g, predicate="dwithin", distance=gap_m):
            if j > i and max(g.area, geoms[j].area) <= size_ratio * min(g.area, geoms[j].area):
                parent[root(j)] = root(i)
    groups = {}
    for i in range(len(free)):
        groups.setdefault(root(i), []).append(free[i])
    for members in groups.values():
        if len(members) < 3:
            continue
        votes = {}
        for k in members:
            r = rows[k]
            if r["COD_RST"] is not None:
                key = (r["COD_RST"], r["COD_JAE"])
                votes[key] = votes.get(key, 0.0) + max(r["confidence"], 0.05)
        if not votes:
            continue
        (rst, jae), weight = max(votes.items(), key=lambda kv: kv[1])
        if weight < 0.5 * sum(votes.values()):
            continue
        for k in members:
            r = rows[k]
            if r["COD_RST"] != rst:
                r["review_flags"] = ";".join(
                    x for x in [r["review_flags"], "type_from_neighbouring_marks"] if x
                )
            r["COD_RST"], r["COD_JAE"] = rst, jae
            r["review_flags"] = ";".join(
                x for x in r["review_flags"].split(";") if x and x != "unclassified_paint"
            )


def road_share(poly, surface, spacing=0.3, max_samples=15):
    """Share of sample points inside ``poly`` with a smooth road surface under them."""
    x0, y0, x1, y1 = poly.bounds
    gx, gy = np.meshgrid(np.arange(x0, x1, spacing), np.arange(y0, y1, spacing))
    pts = shapely.points(gx.ravel(), gy.ravel())
    inside = shapely.get_coordinates(pts[shapely.contains(poly, pts)])
    if not len(inside):
        inside = shapely.get_coordinates(poly.representative_point())
    if len(inside) > max_samples:
        inside = inside[np.linspace(0, len(inside) - 1, max_samples).astype(int)]
    return float(np.mean([surface.z_at(x, y)["ok"] for x, y in inside]))


def _row(poly, code, fit_iou, method, surface, classifier, lines):
    geom3d, support = lift_polygon(poly, surface)
    if support < 0.5:
        return None
    if code is not None:
        rst, jae, confidence = code, classifier.jae.get(code), float(fit_iou)
    else:
        (rst, jae), confidence = classifier.predict(descriptor(poly, lines))
    flags = []
    if rst is None:
        flags.append("unclassified_paint")
    if geom3d is None:
        flags.append("z_unsupported")
    if fit_iou < 0.85:
        flags.append("loose_shape_fit")
    return dict(
        geometry=poly,
        geometry_3d=geom3d,
        COD_RST=rst,
        COD_JAE=jae,
        confidence=confidence,
        shape_method=method,
        shape_iou=round(float(fit_iou), 3),
        vertices=len(poly.exterior.coords) - 1,
        AREA_PINTA=round(poly.area, 3),
        review_state="PENDING_REVIEW",
        review_flags=";".join(flags),
    )


def score_other(rows, reference, observed_area):
    truth = [
        (g, code) for g, code in reference if g.intersection(observed_area).area >= 0.5 * g.area
    ]
    predictions = [
        (row["geometry"], row["COD_RST"])
        for row in rows
        if row["geometry"].intersection(observed_area).area >= 0.5 * row["geometry"].area
    ]
    tree = shapely.STRtree([g for g, _ in truth])
    pairs = []
    for i, (poly, _) in enumerate(predictions):
        for j in tree.query(poly, predicate="intersects"):
            ref = truth[j][0]
            iou = poly.intersection(ref).area / poly.union(ref).area
            if iou >= 0.5:
                pairs.append((iou, i, j))
    matched, used, correct, coded = set(), set(), 0, 0
    for _, i, j in sorted(pairs, reverse=True):
        if i in matched or j in used:
            continue
        matched.add(i)
        used.add(j)
        if truth[j][1] is not None:
            coded += 1
            correct += predictions[i][1] == truth[j][1]
    return dict(
        scanned=len(truth),
        scored_predictions=len(predictions),
        matched=len(matched),
        recall=len(matched) / len(truth) if truth else None,
        precision=len(matched) / len(predictions) if predictions else None,
        class_accuracy=correct / coded if coded else None,
    )
