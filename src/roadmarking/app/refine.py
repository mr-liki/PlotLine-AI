"""Fit each area-mark polygon to its own paint.

The mark extraction picks the shape (a client template or the fewest straight edges); its
position, angle and size can still be a few centimetres off the painted mark (worn or
differently proportioned marks). Here the polygon is moved, turned and stretched slightly
(a similarity transform plus a small stretch along and across its axis) to maximise its overlap
(IoU) with the mark's paint, measured on the same paint raster the extraction uses. The shape
itself (corners, template) is kept; frames with holes are not changed.
"""

from __future__ import annotations

import numpy as np


def _paint_window(poly, paint, grid, pad_m=0.6, exclude=(), fill=True):
    """Paint pixels of the mark: the connected paint component(s) overlapping ``poly`` in a
    window around it (other marks or lines nearby are not counted)."""
    from rasterio.features import rasterize
    from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, label

    from roadmarking.digitization.raster_tiles import xy_to_pixel

    x0, y0, x1, y1 = poly.bounds
    r0, c0 = xy_to_pixel(grid, np.array([x0 - pad_m]), np.array([y1 + pad_m]))
    r1, c1 = xy_to_pixel(grid, np.array([x1 + pad_m]), np.array([y0 - pad_m]))
    r0, c0 = max(int(r0[0]), 0), max(int(c0[0]), 0)
    r1, c1 = min(int(r1[0]) + 2, paint.shape[0]), min(int(c1[0]) + 2, paint.shape[1])
    if r1 - r0 < 4 or c1 - c0 < 4:
        return None
    win = paint[r0:r1, c0:c1].copy()
    from affine import Affine

    tr = grid.affine * Affine.translation(c0, r0)
    if exclude:  # lane-line paint next to the mark is not part of it
        win &= ~rasterize(
            [g.buffer(0.12) for g in exclude], out_shape=win.shape, transform=tr
        ).astype(bool)
    inside = rasterize([poly], out_shape=win.shape, transform=tr).astype(bool)
    lab, _ = label(win)
    ids = np.unique(lab[inside & win])
    ids = ids[ids > 0]
    mark = np.isin(lab, ids) & binary_dilation(inside, iterations=int(0.4 / grid.cell))
    # the painted shape, not its speckle: close small gaps between returns and fill holes
    # (a frame's hole is kept: frames are not refitted)
    mark = binary_closing(mark, iterations=2)
    if fill:
        mark = binary_fill_holes(mark)
    return mark, tr


def paint_iou(poly, paint, grid, lines=()):
    """IoU of ``poly`` with its own painted shape (frames keep their unpainted middle)."""
    near = [ln for ln in lines if ln.distance(poly) < 0.8]
    got = _paint_window(poly, paint, grid, exclude=near, fill=not poly.interiors)
    if got is None or got[0].sum() < 20:
        return None
    return float(_iou(poly, *got))


def _iou(poly, mark, tr):
    from rasterio.features import rasterize

    if poly.is_empty or not poly.is_valid:
        return 0.0
    m = rasterize([poly], out_shape=mark.shape, transform=tr).astype(bool)
    inter = (m & mark).sum()
    union = (m | mark).sum()
    return inter / union if union else 0.0


def fit_to_paint(
    poly, paint, grid, lines=(), max_shift=0.2, max_rot_deg=4.0, max_scale=0.08, min_gain=0.01
):
    """``poly`` moved/turned/stretched to best overlap its paint. Returns (polygon, iou before,
    iou after); the input polygon when nothing improves the overlap by ``min_gain``."""
    from shapely import affinity

    from roadmarking.digitization.polygon_fit import principal_frame

    if poly.interiors or poly.area < 0.2:
        return poly, None, None
    near = [ln for ln in lines if ln.distance(poly) < 0.8]
    got = _paint_window(poly, paint, grid, exclude=near)
    if got is None:
        return poly, None, None
    mark, tr = got
    if mark.sum() < 20:
        return poly, None, None
    c, axis, _, _ = principal_frame(poly)
    ang = np.degrees(np.arctan2(axis[1], axis[0]))
    base = _iou(poly, mark, tr)

    def make(p):
        dx, dy, rot, sa, sc = p
        g = affinity.rotate(poly, -ang, origin=tuple(c))  # axis along x
        g = affinity.scale(g, 1 + sa, 1 + sc, origin=tuple(c))
        g = affinity.rotate(g, ang + rot, origin=tuple(c))
        return affinity.translate(g, dx, dy)

    lim = np.array([max_shift, max_shift, max_rot_deg, max_scale, max_scale])

    def cost(p):
        if np.any(np.abs(p) > lim):
            return 1.0
        return 1.0 - _iou(make(p), mark, tr)

    # coordinate search (the pixel IoU is piecewise constant: no gradients), coarse to fine
    p = np.zeros(5)
    cur = cost(p)
    steps = np.array([0.04, 0.04, 1.0, 0.02, 0.02])
    for _ in range(4):
        improved = True
        while improved:
            improved = False
            for k in range(5):
                for sgn in (1, -1):
                    q = p.copy()
                    q[k] += sgn * steps[k]
                    c_ = cost(q)
                    if c_ < cur - 1e-6:
                        p, cur, improved = q, c_, True
        steps /= 2
    after = 1.0 - cur
    if after < base + min_gain:
        return poly, base, base
    return make(p), base, after


def _edge_offset(a, b, outward, gray, observed, grid, reach=0.3, res=0.02, trim=0.1):
    """Offset (m, along ``outward``) of the paint boundary from edge a-b: the half-brightness
    crossing of the cross-edge profile averaged along the whole edge. None if unclear."""
    from scipy.ndimage import map_coordinates

    from roadmarking.digitization.raster_tiles import xy_to_pixel

    length = float(np.linalg.norm(b - a))
    if length < 0.3:
        return None
    t = (b - a) / length
    offs = np.arange(-reach, reach + 1e-9, res)
    st = np.arange(trim, length - trim + 1e-9, 0.05)
    if not len(st):
        return None
    pts = a[None, None, :] + st[:, None, None] * t + offs[None, :, None] * outward
    r, c = xy_to_pixel(grid, pts[..., 0].ravel(), pts[..., 1].ravel())
    inside = (r >= 0) & (c >= 0) & (r <= gray.shape[0] - 1) & (c <= gray.shape[1] - 1)
    ri, ci = np.rint(r).astype(int), np.rint(c).astype(int)
    ok = inside.copy()
    ok[inside] = observed[ri[inside], ci[inside]]
    v = np.full(r.shape, np.nan)
    v[ok] = map_coordinates(gray.astype(np.float32), [r[ok], c[ok]], order=1)
    v = v.reshape(len(st), len(offs))
    cnt = np.isfinite(v).sum(axis=0)
    if (cnt > 0).mean() < 0.8:
        return None
    prof = np.where(cnt > 0, np.nansum(v, axis=0) / np.maximum(cnt, 1), np.nan)
    good = np.isfinite(prof)
    prof = np.interp(offs, offs[good], prof[good])
    paint = np.percentile(prof[offs < -0.02], 75)  # inside the mark
    road = np.percentile(prof[offs > 0.02], 25)  # outside
    if paint - road < 15:
        return None
    half = 0.5 * (paint + road)
    # the crossing nearest to the current edge
    s = np.sign(prof - half)
    idx = np.flatnonzero(s[:-1] * s[1:] < 0)
    if not len(idx):
        return None
    i = idx[np.argmin(np.abs(offs[idx]))]
    f = (half - prof[i]) / (prof[i + 1] - prof[i])
    return float(offs[i] + f * res)


def _snap_ring(coords, is_hole, gray, observed, grid, max_move):
    """Move every edge of a closed ring onto the paint boundary; corners rebuilt."""
    xy = np.asarray(coords)[:-1, :2]
    n = len(xy)
    if n < 3:
        return None, 0
    # orientation: outward normal points away from the mark's paint
    area2 = np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1])
    ccw = area2 > 0
    lines, moved = [], 0
    for i in range(n):
        a, b = xy[i], xy[(i + 1) % n]
        t = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        right = np.array([t[1], -t[0]])  # outward of a CCW exterior ring
        out = right if ccw else -right
        if is_hole:
            out = -out  # a hole's paint is outside the hole ring
        o = _edge_offset(a, b, out, gray, observed, grid)
        if o is None or abs(o) > max_move:
            o = 0.0
        else:
            moved += 1
        lines.append((a + o * out, t))
    new = []
    for i in range(n):
        p1, d1 = lines[i - 1]
        p2, d2 = lines[i]
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) < np.sin(np.radians(8)):  # nearly straight: keep the shifted vertex
            new.append(0.5 * (p1 + d1 * np.dot(xy[i] - p1, d1) + p2 + d2 * np.dot(xy[i] - p2, d2)))
            continue
        s = ((p2 - p1)[0] * d2[1] - (p2 - p1)[1] * d2[0]) / cross
        q = p1 + s * d1
        if np.linalg.norm(q - xy[i]) > 3 * max_move:  # runaway corner of a thin spike
            q = xy[i]
        new.append(q)
    return np.asarray(new), moved


def snap_edges(poly, gray, observed, grid, max_move=0.15, max_area_change=0.4):
    """Every straight edge of ``poly`` (outline and holes) moved onto the paint boundary
    (half-brightness edge measured along the whole edge), corners rebuilt where neighbouring
    edges meet. Returns (polygon, edges moved); the input when the result is not sound."""
    from shapely.geometry import Polygon

    ext, m0 = _snap_ring(poly.exterior.coords, False, gray, observed, grid, max_move)
    if ext is None:
        return poly, 0
    # corners the shape lacks: split edges where the painted outline bends, snap again
    bent = split_bent_edges(ext, False, gray, observed, grid)
    if len(bent) > len(ext):
        again, m1 = _snap_ring(np.vstack([bent, bent[:1]]), False, gray, observed, grid, max_move)
        if again is not None:
            ext, m0 = again, m0 + m1
    holes, moved = [], m0
    for h in poly.interiors:
        hr, m = _snap_ring(h.coords, True, gray, observed, grid, max_move)
        if hr is not None:
            holes.append(hr)
            moved += m
    new = Polygon(ext, holes)
    if not new.is_valid or new.is_empty:
        new = new.buffer(0)
        if new.geom_type != "Polygon":
            return poly, 0
    if abs(new.area - poly.area) > max_area_change * poly.area or moved == 0:
        return poly, 0
    return new, moved


def retemplate(poly, paint, grid, lines, library, min_iou=0.75):
    """A mark touching a lane line merges with the line's paint and misses the client-shape
    match (the blob is too long for an arrow); it is then fitted with generic straight edges
    (no barb, a lobe into the line). Retry the client shapes on the mark's own paint with the
    lane-line paint removed. Returns (polygon, code, template IoU) or None."""
    from rasterio.features import shapes
    from shapely.geometry import shape

    from roadmarking.digitization.polygon_fit import fit_template

    if library is None or not len(library) or poly.interiors:
        return None
    near = [ln for ln in lines if ln.distance(poly) < 0.8]
    got = _paint_window(poly, paint, grid, exclude=near)
    if got is None or got[0].sum() < 50:
        return None
    mark, tr = got
    blobs = [shape(g) for g, _ in shapes(mark.astype(np.uint8), mask=mark, transform=tr)]
    if not blobs:
        return None
    blob = max(blobs, key=lambda b: b.area).buffer(0)
    if blob.geom_type != "Polygon":
        return None
    tpl, code, iou = fit_template(blob, library, min_iou=min_iou)
    if tpl is None:
        return None
    return tpl, code, float(iou)


def split_bent_edges(xy, is_hole, gray, observed, grid, min_len=0.8, bend=0.04):
    """Insert a corner in the middle of every edge whose two halves sit at different distances
    from the paint boundary (the painted outline bends there: a corner the shape lacks)."""
    n = len(xy)
    area2 = np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1])
    ccw = area2 > 0
    out = []
    for i in range(n):
        a, b = xy[i], xy[(i + 1) % n]
        out.append(a)
        length = np.linalg.norm(b - a)
        if length < min_len:
            continue
        t = (b - a) / length
        right = np.array([t[1], -t[0]])
        o = right if ccw else -right
        if is_hole:
            o = -o
        m = 0.5 * (a + b)
        o1 = _edge_offset(a, m, o, gray, observed, grid)
        o2 = _edge_offset(m, b, o, gray, observed, grid)
        if o1 is not None and o2 is not None and abs(o1 - o2) > bend:
            out.append(m + 0.5 * (o1 + o2) * o)
    return np.asarray(out)


def prune_corners(poly, paint, grid, lines=(), tol=0.004, min_vertices=3):
    """Remove corners that do not help the outline match the paint: repeatedly drop the corner
    whose removal changes the overlap (IoU) with the mark's painted shape the least, while it
    drops by no more than ``tol``. Spurs, hooks and notches at the ends of a bar go; the real
    corners of a shape (an arrow's barb) stay, removing them costs paint. Frames unchanged."""
    from shapely.geometry import Polygon

    if poly.interiors:
        return poly
    near = [ln for ln in lines if ln.distance(poly) < 0.8]
    got = _paint_window(poly, paint, grid, exclude=near)
    if got is None or got[0].sum() < 20:
        return poly
    mark, tr = got
    xy = np.asarray(poly.exterior.coords)[:-1, :2]
    cur = _iou(poly, mark, tr)
    while len(xy) > min_vertices:
        best = None
        for i in range(len(xy)):
            cand = Polygon(np.delete(xy, i, axis=0))
            if not cand.is_valid:
                continue
            v = _iou(cand, mark, tr)
            if best is None or v > best[0]:
                best = (v, i)
        if best is None or best[0] < cur - tol:
            break
        xy = np.delete(xy, best[1], axis=0)
        cur = max(cur, best[0])
    out = Polygon(xy)
    return out if out.is_valid and len(xy) < len(poly.exterior.coords) - 1 else poly


def trace_paint_outline(
    poly, paint, grid, lines=(), gray=None, observed=None, tol_iou=0.01, max_corners=14
):
    """The mark's outline traced from its paint: the painted area (lane-line paint removed,
    scan gaps closed) is outlined and simplified to candidate polygons with 3..``max_corners``
    corners; each candidate's edges are snapped onto the paint boundary (sub-pixel), and the
    fewest corners within ``tol_iou`` of the best overlap with the paint win.
    Returns (polygon, IoU) or None."""
    import cv2
    from rasterio.features import shapes
    from shapely.geometry import Polygon, shape

    if poly.interiors:
        return None
    near = [ln for ln in lines if ln.distance(poly) < 0.8]
    got = _paint_window(poly, paint, grid, exclude=near)
    if got is None or got[0].sum() < 30:
        return None
    mark, tr = got
    blobs = [shape(g) for g, _ in shapes(mark.astype(np.uint8), mask=mark, transform=tr)]
    blobs = [b for b in blobs if b.intersects(poly)]
    if not blobs:
        return None
    blob = max(blobs, key=lambda b: b.intersection(poly).area)
    ring = np.asarray(blob.exterior.coords)[:-1, :2]
    origin = ring.mean(axis=0)
    pts = (ring - origin).astype(np.float32).reshape(-1, 1, 2)
    cands = {}
    for eps in np.arange(0.02, 0.30, 0.005):
        approx = cv2.approxPolyDP(pts, float(eps), True).reshape(-1, 2)
        n = len(approx)
        if 3 <= n <= max_corners and n not in cands:
            cand = Polygon(approx.astype(float) + origin)
            if cand.is_valid and cand.area >= 0.05:
                cands[n] = cand
    scored = []
    for n, cand in cands.items():
        if gray is not None:
            cand, _ = snap_edges(cand, gray, observed, grid)
        scored.append((n, cand, _iou(cand, mark, tr)))
    if not scored:
        return None
    top = max(v for _, _, v in scored)
    n, cand, v = min((s_ for s_ in scored if s_[2] >= top - tol_iou), key=lambda s_: s_[0])
    return cand, v
