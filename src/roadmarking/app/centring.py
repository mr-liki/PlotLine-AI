"""Put every line on the exact centre of its paint stripe.

The tracing fits lines with a smoothing window (stable, but on curves it cuts the inside and on
wide or faint stripes it can sit near one edge). Here each line is moved, station by station,
onto the middle of the paint across it, measured on the intensity raster:

- every ``step`` m the brightness profile across the line is sampled (+- ``reach`` m) and
  averaged with its neighbours over +- 1.5 m (fills the scan-line gaps of one cross-section);
- the stripe is the bright run containing the brightest sample near the current axis; its
  centre is the midpoint of its two half-brightness edges (sub-pixel);
- the measured offsets are smoothed with a robust local linear fit over +- ``window`` m (the
  tracing's error changes slowly along the line, noise does not), so the line follows the
  paint on curves without picking up wobble; stations without a clear stripe take the offset
  interpolated from their neighbours;
- ends that meet another line (junctions) stay attached: the shift fades out towards them.
"""

from __future__ import annotations

import numpy as np


def stripe_offsets(
    line,
    gray,
    observed,
    grid,
    step=0.25,
    reach=0.8,
    res=0.02,
    seek=0.25,
    along=1.5,
    sparse_share=0.6,
):
    """(stations, offsets) of the paint centre across ``line``; offset NaN where no clear
    stripe is found. Positive offsets are to the left of the line direction.

    Cross-sections are averaged over +- ``along`` m of the line before measuring: a single
    cross-section of a scan has unscanned columns (the scan-line pattern), the average of
    neighbouring ones shows the whole stripe. The centre is the midpoint of the two
    half-brightness edges of the stripe (robust to wide stripes and uneven paint)."""
    from scipy.ndimage import map_coordinates

    from roadmarking.digitization.raster_tiles import xy_to_pixel

    length = line.length
    st = np.arange(0.0, length + 1e-9, step) if length > step else np.array([0.0, length])
    offs = np.arange(-reach, reach + 1e-9, res)
    prof = np.full((len(st), len(offs)), np.nan)
    g32 = gray.astype(np.float32)
    for k, s in enumerate(st):
        p = np.asarray(line.interpolate(s).coords[0][:2])
        a = np.asarray(line.interpolate(max(s - 0.5, 0)).coords[0][:2])
        b = np.asarray(line.interpolate(min(s + 0.5, length)).coords[0][:2])
        t = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        n = np.array([-t[1], t[0]])
        q = p[None, :] + offs[:, None] * n[None, :]
        r, c = xy_to_pixel(grid, q[:, 0], q[:, 1])
        inside = (r >= 0) & (c >= 0) & (r <= gray.shape[0] - 1) & (c <= gray.shape[1] - 1)
        ri, ci = np.rint(r).astype(int), np.rint(c).astype(int)
        ok = inside.copy()
        ok[inside] = observed[ri[inside], ci[inside]]
        if ok.any():
            prof[k, ok] = map_coordinates(g32, [r[ok], c[ok]], order=1)
    out = np.full(len(st), np.nan)
    w = max(1, round(along / step))
    near = np.abs(offs) <= seek
    zone = np.abs(offs) <= 0.25
    for k in range(len(st)):
        block = prof[max(k - w, 0) : k + w + 1]
        seen = np.isfinite(block)
        # sparse paint (far edge of a scan): few returns per cross-section -> average over a
        # longer stretch (+-2x) so the stripe, not single bright returns, is measured
        if seen[:, zone].mean() < sparse_share:
            block = prof[max(k - 2 * w, 0) : k + 2 * w + 1]
            seen = np.isfinite(block)
        cnt = seen.sum(axis=0)
        if (cnt > 0).mean() < 0.6:
            continue
        v = np.where(cnt > 0, np.nansum(block, axis=0) / np.maximum(cnt, 1), np.nan)
        # fill remaining single gaps from neighbours along the profile
        good = np.isfinite(v)
        v = np.interp(offs, offs[good], v[good])
        # light smoothing across the line (~6 cm): one bright return cannot set the edges,
        # the centre of a stripe does not move (symmetric kernel)
        v = np.convolve(np.pad(v, 2, mode="edge"), np.array([1, 2, 3, 2, 1]) / 9.0, "valid")
        road = np.percentile(v, 25)
        pk = int(np.argmax(np.where(near, v, -np.inf)))
        peak = v[pk]
        if peak - road < 15:
            continue
        half = road + 0.5 * (peak - road)
        lo = pk
        while lo > 0 and v[lo - 1] >= half:
            lo -= 1
        hi = pk
        while hi < len(offs) - 1 and v[hi + 1] >= half:
            hi += 1
        if lo == 0 or hi == len(offs) - 1:
            continue  # no road on one side within reach: not a stripe
        # sub-pixel half-brightness crossings
        left = offs[lo - 1] + (half - v[lo - 1]) / max(v[lo] - v[lo - 1], 1e-6) * res
        right = offs[hi] + (v[hi] - half) / max(v[hi] - v[hi + 1], 1e-6) * res
        if not 0.06 <= right - left <= 0.7:
            continue
        out[k] = 0.5 * (left + right)
    return st, out


def robust_smooth(st, off, window=10.0, iters=2):
    """Robust local linear fit (tricube x bisquare) of the measured offsets; stations without a
    measurement within ``window`` m are interpolated from the fitted ones."""
    good = np.isfinite(off)
    sm = np.full(len(st), np.nan)
    if good.sum() < 3:
        return sm
    xs, ys = st[good], off[good]
    rob = np.ones(len(xs))
    for _ in range(iters):
        fit = np.full(len(xs), np.nan)
        for i, x0 in enumerate(xs):
            d = np.abs(xs - x0) / window
            w = np.where(d < 1, (1 - d**3) ** 3, 0.0) * rob
            if w.sum() <= 0 or (w > 0).sum() < 3:
                fit[i] = ys[i]
                continue
            a_mat = np.column_stack([np.ones_like(xs), xs - x0])
            aw = a_mat * w[:, None]
            coef, *_ = np.linalg.lstsq(aw.T @ a_mat, aw.T @ ys, rcond=None)
            fit[i] = coef[0]
        res = ys - fit
        mad = np.median(np.abs(res)) + 1e-6
        u = res / (6 * mad)
        rob = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)
    return np.interp(st, xs, fit)


def recentre(line, gray, observed, grid, fixed_ends=(False, False), max_shift=0.25, min_good=0.3):
    """``line`` moved onto the centre of its paint stripe (LineString). Unchanged when too few
    stations show a clear stripe. ``fixed_ends``: (start, end) attached to another line."""
    from shapely.geometry import LineString

    if line.length < 0.5:
        return line, 0.0, None
    st, off = stripe_offsets(line, gray, observed, grid)
    share = float(np.isfinite(off).mean())
    if share < min_good:
        return line, 0.0, share
    sm = robust_smooth(st, off)
    if not np.isfinite(sm).all():
        return line, 0.0, share
    sm = np.clip(sm, -max_shift, max_shift)
    fade = np.ones(len(st))
    for is_end, d in ((fixed_ends[0], st), (fixed_ends[1], line.length - st)):
        if is_end:
            fade = np.minimum(fade, np.clip(d / 1.5, 0, 1))
    sm = sm * fade
    length = line.length
    pts = []
    for s, o in zip(st, sm, strict=True):
        p = np.asarray(line.interpolate(s).coords[0][:2])
        a = np.asarray(line.interpolate(max(s - 0.5, 0)).coords[0][:2])
        b = np.asarray(line.interpolate(min(s + 0.5, length)).coords[0][:2])
        t = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        pts.append(p + o * np.array([-t[1], t[0]]))
    return LineString(pts), float(np.median(np.abs(sm))), share


def off_centre_stretches(line, gray, observed, grid, tol=0.02, min_m=2.0):
    """Stretches of ``line`` (>= ``min_m``) whose measured paint centre is more than ``tol``
    away: [(x, y, length_m, median offset m)]."""
    st, off = stripe_offsets(line, gray, observed, grid)
    bad = np.isfinite(off) & (np.abs(off) > tol)
    out = []
    idx = np.flatnonzero(bad)
    if not len(idx):
        return out
    step = st[1] - st[0] if len(st) > 1 else 0.25
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 2) + 1):
        length = (run[-1] - run[0]) * step
        if length >= min_m:
            mid = line.interpolate(st[run[len(run) // 2]])
            out.append(
                (round(mid.x, 2), round(mid.y, 2), round(length, 2), float(np.median(off[run])))
            )
    return out


def _profiles(line, st, offs, gray, observed, grid, half=0.5):
    """Brightness across ``line`` at stations ``st`` (rows) and offsets ``offs`` (columns),
    bilinear on observed pixels, NaN elsewhere: one vectorised sampling of the raster."""
    import shapely
    from scipy.ndimage import map_coordinates

    from roadmarking.digitization.raster_tiles import xy_to_pixel

    length = line.length
    p = shapely.get_coordinates(shapely.line_interpolate_point(line, st))
    a = shapely.get_coordinates(shapely.line_interpolate_point(line, np.maximum(st - half, 0)))
    b = shapely.get_coordinates(shapely.line_interpolate_point(line, np.minimum(st + half, length)))
    t = b - a
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    n = np.column_stack([-t[:, 1], t[:, 0]])
    q = p[:, None, :] + offs[None, :, None] * n[:, None, :]
    r, c = xy_to_pixel(grid, q[..., 0].ravel(), q[..., 1].ravel())
    inside = (r >= 0) & (c >= 0) & (r <= gray.shape[0] - 1) & (c <= gray.shape[1] - 1)
    ri, ci = np.rint(r).astype(int), np.rint(c).astype(int)
    ok = inside.copy()
    ok[inside] = observed[ri[inside], ci[inside]]
    v = np.full(r.shape, np.nan)
    if ok.any():
        v[ok] = map_coordinates(gray.astype(np.float32), [r[ok], c[ok]], order=1)
    return v.reshape(len(st), len(offs))


def gap_crossings(line, gray, observed, grid, step=0.1, reach=0.5, res=0.02, min_run=0.3):
    """Stretches where ``line`` runs over the dark gap *between* two parallel painted stripes
    (a join between lines that do not merge: both stripes continue side by side). Returns
    [(start m, end m)] along the line."""
    import warnings

    length = line.length
    if length < 1.0:
        return []
    st = np.arange(0.0, length + 1e-9, step)
    offs = np.arange(-reach, reach + 1e-9, res)
    prof = _profiles(line, st, offs, gray, observed, grid)
    w = 3  # average over +-0.3 m along the line (scan-line gaps)
    # running nan-mean along the stations
    seen = np.isfinite(prof)
    vals = np.where(seen, prof, 0.0)
    cs_v = np.vstack([np.zeros((1, len(offs))), np.cumsum(vals, axis=0)])
    cs_n = np.vstack([np.zeros((1, len(offs))), np.cumsum(seen, axis=0)])
    lo = np.clip(np.arange(len(st)) - w, 0, len(st))
    hi = np.clip(np.arange(len(st)) + w + 1, 0, len(st))
    cnt = cs_n[hi] - cs_n[lo]
    v = np.where(cnt > 0, (cs_v[hi] - cs_v[lo]) / np.maximum(cnt, 1), np.nan)
    centre = np.abs(offs) <= 0.04
    left = (offs < -0.08) & (offs >= -0.4)
    right = (offs > 0.08) & (offs <= 0.4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        road = np.nanpercentile(v, 20, axis=1)
        lp = np.nanmax(v[:, left], axis=1)
        rp = np.nanmax(v[:, right], axis=1)
        mid = np.nanmean(v[:, centre], axis=1)
    low = np.minimum(lp, rp)
    crossing = (low - road >= 20) & (mid < road + 0.35 * (low - road))
    crossing &= np.isfinite(mid) & np.isfinite(lp) & np.isfinite(rp)
    out = []
    idx = np.flatnonzero(crossing)
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        if len(run) and (run[-1] - run[0]) * step >= min_run:
            out.append((max(st[run[0]] - 0.1, 0.0), min(st[run[-1]] + 0.1, length)))
    return out
