"""Road-surface Z for centerline vertices from the original LiDAR points.

The source has no ground classification, so each vertex uses a local robust plane:
neighbours within ``radius`` (expanded once) -> least-squares plane z = a + b dx + c dy ->
iterative trimming of residuals beyond max(3 x MAD, ``tol``) (removes vehicles, kerb tops,
multipath) -> Z = a (plane height at the vertex). Support statistics are returned per
vertex. A vertex without enough consistent support gets NaN, never a fabricated value.
Units follow the source (verified metres for the pilot CRS).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


class Surface:
    """``source`` (optional, one id per point) names the acquisition of each point. Passes
    of different surveys disagree in height by a few cm, so each fit uses only the points of
    the pass with the most points near the query."""

    def __init__(self, points, source=None):
        self.points = np.asarray(points, float)
        self.source = None if source is None else np.asarray(source)
        self.tree = cKDTree(self.points[:, :2]) if len(self.points) else None

    def z_at(
        self, x, y, radius=0.30, max_radius=0.60, min_pts=8, tol=0.02, max_rms=0.03, max_slope=0.15
    ):
        out = dict(
            z=np.nan, n=0, rms=np.nan, slope=np.nan, radius=np.nan, ok=False, reason="no_points"
        )
        if self.tree is None:
            return out
        for r in (radius, max_radius):
            idx = self.tree.query_ball_point([x, y], r)
            if len(idx) >= min_pts:
                break
        if len(idx) < min_pts:
            out.update(n=len(idx), reason="too_few_points")
            return out
        if self.source is not None:
            src = self.source[idx]
            ids, counts = np.unique(src, return_counts=True)
            best = ids[np.argmax(counts)]
            if counts.max() >= min_pts:
                idx = np.asarray(idx)[src == best]
        p = self.points[idx]
        d = p[:, :2] - [x, y]
        keep = np.ones(len(p), bool)
        coef = None
        for _ in range(4):
            a = np.column_stack([np.ones(keep.sum()), d[keep]])
            coef, *_ = np.linalg.lstsq(a, p[keep, 2], rcond=None)
            res = p[:, 2] - np.column_stack([np.ones(len(p)), d]) @ coef
            mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
            new = np.abs(res) <= max(3 * mad, tol)
            if new.sum() < min_pts or (new == keep).all():
                keep = new if new.sum() >= min_pts else keep
                break
            keep = new
        a = np.column_stack([np.ones(keep.sum()), d[keep]])
        coef, *_ = np.linalg.lstsq(a, p[keep, 2], rcond=None)
        res = p[keep, 2] - a @ coef
        rms = float(np.sqrt(np.mean(res**2)))
        slope = float(np.hypot(coef[1], coef[2]))
        ok = keep.sum() >= min_pts and rms <= max_rms and slope <= max_slope
        reason = (
            "ok"
            if ok
            else (
                "rough_surface"
                if rms > max_rms
                else ("steep" if slope > max_slope else "too_few_inliers")
            )
        )
        out.update(
            z=float(coef[0]),
            n=int(keep.sum()),
            rms=rms,
            slope=slope,
            radius=r,
            ok=bool(ok),
            reason=reason,
        )
        if not ok:
            out["z"] = np.nan
        return out


def lift_line(surface, xy, max_interp_m=1.0, **kw):
    """Z per vertex; short unsupported runs (<= max_interp_m between supported vertices) are
    linearly interpolated and flagged; longer runs stay NaN (feature needs review)."""
    recs = [surface.z_at(x, y, **kw) for x, y in xy]
    z = np.array([r["z"] for r in recs])
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(np.asarray(xy), axis=0).T))]
    interp = np.zeros(len(z), bool)
    good = np.isfinite(z)
    if good.sum() >= 2:
        i = 0
        while i < len(z):
            if good[i]:
                i += 1
                continue
            j = i
            while j < len(z) and not good[j]:
                j += 1
            if i > 0 and j < len(z) and s[j] - s[i - 1] <= max_interp_m:
                z[i:j] = np.interp(s[i:j], [s[i - 1], s[j]], [z[i - 1], z[j]])
                interp[i:j] = True
            i = j
    return z, interp, recs


__all__ = ["Surface", "lift_line"]
