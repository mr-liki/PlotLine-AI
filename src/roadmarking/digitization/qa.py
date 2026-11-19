"""Digitization QA: pixel-to-vector consistency, identity, 3D and reference comparison."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.ndimage import binary_dilation, map_coordinates
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from shapely.strtree import STRtree


def densify(coords, step=1.0):
    p = np.asarray(coords, float)
    out = [p[0]]
    for a, b in itertools.pairwise(p):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out.extend(a + (b - a) * (k / n) for k in range(1, n + 1))
    return np.asarray(out)


def smoothness_mm(coords_m, spacing=0.25):
    """Jitter: deviation of each resampled vertex from the midpoint of its neighbours (mm).
    A smooth curve of radius R gives ~spacing^2 / (2R) (0.1 mm at R = 300 m)."""
    line = LineString(np.asarray(coords_m)[:, :2])
    if line.length < 3 * spacing:
        return dict(jitter_rms_mm=0.0, jitter_max_mm=0.0)
    p = np.asarray([line.interpolate(d).coords[0] for d in np.arange(0, line.length, spacing)])
    dev = np.linalg.norm(p[1:-1] - (p[:-2] + p[2:]) / 2, axis=1) * 1000
    return dict(jitter_rms_mm=float(np.sqrt(np.mean(dev**2))), jitter_max_mm=float(dev.max()))


def pixel_consistency(refined_rc, raw_rc, connect_mask, paint_mask, width_px):
    """Containment of the centerline in paint, paint coverage, skeleton displacement (px)."""
    d = densify(refined_rc, 0.5)
    inside = map_coordinates(
        binary_dilation(connect_mask).astype(float), d.T, order=0, mode="constant"
    )
    line = LineString(np.asarray(refined_rc)[:, ::-1])
    ys, xs = np.nonzero(paint_mask)
    if len(ys):
        dist = np.array([line.distance(Point(x, y)) for x, y in zip(xs[::3], ys[::3], strict=True)])
        cover = float((dist <= width_px / 2 + 1.0).mean())
    else:
        cover = np.nan
    disp = (
        float(np.mean([line.distance(Point(c, r)) for r, c in raw_rc])) if len(raw_rc) else np.nan
    )
    return dict(
        containment=float(inside.mean()), paint_coverage=cover, skeleton_displacement_px=disp
    )


def identity_checks(features, dup_buffer=0.10, dash_max_m=6.0, oversplit_gap_m=0.5):
    lines = [f["geometry_2d"] for f in features]
    tree = STRtree(lines)
    dup, cross, oversplit = [], [], []
    for i, f in enumerate(features):
        for j in tree.query(lines[i].buffer(dup_buffer)):
            j = int(j)
            if j <= i:
                continue
            g = features[j]
            inter = lines[i].buffer(dup_buffer).intersection(lines[j]).length
            short = min(lines[i].length, lines[j].length)
            if f["class_id"] == g["class_id"] and short > 0 and inter / short > 0.5:
                dup.append((f["mark_id"], g["mark_id"]))
            elif lines[i].crosses(lines[j]):
                cross.append((f["mark_id"], g["mark_id"]))
            elif (
                f["class_id"] == g["class_id"] == 1
                and lines[i].distance(lines[j]) < oversplit_gap_m
            ):
                a, b = (
                    np.diff(np.asarray(lines[i].coords)[[0, -1]], axis=0)[0],
                    np.diff(np.asarray(lines[j].coords)[[0, -1]], axis=0)[0],
                )
                cosang = abs(a @ b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9)
                if cosang > 0.98:
                    oversplit.append((f["mark_id"], g["mark_id"]))
    merged = [f["mark_id"] for f in features if f["class_id"] == 1 and f["length_m"] > dash_max_m]
    return dict(
        duplicates=dup,
        crossings=cross,
        dash_possible_oversplit=oversplit,
        dash_possible_merged=merged,
    )


def reference_compare(features, ref_lines, ref_classes, region, tol):
    """Compare same-class lines within ``region`` (observed support polygon).

    tol: dict(match_buffer, lateral, endpoint). Client geometry is a reference only.
    """
    out = {}
    for cls in (0, 1):
        ours = [f["geometry_2d"] for f in features if f["class_id"] == cls]
        refs = [
            g.intersection(region) for g, c in zip(ref_lines, ref_classes, strict=True) if c == cls
        ]
        refs = [r for r in refs if not r.is_empty and r.length > 0]
        ou = unary_union(ours) if ours else None
        ru = unary_union(refs) if refs else None
        ref_len = sum(r.length for r in refs)
        our_len = sum(o.length for o in ours)
        matched_ref = (
            ru.intersection(ou.buffer(tol["match_buffer"])).length
            if ou is not None and ru is not None
            else 0.0
        )
        matched_our = (
            ou.intersection(ru.buffer(tol["match_buffer"])).length
            if ou is not None and ru is not None
            else 0.0
        )
        lateral = []
        if ru is not None:
            for o in ours:
                for x, y in densify(np.asarray(o.coords), 0.25):
                    d = ru.distance(Point(x, y))
                    if d <= tol["match_buffer"]:
                        lateral.append(d)
        res = dict(
            reference_length=ref_len,
            our_length=our_len,
            matched_reference_length=matched_ref,
            matched_our_length=matched_our,
            recall_length=matched_ref / ref_len if ref_len else None,
            precision_length=matched_our / our_len if our_len else None,
            lateral_median=float(np.median(lateral)) if lateral else None,
            lateral_p95=float(np.percentile(lateral, 95)) if lateral else None,
            lateral_within_tol=(
                float(np.mean(np.asarray(lateral) <= tol["lateral"])) if lateral else None
            ),
            reference_features=len(refs),
            our_features=len(ours),
        )
        if cls == 1 and refs and ours:
            used, pairs, ends = set(), 0, []
            for r in refs:
                cand = [
                    (o.intersection(r.buffer(tol["match_buffer"])).length, k)
                    for k, o in enumerate(ours)
                    if k not in used
                ]
                best = max(cand, default=(0, -1))
                if best[0] >= 0.5 * min(r.length, ours[best[1]].length if best[1] >= 0 else 1):
                    used.add(best[1])
                    pairs += 1
                    o = ours[best[1]]
                    e = (
                        sorted(
                            [
                                Point(r.coords[0]).distance(Point(o.coords[0]))
                                + Point(r.coords[-1]).distance(Point(o.coords[-1])),
                                Point(r.coords[0]).distance(Point(o.coords[-1]))
                                + Point(r.coords[-1]).distance(Point(o.coords[0])),
                            ]
                        )[0]
                        / 2
                    )
                    ends.append(e)
            res.update(
                dash_matched=pairs,
                dash_reference=len(refs),
                dash_ours=len(ours),
                endpoint_error_median=float(np.median(ends)) if ends else None,
                endpoint_within_tol=(
                    float(np.mean(np.asarray(ends) <= tol["endpoint"])) if ends else None
                ),
            )
        out["solid" if cls == 0 else "dash"] = res
    return out


__all__ = [
    "densify",
    "identity_checks",
    "pixel_consistency",
    "reference_compare",
    "smoothness_mm",
]
