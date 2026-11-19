"""Join solid lines across occlusion gaps (vehicles) as explicitly INFERRED geometry.

A gap between two solid centerlines is bridged only when all of these hold:
* the end tangents agree (<= ``max_angle_deg``) and the far endpoint lies on the extrapolated
  line within ``max_lateral_m`` (+0.5 % of the gap): the same physical line continues;
* gap length <= ``max_gap_m``;
* for gaps >= 1 m both sides are >= ``min_piece_m`` long (longer than any dash), so a dash
  rhythm misread as solid pieces is never chained; tiny scan-gap joins run first;
* along the gap, every sample is classified as hidden (no LiDAR returns: vehicle occlusion or
  scan stripe), visible paint, or visible unpainted road; the bridge is refused when more
  than ``max_dark_observed`` of samples are visible unpainted road (a real break stays open);
  ``evidence`` records whether the gap was mostly hidden (>= ``min_unobserved``) or partly
  visible paint;
* the bridge does not cross another feature.
Bridges are mutual-best, greedy by cost, and chains are allowed. The bridge is a cubic
Hermite curve matching both end tangents. No paint pixel is created; every inferred part is
returned separately so it can be reviewed, styled and audited. Dashes are never bridged.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString


def end_state(line: LineString, at_start: bool, span=2.0):
    """Endpoint and outward unit tangent estimated over the last ``span`` metres."""
    length = line.length
    if at_start:
        p = np.asarray(line.interpolate(0).coords[0])
        q = np.asarray(line.interpolate(min(span, length)).coords[0])
    else:
        p = np.asarray(line.interpolate(length).coords[0])
        q = np.asarray(line.interpolate(max(length - span, 0)).coords[0])
    t = p - q
    n = np.linalg.norm(t)
    return p, (t / n if n > 1e-9 else t)


def hermite(p0, t0, p1, t1, step=0.25):
    """Cubic Hermite from p0 (leaving along t0) to p1 (arriving along t1)."""
    d = np.linalg.norm(p1 - p0)
    n = max(2, int(np.ceil(d / step)) + 1)
    s = np.linspace(0, 1, n)[:, None]
    h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
    h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
    return h00 * p0 + h10 * d * t0 + h01 * p1 + h11 * d * t1


def bridge_solids(
    lines,
    observed_at,
    bright_at,
    *,
    max_gap_m=25.0,
    max_angle_deg=8.0,
    max_lateral_m=0.25,
    min_unobserved=0.6,
    max_dark_observed=0.3,
    min_gap_m=0.05,
    max_short_gap_m=0.25,
    min_piece_m=6.0,
):
    """lines: list of LineString (solid). observed_at/bright_at: f(xy[N,2]) -> bool[N].

    Returns (chains, bridges): chains = list of dict(coords, members, inferred=[LineString]);
    bridges = audit records for accepted and rejected candidates.
    """
    cos_max = np.cos(np.radians(max_angle_deg))
    items = [
        dict(coords=np.asarray(line.coords)[:, :2], members=[k], inferred=[])
        for k, line in enumerate(lines)
    ]
    audit = []
    blocked = set()
    while True:
        cands = []
        for i, a in enumerate(items):
            la = LineString(a["coords"])
            for ea in (False, True):  # end of a (False = tail, True = head)
                pa, ta = end_state(la, at_start=ea)
                for j, b in enumerate(items):
                    if j <= i:
                        continue
                    lb = LineString(b["coords"])
                    for eb in (False, True):
                        pb, tb = end_state(lb, at_start=eb)
                        gap = pb - pa
                        dist = float(np.linalg.norm(gap))
                        if not (min_gap_m <= dist <= max_gap_m):
                            continue
                        u = gap / dist
                        # leaving a along ta, entering b against tb (tb points outward of b)
                        if -(ta @ tb) < cos_max:  # the two line directions must agree
                            continue
                        lateral = abs(float(gap @ np.array([-ta[1], ta[0]])))
                        if dist >= 1.0 and min(la.length, lb.length) < min_piece_m:
                            # dash-sized pieces across a long gap look like a dashed line
                            # misread as solid: never chain them into a continuous line
                            continue
                        if dist >= 1.0:
                            # long gap: the gap itself must also run along the line
                            if ta @ u < cos_max or -(tb @ u) < cos_max:
                                continue
                            if lateral > max_lateral_m + 0.005 * dist:
                                continue
                        elif lateral > 0.10 or ta @ u < 0:
                            # short gap: its direction is noise; bound the sideways step
                            continue
                        curve = hermite(pa, ta, pb, -tb)
                        # Evidence samples on the gap itself at ~pixel spacing, ends excluded.
                        k = max(3, int(np.ceil(dist / 0.025)))
                        frac = np.linspace(0, 1, k + 2)[1:-1][:, None]
                        inner = pa + frac * (pb - pa)
                        obs = observed_at(inner)
                        unobs = float((~obs).mean())
                        dark = float((obs & ~bright_at(inner)).mean())
                        rec = dict(
                            a=a["members"],
                            b=b["members"],
                            gap_m=dist,
                            lateral_m=lateral,
                            unobserved_fraction=unobs,
                            dark_observed_fraction=dark,
                        )
                        # Every gap sample is hidden (no returns), visible paint, or visible
                        # unpainted road. Only the last contradicts continuity.
                        if dark > max_dark_observed:
                            audit.append(dict(rec, accepted=False, reason="observed_break"))
                            continue
                        rec["evidence"] = (
                            "short_scan_gap"
                            if dist <= max_short_gap_m
                            else (
                                "occlusion"
                                if unobs >= min_unobserved
                                else "occlusion_and_visible_paint"
                            )
                        )
                        cost = dist + 20 * lateral
                        cands.append((cost, i, ea, j, eb, curve, rec))
        if not cands:
            break
        cands.sort(key=lambda c: c[0])
        chosen = None
        for cand in cands:
            _, i, _, j, _, curve, rec = cand
            if frozenset((i, j)) in blocked:
                continue
            bridge = LineString(curve)
            others = [LineString(it["coords"]) for k, it in enumerate(items) if k not in (i, j)]
            if any(bridge.crosses(o) for o in others):
                audit.append(dict(rec, accepted=False, reason="crosses_other_feature"))
                blocked.add(frozenset((i, j)))
                continue
            chosen = cand
            break
        if chosen is None:
            break
        cost, i, ea, j, eb, curve, rec = chosen
        bridge = LineString(curve)
        blocked = set()  # indices change after a merge
        a, b = items[i], items[j]
        ca = a["coords"] if not ea else a["coords"][::-1]  # orient a so it ENDS at the gap
        cb = b["coords"] if eb else b["coords"][::-1]  # orient b so it STARTS at the gap
        merged = np.vstack([ca, curve[1:-1], cb])
        new = dict(
            coords=merged,
            members=a["members"] + b["members"],
            inferred=a["inferred"] + b["inferred"] + [bridge],
        )
        audit.append(dict(rec, accepted=True, reason="bridged"))
        items = [it for k, it in enumerate(items) if k not in (i, j)] + [new]
    return items, audit


__all__ = ["bridge_solids", "end_state", "hermite"]
