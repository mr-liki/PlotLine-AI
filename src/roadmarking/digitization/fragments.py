"""Observed marking fragments and identity layers from stitched accepted instances.

Identity layers kept distinct:
* query identity      -> instance ``provenance`` (tile, query index, score);
* observed fragment   -> one connected component of the per-class UNION of accepted paint, so a
  marking observed by several overlapping crops becomes one fragment (connectivity may cross
  only LiDAR sampling holes of <= ``sampling_gap_px`` unobserved pixels; no paint is created);
* provisional marking -> the stitched instance (production stitch may link solid fragments
  only across OBSERVED pixels);
* continuation records -> collinear solid fragments separated by a larger gap; recorded as
  ``possible_continuation`` (gap mostly unobserved) or ``observed_break`` and never merged.
A pixel claimed by both classes is a class conflict and is flagged, not resolved silently.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, find_objects, label
from skimage.draw import line as draw_line
from skimage.morphology import disk

EIGHT = np.ones((3, 3), bool)


def _axis(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) < 3:
        return None
    p = np.column_stack([ys, xs]).astype(float)
    c = p.mean(0)
    _, v = np.linalg.eigh(np.cov((p - c).T))
    a = v[:, -1]
    t = (p - c) @ a
    return a, p[np.argmin(t)], p[np.argmax(t)]


def extract_fragments(instances, observed, sampling_gap_px=2, min_px=8):
    shape = observed.shape
    solid_any = np.zeros(shape, bool)
    dash_any = np.zeros(shape, bool)
    for inst in instances:
        (solid_any if inst["class"] == 0 else dash_any)[...] |= inst["mask"]
    conflict = solid_any & dash_any
    frags, rejected = [], []
    se = disk(sampling_gap_px)
    # Tile-overlap reconciliation: the same painted pixels seen by several crops/queries are
    # one observation, so fragments come from the per-class UNION of accepted paint. Distinct
    # dashes stay separate because their observed gaps exceed the sampling-hole bridge.
    for cls, mask in ((0, solid_any), (1, dash_any)):
        members = [i for i in instances if i["class"] == cls]
        closed = binary_closing(mask, structure=se) & (mask | ~observed)  # bridge only holes
        comps, _n = label(closed, structure=EIGHT)
        for k, sl in enumerate(find_objects(comps), start=1):
            if sl is None:
                continue
            region = comps[sl] == k
            paint = region & mask[sl]
            src = [i for i in members if (i["mask"][sl] & paint).any()]
            rec = dict(
                fragment_id=len(frags) + len(rejected),
                instance_id=min(i["instance_id"] for i in src) if src else -1,
                instance_ids=sorted(i["instance_id"] for i in src),
                cls=cls,
                score=max((i["score"] for i in src), default=0.0),
                provenance=[p for i in src for p in i["provenance"]],
                offset=(sl[0].start, sl[1].start),
                mask=paint,
                connect_mask=region,
                paint_px=int(paint.sum()),
                sampling_bridge_px=int((region & ~mask[sl]).sum()),
                class_conflict_px=int((paint & conflict[sl]).sum()),
                duplicate_observations=max(len(src) - 1, 0),
            )
            (frags if rec["paint_px"] >= min_px else rejected).append(rec)
    return frags, rejected, dict(class_conflict_px=int(conflict.sum()))


def bright_paint(gray, observed, delta=40, window=31):
    """Paint-like returns: observed and >= local observed background + ``delta`` (grey levels)."""
    from scipy.ndimage import uniform_filter

    o = observed.astype(np.float32)
    bg = uniform_filter(gray.astype(np.float32) * o, window) / np.maximum(
        uniform_filter(o, window), 1e-3
    )
    return observed & (gray >= bg + delta)


def continuity_relations(
    frags,
    observed,
    max_gap_px=60,
    max_angle=10.0,
    max_lateral_px=3.0,
    bright=None,
    evidence_gap_px=12,
    evidence_fraction=0.6,
):
    """Collinear solid fragment pairs separated by a gap.

    A short gap (<= ``evidence_gap_px``) whose OBSERVED pixels are paint-bright in the LiDAR
    intensity (``bright``) is ``paint_evidence_continuation`` (certain: supported by observed
    paint returns). Every other gap is recorded only: ``possible_continuation`` (mostly
    unobserved) or ``observed_break``; those are never merged.
    """
    sol = [f for f in frags if f["cls"] == 0]
    ends = []
    for f in sol:
        g = np.zeros(observed.shape, bool)
        r, c = f["offset"]
        h, w = f["mask"].shape
        g[r : r + h, c : c + w] = f["mask"]
        ends.append(_axis(g))
    rel = []
    for a in range(len(sol)):
        for b in range(a + 1, len(sol)):
            if ends[a] is None or ends[b] is None:
                continue
            ang = np.degrees(np.arccos(np.clip(abs(float(ends[a][0] @ ends[b][0])), 0, 1)))
            if ang > max_angle:
                continue
            pa, pb = min(
                ((x, y) for x in ends[a][1:] for y in ends[b][1:]),
                key=lambda p: np.linalg.norm(p[0] - p[1]),
            )
            gap = float(np.linalg.norm(pa - pb))
            if gap > max_gap_px or gap < 1:
                continue
            normal = np.array([-ends[a][0][1], ends[a][0][0]])
            if abs(float((pb - pa) @ normal)) > max_lateral_px:
                continue
            rr, cc = draw_line(int(pa[0]), int(pa[1]), int(pb[0]), int(pb[1]))
            rr, cc = rr[1:-1], cc[1:-1]  # gap pixels only (endpoints belong to fragments)
            unobs = float((~observed[rr, cc]).mean()) if len(rr) else 0.0
            obs_gap = observed[rr, cc] if len(rr) else np.zeros(0, bool)
            bright_frac = (
                float(bright[rr, cc][obs_gap].mean())
                if bright is not None and obs_gap.any()
                else 0.0
            )
            if gap <= evidence_gap_px and (
                len(rr) == 0 or (obs_gap.mean() >= 0.5 and bright_frac >= evidence_fraction)
            ):
                kind, certain = "paint_evidence_continuation", True
            else:
                kind, certain = (
                    "possible_continuation" if unobs >= 0.5 else "observed_break"
                ), False
            rel.append(
                dict(
                    from_fragment=sol[a]["fragment_id"],
                    to_fragment=sol[b]["fragment_id"],
                    gap_px=gap,
                    angle_deg=float(ang),
                    unobserved_fraction=unobs,
                    gap_bright_fraction=bright_frac,
                    kind=kind,
                    same_instance=sol[a]["instance_id"] == sol[b]["instance_id"],
                    endpoints_rc=[pa.tolist(), pb.tolist()],
                    certain=certain,
                )
            )
    return rel


def merge_evidence_links(frags, relations, shape):
    """Merge fragments joined by ``paint_evidence_continuation`` (connectivity only; the gap
    line is added to ``connect_mask`` for centerline tracing, never to the paint mask)."""
    parent = {f["fragment_id"]: f["fragment_id"] for f in frags}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    gaps = {}
    for r in relations:
        if r["kind"] == "paint_evidence_continuation":
            a, b = find(r["from_fragment"]), find(r["to_fragment"])
            if a != b:
                parent[a] = b
            gaps.setdefault(r["from_fragment"], []).append(r["endpoints_rc"])
    groups = {}
    for f in frags:
        groups.setdefault(find(f["fragment_id"]), []).append(f)
    out = []
    for members in groups.values():
        if len(members) == 1:
            out.append(
                dict(members[0], merged_fragments=[members[0]["fragment_id"]], evidence_bridge_px=0)
            )
            continue
        paint = np.zeros(shape, bool)
        conn = np.zeros(shape, bool)
        for f in members:
            r0, c0 = f["offset"]
            h, w = f["mask"].shape
            paint[r0 : r0 + h, c0 : c0 + w] |= f["mask"]
            conn[r0 : r0 + h, c0 : c0 + w] |= f["connect_mask"]
        bridge = np.zeros(shape, bool)
        for f in members:
            for pa, pb in gaps.get(f["fragment_id"], []):
                rr, cc = draw_line(int(pa[0]), int(pa[1]), int(pb[0]), int(pb[1]))
                bridge[rr, cc] = True
        bridge = binary_dilation(bridge, iterations=1) & ~conn
        conn |= bridge
        ys, xs = np.nonzero(conn)
        r0, r1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        base = max(members, key=lambda f: f["paint_px"])
        out.append(
            dict(
                base,
                fragment_id=base["fragment_id"],
                merged_fragments=sorted(f["fragment_id"] for f in members),
                instance_ids=sorted({i for f in members for i in f["instance_ids"]}),
                provenance=[p for f in members for p in f["provenance"]],
                score=max(f["score"] for f in members),
                offset=(r0, c0),
                mask=paint[r0:r1, c0:c1],
                connect_mask=conn[r0:r1, c0:c1],
                paint_px=int(paint.sum()),
                evidence_bridge_px=int(bridge.sum()),
                sampling_bridge_px=sum(f["sampling_bridge_px"] for f in members),
                class_conflict_px=sum(f["class_conflict_px"] for f in members),
                duplicate_observations=sum(f["duplicate_observations"] for f in members),
            )
        )
    return out


__all__ = ["bright_paint", "continuity_relations", "extract_fragments", "merge_evidence_links"]
