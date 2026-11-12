"""Stage 7F: classify stitched/merged outputs that touch several ground-truth IDs.

The Stage-3 rule counted an output solid as a *false link* whenever it overlapped two
solid client IDs (>= 10% of its pixels each). A client may split one painted line into
consecutive IDs, so that rule mixes harmless joins with real errors. Here client IDs are
first grouped into *physical lines* (end-to-end, collinear continuation), and each
multi-ID output is classified by physical identity:

- ``same_physical_line``: all touched IDs belong to one physical line (not an error)
- ``dashes_joined``: an output joins two or more dash IDs
- ``solid_dash_joined``: an output joins IDs of different classes
- ``parallel_lines_joined``: touched solid IDs run side by side (lateral offset)
- ``different_lines_joined``: any other join of distinct physical lines

``nodata_bridge`` flags joins whose touched parts are disconnected inside the valid area
and separated mostly by NoData.
"""

import numpy as np
from scipy.ndimage import binary_dilation, label
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

HIT_FRACTION = 0.1  # identical to the Stage-3 false-link rule


def _points(mask):
    skel = skeletonize(mask)
    pts = np.argwhere(skel if skel.any() else mask).astype(float)
    return pts


def _direction(pts, centre, radius=30.0):
    near = pts[np.hypot(*(pts - centre).T) <= radius]
    if len(near) < 3:
        near = pts
    if len(near) < 2:
        return None
    _, vecs = np.linalg.eigh(np.cov(near.T))
    return vecs[:, -1]


def _angle(u, v):
    c = abs(float(np.dot(u, v)) / max(1e-9, np.linalg.norm(u) * np.linalg.norm(v)))
    return float(np.degrees(np.arccos(min(1.0, c))))


def relation(a, b, gap_px=20.0, max_angle=10.0, lateral_px=4.0):
    """'continuation', 'parallel' or 'other' for two target masks."""
    pa, pb = _points(a), _points(b)
    if not len(pa) or not len(pb):
        return "other"
    dist, idx = cKDTree(pb).query(pa)
    i = int(np.argmin(dist))
    ca, cb, d = pa[i], pb[idx[i]], float(dist[i])
    ua, ub = _direction(pa, ca), _direction(pb, cb)
    if ua is None or ub is None or _angle(ua, ub) > max_angle:
        return "other"
    axis = (ua + np.sign(np.dot(ua, ub) or 1) * ub) / 2
    disp = cb - ca
    lateral = abs(float(disp[0] * axis[1] - disp[1] * axis[0])) / max(1e-9, np.linalg.norm(axis))
    if d <= gap_px and lateral <= lateral_px:
        return "continuation"
    # Side-by-side: substantial overlap along the axis but separated laterally.
    proj_a, proj_b = pa @ axis, pb @ axis
    overlap = min(proj_a.max(), proj_b.max()) - max(proj_a.min(), proj_b.min())
    if overlap > 0 and lateral > lateral_px:
        return "parallel"
    return "other"


def physical_lines(targets, classes, **kw):
    """Union-find of same-class IDs that continue each other end to end."""
    n = len(targets)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    boxes = [np.argwhere(t) for t in targets]
    for i in range(n):
        for j in range(i + 1, n):
            if classes[i] != classes[j] or classes[i] != 0 or not len(boxes[i]):
                continue
            if not len(boxes[j]):
                continue
            lo_i, hi_i = boxes[i].min(0), boxes[i].max(0)
            lo_j, hi_j = boxes[j].min(0), boxes[j].max(0)
            if (np.maximum(lo_i, lo_j) - np.minimum(hi_i, hi_j) > kw.get("gap_px", 20.0)).any():
                continue
            if relation(targets[i], targets[j], **kw) == "continuation":
                parent[find(i)] = find(j)
    return [find(i) for i in range(n)]


def classify_outputs(masks, classes, keep, targets, target_classes, valid, lines=None, **kw):
    """Classify every kept output touching >= 2 IDs; return records and counts."""
    target_classes = np.asarray(target_classes, int)
    lines = lines if lines is not None else physical_lines(targets, target_classes, **kw)
    records = []
    for k in keep:
        px = max(1, int((masks[k] & valid).sum()))
        hits = [
            j
            for j in range(len(targets))
            if int((masks[k] & targets[j]).sum()) >= HIT_FRACTION * px
        ]
        if len(hits) < 2:
            continue
        hit_classes = {int(target_classes[j]) for j in hits}
        solid_hits = [j for j in hits if target_classes[j] == 0]
        legacy = int(classes[k]) == 0 and len(solid_hits) >= 2
        if len(hit_classes) > 1:
            kind = "solid_dash_joined"
        elif hit_classes == {1}:
            kind = "dashes_joined"
        elif len({lines[j] for j in hits}) == 1:
            kind = "same_physical_line"
        else:
            rel = {relation(targets[a], targets[b], **kw) for a in hits for b in hits if a < b}
            kind = "parallel_lines_joined" if "parallel" in rel else "different_lines_joined"
        records.append(
            {
                "output": int(k),
                "output_class": int(classes[k]),
                "touched_ids": hits,
                "physical_lines": sorted({int(lines[j]) for j in hits}),
                "kind": kind,
                "stage3_rule_false_link": legacy,
                "nodata_bridge": nodata_bridge(masks[k], [targets[j] for j in hits], valid),
            }
        )
    counts = {}
    for r in records:
        counts[r["kind"]] = counts.get(r["kind"], 0) + 1
    counts["stage3_rule_false_links"] = sum(r["stage3_rule_false_link"] for r in records)
    counts["true_false_links"] = sum(r["kind"] != "same_physical_line" for r in records)
    counts["nodata_bridges"] = sum(r["nodata_bridge"] for r in records)
    return records, counts


def nodata_bridge(mask, touched, valid):
    """Touched parts lie in different valid components of the output and the straight gap
    between their closest pixels is mostly NoData."""
    lab, _ = label(mask & valid)
    parts = [set(np.unique(lab[mask & t & valid])) - {0} for t in touched]
    comps = sorted(set().union(*parts)) if parts else []
    if len(comps) < 2 or all(parts[0] & p for p in parts[1:]):
        return False
    for i, a in enumerate(comps):
        for b in comps[i + 1 :]:
            pa, pb = np.argwhere(lab == a), np.argwhere(lab == b)
            dist, idx = cKDTree(pb).query(pa)
            k = int(np.argmin(dist))
            n = max(2, int(np.ceil(dist[k])) + 1)
            line = np.linspace(pa[k], pb[idx[k]], n).round().astype(int)[1:-1]
            if len(line) and (~valid[line[:, 0], line[:, 1]]).mean() > 0.5:
                return True
    return False


def duplicates(masks, classes, keep, targets, target_classes, ios=0.5):
    """Targets covered by >= 2 kept same-class outputs (each >= ``ios`` inside the target)."""
    out = 0
    for j, t in enumerate(targets):
        cover = sum(
            1
            for k in keep
            if classes[k] == target_classes[j]
            and (masks[k] & t).sum() >= ios * max(1, masks[k].sum())
        )
        out += cover >= 2
    return out


# ---------------------------------------------------------------- Stage 8 connection audit
ACCEPTED = "continuous_marking"


def _mst(edges, nodes):
    parent = {n: n for n in nodes}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    out = []
    for e in sorted(edges, key=lambda e: e[0]):
        a, b = find(e[1]), find(e[2])
        if a != b:
            parent[a] = b
            out.append(e)
    return out


def output_gaps(mask, valid, observed, paint, member_masks=None, min_component=10):
    """Connections (minimum spanning tree) between the valid components of one output.

    Each connection is the straight segment between the closest pixels of two components.
    ``unobserved`` is the fraction of its interior pixels without LiDAR observation,
    ``excluded`` observed-but-unscored pixels (symbols, ignore zones), ``painted`` the
    fraction lying on observed labelled paint (``paint``). ``bridged_by`` says whether one
    model mask already spans the gap or only the stitching merge joined the parts.
    """
    lab, n = label(mask & valid, structure=np.ones((3, 3), bool))
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    comps = [i for i in range(1, n + 1) if sizes[i] >= min_component]
    if len(comps) < 2:
        return []
    pts = {i: np.argwhere(lab == i) for i in comps}
    edges = []
    for i, a in enumerate(comps):
        tree = cKDTree(pts[a])
        for b in comps[i + 1 :]:
            dist, idx = tree.query(pts[b])
            k = int(np.argmin(dist))
            edges.append((float(dist[k]), a, b, pts[a][idx[k]], pts[b][k]))
    out = []
    for d, a, b, pa, pb in _mst(edges, comps):
        steps = max(2, int(np.ceil(d)) + 1)
        line = np.unique(np.linspace(pa, pb, steps).round().astype(int), axis=0)
        inner = np.array([p for p in line if lab[p[0], p[1]] not in (a, b)], dtype=int).reshape(
            -1, 2
        )
        if not len(inner):
            continue
        r, c = inner[:, 0], inner[:, 1]
        unobserved = float((~observed[r, c]).mean())
        excluded = float((observed[r, c] & ~valid[r, c]).mean())
        painted = float(paint[r, c].mean())
        bridged = "single_model_mask"
        if member_masks is not None and len(member_masks) > 1:
            spans = any((m & (lab == a)).any() and (m & (lab == b)).any() for m in member_masks)
            bridged = "single_model_mask" if spans else "stitch_merge"
        if painted >= 0.5:
            kind = "paint_supported"
        elif unobserved > 0.5:
            kind = "nodata_unsupported"
        else:
            kind = "observed_unpainted"
        out.append(
            {
                "from_px": [int(pa[0]), int(pa[1])],
                "to_px": [int(pb[0]), int(pb[1])],
                "length_px": d,
                "unobserved_fraction": unobserved,
                "excluded_fraction": excluded,
                "painted_fraction": painted,
                "bridged_by": bridged,
                "kind": kind,
            }
        )
    return out


def audit_connections(
    masks, classes, keep, targets, target_classes, valid, observed, members=None, lines=None, **kw
):
    """Every kept output that joins >= 2 IDs or contains a gap without paint support.

    Final ``category`` (one per output, first rule that applies):
    ``nodata_crossing_unsupported`` > ``separate_dashes`` > ``different_or_parallel`` >
    ``observed_gap_unpainted`` > ``continuous_marking`` (the only accepted outcome).
    """
    target_classes = np.asarray(target_classes, int)
    lines = lines if lines is not None else physical_lines(targets, target_classes, **kw)
    paint = (
        binary_dilation(targets.any(0), iterations=1) & observed
        if len(targets)
        else np.zeros(valid.shape, bool)
    )
    records = []
    for k in keep:
        px = max(1, int((masks[k] & valid).sum()))
        hits = [
            j
            for j in range(len(targets))
            if int((masks[k] & targets[j]).sum()) >= HIT_FRACTION * px
        ]
        gaps = output_gaps(masks[k], valid, observed, paint, members[k] if members else None)
        weak = [g for g in gaps if g["kind"] != "paint_supported"]
        if len(hits) < 2 and not weak:
            continue
        hit_classes = {int(target_classes[j]) for j in hits}
        if len(hits) < 2:
            identity = "single_id" if hits else "no_id"
        elif len(hit_classes) > 1:
            identity = "solid_dash_joined"
        elif hit_classes == {1}:
            identity = "dashes_joined"
        elif len({lines[j] for j in hits}) == 1:
            identity = "same_physical_line"
        else:
            rel = {relation(targets[a], targets[b], **kw) for a in hits for b in hits if a < b}
            identity = "parallel_lines_joined" if "parallel" in rel else "different_lines_joined"
        if any(g["kind"] == "nodata_unsupported" for g in gaps):
            category = "nodata_crossing_unsupported"
        elif identity == "dashes_joined":
            category = "separate_dashes"
        elif identity in ("solid_dash_joined", "parallel_lines_joined", "different_lines_joined"):
            category = "different_or_parallel"
        elif weak:
            category = "observed_gap_unpainted"
        else:
            category = ACCEPTED
        records.append(
            {
                "output": int(k),
                "output_class": int(classes[k]),
                "members": len(members[k]) if members else 1,
                "touched_ids": hits,
                "identity": identity,
                "category": category,
                "accepted": category == ACCEPTED,
                "gaps": gaps,
                "stage7f_counted": len(hits) >= 2,
                "stage7f_nodata_bridge": (
                    nodata_bridge(masks[k], [targets[j] for j in hits], valid)
                    if len(hits) >= 2
                    else False
                ),
            }
        )
    return records
