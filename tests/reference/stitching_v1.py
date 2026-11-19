"""REFERENCE copy of the production stitching before the speed-up (kept for exactness tests).

Evidence-constrained cross-crop instance stitching (Stage 3).

Solid fragments from overlapping crops are merged only when they overlap inside the shared
crop strip (intersection over the smaller in-strip area), their local directions agree and
the pairing is mutual-best. Short gaps between merged solids are linked (identity only, no
pixels added) when the straight bridge lies entirely on observed pixels. Dash duplicates
across crops keep the copy whose centroid is most central in its crop. Thresholds are
applied AFTER merging. Masks are only ever unioned: no pixel is created.
"""

import numpy as np
from skimage.draw import line as draw_line


def _axis(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) < 3:
        return None, None, None
    pts = np.column_stack((ys, xs)).astype(float)
    centre = pts.mean(0)
    _, vecs = np.linalg.eigh(np.cov((pts - centre).T))
    axis = vecs[:, -1]
    proj = (pts - centre) @ axis
    return axis, pts[np.argmin(proj)], pts[np.argmax(proj)]


def _angle(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), 0, 1))))


def _window_box(w):
    r, c = w
    return r, c, r + 512, c + 512


def stitch(
    masks,
    scores,
    classes,
    crop_ids,
    windows,
    valid,
    thresholds,
    ios=0.3,
    max_angle=15.0,
    link_px=20,
    link_angle=10.0,
    link_lateral=3.0,
    dash_ios=0.5,
):
    """Return merged instances [{'mask','score','class','members'}] and audit counters."""
    n = len(scores)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    solids = [i for i in range(n) if classes[i] == 0]
    # ---- in-strip overlap merging for solids (mutual best per crop pair)
    best = {}
    for i in solids:
        for j in solids:
            if crop_ids[i] == crop_ids[j]:
                continue
            a, b = _window_box(windows[crop_ids[i]]), _window_box(windows[crop_ids[j]])
            r0, c0, r1, c1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
            if r1 <= r0 or c1 <= c0:
                continue
            mi, mj = masks[i][r0:r1, c0:c1], masks[j][r0:r1, c0:c1]
            ai, aj = int(mi.sum()), int(mj.sum())
            if not ai or not aj:
                continue
            inter = int((mi & mj).sum())
            value = inter / min(ai, aj)
            if value < ios:
                continue
            ax_i, ax_j = _axis(mi)[0], _axis(mj)[0]
            if ax_i is None or ax_j is None or _angle(ax_i, ax_j) > max_angle:
                continue
            key = (i, crop_ids[j])
            if value > best.get(key, (0, -1))[0]:
                best[key] = (value, j)
    merges = 0
    for (i, _), (_, j) in best.items():
        if best.get((j, crop_ids[i]), (0, -1))[1] == i and find(i) != find(j):
            parent[find(i)] = find(j)
            merges += 1
    groups = {}
    for i in solids:
        groups.setdefault(find(i), []).append(i)
    merged = [
        {
            "members": sorted(g),
            "class": 0,
            "mask": np.any([masks[k] for k in g], axis=0),
            "score": float(max(scores[k] for k in g)),
        }
        for g in groups.values()
    ]
    # ---- observed-only gap linking (identity only)
    links = 0
    ends = [_axis(m["mask"]) for m in merged]
    proposals = []
    for a in range(len(merged)):
        for b in range(a + 1, len(merged)):
            if ends[a][0] is None or ends[b][0] is None:
                continue
            if _angle(ends[a][0], ends[b][0]) > link_angle:
                continue
            pairs = [(ea, eb) for ea in ends[a][1:] for eb in ends[b][1:]]
            ea, eb = min(pairs, key=lambda p: np.linalg.norm(p[0] - p[1]))
            gap = float(np.linalg.norm(ea - eb))
            if gap > link_px:
                continue
            normal = np.array([-ends[a][0][1], ends[a][0][0]])
            if abs(float((eb - ea) @ normal)) > link_lateral:
                continue
            rr, cc = draw_line(int(ea[0]), int(ea[1]), int(eb[0]), int(eb[1]))
            if not valid[rr, cc].all():
                continue  # never bridge NoData
            proposals.append((gap, a, b))
    used = set()
    lparent = list(range(len(merged)))

    def lfind(i):
        while lparent[i] != i:
            lparent[i] = lparent[lparent[i]]
            i = lparent[i]
        return i

    for _, a, b in sorted(proposals):
        if (a, "end") in used or (b, "end") in used:
            continue
        lparent[lfind(a)] = lfind(b)
        used.update({(a, "end"), (b, "end")})
        links += 1
    linked = {}
    for k, m in enumerate(merged):
        linked.setdefault(lfind(k), []).append(m)
    solids_out = [
        {
            "members": sorted(x for m in g for x in m["members"]),
            "class": 0,
            "mask": np.any([m["mask"] for m in g], axis=0),
            "score": float(max(m["score"] for m in g)),
        }
        for g in linked.values()
    ]
    # ---- dash duplicates: keep the most central copy
    dashes = [i for i in range(n) if classes[i] == 1]

    def centrality(i):
        ys, xs = np.nonzero(masks[i])
        r, c, r1, c1 = _window_box(windows[crop_ids[i]])
        cy, cx = ys.mean(), xs.mean()
        return min(cy - r, r1 - cy, cx - c, c1 - cx)

    removed = set()
    for i in sorted(dashes, key=lambda k: -centrality(k)):
        if i in removed:
            continue
        for j in dashes:
            if j == i or j in removed or crop_ids[j] == crop_ids[i]:
                continue
            inter = int((masks[i] & masks[j]).sum())
            if inter and inter / min(masks[i].sum(), masks[j].sum()) >= dash_ios:
                removed.add(j)
    dash_out = [
        {"members": [i], "class": 1, "mask": masks[i], "score": float(scores[i])}
        for i in dashes
        if i not in removed
    ]
    instances = [m for m in solids_out + dash_out if m["score"] >= thresholds[str(m["class"])]]
    return instances, {
        "solid_overlap_merges": merges,
        "solid_gap_links": links,
        "dash_duplicates_removed": len(removed),
    }
