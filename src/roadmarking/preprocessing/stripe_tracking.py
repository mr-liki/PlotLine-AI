"""Finite, evidence-constrained stripe paths, independent of corridor references.

The legacy PCA reference detector remains unchanged for reproducible comparisons.
Here local-contrast ridges are thinned to a graph, split at junctions, centered on
original bright observations and joined only through observed PAINT support.
Dashed strokes have group IDs but never acquire connecting painted geometry.
"""

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from roadmarking.preprocessing.stripe_reference import local_contrast
from roadmarking.rasterization.rasterizer import pixel_to_world, world_to_pixel


@dataclass(frozen=True)
class TrackingSettings:
    min_length_m: float = 1.0
    max_width_m: float = 0.8
    max_turn_degrees: float = 35.0
    max_interpolation_m: float = 0.3
    max_dash_group_gap_m: float = 8.0
    dash_length_min_m: float = 2.0
    dash_length_max_m: float = 12.0
    minimum_supported_fraction: float = 0.95
    minimum_overlap_fraction: float = 0.55


def thin(mask):
    """Zhang-Suen binary thinning; changes graph topology only, never raster data."""
    image = np.pad(mask.astype(bool), 1)
    for _ in range(128):
        changed = False
        for phase in (0, 1):
            p = [
                np.roll(image, shift, axis=(0, 1))
                for shift in ((1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1))
            ]
            neighbors = sum(v.astype(np.uint8) for v in p)
            transitions = sum((~p[i] & p[(i + 1) % 8]).astype(np.uint8) for i in range(8))
            if phase == 0:
                corner = ~(p[0] & p[2] & p[4]) & ~(p[2] & p[4] & p[6])
            else:
                corner = ~(p[0] & p[2] & p[6]) & ~(p[0] & p[4] & p[6])
            remove = image & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1) & corner
            remove[[0, -1], :] = False
            remove[:, [0, -1]] = False
            if remove.any():
                changed = True
                image[remove] = False
        if not changed:
            break
    return image[1:-1, 1:-1]


def graph_paths(skeleton):
    pixels = {tuple(p) for p in np.argwhere(skeleton)}
    graph = {}
    for r, c in sorted(pixels):
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if not (dr or dc) or (r + dr, c + dc) not in pixels:
                    continue
                # Avoid diagonal shortcuts around an existing right-angle pixel.
                if dr and dc and ((r + dr, c) in pixels or (r, c + dc) in pixels):
                    continue
                neighbors.append((r + dr, c + dc))
        graph[(r, c)] = sorted(neighbors)
    used, paths = set(), []
    starts = sorted(graph, key=lambda p: (len(graph[p]) == 2, p))
    for start in starts:
        for neighbor in graph[start]:
            edge = tuple(sorted((start, neighbor)))
            if edge in used:
                continue
            path = [start, neighbor]
            used.add(edge)
            previous, current = start, neighbor
            while len(graph[current]) == 2:
                nxt = next(p for p in graph[current] if p != previous)
                edge = tuple(sorted((current, nxt)))
                if edge in used:
                    break
                used.add(edge)
                path.append(nxt)
                previous, current = current, nxt
            paths.append(path)
    return paths


def densify(coords, step):
    coords = np.asarray(coords, float)
    lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    good = lengths > 1e-9
    if not good.any():
        return np.empty((0, 2)), np.empty(0)
    starts, delta, lengths = coords[:-1][good], np.diff(coords, axis=0)[good], lengths[good]
    cumulative = np.r_[0, lengths.cumsum()]
    distances = np.linspace(0, cumulative[-1], max(2, int(np.ceil(cumulative[-1] / step)) + 1))
    indices = np.minimum(np.searchsorted(cumulative[1:], distances, side="right"), len(lengths) - 1)
    xy = (
        starts[indices]
        + ((distances - cumulative[indices]) / lengths[indices])[:, None] * delta[indices]
    )
    return xy, distances


def tangent(coords, at_end=False):
    coords = np.asarray(coords, float)
    if at_end:
        delta = coords[-1] - coords[max(0, len(coords) - 4)]
    else:
        delta = coords[min(3, len(coords) - 1)] - coords[0]
    return delta / max(np.linalg.norm(delta), 1e-12)


class LineEvidence:
    def __init__(self, image, observed, transform, contrast=None, paint=None):
        self.image, self.observed, self.transform = image, observed.astype(bool), transform
        if contrast is None or paint is None:
            contrast, paint = local_contrast(image, self.observed, transform.a)
        self.contrast, self.paint = contrast, paint.astype(bool)
        rows, cols = np.nonzero(self.paint)
        x, y = pixel_to_world(transform, cols, rows)
        self.bright_xy = np.column_stack((x, y))
        self.tree = cKDTree(self.bright_xy) if len(rows) else None
        if len(rows) > 1:
            spacing = self.tree.query(self.bright_xy[:: max(1, len(rows) // 10000)], k=2)[0][:, 1]
            self.spacing = float(np.median(spacing))
        else:
            self.spacing = transform.a
        self.support_radius = max(1.5 * transform.a, min(0.4, 1.5 * self.spacing))

    def sample(self, coords):
        xy, distances = densify(coords, self.transform.a / 2)
        if not len(xy):
            raise ValueError("Finite, nonzero line required")
        c, r = world_to_pixel(self.transform, xy[:, 0], xy[:, 1])
        inside = (c >= 0) & (r >= 0) & (c < self.image.shape[1]) & (r < self.image.shape[0])
        observed = np.zeros(len(xy), bool)
        paint = np.zeros(len(xy), bool)
        contrast = np.zeros(len(xy))
        observed[inside] = self.observed[r[inside], c[inside]]
        paint[inside] = self.paint[r[inside], c[inside]]
        contrast[inside] = self.contrast[r[inside], c[inside]]
        distance = self.tree.query(xy)[0] if self.tree else np.full(len(xy), np.inf)
        supported = observed & (distance <= self.support_radius)
        return xy, distances, observed, paint, contrast, supported


def line_quality(coords, evidence, search=None):
    import shapely

    xy, s, observed, paint, contrast, supported = evidence.sample(coords)
    interval = np.diff(s)
    support_length = float(np.sum(interval * (supported[:-1] & supported[1:])))
    literal_length = float(np.sum(interval * (paint[:-1] & paint[1:])))
    delta = np.diff(np.asarray(coords), axis=0)
    norm = np.linalg.norm(delta, axis=1)
    unit = delta / np.maximum(norm[:, None], 1e-12)
    turns = np.rad2deg(np.arccos(np.clip((unit[:-1] * unit[1:]).sum(axis=1), -1, 1)))
    in_search = (
        shapely.intersects_xy(search, xy[:, 0], xy[:, 1])
        if search is not None
        else np.ones(len(xy), bool)
    )
    return {
        "total_length": float(s[-1]),
        "supported_length": support_length,
        "unsupported_length": float(s[-1] - support_length),
        "interpolated_or_between_bright_cells_length": float(s[-1] - literal_length),
        "supported_fraction": support_length / max(float(s[-1]), 1e-12),
        "outside_observed_fraction": float((~observed).mean()),
        "inside_search_fraction": float(in_search.mean()),
        "mean_local_contrast": float(contrast.mean()),
        "minimum_local_contrast": float(contrast.min()),
        "max_turn_degrees": float(turns.max(initial=0)),
        "mean_turn_degrees": float(turns.mean()) if len(turns) else 0.0,
        "disconnected_components": 1,
        "cross_tile_joins": 0,
        "review_status": "PROPOSED_UNREVIEWED",
        "support_radius": evidence.support_radius,
        "measured_bright_cell_spacing": evidence.spacing,
    }


def _candidate(coords, evidence, cfg, width):
    coords = np.asarray(coords, float)
    if len(coords) > 1:
        coords = coords[np.r_[True, np.linalg.norm(np.diff(coords, axis=0), axis=1) > 1e-9]]
    if len(coords) < 2 or not np.isfinite(coords).all():
        return None
    # Direction canonicalization gives deterministic IDs independent of graph traversal.
    if tuple(coords[-1]) < tuple(coords[0]):
        coords = coords[::-1]
    quality = line_quality(coords, evidence)
    flags = []
    if quality["total_length"] < cfg.min_length_m:
        flags.append("SHORT_UNRESOLVED_FRAGMENT")
    if quality["supported_fraction"] < cfg.minimum_supported_fraction:
        flags.append("INSUFFICIENT_PAINT_SUPPORT")
    if quality["outside_observed_fraction"] > 0:
        flags.append("NODATA_INTERSECTION")
    if width > cfg.max_width_m:
        flags.append("BROAD_BRIGHT_STRUCTURE")
    if quality["max_turn_degrees"] > cfg.max_turn_degrees:
        flags.append("ABRUPT_TURN_OR_JUNCTION")
    identifier = hashlib.sha256(np.round(coords, 5).tobytes()).hexdigest()[:16]
    return {
        "id": f"stripe_{identifier}",
        "coords": coords.tolist(),
        "width": float(width),
        "candidate_class": (
            "DASH_OR_SHORT_STROKE_CANDIDATE"
            if cfg.dash_length_min_m <= quality["total_length"] <= cfg.dash_length_max_m
            else "CONTINUOUS_OR_FRAGMENT_CANDIDATE"
        ),
        "quality": quality,
        "flags": flags,
        "review_status": "PROPOSED_UNREVIEWED",
        "role": "CANDIDATE_PAINTED_STRIPE",
        "joins": [],
        "origin_ids": [f"stripe_{identifier}"],
    }


def trace_stripes(evidence, settings=None, mask=None):
    cfg = settings or TrackingSettings()
    bright = evidence.paint if mask is None else evidence.paint & mask
    if not bright.any():
        return [], [], {"status": "NO_PAINT_REFERENCE", "bright_pixels": 0}
    # A one-cell neighborhood bridges scan sampling only. Every traced interval
    # is independently checked against the ORIGINAL bright and observed masks.
    working = ndimage.binary_dilation(bright, iterations=1)
    skeleton = thin(working)
    widths = ndimage.distance_transform_edt(working) * 2 * evidence.transform.a
    candidates, qa = [], []
    for path in graph_paths(skeleton):
        rc = np.asarray(path)
        x, y = pixel_to_world(evidence.transform, rc[:, 1], rc[:, 0])
        coords = np.column_stack((x, y))
        # Trim dilation endpoints to actual supporting bright-cell centers.
        if evidence.tree:
            _, indices = evidence.tree.query(coords[[0, -1]])
            coords[[0, -1]] = evidence.bright_xy[indices]
        # Local smoothing is allowed only when the proposed positions remain
        # close to ORIGINAL paint evidence. Endpoints and junctions stay fixed.
        if len(coords) > 4:
            smooth = ndimage.gaussian_filter1d(coords, 1, axis=0, mode="nearest")
            distance = evidence.tree.query(smooth)[0]
            accept = distance <= evidence.support_radius
            accept[[0, -1]] = False
            coords[accept] = smooth[accept]
        width = max(
            evidence.transform.a,
            float(np.median(widths[rc[:, 0], rc[:, 1]])) - 2 * evidence.transform.a,
        )
        candidate = _candidate(coords, evidence, cfg, width)
        if candidate:
            (qa if candidate["flags"] else candidates).append(candidate)
            # A single sampling hole must not suppress all observed portions of
            # a long stroke. Keep the original in QA and emit supported runs.
            if candidate["quality"]["total_length"] >= cfg.min_length_m and any(
                flag in candidate["flags"]
                for flag in ("NODATA_INTERSECTION", "INSUFFICIENT_PAINT_SUPPORT")
            ):
                xy, _, observed, _, _, supported = evidence.sample(coords)
                valid = observed & supported
                edges = np.flatnonzero(np.diff(np.r_[False, valid, False].astype(np.int8)))
                for begin, end in zip(edges[::2], edges[1::2], strict=True):
                    if end - begin < 2:
                        continue
                    piece = _candidate(xy[begin:end], evidence, cfg, width)
                    if piece and not piece["flags"]:
                        piece["origin_ids"] = [candidate["id"]]
                        candidates.append(piece)
    return (
        candidates,
        qa,
        {
            "status": "PROVISIONAL",
            "bright_pixels": int(bright.sum()),
            "skeleton_pixels": int(skeleton.sum()),
            "support_radius": evidence.support_radius,
            "measured_spacing": evidence.spacing,
        },
    )


def _compatible(a, b, cfg):
    return (
        a["candidate_class"] == b["candidate_class"]
        and max(a["width"], b["width"]) / max(min(a["width"], b["width"]), 1e-8) <= 2
    )


def stitch_candidates(candidates, evidence, settings=None):
    """Join only short paint-supported endpoint gaps; never turn dash groups solid."""
    cfg = settings or TrackingSettings()
    items = [dict(c, origin_ids=list(c.get("origin_ids", [c["id"]]))) for c in candidates]
    duplicate_ids = []
    # Remove duplicate overlap observations, not distinct parallel painted lines.
    keep = []
    for candidate in sorted(items, key=lambda c: (-c["quality"]["total_length"], c["id"])):
        geometry = LineString(candidate["coords"])
        duplicate = None
        for other in keep:
            if not _compatible(candidate, other, cfg):
                continue
            covering = geometry.intersection(
                LineString(other["coords"]).buffer(evidence.transform.a * 0.75)
            ).length / max(geometry.length, 1e-8)
            if covering >= max(0.95, cfg.minimum_overlap_fraction) and abs(
                np.dot(tangent(candidate["coords"]), tangent(other["coords"]))
            ) >= np.cos(np.deg2rad(cfg.max_turn_degrees)):
                # Only discard a geometrically covered duplicate. Partial overlap
                # fragments with unique tails remain available for supported joins.
                duplicate = other
                break
        if duplicate:
            duplicate["origin_ids"] += candidate["origin_ids"]
            duplicate_ids.append(candidate["id"])
        else:
            keep.append(candidate)
    joins = []
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(keep):
            if changed:
                break
            for j, b in enumerate(keep):
                if j <= i or not _compatible(a, b, cfg):
                    continue
                for reverse_a, reverse_b in (
                    (False, False),
                    (False, True),
                    (True, False),
                    (True, True),
                ):
                    pa = np.asarray(a["coords"])[:: -1 if reverse_a else 1]
                    pb = np.asarray(b["coords"])[:: -1 if reverse_b else 1]
                    delta = pb[0] - pa[-1]
                    gap = float(np.linalg.norm(delta))
                    if (
                        gap
                        > min(
                            cfg.max_interpolation_m,
                            max(2 * evidence.transform.a, 3 * evidence.spacing),
                        )
                        or gap < 1e-9
                    ):
                        continue
                    cosine = np.cos(np.deg2rad(cfg.max_turn_degrees))
                    if (
                        np.dot(tangent(pa, True), tangent(pb)) < cosine
                        or np.dot(tangent(pa, True), delta / gap) < cosine
                    ):
                        continue
                    q = line_quality([pa[-1], pb[0]], evidence)
                    if q["supported_fraction"] < 1 or q["outside_observed_fraction"] > 0:
                        continue
                    combined = _candidate(
                        np.vstack((pa, pb)), evidence, cfg, (a["width"] + b["width"]) / 2
                    )
                    if combined is None or combined["flags"]:
                        continue
                    join = {
                        "a": a["id"],
                        "b": b["id"],
                        "gap": gap,
                        "status": "SHORT_PAINT_SUPPORTED_INTERPOLATION",
                        "review_status": "PROPOSED_UNREVIEWED",
                    }
                    combined["joins"] = a["joins"] + b["joins"] + [join]
                    combined["origin_ids"] = a["origin_ids"] + b["origin_ids"]
                    keep[i] = combined
                    keep.pop(j)
                    joins.append(join)
                    changed = True
                    break
                if changed:
                    break
    # Group dashed/short strokes without changing their coordinates or connecting gaps.
    group = 0
    for i, a in enumerate(keep):
        if a.get("dash_group_id"):
            continue
        related = [a]
        if a["candidate_class"] == "DASH_OR_SHORT_STROKE_CANDIDATE":
            for b in keep[i + 1 :]:
                if not _compatible(a, b, cfg):
                    continue
                pa, pb = np.asarray(a["coords"]), np.asarray(b["coords"])
                delta = pb.mean(axis=0) - pa.mean(axis=0)
                direction = tangent(pa)
                lateral = abs(direction[0] * delta[1] - direction[1] * delta[0])
                if (
                    LineString(pa).distance(LineString(pb)) <= cfg.max_dash_group_gap_m
                    and lateral <= max(a["width"], b["width"], 2 * evidence.transform.a)
                    and abs(np.dot(direction, tangent(pb)))
                    >= np.cos(np.deg2rad(cfg.max_turn_degrees))
                ):
                    related.append(b)
        if len(related) > 1:
            identifier = f"dash_group_{group:04d}"
            group += 1
            for member in related:
                member["dash_group_id"] = identifier
    return sorted(keep, key=lambda c: c["id"]), {
        "duplicate_ids": duplicate_ids,
        "joins": joins,
        "dash_groups": group,
    }


def clip_candidates(candidates, extent, evidence, settings=None):
    cfg = settings or TrackingSettings()
    output = []
    for candidate in candidates:
        intersection = LineString(candidate["coords"]).intersection(extent)
        parts = (
            [intersection]
            if intersection.geom_type == "LineString"
            else list(getattr(intersection, "geoms", []))
        )
        for part in parts:
            if part.geom_type != "LineString" or part.length <= 0:
                continue
            result = _candidate(np.asarray(part.coords), evidence, cfg, candidate["width"])
            if result:
                result["parent_id"] = candidate["id"]
                result["dash_group_id"] = candidate.get("dash_group_id")
                result["joins"] = candidate["joins"]
                result["origin_ids"] = candidate["origin_ids"]
                output.append(result)
    return output
