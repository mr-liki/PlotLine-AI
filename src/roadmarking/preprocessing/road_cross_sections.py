"""Station-based native-XY profiles queried from one bounded original-point KD-tree."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SectionSettings:
    spacing_m: float = 1.0
    half_width_m: float = 30.0
    along_half_length_m: float = 0.6
    bin_m: float = 0.25
    min_bin_points: int = 3
    noise_floor_m: float = 0.01
    noise_cap_m: float = 0.04
    max_step_m: float = 0.15
    max_slope: float = 0.30
    max_slope_change: float = 0.12
    max_station_width_change_m: float = 1.5


def stations(coords, spacing=1.0):
    points = np.asarray(coords, float)[:, :2]
    if not np.isfinite(points).all() or spacing <= 0 or len(points) < 2:
        raise ValueError("Finite native XY polyline and positive spacing required")
    delta = np.diff(points, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    good = lengths > 1e-8
    starts, delta, lengths = points[:-1][good], delta[good], lengths[good]
    if not len(lengths):
        raise ValueError("Reference has zero length")
    cumulative = np.r_[0, lengths.cumsum()]
    distances = np.arange(0, cumulative[-1] + 1e-8, spacing)
    indices = np.minimum(np.searchsorted(cumulative[1:], distances, side="right"), len(lengths) - 1)
    fractions = (distances - cumulative[indices]) / lengths[indices]
    centers = starts[indices] + fractions[:, None] * delta[indices]
    tangents = delta[indices] / lengths[indices, None]
    return distances, centers, tangents


def _profile(points, offsets, center, normal, tangent, station, cfg):
    edges = np.arange(-cfg.half_width_m, cfg.half_width_m + cfg.bin_m * 0.5, cfg.bin_m)
    centers = (edges[:-1] + edges[1:]) / 2
    indices = np.floor((offsets + cfg.half_width_m) / cfg.bin_m).astype(int)
    n = len(centers)
    z, noise = np.full(n, np.nan), np.full(n, np.nan)
    counts = np.bincount(indices[(indices >= 0) & (indices < n)], minlength=n)
    # Density-adaptive minimum, bounded to avoid asserting a surface from one return.
    positive = counts[counts > 0]
    minimum = (
        max(cfg.min_bin_points, min(12, int(np.median(positive) * 0.1)))
        if len(positive)
        else cfg.min_bin_points
    )
    order = np.argsort(indices)
    sorted_ids = indices[order]
    ids, starts, sizes = np.unique(sorted_ids, return_index=True, return_counts=True)
    for index, start, size in zip(ids, starts, sizes, strict=True):
        if index < 0 or index >= n or size < minimum:
            continue
        values = points[order[start : start + size], 2]
        # Median does not assume that the lowest return is pavement.
        z[index] = np.median(values)
        noise[index] = 1.4826 * np.median(np.abs(values - z[index]))
    valid = np.isfinite(z)
    runs = []
    current = []
    breaks = []
    for index in range(n):
        reason = None
        if not valid[index]:
            reason = "insufficient_observations"
        elif noise[index] > cfg.noise_cap_m:
            reason = "mixed_or_rough_bin"
        elif current:
            last = current[-1]
            slope = (z[index] - z[last]) / cfg.bin_m
            if abs(z[index] - z[last]) > cfg.max_step_m:
                reason = "height_step"
            elif abs(slope) > cfg.max_slope:
                reason = "slope_break"
            elif len(current) >= 2:
                previous = (z[last] - z[current[-2]]) / cfg.bin_m
                # Noise-aware slope differences; no global horizontal-plane assumption.
                tolerance = max(
                    cfg.max_slope_change,
                    3 * max(cfg.noise_floor_m, noise[index], noise[last]) / cfg.bin_m,
                )
                if abs(slope - previous) > tolerance:
                    reason = "normal_change"
        if reason:
            if len(current) >= 3:
                runs.append(current)
            current = []
            if valid[index] and reason not in ("mixed_or_rough_bin",):
                current = [index]
            breaks.append({"offset": float(centers[index]), "reason": reason})
        else:
            current.append(index)
    if len(current) >= 3:
        runs.append(current)
    segments = []
    for run in runs:
        rr = np.array(run)
        # Fit local profile and record residual; curved/banked profiles need not be flat.
        coefficient = np.polyfit(centers[rr], z[rr], min(2, len(rr) - 1))
        residual = z[rr] - np.polyval(coefficient, centers[rr])
        segments.append(
            {
                "left": float(edges[rr[0]]),
                "right": float(edges[rr[-1] + 1]),
                "median_z": float(np.median(z[rr])),
                "rms": float(np.sqrt(np.mean(residual**2))),
                "noise_median": float(np.median(noise[rr])),
                "point_count": int(counts[rr].sum()),
                "near_reference": bool(edges[rr[0]] <= 0 <= edges[rr[-1] + 1]),
                "edge_status": "UNCERTAIN_SMOOTH_TRANSITION_OR_SAMPLING_LIMIT",
            }
        )
    return {
        "station": float(station),
        "center": center.tolist(),
        "normal": normal.tolist(),
        "tangent": tangent.tolist(),
        "offsets": centers.tolist(),
        "elevations": [float(v) if np.isfinite(v) else None for v in z],
        "noise": [float(v) if np.isfinite(v) else None for v in noise],
        "counts": counts.tolist(),
        "adaptive_min_points": minimum,
        "segments": segments,
        "break_candidates": breaks,
        "review_status": "PROPOSED_UNREVIEWED",
    }


def extract_sections(xyz, references, settings=None):
    cfg = settings or SectionSettings()
    if min(cfg.spacing_m, cfg.half_width_m, cfg.bin_m, cfg.along_half_length_m) <= 0:
        raise ValueError("Positive section dimensions required")
    tree = cKDTree(xyz[:, :2])
    sections = []
    radius = np.hypot(cfg.half_width_m, cfg.along_half_length_m)
    for reference in references:
        if not reference.get("reference_supported", False):
            continue
        distances, centers, tangents = stations(reference["coords"], cfg.spacing_m)
        for station, center, tangent in zip(distances, centers, tangents, strict=True):
            normal = np.array([-tangent[1], tangent[0]])
            ids = tree.query_ball_point(center, radius)
            points = xyz[ids]
            delta = points[:, :2] - center
            along, cross = delta @ tangent, delta @ normal
            keep = (np.abs(along) <= cfg.along_half_length_m) & (np.abs(cross) < cfg.half_width_m)
            profile = _profile(points[keep], cross[keep], center, normal, tangent, station, cfg)
            profile["reference_id"] = reference["id"]
            sections.append(profile)
    return sections
