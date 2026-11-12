"""Training-coverage descriptors: solid curvature, geographic sections and paint contrast.

These are dataset *descriptors* used for audit and region prioritisation. They never alter
labels, geometry or approval scope.
"""

import math

import numpy as np
from scipy.ndimage import binary_dilation
from shapely.geometry import LineString, MultiLineString


def _parts(geom):
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    if hasattr(geom, "geoms"):
        return [p for g in geom.geoms for p in _parts(g)]
    return []


def turning_degrees(geom, step=5.0):
    """Total absolute heading change (degrees) along each line part, densified at ``step``.

    Sub-step vertex jitter is removed by resampling, so digitising noise below ``step``
    does not register as curvature.
    """
    total = 0.0
    for part in _parts(geom):
        if part.length < 2 * step:
            continue
        distances = np.arange(0.0, part.length + 1e-9, step)
        points = np.array([part.interpolate(d).coords[0][:2] for d in distances])
        vectors = np.diff(points, axis=0)
        headings = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))
        total += float(np.degrees(np.abs(np.diff(headings)).sum()))
    return total


def solid_shape(geom, minimum_length=10.0, curved_degrees_per_50=5.0, step=5.0):
    """Classify a clipped solid centerline as ``curved``, ``straight`` or ``short``.

    Curvature is normalised to degrees of heading change per 50 native units so windows of
    different clipped length are comparable. A 5 deg / 50 m threshold corresponds to a
    radius of roughly 570 m.
    """
    length = sum(p.length for p in _parts(geom))
    if length < minimum_length:
        return "short", length, 0.0
    # Chord headings sample between first and last chord midpoints: span = length - step.
    span = sum(p.length - step for p in _parts(geom) if p.length >= 2 * step)
    if span <= 0:
        return "short", length, 0.0
    rate = turning_degrees(geom, step) * 50.0 / span
    return ("curved" if rate >= curved_degrees_per_50 else "straight"), length, rate


def cluster_sections(boxes, guard):
    """Union-find clustering of shapely geometries closer than ``guard``; returns labels."""
    parent = list(range(len(boxes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].distance(boxes[j]) < guard:
                parent[find(i)] = find(j)
    roots = {}
    return [roots.setdefault(find(i), len(roots)) for i in range(len(boxes))]


def paint_contrast(gray, mask, observed, exclude=None, inner=2, outer=6, minimum=8):
    """Median observed intensity inside ``mask`` minus median of an observed outer ring.

    Returns ``None`` when either side has fewer than ``minimum`` observed pixels. ``exclude``
    (e.g. all other paint) is removed from the ring so neighbouring markings do not mask
    fading.
    """
    mask = np.asarray(mask, bool) & observed
    if mask.sum() < minimum:
        return None
    ring = binary_dilation(mask, iterations=outer) & ~binary_dilation(mask, iterations=inner)
    ring &= observed
    if exclude is not None:
        ring &= ~exclude
    if ring.sum() < minimum:
        return None
    return float(np.median(gray[mask]) - np.median(gray[ring]))


def radius_from_rate(degrees_per_50):
    """Approximate circular radius (native units) for a turning rate in deg / 50 units."""
    if degrees_per_50 <= 0:
        return math.inf
    return 50.0 / math.radians(degrees_per_50)
