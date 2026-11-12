"""Three explicit reference modes with an auditable native-coordinate review gate."""

import json
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, box, shape

from roadmarking.preprocessing.stripe_reference import detect_references


def load_seed_points(path, source_sha256, extent):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("coordinate_frame") != "NATIVE_XY" or data.get("crs"):
        raise ValueError("Seeds require an explicit NATIVE_XY frame")
    if data.get("source_sha256") != source_sha256:
        raise ValueError("Seed source hash mismatch")
    points = []
    for feature in data.get("features", []):
        geometry = shape(feature["geometry"])
        if geometry.geom_type not in ("Point", "MultiPoint") or geometry.is_empty:
            raise ValueError("Seed Point or MultiPoint geometry required")
        parts = [geometry] if geometry.geom_type == "Point" else list(geometry.geoms)
        for point in parts:
            xy = np.asarray(point.coords)[0, :2]
            if not np.isfinite(xy).all() or not box(*extent).covers(point):
                raise ValueError("Seed must be finite and inside the source extent")
            points.append(xy)
    if not points:
        raise ValueError("No seed points supplied")
    return np.asarray(points)


def load_references(path, source_sha256, extent, *, approved=False):
    """Native-XY JSON overlay, never reprojecting/repairing geometry or inventing CRS."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("crs") or data.get("coordinate_frame") != "NATIVE_XY":
        raise ValueError("Explicit NATIVE_XY frame required; declared CRS is not accepted")
    if data.get("source_sha256") != source_sha256:
        raise ValueError("Reference source hash mismatch")
    out = []
    for i, feature in enumerate(data.get("features", [])):
        geometry = shape(feature["geometry"])
        props = feature.get("properties", {})
        if geometry.geom_type != "LineString" or not geometry.is_valid or geometry.length <= 0:
            raise ValueError("Valid nonzero alignment LineString required")
        if not np.isfinite(np.asarray(geometry.coords)).all() or not geometry.intersects(
            box(*extent)
        ):
            raise ValueError("Reference must overlap source native extent")
        if approved and (
            props.get("review_status") != "APPROVED"
            or not props.get("reviewer")
            or not props.get("reviewed_at")
        ):
            raise ValueError("Validated processing requires approved, attributed references")
        out.append(
            {
                "id": f"guided_{i:04d}",
                "coords": np.asarray(geometry.coords)[:, :2].tolist(),
                "reference_supported": True,
                "review_status": props.get("review_status", "PROPOSED_UNREVIEWED"),
                "method": "user_supplied_native_reference",
                "length_m": geometry.length,
                "uncertainty": [] if approved else ["provisional_user_reference"],
            }
        )
    if not out:
        raise ValueError("No reference lines supplied")
    return out


def discover(
    image,
    observed,
    transform,
    *,
    mode="intensity",
    references=None,
    geometry_xy=None,
    settings=None,
    seed_points=None,
):
    if mode == "reference":
        if not references:
            raise ValueError("Reference-guided mode needs supplied lines")
        return references, {}, "USER_REFERENCE_REQUIRES_SURFACE_QA"
    if mode not in ("intensity", "geometry"):
        raise ValueError("Unknown discovery mode")
    if mode == "intensity":
        lines, evidence = detect_references(image, observed, transform, settings)
        if any(line["reference_supported"] for line in lines):
            return lines, evidence, "PROVISIONAL_INTENSITY_REFERENCES"
    else:
        lines, evidence = [], {}
    # Geometry supplies a REVIEW path only. No densest/lowest surface assumption.
    if geometry_xy is not None and len(geometry_xy) >= 8:
        xy = np.asarray(geometry_xy)
        if seed_points is not None:
            seeds = np.asarray(seed_points, float)
            if seeds.ndim != 2 or seeds.shape[1] != 2 or not np.isfinite(seeds).all():
                raise ValueError("Finite native XY seed points required")
            distance = np.min(np.linalg.norm(xy[:, None, :] - seeds[None, :, :], axis=2), axis=1)
            xy = xy[distance <= 15.0]
            if len(xy) < 8:
                return lines, evidence, "REVIEW_REQUIRED_SEEDS_HAVE_INSUFFICIENT_SUPPORT"
        center = np.median(xy, axis=0)
        _, _, axes = np.linalg.svd(xy - center, full_matrices=False)
        along = (xy - center) @ axes[0]
        endpoints = center + np.percentile(along, [5, 95])[:, None] * axes[0]
        lines.append(
            {
                "id": "geometry_review_0000",
                "coords": endpoints.tolist(),
                "length_m": LineString(endpoints).length,
                "method": "geometry_PCA_review_only",
                "reference_supported": False,
                "review_status": "PROPOSED_UNREVIEWED",
                "uncertainty": ["geometry_does_not_identify_road", "reference_acceptance_required"],
                "seed_points": np.asarray(seed_points).tolist() if seed_points is not None else [],
            }
        )
    return lines, evidence, "REVIEW_REQUIRED_NO_RELIABLE_REFERENCE"
