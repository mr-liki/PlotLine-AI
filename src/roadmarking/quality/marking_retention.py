"""Observation retention metrics and human-reference validation (never proxy recall)."""

import json
from pathlib import Path

import numpy as np
from rasterio.features import rasterize
from shapely.geometry import shape


def retention_metrics(raw, old, new, candidate_paint, resolution=0.05):
    observed, previous, retained = raw["observed"] > 0, old["observed"] > 0, new["observed"] > 0
    paint = np.asarray(candidate_paint, bool) & observed
    denominator = int(paint.sum())
    both = previous & retained
    return {
        "wide_observed_pixels": int(observed.sum()),
        "old_observed_pixels": int(previous.sum()),
        "new_observed_pixels": int(retained.sum()),
        "previously_excluded_observed_recovered": int((retained & ~previous).sum()),
        "old_observed_now_missing": int((previous & ~retained).sum()),
        "candidate_paint_pixels": denominator,
        "candidate_paint_preserved": int((paint & retained).sum()),
        "candidate_paint_preserved_pct": 100 * int((paint & retained).sum()) / max(denominator, 1),
        "verified_marking_recall": None,
        "raw_max_equal_original_where_retained": bool(
            np.array_equal(raw["maximum"][retained], new["maximum"][retained])
        ),
        "old_max_equal_where_both": bool(
            np.array_equal(old["maximum"][both], new["maximum"][both])
        ),
        "raw_mean_equal_original_where_retained": bool(
            np.array_equal(raw["mean"][retained], new["mean"][retained])
        ),
        "normalized_equal_original_where_retained": bool(
            np.array_equal(raw["normalized"][retained], new["normalized"][retained])
        ),
        "search_area": float(new["marking_search"].sum() * resolution**2),
        "strict_candidate_area": float(new["strict_surface_candidate"].sum() * resolution**2),
        "confirmed_road_area": float(new["confirmed_road"].sum() * resolution**2),
        "uncertain_search_area": float(new["uncertain_road"].sum() * resolution**2),
        "new_observed_outside_strict": int(
            (retained & ~new["strict_surface_candidate"].astype(bool)).sum()
        ),
        "new_nonroad_area": None,
        "nonroad_assessment": "unknown_without_manual_boundary_reference",
        "nodata_pct_in_search": 100
        * int((new["marking_search"].astype(bool) & ~retained).sum())
        / max(int(new["marking_search"].sum()), 1),
        "filled_pixels": int(new["filled_mask"].sum()),
    }


def reviewed_samples(path, source_sha256):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("coordinate_frame") != "NATIVE_XY" or data.get("crs"):
        raise ValueError("Review must explicitly use source native XY")
    if data.get("source_sha256") != source_sha256:
        raise ValueError("Review source identity mismatch")
    features = data.get("features", [])
    if not features:
        raise ValueError("Manual review is empty")
    for feature in features:
        props = feature.get("properties", {})
        if (
            props.get("review_status") != "MANUALLY_VERIFIED"
            or not props.get("reviewer")
            or not props.get("reviewed_at")
        ):
            raise ValueError("Proposed features cannot be used as manually verified truth")
        geom = shape(feature["geometry"])
        if (
            not geom.is_valid
            or geom.is_empty
            or props.get("kind") not in ("paint", "road_boundary")
        ):
            raise ValueError("Valid attributed paint or road_boundary geometry required")
        if not np.isfinite(np.asarray(geom.bounds)).all():
            raise ValueError("Nonfinite review geometry")
    return features


def manual_retention(features, observed, retained, transform):
    strokes = [f["geometry"] for f in features if f["properties"]["kind"] == "paint"]
    boundaries = [f["geometry"] for f in features if f["properties"]["kind"] == "road_boundary"]
    if not strokes or not boundaries:
        raise ValueError("Both manually verified paint and road boundary samples required")
    mask = rasterize(
        [(g, 1) for g in strokes], out_shape=observed.shape, transform=transform, all_touched=True
    ).astype(bool)
    visible = mask & observed
    if not visible.any():
        raise ValueError("Review has no observed paint in this processing extent")
    return {
        "review_status": "MANUALLY_VERIFIED",
        "observed_paint_pixels": int(visible.sum()),
        "retained_paint_pixels": int((visible & retained).sum()),
        "observed_paint_retention": float((visible & retained).sum() / visible.sum()),
    }
