"""Validation never approves annotations; eligibility needs a matching human event."""

import hashlib
import math
from collections import Counter
from pathlib import Path

from shapely.geometry import box
from shapely.ops import unary_union

from roadmarking.dataset.annotation_store import (
    STATUSES,
    current_review,
    frame_id,
    metadata,
    read_annotations,
)


def validate_record(record, meta):
    errors = []
    geometry = record.get("geometry")
    if geometry is None or geometry.is_empty or not geometry.is_valid:
        return ["invalid_or_empty_geometry"]
    for field in (
        "annotation_id",
        "source_track",
        "source_raster_id",
        "source_tile_id",
        "annotation_source",
        "review_status",
        "source_coordinate_frame",
        "purpose",
    ):
        if not record.get(field):
            errors.append(f"missing_{field}")
    if record.get("review_status") not in STATUSES:
        errors.append("unknown_review_status")
    source = record.get("source_track")
    tiles = [t for t in meta["tiles"] if t["source"] == source]
    anchor = next((t for t in tiles if t["tile_id"] == record.get("source_tile_id")), None)
    if anchor is None or record.get("source_raster_id") != record.get("source_tile_id"):
        errors.append("unknown_source_raster_or_anchor")
    elif not geometry.intersects(box(*anchor["bounds"])):
        errors.append("annotation_misses_anchor_image")
    if record.get("source_coordinate_frame") != frame_id(source):
        errors.append("coordinate_frame_mismatch")
    if tiles and not unary_union([box(*t["bounds"]) for t in tiles]).buffer(1e-8).covers(geometry):
        errors.append("geometry_outside_available_images")
    is_line = record["layer"] == "paint_centerlines"
    expected = {"LineString", "MultiLineString"} if is_line else {"Polygon", "MultiPolygon"}
    if geometry.geom_type not in expected:
        errors.append("wrong_geometry_family")
    if record.get("geometry_type") not in expected:
        errors.append("geometry_type_field_mismatch")
    if record["layer"] == "review_areas":
        if record.get("purpose") not in {"COMPLETE_REGION", "BACKGROUND", "UNCERTAIN"}:
            errors.append("invalid_review_area_purpose")
    else:
        if record.get("purpose") != "PAINT" or not record.get("global_instance_id"):
            errors.append("missing_paint_instance_identity")
        classes = {c["id"]: c for c in meta["taxonomy"]["visual_classes"]}
        entry = classes.get(record.get("visual_class_id"))
        if entry is None or entry["name"] != record.get("visual_class_name"):
            errors.append("class_id_name_mismatch")
        elif not entry["enabled_for_annotation"]:
            errors.append("class_not_located_or_catalog_unresolved")
        if is_line:
            width = record.get("marking_width")
            if (
                not isinstance(width, int | float)
                or not math.isfinite(width)
                or width <= 0
                or record.get("width_verified") != 1
                or record.get("mask_reviewed") != 1
            ):
                errors.append("centerline_needs_verified_native_width_and_area_review")
    return errors


def validate_store(path):
    meta = metadata(path)
    global_errors = []
    for filename, digest in meta["source_artifact_sha256"].items():
        source = Path(filename)
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            global_errors.append(f"source_artifact_changed:{filename}")
    records = read_annotations(path)
    ids = Counter(r.get("annotation_id") for r in records)
    global_ids = Counter(
        r.get("global_instance_id") for r in records if r["layer"] != "review_areas"
    )
    geometries = Counter(
        (r["source_track"], r["geometry"].wkb_hex) for r in records if r["geometry"] is not None
    )
    result = []
    for record in records:
        errors = validate_record(record, meta)
        if (
            record["layer"] == "review_areas"
            and record.get("purpose") == "BACKGROUND"
            and record["geometry"] is not None
            and any(
                other["layer"] != "review_areas"
                and other["source_track"] == record["source_track"]
                and other["review_status"] != "REJECTED"
                and other["geometry"] is not None
                and other["geometry"].intersection(record["geometry"]).area > 0
                for other in records
            )
        ):
            errors.append("background_overlaps_possible_paint")
        if ids[record.get("annotation_id")] > 1:
            errors.append("duplicate_annotation_id")
        if record["layer"] != "review_areas" and global_ids[record.get("global_instance_id")] > 1:
            errors.append("duplicate_global_instance_id_use_one_multipart_feature")
        if (
            record["geometry"] is not None
            and geometries[(record["source_track"], record["geometry"].wkb_hex)] > 1
        ):
            errors.append("duplicate_geometry")
        reviewed = not errors and not global_errors and current_review(path, record)
        if record.get("review_status") == "VERIFIED" and not reviewed:
            errors.append("missing_or_stale_human_review_event")
        result.append(
            {
                "annotation_id": record.get("annotation_id"),
                "layer": record["layer"],
                "review_status": record.get("review_status"),
                "errors": errors,
                "eligible": bool(reviewed and not errors),
            }
        )
    return {
        "records": result,
        "global_errors": global_errors,
        "counts_by_status": dict(Counter(r.get("review_status") for r in records)),
        "total_features": len(records),
        "eligible_features": sum(r["eligible"] for r in result),
        "invalid_features": sum(bool(r["errors"]) for r in result),
        "duplicate_annotations": sum("duplicate_geometry" in r["errors"] for r in result),
    }
