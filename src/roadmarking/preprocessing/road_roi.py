"""Manually reviewed road-corridor ROIs: strict loading, validation and frame checks.

Accepted formats: GeoPackage, Shapefile, GeoJSON, or the project's native JSON
(``{"coordinate_system": "native XY ...", "features": [...]}``).

Frame rule (the LiDAR CRS is unconfirmed): an ROI is accepted only when it declares
no CRS, or declares a CRS identical to the source header WKT. An ROI that declares a
CRS the LiDAR lacks (for example EPSG:3763) is refused unless the caller explicitly
asserts it was digitised in the LiDAR's native frame. The ROI must also overlap the
source header extent. Coordinates are never transformed, shifted or repaired.

Review rule: features whose ``review_status`` is ``PROPOSED_UNREVIEWED`` (as written by
the Phase 3.1 proposal step) are refused, so a generated proposal can never be fed back
as if it were approved.
"""

from __future__ import annotations

import json
from pathlib import Path

import shapely
from shapely.geometry import box, shape

PROPOSED = "PROPOSED_UNREVIEWED"
APPROVED_VALUES = {"APPROVED", "MANUALLY_REVIEWED", "REVIEWED"}


class RoadRoiError(ValueError):
    pass


def _read_features(path: Path) -> tuple[list, list[dict], str | None]:
    suffix = path.suffix.casefold()
    if suffix in {".json", ".geojson"}:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("type") == "FeatureCollection" or "features" in doc:
            feats = doc["features"]
        elif doc.get("type") == "Feature":
            feats = [doc]
        else:
            feats = [{"type": "Feature", "geometry": doc, "properties": {}}]
        crs = doc.get("crs")
        crs_text = json.dumps(crs) if crs else None
        geoms = [shape(f["geometry"]) if f.get("geometry") else None for f in feats]
        return geoms, [f.get("properties") or {} for f in feats], crs_text
    if suffix in {".gpkg", ".shp"}:
        import geopandas as gpd
        import pyogrio

        frames = []
        for layer, _ in pyogrio.list_layers(path):
            frames.append(gpd.read_file(path, layer=layer, engine="pyogrio"))
        crs = {f.crs.to_wkt() if f.crs is not None else None for f in frames}
        if len(crs) > 1:
            raise RoadRoiError("ROI layers declare different CRSs")
        geoms, props = [], []
        for f in frames:
            geoms += list(f.geometry)
            props += f.drop(columns="geometry").to_dict("records")
        return geoms, props, next(iter(crs))
    raise RoadRoiError(f"Unsupported ROI format: {path.suffix}")


def load_road_roi(path, header, *, assert_native_frame: bool = False):
    """Return (union polygon in native XY, report). Raises RoadRoiError on any problem."""
    path = Path(path)
    if not path.is_file():
        raise RoadRoiError(f"ROI file not found: {path}")
    geoms, props, roi_crs = _read_features(path)
    if not geoms:
        raise RoadRoiError("ROI contains no features")
    problems = []
    polygons = []
    holes = 0
    for i, (g, p) in enumerate(zip(geoms, props, strict=True)):
        status = str(p.get("review_status", "")).strip().upper()
        if status == PROPOSED:
            problems.append(f"feature {i}: review_status {PROPOSED} is not an approved ROI")
            continue
        if g is None or g.is_empty:
            problems.append(f"feature {i}: empty geometry")
            continue
        g = shapely.force_2d(g)
        if g.geom_type not in ("Polygon", "MultiPolygon"):
            problems.append(f"feature {i}: {g.geom_type} is not a polygon")
            continue
        if not g.is_valid:
            problems.append(f"feature {i}: invalid geometry ({shapely.is_valid_reason(g)})")
            continue
        parts = g.geoms if g.geom_type == "MultiPolygon" else [g]
        holes += sum(len(part.interiors) for part in parts)
        polygons.append(g)
    if problems:
        raise RoadRoiError("; ".join(problems))
    source_crs = getattr(header, "crs_wkt", None)
    if roi_crs is not None and roi_crs != source_crs and not assert_native_frame:
        raise RoadRoiError(
            "ROI declares a CRS but the LiDAR source "
            + ("has none" if source_crs is None else "declares a different one")
            + "; refusing to assume equivalence (use an explicit native-frame assertion)"
        )
    union = shapely.union_all(polygons)
    total = sum(g.area for g in polygons)
    overlap_area = total - union.area
    xmin, ymin = float(header.mins[0]), float(header.mins[1])
    xmax, ymax = float(header.maxs[0]), float(header.maxs[1])
    extent = box(xmin, ymin, xmax, ymax)
    if not union.intersects(extent):
        raise RoadRoiError(
            "ROI does not overlap the source extent in native coordinates; "
            "it is not in this source's frame"
        )
    statuses = sorted({str(p.get("review_status", "UNSPECIFIED")) for p in props})
    report = dict(
        path=str(path),
        features=len(polygons),
        holes=holes,
        overlapping_area_native_units2=float(overlap_area),
        union_area_native_units2=float(union.area),
        area_outside_source_extent_native_units2=float(union.difference(extent).area),
        declared_crs=roi_crs,
        native_frame_assertion=bool(assert_native_frame),
        review_statuses=statuses,
        review_status=(
            "APPROVED"
            if statuses and set(s.upper() for s in statuses) <= APPROVED_VALUES
            else "USER_SUPPLIED_UNSPECIFIED"
        ),
        coordinates_modified=False,
    )
    return union, report
