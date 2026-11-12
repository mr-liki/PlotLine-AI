"""Read-only annotation inspection and per-feature quality control.

Valid geometry never implies a VERIFIED label: VERIFIED requires (a) a passed raster/
vector alignment gate and (b) an explicit human review record for the feature.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from itertools import pairwise
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from shapely import STRtree
from shapely.validation import explain_validity

from roadmarking.dataset.geometry import line_parts, polygon_parts
from roadmarking.dataset.taxonomy import Taxonomy, _blank, to_float

QC_STATUSES = ("VERIFIED", "NEEDS_REVIEW", "MISALIGNED", "INCOMPLETE", "UNUSABLE")
LINE_TYPES = {"LineString", "MultiLineString"}
POLY_TYPES = {"Polygon", "MultiPolygon"}
ID_FIELD = "ID_SIG"


def _json_value(value):
    if value is None:
        return None
    if isinstance(value, float | np.floating):
        return None if math.isnan(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value if isinstance(value, int | str | bool) else str(value)


def _num(value) -> float | None:
    """Float or None; pandas turns missing numerics into NaN, which is not a value."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def read_layer(path: Path, layer: str) -> gpd.GeoDataFrame:
    # fid_as_index keeps the GeoPackage feature id as the stable source identifier.
    return gpd.read_file(path, layer=layer, engine="pyogrio", fid_as_index=True)


def inspect_file(path: Path, taxonomy: Taxonomy) -> dict:
    """Layer-level report required by Phase 4 step 1. Never writes to ``path``."""
    record: dict = {"path": str(path), "name": path.name, "layers": []}
    try:
        layers = pyogrio.list_layers(path)
    except Exception as exc:  # GDAL raises driver-specific errors
        record["status"] = "UNREADABLE"
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    record["status"] = "READ"
    for name, _gtype in layers:
        info = pyogrio.read_info(path, layer=name)
        df = read_layer(path, name)
        kinds = Counter(df.geom_type.fillna("None"))
        ids = df[ID_FIELD] if ID_FIELD in df.columns else pd.Series([None] * len(df))
        missing_ids = int(sum(_blank(v) for v in ids))
        dup_ids = int(ids[~ids.map(_blank)].duplicated().sum())
        is_line = set(kinds) <= LINE_TYPES
        class_field = "DESIGNACAO" if is_line else "COD_RST"
        class_values = (
            {str(k): int(v) for k, v in df[class_field].value_counts(dropna=False).items()}
            if class_field in df.columns
            else None
        )
        decisions = [
            (taxonomy.classify_line if is_line else taxonomy.classify_polygon)(row)
            for row in df.drop(columns="geometry").to_dict("records")
        ]
        unknown = Counter(
            str(row.get(class_field))
            for row, dec in zip(df.to_dict("records"), decisions, strict=True)
            if "unknown_class_code" in dec.flags
            or "unknown_designation" in dec.flags
            or "missing_class_code" in dec.flags
            or "missing_designation" in dec.flags
        )
        invalid = df.geometry.notna() & ~df.geometry.is_valid
        has_z = df.geometry.has_z
        z = shapely.get_coordinates(df.geometry.values, include_z=True)[:, 2] if len(df) else []
        record["layers"].append(
            {
                "layer": name,
                "feature_count": len(df),
                "declared_geometry_type": info["geometry_type"],
                "geometry_types": dict(kinds),
                "declared_crs": info["crs"],
                "extent": [float(v) for v in df.total_bounds],
                "z_available": bool(has_z.all()) if len(df) else False,
                "z_range": (
                    [float(np.nanmin(z)), float(np.nanmax(z))] if np.isfinite(z).any() else None
                ),
                "attribute_fields": {
                    k: str(v) for k, v in zip(info["fields"], info["dtypes"], strict=True)
                },
                "classification_field": class_field,
                "classification_values": class_values,
                "invalid_geometries": int(invalid.sum()),
                "invalid_details": [
                    {"fid": int(fid), "reason": explain_validity(g)}
                    for fid, g in df.geometry[invalid].items()
                ],
                "empty_geometries": int((df.geometry.isna() | df.geometry.is_empty).sum()),
                "missing_identifiers": missing_ids,
                "duplicate_identifiers": dup_ids,
                "identifier_field": ID_FIELD,
                "unknown_or_missing_class_codes": dict(unknown),
                "multipart_features": int(sum(shapely.get_num_geometries(df.geometry.values) > 1)),
                "clockwise_exterior_rings": (
                    int(
                        sum(
                            any(not p.exterior.is_ccw for p in polygon_parts(g))
                            for g in df.geometry
                        )
                    )
                    if not is_line
                    else 0
                ),
                "polygons_with_holes": (
                    int(sum(any(len(p.interiors) for p in polygon_parts(g)) for g in df.geometry))
                    if not is_line
                    else 0
                ),
            }
        )
    return record


def load_review_records(path: Path | None) -> dict[str, str]:
    """Optional human review CSV with columns global_id, status (VERIFIED/...)."""
    if path is None or not Path(path).is_file():
        return {}
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = {}
    for row in rows:
        status = row["status"].strip().upper()
        if status not in QC_STATUSES:
            raise ValueError(f"Unknown review status {status!r} in {path}")
        out[row["global_id"].strip()] = status
    return out


def normalize_features(
    files: list[Path], taxonomy: Taxonomy, width_range: tuple[float, float]
) -> gpd.GeoDataFrame:
    """One row per source feature with label levels and QC reasons; originals untouched.

    Geometry stays in its stored coordinates (no reprojection). All source attributes are
    kept verbatim in ``attributes`` (JSON) so nothing unknown is discarded.
    """
    frames = []
    for path in files:
        for layer, _ in pyogrio.list_layers(path):
            df = read_layer(path, layer)
            crs = df.crs.to_string() if df.crs is not None else None
            rows = []
            for fid, row in df.iterrows():
                attrs = {k: _json_value(v) for k, v in row.drop(labels="geometry").items()}
                geom = row.geometry
                gtype = None if geom is None else geom.geom_type
                kind = "line" if gtype in LINE_TYPES else "polygon" if gtype in POLY_TYPES else None
                if kind == "line":
                    dec = taxonomy.classify_line(attrs)
                elif kind == "polygon":
                    dec = taxonomy.classify_polygon(attrs)
                else:
                    dec = None
                rows.append(
                    {
                        "global_id": f"{path.name}|{layer}|fid={int(fid)}",
                        "source_file": path.name,
                        "source_group": path.parent.name,
                        "layer": layer,
                        "fid": int(fid),
                        "id_sig": attrs.get(ID_FIELD),
                        "declared_crs": crs,
                        "kind": kind,
                        "visual_class": dec.visual_class if dec else None,
                        "visual_id": dec.visual_id if dec else None,
                        "client_code": dec.client_code if dec else None,
                        "client_name": dec.client_name if dec else None,
                        "label_flags": list(dec.flags) if dec else ["unsupported_geometry_type"],
                        "width_m": to_float(attrs.get("LARGURA")),
                        "dash_m": to_float(attrs.get("TRAÇO")),
                        "gap_m": to_float(attrs.get("ESPAÇO")),
                        "group_id": (
                            None if _blank(attrs.get("ID_GRUPO")) else str(attrs.get("ID_GRUPO"))
                        ),
                        "elevation_attr": to_float(attrs.get("ELEVATION")),
                        "attributes": json.dumps(attrs, ensure_ascii=False, sort_keys=True),
                        "geometry": geom,
                    }
                )
            frames.append(gpd.GeoDataFrame(rows, geometry="geometry", crs=df.crs))
    if not frames:
        return gpd.GeoDataFrame(columns=["global_id", "geometry"], geometry="geometry")
    crs_values = {f.crs.to_string() if f.crs else None for f in frames}
    out = pd.concat(frames, ignore_index=True)
    out = gpd.GeoDataFrame(
        out, geometry="geometry", crs=frames[0].crs if len(crs_values) == 1 else None
    )
    out = out.sort_values("global_id", kind="stable").reset_index(drop=True)
    out["instance_uid"] = np.arange(1, len(out) + 1, dtype=np.int64)
    _geometry_qc(out, width_range)
    return out


def _geometry_qc(df: gpd.GeoDataFrame, width_range: tuple[float, float]) -> None:
    reasons: list[list[str]] = [list(f) for f in df["label_flags"]]
    stats = {"parts": [], "holes": [], "length_m": [], "area_m2": [], "z_minus_elevation": []}
    for i, (geom, kind) in enumerate(zip(df.geometry, df["kind"], strict=True)):
        r = reasons[i]
        parts = holes = 0
        length = area = dz = None
        if geom is None or geom.is_empty:
            r.append("empty_geometry")
        elif kind is None:
            r.append("unsupported_geometry_type")
        else:
            if not geom.is_valid:
                r.append("invalid_geometry")
            if not geom.has_z:
                r.append("no_z")
            else:
                z = shapely.get_coordinates(geom, include_z=True)[:, 2]
                elev = _num(df.at[i, "elevation_attr"])
                if elev is not None and np.isfinite(z).all():
                    dz = float(np.median(z) - elev)
            if kind == "line":
                p = line_parts(geom)
                parts = len(p)
                length = float(sum(x.length for x in p))
                if parts > 1 and shapely.union_all(p).geom_type != "LineString":
                    gaps = [a.distance(b) for a, b in pairwise(p)]
                    if max(gaps, default=0) > 0:
                        r.append("multipart_disjoint")
                width = _num(df.at[i, "width_m"])
                if width is None:
                    r.append("missing_width")
                elif not width_range[0] <= width <= width_range[1]:
                    r.append("implausible_width")
                else:
                    r.append("width_documented_unmeasured")
                dash = _num(df.at[i, "dash_m"])
                if dash and length is not None and abs(length - dash) / dash > 0.25:
                    r.append("dash_length_mismatch")
            else:
                p = polygon_parts(geom)
                parts = len(p)
                holes = sum(len(x.interiors) for x in p)
                area = float(sum(x.area for x in p))
        stats["parts"].append(parts)
        stats["holes"].append(holes)
        stats["length_m"].append(length)
        stats["area_m2"].append(area)
        stats["z_minus_elevation"].append(dz)
    for key, values in stats.items():
        df[key] = values
    _overlap_and_duplicate_qc(df, reasons)
    df["qc_reasons"] = [sorted(set(r)) for r in reasons]


def _overlap_and_duplicate_qc(df: gpd.GeoDataFrame, reasons: list[list[str]]) -> None:
    """Flag polygon overlaps and near-identical features repeated in different files."""
    geoms = shapely.force_2d(df.geometry.values)
    ok = np.array([g is not None and not g.is_empty and g.is_valid for g in geoms])
    idx = np.nonzero(ok)[0]
    if not len(idx):
        df["duplicate_of"] = None
        return
    tree = STRtree(geoms[idx])
    left, right = tree.query(geoms[idx], predicate="intersects")
    duplicate_of: list[str | None] = [None] * len(df)
    kinds = df["kind"].to_numpy()
    files = df["source_file"].to_numpy()
    gids = df["global_id"].to_numpy()
    for a, b in zip(idx[left], idx[right], strict=True):
        if a >= b or kinds[a] != kinds[b]:
            continue
        ga, gb = geoms[a], geoms[b]
        if shapely.hausdorff_distance(ga, gb) < 0.02:
            tag = "duplicate_in_other_file" if files[a] != files[b] else "duplicate_in_same_file"
            reasons[a].append(tag)
            reasons[b].append(tag)
            duplicate_of[b] = duplicate_of[b] or gids[a]
        elif kinds[a] == "polygon":
            inter = shapely.intersection(ga, gb).area
            if inter > 1e-4:
                reasons[a].append("polygon_overlap")
                reasons[b].append("polygon_overlap")
    df["duplicate_of"] = duplicate_of


BLOCKING = {
    "empty_geometry": "UNUSABLE",
    "unsupported_geometry_type": "UNUSABLE",
    "invalid_geometry": "UNUSABLE",
    "missing_class_code": "INCOMPLETE",
    "unknown_class_code": "INCOMPLETE",
    "missing_designation": "INCOMPLETE",
    "unknown_designation": "INCOMPLETE",
    "missing_width": "INCOMPLETE",
    "implausible_width": "INCOMPLETE",
}


def assign_status(
    df: gpd.GeoDataFrame,
    alignment_passed: bool,
    reviews: dict[str, str],
    misaligned_ids: set[str] | None = None,
) -> None:
    """Final QC status. VERIFIED only with alignment passed + explicit review record."""
    misaligned_ids = misaligned_ids or set()
    severity = {
        s: i
        for i, s in enumerate(("VERIFIED", "NEEDS_REVIEW", "MISALIGNED", "INCOMPLETE", "UNUSABLE"))
    }
    statuses, reasons_out = [], []
    for gid, reasons in zip(df["global_id"], df["qc_reasons"], strict=True):
        reasons = list(reasons)
        status = "NEEDS_REVIEW"
        for reason in reasons:
            blocked = BLOCKING.get(reason)
            if blocked and severity[blocked] > severity[status]:
                status = blocked
        if gid in misaligned_ids and severity["MISALIGNED"] > severity[status]:
            status = "MISALIGNED"
        if not alignment_passed:
            reasons.append("alignment_unverified")
        review = reviews.get(gid)
        if review is None:
            reasons.append("no_human_review_record")
        if status == "NEEDS_REVIEW" and alignment_passed and review is not None:
            status = review
        elif review in {"MISALIGNED", "INCOMPLETE", "UNUSABLE"}:
            status = review if severity[review] > severity[status] else status
        statuses.append(status)
        reasons_out.append(sorted(set(reasons)))
    df["qc_status"] = statuses
    df["qc_reasons"] = reasons_out
