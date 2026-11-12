"""Native-XY QGIS annotation stores with append-only edit and review provenance."""

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
from shapely.geometry import box

STATUSES = ("DRAFT", "PROPOSED_UNREVIEWED", "NEEDS_CORRECTION", "VERIFIED", "REJECTED", "UNCERTAIN")
LAYERS = {
    "paint_polygons": "MultiPolygon",
    "paint_centerlines": "MultiLineString",
    "review_areas": "MultiPolygon",
}
TEXT_FIELDS = (
    "annotation_id",
    "global_instance_id",
    "source_track",
    "source_raster_id",
    "source_tile_id",
    "visual_class_name",
    "client_code",
    "geometry_type",
    "group_id",
    "annotation_source",
    "review_status",
    "reviewer_id",
    "review_timestamp",
    "confidence_or_quality_note",
    "comment",
    "source_coordinate_frame",
    "purpose",
)
INT_FIELDS = ("visual_class_id", "width_verified", "mask_reviewed")
FIELDS = (*TEXT_FIELDS, *INT_FIELDS, "marking_width")
CHECKLIST = (
    "class",
    "paint_outline",
    "placement",
    "instance_separation",
    "grouping",
    "no_false_paint",
    "no_missing_paint_in_review_area",
    "no_connections",
    "uncertainty",
)
VERSION = "phase4-verified-native-v2"


def utc_now():
    return datetime.now(UTC).isoformat()


def create_store(path, tiles, taxonomy):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for layer, geometry_type in LAYERS.items():
        columns = {k: pd.Series(dtype="str") for k in TEXT_FIELDS}
        columns.update({k: pd.Series(dtype="Int64") for k in INT_FIELDS})
        columns["marking_width"] = pd.Series(dtype="float64")
        frame = gpd.GeoDataFrame(columns, geometry=gpd.GeoSeries([], crs=None))
        pyogrio.write_dataframe(
            frame,
            path,
            layer=layer,
            driver="GPKG",
            geometry_type=geometry_type,
            layer_options={"SPATIAL_INDEX": "NO"},
        )
    regions = gpd.GeoDataFrame(
        [
            {
                "tile_id": t["tile_id"],
                "source_track": t["source"],
                "coordinate_frame": frame_id(t["source"]),
                "role": "PLANNING_FOOTPRINT_NOT_GROUND_TRUTH",
                "geometry": box(*t["bounds"]),
            }
            for t in tiles
        ],
        crs=None,
    )
    pyogrio.write_dataframe(
        regions,
        path,
        layer="annotation_regions",
        driver="GPKG",
        layer_options={"SPATIAL_INDEX": "NO"},
    )
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE project_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        paths = set()
        for tile in tiles:
            paths.update(
                path for field in ("layers", "images", "masks") for path in tile[field].values()
            )
            paths.update(
                path
                for window in tile.get("source_windows", [])
                for path in window["rasters"].values()
            )
        source_hashes = {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths
        }
        for key, value in {
            "version": VERSION,
            "coordinate_status": "CRS_UNCONFIRMED",
            "tiles": tiles,
            "taxonomy": taxonomy,
            "source_artifact_sha256": source_hashes,
        }.items():
            db.execute("INSERT INTO project_metadata VALUES (?,?)", (key, json.dumps(value)))
        db.execute("""CREATE TABLE annotation_audit (event_id INTEGER PRIMARY KEY,
            layer_name TEXT, annotation_id TEXT, action TEXT, recorded_at TEXT,
            old_geometry BLOB, new_geometry BLOB, old_attributes TEXT, new_attributes TEXT)""")
        db.execute("""CREATE TABLE review_events (event_id INTEGER PRIMARY KEY,
            layer_name TEXT NOT NULL, annotation_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
            status TEXT NOT NULL, reviewer_id TEXT NOT NULL, review_timestamp TEXT NOT NULL,
            checklist_json TEXT NOT NULL)""")
        for table in ("annotation_audit", "review_events"):
            for action in ("UPDATE", "DELETE"):
                db.execute(f"""CREATE TRIGGER immutable_{table}_{action} BEFORE {action} ON {table}
                    BEGIN SELECT RAISE(ABORT, 'Audit history is append-only'); END""")
        for layer in LAYERS:
            for action in ("INSERT", "UPDATE", "DELETE"):
                old = "NULL" if action == "INSERT" else "OLD.geom"
                new = "NULL" if action == "DELETE" else "NEW.geom"
                ref = "OLD" if action == "DELETE" else "NEW"

                def attrs(prefix):
                    return (
                        "json_object(" + ",".join(f"'{f}',{prefix}.\"{f}\"" for f in FIELDS) + ")"
                    )

                old_attrs = "NULL" if action == "INSERT" else attrs("OLD")
                new_attrs = "NULL" if action == "DELETE" else attrs("NEW")
                db.execute(f"""CREATE TRIGGER audit_{layer}_{action} AFTER {action} ON {layer}
                    BEGIN INSERT INTO annotation_audit
                    (layer_name,annotation_id,action,recorded_at,old_geometry,new_geometry,
                     old_attributes,new_attributes) VALUES ('{layer}',{ref}.annotation_id,
                     '{action}',strftime('%Y-%m-%dT%H:%M:%fZ','now'),{old},{new},
                     {old_attrs},{new_attrs}); END""")
    return path


def frame_id(source):
    return f"NATIVE_XY_CRS_UNCONFIRMED:{source}"


def metadata(path):
    with sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True) as db:
        result = {
            key: json.loads(value)
            for key, value in db.execute("SELECT key,value FROM project_metadata")
        }
    if result.get("version") != VERSION:
        raise ValueError("Unsupported annotation workspace version")
    return result


def clean(value):
    return None if pd.isna(value) else value.item() if hasattr(value, "item") else value


def fingerprint(record):
    fields = {
        k: clean(record.get(k))
        for k in FIELDS
        if k not in {"review_status", "reviewer_id", "review_timestamp"}
    }
    fields["geometry_wkb"] = record["geometry"].wkb_hex
    return hashlib.sha256(json.dumps(fields, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_annotations(path):
    records = []
    for layer in LAYERS:
        frame = gpd.read_file(path, layer=layer, engine="pyogrio")
        for row in frame.to_dict("records"):
            records.append(
                {
                    **{k: clean(v) for k, v in row.items() if k != "geometry"},
                    "geometry": row["geometry"],
                    "layer": layer,
                }
            )
    return records


def current_review(path, record):
    with sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True) as db:
        row = db.execute(
            """SELECT fingerprint,status,reviewer_id,review_timestamp,checklist_json
            FROM review_events WHERE layer_name=? AND annotation_id=?
            ORDER BY event_id DESC LIMIT 1""",
            (record["layer"], record["annotation_id"]),
        ).fetchone()
    return bool(
        row
        and row[0] == fingerprint(record)
        and row[1] == "VERIFIED"
        and record["review_status"] == "VERIFIED"
        and record["reviewer_id"] == row[2]
        and record["review_timestamp"] == row[3]
        and all(json.loads(row[4]).get(k) is True for k in CHECKLIST)
    )


def record_review(path, annotation_id, reviewer_id, status, checklist):
    """Called explicitly by the human reviewer; never called by auto-validation."""
    if status not in STATUSES or not reviewer_id.strip():
        raise ValueError("Known status and attributed human reviewer required")
    records = [r for r in read_annotations(path) if r["annotation_id"] == annotation_id]
    if len(records) != 1:
        raise ValueError("Annotation ID must select exactly one feature")
    record = records[0]
    if status == "VERIFIED":
        if not all(checklist.get(k) is True for k in CHECKLIST):
            raise ValueError("Every human verification checklist item must be explicitly true")
        from roadmarking.dataset.annotation_validation import validate_record

        errors = validate_record(record, metadata(path))
        from roadmarking.dataset.annotation_validation import validate_store

        store_check = validate_store(path)
        errors.extend(store_check["global_errors"])
        errors.extend(
            error
            for r in store_check["records"]
            if r["annotation_id"] == annotation_id
            for error in r["errors"]
            if error != "missing_or_stale_human_review_event"
        )
        if errors:
            raise ValueError(errors)
    timestamp = utc_now()
    with sqlite3.connect(path) as db:
        db.execute(
            f'UPDATE "{record["layer"]}" SET review_status=?,reviewer_id=?,review_timestamp=? '
            "WHERE annotation_id=?",
            (status, reviewer_id, timestamp, annotation_id),
        )
        db.execute(
            "INSERT INTO review_events VALUES (NULL,?,?,?,?,?,?,?)",
            (
                record["layer"],
                annotation_id,
                fingerprint(record),
                status,
                reviewer_id,
                timestamp,
                json.dumps(checklist, sort_keys=True),
            ),
        )
    return timestamp


def snapshot(source, destination):
    """Consistent GeoPackage import/export copy, retaining all audit history."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    metadata(source)  # Reject an unrelated client GeoPackage.
    with sqlite3.connect(source) as src, sqlite3.connect(destination) as dst:
        src.backup(dst)
    return str(destination)
