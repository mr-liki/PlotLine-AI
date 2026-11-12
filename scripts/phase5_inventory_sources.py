"""Live read-only inventory, actual-record sampling and annotation approval audit."""

import json
import shutil
import sqlite3
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import shapely
import yaml
from shapely.geometry import box
from shapely.validation import explain_validity

from roadmarking.io.source_inventory import discover, inspect_records
from roadmarking.rasterization.pipeline import dump, sha256


def run(root):
    started = time.perf_counter()
    config = yaml.safe_load((root / "configs/phase4_network_sources.yaml").read_text())
    annotation_root = Path(config["annotation_root"])
    raw_root = Path(config["candidate_raw_roots"][0])
    out = root / "outputs/reports/phase5" / (time.strftime("%Y%m%dT%H%M%S") + "_source_audit")
    out.mkdir(parents=True, exist_ok=False)
    inventories = {"annotations": discover(annotation_root), "raw": discover(raw_root)}
    dump(out / "file_inventory.json", inventories)
    paths = [
        Path(x["path"])
        for x in inventories["raw"]
        if x.get("kind") == "file" and x["extension"] in {".las", ".laz"}
    ]
    print(f"Auditing actual point windows in {len(paths)} raw files", flush=True)
    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, record in enumerate(pool.map(inspect_records, paths), 1):
            records.append(record)
            dump(out / "raw_integrity.json", records)
            if index % 10 == 0 or index == len(paths):
                print(f"Actual-record audit {index}/{len(paths)}", flush=True)
    boxes = [
        box(r["mins"][0], r["mins"][1], r["maxs"][0], r["maxs"][1])
        for r in records
        if r["status"] == "SAMPLE_QC_PASS_FULL_SCAN_PENDING"
    ]
    tree = shapely.STRtree(boxes)
    layers, hashes = [], {}
    for entry in inventories["annotations"]:
        if entry.get("kind") != "file":
            continue
        source = Path(entry["path"])
        hashes[str(source)] = sha256(source)
        if source.suffix.lower() not in {".gpkg", ".geojson", ".shp"}:
            continue
        # GDAL interprets the exclamation marks in this share as archive separators.
        local = out / "annotation_snapshots" / source.relative_to(annotation_root)
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, local)
        if source.suffix.lower() == ".shp":
            for suffix in (".shx", ".dbf", ".prj", ".cpg"):
                sidecar = source.with_suffix(suffix)
                if not sidecar.exists() and suffix in {".shx", ".dbf"}:
                    raise ValueError(f"Missing required shapefile sidecar: {sidecar}")
                if sidecar.exists():
                    shutil.copyfile(sidecar, local.with_suffix(suffix))
        if sha256(local) != hashes[str(source)]:
            raise ValueError("Annotation snapshot checksum mismatch")
        for name, kind in pyogrio.list_layers(local):
            if kind is None:
                continue
            frame = gpd.read_file(local, layer=name, fid_as_index=True)
            valid = frame.geometry.notna() & ~frame.geometry.is_empty & frame.geometry.is_valid
            geoms = frame.geometry[valid]
            hits = tree.query(geoms.array, predicate="intersects")
            widths = {}
            for field in frame.columns:
                if any(token in field.lower() for token in ("larg", "width", "review", "aprov")):
                    widths[field] = {
                        str(k): int(v)
                        for k, v in frame[field].value_counts(dropna=False).head(30).items()
                    }
            tables = []
            if local.suffix.lower() == ".gpkg":
                with sqlite3.connect(f"file:{local.as_posix()}?mode=ro", uri=True) as db:
                    tables = [
                        r[0]
                        for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    ]
            coords = shapely.get_coordinates(frame.geometry.array, include_z=True)
            layers.append(
                {
                    "source": str(source),
                    "local_snapshot": str(local),
                    "layer": name,
                    "sha256": hashes[str(source)],
                    "features": len(frame),
                    "valid_features": int(valid.sum()),
                    "geometry_types": dict(Counter(frame.geom_type)),
                    "fields": {str(k): str(v) for k, v in frame.dtypes.items()},
                    "bounds": frame.total_bounds.tolist(),
                    "crs": str(frame.crs),
                    "native_header_hits": len(np.unique(hits[0])),
                    "duplicate_geometry_wkb": int(frame.geometry.to_wkb().duplicated().sum()),
                    "duplicate_source_ids": (
                        int(frame["ID_SIG"].dropna().duplicated().sum())
                        if "ID_SIG" in frame
                        else None
                    ),
                    "class_attributes": {
                        field: {
                            str(k): int(v)
                            for k, v in frame[field].value_counts(dropna=False).items()
                        }
                        for field in ("DESIGNACAO", "COD_RST")
                        if field in frame
                    },
                    "width_and_approval_attributes": widths,
                    "database_tables": tables,
                    "finite_z_vertices": int(np.isfinite(coords[:, 2]).sum()),
                    "vertices": len(coords),
                    "invalid": [
                        {"fid": int(fid), "reason": explain_validity(g)}
                        for fid, g in frame.geometry[~valid].items()
                    ],
                    "paint_area_approval_evidence": "NOT_ESTABLISHED_BY_AVAILABLE_FILES",
                }
            )
    changed = [p for p, h in hashes.items() if sha256(Path(p)) != h]
    if changed:
        raise ValueError("Annotation sources changed during audit")
    duplicate_keys = Counter(
        (
            r.get("point_count"),
            tuple(r.get("mins", [])),
            tuple(r.get("maxs", [])),
            r.get("fingerprint", {}).get("sha256_partial"),
        )
        for r in records
    )
    summary = {
        "annotation_root": str(annotation_root),
        "raw_root": str(raw_root),
        "annotation_path_note": (
            "Phase 5 document omits separator before !!!!!geetha; "
            "used existing user-confirmed path"
        ),
        "report_directory": str(out),
        "raw_files": len(records),
        "raw_statuses": dict(Counter(r["status"] for r in records)),
        "sampled_valid_points": sum(r.get("sample_points_passed", 0) for r in records),
        "full_integrity_validated_files": 0,
        "annotation_layers": layers,
        "source_annotation_hashes": hashes,
        "source_annotation_hash_mismatches": len(changed),
        "candidate_features": sum(x["features"] for x in layers),
        "valid_features": sum(x["valid_features"] for x in layers),
        "native_header_hits": sum(x["native_header_hits"] for x in layers),
        "potential_duplicate_fingerprint_groups": sum(v > 1 for v in duplicate_keys.values()),
        "full_file_duplicate_hashes_established": False,
        "human_paint_area_approvals_established": 0,
        "annotation_source_claim": "User identifies manually digitized verified client GIS source",
        "approval_limit": (
            "No supplied record establishes paint width, buffered-area approval "
            "or complete reviewed background"
        ),
        "dgn_support": (
            "DGNv8 previously unsupported by installed GDAL; preserved as source references"
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    dump(out / "report.json", summary)
    dump(
        root / "outputs/reports/phase5/source_audit_latest.json",
        {"report": str(out / "report.json")},
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "raw_files",
                    "raw_statuses",
                    "valid_features",
                    "native_header_hits",
                    "elapsed_seconds",
                )
            },
            indent=2,
        )
    )
    return summary


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[1])
