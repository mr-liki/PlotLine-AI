"""Read-only native-coordinate audit of supplied annotations and raw survey headers."""

import argparse
import hashlib
import io
import json
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pyogrio
from shapely import STRtree
from shapely.geometry import box
from shapely.ops import unary_union

from roadmarking.rasterization.pipeline import dump, sha256

ANNOTATIONS = (
    r"\\Nast\naksha\DAILY ACTIVITY REPORT\!!!!!geetha\2026\NT895\2026"
    r"\Highway inventory project - A4\Road_marking"
)
RAW = r"\\192.168.10.6\d\NT895\A6\LAS new"


def header_record(path):
    try:
        stat = path.stat()
        with path.open("rb") as stream:
            prefix = stream.read(227)
            if len(prefix) < 227 or prefix[:4] != b"LASF":
                raise ValueError("Missing LASF signature or truncated fixed header")
            offset = int.from_bytes(prefix[96:100], "little")
            if offset < 227 or offset > min(stat.st_size, 4 * 1024**2):
                raise ValueError(f"Invalid or over-budget point-data offset: {offset}")
            payload = prefix + stream.read(offset - len(prefix))
            if len(payload) != offset:
                raise ValueError("Truncated header/VLR region")
            h = laspy.LasHeader.read_from(io.BytesIO(payload), read_evlrs=False)
            valid = bool(
                h.point_count
                and np.isfinite(h.mins).all()
                and np.isfinite(h.maxs).all()
                and (h.maxs >= h.mins).all()
                and (h.scales > 0).all()
            )
            crs = h.parse_crs()
            required = h.offset_to_point_data + h.point_count * h.point_format.size
            return {
                "path": str(path),
                "name": path.name,
                "size_bytes": stat.st_size,
                "modified_ns": stat.st_mtime_ns,
                "point_count": h.point_count,
                "xyz_min": list(h.mins),
                "xyz_max": list(h.maxs),
                "native_xy_bbox": [h.mins[0], h.mins[1], h.maxs[0], h.maxs[1]],
                "crs": crs.to_wkt() if crs else None,
                "valid_header_bounds": valid,
                "compressed": h.are_points_compressed,
                "declared_records_fit_file": (
                    h.offset_to_point_data < stat.st_size
                    if h.are_points_compressed
                    else required <= stat.st_size
                ),
                "full_point_integrity_validated": False,
            }
    except Exception as exc:
        return {
            "path": str(path),
            "error": f"{type(exc).__name__}: {exc}",
            "valid_header_bounds": False,
        }


def run(annotation_root, raw_root, project, *, recursive=False):
    started = time.perf_counter()
    ident = hashlib.sha256(str(raw_root).encode()).hexdigest()[:8]
    out = (
        project
        / "outputs/reports/phase4_source_alignment"
        / (time.strftime("%Y%m%dT%H%M%S") + "_" + ident)
    )
    out.mkdir(parents=True, exist_ok=False)
    entries = raw_root.rglob("*") if recursive else raw_root.iterdir()
    paths = sorted(p for p in entries if p.suffix.lower() in {".las", ".laz"})
    print(f"Reading {len(paths)} LAS/LAZ headers only", flush=True)
    headers = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, record in enumerate(pool.map(header_record, paths), 1):
            headers.append(record)
            if index % 50 == 0:
                print(f"Headers inspected: {index}/{len(paths)}", flush=True)
    dump(out / "raw_header_inventory.json", headers)
    valid = [h for h in headers if h["valid_header_bounds"] and h.get("declared_records_fit_file")]
    footprints = [box(*h["native_xy_bbox"]) for h in valid]
    tree = STRtree(footprints)
    extent = unary_union(footprints)
    layers, matching_files = [], Counter()
    for path in sorted(annotation_root.rglob("*.gpkg")):
        digest = sha256(path)
        relative = path.relative_to(annotation_root)
        local = project / "data/raw/annotations" / relative
        same_local = local.is_file() and sha256(local) == digest
        # GDAL's path parser treats exclamation marks in this UNC directory as
        # archive separators. Inspect an identical existing copy or a new snapshot.
        readable = local if same_local else out / "input_snapshots" / relative
        if not same_local:
            readable.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, readable)
            if sha256(readable) != digest:
                raise ValueError("Annotation snapshot identity mismatch")
        for name, geometry_type in pyogrio.list_layers(readable):
            if geometry_type is None:
                continue
            frame = gpd.read_file(readable, layer=name, engine="pyogrio")
            geoms = frame.geometry
            usable = geoms.notna() & ~geoms.is_empty & geoms.is_valid
            subset = geoms[usable]
            hit = tree.query(subset.array, predicate="intersects")
            for index, count in Counter(hit[1]).items():
                matching_files[valid[int(index)]["path"]] += count
            fields = {
                field: {str(k): int(v) for k, v in frame[field].value_counts(dropna=False).items()}
                for field in ("DESIGNACAO", "COD_RST")
                if field in frame
            }
            distances = subset.distance(extent)
            layers.append(
                {
                    "path": str(path),
                    "layer": name,
                    "sha256": digest,
                    "identical_to_existing_local_copy": same_local,
                    "readable_inspection_copy": str(readable),
                    "features": len(frame),
                    "valid_features": int(usable.sum()),
                    "declared_crs": frame.crs.to_wkt() if frame.crs else None,
                    "native_xy_bbox": frame.total_bounds.tolist(),
                    "features_intersecting_raw_header_footprints": int(np.unique(hit[0]).size),
                    "minimum_native_xy_distance_to_raw_footprints": float(distances.min()),
                    "class_attributes": fields,
                    "human_pixel_area_review_verified": False,
                }
            )
            print(
                f"{relative} / {name}: {len(frame)} features; "
                f"{np.unique(hit[0]).size} intersect source footprints",
                flush=True,
            )
        if sha256(path) != digest:
            raise ValueError("Annotation source changed during audit")
    hits = sum(layer["features_intersecting_raw_header_footprints"] for layer in layers)
    status = (
        "OVERLAPPING_HEADER_FOOTPRINTS_REQUIRE_PIXEL_ALIGNMENT_REVIEW"
        if hits
        else "BLOCKED_NO_NATIVE_XY_OVERLAP"
    )
    report = {
        "status": status,
        "annotation_root": str(annotation_root),
        "raw_root": str(raw_root),
        "raw_files": len(headers),
        "scan_recursive": recursive,
        "valid_header_files": len(valid),
        "raw_batches": dict(Counter(h["name"].split("_Track")[0] for h in valid)),
        "raw_native_xy_union_bounds": list(extent.bounds),
        "annotation_layers": layers,
        "annotation_features": sum(layer["features"] for layer in layers),
        "annotation_features_intersecting_headers": hits,
        "matching_raw_files": dict(matching_files),
        "seconds": time.perf_counter() - started,
        "no_coordinate_transform_applied": True,
        "full_raw_files_copied": False,
        "training_or_inference_run": False,
        "report_directory": str(out),
        "limitations": [
            "Header footprints establish possible coverage, not point integrity or paint alignment",
            "Manual GIS geometry still needs visible-paint segmentation review",
            "No source transformations or guessed CRS; report compares stored native XY only",
        ],
    }
    dump(out / "report.json", report)
    text = [
        "# Supplied annotation / raw-survey alignment audit",
        "",
        f"Status: **{status}**",
        "",
        f"Raw files: {len(headers)}; usable headers: {len(valid)}.",
        f"Annotation features: {report['annotation_features']}; "
        f"intersecting raw header footprints: {hits}.",
        "",
        "These shares are read-only inputs. The previous Phase 3/4 products are preserved. "
        "No point data were rasterized and no labels were approved.",
        "",
        "| Annotation file | Features | Native XY footprint hits | Same as local copy |",
        "|---|---:|---:|---|",
    ]
    text.extend(
        f"| {Path(layer['path']).name} | {layer['features']} "
        f"| {layer['features_intersecting_raw_header_footprints']} "
        f"| {layer['identical_to_existing_local_copy']} |"
        for layer in layers
    )
    text += [
        "",
        "Coordinate overlap must be resolved before preparing training masks. "
        "Header overlap still requires an intensity/vector overlay and paint-area review.",
        "",
    ]
    (out / "report.md").write_text("\n".join(text), encoding="utf-8")
    dump(
        out.parent / "latest.json",
        {"report": str(out / "report.json"), "report_directory": str(out)},
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path(ANNOTATIONS))
    parser.add_argument("--raw", type=Path, default=Path(RAW))
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()
    report = run(
        args.annotations, args.raw, Path(__file__).resolve().parents[1], recursive=args.recursive
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "raw_files",
                    "annotation_features",
                    "annotation_features_intersecting_headers",
                    "report_directory",
                )
            },
            indent=2,
        )
    )
