"""Make an explicit UNVERIFIED rigid-placement copy; never creates training labels."""

import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import laspy
import matplotlib
import numpy as np
import pyogrio
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import box

from roadmarking.dataset.rigid_registration import (
    apply_xy,
    distance_stats,
    fit_rigid_multistart,
    grid_sample,
    transform_geometry_xy,
)
from roadmarking.io.pointcloud import point_validity
from roadmarking.rasterization.pipeline import dump, sha256

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sample_file(record, *, windows=8, size=512):
    path = Path(record["path"])
    try:
        with path.open("rb") as stream:
            prefix = stream.read(227)
            offset = int.from_bytes(prefix[96:100], "little")
            if prefix[:4] != b"LASF" or not 227 <= offset <= 4 * 1024**2:
                raise ValueError("Invalid or over-budget header")
            header = laspy.LasHeader.read_from(io.BytesIO(prefix + stream.read(offset - 227)))
            if header.are_points_compressed:
                raise ValueError("Only uncompressed LAS accepted by bounded sampler")
            starts = np.unique(
                np.linspace(0, max(0, header.point_count - size), windows, dtype=np.int64)
            )
            xyzs = []
            intensities = []
            positions = []
            for start in starts:
                n = min(size, header.point_count - int(start))
                stream.seek(offset + int(start) * header.point_format.size)
                payload = stream.read(n * header.point_format.size)
                if len(payload) != n * header.point_format.size:
                    raise ValueError("Truncated sample")
                packed = laspy.PackedPointRecord.from_buffer(payload, header.point_format)
                points = laspy.ScaleAwarePointRecord(
                    packed.array, header.point_format, header.scales, header.offsets
                )
                valid = point_validity(
                    points,
                    header.mins,
                    header.maxs,
                    gps_week_time=not bool(header.global_encoding.gps_time_type),
                )
                if not valid.all():
                    raise ValueError(f"Sample QC failed: {int((~valid).sum())}/{n} at {start}")
                xyzs.append(np.column_stack([points.x, points.y, points.z]))
                intensities.append(np.asarray(points.intensity).copy())
                positions.append(int(start))
        xyz = np.concatenate(xyzs)
        intensity = np.concatenate(intensities)
        cutoff = float(np.quantile(intensity, 0.95))
        bright = intensity >= cutoff
        return {
            "source": str(path),
            "status": "SAMPLED_QC_PASSED_NOT_FULLY_VALIDATED",
            "windows": positions,
            "sample_points": len(xyz),
            "bright_proxy_cutoff": cutoff,
            "bright_proxy_points": int(bright.sum()),
            "size_bytes": path.stat().st_size,
            "modified_ns": path.stat().st_mtime_ns,
            "xy": xyz[:, :2],
            "bright_xy": xyz[bright, :2],
        }
    except Exception as exc:
        return {
            "source": str(path),
            "status": "EXCLUDED_SAMPLE_QC_OR_READ_ERROR",
            "error": f"{type(exc).__name__}: {exc}",
        }


def run(root):
    started = time.perf_counter()
    audit_base = root / "outputs/reports/phase4_source_alignment"
    audits = []
    for path in sorted(audit_base.glob("*/report.json")):
        data = json.loads(path.read_text())
        if data.get("raw_root", "").endswith(r"A6\LAS new") and "annotation_layers" in data:
            audits.append(data)
    audit = audits[-1]
    headers = json.loads(
        (Path(audit["report_directory"]) / "raw_header_inventory.json").read_text()
    )
    previous = json.loads((audit_base / "exact_pair_latest.json").read_text())
    sample_qc = json.loads((Path(previous["report_directory"]) / "report.json").read_text())
    known_bad = set(sample_qc["sampled_files_with_xyz_anomalies"])
    sources = [
        h
        for h in headers
        if h["valid_header_bounds"]
        and h.get("declared_records_fit_file")
        and h["path"] not in known_bad
    ]
    run_id = time.strftime("%Y%m%dT%H%M%S") + "_rigid_unverified"
    out = root / "data/processed/phase4_fitted_annotations" / run_id
    for folder in ("metadata", "previews", "source_references"):
        (out / folder).mkdir(parents=True, exist_ok=False)
    report_dir = root / "outputs/reports/phase4_fitted_annotations"
    report_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = {}
    frames = []
    fit_coordinates = []
    for layer in audit["annotation_layers"]:
        original = Path(layer["path"])
        local = Path(layer["readable_inspection_copy"])
        digest = sha256(original)
        if digest != layer["sha256"] or sha256(local) != digest:
            raise ValueError("Annotation identity changed")
        source_hashes[str(original)] = digest
        frame = gpd.read_file(local, layer=layer["layer"], fid_as_index=True, engine="pyogrio")
        frames.append((layer, frame))
        if "Longitudinal" in local.name:
            valid = frame.geometry.notna() & ~frame.geometry.is_empty & frame.geometry.is_valid
            dense = shapely.segmentize(
                shapely.force_2d(frame.geometry[valid].array), max_segment_length=50
            )
            fit_coordinates.append(shapely.get_coordinates(dense))
    source_xy = grid_sample(np.concatenate(fit_coordinates), 50)
    records = []
    fit_points = []
    check_points = []
    all_points = []
    print(f"Sampling {len(sources)} LAS files with bounded record windows", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, result in enumerate(pool.map(sample_file, sources)):
            if "xy" in result:
                all_points.append(grid_sample(result.pop("xy"), 10))
                proxy = grid_sample(result.pop("bright_xy"), 2)
                (check_points if index % 5 == 0 else fit_points).append(proxy)
                result["partition"] = (
                    "CHECK_RECORDS_NOT_SPATIALLY_INDEPENDENT" if index % 5 == 0 else "FIT_RECORDS"
                )
            records.append(result)
            if (index + 1) % 50 == 0:
                print(f"Sampled {index+1}/{len(sources)} files", flush=True)
    dump(out / "metadata/raw_sampling.json", records)
    if not fit_points or not check_points:
        raise ValueError("Insufficient QC-passed sample evidence")
    target = grid_sample(np.concatenate(fit_points), 2)
    check = grid_sample(np.concatenate(check_points), 2)
    background = grid_sample(np.concatenate(all_points), 5)
    np.savez_compressed(
        out / "metadata/registration_samples.npz",
        source=source_xy,
        target=target,
        check=check,
        observed_sample=background,
    )
    print(
        f"Registering {len(source_xy)} annotation cells to {len(target)} bright-proxy cells",
        flush=True,
    )
    fit = fit_rigid_multistart(source_xy, target)
    rotation, translation = fit["rotation"], fit["translation"]
    matrix = np.eye(3)
    matrix[:2, :2] = rotation
    matrix[:2, 2] = translation
    transformed = apply_xy(source_xy, rotation, translation)
    before = distance_stats(source_xy, target)
    after = distance_stats(transformed, target)
    check_stats = distance_stats(transformed, check)
    gpkg = out / "annotations_rigid_fit_UNVERIFIED.gpkg"
    exported = []
    for index, (layer, frame) in enumerate(frames):
        original = Path(layer["path"])
        result = frame.copy()
        result["fit_source_geometry_valid"] = result.geometry.is_valid.astype(int)
        result.geometry = result.geometry.map(
            lambda geom: transform_geometry_xy(geom, rotation, translation)
        )
        result = result.set_crs(None, allow_override=True)
        result["fit_source_file"] = str(original)
        result["fit_source_layer"] = layer["layer"]
        result["fit_source_fid"] = result.index.astype(str)
        result["fit_global_id"] = [
            f"{original.parent.name}/{original.name}|{layer['layer']}|fid={fid}"
            for fid in result.index
        ]
        result["fit_status"] = "UNVERIFIED_GEOMETRIC_PLACEMENT"
        result["fit_review_status"] = "PROPOSED_UNREVIEWED"
        result["fit_training_eligible"] = 0
        result["fit_z_status"] = "ORIGINAL_Z_PRESERVED_NOT_REGISTERED"
        kind = "longitudinal" if "Longitudinal" in original.name else "other"
        name = f"fit_{index+1}_{original.parent.name.lower()}_{kind}"
        pyogrio.write_dataframe(result, gpkg, layer=name, driver="GPKG")
        readback = gpd.read_file(gpkg, layer=name, engine="pyogrio")
        if len(readback) != len(frame):
            raise ValueError("Export changed feature count")
        exported.append(
            {
                "layer": name,
                "features": len(result),
                "source": str(original),
                "source_sha256": source_hashes[str(original)],
            }
        )
    footprint = gpd.GeoDataFrame(
        [
            {
                "source_las": h["path"],
                "role": "HEADER_FOOTPRINT_NOT_VERIFIED_ROAD",
                "geometry": box(*h["native_xy_bbox"]),
            }
            for h in sources
        ],
        crs=None,
    )
    pyogrio.write_dataframe(footprint, gpkg, layer="raw_header_footprints", driver="GPKG")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].scatter(
        background[:, 0], background[:, 1], s=1, c="#9ca3af", label="QC-passed sampled observations"
    )
    axes[0].scatter(
        target[:, 0],
        target[:, 1],
        s=2,
        c="#2563eb",
        label="Sampled bright proxy (not verified paint)",
    )
    axes[0].scatter(
        transformed[:, 0],
        transformed[:, 1],
        s=3,
        c="#ea580c",
        label="Rigidly fitted annotation cells",
    )
    axes[0].set_title("UNVERIFIED global rigid fit — native XY")
    axes[0].legend(fontsize=7)
    axes[0].set_aspect("equal", adjustable="box")
    axes[1].plot(
        np.sort(cKDTree(target).query(transformed)[0]), np.linspace(0, 100, len(transformed))
    )
    axes[1].set_xlabel("Nearest sampled bright-proxy distance (native units)")
    axes[1].set_ylabel("Annotation cells (%)")
    axes[1].set_title(f"Residual distribution; median {after['p50']:.2f}, p95 {after['p95']:.2f}")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("DIAGNOSTIC ONLY — no ground-truth or paint-alignment approval")
    fig.tight_layout()
    preview = out / "previews/rigid_fit_review.png"
    fig.savefig(preview, dpi=150)
    plt.close(fig)
    if not all(sha256(Path(p)) == h for p, h in source_hashes.items()):
        raise ValueError("Original annotations changed")
    report = {
        "status": "UNVERIFIED_RIGID_FIT_EXPORTED",
        "output_root": str(out),
        "geopackage": str(gpkg),
        "raw_root": audit["raw_root"],
        "annotation_root": audit["annotation_root"],
        "method": (
            "Deterministic multistart trimmed nearest-neighbor rigid ICP; sampled "
            "high-intensity points are proxies, not landmark correspondences"
        ),
        "scale": 1.0,
        "reflection": False,
        "rotation_degrees": float(np.rad2deg(np.arctan2(rotation[1, 0], rotation[0, 0]))),
        "transform_3x3": matrix.tolist(),
        "inverse_transform_3x3": np.linalg.inv(matrix).tolist(),
        "source_fit_cells": len(source_xy),
        "target_bright_proxy_cells": len(target),
        "check_bright_proxy_cells": len(check),
        "before_native_distances": before,
        "fitted_native_distances": after,
        "check_native_distances": check_stats,
        "fit_candidates": fit["candidates"],
        "trim_fraction": 0.9,
        "raw_sample_files": len(records),
        "raw_sample_qc_passed": sum(r["status"].startswith("SAMPLED") for r in records),
        "raw_sample_excluded": sum(r["status"].startswith("EXCLUDED") for r in records),
        "previously_known_bad_files_skipped": sorted(known_bad),
        "sampled_points": sum(r.get("sample_points", 0) for r in records),
        "exported_layers": exported,
        "features_exported": sum(r["features"] for r in exported),
        "source_hashes": source_hashes,
        "original_annotations_unchanged": True,
        "preview": str(preview),
        "training_eligible": False,
        "alignment_verified": False,
        "masks_generated": 0,
        "limitations": [
            (
                "This fit is an arbitrary geometric placement, not a CRS conversion or "
                "surveyed registration."
            ),
            (
                "Source geometry and width preserved at scale 1; original Z retained "
                "without registration."
            ),
            (
                "Bright samples include non-paint; source files can overlap "
                "geographically across fit/check partitions."
            ),
            "Nearest sparse-sample distances cannot prove subpixel paint alignment.",
            "No scale/shear/nonrigid warp used to force differently shaped road sections together.",
            (
                "No exhaustive raw-source integrity validation or full-resolution "
                "paint-retention review."
            ),
        ],
        "processing_seconds": time.perf_counter() - started,
    }
    dump(out / "metadata/fit_report.json", report)
    dump(report_dir / "report.json", report)
    dump(
        report_dir / "latest.json",
        {"output_root": str(out), "report": str(report_dir / "report.json")},
    )
    lines = [
        "# Requested geometric placement of existing annotations",
        "",
        "Status: **UNVERIFIED RIGID FIT EXPORTED**.",
        "",
        f"GeoPackage: `{gpkg}`",
        "",
        f"Features: {report['features_exported']:,}; scale 1; "
        f"rotation {report['rotation_degrees']:.6f} degrees.",
        f"Translation X/Y: {translation[0]:.6f}, {translation[1]:.6f} native units.",
        "",
        f"Median nearest bright-sample residual: {after['p50']:.3f}; "
        f"p95: {after['p95']:.3f} native units.",
        f"Check-record median: {check_stats['p50']:.3f}; p95: {check_stats['p95']:.3f}.",
        "",
        (
            "These are geometric proxy residuals, not verified marking accuracy. "
            "The fitted copies remain PROPOSED_UNREVIEWED and training-ineligible. "
            "Source files are unchanged. Original Z was retained and is not aligned"
            " to the raw survey."
        ),
        "",
        *[f"- {line}" for line in report["limitations"]],
        "",
        f"Preview: `{preview}`",
        "",
    ]
    (report_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "geopackage": str(gpkg),
                "median_residual": after["p50"],
                "p95_residual": after["p95"],
                "features": report["features_exported"],
            },
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[1])
