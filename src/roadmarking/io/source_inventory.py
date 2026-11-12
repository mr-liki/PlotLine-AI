"""Read-only source discovery and bounded LAS header/actual-record audits."""

import hashlib
import io
from collections import Counter
from pathlib import Path

import laspy
import numpy as np

from roadmarking.io.pointcloud import point_validity


def safe_header(path):
    """Reject malformed offsets before allowing a LAS reader to allocate VLRs."""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(227)
        if len(prefix) != 227 or prefix[:4] != b"LASF":
            raise ValueError("Missing LASF signature or truncated fixed header")
        offset = int.from_bytes(prefix[96:100], "little")
        if not 227 <= offset <= min(size, 4 * 1024**2):
            raise ValueError(f"Invalid or over-budget point-data offset: {offset}")
        payload = prefix + stream.read(offset - 227)
    if len(payload) != offset:
        raise ValueError("Truncated header/VLR region")
    header = laspy.LasHeader.read_from(io.BytesIO(payload), read_evlrs=False)
    if (
        not header.point_count
        or not np.isfinite([header.mins, header.maxs, header.scales, header.offsets]).all()
        or (header.maxs < header.mins).any()
        or (header.scales <= 0).any()
    ):
        raise ValueError("Invalid LAS bounds/count/scales")
    minimum = offset + (
        0 if header.are_points_compressed else header.point_count * header.point_format.size
    )
    if minimum > size or offset >= size:
        raise ValueError("Declared point records do not fit file")
    return header


def fingerprint(path, block_size=1024**2):
    """Partial fingerprint, explicitly not a whole-file cryptographic checksum."""
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        digest.update(stream.read(block_size))
        stream.seek(max(block_size, before.st_size - block_size))
        digest.update(stream.read(block_size))
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Source changed while fingerprinting")
    return {
        "method": "sha256(first_1MiB+last_1MiB); size+mtime; NOT_FULL_FILE_HASH",
        "sha256_partial": digest.hexdigest(),
        "size_bytes": before.st_size,
        "mtime_ns": before.st_mtime_ns,
    }


def discover(root):
    """Inventory recursively without following directory symlinks or writing sources."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            continue
        try:
            stat = path.stat()
            records.append(
                {
                    "path": str(path),
                    "relative": str(path.relative_to(root)),
                    "kind": "directory" if path.is_dir() else "file",
                    "extension": path.suffix.lower(),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        except OSError as exc:
            records.append({"path": str(path), "error": str(exc)})
    return records


def inspect_records(path, sample_points=5000):
    """Validate three actual point windows; never promote sampling to full integrity."""
    path = Path(path)
    result = {"path": str(path), "full_point_integrity_validated": False}
    try:
        header = safe_header(path)
        crs = header.parse_crs()
        result.update(
            {
                "fingerprint": fingerprint(path),
                "point_count": int(header.point_count),
                "version": str(header.version),
                "point_format": int(header.point_format.id),
                "scales": header.scales.tolist(),
                "offsets": header.offsets.tolist(),
                "mins": header.mins.tolist(),
                "maxs": header.maxs.tolist(),
                "crs_wkt": crs.to_wkt() if crs else None,
                "gps_week_time": not bool(header.global_encoding.gps_time_type),
                "dimensions": list(header.point_format.dimension_names),
                "vertical_reference": "SOURCE_WKT_ONLY_NOT_INDEPENDENTLY_CONFIRMED",
                "declared_bbox_density_points_per_native_square": float(
                    header.point_count / max(np.prod((header.maxs - header.mins)[:2]), 1e-12)
                ),
            }
        )
        n = min(sample_points, header.point_count)
        starts = sorted({0, max(0, header.point_count // 2 - n // 2), header.point_count - n})
        windows, classes, hist = [], Counter(), np.zeros(65536, np.uint64)
        with laspy.open(
            path, mode="r", read_evlrs=False, laz_backend=laspy.LazBackend.Lazrs
        ) as reader:
            for start in starts:
                try:
                    reader.seek(start)
                    points = reader.read_points(n)
                    valid = point_validity(
                        points, header.mins, header.maxs, gps_week_time=result["gps_week_time"]
                    )
                    if len(points) != n or not valid.all():
                        raise ValueError(
                            f"short/invalid sample: {len(points)}/{n}; "
                            f"invalid={int((~valid).sum())}"
                        )
                    hist += np.bincount(points.intensity, minlength=65536).astype(np.uint64)
                    values, counts = np.unique(
                        np.asarray(points.classification), return_counts=True
                    )
                    classes.update(dict(zip(map(int, values), map(int, counts), strict=True)))
                    windows.append({"start": int(start), "count": int(n), "status": "PASS"})
                except Exception as exc:
                    windows.append(
                        {
                            "start": int(start),
                            "count": int(n),
                            "status": "REJECTED",
                            "error": str(exc),
                        }
                    )
        occupied = np.flatnonzero(hist)
        result.update(
            {
                "status": (
                    "SAMPLE_QC_PASS_FULL_SCAN_PENDING"
                    if all(w["status"] == "PASS" for w in windows)
                    else "SAMPLE_QC_FAILED"
                ),
                "point_windows": windows,
                "sample_classifications": dict(classes),
                "sample_intensity_range": (
                    [int(occupied[0]), int(occupied[-1])] if occupied.size else None
                ),
                "sample_points_passed": int(hist.sum()),
            }
        )
    except Exception as exc:
        result.update(status="UNREADABLE_OR_INVALID_HEADER", error=f"{type(exc).__name__}: {exc}")
    return result
