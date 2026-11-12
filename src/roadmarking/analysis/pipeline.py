"""Phase 2 inspection pipeline: stats cache -> report JSON/Markdown (read-only on inputs)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from roadmarking.analysis.lidar_stats import collect_file_stats, intensity_summary
from roadmarking.io.pointcloud import PointCloudReadError, read_header
from roadmarking.utils.logging import get_logger

log = get_logger(__name__)


def json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o).__name__)


def cache_dir(cfg: Any) -> Path:
    d = cfg.path("outputs.reports") / "lidar_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def file_record(path: Path, grid_cell: float) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Header + streaming statistics of one file, and the 1 m grid arrays to cache."""
    rec: dict[str, Any] = {"file": path.name, "size_bytes": path.stat().st_size}
    try:
        hdr = read_header(path)
    except PointCloudReadError as exc:
        rec["error"] = str(exc)
        return rec, {}
    rec["header"] = {
        "version": hdr.version,
        "point_format": hdr.point_format,
        "point_count": hdr.point_count,
        "scales": hdr.scales.tolist(),
        "offsets": hdr.offsets.tolist(),
        "mins": hdr.mins.tolist(),
        "maxs": hdr.maxs.tolist(),
        "crs_wkt": hdr.crs_wkt,
        "extra_dimensions": hdr.extra_dimensions,
    }
    if hdr.point_count == 0:
        rec["error"] = "header point_count is 0 (header not finalised; chunk table missing)"
        return rec, {}
    t0 = time.perf_counter()
    st = collect_file_stats(path, grid_cell=grid_cell)
    cr = st.chunk_report
    rec.update(
        {
            "stream_seconds": round(time.perf_counter() - t0, 1),
            "streamed_point_count": st.point_count,
            "read_error": st.read_error,
            "laz_chunks": (
                None
                if cr is None
                else {
                    "chunk_size": cr.chunk_size,
                    "n_chunks": cr.n_chunks,
                    "unreadable_chunks": [list(b) for b in cr.bad_chunks],
                    "corrupt_chunks": [list(b) for b in cr.corrupt_chunks],
                    "dropped_points": cr.dropped_points,
                    "valid_points": cr.points_read,
                }
            ),
            "complete": st.read_error is None and st.point_count == hdr.point_count,
            "xyz_min": st.xyz_min.tolist(),
            "xyz_max": st.xyz_max.tolist(),
            "gps_time": [st.gps_time_min, st.gps_time_max],
            "class_counts": {int(c): int(n) for c, n in enumerate(st.class_counts) if n},
            "return_matrix": st.return_matrix.tolist(),
            "scan_angle_counts": {
                int(a) - 128: int(n) for a, n in enumerate(st.scan_angle_hist) if n
            },
            "user_data_counts": {int(c): int(n) for c, n in enumerate(st.user_data_counts) if n},
            "flag_counts": st.flag_counts,
            "point_source_counts": {str(k): v for k, v in sorted(st.point_source_counts.items())},
            "rgb": (
                None
                if st.rgb_min is None
                else {
                    "min": st.rgb_min.tolist(),
                    "max": st.rgb_max.tolist(),
                    "all_zero_points": st.rgb_all_zero,
                }
            ),
            "intensity": intensity_summary(st.intensity_hist),
        }
    )
    arrays: dict[str, np.ndarray] = {"intensity_hist": st.intensity_hist}
    if st.grid is not None and st.point_count:
        rec["grid"] = st.grid.spec.as_dict()
        arrays.update(
            count=st.grid.count,
            single_return=st.grid.single_return,
            intensity_sum=st.grid.intensity_sum,
            z_min=st.grid.z_min,
            z_max=st.grid.z_max,
        )
    return rec, arrays


def compute_stats_cache(
    cfg: Any, files: list[Path], grid_cell: float = 1.0, force: bool = False
) -> list[dict[str, Any]]:
    """Stream every file once and cache ``<stem>.json`` + ``<stem>_grid.npz``."""
    out = cache_dir(cfg)
    records = []
    for path in files:
        target = out / f"{path.stem}.json"
        if target.exists() and not force:
            records.append(json.loads(target.read_text(encoding="utf-8")))
            continue
        rec, arrays = file_record(path, grid_cell)
        if "grid" in rec:
            np.savez_compressed(out / f"{path.stem}_grid.npz", **arrays)
        target.write_text(json.dumps(rec, indent=2, default=json_default), encoding="utf-8")
        log.info(
            "%s: %s valid points (%s)",
            path.name,
            rec.get("streamed_point_count"),
            rec.get("error") or rec.get("read_error") or "complete",
        )
        records.append(rec)
    return records


def write_reports(cfg: Any, files: list[Path]) -> tuple[Path, Path]:
    """Build the report from the cache and write the JSON and Markdown data reports."""
    from roadmarking.analysis.report import build_report
    from roadmarking.analysis.report_md import render_markdown

    result = build_report(cfg, files)
    reports = cfg.path("outputs.reports")
    js = reports / "lidar_inspection_report.json"
    js.write_text(json.dumps(result, indent=2, default=json_default), encoding="utf-8")
    md = reports / "lidar_inspection_report.md"
    md.write_text(render_markdown(json.loads(js.read_text(encoding="utf-8"))), encoding="utf-8")
    return js, md
