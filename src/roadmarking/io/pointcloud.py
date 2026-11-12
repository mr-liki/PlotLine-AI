"""Stage 1 — LAS/LAZ point-cloud input (laspy + lazrs backend)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from roadmarking.utils.logging import get_logger

log = get_logger(__name__)


class PointCloudReadError(IOError):
    """Raised when a LAS/LAZ file cannot be read."""


@dataclass
class PointCloudHeader:
    """Metadata needed to keep original coordinates and CRS intact."""

    path: Path
    point_count: int
    point_format: int
    version: str
    scales: np.ndarray
    offsets: np.ndarray
    mins: np.ndarray
    maxs: np.ndarray
    crs_wkt: str | None
    extra_dimensions: list[str] = field(default_factory=list)


@dataclass
class PointCloud:
    """Points in real-world coordinates (float64) plus raw intensity."""

    xyz: np.ndarray  # (N, 3) float64, scaled + offset applied
    intensity: np.ndarray  # (N,) uint16, as stored
    classification: np.ndarray | None
    header: PointCloudHeader

    def __len__(self) -> int:
        return int(self.xyz.shape[0])


def _header_from_laspy(path: Path, hdr: Any) -> PointCloudHeader:
    crs = None
    try:
        parsed = hdr.parse_crs()
        crs = parsed.to_wkt() if parsed is not None else None
    except Exception as exc:  # malformed or unsupported CRS VLRs must not block reading
        log.warning("Could not parse CRS from %s: %s", path.name, exc)
    return PointCloudHeader(
        path=path,
        point_count=int(hdr.point_count),
        point_format=int(hdr.point_format.id),
        version=str(hdr.version),
        scales=np.asarray(hdr.scales, dtype=np.float64),
        offsets=np.asarray(hdr.offsets, dtype=np.float64),
        mins=np.asarray(hdr.mins, dtype=np.float64),
        maxs=np.asarray(hdr.maxs, dtype=np.float64),
        crs_wkt=crs,
        extra_dimensions=list(hdr.point_format.extra_dimension_names),
    )


def read_header(path: str | Path) -> PointCloudHeader:
    """Read only the LAS/LAZ header (fast, no point data)."""
    import laspy

    path = Path(path)
    try:
        with laspy.open(path) as reader:
            return _header_from_laspy(path, reader.header)
    except Exception as exc:
        raise PointCloudReadError(f"Cannot read header of {path}: {exc}") from exc


@dataclass
class ChunkReadReport:
    """Outcome of a chunk-table read: which LAZ chunks decoded and which did not."""

    path: Path
    chunk_size: int
    n_chunks: int
    points_read: int = 0
    bad_chunks: list[tuple[int, str]] = field(default_factory=list)
    corrupt_chunks: list[tuple[int, int]] = field(default_factory=list)  # (chunk, first bad pt)
    dropped_points: int = 0

    @property
    def points_expected(self) -> int:
        return self.points_read + self.dropped_points


def laz_chunk_size(path: str | Path) -> int | None:
    """Return the LASzip fixed chunk size (points per chunk), or None if not LAZ/variable."""
    import struct

    import laspy

    with laspy.open(Path(path)) as reader:
        vlr = next((v for v in reader.header.vlrs if v.user_id == "laszip encoded"), None)
    if vlr is None:
        return None
    size = struct.unpack_from("<I", vlr.record_data, 12)[0]
    return None if size == 0xFFFFFFFF else int(size)  # 0xFFFFFFFF = variable-size chunks


GPS_WEEK_SECONDS = 604_800.0


def point_validity(
    pts: Any, mins: np.ndarray, maxs: np.ndarray, *, gps_week_time: bool, tolerance: float = 1e-3
) -> np.ndarray:
    """Boolean mask of physically plausible points (used to detect corrupt LAZ decodes).

    A point is valid if it lies inside the header bounding box, has
    ``1 <= return_number <= number_of_returns``, and (if present) a finite GPS time that is
    within one GPS week when the header declares GPS-week time.
    """
    x, y, z = np.asarray(pts.x), np.asarray(pts.y), np.asarray(pts.z)
    lo, hi = np.asarray(mins) - tolerance, np.asarray(maxs) + tolerance
    ok = (x >= lo[0]) & (x <= hi[0]) & (y >= lo[1]) & (y <= hi[1]) & (z >= lo[2]) & (z <= hi[2])
    rn, nr = np.asarray(pts.return_number), np.asarray(pts.number_of_returns)
    ok &= (rn >= 1) & (rn <= nr)
    names = set(pts.point_format.dimension_names)
    if "gps_time" in names:
        g = np.asarray(pts.gps_time)
        ok &= np.isfinite(g)
        if gps_week_time:
            ok &= (g >= 0) & (g < GPS_WEEK_SECONDS)
    return ok


def iter_laz_chunks(path: str | Path, report: ChunkReadReport | None = None) -> Iterator[Any]:
    """Yield validated points one LAZ chunk at a time, seeking via the chunk table.

    Seeking re-initialises the LAZ decoder at every chunk, so one corrupt chunk cannot
    spoil the following ones (front-to-back readers in laspy and PDAL fail or crash on
    the client files). Corrupt chunks do not always raise: they can decode to garbage.
    Every chunk is therefore checked with :func:`point_validity`; a chunk containing any
    invalid point is discarded entirely, because plausible-looking garbage can precede the
    first detectable one (recorded in ``report.corrupt_chunks`` / ``report.dropped_points``).
    Chunks that raise are recorded in ``report.bad_chunks``. The file is opened read-only.
    """
    import laspy

    path = Path(path)
    chunk = laz_chunk_size(path)
    with laspy.open(path, mode="r", laz_backend=laspy.LazBackend.Lazrs) as reader:
        hdr = reader.header
        total = int(hdr.point_count)
        gps_week = not bool(hdr.global_encoding.gps_time_type)
        if chunk is None:  # plain LAS or variable chunks: ordinary streaming
            chunk = 1_000_000
        n_chunks = -(-total // chunk) if total else 0
        if report is not None:
            report.chunk_size, report.n_chunks = chunk, n_chunks
        for ci in range(n_chunks):
            start = ci * chunk
            n = min(chunk, total - start)
            try:
                reader.seek(start)
                pts = reader.read_points(n)
                if len(pts) != n:
                    raise PointCloudReadError(f"short read: {len(pts)} of {n}")
            except Exception as exc:
                if report is not None:
                    report.bad_chunks.append((ci, f"{type(exc).__name__}: {exc}"))
                    report.dropped_points += n
                log.warning("%s: chunk %d unreadable: %s", path.name, ci, exc)
                continue
            ok = point_validity(pts, hdr.mins, hdr.maxs, gps_week_time=gps_week)
            if not ok.all():
                # Garbage can precede the first detectably invalid point, so the whole
                # chunk is discarded (conservative: no undetected garbage downstream).
                first_bad = int(np.argmin(ok))
                if report is not None:
                    report.corrupt_chunks.append((ci, first_bad))
                    report.dropped_points += n
                log.warning(
                    "%s: chunk %d corrupt (first invalid point %d); discarded",
                    path.name,
                    ci,
                    first_bad,
                )
                continue
            if report is not None:
                report.points_read += len(pts)
            yield pts


def read_pointcloud(path: str | Path) -> PointCloud:
    """Read a whole LAS/LAZ file into memory.

    Large files should later be streamed in chunks (``pointcloud.chunk_size``); chunked
    reading is part of Phase 2.
    """
    import laspy

    path = Path(path)
    try:
        las = laspy.read(path)
    except Exception as exc:
        raise PointCloudReadError(f"Cannot read {path}: {exc}") from exc
    xyz = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)])
    names = set(las.point_format.dimension_names)
    return PointCloud(
        xyz=xyz.astype(np.float64, copy=False),
        intensity=np.asarray(las.intensity),
        classification=np.asarray(las.classification) if "classification" in names else None,
        header=_header_from_laspy(path, las.header),
    )


@dataclass
class StrictReadReport:
    """Strict streaming validation; Phase 3 never invokes the recovery reader."""

    points_read: int = 0
    invalid_points: int = 0
    warnings: list[str] = field(default_factory=list)


def iter_pointcloud_chunks(path, chunk_size=250_000, report=None):
    """Yield (source record offset, PointCloud) or fail on any invalid record.

    Rules: finite coordinates within header bounds (one source quantization step),
    valid returns and GPS time, positive finite scales and ordered finite bounds.
    No universal absolute XY limit is imposed because native CRS is unconfirmed.
    Header extents are additionally budget-checked by the coarse-grid constructor.
    """
    import laspy

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    report = report if report is not None else StrictReadReport()
    path = Path(path)
    try:
        with laspy.open(path, mode="r", laz_backend=laspy.LazBackend.Lazrs) as reader:
            h = _header_from_laspy(path, reader.header)
            if (
                h.point_count <= 0
                or not np.isfinite(h.scales).all()
                or (h.scales <= 0).any()
                or not np.isfinite(h.offsets).all()
                or not np.isfinite(h.mins).all()
                or not np.isfinite(h.maxs).all()
                or (h.maxs < h.mins).any()
            ):
                raise PointCloudReadError("Invalid or empty source header")
            if h.crs_wkt is None:
                report.warnings.append("CRS_UNCONFIRMED; native coordinate units unverified")
            start = 0
            for pts in reader.chunk_iterator(chunk_size):
                valid = point_validity(
                    pts,
                    h.mins,
                    h.maxs,
                    gps_week_time=not bool(reader.header.global_encoding.gps_time_type),
                    tolerance=float(h.scales.max()),
                )
                xyz = np.column_stack((pts.x, pts.y, pts.z)).astype(np.float64)
                valid &= np.isfinite(xyz).all(axis=1)
                bad = int((~valid).sum())
                report.invalid_points += bad
                if bad:
                    raise PointCloudReadError(
                        f"{path.name}: {bad} invalid records at source offset {start}; no recovery"
                    )
                report.points_read += len(pts)
                yield start, PointCloud(
                    xyz, np.asarray(pts.intensity).copy(), np.asarray(pts.classification).copy(), h
                )
                start += len(pts)
            if start != h.point_count:
                raise PointCloudReadError(f"Short source: {start} != {h.point_count}")
    except PointCloudReadError:
        raise
    except Exception as exc:
        raise PointCloudReadError(f"Strict read failed for {path}: {exc}") from exc
