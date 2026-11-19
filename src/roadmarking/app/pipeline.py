"""Plotline-AI end-to-end job: one LAS/LAZ file in, one client-layout GeoPackage out.

Stages (each reports measured progress and can be cancelled between safe units):

1. VALIDATING     header, coordinates, intensity, CRS, disk space, output location;
2. RASTERIZING    one streaming pass over the file: whole-file intensity histogram (production
                  calibration), corrupt-chunk accounting, and the points of every processing
                  block (~96 m cores with 6 m margins) written to the job's extract cache;
3. RECOGNIZING    per block: lowest-road and bridge-deck rasters + the registered model
                  (loaded and hash-verified once) -> accepted masks, clipped to observed support;
4. DIGITIZING     per block: locked line tracing (centerlines, dashes, junction joins) and the
                  locked closed-polygon extraction for other marks; features are kept in the
                  block core, lines cut at block borders are stitched back together;
5. RECOVERING_Z   per block: road-surface Z from the source points (plane fit, never invented);
6. EXPORTING      the client layers "Highway" (MultiLineString Z) and "Other Marks"
                  (MultiPolygon Z) + a run-metadata table, into a job-temporary GeoPackage;
7. VERIFYING      reopen, geometry/Z/CRS/bounds/ID/spatial-index checks, source unchanged;
8. delivery       atomic move to the user's destination (nothing else is written there).

The source LAS/LAZ is opened read-only and is hashed before and after the job.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import sqlite3
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

STAGES = (
    ("VALIDATING", "Validating LiDAR", 0.00, 0.03),
    ("RASTERIZING", "Extracting intensity", 0.03, 0.20),
    ("RECOGNIZING", "Running AI recognition", 0.20, 0.70),
    ("DIGITIZING", "Digitizing road markings", 0.70, 0.85),
    ("RECOVERING_Z", "Recovering 3D coordinates", 0.85, 0.92),
    ("EXPORTING", "Exporting GeoPackage", 0.92, 0.96),
    ("VERIFYING", "Validating output", 0.96, 1.00),
)
CORE_M, MARGIN_M, EXTRACT_MARGIN_M = 96.0, 6.0, 2.0
MIN_BLOCK_POINTS = 20000
MAX_DROPPED_FRACTION = 0.01  # more unreadable points than this: the file is not processed
CLIENT_LINE_LAYER, CLIENT_POLY_LAYER = "Highway", "Other Marks"
# Locked release (configs/app_lock.yaml); bump with every deliberate change after the lock.
APP_VERSION = "2.0"
LINE_FIELDS = [
    ("ID_SIG", "float64"),
    ("POSICAO", "object"),
    ("DESIGNACAO", "object"),
    ("LARGURA", "float64"),
    ("COMENTARIO", "object"),
    ("ID_HIST", "object"),
    ("COD_SUBLAN", "object"),
    ("SUBLANCO", "object"),
    ("NO", "object"),
    ("RAMO", "object"),
    ("TRAÇO", "object"),
    ("ESPAÇO", "object"),
    ("OBS", "object"),
    ("ID_GRUPO", "object"),
    ("COD_JAE", "object"),
    ("ELEVATION", "object"),
]
POLY_FIELDS = [
    ("ID_SIG", "int32"),
    ("POSICAO", "object"),
    ("COD_RST", "object"),
    ("ALGARISMO", "object"),
    ("LETRA", "object"),
    ("ALTURA", "float64"),
    ("AREA_PINTA", "float64"),
    ("ID_GRUPO", "object"),
    ("COMENTARIO", "object"),
    ("ID_HIST", "object"),
    ("ELEVATION", "float64"),
    ("OBS", "object"),
    ("COD_SUBLANCO", "object"),
    ("COD_JAE", "object"),
    ("LARGURA", "object"),
]


class JobError(RuntimeError):
    """A problem the user must see (bad input, unsupported setup, failed output check)."""


class Cancelled(RuntimeError):  # noqa: N818 - user-facing state name
    pass


@dataclass
class JobResult:
    output: Path
    seconds: float
    lines: int
    polygons: int
    crs: str
    warnings: list = field(default_factory=list)
    report: Path | None = None


def app_home() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "Plotline-AI"


class Progress:
    """Maps stage-local fractions to overall progress and checks for cancellation."""

    def __init__(self, callback=None, cancel: threading.Event | None = None):
        self.callback, self.cancel = callback, cancel or threading.Event()
        self.stage = STAGES[0]

    def enter(self, name, message=None):
        self.stage = next(s for s in STAGES if s[0] == name)
        self.update(0.0, message or self.stage[1])

    def update(self, fraction, message=None):
        if self.cancel.is_set():
            raise Cancelled("Processing was cancelled.")
        _, label, lo, hi = self.stage
        if self.callback:
            self.callback(
                self.stage[0], lo + (hi - lo) * float(np.clip(fraction, 0, 1)), message or label
            )


# ---------------------------------------------------------------- 1. validation
def resolve_crs(header_crs):
    """(crs for the output, human description, warning or None). Never guesses: EPSG:3763 is
    used only when the source horizontal CRS is the same PT-TM06 projection on GRS 1980."""
    if header_crs is None:
        return (
            None,
            "none in source (native coordinates kept)",
            (
                "The LAS/LAZ has no coordinate reference system. Native coordinates are kept "
                "and the "
                "GeoPackage has no CRS; assign the source CRS when loading it."
            ),
        )
    import pyproj

    horiz = header_crs.sub_crs_list[0] if header_crs.is_compound else header_crs
    vert = header_crs.sub_crs_list[1].name if header_crs.is_compound else "none declared"
    try:
        ref = pyproj.CRS.from_epsg(3763)
        same = (
            horiz.is_projected
            and horiz.coordinate_operation is not None
            and horiz.coordinate_operation.method_name == ref.coordinate_operation.method_name
            and all(
                abs(a.value - b.value) < 1e-9
                for a, b in zip(
                    horiz.coordinate_operation.params, ref.coordinate_operation.params, strict=True
                )
            )
            and horiz.ellipsoid.semi_major_metre == ref.ellipsoid.semi_major_metre
            and abs(horiz.ellipsoid.inverse_flattening - ref.ellipsoid.inverse_flattening) < 1e-9
        )
    except Exception:
        same = False
    if same:
        return "EPSG:3763", f"EPSG:3763 (PT-TM06/ETRS89; heights: {vert})", None
    return horiz.to_wkt(), f"{horiz.name} (from source; heights: {vert})", None


def validate_input(las_path: Path, out_path: Path, work: Path):
    import laspy

    if not las_path.is_file():
        raise JobError(f"Input file not found: {las_path}")
    if las_path.suffix.lower() not in (".las", ".laz"):
        raise JobError("Input must be a .las or .laz file.")
    if not os.access(las_path, os.R_OK):
        raise JobError("The input file cannot be read (permission denied).")
    if out_path.suffix.lower() != ".gpkg":
        raise JobError("The output file must have the .gpkg extension.")
    if out_path.resolve() == las_path.resolve():
        raise JobError("The output cannot overwrite the input file.")
    out_dir = out_path.parent
    if not out_dir.is_dir():
        raise JobError(f"Output folder does not exist: {out_dir}")
    probe = out_dir / f".plotline_write_test_{os.getpid()}"
    try:
        probe.write_bytes(b"x")
        probe.unlink()
    except OSError as e:
        raise JobError(f"The output folder is not writable: {out_dir} ({e})") from e
    try:
        with laspy.open(las_path) as f:
            h = f.header
            dims = set(h.point_format.dimension_names)
            crs = h.parse_crs()
            hdr = dict(
                points=int(h.point_count),
                version=str(h.version),
                mins=[float(v) for v in h.mins],
                maxs=[float(v) for v in h.maxs],
                scales=[float(v) for v in h.scales],
                offsets=[float(v) for v in h.offsets],
            )
    except Exception as e:
        raise JobError(f"The file cannot be opened as LAS/LAZ: {e}") from e
    if hdr["points"] <= 0:
        raise JobError("The LAS/LAZ file contains no points.")
    if "intensity" not in dims:
        raise JobError("The LAS/LAZ file has no intensity values; road markings cannot be read.")
    if not (np.all(np.isfinite(hdr["mins"] + hdr["maxs"])) and all(s > 0 for s in hdr["scales"])):
        raise JobError("The LAS/LAZ header has invalid bounds or scale factors.")
    if any(b <= a for a, b in zip(hdr["mins"][:2], hdr["maxs"][:2], strict=True)):
        raise JobError("The LAS/LAZ header bounds are empty.")
    if las_path.suffix.lower() == ".laz":
        try:
            import lazrs  # noqa: F401
        except ImportError as e:
            raise JobError("LAZ decompression (lazrs) is not available.") from e
    need = max(3 * las_path.stat().st_size, 4 * 2**30)
    for path in {work.anchor, out_dir.anchor}:
        free = shutil.disk_usage(path).free
        if free < need:
            raise JobError(
                f"Not enough free disk space on {path}: {free / 2**30:.1f} GB free, "
                f"{need / 2**30:.1f} GB needed for processing."
            )
    out_crs, crs_text, warning = resolve_crs(crs)
    return hdr, out_crs, crs_text, warning


# ---------------------------------------------------------------- 2. streaming extraction
def stream_blocks(las_paths, hdrs, cache: Path, progress: Progress):
    """One read-only pass per source file: that file's whole-file histogram (production
    calibration), corrupt-chunk accounting, and its points binned into ~96 m cells on disk.
    Blocks (cores with points + 6 m margins) are then assembled from the cells and saved in the
    extract-cache format of ``raster_tiles.extract_regions``, one per file, so the locked
    stages read them without touching the source again. Memory stays bounded by one file's
    chunk plus one block."""
    from roadmarking.digitization.raster_tiles import _extract_key, make_grid, sha256
    from roadmarking.io.pointcloud import ChunkReadReport, iter_laz_chunks

    mins = np.min([h["mins"] for h in hdrs], axis=0)
    maxs = np.max([h["maxs"] for h in hdrs], axis=0)
    x0 = np.floor(mins[0] / CORE_M) * CORE_M
    y0 = np.floor(mins[1] / CORE_M) * CORE_M
    nx = int(np.ceil((maxs[0] - x0) / CORE_M)) + 1
    cells = cache / "cells"
    cells.mkdir(exist_ok=True)
    total_points = sum(h["points"] for h in hdrs)
    counts, files, done = {}, [], 0
    for fi, las_path in enumerate(las_paths):
        hist = np.zeros(65536, np.uint64)
        report = ChunkReadReport(las_path, 0, 0)
        parts: dict[tuple, list] = {}
        for chunk in iter_laz_chunks(las_path, report):
            x, y, z = np.asarray(chunk.x), np.asarray(chunk.y), np.asarray(chunk.z)
            raw = np.asarray(chunk.intensity)
            ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            x, y, z, raw = x[ok], y[ok], z[ok], raw[ok]
            hist += np.bincount(raw, minlength=65536).astype(np.uint64)
            key = np.floor((y - y0) / CORE_M).astype(np.int64) * nx + np.floor(
                (x - x0) / CORE_M
            ).astype(np.int64)
            order = np.argsort(key, kind="stable")
            ks = key[order]
            starts = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
            for a, b in zip(starts, np.r_[starts[1:], len(ks)], strict=True):
                sel = order[a:b]
                k = (int(ks[a] % nx), int(ks[a] // nx))
                parts.setdefault(k, []).append((x[sel], y[sel], z[sel], raw[sel]))
            done += len(chunk)
            progress.update(
                0.8 * done / max(total_points, 1),
                f"Extracting intensity ({done:,} of {total_points:,} points)",
            )
        dropped = report.dropped_points / max(report.points_read + report.dropped_points, 1)
        if report.points_read == 0:
            raise JobError(f"No readable points were found in {las_path.name}.")
        if dropped > MAX_DROPPED_FRACTION:
            raise JobError(
                f"{dropped:.1%} of the points of {las_path.name} could not be decoded (corrupt "
                "LAZ chunks). The file is not processed because the result would silently miss "
                "road sections."
            )
        for k, v in parts.items():
            cols = [np.concatenate(c) for c in zip(*v, strict=True)]
            np.savez(cells / f"{fi}_{k[0]}_{k[1]}.npz", x=cols[0], y=cols[1], z=cols[2], i=cols[3])
            counts[k] = counts.get(k, 0) + len(cols[0])
        del parts
        files.append(
            dict(
                path=las_path,
                hist=hist,
                sha256=sha256(las_path),
                points_read=int(report.points_read),
                dropped_points=int(report.dropped_points),
                corrupt_chunks=np.asarray(report.corrupt_chunks, np.int64).reshape(-1, 2),
            )
        )
    cores = sorted(
        (k for k, n in counts.items() if n >= MIN_BLOCK_POINTS), key=lambda k: (k[1], k[0])
    )
    blocks = []
    for n, (i, j) in enumerate(cores):
        core = (x0 + i * CORE_M, y0 + j * CORE_M, x0 + (i + 1) * CORE_M, y0 + (j + 1) * CORE_M)
        bbox = (core[0] - MARGIN_M, core[1] - MARGIN_M, core[2] + MARGIN_M, core[3] + MARGIN_M)
        gb = make_grid(bbox).bounds
        m = EXTRACT_MARGIN_M
        for fi, f in enumerate(files):
            sel = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    cf = cells / f"{fi}_{i + di}_{j + dj}.npz"
                    if not cf.exists():
                        continue
                    with np.load(cf) as c:
                        px, py, pz, pi = c["x"], c["y"], c["z"], c["i"]
                    inb = (
                        (px >= gb[0] - m) & (px < gb[2] + m) & (py > gb[1] - m) & (py <= gb[3] + m)
                    )
                    if inb.any():
                        sel.append((px[inb], py[inb], pz[inb], pi[inb]))
            cols = (
                [np.concatenate(v) for v in zip(*sel, strict=True)]
                if sel
                else [np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, np.uint16)]
            )
            np.savez(
                cache / "extracts" / _extract_key(f["path"], gb, m),
                x=cols[0],
                y=cols[1],
                z=cols[2],
                intensity=cols[3].astype(np.uint16),
                hist=f["hist"],
                points_read=np.int64(f["points_read"]),
                dropped_points=np.int64(f["dropped_points"]),
                corrupt_chunks=f["corrupt_chunks"],
                sha256=np.str_(f["sha256"]),
                seconds=np.float64(0.0),
            )
        blocks.append(dict(id=n, core=core, bbox=bbox, points=int(counts[(i, j)])))
        progress.update(0.8 + 0.2 * (n + 1) / max(len(cores), 1), "Preparing AI tiles")
    shutil.rmtree(cells, ignore_errors=True)
    return blocks, files


# ---------------------------------------------------------------- 3-5. per block
def lift_feature(f, surface, spacing=1.0):
    """Road-surface Z every <= ``spacing`` m along the 2D line (same estimator as the
    validated pipeline); inferred joins take Z interpolated between supported vertices only."""
    from shapely.geometry import LineString, Point

    from roadmarking.digitization.qa import densify
    from roadmarking.digitization.surface import lift_line

    xy = densify(np.asarray(f["geometry_2d"].coords)[:, :2], spacing)
    z, _interp, recs = lift_line(surface, xy)
    if f.get("inferred_segments"):
        near = np.array(
            [min(sg.distance(Point(p)) for sg in f["inferred_segments"]) < 0.05 for p in xy]
        )
        fin = np.isfinite(z) & ~near
        if near.any() and fin.sum() >= 2:
            s = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
            first, last = np.flatnonzero(near)[[0, -1]]
            if fin[:first].any() and fin[last + 1 :].any():
                z[near] = np.interp(s[near], s[fin], z[fin])
    good = np.isfinite(z)
    f["z_support_frac"] = float(np.mean([r["ok"] for r in recs]))
    if not good.all() and f["z_support_frac"] >= 0.5 and good.sum() >= 2:
        # Sparse surface (far edge of the scan): keep the stretch between the first and last
        # vertices with a measured height; heights in between are interpolated along the line,
        # never extrapolated beyond measured ones.
        first, last = np.flatnonzero(good)[[0, -1]]
        xy, z, good = xy[first : last + 1], z[first : last + 1], good[first : last + 1]
        if (~good).any():
            s = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
            z[~good] = np.interp(s[~good], s[good], z[good])
            f["flags"] = sorted(set(f["flags"]) | {"z_interpolated_between_measured_points"})
        f["geometry_2d"] = LineString(xy)
        f["length_m"] = f["geometry_2d"].length
        good = np.isfinite(z)
    f["mean_z"] = float(np.nanmean(z)) if good.any() else np.nan
    f["geometry_3d"] = (
        LineString(np.column_stack([xy, z]))
        if good.all() and f["z_support_frac"] >= 0.5 and len(xy) >= 2
        else None
    )


def paint_raster(levels):
    """Per level: paint evidence used for line ends (bright observed return or AI mask), the
    same evidence the line tracing uses."""
    from roadmarking.digitization.fragments import bright_paint

    return [
        (bright_paint(lv["gray"], lv["observed"], delta=40) & lv["observed"])
        | lv["solid"]
        | lv["dash"]
        for lv in levels
    ]


def painted_at(line, dist, paint, grid, lateral=0.10, min_share=None):
    """True when ``paint`` has a pixel within ``lateral`` of the line axis at ``dist``. With
    ``min_share``: when at least that share of the pixels within ``lateral`` across and 0.1 m
    along is paint (isolated bright speckle is not)."""
    from roadmarking.digitization.raster_tiles import xy_to_pixel

    p = np.asarray(line.interpolate(dist).coords[0])[:2]
    q = np.asarray(line.interpolate(min(dist + 0.1, line.length)).coords[0])[:2]
    if np.linalg.norm(q - p) < 1e-6:
        q = p + (p - np.asarray(line.interpolate(max(dist - 0.1, 0)).coords[0])[:2])
    t = (q - p) / max(np.linalg.norm(q - p), 1e-9)
    n = np.array([-t[1], t[0]])
    pts = p + np.linspace(-lateral, lateral, 9)[:, None] * n
    if min_share is not None:
        along = np.linspace(-0.1, 0.1, 5)
        pts = (pts[None, :, :] + along[:, None, None] * t).reshape(-1, 2)
    r, c = xy_to_pixel(grid, pts[:, 0], pts[:, 1])
    r, c = np.rint(r).astype(int), np.rint(c).astype(int)
    ok = (r >= 0) & (c >= 0) & (r < paint.shape[0]) & (c < paint.shape[1])
    if min_share is not None:
        return bool(ok.any() and paint[r[ok], c[ok]].mean() >= min_share)
    return bool(paint[r[ok], c[ok]].any())


def polygon_paint_share(poly, paint, observed, grid, inset=0.05):
    """Share of the observed cells inside ``poly`` (inset by ``inset`` m) within 0.15 m of
    bright paint (the bright-paint test marks the edges and texture of a solid mark, not
    every pixel of it; speckle on verges leaves most of a blob far from any bright return)."""
    from rasterio.features import rasterize

    g = poly.buffer(-inset)
    if g.is_empty:
        g = poly
    from scipy.ndimage import binary_dilation

    inside = rasterize([g], out_shape=paint.shape, transform=grid.affine).astype(bool)
    seen = inside & observed
    if seen.sum() < 10:
        return None
    r, c = np.nonzero(inside)
    r0, r1, c0, c1 = max(r.min() - 4, 0), r.max() + 5, max(c.min() - 4, 0), c.max() + 5
    near = binary_dilation(paint[r0:r1, c0:c1], iterations=3)  # within 0.15 m of bright paint
    return float(near[seen[r0:r1, c0:c1]].mean())


def paint_support(line, paint, observed, grid, spacing=0.5):
    """Share of the line's observed stations (every ``spacing`` m) with paint on the axis.
    Hidden stations (vehicles, no returns) are not counted. None when nothing is observed."""
    hits = seen = 0
    for d in np.arange(spacing / 2, max(line.length, spacing / 2 + 1e-6), spacing):
        if not painted_at(line, d, observed, grid, lateral=0.0):
            continue
        seen += 1
        hits += painted_at(line, d, paint, grid)
    return hits / seen if seen else None


def hatch_zones(mark_geoms, link_m=1.5, min_marks=3, inset_m=0.3):
    """Areas covered by groups of >= ``min_marks`` painted marks lying close together (hatch
    bars, chevrons): the convex hull of each group, shrunk by ``inset_m`` so that lines running
    along the zone's edge are outside."""
    import shapely

    if len(mark_geoms) < min_marks:
        return []
    merged = shapely.union_all([g.buffer(link_m / 2) for g in mark_geoms])
    zones = []
    for blob in getattr(merged, "geoms", [merged]):
        members = [g for g in mark_geoms if blob.intersects(g)]
        if len(members) >= min_marks:
            z = shapely.union_all(members).convex_hull.buffer(-inset_m)
            if not z.is_empty:
                zones.append(z)
    return zones


def runs_through_marks(line, mark_geoms, max_len=6.0, zones=None, in_mark=0.4):
    """A short line lying inside a hatch zone (between or across its bars), or largely inside
    one mark (an arrow next to a line), is traced from the marks' own paint, not a lane line.
    Lines along a zone's edge are kept."""
    if line.length > max_len or not mark_geoms:
        return False
    zones = hatch_zones(mark_geoms) if zones is None else zones
    inside = sum(line.intersection(z).length for z in zones)
    if inside >= 0.5 * line.length:
        return True
    return any(line.intersection(g).length >= in_mark * line.length for g in mark_geoms)


def trim_ends_in_marks(line, mark_geoms, near_m=0.15, out_m=0.25, step=0.1, max_trim=8.0):
    """A line end running into an area mark (traced into an arrow next to it) is cut back to
    where the line leaves the mark. Returns the line, or None when nothing is left."""
    import shapely
    from shapely.ops import substring

    marks = [g for g in mark_geoms if g.distance(line) <= near_m]
    if not marks:
        return line
    inside = shapely.union_all([g.buffer(near_m) for g in marks])
    outside = shapely.union_all([g.buffer(out_m) for g in marks])
    a, b = 0.0, line.length
    if inside.contains(shapely.Point(line.coords[0])):
        while a < min(max_trim, b) and outside.contains(line.interpolate(a)):
            a += step
    if inside.contains(shapely.Point(line.coords[-1])):
        while b > max(line.length - max_trim, a) and outside.contains(line.interpolate(b)):
            b -= step
    if b - a < 0.3:
        return None
    return line if (a == 0 and b == line.length) else substring(line, a, b)


def clip_at_lines(poly, lines, half_width=0.1, min_line_m=10.0, max_cut=0.35):
    """A mark touching a lane line (an arrow next to the edge line) is fitted over part of the
    line's paint. Cut it back at every long line crossing it and keep the part holding the
    mark, when what is cut away is small. Returns the polygon (unchanged if nothing to cut)."""
    import shapely

    for ln in lines:
        if ln.length < min_line_m or not ln.intersects(poly):
            continue
        ends = [shapely.Point(ln.coords[0]), shapely.Point(ln.coords[-1])]
        if any(poly.buffer(0.5).contains(e) for e in ends):
            continue  # only a line passing right through the mark, not one ending in it
        rest = poly.difference(ln.buffer(half_width, cap_style="flat"))
        parts = [g for g in getattr(rest, "geoms", [rest]) if g.geom_type == "Polygon"]
        if not parts:
            continue
        keep = max(parts, key=lambda g: g.area)
        if keep.area >= (1 - max_cut) * poly.area:
            poly = keep
    return poly


def stripe_support(
    line,
    gray,
    observed,
    grid,
    spacing=0.5,
    delta=20.0,
    near=0.08,
    side=(0.15, 0.6),
    typical_side=False,
):
    """Share of the line's observed stations where the axis is a bright *stripe*: brighter by
    ``delta`` grey levels than the darkest road on each side (``side`` m from the axis). The
    edge of a brighter area (another scan pass, a shadow) is bright on one side only and is not
    paint. ``typical_side``: compare with the median of each side instead of its darkest pixel,
    so a single dark pixel (a gap, the foot of a post) does not make a wide bright band (a
    barrier base, a light verge) look like a stripe. None when nothing is observed."""
    from roadmarking.digitization.raster_tiles import xy_to_pixel

    def sample(pts):
        r, c = xy_to_pixel(grid, pts[:, 0], pts[:, 1])
        r, c = np.rint(r).astype(int), np.rint(c).astype(int)
        ok = (r >= 0) & (c >= 0) & (r < gray.shape[0]) & (c < gray.shape[1])
        ok[ok] &= observed[r[ok], c[ok]]
        v = np.full(len(pts), np.nan)
        v[ok] = gray[r[ok], c[ok]]
        return v

    side_level = np.nanmedian if typical_side else np.nanmin
    hits = seen = 0
    for d in np.arange(spacing / 2, max(line.length, spacing / 2 + 1e-6), spacing):
        p = np.asarray(line.interpolate(d).coords[0])[:2]
        q = np.asarray(line.interpolate(min(d + 0.1, line.length)).coords[0])[:2]
        if np.linalg.norm(q - p) < 1e-6:
            q = p + (p - np.asarray(line.interpolate(max(d - 0.1, 0)).coords[0])[:2])
        t = (q - p) / max(np.linalg.norm(q - p), 1e-9)
        n = np.array([-t[1], t[0]])
        axis = sample(p + np.linspace(-near, near, 5)[:, None] * n)
        if np.all(np.isnan(axis)):
            continue
        offs = np.arange(side[0], side[1] + 1e-9, 0.05)
        left = sample(p + offs[:, None] * n)
        right = sample(p - offs[:, None] * n)
        if np.all(np.isnan(left)) or np.all(np.isnan(right)):
            continue
        seen += 1
        a = np.nanmax(axis)
        hits += a - side_level(left) >= delta and a - side_level(right) >= delta
    return hits / seen if seen else None


def trim_to_paint(
    line, paint, grid, other_lines=(), step=0.05, lateral=0.10, max_trim=6.0, min_share=0.2
):
    """Shorten ``line`` at both ends to the last station with paint within ``lateral`` of
    the axis. An end lying on another line (a junction join) is kept. Returns the trimmed line
    or None when no paint is found at all."""
    from shapely.geometry import Point
    from shapely.ops import substring

    def painted(dist):
        return painted_at(line, dist, paint, grid, lateral, min_share)

    def on_other(dist):
        pt = Point(line.interpolate(dist).coords[0])
        return any(o.distance(pt) < 0.02 for o in other_lines)

    length = line.length
    a, b = 0.0, length
    if not on_other(0.0):
        while a < min(max_trim, length) and not painted(a):
            a += step
    if not on_other(length):
        while b > max(length - max_trim, a) and not painted(b):
            b -= step
    if b - a < 0.3:
        return None
    if a <= 0 and b >= length:
        return line
    return substring(line, a, b)


def drop_line_paint_polygons(polys, lines, max_angle_deg=15.0, min_cover=0.5, min_ratio=3.0):
    """Remove 'other mark' polygons that are only the merged paint of lines converging at a
    junction: the polygon runs parallel to traced lines that are much longer than it and that
    cover most of its length. Hatch bars cross lines at an angle and are kept."""
    import shapely

    from roadmarking.digitization.polygon_fit import principal_frame

    if not polys or not lines:
        return polys, 0
    tree = shapely.STRtree(lines)
    kept, dropped = [], 0
    for r in polys:
        g = r["geometry"]
        _, axis, length, _ = principal_frame(g)
        near = [lines[i] for i in tree.query(g, predicate="intersects")]
        parallel = []
        for ln in near:
            if ln.length < max(min_ratio * length, 10.0):
                continue
            d = ln.project(g.centroid)
            a = np.asarray(ln.interpolate(max(d - 1, 0)).coords[0])[:2]
            b = np.asarray(ln.interpolate(min(d + 1, ln.length)).coords[0])[:2]
            u = (b - a) / max(np.linalg.norm(b - a), 1e-9)
            if abs(float(u @ axis)) >= np.cos(np.radians(max_angle_deg)):
                parallel.append(ln)
        if parallel:
            covered = shapely.union_all(parallel).intersection(g).length
            if covered >= min_cover * length:
                dropped += 1
                continue
        kept.append(r)
    return kept, dropped


def is_line_paint_polygon(poly, max_width=0.25, min_length=2.0):
    """A long strip no wider than a lane line is line paint (e.g. the untraced border of a hatch
    zone), not an area mark: 3 of the client's 3245 area marks have this shape."""
    from roadmarking.digitization.polygon_fit import principal_frame

    parts = getattr(poly, "geoms", [poly])
    big = max(parts, key=lambda g: g.area)
    _, _, length, width = principal_frame(big)
    return width <= max_width and length >= min_length


def off_lane_polygon(poly, lines, tree=None, reach=8.0, min_area=0.2, verge_m=0.0, along_deg=10.0):
    """Junk 'mark' outside the lanes: no traced line on one side within ``reach`` m and either
    tiny (< ``min_area`` m2) or a strip running along the road (within ``along_deg``; verge
    speckle beyond the edge line). On Track01_4 no client mark matched this: one-sided client
    marks (hatch bars, chevrons) are >= 0.22 m2 and turned >= 17 deg to the road."""
    import shapely
    from shapely.geometry import LineString

    from roadmarking.digitization.polygon_fit import principal_frame

    if not lines:
        return False
    tree = shapely.STRtree(lines) if tree is None else tree
    big = max(getattr(poly, "geoms", [poly]), key=lambda g: g.area)
    c = big.centroid
    ln = lines[tree.nearest(c)]
    d = ln.project(c)
    a = np.asarray(ln.interpolate(max(d - 1, 0)).coords[0])[:2]
    b = np.asarray(ln.interpolate(min(d + 1, ln.length)).coords[0])[:2]
    t = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    n = np.array([-t[1], t[0]])
    p = np.asarray(c.coords[0])[:2]
    sides = 0
    for sgn in (1, -1):
        ray = LineString([p, p + sgn * reach * n])
        sides += any(lines[k].intersects(ray) for k in tree.query(ray))
    if sides == 2:
        return False
    if big.area < min_area:
        return True
    _, axis, _, _ = principal_frame(big)
    along = np.degrees(np.arccos(min(1.0, abs(float(axis @ t))))) <= along_deg
    return along and ln.distance(c) >= verge_m


def paint_not_digitized(
    level, paint, grid, lines, polys, core, min_area=1.0, near_m=4.0, min_solid=0.75
):
    """Solid paint patches (>= ``min_area`` m2) in the lanes that no line and no polygon covers:
    a mark the extraction or a filter missed. [(x, y, area)] inside ``core``."""
    from rasterio.features import rasterize, shapes
    from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening
    from shapely.geometry import box, shape

    geoms = [f["geometry_2d"] for f in lines if f.get("layer", 0) == 0]
    if not geoms:
        return []
    lanes = rasterize(
        [g.buffer(near_m) for g in geoms], out_shape=paint.shape, transform=grid.affine
    ).astype(bool)
    covered = rasterize(
        [g.buffer(0.25) for g in geoms] + [r["geometry"].buffer(0.15) for r in polys],
        out_shape=paint.shape,
        transform=grid.affine,
    ).astype(bool)
    m = binary_fill_holes(binary_closing(paint & lanes, iterations=2))
    m = binary_opening(m & ~covered, iterations=2) & level["observed"]
    raw = paint & level["observed"]
    out, cb = [], box(*core)
    for g, _ in shapes(m.astype(np.uint8), mask=m, transform=grid.affine):
        poly = shape(g)
        c = poly.centroid
        if poly.area < min_area or not cb.contains(c):
            continue
        if poly.area / max(poly.convex_hull.area, 1e-9) < 0.3:
            continue  # a speckle net, not a patch
        inside = rasterize([poly], out_shape=paint.shape, transform=grid.affine).astype(bool)
        seen = inside & level["observed"]
        if seen.sum() < 20 or raw[seen].mean() < min_solid:
            continue  # gravel / shoulder speckle: paint pixels scattered, not solid
        out.append([round(c.x, 2), round(c.y, 2), round(poly.area, 2)])
    return out


def touches_line_paint(poly, lines, max_width=0.3, touch_m=0.12, max_angle_deg=10.0):
    """A thin strip parallel to a traced line and touching its paint (within ``touch_m``) is
    a piece of that line's paint (e.g. where two side-by-side stripes end), not an area mark.
    None of the client's 3248 marks is like this: they always keep clear of the lines."""
    from roadmarking.digitization.polygon_fit import principal_frame

    big = max(getattr(poly, "geoms", [poly]), key=lambda g: g.area)
    _, axis, _, width = principal_frame(big)
    if width > max_width:
        return False
    for ln in lines:
        if ln.length < 2.0 or ln.distance(big) > touch_m:
            continue
        d = ln.project(big.centroid)
        a = np.asarray(ln.interpolate(max(d - 1, 0)).coords[0])[:2]
        b = np.asarray(ln.interpolate(min(d + 1, ln.length)).coords[0])[:2]
        t = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        if np.degrees(np.arccos(min(1.0, abs(float(axis @ t))))) <= max_angle_deg:
            return True
    return False


def mark_paint_raster(level, gap_px=4):
    """Paint of area marks (same test as the mark extraction), with unscanned gaps of up to
    ~2 x ``gap_px`` pixels between paint filled in: a narrow unscanned strip across a mark (scan
    pattern) must not split it or cut off a part. Scanned bare road is never filled."""
    from scipy.ndimage import binary_closing

    from roadmarking.digitization.fragments import bright_paint

    paint = bright_paint(level["gray"], level["observed"], delta=25, window=81) & level["observed"]
    bridged = binary_closing(paint, iterations=gap_px)
    return paint | (bridged & ~level["observed"])


def simplify_3d(coords, keep, tol_xy=0.01, tol_z=0.01):
    """Douglas-Peucker on a 3D polyline: fewest vertices keeping every removed vertex within
    ``tol_xy`` sideways and ``tol_z`` in height of the simplified line. Indices in ``keep``
    (ends, shared junction vertices) are never removed."""
    c = np.asarray(coords, float)
    if len(c) <= 2:
        return c
    keep = sorted({0, len(c) - 1} | {int(k) for k in keep})

    def dp(i, j, out):
        if j <= i + 1:
            return
        a, b = c[i], c[j]
        seg = b[:2] - a[:2]
        seg_len = np.linalg.norm(seg)
        pts = c[i + 1 : j]
        if seg_len < 1e-9:
            t = np.zeros(len(pts))
            lat = np.linalg.norm(pts[:, :2] - a[:2], axis=1)
        else:
            u = seg / seg_len
            t = np.clip((pts[:, :2] - a[:2]) @ u / seg_len, 0, 1)
            lat = np.abs((pts[:, :2] - a[:2]) @ np.array([-u[1], u[0]]))
        dz = np.abs(pts[:, 2] - (a[2] + t * (b[2] - a[2])))
        score = np.maximum(lat / tol_xy, dz / tol_z)
        k = int(np.argmax(score))
        if score[k] > 1.0:
            m = i + 1 + k
            dp(i, m, out)
            out.append(m)
            dp(m, j, out)

    out = []
    for i, j in itertools.pairwise(keep):
        out.append(i)
        dp(i, j, out)
    out.append(keep[-1])
    return c[sorted(set(out))]


COVER_CELL_M = 0.25
_KEY = 2**31


def cell_keys(xs, ys):
    """Integer keys of the global 0.25 m cells containing (xs, ys)."""
    ix = np.floor(np.asarray(xs) / COVER_CELL_M).astype(np.int64)
    iy = np.floor(np.asarray(ys) / COVER_CELL_M).astype(np.int64)
    return (ix + _KEY) * (2 * _KEY) + (iy + _KEY)


def coverage_keys(grid, observed, paint, core):
    """Sorted unique keys of the 0.25 m cells inside ``core`` with road-surface returns, and of
    those with paint (vehicles and structures are not part of the road surface)."""
    from roadmarking.digitization.raster_tiles import pixel_to_xy

    out = []
    for mask in (observed, paint & observed):
        r, c = np.nonzero(mask)
        x, y = pixel_to_xy(grid, r, c)
        inside = (x >= core[0]) & (x < core[2]) & (y >= core[1]) & (y < core[3])
        out.append(np.unique(cell_keys(x[inside], y[inside])))
    return out


# A line is exported only if visible paint (bright returns) lies on its axis at this share
# of its observed stations (measured against the client's lines, see the report).
MIN_PAINT_SUPPORT = 0.3
# ... and the paint must be a stripe (darker road on both sides) at this share of stations.
MIN_STRIPE_SUPPORT = 0.4
# ... and brighter than the *typical* road on both sides at this share (a line along a barrier
# base or a light verge band is brighter than its darkest neighbour pixel, not than the road).
MIN_TYPICAL_STRIPE_SUPPORT = 0.4
# An area mark is exported only if this share of its observed cells is visible paint.
MIN_POLYGON_PAINT = 0.6
# Self-check: a mark's outline should overlap its own paint at least this much (IoU).
MIN_MARK_IOU = 0.75


def process_block(block, las_paths, cache, bundle, classifier, library, progress, frac):
    from shapely.geometry import Point, Polygon, box

    from roadmarking.digitization.other_marks import extract_other
    from roadmarking.digitization.raster_tiles import make_grid
    from roadmarking.digitization.stages import build_levels, level_pieces, trace_features

    grid = make_grid(block["bbox"])
    levels, diag, _ = build_levels([str(p) for p in las_paths], grid, cache, bundle=bundle)
    progress.update(frac[0], f"Running AI recognition (block {block['id'] + 1})")
    seeds = level_pieces(levels, grid)
    lines, _ = trace_features(seeds, levels, grid)
    lines = [
        f
        for f in lines
        if not any("off-road candidate" in x for x in f["flags"]) or f["length_m"] >= 3
    ]
    polys = []
    for k, lv in enumerate(levels):
        level_lines = [f["geometry_2d"] for f in lines if f.get("layer", 0) == k]
        for row in extract_other(
            lv["gray"], lv["observed"], grid, level_lines, lv["surface"], classifier, library
        ):
            row["layer"] = k
            polys.append(row)
    # audit of every mark dropped by a filter below (job log: polygons_dropped)
    dropped_log = []

    def drop(rows, keep, reason):
        kept_ids = {id(r) for r in keep}
        for r in rows:
            if id(r) not in kept_ids:
                c = r["geometry"].centroid
                dropped_log.append(
                    [round(c.x, 2), round(c.y, 2), round(r["geometry"].area, 2), reason]
                )
        return keep

    extracted_count = len(polys)
    before = polys
    polys, junction_polys = drop_line_paint_polygons(polys, [f["geometry_2d"] for f in lines])
    polys = drop(before, polys, "junction_line_paint")
    line_like = sum(is_line_paint_polygon(r["geometry"]) for r in polys)
    polys = drop(
        polys, [r for r in polys if not is_line_paint_polygon(r["geometry"])], "line_paint_strip"
    )
    from roadmarking.digitization.fragments import bright_paint

    bright = [bright_paint(lv["gray"], lv["observed"], delta=40) & lv["observed"] for lv in levels]
    from roadmarking.app.refine import retemplate

    early_mark_paint = [mark_paint_raster(lv) for lv in levels]
    # Area marks must be solid visible paint: speckle on verges and shoulders is not a mark.
    poly_log, kept_p = [], []
    for r in polys:
        dens = polygon_paint_share(
            r["geometry"], bright[r.get("layer", 0)], levels[r.get("layer", 0)]["observed"], grid
        )
        c = r["geometry"].centroid
        poly_log.append([round(c.x, 2), round(c.y, 2), round(r["geometry"].area, 2), dens])
        # a mark that matches a client shape (template IoU >= 0.75) is real paint even when
        # faint (worn or less reflective): speckle does not take an arrow's shape
        if dens is None or dens >= MIN_POLYGON_PAINT or r.get("shape_method") == "client_shape":
            kept_p.append(r)
            continue
        k_ = r.get("layer", 0)
        if r["geometry"].area < 1.0 or r.get("shape_method") == "outline_mark":
            continue
        # faint paint can leave gaps (holes) in the extracted outline: match its outer outline
        got = retemplate(
            Polygon(r["geometry"].exterior),
            early_mark_paint[k_],
            grid,
            [f["geometry_2d"] for f in lines],
            library,
        )
        if got is not None:
            tpl, code, _t_iou = got
            from roadmarking.digitization.other_marks import lift_polygon

            flags = [x for x in (r.get("review_flags") or "").split(";") if x]
            r.update(
                geometry=tpl,
                COD_RST=code,
                COD_JAE=classifier.jae.get(code),
                shape_method="client_shape",
                review_flags=";".join([*flags, "faint_paint_client_shape"]),
            )
            r["geometry_3d"], _ = lift_polygon(tpl, levels[k_]["surface"])
            kept_p.append(r)
    speckle_polys = len(polys) - len(kept_p)
    polys = drop(polys, kept_p, "speckle_low_paint_share")
    if polys:  # lines inside or along a fitted mark are not lane lines (validated rule)
        import shapely

        marks = shapely.union_all([r["geometry"].buffer(0.25) for r in polys])
        lines = [
            f for f in lines if f["geometry_2d"].intersection(marks).length < 0.7 * f["length_m"]
        ]
    # Short lines traced across hatch bars / chevrons are the marks' paint, not lane lines.
    mark_geoms = [r["geometry"] for r in polys]
    zones = hatch_zones(mark_geoms)
    inside = [runs_through_marks(f["geometry_2d"], mark_geoms, zones=zones) for f in lines]
    through_marks = sum(inside)
    lines = [f for f, bad in zip(lines, inside, strict=True) if not bad]
    trimmed_ends = []
    for f in lines:  # no line end runs into a mark next to it
        t = trim_ends_in_marks(f["geometry_2d"], mark_geoms)
        if t is not None:
            trimmed_ends.append(
                f if t is f["geometry_2d"] else dict(f, geometry_2d=t, length_m=t.length)
            )
    lines = trimmed_ends
    # A dash merged with the paint of the continuous line beside it is fitted as a polygon by
    # the mark extraction: recover it as a dash (it passes the same paint tests below).
    from roadmarking.app.recover import _feature, dash_beside_line

    base = [f["geometry_2d"] for f in lines]
    kept_p, recovered_dashes = [], 0
    for r in polys:
        got = dash_beside_line(r["geometry"], base)
        if got is None:
            kept_p.append(r)
            continue
        lines.append(
            _feature(got[0], 1, got[1], ["dash_beside_line (from paint)"], r.get("layer", 0))
        )
        recovered_dashes += 1
    polys = drop(polys, kept_p, "converted_to_dash")
    # Visible paint: bright LiDAR returns above the local road (the AI mask can spread past
    # the paint). Line ends stop at the last visibly painted station, and a line with almost
    # no visible paint under it is not exported.
    paint = paint_raster(levels)
    seen_keys, paint_keys = coverage_keys(grid, levels[0]["observed"], bright[0], block["core"])
    trimmed, no_paint, support_log = [], 0, []
    for f in lines:
        k = f.get("layer", 0)
        sup = paint_support(f["geometry_2d"], bright[k], levels[k]["observed"], grid)
        stripe = stripe_support(f["geometry_2d"], levels[k]["gray"], levels[k]["observed"], grid)
        typical = stripe_support(
            f["geometry_2d"], levels[k]["gray"], levels[k]["observed"], grid, typical_side=True
        )
        mid = f["geometry_2d"].interpolate(0.5, normalized=True)
        support_log.append(
            [round(mid.x, 2), round(mid.y, 2), round(f["length_m"], 2), sup, stripe, typical]
        )
        if (
            (sup is not None and sup < MIN_PAINT_SUPPORT)
            or (stripe is not None and stripe < MIN_STRIPE_SUPPORT)
            or (typical is not None and typical < MIN_TYPICAL_STRIPE_SUPPORT)
        ):
            no_paint += 1
            continue
        others = [g["geometry_2d"] for g in lines if g is not f]
        # Trim to visible paint; a line whose paint is only in the AI mask (worn) keeps the
        # AI evidence for its ends.
        ends_paint = bright[k] if (sup or 0) >= 0.5 else paint[k]
        t = trim_to_paint(f["geometry_2d"], ends_paint, grid, others)
        if t is not None:
            trimmed.append(dict(f, geometry_2d=t, length_m=t.length))
    lines = trimmed
    # Every line on the exact centre of its paint stripe (curves, wide and faint stripes);
    # ends meeting another line stay attached.
    import shapely

    # a line crossing the dark gap between two side-by-side stripes (joined although the
    # stripes do not merge) is cut there; a leftover piece repeating another line goes
    from shapely.ops import substring

    from roadmarking.app.centring import gap_crossings, off_centre_stretches, recentre

    split, cut_count = [], 0
    for f in lines:
        k = f.get("layer", 0)
        g = f["geometry_2d"]
        gaps = gap_crossings(g, levels[k]["gray"], levels[k]["observed"], grid)
        if not gaps:
            split.append(f)
            continue
        cut_count += len(gaps)
        a = 0.0
        for s0, s1 in [*gaps, (g.length, g.length)]:
            if s0 - a >= 0.5:
                piece = substring(g, a, s0)
                split.append(
                    dict(
                        f,
                        geometry_2d=piece,
                        length_m=piece.length,
                        flags=sorted(set(f["flags"]) | {"cut_where_crossing_between_stripes"}),
                    )
                )
            a = s1
    deduped = []
    for f in split:
        g = f["geometry_2d"]
        others = [
            h["geometry_2d"] for h in split if h is not f and h["geometry_2d"].length > g.length
        ]
        if (
            others
            and g.intersection(shapely.union_all([o.buffer(0.12) for o in others])).length
            >= 0.8 * g.length
        ):
            continue  # the same paint as a longer line
        deduped.append(f)
    lines = deduped
    geoms = [f["geometry_2d"] for f in lines]
    gtree = shapely.STRtree(geoms) if geoms else None
    centred, shifts, centre_log = [], [], []
    for i, f in enumerate(lines):
        g = f["geometry_2d"]
        ends = []
        for pt in (shapely.Point(g.coords[0]), shapely.Point(g.coords[-1])):
            near = gtree.query(pt.buffer(0.05)) if gtree is not None else []
            ends.append(any(j != i and geoms[j].distance(pt) < 0.05 for j in near))
        k = f.get("layer", 0)
        ng, shift, share = recentre(g, levels[k]["gray"], levels[k]["observed"], grid, tuple(ends))
        shifts.append(shift)
        mid = g.interpolate(0.5, normalized=True)
        centre_log.append([round(mid.x, 2), round(mid.y, 2), round(g.length, 2), share])
        centred.append(f if ng is g else dict(f, geometry_2d=ng, length_m=ng.length))
    lines = centred
    # centring self-check: every final line re-measured on the paint
    off_centre = []
    for f in lines:
        k = f.get("layer", 0)
        for x_, y_, len_, o_ in off_centre_stretches(
            f["geometry_2d"], levels[k]["gray"], levels[k]["observed"], grid
        ):
            if (
                block["core"][0] <= x_ < block["core"][2]
                and block["core"][1] <= y_ < block["core"][3]
            ):
                off_centre.append([x_, y_, len_, round(o_, 3)])
    core = box(*block["core"])
    kept_lines = []
    for f in lines:
        clipped = f["geometry_2d"].intersection(core)
        for part in getattr(clipped, "geoms", [clipped]):
            if part.geom_type == "LineString" and part.length >= 0.3:
                kept_lines.append(
                    dict(f, geometry_2d=part, length_m=part.length, block=block["id"])
                )
    for f in kept_lines:
        lift_feature(f, levels[f.get("layer", 0)]["surface"])
    # Marks lie in the lanes: junk outside them (verge speckle, tiny one-sided blobs) is dropped.
    import shapely

    final_lines = [f["geometry_2d"] for f in lines]
    for r in polys:  # a mark never covers the paint of a lane line next to it
        clipped = clip_at_lines(r["geometry"], final_lines)
        if clipped is not r["geometry"]:
            from roadmarking.digitization.other_marks import lift_polygon

            r["geometry"] = clipped
            r["geometry_3d"], _ = lift_polygon(clipped, levels[r.get("layer", 0)]["surface"])
            r["review_flags"] = ";".join(
                x for x in [r.get("review_flags"), "cut_back_at_lane_line"] if x
            )
    ltree = shapely.STRtree(final_lines) if final_lines else None
    # junk outside the lanes first (judged on the shape as extracted), then the kept marks are
    # refined: refining never turns a dropped blob into a kept mark
    off_lane = [off_lane_polygon(r["geometry"], final_lines, ltree) for r in polys]
    polys = drop(polys, [r for r, bad in zip(polys, off_lane, strict=True) if not bad], "off_lane")
    polys = drop(
        polys,
        [r for r in polys if not touches_line_paint(r["geometry"], final_lines)],
        "touches_line_paint",
    )
    # every mark fitted to its own paint (position, angle, size; shape kept)
    from roadmarking.app.refine import (
        fit_to_paint,
        paint_iou,
        prune_corners,
        retemplate,
        snap_edges,
        trace_paint_outline,
    )
    from roadmarking.digitization.other_marks import lift_polygon

    mark_paint = early_mark_paint
    refit_gain, fit_log, edges_snapped, corners_removed, traced_outlines = [], [], [], [], []
    retemplated = 0
    for r in polys:
        k = r.get("layer", 0)
        # a mark that missed the client-shape match (merged with a lane line's paint): retry
        # the client shapes on its own paint; keep the template if it covers the paint better
        if r.get("shape_method") != "client_shape" and r["geometry"].area >= 1.0:
            got = retemplate(r["geometry"], mark_paint[k], grid, final_lines, library)
            if got is not None:
                tpl, code, t_iou = got
                cur = paint_iou(r["geometry"], mark_paint[k], grid, final_lines) or 0.0
                new_iou = paint_iou(tpl, mark_paint[k], grid, final_lines) or 0.0
                if new_iou > cur + 0.02:
                    flags = [x for x in (r.get("review_flags") or "").split(";") if x]
                    flags = [x for x in flags if x != "loose_shape_fit" or t_iou < 0.85]
                    flags.append("client_shape_after_line_paint_removed")
                    r.update(
                        geometry=tpl,
                        COD_RST=code,
                        COD_JAE=classifier.jae.get(code),
                        shape_method="client_shape",
                        review_flags=";".join(flags),
                    )
                    r["geometry_3d"], _ = lift_polygon(tpl, levels[k]["surface"])
                    retemplated += 1
        new, before, after = fit_to_paint(r["geometry"], mark_paint[k], grid, final_lines)
        if new is not r["geometry"]:
            refit_gain.append(round(after - before, 3))
        # then every edge (outline and holes) onto the paint boundary
        snapped, n_moved = snap_edges(new, levels[k]["gray"], levels[k]["observed"], grid)
        if n_moved:
            edges_snapped.append(n_moved)
            new = snapped
        # corners that do not help the outline match the paint go (spurs, hooks, notches at
        # bar ends); the remaining edges are snapped onto the paint boundary again
        # (marks fitted with a client shape keep it: the client's own template is exact, and
        # tracing or pruning would lose details such as an arrow's barb)
        template = r.get("shape_method") == "client_shape"
        pruned = new if template else prune_corners(new, mark_paint[k], grid, final_lines)
        if pruned is not new:
            corners_removed.append(len(new.exterior.coords) - len(pruned.exterior.coords))
            resnapped, _ = snap_edges(pruned, levels[k]["gray"], levels[k]["observed"], grid)
            new = resnapped
        # the outline traced from the paint itself (no template / cut / fit artefacts): kept
        # when it matches the paint at least as well as the outline built above
        traced = (
            None
            if template
            else trace_paint_outline(
                new,
                mark_paint[k],
                grid,
                final_lines,
                gray=levels[k]["gray"],
                observed=levels[k]["observed"],
            )
        )
        if traced is not None:
            cand, _ = traced
            now = paint_iou(new, mark_paint[k], grid, final_lines) or 0.0
            alt = paint_iou(cand, mark_paint[k], grid, final_lines) or 0.0
            if alt >= now + 0.005:
                traced_outlines.append(round(alt - now, 3))
                new = cand
        if new is not r["geometry"]:
            r["geometry"] = new
            r["geometry_3d"], _ = lift_polygon(new, levels[k]["surface"])
        iou_now = paint_iou(new, mark_paint[k], grid, final_lines)
        c_ = new.centroid
        if iou_now is not None and core.contains(c_):
            fit_log.append([round(c_.x, 2), round(c_.y, 2), round(iou_now, 3)])
    # two polygons outlining the same painted mark (one mark split by a scan gap): keep one
    uniq = []
    for r in sorted(polys, key=lambda r: -r["geometry"].area):
        g = r["geometry"]
        if any(g.intersection(u["geometry"]).area > 0.7 * g.area for u in uniq):
            continue
        uniq.append(r)
    polys = uniq
    kept_polys = [r for r in polys if core.contains(r["geometry"].centroid)]
    undigitized = paint_not_digitized(levels[0], mark_paint[0], grid, lines, polys, block["core"])
    stats = dict(
        block=block["id"],
        instances=sum(len(lv["instances"]) for lv in levels),
        levels=len(levels),
        tiles=len(grid.windows),
        raw_px_outside_support=sum(d.get("raw_mask_px_on_nodata", 0) for d in diag.values()),
        junction_polygons_dropped=junction_polys,
        lines_through_marks_dropped=through_marks,
        line_paint_polygons_dropped=line_like,
        speckle_polygons_dropped=speckle_polys,
        off_lane_polygons_dropped=int(sum(off_lane)),
        dashes_beside_lines_recovered=recovered_dashes,
        polygons_refit_to_paint=refit_gain,
        polygon_edges_snapped=edges_snapped,
        polygon_corners_removed=corners_removed,
        polygons_traced_from_paint=traced_outlines,
        polygons_retemplated=retemplated,
        lines_cut_between_stripes=cut_count,
        paint_not_digitized=undigitized,
        polygons_extracted=extracted_count,
        polygons_dropped=[
            d_ for d_ in dropped_log if box(*block["core"]).contains(Point(d_[0], d_[1]))
        ],
        polygon_paint_iou=fit_log,
        recentre_shift_median_m=float(np.median(shifts)) if shifts else 0.0,
        centre_stripe_share=centre_log,
        off_centre_stretches=off_centre,
        recentre_shift_max_m=float(np.max(shifts)) if shifts else 0.0,
        polygon_paint_share=poly_log,
        lines_without_visible_paint_dropped=no_paint,
        line_paint_support=support_log,
        seen_keys=seen_keys,
        paint_keys=paint_keys,
        accepted_px_outside_support=sum(
            int((lv["solid"] | lv["dash"])[~lv["observed"]].sum()) for lv in levels
        ),
    )
    return kept_lines, kept_polys, stats


# ---------------------------------------------------------------- stitching blocks
def stitch_lines(features, tol=0.2, max_angle_deg=20.0):
    """Join line pieces cut at block borders: end points within ``tol`` of each other, same
    class and direction. Dashes cut by a border become one dash again."""
    from shapely.geometry import LineString

    def ends(f):
        c = np.asarray(
            f["geometry_3d"].coords if f["geometry_3d"] is not None else f["geometry_2d"].coords
        )
        return c

    changed = True
    while changed:
        changed = False
        pts = []
        for k, f in enumerate(features):
            c = ends(f)
            pts.append((k, 0, c[0][:2], c[1][:2] - c[0][:2]))
            pts.append((k, 1, c[-1][:2], c[-2][:2] - c[-1][:2]))
        if not pts:
            break
        from scipy.spatial import cKDTree

        tree = cKDTree(np.array([p[2] for p in pts]))
        cos = np.cos(np.radians(max_angle_deg))
        for a, b in sorted(tree.query_pairs(tol)):
            ka, ea, _pa, ta = pts[a]
            kb, eb, _pb, tb = pts[b]
            fa, fb = features[ka], features[kb]
            if ka == kb or fa.get("block") == fb.get("block") or fa["class_id"] != fb["class_id"]:
                continue
            ta, tb = ta / max(np.linalg.norm(ta), 1e-9), tb / max(np.linalg.norm(tb), 1e-9)
            if -(ta @ tb) < cos:  # the two pieces must continue each other
                continue
            ca, cb = ends(fa), ends(fb)
            ca = ca if ea == 1 else ca[::-1]
            cb = cb if eb == 0 else cb[::-1]
            if ca.shape[1] != cb.shape[1]:
                ca, cb = ca[:, :2], cb[:, :2]
            merged = np.vstack([ca, cb[1:]])
            g = LineString(merged)
            nf = dict(fa)
            nf["geometry_2d"] = LineString(merged[:, :2])
            nf["geometry_3d"] = g if g.has_z else None
            nf["length_m"] = nf["geometry_2d"].length
            nf["flags"] = sorted(set(fa["flags"]) | set(fb["flags"]))
            nf["block"] = f"{fa.get('block')}+{fb.get('block')}"
            nf["mean_z"] = float(np.nanmean(merged[:, 2])) if merged.shape[1] == 3 else np.nan
            features = [f for k, f in enumerate(features) if k not in (ka, kb)] + [nf]
            changed = True
            break
    return features


# ---------------------------------------------------------------- whole-file occlusion joins
BRIDGE_RULES = dict(
    max_gap_m=40.0,
    max_angle_deg=8.0,
    max_lateral_m=0.25,
    min_long_side_m=6.0,
    min_side_m=1.5,
    min_hidden=0.6,
    max_dark=0.2,
)


def cell_state(keys, xy):
    """True where the 0.25 m cell of each point is in the sorted key set."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    k = cell_keys(xy[:, 0], xy[:, 1])
    if not len(keys):
        return np.zeros(len(k), bool)
    idx = np.minimum(np.searchsorted(keys, k), len(keys) - 1)
    return keys[idx] == k


def line_ends(f, back=2.0):
    """((is_start, end point xyz, outward unit direction), ...) for both ends of a line."""
    c = np.asarray(f["geometry_3d"].coords)
    g = f["geometry_2d"]
    out = []
    for start in (True, False):
        p = c[0] if start else c[-1]
        q = np.asarray(
            g.interpolate(min(back, g.length) if start else max(g.length - back, 0)).coords[0]
        )[:2]
        t = p[:2] - q
        out.append((start, p, t / max(np.linalg.norm(t), 1e-9)))
    return out


def hidden_gap_candidates(lines, seen_keys, paint_keys, rules=None, check_crossing=True):
    """Pairs of solid-line ends that face each other across a gap hidden from the scanner
    (vehicle shadow): same direction, aligned, one side >= 6 m, gap mostly without road
    returns and no visible unpainted road. Returns [(cost, a, a_is_start, b, b_is_start,
    curve, gap_m)]; with ``check_crossing`` a join that would cross another line is left out."""
    import shapely
    from scipy.spatial import cKDTree
    from shapely.geometry import LineString

    from roadmarking.digitization.gap_bridging import hermite

    r = dict(BRIDGE_RULES, **(rules or {}))
    cos = np.cos(np.radians(r["max_angle_deg"]))
    solid = [k for k, f in enumerate(lines) if f["class_id"] == 0 and f["geometry_3d"] is not None]
    end_rec = [(k, *e) for k in solid for e in line_ends(lines[k])]
    if len(end_rec) < 2:
        return []
    others = shapely.STRtree([f["geometry_2d"] for f in lines]) if check_crossing else None
    etree = cKDTree(np.array([e[2][:2] for e in end_rec]))
    out = []
    for i, j in sorted(etree.query_pairs(r["max_gap_m"])):
        a, sa, pa, ta = end_rec[i]
        b, sb, pb, tb = end_rec[j]
        if a == b:
            continue
        la, lb = lines[a]["length_m"], lines[b]["length_m"]
        if max(la, lb) < r["min_long_side_m"] or min(la, lb) < r["min_side_m"]:
            continue
        d = pb[:2] - pa[:2]
        gap = float(np.linalg.norm(d))
        if not 0.3 <= gap <= r["max_gap_m"] or -(ta @ tb) < cos:
            continue
        if ta @ d < cos * gap or -(tb @ d) < cos * gap:
            continue
        lateral = abs(float(d @ np.array([-ta[1], ta[0]])))
        if lateral > r["max_lateral_m"] + 0.005 * gap:
            continue
        k = max(8, int(gap / 0.1))
        mid = pa[:2] + np.linspace(0, 1, k + 2)[1:-1, None] * d
        seen = cell_state(seen_keys, mid)
        painted = cell_state(paint_keys, mid)
        if (
            float((~seen).mean()) < r["min_hidden"]
            or float((seen & ~painted).mean()) > r["max_dark"]
        ):
            continue
        curve = hermite(pa[:2], ta, pb[:2], -tb, step=0.25)
        if check_crossing:
            bridge = LineString(curve)
            if any(
                m not in (a, b) and bridge.crosses(lines[m]["geometry_2d"])
                for m in others.query(bridge, predicate="intersects")
            ):
                continue
        out.append((gap + 20 * lateral, a, sa, b, sb, curve, gap))
    return out


def bridge_occlusions(lines, seen_keys, paint_keys, rules=None):
    """Join solid lines across gaps hidden by vehicles, over the whole file (gaps that cross
    a block border are not seen by the per-block tracing). Best (shortest, best aligned) join
    first, repeated until none is left. Visible unpainted road (a real break) is never joined.
    Returns (lines, number of joins)."""
    from shapely.geometry import LineString

    joins = 0
    while True:
        cands = hidden_gap_candidates(lines, seen_keys, paint_keys, rules)
        if not cands:
            break
        _, a, sa, b, sb, curve, _gap = min(cands, key=lambda c: c[0])
        fa, fb = lines[a], lines[b]
        ca = np.asarray(fa["geometry_3d"].coords)
        cb = np.asarray(fb["geometry_3d"].coords)
        ca = ca[::-1] if sa else ca  # ca ends at the gap
        cb = cb if sb else cb[::-1]  # cb starts at the gap
        za, zb = ca[-1, 2], cb[0, 2]
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
        zc = za + (zb - za) * s / max(s[-1], 1e-9)  # height interpolated across the hidden gap
        merged = np.vstack([ca, np.column_stack([curve[1:-1], zc[1:-1]]), cb])
        nf = dict(fa)
        nf["geometry_3d"] = LineString(merged)
        nf["geometry_2d"] = LineString(merged[:, :2])
        nf["length_m"] = nf["geometry_2d"].length
        nf["mean_z"] = float(np.mean(merged[:, 2]))
        nf["flags"] = sorted(
            set(fa["flags"]) | set(fb["flags"]) | {"INFERRED_UNDER_OCCLUSION (hidden by vehicle)"}
        )
        nf["inferred_segments"] = (
            list(fa.get("inferred_segments", []))
            + list(fb.get("inferred_segments", []))
            + [LineString(curve)]
        )
        lines = [f for k, f in enumerate(lines) if k not in (a, b)] + [nf]
        joins += 1
    return lines, joins


def drop_lines_in_marks(lines, polys, share=0.5):
    """Whole file (marks and lines may come from different blocks): a line running mostly
    inside area marks (traced along a frame's or arrow's own paint), or a short one largely
    inside one mark, is the mark's paint, not a lane line."""
    import shapely

    marks = [r["geometry"] for r in polys if r.get("geometry") is not None]
    if not marks:
        return lines, 0
    tree = shapely.STRtree(marks)
    kept, dropped = [], 0
    for f in lines:
        g = f["geometry_2d"]
        near = [marks[i] for i in tree.query(g.buffer(0.05))]
        if near:
            inside = g.intersection(shapely.union_all([m.buffer(0.05) for m in near])).length
            if inside >= share * g.length or runs_through_marks(g, near):
                dropped += 1
                continue
        kept.append(f)
    return kept, dropped


# ---------------------------------------------------------------- self-check of every output
def self_check(lines, polys, seen_keys, paint_keys, max_where=25):
    """Checks every job's final output for the mistakes found in review, so they are caught
    automatically instead of by eye. Returns {check: {"count": n, "where": [[x, y], ...]}}.
    Checks:
    - unjoined_hidden_gaps: a solid line broken where vehicles hid the road (should be joined);
    - ends_past_paint: a line end running over visible unpainted road;
    - junction_paint_polygons: an 'other mark' polygon that is only merged line paint;
    - overlapping_lines: two lines drawn on top of each other over more than 2 m;
    - excess_vertices: a line keeping vertices that add < 1 cm (should be simplified);
    - lines_without_visible_paint: a line on road with (almost) no bright paint returns;
    - lines_through_marks: a short line drawn across hatch bars or chevrons;
    - zone_outline_polygons: a polygon around a hatch zone (inside not solid paint, or paint
      inside the hole of a frame) instead of one polygon per bar;
    - line_paint_polygons: a thin strip of line paint exported as an area mark;
    - off_lane_polygons: an area mark outside the lanes (verge speckle, tiny one-sided blob);
    - marks_over_lines: an area mark drawn over the paint of a lane line next to it."""
    import shapely

    lines = [dict(f, geometry_2d=shapely.force_2d(f["geometry_3d"])) for f in lines]
    res = {}

    def put(name, pts):
        res[name] = dict(
            count=len(pts),
            where=[[round(float(x), 2), round(float(y), 2)] for x, y in pts[:max_where]],
        )

    # 1. vehicle-hidden gaps left open (including those a join would cross another line)
    cands = hidden_gap_candidates(lines, seen_keys, paint_keys, check_crossing=False)
    put("unjoined_hidden_gaps", [c[5][len(c[5]) // 2] for c in cands])

    # 2. line ends over visible, unpainted road (last 0.5 m before the end)
    past = []
    g2 = [f["geometry_2d"] for f in lines]
    tree = shapely.STRtree(g2)
    for k, f in enumerate(lines):
        g = f["geometry_2d"]
        if g.length < 1.0:
            continue
        for start, p, _ in line_ends(f):
            pt = shapely.Point(p[:2])
            if any(m != k and g2[m].distance(pt) < 0.05 for m in tree.query(pt.buffer(0.05))):
                continue  # junction end on another line
            ds = np.linspace(0.0, 0.5, 6)
            st = [g.interpolate(d if start else g.length - d).coords[0] for d in ds]
            seen = cell_state(seen_keys, st)
            painted = cell_state(paint_keys, st)
            if (seen & ~painted).mean() >= 0.8:
                past.append(p[:2])
    put("ends_past_paint", past)

    # 3. polygons that are only the merged paint of converging lines
    kept, _ = drop_line_paint_polygons(list(polys), g2)
    kept_ids = {id(r) for r in kept}
    put(
        "junction_paint_polygons",
        [np.asarray(r["geometry"].centroid.coords[0]) for r in polys if id(r) not in kept_ids],
    )

    # 4. lines drawn on top of each other
    over = []
    for a, b in zip(*tree.query(g2, predicate="dwithin", distance=0.1), strict=True):
        if a < b:
            shared = g2[a].intersection(g2[b].buffer(0.1))
            if shared.length > 2.0:
                over.append(np.asarray(shared.centroid.coords[0]))
    put("overlapping_lines", over)

    # 5. vertices the 1 cm simplification would still remove
    extra = []
    for f in lines:
        c = np.asarray(f["geometry_3d"].coords)
        if len(c) > 2 and len(simplify_3d(c, [], tol_xy=0.009, tol_z=0.009)) < 0.5 * len(c):
            extra.append(c[len(c) // 2, :2])
    put("excess_vertices", extra)

    # 6. lines with (almost) no visible paint under them
    weak = []
    for f in lines:
        g = f["geometry_2d"]
        st = [g.interpolate(d).coords[0] for d in np.arange(0.25, max(g.length, 0.3), 0.5)]
        seen = cell_state(seen_keys, st)
        if seen.sum() >= 3 and (cell_state(paint_keys, st) & seen).sum() < 0.3 * seen.sum():
            weak.append(np.asarray(g.interpolate(0.5, normalized=True).coords[0]))
    put("lines_without_visible_paint", weak)

    # 7. short lines traced across hatch bars / chevrons
    marks = [r["geometry"] for r in polys]
    zones = hatch_zones(marks)
    put(
        "lines_through_marks",
        [
            np.asarray(f["geometry_2d"].interpolate(0.5, normalized=True).coords[0])
            for f in lines
            if runs_through_marks(f["geometry_2d"], marks, zones=zones)
        ],
    )

    # 8. polygons drawn around a zone instead of the paint (hatch-zone outlines): the inside is
    # not solid paint, or the hole of a frame holds paint (bars)
    zone = []
    for r in polys:
        if r.get("frame_part") == "inner":
            continue  # the inner outline of a frame encloses bare road by design
        g = shapely.force_2d(r["geometry"])
        for part in getattr(g, "geoms", [g]):
            filled = shapely.Polygon(part.exterior)
            if filled.area < 1.0:
                continue
            x0, y0, x1, y1 = filled.bounds
            gx, gy = np.meshgrid(np.arange(x0, x1, 0.1), np.arange(y0, y1, 0.1))
            pts = np.column_stack([gx.ravel(), gy.ravel()])
            in_part = shapely.contains_xy(part, pts[:, 0], pts[:, 1])
            # the hole away from its painted border (0.25 m cells next to paint read as paint)
            hole_core = filled.difference(part.buffer(0.3))
            in_hole = shapely.contains_xy(hole_core, pts[:, 0], pts[:, 1])
            seen = cell_state(seen_keys, pts)
            painted = cell_state(paint_keys, pts) & seen
            solid = painted[in_part].sum() / max((seen & in_part).sum(), 1)
            hole_paint = painted[in_hole].sum() / max((seen & in_hole).sum(), 1)
            if (seen & in_part).sum() >= 20 and (solid < 0.6 or hole_paint > 0.3):
                zone.append(np.asarray(part.representative_point().coords[0]))
    put("zone_outline_polygons", zone)

    # 10. area marks outside the lanes
    tree2 = shapely.STRtree(g2) if g2 else None
    put(
        "off_lane_polygons",
        [
            np.asarray(shapely.force_2d(r["geometry"]).representative_point().coords[0])
            for r in polys
            if off_lane_polygon(shapely.force_2d(r["geometry"]), g2, tree2)
        ],
    )

    # 11. marks drawn over the paint of a lane line next to them
    put(
        "marks_over_lines",
        [
            np.asarray(shapely.force_2d(r["geometry"]).representative_point().coords[0])
            for r in polys
            if any(
                ln.length >= 10.0 and ln.intersection(shapely.force_2d(r["geometry"])).length > 0.3
                for ln in g2
            )
        ],
    )

    # 9. line paint exported as an area mark (a thin strip along a line)
    put(
        "line_paint_polygons",
        [
            np.asarray(shapely.force_2d(r["geometry"]).representative_point().coords[0])
            for r in polys
            if is_line_paint_polygon(shapely.force_2d(r["geometry"]))
            or touches_line_paint(shapely.force_2d(r["geometry"]), g2)
        ],
    )
    return res


SELF_CHECK_TEXT = dict(
    unjoined_hidden_gaps="solid lines left broken where vehicles hid the road",
    ends_past_paint="line ends running past the painted marking",
    junction_paint_polygons="junction paint exported as an 'other mark' polygon",
    overlapping_lines="lines drawn on top of each other",
    excess_vertices="lines with unnecessary vertices",
    lines_without_visible_paint="lines with almost no visible paint under them",
    lines_through_marks="short lines drawn across hatch bars or chevrons",
    zone_outline_polygons="polygons drawn around a hatch zone instead of each bar",
    line_paint_polygons="thin strips of line paint exported as area marks",
    off_lane_polygons="area marks outside the lanes (verge speckle, tiny blobs)",
    marks_over_lines="area marks drawn over the paint of a lane line",
    off_centre_lines="line stretches (>= 2 m) more than 2 cm off the paint centre",
    polygons_not_covering_paint="area marks whose outline does not match their paint",
    paint_not_digitized="painted patches (>= 1 m2) in the lanes with no line or polygon",
)


# ---------------------------------------------------------------- 6. export
def simplify_lines(lines):
    """Fewest 3D vertices within 1 cm sideways and 1 cm in height; ends and vertices shared
    with another line's end (junction snaps) are kept."""
    from shapely.geometry import LineString

    ends = np.array(
        [c for f in lines for c in (f["geometry_3d"].coords[0], f["geometry_3d"].coords[-1])]
    )[:, :2]
    from scipy.spatial import cKDTree

    tree = cKDTree(ends) if len(ends) else None
    for f in lines:
        c = np.asarray(f["geometry_3d"].coords)
        keep = []
        if tree is not None:
            d, _ = tree.query(c[:, :2], k=1)
            keep = list(np.flatnonzero(d < 1e-6))
        f["geometry_3d"] = LineString(simplify_3d(c, keep))
    return lines


def client_line_rows(lines, cod_jae):
    from shapely.geometry import MultiLineString

    from roadmarking.digitization.export import client_rows

    rows, unresolved = client_rows(lines, cod_jae)
    for row, f in zip(rows, lines, strict=True):
        if row["LARGURA"] is not None and not 0.08 <= row["LARGURA"] <= 0.6:
            row["LARGURA"] = None  # implausible measured width (thin scan): left for review
        row["COMENTARIO"] = "Ativo novo"  # client enum: a new asset (not in the old database)
        flags = [x for x in f["flags"] if x]
        row["OBS"] = "Plotline-AI automatic digitization; pending review" + (
            f"; flags: {', '.join(sorted(set(flags)))}" if flags else ""
        )
        row["geometry"] = MultiLineString([f["geometry_3d"]])
    return rows, unresolved


def split_frames(polys):
    """Client format for framed marks (give-way triangles drawn as a painted border): two
    solid polygons, the outer outline and the inner outline, instead of one polygon with a
    hole. Both keep the mark's type; heights come from the fitted 3D polygon."""
    from shapely.geometry import Polygon

    out = []
    for r in polys:
        g2, g3 = r["geometry"], r["geometry_3d"]
        if g2.geom_type != "Polygon" or not g2.interiors:
            out.append(r)
            continue
        flags = r.get("review_flags") or ""
        out.append(
            dict(
                r,
                geometry=Polygon(g2.exterior),
                geometry_3d=Polygon(g3.exterior),
                frame_part="outer",
                review_flags=";".join(x for x in [flags, "frame_outer"] if x),
            )
        )
        for h2, h3 in zip(g2.interiors, g3.interiors, strict=True):
            out.append(
                dict(
                    r,
                    geometry=Polygon(h2),
                    geometry_3d=Polygon(h3),
                    frame_part="inner",
                    review_flags=";".join(x for x in [flags, "frame_inner"] if x),
                )
            )
    return out


def client_poly_rows(polys):
    from shapely.geometry import MultiPolygon

    rows = []
    for i, r in enumerate(polys, start=1):
        g = r["geometry_3d"]
        z = np.asarray(g.exterior.coords)[:, 2]
        rows.append(
            {
                "ID_SIG": i,
                "POSICAO": None,
                "COD_RST": r["COD_RST"],
                "ALGARISMO": None,
                "LETRA": None,
                "ALTURA": None,
                "AREA_PINTA": round(float(r["geometry"].area), 3),
                "ID_GRUPO": None,
                "COMENTARIO": "Ativo novo",
                "ID_HIST": None,
                "ELEVATION": round(float(np.mean(z)), 3),
                "OBS": "Plotline-AI automatic digitization; pending review; type is a suggestion"
                + (f"; flags: {r['review_flags']}" if r.get("review_flags") else ""),
                "COD_SUBLANCO": None,
                "COD_JAE": r["COD_JAE"],
                "LARGURA": None,
                "geometry": MultiPolygon([g]),
            }
        )
    return rows


def write_output(path: Path, line_rows, poly_rows, crs, metadata):
    import geopandas as gpd
    import pandas as pd
    import pyogrio

    for name, rows, fields, gtype in (
        (CLIENT_LINE_LAYER, line_rows, LINE_FIELDS, "MultiLineString Z"),
        (CLIENT_POLY_LAYER, poly_rows, POLY_FIELDS, "MultiPolygon Z"),
    ):
        frame = pd.DataFrame(rows, columns=[f for f, _ in fields] + ["geometry"])
        for f, dt in fields:
            frame[f] = frame[f].astype(dt) if dt != "object" else frame[f].astype(object)
        gdf = gpd.GeoDataFrame(frame, geometry="geometry", crs=crs)
        pyogrio.write_dataframe(
            gdf,
            path,
            layer=name,
            driver="GPKG",
            append=path.exists(),
            geometry_type=gtype,
            layer_options={"SPATIAL_INDEX": "YES"},
        )
    if metadata:
        meta = pd.DataFrame(
            [
                dict(key=k, value=v if isinstance(v, str) else json.dumps(v))
                for k, v in metadata.items()
            ]
        )
        pyogrio.write_dataframe(
            meta, path, layer="plotline_run_metadata", driver="GPKG", append=True
        )


# ---------------------------------------------------------------- 7. verification
def verify_output(path: Path, crs, las_bounds, expect_lines, expect_polys):
    import pyogrio
    import shapely

    problems = []
    layers = dict(pyogrio.list_layers(path))
    for name, gtype in (
        (CLIENT_LINE_LAYER, "MultiLineString Z"),
        (CLIENT_POLY_LAYER, "MultiPolygon Z"),
    ):
        if name not in layers:
            problems.append(f"missing layer {name}")
            continue
        info = pyogrio.read_info(path, layer=name)
        df = pyogrio.read_dataframe(path, layer=name)
        expected = expect_lines if name == CLIENT_LINE_LAYER else expect_polys
        if len(df) != expected:
            problems.append(f"{name}: {len(df)} features read back, {expected} written")
        if info["geometry_type"] != gtype:
            problems.append(f"{name}: geometry type {info['geometry_type']} (expected {gtype})")
        fields = LINE_FIELDS if name == CLIENT_LINE_LAYER else POLY_FIELDS
        if [f for f, _ in fields] != list(info["fields"]):
            problems.append(f"{name}: fields differ from the client schema")
        if len(df):
            if not df.geometry.is_valid.all():
                problems.append(f"{name}: invalid geometries")
            if not df.geometry.has_z.all():
                problems.append(f"{name}: features without Z")
            xyz = shapely.get_coordinates(df.geometry.values, include_z=True)
            if not np.isfinite(xyz).all():
                problems.append(f"{name}: non-finite coordinates")
            mn, mx = xyz.min(0), xyz.max(0)
            lo, hi = np.asarray(las_bounds[0]), np.asarray(las_bounds[1])
            if (mn[:2] < lo[:2] - 1).any() or (mx[:2] > hi[:2] + 1).any():
                problems.append(f"{name}: coordinates outside the LAS/LAZ footprint")
            if (mn[2] < lo[2] - 1) or (mx[2] > hi[2] + 1):
                problems.append(f"{name}: heights outside the LAS/LAZ height range")
            if df["ID_SIG"].duplicated().any():
                problems.append(f"{name}: duplicate ID_SIG")
            if name == CLIENT_LINE_LAYER and (df.geometry.length < 0.05).any():
                problems.append(f"{name}: zero-length lines")
        if crs is not None and info["crs"] is None:
            problems.append(f"{name}: CRS missing")
    con = sqlite3.connect(path)
    try:  # closed explicitly: an open handle keeps the file locked on Windows
        rtrees = {
            r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE name LIKE 'rtree_%'")
        }
        ok = con.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        con.close()
    for name in (CLIENT_LINE_LAYER, CLIENT_POLY_LAYER):
        if not any(r.startswith(f"rtree_{name}_") for r in rtrees):
            problems.append(f"{name}: spatial index missing")
    if ok != "ok":
        problems.append(f"GeoPackage integrity check: {ok}")
    return problems


# ---------------------------------------------------------------- reference shapes
def shape_references(exclude_box):
    """Client polygons as shape examples for the other-marks classifier and arrow library
    (nothing inside the job's own footprint, so the job never copies the client's answers)."""
    import pandas as pd
    import pyogrio
    import shapely

    from roadmarking.config import PROJECT_ROOT
    from roadmarking.digitization.other_marks import ShapeClassifier, descriptor
    from roadmarking.digitization.polygon_fit import ShapeLibrary

    ann = PROJECT_ROOT / "data/raw/annotations"
    examples, shapes = [], []
    pairs = (
        ("Highway/Other Marks.gpkg", "Highway/Longitudinal Marks .gpkg"),
        (
            "Interchange/interchange_Other Marks.gpkg",
            "Interchange/Interchange-Longitudinal Marks.gpkg",
        ),
    )
    cod = {}
    for other, line in pairs:
        if not (ann / other).exists():
            continue
        frame = pyogrio.read_dataframe(ann / other, columns=["COD_RST", "COD_JAE"])
        lines = pyogrio.read_dataframe(ann / line, columns=["DESIGNACAO", "COD_JAE"]).explode(
            index_parts=False
        )
        for d, v in lines.groupby("DESIGNACAO")["COD_JAE"]:
            if v.nunique() == 1:
                cod.setdefault(d, v.iloc[0])
        lines = lines[lines.geometry.notna() & ~lines.intersects(exclude_box)]
        geoms = list(lines.geometry)
        tree = shapely.STRtree(geoms)
        for row in frame.explode(index_parts=False).itertuples():
            g = row.geometry
            if g is None or not g.is_valid or g.area <= 0 or g.intersects(exclude_box):
                continue
            rst = row.COD_RST if pd.notna(row.COD_RST) else None
            jae = row.COD_JAE if pd.notna(row.COD_JAE) else None
            if rst is None:
                continue
            examples.append((descriptor(g, [geoms[tree.nearest(g)]] if geoms else []), (rst, jae)))
            shapes.append((shapely.force_2d(g), rst))
    return ShapeClassifier(examples), ShapeLibrary(shapes), cod


# ---------------------------------------------------------------- parallel blocks
_WORKER = {}
GPU_MB_PER_WORKER = 1100  # CUDA context (~0.4 GB) + model peak (measured ~0.7 GB)


def resolve_model():
    from roadmarking.models.production import resolve

    return resolve(None, verify=True)


def choose_workers(n_blocks, max_workers=4):
    """Blocks are independent, and most of a block's time is CPU work, so several blocks run
    at once, each worker with its own copy of the model. Bounded by CPU cores, free GPU memory
    and the number of blocks."""
    import psutil
    import torch

    cores = psutil.cpu_count(logical=False) or 2
    free_mb = torch.cuda.mem_get_info()[0] / 2**20
    by_gpu = int((free_mb - 500) // GPU_MB_PER_WORKER)
    return max(1, min(max_workers, cores // 2, by_gpu, n_blocks))


def raise_priority():
    """Plotline-AI's own processes run at "above normal" priority, so other programs (an open
    GIS window, a viewer) do not take CPU time from a job. Not a system setting: it applies
    to these processes only, while they run."""
    import contextlib

    import psutil

    with contextlib.suppress(Exception):
        psutil.Process().nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)


def choose_pools(n_blocks, max_gpu=2):
    """(GPU workers, CPU workers). Recognition (GPU, about half of a block's time) runs in one
    or two processes that each hold the model; the digitizing (CPU) runs from the cached
    recognition in separate processes that need no GPU memory - so a program holding GPU memory
    (an open 3D view) no longer limits how many blocks are digitized at once."""
    import psutil
    import torch

    cores = psutil.cpu_count(logical=False) or 2
    free_mb = torch.cuda.mem_get_info()[0] / 2**20
    gpu = max(1, min(max_gpu, int((free_mb - 500) // GPU_MB_PER_WORKER), n_blocks))
    cpu = max(1, min(cores // 2, n_blocks))
    return gpu, cpu


def _init_gpu_worker():
    import torch

    from roadmarking.models.production import load_production_model

    raise_priority()
    torch.set_num_threads(2)
    _WORKER.update(bundle=load_production_model(None, "cuda"))


def _gpu_stage(block, las_paths, work):
    """Rasterize the block and run the recognition; the result lands in the job's cache, where
    the digitizing reads it (same inputs -> same cache key -> identical result)."""
    from roadmarking.digitization.raster_tiles import make_grid
    from roadmarking.digitization.stages import build_levels

    grid = make_grid(block["bbox"])
    build_levels([str(p) for p in las_paths], grid, work, bundle=_WORKER["bundle"])
    return block["id"]


def _init_cpu_worker(classifier, library):
    import torch

    raise_priority()
    torch.set_num_threads(2)
    _WORKER.update(bundle=None, classifier=classifier, library=library)


def _init_worker(classifier, library):
    import torch

    from roadmarking.models.production import load_production_model

    raise_priority()
    torch.set_num_threads(2)
    _WORKER.update(
        bundle=load_production_model(None, "cuda"), classifier=classifier, library=library
    )


def _block_worker(block, las_paths, work):
    progress = Progress(None, threading.Event())
    progress.enter("RECOGNIZING")
    return process_block(
        block,
        las_paths,
        work,
        _WORKER["bundle"],
        _WORKER["classifier"],
        _WORKER["library"],
        progress,
        (0, 1),
    )


def run_blocks(blocks, las_paths, work, classifier, library, progress, workers):
    """Process every block; results in block order (identical to running them one by one)."""
    total = len(blocks)
    results = {}
    if workers == (1, 1) or workers == 1:
        _init_worker(classifier, library)
        try:
            for n, b in enumerate(blocks):
                progress.update(n / total, f"Running AI recognition (block {n + 1} of {total})")
                results[n] = _block_worker(b, las_paths, work)
        finally:
            _WORKER.clear()
    else:
        import multiprocessing as mp
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

        n_gpu, n_cpu = workers
        ctx = mp.get_context("spawn")
        gpu_pool = ProcessPoolExecutor(n_gpu, mp_context=ctx, initializer=_init_gpu_worker)
        cpu_pool = ProcessPoolExecutor(
            n_cpu, mp_context=ctx, initializer=_init_cpu_worker, initargs=(classifier, library)
        )
        pools = (gpu_pool, cpu_pool)
        try:
            progress.update(0.0, f"Starting {n_gpu} recognition + {n_cpu} digitizing workers")
            recog = {
                gpu_pool.submit(_gpu_stage, b, las_paths, work): n for n, b in enumerate(blocks)
            }
            digit, recognized = {}, 0
            while recog or digit:
                done, _ = wait(list(recog) + list(digit), timeout=1.0, return_when=FIRST_COMPLETED)
                for fut in done:
                    if fut in recog:
                        n = recog.pop(fut)
                        fut.result()  # recognition errors surface here
                        recognized += 1
                        digit[cpu_pool.submit(_block_worker, blocks[n], las_paths, work)] = n
                    else:
                        results[digit.pop(fut)] = fut.result()
                progress.update(
                    (recognized + len(results)) / (2 * total),
                    f"Running AI recognition ({recognized} of {total} blocks recognized, "
                    f"{len(results)} digitized; {n_gpu} + {n_cpu} workers)",
                )
        except BaseException:
            for pool in pools:
                for proc in list(getattr(pool, "_processes", {}).values()):
                    proc.terminate()  # cancel or failure: stop workers now, not after their block
                pool.shutdown(wait=False, cancel_futures=True)
            raise
        for pool in pools:
            pool.shutdown(wait=True)
    lines, polys, stats = [], [], []
    for n in range(total):
        bl, bp, st = results[n]
        lines += bl
        polys += bp
        stats.append(st)
    return lines, polys, stats


# ---------------------------------------------------------------- driver
def run_job(
    las_path,
    out_path,
    progress_cb=None,
    cancel=None,
    keep_work=False,
    overwrite=False,
    extra_las=(),
):
    """Run the full Plotline-AI workflow. Returns JobResult; raises JobError or Cancelled.

    ``extra_las``: optional overlapping files of the same survey (other passes/directions). A
    single file usually covers only part of the road; overlapping files are fused per cell.
    """
    import psutil
    import torch
    from shapely.geometry import box

    t0 = time.perf_counter()
    las_path, out_path = Path(las_path).resolve(), Path(out_path).resolve()
    las_paths = [las_path] + [Path(p).resolve() for p in extra_las if Path(p).resolve() != las_path]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    work = app_home() / "jobs" / f"{stamp}_{las_path.stem}"
    logs = app_home() / "logs"
    (work / "extracts").mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    progress = Progress(progress_cb, cancel)
    times, warnings, qa = {}, [], {}
    log = logs / f"{stamp}_{las_path.stem}.json"
    success = False
    try:
        if out_path.exists() and not overwrite:
            raise JobError(f"The output file already exists: {out_path}")
        # 1
        progress.enter("VALIDATING")
        t = time.perf_counter()
        checked = [validate_input(p, out_path, work) for p in las_paths]
        _hdr, out_crs, crs_text, crs_warning = checked[0]
        if any(c[1] != out_crs for c in checked[1:]):
            raise JobError(
                "The selected LAS/LAZ files do not share the same coordinate reference system."
            )
        if crs_warning:
            warnings.append(crs_warning)
        hdrs = [c[0] for c in checked]
        footprint_mins = np.min([h["mins"] for h in hdrs], axis=0)
        footprint_maxs = np.max([h["maxs"] for h in hdrs], axis=0)
        if not torch.cuda.is_available():
            raise JobError(
                "No CUDA GPU is available. The accepted recognition model is validated on GPU "
                "(FP16 autocast); processing on CPU is not an approved configuration."
            )
        times["validation_s"] = time.perf_counter() - t
        # 2
        progress.enter("RASTERIZING")
        t = time.perf_counter()
        blocks, files = stream_blocks(las_paths, hdrs, work, progress)
        if not blocks:
            raise JobError("The file has too few points for road-marking processing.")
        times["intensity_extraction_s"] = time.perf_counter() - t
        # 3-5
        progress.enter("RECOGNIZING", "Loading the accepted AI model")

        spec = resolve_model()  # hash-verified once here; workers load the same files
        footprint = box(
            footprint_mins[0], footprint_mins[1], footprint_maxs[0], footprint_maxs[1]
        ).buffer(200)
        classifier, library, cod = shape_references(footprint)
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        raise_priority()
        workers = choose_pools(len(blocks))
        times["parallel_workers"] = workers
        lines, polys, block_stats = run_blocks(
            blocks, las_paths, work, classifier, library, progress, workers
        )
        times["recognition_digitization_z_s"] = time.perf_counter() - t
        progress.enter("DIGITIZING", "Joining road markings across blocks")
        t = time.perf_counter()
        lines = stitch_lines(lines)
        seen_keys = np.unique(np.concatenate([s_.pop("seen_keys") for s_ in block_stats]))
        paint_keys = np.unique(np.concatenate([s_.pop("paint_keys") for s_ in block_stats]))
        lines, occlusion_joins = bridge_occlusions(lines, seen_keys, paint_keys)
        lines, lines_in_marks = drop_lines_in_marks(lines, polys)
        times["stitching_s"] = time.perf_counter() - t
        if any(s["accepted_px_outside_support"] for s in block_stats):
            raise JobError(
                "Accepted paint was found outside observed LiDAR support (safety check)."
            )
        progress.enter("RECOVERING_Z")
        export_lines = [f for f in lines if f["geometry_3d"] is not None and f["length_m"] >= 0.3]
        export_polys = split_frames([r for r in polys if r["geometry_3d"] is not None])
        skipped = dict(
            lines_without_z=sum(f["geometry_3d"] is None for f in lines),
            polygons_without_z=sum(r["geometry_3d"] is None for r in polys),
        )
        if skipped["lines_without_z"] or skipped["polygons_without_z"]:
            warnings.append(
                f"{skipped['lines_without_z']} lines and {skipped['polygons_without_z']} polygons "
                "had no reliable road-surface height and were not exported (3D layers)."
            )
        # 6
        progress.enter("EXPORTING")
        t = time.perf_counter()
        export_lines = simplify_lines(export_lines)
        line_rows, unresolved = client_line_rows(export_lines, cod)
        poly_rows = client_poly_rows(export_polys)
        tmp = work / "output.gpkg"
        if tmp.exists():
            tmp.unlink()
        metadata = dict(
            application="Plotline-AI",
            application_version=APP_VERSION,
            status="AUTOMATIC DIGITIZATION - PENDING HUMAN REVIEW",
            source_files=[
                dict(
                    path=str(f["path"]),
                    sha256=f["sha256"],
                    points_read=f["points_read"],
                    points_dropped=f["dropped_points"],
                )
                for f in files
            ],
            coordinate_reference=crs_text,
            model=spec.name,
            model_weights_sha256=spec.entry["weights_sha256"],
            model_classes=spec.classes,
            thresholds=spec.thresholds_v1,
            supported_marks=dict(
                lines="solid and dashed longitudinal lines (AI model)",
                polygons="arrows (client shapes), hatch bars, chevrons, colour bands, give-way and "
                "framed marks (paint geometry; COD_RST is a suggestion)",
                not_supported="road text (letters/digits), pedestrian crossings, stop lines",
            ),
            client_fields_not_determined=unresolved,
            blocks=len(blocks),
            created_utc=stamp,
        )
        progress.update(0.5, "Self-checking the result")
        checks = self_check(
            export_lines,
            [r for r in export_polys if r.get("frame_part") != "inner"],  # bare road by design
            seen_keys,
            paint_keys,
        )
        loose = [
            p_ for b in block_stats for p_ in b.get("polygon_paint_iou", []) if p_[2] < MIN_MARK_IOU
        ]
        checks["polygons_not_covering_paint"] = dict(
            count=len(loose), where=[[x_, y_] for x_, y_, _ in loose[:25]]
        )
        patches = [p_ for b in block_stats for p_ in b.get("paint_not_digitized", [])]
        checks["paint_not_digitized"] = dict(
            count=len(patches), where=[[x_, y_] for x_, y_, _ in patches[:25]]
        )
        stretches = [s_ for b in block_stats for s_ in b.get("off_centre_stretches", [])]
        checks["off_centre_lines"] = dict(
            count=len(stretches), where=[[x_, y_] for x_, y_, *_ in stretches[:25]]
        )
        found = {k: v["count"] for k, v in checks.items() if v["count"]}
        metadata["self_check"] = {k: v["count"] for k, v in checks.items()}
        if found:
            warnings.append(
                "Self-check found places to review: "
                + "; ".join(f"{n} {SELF_CHECK_TEXT[k]}" for k, n in found.items())
                + " (coordinates in the job log)."
            )
        write_output(tmp, line_rows, poly_rows, out_crs, metadata)
        times["export_s"] = time.perf_counter() - t
        # 7
        progress.enter("VERIFYING")
        problems = verify_output(
            tmp, out_crs, (footprint_mins, footprint_maxs), len(line_rows), len(poly_rows)
        )
        from roadmarking.digitization.raster_tiles import sha256

        for f in files:
            if sha256(f["path"]) != f["sha256"]:
                problems.append(f"the source {f['path'].name} changed during processing")
        if problems:
            raise JobError("The output failed validation: " + "; ".join(problems))
        progress.update(0.8, "Saving the GeoPackage")
        partial = out_path.with_name(f".{out_path.name}.partial")
        shutil.copyfile(tmp, partial)
        os.replace(partial, out_path)
        progress.update(1.0, "Completed")
        success = True
        times["total_s"] = time.perf_counter() - t0
        qa = dict(
            blocks=block_stats,
            tiles=sum(s["tiles"] for s in block_stats),
            instances=sum(s["instances"] for s in block_stats),
            exported=dict(lines=len(line_rows), polygons=len(poly_rows)),
            skipped=skipped,
            occlusion_joins_across_blocks=occlusion_joins,
            lines_in_marks_dropped_whole_file=lines_in_marks,
            self_check=checks,
            times=times,
            peak_ram_mib=getattr(psutil.Process().memory_info(), "peak_wset", 0) / 2**20,
            peak_gpu_mib=torch.cuda.max_memory_allocated() / 2**20,
            warnings=warnings,
            metadata=metadata,
        )
        return JobResult(
            out_path, times["total_s"], len(line_rows), len(poly_rows), crs_text, warnings, log
        )
    except Cancelled:
        qa = dict(status="cancelled")
        raise
    except JobError as e:
        qa = dict(status="failed", error=str(e))
        raise
    except Exception as e:
        qa = dict(status="failed", error=repr(e), trace=traceback.format_exc())
        raise JobError(f"Processing failed: {e}") from e
    finally:
        qa.setdefault("status", "completed" if success else "failed")
        qa.update(input=[str(p) for p in las_paths], output=str(out_path))
        log.write_text(json.dumps(qa, indent=1, default=str))
        if not keep_work:
            shutil.rmtree(work, ignore_errors=True)


__all__ = ["STAGES", "Cancelled", "JobError", "JobResult", "run_job", "validate_input"]
