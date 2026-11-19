"""Production-compatible intensity rasterization of a region on an overlapping tile grid.

Same path as the approved crops: per-cell maximum intensity (RasterAccumulator), per-
acquisition 1-99.5 % calibration over every point of the source file, gamma 0.7 grayscale.
Pixel (row, col) covers [x0 + col*c, x0 + (col+1)*c) x (y0 - (row+1)*c, y0 - row*c];
vertex coordinates use pixel CENTRES. Region points (x, y, z) are kept for Z recovery.

The client specification requires every track covering a road part to be used (both sides of
the road), so a region may be built from several overlapping source files: each file keeps its
own calibration and the files are fused per cell (``fuse_sources``).
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from affine import Affine

from roadmarking.io.pointcloud import ChunkReadReport, iter_laz_chunks
from roadmarking.preprocessing.intensity import calibrate_histogram
from roadmarking.rasterization.rasterizer import RasterAccumulator, world_to_pixel
from roadmarking.visualization.intensity_render import grayscale_intensity

CELL = 0.05
TILE = 512
STRIDE = 384


@dataclass
class Grid:
    x0: float
    y0: float
    rows: int
    cols: int
    cell: float = CELL
    tile: int = TILE
    stride: int = STRIDE
    windows: dict = field(default_factory=dict)

    @property
    def affine(self) -> Affine:
        return Affine(self.cell, 0, self.x0, 0, -self.cell, self.y0)

    @property
    def bounds(self):
        return (self.x0, self.y0 - self.rows * self.cell, self.x0 + self.cols * self.cell, self.y0)


def make_grid(bbox, cell=CELL, tile=TILE, stride=STRIDE) -> Grid:
    """Tile grid on the global ``cell`` lattice covering ``bbox`` = (minx, miny, maxx, maxy)."""
    minx, miny, maxx, maxy = bbox
    x0 = math.floor(minx / cell) * cell
    y0 = math.ceil(maxy / cell) * cell
    need_c = math.ceil((maxx - x0) / cell)
    need_r = math.ceil((y0 - miny) / cell)
    ntx = max(1, math.ceil(max(need_c - tile, 0) / stride) + 1)
    nty = max(1, math.ceil(max(need_r - tile, 0) / stride) + 1)
    grid = Grid(x0, y0, tile + (nty - 1) * stride, tile + (ntx - 1) * stride, cell, tile, stride)
    for i in range(nty):
        for j in range(ntx):
            grid.windows[f"T{i:02d}_{j:02d}"] = (i * stride, j * stride)
    return grid


def pixel_to_xy(grid: Grid, rows, cols):
    """Native XY of (sub)pixel coordinates; integer indices map to pixel centres."""
    rows, cols = np.asarray(rows, float), np.asarray(cols, float)
    return grid.x0 + (cols + 0.5) * grid.cell, grid.y0 - (rows + 0.5) * grid.cell


def xy_to_pixel(grid: Grid, xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    return (grid.y0 - ys) / grid.cell - 0.5, (xs - grid.x0) / grid.cell - 0.5


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _extract_key(las_path, bounds, margin):
    p = Path(las_path)
    tag = f"{p.name}|{p.stat().st_size}|{[round(float(v), 3) for v in bounds]}|{margin}"
    return f"{p.stem}__{hashlib.sha1(tag.encode()).hexdigest()[:12]}.npz"


def _finish_extract(rec, las_path):
    return dict(
        path=str(las_path),
        name=Path(las_path).name,
        x=rec["x"],
        y=rec["y"],
        z=rec["z"],
        intensity=rec["intensity"],
        hist=rec["hist"],
        sha256=str(rec["sha256"]),
        seconds=float(rec["seconds"]),
        read_report=dict(
            points_read=int(rec["points_read"]),
            dropped_points=int(rec["dropped_points"]),
            corrupt_chunks=[list(map(int, c)) for c in rec["corrupt_chunks"]],
        ),
    )


def extract_regions(las_path, bounds_list, margin=2.0, cache_dir=None):
    """One streaming pass over a source file: the whole-file intensity histogram (for the
    per-acquisition calibration) and, per region, the points inside ``bounds`` (+ ``margin``).

    With ``cache_dir`` each extract is stored once and reused, so a region can be re-processed
    (and several overlapping files fused) without reading the LAS/LAZ again. Sources are only
    ever opened read-only.
    """
    las_path = Path(las_path)
    cache = [
        (Path(cache_dir) / _extract_key(las_path, b, margin)) if cache_dir else None
        for b in bounds_list
    ]
    out = [None] * len(bounds_list)
    for k, c in enumerate(cache):
        if c is not None and c.exists():
            with np.load(c, allow_pickle=False) as z:
                out[k] = {key: z[key] for key in z.files}
    if all(o is not None for o in out):
        return [_finish_extract(o, las_path) for o in out]
    t0 = time.perf_counter()
    hist = np.zeros(65536, np.uint64)
    report = ChunkReadReport(las_path, 0, 0)
    parts = [[] for _ in bounds_list]
    for chunk in iter_laz_chunks(las_path, report):
        x, y = np.asarray(chunk.x), np.asarray(chunk.y)
        raw = np.asarray(chunk.intensity)
        hist += np.bincount(raw, minlength=65536).astype(np.uint64)
        for k, (minx, miny, maxx, maxy) in enumerate(bounds_list):
            if out[k] is not None:
                continue
            near = (x >= minx - margin) & (x < maxx + margin)
            near &= (y > miny - margin) & (y <= maxy + margin)
            if near.any():
                parts[k].append((x[near], y[near], np.asarray(chunk.z)[near], raw[near]))
    digest = sha256(las_path)
    for k, c in enumerate(cache):
        if out[k] is not None:
            continue
        cols = [np.concatenate(v) for v in zip(*parts[k], strict=True)] if parts[k] else None
        rec = dict(
            x=cols[0] if cols else np.zeros(0),
            y=cols[1] if cols else np.zeros(0),
            z=cols[2] if cols else np.zeros(0),
            intensity=(cols[3] if cols else np.zeros(0)).astype(np.uint16),
            hist=hist,
            points_read=np.int64(report.points_read),
            dropped_points=np.int64(report.dropped_points),
            corrupt_chunks=np.asarray(report.corrupt_chunks, np.int64).reshape(-1, 2),
            sha256=np.str_(digest),
            seconds=np.float64(time.perf_counter() - t0),
        )
        if c is not None:
            c.parent.mkdir(parents=True, exist_ok=True)
            tmp = c.with_name(c.stem + ".tmp.npz")
            np.savez_compressed(tmp, **rec)
            tmp.replace(c)
        out[k] = rec
    return [_finish_extract(o, las_path) for o in out]


@dataclass
class HeightGrid:
    """Surface height on a coarse lattice: ``z[iy, ix]`` for the column whose lower-left
    corner is (x0 + ix*cell, y0 + iy*cell); NaN where the column holds no surface."""

    x0: float
    y0: float
    cell: float
    z: np.ndarray

    def at(self, x, y, same_surface_m=0.75):
        """Height at (x, y): bilinear between neighbouring columns that belong to the same
        surface as the point's own column (a deck edge or a wall is never averaged with the
        ground beside it). NaN where no surface is known."""
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        ny, nx = self.z.shape

        def cellz(iy, ix):
            ok = (iy >= 0) & (ix >= 0) & (iy < ny) & (ix < nx)
            out = np.full(np.shape(ix), np.nan)
            out[ok] = self.z[iy[ok], ix[ok]]
            return out

        gx, gy = (x - self.x0) / self.cell, (y - self.y0) / self.cell
        own = cellz(np.floor(gy).astype(np.int64), np.floor(gx).astype(np.int64))
        ix0, iy0 = np.floor(gx - 0.5).astype(np.int64), np.floor(gy - 0.5).astype(np.int64)
        tx, ty = gx - 0.5 - ix0, gy - 0.5 - iy0
        acc, wsum = np.zeros(x.shape), np.zeros(x.shape)
        for dy in (0, 1):
            for dx in (0, 1):
                b = cellz(iy0 + dy, ix0 + dx)
                w = (tx if dx else 1 - tx) * (ty if dy else 1 - ty)
                good = np.isfinite(b) & (~np.isfinite(own) | (np.abs(b - own) <= same_surface_m))
                acc += np.where(good, w * b, 0.0)
                wsum += np.where(good, w, 0.0)
        return np.where(wsum > 1e-9, acc / np.maximum(wsum, 1e-9), own)


def surface_band(x, y, z, cell=1.0, band=(-0.15, 0.25), cluster_m=0.2, min_cluster=3, top=False):
    """Points on the lowest (or, with ``top``, the highest) dense surface of each
    ``cell`` x ``cell`` column. Returns (keep mask, HeightGrid of that surface).

    The surface height of a column is the lowest z that has at least ``min_cluster`` points
    within ``cluster_m`` above it, so isolated below-ground noise cannot define it. Points
    inside ``band`` (metres relative to the interpolated surface) are kept. Used with all
    overlapping acquisitions together: tree canopy, gantries, a bridge deck above the road
    and vehicle bodies are dropped, and another pass that saw the pavement fills the cell.
    """
    x, y, z = (np.asarray(v, np.float64) for v in (x, y, z))
    if len(z) == 0:
        return np.zeros(0, bool), HeightGrid(0.0, 0.0, cell, np.full((1, 1), np.nan))
    if top:
        keep, grid = surface_band(x, y, -z, cell, band, cluster_m, min_cluster)
        return keep, HeightGrid(grid.x0, grid.y0, cell, -grid.z)
    x0, y0 = float(x.min()), float(y.min())
    ix = np.floor((x - x0) / cell).astype(np.int64)
    iy = np.floor((y - y0) / cell).astype(np.int64)
    nx, ny = int(ix.max()) + 1, int(iy.max()) + 1
    key = iy * nx + ix
    order = np.lexsort((z, key))
    ks, zs = key[order], z[order]
    comp = ks.astype(np.float64) * 1e5 + (zs - zs.min())  # z range is far below 1e5 m
    dense = np.searchsorted(comp, comp + cluster_m, side="right") - np.arange(len(comp))
    starts = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
    counts = np.diff(np.r_[starts, len(ks)])
    ok = dense >= np.repeat(np.minimum(min_cluster, counts), counts)
    first = np.minimum.reduceat(np.where(ok, np.arange(len(ks)), len(ks)), starts)
    height = np.full(ny * nx, np.nan)
    height[ks[starts]] = zs[np.minimum(first, len(ks) - 1)]
    grid = HeightGrid(x0, y0, cell, height.reshape(ny, nx))
    rel = z - grid.at(x, y)
    return (rel >= band[0]) & (rel <= band[1]), grid


def fuse_sources(normalized, counts, density_window=41):
    """Per-cell choice between overlapping acquisitions: the file with the highest local point
    density that observed the cell wins (it scanned that lane from the shortest range); the
    other files only fill cells the winner left empty. Returns (normalized, observed, source)."""
    from scipy.ndimage import uniform_filter

    counts = np.asarray(counts)
    seen = counts > 0
    density = np.stack([uniform_filter(c.astype(np.float32), density_window) for c in counts])
    score = np.where(seen, density + 1e-6, -1.0)
    source = score.argmax(0)
    observed = seen.any(0)
    fused = np.take_along_axis(np.asarray(normalized), source[None], 0)[0]
    return (
        np.where(observed, fused, 0).astype(np.float32),
        observed,
        np.where(observed, source, 255).astype(np.uint8),
    )


def rasterize_region(las_paths, grid: Grid, point_margin=2.0, cache_dir=None, surface="low"):
    """Region raster from one or several overlapping source files, plus region points for Z.

    Each file keeps the production contract (per-cell maximum intensity, calibration from that
    file's own whole-file histogram); several files are fused per cell by ``fuse_sources``.
    ``surface``: "low" keeps only points on the lowest dense surface of all files together
    (``surface_band``), "top" the highest one (bridge decks), "all" every point.
    """
    t0 = time.perf_counter()
    paths = [las_paths] if isinstance(las_paths, str | Path) else list(las_paths)
    extracts = [extract_regions(p, [grid.bounds], point_margin, cache_dir)[0] for p in paths]
    minx, miny, maxx, maxy = grid.bounds
    sizes = [len(e["x"]) for e in extracts]
    height = None
    if surface == "all":
        on_surface = [np.ones(n, bool) for n in sizes]
    else:
        keep, height = surface_band(
            np.concatenate([e["x"] for e in extracts]),
            np.concatenate([e["y"] for e in extracts]),
            np.concatenate([e["z"] for e in extracts]),
            top=surface == "top",
        )
        on_surface = np.split(keep, np.cumsum(sizes)[:-1])
    normalized, counts, sources, keep_pts, point_source = [], [], [], [], []
    for k, e in enumerate(extracts):
        sel = on_surface[k]
        x, y = e["x"][sel], e["y"][sel]
        acc = RasterAccumulator((grid.rows, grid.cols))
        inside = (x >= minx) & (x < maxx) & (y > miny) & (y <= maxy)
        if inside.any():
            cols, rows = world_to_pixel(grid.affine, x[inside], y[inside])
            cols = np.floor(np.asarray(cols)).astype(np.int64)
            rows = np.floor(np.asarray(rows)).astype(np.int64)
            ok = (cols >= 0) & (rows >= 0) & (cols < grid.cols) & (rows < grid.rows)
            acc.add(cols[ok], rows[ok], e["intensity"][sel][inside][ok])
        calibration = calibrate_histogram(e["hist"], low=1, high=99.5)
        layers = acc.layers(calibration)
        normalized.append(np.clip(np.nan_to_num(layers["normalized"]), 0, 1))
        counts.append(layers["count"])
        keep_pts.append(np.column_stack([x, y, e["z"][sel]]))
        point_source.append(np.full(len(x), k, np.uint8))
        sources.append(
            dict(
                path=e["path"],
                name=e["name"],
                sha256=e["sha256"],
                calibration=calibration.as_dict(),
                read_report=e["read_report"],
                region_points=int(inside.sum()),
                points_off_surface=int((~sel).sum()),
            )
        )
    fused, observed, source = fuse_sources(normalized, counts)
    for k, src in enumerate(sources):
        src["observed_cells"] = int((counts[k] > 0).sum())
        src["cells_used"] = int((source == k).sum())
    gray = grayscale_intensity(fused, observed)
    points = np.concatenate(keep_pts) if keep_pts else np.zeros((0, 3))
    return dict(
        gray=gray,
        observed=observed,
        count=np.sum(counts, axis=0),
        source=source,
        points=points,
        point_source=np.concatenate(point_source) if point_source else np.zeros(0, np.uint8),
        sources=sources,
        surface=surface,
        height=height,
        calibration=sources[0]["calibration"],
        read_report=sources[0]["read_report"],
        seconds=time.perf_counter() - t0,
    )


__all__ = [
    "CELL",
    "STRIDE",
    "TILE",
    "Grid",
    "HeightGrid",
    "extract_regions",
    "fuse_sources",
    "make_grid",
    "pixel_to_xy",
    "rasterize_region",
    "sha256",
    "surface_band",
    "xy_to_pixel",
]
