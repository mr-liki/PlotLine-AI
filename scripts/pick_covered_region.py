"""Pick a review box inside a LAS file that is actually scanned and has client lines.

Usage: python scripts/pick_covered_region.py <file.laz> <name> [--prefer curved|lines]
Prints a JSON job for digitize.py. Boxes overlapping protected footprints (+100 m) are skipped.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from digitize import DASH, SOLID, protected_footprints  # noqa: E402
from roadmarking.io.pointcloud import ChunkReadReport, iter_laz_chunks  # noqa: E402

SIZE, CELL = 102.4, 5.0


def main(las, name, prefer="curved"):
    las = Path(las)
    xs, ys = [], []
    for ch in iter_laz_chunks(las, ChunkReadReport(las, 0, 0)):
        xs.append(np.asarray(ch.x)[::20])
        ys.append(np.asarray(ch.y)[::20])
    x, y = np.concatenate(xs), np.concatenate(ys)
    x0, y0 = x.min(), y.min()
    gi, gj = ((x - x0) // CELL).astype(int), ((y - y0) // CELL).astype(int)
    grid = np.zeros((gi.max() + 1, gj.max() + 1), int)
    np.add.at(grid, (gi, gj), 1)
    scanned = grid >= 20

    def coverage(b):
        i0, i1 = int((b[0] - x0) // CELL), int((b[2] - x0) // CELL)
        j0, j1 = int((b[1] - y0) // CELL), int((b[3] - y0) // CELL)
        sub = scanned[max(i0, 0) : max(i1, 0), max(j0, 0) : max(j1, 0)]
        return float(sub.mean()) if sub.size else 0.0

    protected, _ = protected_footprints()
    keep_out = protected.buffer(100)
    ann = ROOT / "data/raw/annotations"
    lines = pd.concat(
        [
            pyogrio.read_dataframe(ann / "Highway/Longitudinal Marks .gpkg"),
            pyogrio.read_dataframe(ann / "Interchange/Interchange-Longitudinal Marks.gpkg"),
        ]
    )
    lines = lines[lines.geometry.notna() & lines["DESIGNACAO"].isin(SOLID | DASH)]
    fb = box(x.min(), y.min(), x.max(), y.max()).buffer(-SIZE / 2)
    best = None
    for g in lines.geometry[lines.intersects(fb)]:
        for part in g.geoms:
            for d in np.arange(0, part.length, 25.0):
                p = part.interpolate(d)
                b = (p.x - SIZE / 2, p.y - SIZE / 2, p.x + SIZE / 2, p.y + SIZE / 2)
                if not fb.contains(p) or box(*b).intersects(keep_out):
                    continue
                cov = coverage(b)
                if cov < 0.25:
                    continue
                inb = lines[lines.intersects(box(*b))].geometry.intersection(box(*b))
                straight = [
                    np.linalg.norm(np.asarray(pp.coords)[-1, :2] - np.asarray(pp.coords)[0, :2])
                    / pp.length
                    for gg in inb
                    for pp in getattr(gg, "geoms", [gg])
                    if pp.length > 30
                ]
                curv = 1 - float(np.median(straight)) if straight else 0.0
                score = (
                    (curv if prefer == "curved" else 0) * 100
                    + cov
                    + 0.001 * float(inb.length.sum())
                )
                if best is None or score > best[0]:
                    best = (score, b, cov, curv, len(inb))
    if best is None:
        raise SystemExit("No scanned, unprotected box with client lines found")
    _, b, cov, curv, n = best
    job = dict(
        las=str(las),
        bbox=list(b),
        name=name,
        scanned_cell_fraction=round(cov, 2),
        median_line_curvature=round(curv, 4),
        client_lines=n,
    )
    print(json.dumps(job))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "curved")
