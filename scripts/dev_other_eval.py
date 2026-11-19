"""Development loop for painted area marks (polygons): extract, fit clean shapes, score.

Usage: python scripts/dev_other_eval.py <regions.json> [--names ...] [--save DIR]

Longitudinal lines come from the latest digitization run of each area (they are only removed
from the paint mask). Shapes and types are learnt from client polygons outside the protected
footprints (+100 m) and outside the area being scored (+200 m).
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
import shapely
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from roadmarking.digitization.other_marks import (  # noqa: E402
    ShapeClassifier,
    descriptor,
    extract_other,
)
from roadmarking.digitization.polygon_fit import ShapeLibrary, iou  # noqa: E402
from roadmarking.digitization.raster_tiles import make_grid  # noqa: E402
from roadmarking.digitization.stages import build_levels  # noqa: E402

CACHE = ROOT / "outputs/digitization/_cache"
ANN = ROOT / "data/raw/annotations"
PAIRS = (
    ("Highway/Other Marks.gpkg", "Highway/Longitudinal Marks .gpkg"),
    ("Interchange/interchange_Other Marks.gpkg", "Interchange/Interchange-Longitudinal Marks.gpkg"),
)


def references(region_box, exclusion):
    """(client polygons in the area [(poly, code)], training examples, library examples)."""
    inside, examples, shapes = [], [], []
    for other_path, line_path in PAIRS:
        frame = pyogrio.read_dataframe(ANN / other_path, columns=["COD_RST", "COD_JAE"])
        train_lines = pyogrio.read_dataframe(ANN / line_path, columns=[]).explode(index_parts=False)
        train_lines = train_lines[train_lines.geometry.notna() & ~train_lines.intersects(exclusion)]
        line_geometries = list(train_lines.geometry)
        tree = shapely.STRtree(line_geometries)
        for row in frame.explode(index_parts=False).itertuples():
            geom = row.geometry
            if geom is None or not geom.is_valid or geom.area <= 0:
                continue
            geom = shapely.force_2d(geom)
            rst = row.COD_RST if pd.notna(row.COD_RST) else None
            jae = row.COD_JAE if pd.notna(row.COD_JAE) else None
            if geom.intersects(region_box):
                inside.append((geom, rst))
            elif rst is not None and not geom.intersects(exclusion):
                nearest = [line_geometries[tree.nearest(geom)]] if line_geometries else []
                examples.append((descriptor(geom, nearest), (rst, jae)))
                shapes.append((geom, rst))
    return inside, examples, shapes


def latest_run(name):
    runs = sorted((ROOT / "outputs/digitization").glob(f"2026*_{name}"))
    runs = [r for r in runs if (r / "qa/qa_summary.json").exists()]
    return runs[-1]


def score(rows, truth, observed_poly):
    truth = [(g, c) for g, c in truth if g.intersection(observed_poly).area >= 0.5 * g.area]
    preds = [
        r
        for r in rows
        if r["geometry"].intersection(observed_poly).area >= 0.5 * r["geometry"].area
    ]
    tree = shapely.STRtree([g for g, _ in truth]) if truth else None
    pairs = []
    for i, r in enumerate(preds):
        if tree is None:
            break
        for j in tree.query(r["geometry"], predicate="intersects"):
            v = iou(r["geometry"], truth[j][0])
            if v >= 0.5:
                pairs.append((v, i, j))
    used_i, used_j, matched = set(), set(), []
    for v, i, j in sorted(pairs, reverse=True):
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        matched.append((v, preds[i], truth[j]))
    per_code = {}
    for _g, c in truth:
        per_code.setdefault(c or "NULL", [0, 0])[1] += 1
    for _, _, (_g, c) in matched:
        per_code[c or "NULL"][0] += 1
    coded = [(p, t) for _, p, t in matched if t[1] is not None]
    return dict(
        truth=len(truth),
        preds=len(preds),
        matched=len(matched),
        recall=len(matched) / len(truth) if truth else None,
        precision=len(matched) / len(preds) if preds else None,
        iou_median=float(np.median([v for v, _, _ in matched])) if matched else None,
        class_accuracy=(
            (sum(p["COD_RST"] == t[1] for p, t in coded) / len(coded)) if coded else None
        ),
        vertices_ours=float(np.median([p["vertices"] for _, p, _ in matched])) if matched else None,
        vertices_client=(
            float(np.median([len(t[0].exterior.coords) - 1 for _, _, t in matched]))
            if matched
            else None
        ),
        per_code={k: f"{a}/{b}" for k, (a, b) in sorted(per_code.items())},
        methods=pd.Series([r["shape_method"] for r in rows]).value_counts().to_dict(),
    )


def render(path, gray, grid, truth, rows, window, size=900):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x0, y0, x1, y1 = grid.bounds
    fig, ax = plt.subplots(figsize=(size / 100, size / 100), dpi=100)
    ax.imshow(gray, cmap="gray", extent=[x0, x1, y0, y1], vmin=0, vmax=255, interpolation="nearest")
    for g, _ in truth:
        x, y = g.exterior.xy
        ax.plot(x, y, color="yellow", lw=3, alpha=0.5)
    for r in rows:
        x, y = r["geometry"].exterior.xy
        color = "lime" if r["shape_method"] == "client_shape" else "deepskyblue"
        ax.plot(x, y, color=color, lw=1.2)
        ax.plot(x, y, ".", color="red", ms=4)
        c = r["geometry"].centroid
        if window[0] < c.x < window[2] and window[1] < c.y < window[3]:
            ax.text(c.x, c.y, r["COD_RST"] or "?", color="orange", fontsize=8)
    ax.set_xlim(window[0], window[2])
    ax.set_ylim(window[1], window[3])
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path)
    plt.close(fig)


def main():
    from digitize import protected_footprints

    ap = argparse.ArgumentParser()
    ap.add_argument("regions")
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--save")
    a = ap.parse_args()
    protected, _ = protected_footprints()
    for region in json.loads(Path(a.regions).read_text()):
        if a.names and region["name"] not in a.names:
            continue
        t = time.perf_counter()
        grid = make_grid(region["bbox"])
        region_box = box(*region["bbox"])
        truth, examples, shapes = references(
            region_box, protected.buffer(100).union(region_box.buffer(200))
        )
        levels, _, _ = build_levels(region["las"], grid, CACHE)
        eng = latest_run(region["name"]) / "geopackage/digitization_engineering.gpkg"
        cl = pyogrio.read_dataframe(eng, layer="centerlines_refined_2d")
        rows = []
        for lv in levels:
            lines = list(cl.geometry[cl["height_level"] == lv["name"]])
            rows += extract_other(
                lv["gray"],
                lv["observed"],
                grid,
                lines,
                lv["surface"],
                ShapeClassifier(examples),
                ShapeLibrary(shapes),
            )
        from rasterio.features import shapes as rshapes
        from scipy.ndimage import binary_dilation
        from shapely.geometry import shape

        obs = np.any([binary_dilation(lv["observed"], iterations=2) for lv in levels], axis=0)
        observed_poly = shapely.union_all(
            [shape(g) for g, v in rshapes(obs.astype(np.uint8), mask=obs, transform=grid.affine)]
        )
        sc = score(rows, truth, observed_poly)
        print(region["name"], json.dumps(sc), f"{time.perf_counter() - t:.0f}s", flush=True)
        if a.save:
            out = Path(a.save)
            out.mkdir(parents=True, exist_ok=True)
            gray = np.maximum.reduce([lv["gray"] for lv in levels])
            with (out / f"{region['name']}_other.pkl").open("wb") as fh:
                pickle.dump(dict(rows=rows, truth=truth, score=sc), fh)
            for k, row in enumerate(r for r in rows if r["shape_method"] == "outline_mark"):
                x0, y0, x1, y1 = row["geometry"].bounds
                render(
                    out / f"{region['name']}_outline{k}.png",
                    gray,
                    grid,
                    truth,
                    rows,
                    (x0 - 1, y0 - 1, x1 + 1, y1 + 1),
                )
            seen = [g for g, _ in truth if g.intersection(observed_poly).area >= 0.5 * g.area]
            for k, g in enumerate(seen[:: max(1, len(seen) // 6)][:6]):
                c = g.centroid
                render(
                    out / f"{region['name']}_mark{k}.png",
                    gray,
                    grid,
                    truth,
                    rows,
                    (c.x - 6, c.y - 6, c.x + 6, c.y + 6),
                )


if __name__ == "__main__":
    main()
