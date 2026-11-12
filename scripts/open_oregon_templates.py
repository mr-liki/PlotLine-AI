"""Convert the CC0 Oregon DOT (Philomath) labelled LiDAR into 5 cm marking-template rasters.

The LAS keeps intensity only on marking points; background (class 0) intensity is zero and its
RGB is camera colourisation, so it is NOT used as image evidence. Background points are used
only as the observed-footprint mask. Lane-marking components (class 1) are classified as dash or
solid from their measured length; ambiguous lengths are kept as "unknown" (never guessed).

Run from scripts/ with ``python -I`` (input is downloaded data).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import laspy
import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, find_objects, label

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/raw/open_sources/oregon_philomath_cc0/philomath_road_markings.las"
OUT = ROOT / "data/processed/open_sources/oregon_philomath_templates"
CELL = 0.05
CLASSES = {
    1: "lane_marking",
    2: "crosswalk_text",
    3: "bike",
    4: "arrow_left",
    5: "arrow_right",
    6: "arrow_straight",
}
DASH_RANGE = (1.5, 4.6)  # metres; observed peak checked in the summary
SOLID_MIN = 8.0
MARGIN = 20  # px of context around each template


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    las = laspy.read(SRC)
    x, y = np.asarray(las.x), np.asarray(las.y)
    c = np.asarray(las.classification).astype(np.uint8)
    inten = np.asarray(las.intensity).astype(np.float32)
    x0, y1 = x.min(), y.max()
    col = ((x - x0) / CELL).astype(np.int64)
    row = ((y1 - y) / CELL).astype(np.int64)
    h, w = row.max() + 1, col.max() + 1
    observed = np.zeros((h, w), bool)
    observed[row, col] = True
    m = c > 0
    cls = np.zeros((h, w), np.uint8)
    np.maximum.at(cls, (row[m], col[m]), c[m])  # higher id wins on rare shared cells
    s = np.zeros((h, w), np.float32)
    n = np.zeros((h, w), np.float32)
    np.add.at(s, (row[m], col[m]), inten[m])
    np.add.at(n, (row[m], col[m]), 1)
    paint_int = np.where(n > 0, s / np.maximum(n, 1), 0)
    # Lane components: close 2-cell dropout gaps along the stripe before measuring.
    lane = binary_closing(cls == 1, structure=np.ones((5, 5), bool))
    comps, _k = label(lane, structure=np.ones((3, 3), bool))
    records, lengths = [], []
    for i, sl in enumerate(find_objects(comps), start=1):
        piece = comps[sl] == i
        ys, xs = np.nonzero(piece)
        if len(ys) < 20:
            continue
        pts = np.stack([ys, xs], 1).astype(float) * CELL
        pts -= pts.mean(0)
        _ev, evec = np.linalg.eigh(np.cov(pts.T))
        proj = pts @ evec
        length = float(np.ptp(proj[:, 1]))
        width = float(np.ptp(proj[:, 0]))
        lengths.append(length)
        kind = (
            "dash"
            if DASH_RANGE[0] <= length <= DASH_RANGE[1]
            else "solid" if length >= SOLID_MIN else "unknown"
        )
        records.append(
            dict(
                kind=kind,
                length_m=length,
                width_m=width,
                bbox=[sl[0].start, sl[0].stop, sl[1].start, sl[1].stop],
                comp=i,
            )
        )
    # Other marking classes as whole connected objects.
    for cid, name in CLASSES.items():
        if cid == 1:
            continue
        cc, _kk = label(binary_closing(cls == cid, np.ones((5, 5), bool)), np.ones((3, 3), bool))
        for j, sl in enumerate(find_objects(cc), start=1):
            if (cc[sl] == j).sum() >= 20:
                records.append(
                    dict(kind=name, bbox=[sl[0].start, sl[0].stop, sl[1].start, sl[1].stop])
                )
    np.savez_compressed(
        OUT / "scene_rasters.npz",
        observed=observed,
        cls=cls,
        paint_intensity=paint_int,
        lane_components=comps.astype(np.int32),
        cell_m=CELL,
        origin=[x0, y1],
    )
    # Width of every lane component (minor-axis extent) as a unit sanity check.
    lw = [r["width_m"] for r in records if "width_m" in r and r["kind"] in ("dash", "solid")]
    summary = dict(
        source=str(SRC),
        cell_m=CELL,
        raster_shape=[int(h), int(w)],
        observed_fraction=float(observed.mean()),
        counts={k2: sum(r["kind"] == k2 for r in records) for k2 in {r["kind"] for r in records}},
        lane_length_hist_m=np.histogram(
            lengths, bins=[0, 1, 1.5, 2.5, 3.5, 4.6, 6, 8, 15, 50, 1000]
        )[0].tolist(),
        lane_length_bins=[0, 1, 1.5, 2.5, 3.5, 4.6, 6, 8, 15, 50, 1000],
        dash_length_median_m=(
            float(np.median([r["length_m"] for r in records if r["kind"] == "dash"]))
            if any(r["kind"] == "dash" for r in records)
            else None
        ),
        lane_width_median_m=float(np.median(lw)) if lw else None,
        paint_intensity_pct=np.percentile(paint_int[cls > 0], [5, 25, 50, 75, 95]).tolist(),
        paint_points_per_cell_median=float(np.median(n[n > 0])),
        note="Background intensity absent in source; background usable only as observed footprint.",
    )
    (OUT / "templates.json").write_text(
        json.dumps(dict(summary=summary, templates=records), indent=1)
    )
    # Preview: grey = observed, colours = classes (downsampled 4x).
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[observed] = 60
    palette = {
        1: (255, 255, 255),
        2: (255, 200, 0),
        3: (0, 200, 255),
        4: (255, 0, 0),
        5: (0, 255, 0),
        6: (255, 0, 255),
    }
    for cid, col_ in palette.items():
        rgb[cls == cid] = col_
    for r in records:
        if r["kind"] in ("dash", "solid"):
            r0, r1, c0, c1 = r["bbox"]
            sel = comps[r0:r1, c0:c1] == r["comp"]
            rgb[r0:r1, c0:c1][sel] = (255, 60, 60) if r["kind"] == "solid" else (60, 220, 255)
    Image.fromarray(rgb[::4, ::4]).save(OUT / "preview.png")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.exit(main())
