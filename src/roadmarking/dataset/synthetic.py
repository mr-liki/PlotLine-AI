"""Synthetic converter fixtures (tests and the labelled converter self-test only).

Nothing here is client data. Coordinates are an arbitrary local frame; output figures
are explicitly titled SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from affine import Affine
from rasterio import features as rio_features
from shapely.geometry import LineString, MultiLineString, Polygon

from roadmarking.dataset.labels import LabelFeature

RES = 0.05


def make_tile(tile_id: str, row_off: int, col_off: int, size: int = 256, x0=1000.0, y0=2000.0):
    """Tile on a shared lattice anchored at (x0, y0) = top-left corner of row/col 0."""
    c = x0 + col_off * RES
    f = y0 - row_off * RES
    return {
        "tile_id": tile_id,
        "source_file": "synthetic.laz",
        "row_offset": row_off,
        "col_offset": col_off,
        "height": size,
        "width": size,
        "resolution": RES,
        "affine": [RES, 0.0, c, 0.0, -RES, f],
        "native_xy_bbox": [c, f - size * RES, c + size * RES, f],
        "crs_status": "CRS_UNCONFIRMED",
    }


def arrow(x, y, length=5.0, shaft=0.3, head=1.0, head_len=1.5) -> Polygon:
    s = shaft / 2
    return Polygon(
        [
            (x, y - s),
            (x + length - head_len, y - s),
            (x + length - head_len, y - head / 2),
            (x + length, y),
            (x + length - head_len, y + head / 2),
            (x + length - head_len, y + s),
            (x, y + s),
        ]
    )


def features(x0=1000.0, y0=2000.0) -> list[LabelFeature]:
    """Narrow solid line, three dashes, a curve, an arrow, a holed polygon, a tile-crosser."""
    out = []
    uid = 0

    def add(kind, geom, visual, width=None, group=None):
        nonlocal uid
        uid += 1
        out.append(LabelFeature(uid, f"synthetic|fid={uid}", visual, kind, geom, width, group))

    add("line", LineString([(x0 + 0.5, y0 - 2.0, 10.0), (x0 + 12.0, y0 - 2.0, 10.0)]), 1, 0.15)
    for i in range(3):
        a = x0 + 0.5 + i * 3.0
        add("line", LineString([(a, y0 - 3.5), (a + 2.0, y0 - 3.5)]), 2, 0.15, group="dash-group")
    t = np.linspace(0, np.pi / 2, 40)
    add("line", LineString(np.c_[x0 + 6 + 4 * np.cos(t), y0 - 12 + 4 * np.sin(t)]), 1, 0.30)
    add("polygon", arrow(x0 + 1.0, y0 - 6.0), 3)
    outer = [(x0 + 1, y0 - 9), (x0 + 5, y0 - 9), (x0 + 5, y0 - 11.5), (x0 + 1, y0 - 11.5)]
    hole = [(x0 + 2, y0 - 9.8), (x0 + 4, y0 - 9.8), (x0 + 4, y0 - 10.7), (x0 + 2, y0 - 10.7)]
    add("polygon", Polygon(outer, [hole]), 5)
    # crosses the boundary between two horizontally adjacent tiles
    add(
        "polygon",
        Polygon([(x0 + 10, y0 - 7), (x0 + 14, y0 - 7), (x0 + 14, y0 - 8), (x0 + 10, y0 - 8)]),
        7,
    )
    add("line", MultiLineString([[(x0 + 0.5, y0 - 4.5), (x0 + 3, y0 - 4.5)]]), 2, 0.15)
    return out


def paint_raster(tile: dict, feats: list[LabelFeature], shift=(0.0, 0.0), seed=0, missing=0.2):
    """Intensity image with paint at feature locations (optionally shifted) and holes."""
    a = Affine(*tile["affine"])
    shape = (tile["height"], tile["width"])
    shapes = []
    for f in feats:
        g = f.painted_area()
        from shapely import affinity

        shapes.append((affinity.translate(g, *shift), 1))
    paint = rio_features.rasterize(shapes, out_shape=shape, transform=a, fill=0, dtype="uint8")
    rng = np.random.default_rng(seed)
    image = 0.15 + 0.05 * rng.standard_normal(shape) + 0.7 * paint
    observed = (rng.random(shape) >= missing).astype(np.uint8)
    image = np.where(observed > 0, image, -9999.0).astype(np.float32)
    return image, observed
