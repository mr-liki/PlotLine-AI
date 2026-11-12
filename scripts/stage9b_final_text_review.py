"""Build the focused N9_01 review delta without editing earlier geometry."""

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import rasterio
from affine import Affine
from PIL import Image, ImageDraw
from rasterio.features import rasterize
from scipy.ndimage import convolve

from stage9b_annotations import geometry

base = Path.cwd() / "outputs/reports/stage9b"
run = json.loads((base / "latest.json").read_text())
spec = json.loads((Path(run["report_dir"]) / "review_spec.json").read_text())
tile = next(tile for tile in spec["tiles"] if tile["candidate_id"] == "N9_01")
residual = "--lowest-group" in sys.argv
previous = base / "final_text_review"
bottom = "--bottom-strokes" in sys.argv
gaps = "--stroke-gaps" in sys.argv
out = base / "lowest_group_review" if residual else previous
if bottom:
    out = base / ("stroke_gap_review" if gaps else "bottom_stroke_review")
out.mkdir(exist_ok=True)
width, height = 234, 254
crop = np.s_[:height, :width]
affine = Affine(*tile["affine"])


def read(channel):
    with rasterio.open(tile["layers"][channel]) as dataset:
        return dataset.read(1, window=((0, height), (0, width)))


def mask(shapes):
    return rasterize([(g, 1) for g in shapes], out_shape=(height, width), transform=affine).astype(
        bool
    )


observed = read("observed").astype(bool)
intensity = read("normalized")
frames = [
    pyogrio.read_dataframe(spec["text_candidates"], layer=layer, where="candidate_id='N9_01'")
    for layer in ("character_candidates", "group_and_unknown_fragments")
]
for folder in ("coverage_followup", "text_coverage_refinement"):
    frames.append(
        pyogrio.read_dataframe(
            base / folder / "coverage_followup.gpkg",
            layer="additional_bright_support",
            where="region='N9_01'",
        )
    )
if residual:
    frames.append(
        pyogrio.read_dataframe(previous / "coverage_followup.gpkg", layer="stroke_candidates")
    )
if bottom:
    frames.append(
        pyogrio.read_dataframe(
            base / "lowest_group_review" / "coverage_followup.gpkg", layer="stroke_candidates"
        )
    )
if gaps:
    frames.append(
        pyogrio.read_dataframe(
            base / "bottom_stroke_review" / "coverage_followup.gpkg", layer="stroke_candidates"
        )
    )
prior = mask(g for frame in frames for g in frame.geometry)
excluded = mask(
    g
    for layer in ("candidate_paint_footprints", "unsupported_marking_regions")
    for g in pyogrio.read_dataframe(spec["base_stage9_candidates"], layer=layer).geometry
)
# Compare each pixel to observed road samples above and below the same stroke.
flanks = np.ones((9, 5), dtype="float32")
flanks[2:7] = 0
context = observed & ~prior & ~excluded
samples = convolve(context.astype("float32"), flanks, mode="constant")
background = convolve(np.where(context, intensity, 0), flanks, mode="constant")
background /= np.maximum(samples, 1)
contrast = intensity - background
bright = observed & ~excluded & (samples >= 5) & (contrast >= 0.027) & (intensity >= 0.055)
support = convolve((bright | prior).astype("uint8"), np.ones((3, 7)), mode="constant")
boxes = [
    (0, 0, 66, 47),
    (133, 0, 228, 47),
    (55, 43, 146, 109),
    (0, 117, 67, 179),
    (132, 115, 225, 180),
    (59, 182, 151, 250),
]
rows = []
delta = np.zeros_like(observed)
for i, (x0, y0, x1, y1) in enumerate(boxes, 1):
    extent = np.zeros_like(observed)
    extent[y0:y1, x0:x1] = True
    unresolved = extent & observed & ~excluded & ~prior
    addition = unresolved & bright & (support >= 7)
    if residual:
        if i != 6:
            continue
        bands = np.zeros_like(observed)
        bands[214:219, 102:146] = True
        bands[208:212, 126:146] = True
        if bottom:
            bands[:] = False
            bands[235:240, 65:145] = True
            bands[244:248, 65:145] = True
        if gaps:
            bands[:] = False
            bands[222:232, 65:145] = True
            bands[239:245, 65:145] = True
            bands[247:250, 65:145] = True
        stroke = observed & (intensity >= 0.105) & (contrast >= 0.018)
        aligned = convolve(stroke.astype("uint8"), np.ones((3, 7)), mode="constant")
        addition = unresolved & bands & stroke & (aligned >= 4)
    delta |= addition
    for kind, pixels in (("stroke_candidates", addition), ("unresolved_extent", unresolved)):
        polygon = geometry(pixels, affine)
        assert np.array_equal(pixels, mask([polygon]))
        rows.append(
            dict(
                kind=kind,
                text_group_id=f"N9_01_TEXT_G{i:02d}",
                character_id=None,
                character_value=None,
                observed_pixels=int(pixels.sum()),
                supervision="IGNORE_PENDING_REVIEW",
                geometry=polygon,
            )
        )
assert not (delta & (~observed | prior | excluded)).any()
store = out / "coverage_followup.gpkg"
for kind in ("stroke_candidates", "unresolved_extent"):
    frame = gpd.GeoDataFrame([row for row in rows if row["kind"] == kind])
    frame.to_file(store, layer=kind)
raw = np.array(Image.open(tile["images"]["grayscale"]).convert("RGB"))[crop]
before = raw.copy()
before[prior] = (255, 165, 0)
lines = pyogrio.read_dataframe(
    spec["base_stage9_candidates"],
    layer="candidate_paint_footprints",
    where="candidate_id = 'N9_01'",
)
for visual_id, color in ((1, (255, 65, 65)), (2, (70, 145, 255))):
    line_pixels = mask(lines[lines.visual_id == visual_id].geometry)
    before[line_pixels] = color
after = before.copy()
after[delta] = (115, 255, 105)
panel = Image.new("RGB", (width * 6, height * 2 + 22), "white")
for i, (pixels, title) in enumerate(
    (
        (raw, "Native source"),
        (before, "Prior coverage"),
        (after, "Green: text additions | Blue: existing dashes"),
    )
):
    panel.paste(Image.fromarray(pixels).resize((width * 2, height * 2), 0), (i * width * 2, 22))
    ImageDraw.Draw(panel).text((i * width * 2 + 4, 4), title, fill="black")
panel.save(out / "N9_01_final_comparison.png")
(out / "coverage_audit.json").write_text(
    json.dumps([{k: v for k, v in row.items() if k != "geometry"} for row in rows], indent=2)
)
