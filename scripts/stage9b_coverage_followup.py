"""Localize Stage 9B reviewer feedback on the native observed grid."""

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
from scipy.ndimage import convolve, median_filter

from stage9b_annotations import geometry

root = Path(__file__).resolve().parents[1]
base = root / "outputs/reports/stage9b"
run = json.loads((base / "latest.json").read_text())
report = Path(run["report_dir"])
spec = json.loads((report / "review_spec.json").read_text())
refine = "--refine-text" in sys.argv
previous = base / "coverage_followup"
out = base / "text_coverage_refinement" if refine else previous
out.mkdir(exist_ok=True)
source = spec["text_candidates"]
prior = [
    pyogrio.read_dataframe(source, layer=name)
    for name in ("character_candidates", "group_and_unknown_fragments")
]
rois = {
    "N9_01": [
        (0, 0, 64, 46),
        (134, 0, 229, 48),
        (52, 42, 150, 112),
        (0, 117, 75, 181),
        (131, 116, 221, 182),
        (51, 178, 131, 238),
    ],
    "N9_03": [(101, 785, 121, 815), (119, 819, 157, 840), (94, 817, 119, 836)],
}
if refine:
    prior.append(
        pyogrio.read_dataframe(
            previous / "coverage_followup.gpkg", layer="additional_bright_support"
        ).rename(columns={"region": "candidate_id"})
    )
    original_paint = pyogrio.read_dataframe(
        spec["base_stage9_candidates"], layer="candidate_paint_footprints"
    )
    rois = {
        "N9_01": [
            (0, 0, 66, 47),
            (133, 0, 228, 47),
            (55, 43, 146, 109),
            (0, 117, 67, 179),
            (132, 115, 225, 180),
            (59, 182, 151, 248),
        ]
    }
rows = []
audit = {}
for tile in spec["tiles"]:
    region = tile["candidate_id"]
    if region not in rois:
        continue
    with rasterio.open(tile["layers"]["normalized"]) as ds:
        intensity = ds.read(1)
    with rasterio.open(tile["layers"]["observed"]) as ds:
        observed = ds.read(1).astype(bool)
    affine = Affine(*tile["affine"])
    known_shapes = [(g, 1) for frame in prior for g in frame[frame.candidate_id == region].geometry]
    known = rasterize(known_shapes, out_shape=observed.shape, transform=affine).astype(bool)
    if refine:
        line_mask = rasterize(
            [(g, 1) for g in original_paint.geometry],
            out_shape=observed.shape,
            transform=affine,
        ).astype(bool)
        local_background = median_filter(intensity[:250, :235], size=7)
    raw = Image.open(tile["images"]["grayscale"]).convert("RGB")
    before = np.array(raw)
    before[known] = (255, 155, 0)
    after = before.copy()
    audit[region] = []
    for index, (x0, y0, x1, y1) in enumerate(rois[region], 1):
        extent = np.zeros_like(observed)
        extent[y0:y1, x0:x1] = observed[y0:y1, x0:x1]
        cutoff = 0.16 if region == "N9_01" else 0.24
        bright = extent & (intensity > cutoff)
        neighbors = convolve(
            bright.astype("uint8"), np.ones((3, 3), dtype="uint8"), mode="constant"
        )
        added = bright & (neighbors >= 3) & ~known
        if refine:
            faint = extent.copy()
            faint[:250, :235] &= (intensity[:250, :235] >= 0.09) & (
                intensity[:250, :235] - local_background >= 0.045
            )
            horizontal_support = convolve(
                faint.astype("uint8"), np.ones((1, 7), dtype="uint8"), mode="constant"
            )
            extent &= ~line_mask
            added = faint & extent & (horizontal_support >= 3) & ~known
        assert not (added & ~observed).any()
        label = f"{region}_FOLLOWUP_{index:02d}"
        for kind, mask in (
            ("observed_review_extent", extent),
            ("additional_bright_support", added),
        ):
            if not mask.any():
                continue
            polygon = geometry(mask, affine)
            assert np.array_equal(
                mask, rasterize([(polygon, 1)], out_shape=mask.shape, transform=affine).astype(bool)
            )
            rows.append(
                dict(
                    region=region,
                    issue=label,
                    kind=kind,
                    observed_pixels=int(mask.sum()),
                    review_status="UNCONFIRMED_IGNORE",
                    geometry=polygon,
                )
            )
        after[added] = (80, 230, 255)
        audit[region].append(
            dict(
                issue=label,
                native_pixel_box=[x0, y0, x1, y1],
                observed_pixels=int(extent.sum()),
                additional_pixels=int(added.sum()),
            )
        )
    crop = (0, 0, 234, 250) if region == "N9_01" else (75, 765, 175, 855)
    panels = [
        im.crop(crop).resize(
            ((crop[2] - crop[0]) * 2, (crop[3] - crop[1]) * 2), Image.Resampling.NEAREST
        )
        for im in (raw, Image.fromarray(before), Image.fromarray(after))
    ]
    w, h = panels[0].size
    panel = Image.new("RGB", (3 * w, h + 24), "white")
    for i, (im, caption) in enumerate(
        zip(panels, ("Source", "Prior draft", "Cyan: added review support"), strict=True)
    ):
        panel.paste(im, (i * w, 24))
        ImageDraw.Draw(panel).text((i * w + 3, 5), caption, fill="black")
    panel.save(out / f"{region}_coverage_comparison.png")
store = out / "coverage_followup.gpkg"
for kind in ("observed_review_extent", "additional_bright_support"):
    gpd.GeoDataFrame([row for row in rows if row["kind"] == kind]).to_file(store, layer=kind)
(out / "coverage_audit.json").write_text(json.dumps(audit, indent=2))
