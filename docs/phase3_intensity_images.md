# Phase 3: clean top-down LiDAR intensity images

The primary output is now a raster-only grayscale image. Reference lines,
corridor polygons, stripe proposals and diagnostic figures remain separate review
products. None is drawn into the final PNGs.

## Run and defaults

```powershell
$mm = "$env:USERPROFILE/micromamba/micromamba.exe"
# Nine representative tiles from both intact tracks:
& $mm run -n roadmarking-ai python scripts/run_phase3.py
# Dedicated entry point / explicit selection:
& $mm run -n roadmarking-ai python scripts/run_phase3_intensity_images.py
& $mm run -n roadmarking-ai python scripts/run_phase3_intensity_images.py --sources Track01_3
& $mm run -n roadmarking-ai python scripts/run_phase3_intensity_images.py --tile-id Job_20260325_0838_Track01_8__r0029952_c0018432
# One source-grid image tile:
& $mm run -n roadmarking-ai python scripts/run_phase3.py --smoke
# Historical geometric extraction / diagnostic plots:
& $mm run -n roadmarking-ai python scripts/run_phase3.py --output-mode legacy --smoke --previews
```

`--output-mode intensity` is the default. The old geometry/ROI/fill workflow
requires `--output-mode legacy`. Clean mode always writes PNGs; `--previews` is
unnecessary. It rejects geometry ROI/fill flags instead of silently clipping or
inventing observations. Legacy processing remains available and unchanged in
behavior; its old multi-panel previews are diagnostics.

The pilot includes straight/overlap, boundary markings, surrounding crossing
context, an ambiguous context region and a curved section. Only Track01_3 and
Track01_8 source identities are accepted. `--bounds` selects intersecting existing
cached pilot tiles without moving their affines; `--tile-limit` bounds development
runs. Context outside verified cache coverage is refused, not called NoData.
No full-survey processing is implicit.

## Two visual products

**Road-focused:** original 1024 × 1024 tile, 0.05 native-unit pixels, 51.2 × 51.2
native-unit field. Original 768 stride / 256 overlap is retained. All validated
observations inside the window can contribute; no strict road mask suppresses
intensity. Nearby useful context remains visible.

**Wider context:** 2048 × 2048 original-observation window over 102.4 × 102.4 native
units at the same resolution. It is not an enlarged/blurred tile. Its origin is
exactly 512 pixels before the original tile. Two larger section mosaics stitch
neighboring contexts from the same source.

Distances are provisionally metric (5 cm pixels); CRS, units and datum remain
unconfirmed. Native X-right/Y-up alignment is preserved; no road-axis rotation or
geodetic north direction is inferred.

## Output layout

```text
data/processed/phase3_intensity_images/<run_id>/
  rasters/<tile>/
    raw_max_intensity.tif
    raw_mean_intensity.tif
    normalized_intensity.tif
    observed_mask.tif
    point_count.tif
  rasters/context/<tile>/          # same five layers, 2048²
  rasters/mosaics/<section>/       # same five exact stitched layers
  tiles/<tile>/
    intensity_model_input.png     # L grayscale, 1024²
    intensity_model_input_rgb.png # RGB, three identical intensity channels
  previews/<tile>/
    intensity_preview.png         # clean image only
    intensity_context_preview.png # wider observations, clean image only
  mosaics/<section>/
    intensity_context_preview.png
  metadata/
    manifest.json
    identity.json
    completion.json
```

PNG georeferencing is separate: `.pgw` pixel-center worldfile and `.json` affine,
shape, gamma and missing-data sidecar. PNG pixels contain no axes, labels, legends,
outlines, colors or vectors. RGB means replicated intensity, not false color.

| GeoTIFF layer | Type | Missingness |
|---|---|---|
| raw maximum | int32, original uint16 range | -1 absent |
| raw mean | float32, original intensity units | -9999 absent |
| normalized intensity | float32 [0,1] | -9999 absent |
| observed mask | uint8 | 1 measured / 0 absent |
| point count | uint32 | 0 absent |

Raw data are never gamma-corrected, filtered or gap-filled. PNG missing pixels are
black (0), but a genuinely weak measured return can also be black. Therefore
**observed_mask.tif is authoritative** when distinguishing missing and dark data.

## Shared normalization and contrast

The existing full-file Phase 3 calibration is reused separately for each source:
p1/p99.5 of geometry-candidate points across the whole file, not tile histograms.
Its population, raw percentiles and source hash are retained in the manifest.

```text
normalized = clip((raw_max - file_low) / max(file_high - file_low, 1024), 0, 1)
PNG observed pixel = round(255 * normalized**0.7)
PNG missing pixel  = 0
```

The one fixed monotone gamma exposes darker asphalt while retaining strong paint
contrast. It is identical across tiles, contexts, RGB copies and mosaics. No
per-tile histogram adjustment or local enhancement is used. The dedicated CLI's
`--gamma` creates an explicitly configured, differently fingerprinted run; raw
and linear data remain unchanged.

Bright objects are not automatically paint. Barriers, trees, scan sampling and
other measured context can remain visible. No smoothing, denoising, vector
rasterization, filling or paint interpolation occurs. Observed solid lines stay
continuous where supported; dashes and real sampling gaps are never drawn together.

## Mosaics and verification

Mosaic pixels are copied once on the shared lattice. Every overlapping raw max,
raw mean, normalized intensity, mask and count value must match exactly. There
is no averaging, summation, feathering or resampling. Unprocessed holes cannot
be claimed as observed coverage. Independently calibrated source files are not
mixed in a mosaic.

- `rasterization/rasterizer.py`: aligned accumulation of all window observations.
- `preprocessing/intensity.py`: existing whole-file calibration, reused.
- `visualization/intensity_render.py`: numerical arrays directly to L/RGB PNG;
  no vector, drawing, plotting or text API.
- `visualization/mosaic.py`: exact-lattice stitching and seam rejection.
- `visualization/intensity_pipeline.py`: source/cache integrity, clean products,
  sidecars, readback, overlap checks and versioned completion/report.
- `scripts/run_phase3.py`: clean output default, explicit legacy mode.

Every PNG is decoded and compared with the exact transfer equation. RGB channels
must be identical. GeoTIFF arrays/affines read back exactly and CRS stays unset.
Completed artifacts are hash-verified before reuse; corrupt or incomplete runs
are not overwritten silently.

Tests cover grayscale/monotone transfer, no overlay path, NoData, RGB replication,
shared calibration, unchanged raw statistics, source-grid worldfiles, exact
overlap, mismatched grids/seams, unprocessed holes, source-file immutability and
end-to-end TIFF/PNG/cache export. Full pytest, healthcheck, Ruff and Black are run.

## Debug separation and acceptance scope

Required A–G reports and validation are under
`outputs/reports/phase3_intensity_images/`. Its `debug/` folder links prior maps
and review packages; they are not final image products. Old raw intensity TIFFs
never contained those diagnostic vectors. The new default previews are clean
PNG arrays, not screenshots or plots of GIS layers.

Clean-image export can pass while training remains blocked. Images have the
requested future model-input format; matching, reviewed annotations/labels are
still required before Mask2Former training. No road-boundary or marking accuracy
is asserted by grayscale rendering. No model architecture change, weights
download, training, final marking vectors, client-file changes or Git commit.
