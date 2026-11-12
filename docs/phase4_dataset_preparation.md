# Phase 4: annotation alignment and Mask2Former dataset preparation

The current [human-verified annotation workflow](phase4_verified_annotations.md)
starts directly on Phase 3.3 images and preserves this older blocked run.
Installed-processor checks establish that an ignore label does not mask unknown
pixels out of stock mask losses. Partial samples remain review-only pending a
validated loss-validity strategy.

**Outcome (2026-11-04): stopped at the coordinate-alignment gate. No training labels were
generated and the dataset is not ready for Phase 5.** The delivered annotations and the
LiDAR rasters describe places that do not overlap in any coordinate interpretation that
can be justified without inventing a transformation.

Measured results: [phase4_report.md](../outputs/reports/phase4/phase4_report.md) and
[phase4_report.json](../outputs/reports/phase4/phase4_report.json).

## Run

```powershell
micromamba run -n roadmarking-ai python -m pytest --junitxml=outputs/reports/phase4/pytest.xml
micromamba run -n roadmarking-ai python -m roadmarking.healthcheck --json outputs/reports/phase4/healthcheck.json
micromamba run -n roadmarking-ai python scripts/run_phase4.py --config configs/default.yaml
micromamba run -n roadmarking-ai python scripts/run_phase4.py --review-csv path/to/review.csv
```

Exit code 0 = dataset generated, 2 = stopped at a documented gate. `--preflight-only` runs
the earlier availability check. A run takes ~23 s on this machine (CPU only, no LAZ reads).
Outputs go to `data/processed/phase4/run_<hash>/` where the hash covers the annotation
file hashes, Phase 3 manifest hash, taxonomy file, Phase 4 code and options, so a changed
input or setting never reuses stale data. Phase 3 outputs and client files are hashed
before and after the run and must be unchanged.

## What was delivered

Four GeoPackages (Highway and Interchange: longitudinal MultiLineString Z and other-marks
MultiPolygon Z; 16,521 features, all declaring EPSG:3763) and two MicroStation DGN v8
files. The installed GDAL has no DGNv8 driver, so the DGNs are reported as unreadable;
nothing was installed. No Shapefiles were delivered.

## Alignment finding (blocking)

| | X | Y | Z |
|---|---|---|---|
| Annotations (EPSG:3763 declared) | −38.5 … +5.1 km | +166.2 … +177.2 km | 75 … 371 |
| All 10 LAZ headers (no CRS) | −57.5 … −44.3 km | −118.3 … −86.4 km | 13 … 86 |
| Phase 3 rasters (Track01_3, _8) | −56.9 … −48.1 km | −97.8 … −88.4 km | — |

0 of 16,521 features intersect any of the 280 raster tiles; the nearest annotation is
about 255 km (native units) away. Axis swaps and sign flips were evaluated as diagnostics
only and none creates overlap. Elevation ranges also disagree. The corridor shapes differ
(annotations ≈ 43 × 11 km, east-west; LiDAR ≈ 13 × 32 km, north-west to south-east).

Following the rules, no EPSG was assigned to the LiDAR and no shift, rotation, scale or
reprojection was estimated. The displacement method (perpendicular intensity-profile
offsets of centrelines, tolerance 0.05 native units = 1 pixel) is implemented and validated
on synthetic shifts (0, +10, −15 cm), but there were no shared features to measure.

**Needed from the client:** annotations of the surveyed LiDAR area, or LiDAR of the
annotated corridor, or the LiDAR CRS / vertical datum and any documented transformation
between the two datasets.

## Road-surface candidate review (step 3)

Evidence: per-tile 5 cm layers, plus the Phase 2 all-point 1 m grid compared against
the Phase 3 1 m surface state on the identical lattice. Figures are in
`outputs/reports/phase4/road_review/`.

* The candidate is mainly one ~10–12 m strip per track. The edge line falls inside it, but
  lane dashes run along its edge (42 tiles flagged). This strongly suggests that adjacent
  lanes or the shoulder were excluded.
* Flat (z-range < 10 cm), well-sampled rejected 1 m cells within 15 m of the candidate:
  2,954 (Track01_3) and 5,287 (Track01_8). Bright rejected cells within 5 m: 4,435 and 2,234.
* Additional candidate components (12 and 22) contain 5,658 and 14,338 cells. They include
  sparse parallel strips and irregular blobs whose road status is unknown, so they could be
  either missed carriageway or off-road surface.
* About 33 % of candidate pixels are unobserved; 64 tiles are near-empty (<1 % observed).
* Boundaries follow 1 m cells (stair-steps).

Policy implemented: near-empty tiles are excluded from labels. Unobserved pixels and
pixels outside reviewed-complete regions are ignored, not treated as background.
**Recommended correction (not applied):** provide a confirmed road ROI and rerun Phase 3
with `--road-roi`, or revise the geometric selection. Either step must be reviewed before
any labels are trusted. Phase 3 was not rewritten.

## Taxonomy (`configs/phase4_taxonomy.yaml`)

Four separate levels:

1. **Visual classes** (pixel semantics, ids 1–9): longitudinal_solid, longitudinal_dash,
   arrow, detour_arrow, diagonal_stripe, give_way, colour_band, safety_marking, character.
   0 = background, 255 = ignore.
2. **Instance identity**: `global_id = "<file>|<layer>|fid=<fid>"` (ID_SIG restarts at 1 in
   each file, so it cannot be used alone); integer `instance_uid` follows sorted global_id;
   per-tile local ids 1..N are listed in each tile's metadata.
3. **Client codes**: DESIGNACAO text → 1–14; COD_RST kept verbatim.
4. **Context attributes** (POSICAO, ID_GRUPO, TRAÇO, ESPAÇO, LARGURA, COD_JAE, ELEVATION,
   ID_HIST) are metadata only. All source attributes are retained verbatim.

Lines are classified dash/solid by whether TRAÇO is present. The client designation is not
learned from pixels. Open items (flagged, not invented): 'Linha mista' is code 4 or 5;
no Portuguese labels in the data for codes 8, 10, 11, 12; COD_RST visual groupings are
`to_confirm`; 156 polygons have COD_RST 'NULL'.

## Annotation QC

Statuses: VERIFIED, NEEDS_REVIEW, MISALIGNED, INCOMPLETE, UNUSABLE. A feature can be
VERIFIED **only** if the alignment gate passes **and** a human review CSV
(`global_id,status`) marks it so. Valid geometry is never enough. Current result: 0 verified,
16,364 needs-review, 156 incomplete (no class code), 1 unusable (self-intersecting polygon).

Findings to review with the client:

* 3,326 dashes whose length differs from TRAÇO by >25 %. Many 'Linha descontínua'
  features are ~10 m long against TRAÇO 4 / ESPAÇO 10, so dash and gap may be swapped.
* 180 duplicate flags: the Interchange files repeat Highway features. Duplicates are kept
  once (earlier global_id) if labels are generated.
* 259 overlapping polygons (mostly NULL-code and M9a).
* 178 features where vertex Z differs from the ELEVATION attribute by >1 m.
* Edge lines stored as single features up to 40.6 km long.
* All 13,272 centrelines have LARGURA (0.15 / 0.30 m), but this is a documented width,
  not one measured against the imagery.

## Label conversion (implemented, synthetic-validated only)

* Exact Phase 3 affines (corner-based, north-up). Pixel centre = corner + 0.5. Coverage is
  computed with 4× supersampled centre sampling inside each feature's window only.
* Pixels with coverage >0.5 belong to the instance. Partially covered pixels (including
  exactly 0.5) are boundary-ignore. Pixels claimed by two instances are conflict-ignore, so
  instances are never merged.
* Polygon holes are preserved. Centrelines are buffered by LARGURA/2 with flat caps (no
  extension past endpoints) and mitre joins. Each part is buffered separately, so dashes
  are never bridged. Lines without a valid width raise an error rather than getting a
  default.
* Label states: no-data, unknown, verified background (only inside supplied
  reviewed-complete regions), instance, boundary, conflict.
* Per tile: `images/<id>.tif` (normalized intensity with its NoData kept, observed,
  filled_mask; float32, original affine, no CRS, no contrast change);
  `instance_masks/<id>_instance.tif` (uint32), `_semantic.tif` (uint8, 255 ignore),
  `_state.tif`; and `metadata/tiles/<id>.json` with `instance_id_to_semantic_id`, global ids
  and file hashes. This matches the HF Mask2Former image-processor inputs; the 3-channel
  conversion is left to Phase 5.
* Features crossing tiles keep the same uid in every tile. The synthetic self-test confirms
  identical world pixels in the 64-pixel overlap (`outputs/reports/phase4/converter_selftest/`,
  titled SYNTHETIC).

## Spatial splits (implemented, synthetic-validated only)

Labelled tiles are ordered along each corridor (principal axis), grouped into contiguous
blocks, and assigned to train/val/test as contiguous runs by instance weight (target
70/15/15). Guard tiles are then removed if their footprint overlaps another split
(the 256-pixel overlap) or they share a compact instance. Features longer than 25 m (edge
lines) are treated as segmented, and the footprint guard prevents any shared painted pixel.
Validation or test sets with fewer than 50 instances are reported as insufficient rather
than forced. With two separate ~2 km tracks, a credible independent test set needs
annotations on both tracks.

## Tests

`tests/test_phase4_dataset.py` (24 tests) covers:

* world↔pixel round trips with no half-pixel shift
* holes, flat-capped widths and curves
* dashes and parallel lines kept separate, conflicts, boundary ignore
* no silent background, rejection of unverified widths
* cross-tile identity and the GeoTIFF writer (dimensions, transform, NoData,
  filled-band distinction)
* displacement measurement on synthetic shifts
* axis diagnostics and taxonomy levels
* QC statuses (no VERIFIED without the gate), duplicates, unchanged originals and
  retained attributes
* splits: no overlap or instance leakage, long instances, insufficient sets
* three end-to-end runs: aligned → dataset generated with no leakage; 30 cm shifted →
  FAILED, no masks; disjoint frames → BLOCKED

Full suite: 117 passed, 1 skipped (network).
