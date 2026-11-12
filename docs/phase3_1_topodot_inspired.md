# Phase 3.1: independent alignment and cross-section pilot

This replaces the destructive *architecture* for the new experiment only. Phase 3,
the earlier Phase 3.1 experiment and Phase 4 outputs remain read-only. The pilot
status is provisional until an attributed human review establishes road identity,
boundary interpretation and observed paint retention. No training is performed.

## Public workflow basis

Reviewed November 4, 2026:

| Public reference | Principle used | Independent implementation here |
|---|---|---|
| [Infrastructure technical note, pages 7 and 9](https://www.certainty3d.com/pdf/technotes/TopoDOT_Infrastructure.pdf) | Paint stripes / imported alignment provide a reference for station cross-sections | Local-contrast hypotheses or native-XY supplied polylines; perpendicular original-point profiles |
| [Product capabilities](https://www.topodot.com/solutions/topodot) | Review and operator control accompany extraction | Explicit provisional/approved reference states and manually verified sample gate |
| [2022 release highlights, Extraction by Intensity](https://blog.topodot.com/1039-topodot-version-2022-0-release-highlights/) | Direction, turn handling and continuous line extraction matter | Local PCA fragments; compatible endpoint graph with bounded, observed gaps |
| [Feature-specific extraction discussion](https://blog.topodot.com/1039-is-the-requirement-for-point-cloud-classification-dead/) | Different structures need different extraction rules | Paint proposal, section geometry and raster evidence have separate responsibilities |

Only public workflow principles were adopted. These are our own algorithms; they
are not TopoDOT algorithms, a reproduction of TopoDOT, or evidence of comparable
accuracy/performance. No Naksha PlotLine plugin code is used by the new detector.
The acquisition platform of this dataset remains unknown.

## Two spatial representations

1. **Marking search**: a permissive, unverified observation envelope. It combines
   profile support with an adaptive one-native-unit observation footprint in each
   bounded pilot review window. This explicit context fallback preserves evidence
   when a stripe stops, an alignment is ambiguous, or the strict profile fails.
   It includes possible non-road objects and is **not** verified pavement.
2. **Strict surface candidate**: locally supported profile segments containing a
   provisional/supplied reference, constrained by adjacent-station width change.
   Alternatives at disconnected levels stay uncertain. Smooth edges are not
   asserted as surveyed road boundaries. This layer is provisional geometry.

`confirmed_road` remains zero in automatic processing. The strict geometric mask
does not filter the intensity accumulator. Thus an original marking can survive
in the model-input candidate raster even where road classification is uncertain.
The candidate raster still needs review before it is used for model training.

## Components

- `preprocessing/stripe_reference.py`: all-observation local mean/variance
  contrast, original bright-cell support, short-window PCA elongation/width tests,
  endpoint direction and observed-gap checks. Paths and parallel competitors are
  retained. A supported path means sufficient linear evidence, not verified paint.
- `preprocessing/corridor_discovery.py`: reference-guided, intensity-assisted and
  geometry-assisted modes. Geometry PCA is a review artifact, never a reliable
  roadway assertion. Neither density nor lowest elevation selects a road.
- `preprocessing/road_cross_sections.py`: polyline station coordinates/tangents;
  bounded-window KD-tree over original XYZ; perpendicular point queries; adaptive
  counts, medians/MAD, local slope and normal changes, possible grade/height breaks.
  Profile residuals are recorded. Point source identities remain in the cache.
- `preprocessing/corridor_boundaries.py`: longitudinal station tracking with a
  width-change bound; strict segment containing the reference; disconnected
  alternatives, uncertain edges, mixed/rough rejection and insufficient support.
  Large jumps are exposed as uncertainty rather than hidden by smoothing.
- `rasterization/rasterizer.py`: independent search-corridor accumulator; raw
  max/mean/count, shared Phase 3 normalization, observed/filled provenance and
  strict/confirmed/uncertain/search layers. No filling or artificial paint.
- `quality/marking_retention.py`: observation/proxy retention and explicit
  attributed human-reference validation. Proxy brightness is never marking recall.
- `analysis/corridor_pilot.py`: strict source streaming, deterministic spatial
  partitions, version identity, pilot execution, exact TIFF readback and pairwise
  overlap checks, source hashes, A–G views, profiles and the A–M report.

Configuration is `configs/phase3_1_topodot_inspired.yaml`. Distances named `_m`
are provisional native units, **not** a confirmed metric CRS. Initial settings:
1-unit stations, ±30-unit search, 0.25-unit profile bins, minimum three points with
density adaptation, 1–4 cm equivalent noise bounds and local slope/step limits.
They are engineering starting settings, not universal road-width rules. They do
not impose a fixed number of lanes. Inspect recorded noise, alternative segments
and width distributions before changing them.

## Controlled modes

Run inside the existing environment; no installation is needed:

```powershell
$mm = "$env:USERPROFILE/micromamba/micromamba.exe"
& $mm run -n roadmarking-ai python scripts/run_phase3_1_topodot_inspired.py --mode diagnose
& $mm run -n roadmarking-ai python scripts/run_phase3_1_topodot_inspired.py --mode propose
& $mm run -n roadmarking-ai python scripts/run_phase3_1_topodot_inspired.py --mode propose --discovery-mode geometry
& $mm run -n roadmarking-ai python scripts/run_phase3_1_topodot_inspired.py --mode guided --reference alignment.geojson
& $mm run -n roadmarking-ai python scripts/run_phase3_1_topodot_inspired.py --mode validated --reference approved.geojson --manual-review reviewed_samples.geojson
& $mm run -n roadmarking-ai python scripts/run_phase3_1_topodot_inspired.py --mode compare
```

- **Diagnose** reads old models and records first-failing criteria. It changes no
  earlier result. Original-point diagnostic evidence is linked in its report.
- **Propose** audits all 42 historical boundary flags plus representative cases
  from both intact tracks. It creates proposed references, profiles and rasters.
- **Guided** uses supplied native-XY reference polylines (including a polyline
  drawn through operator seed locations); unapproved references remain provisional.
  `--seed-points seeds.geojson` also accepts native-XY Point/MultiPoint hints with
  the same source-hash/frame declaration. Seed-only geometry proposals remain
  review-only: a point does not establish an accepted alignment direction.
- **Validated** requires attributed approved references and manually verified
  paint/boundary samples. It does not automatically promote every geometric
  candidate to confirmed road or assert full-survey validation.
- **Compare** rebuilds the latest report from persisted metrics without new LAZ
  decoding. `--tile-limit` is for a bounded development pilot only.

The CLI deliberately runs pilot windows. There is no implicit full-survey mode.
Automatic technical checks alone do not satisfy the full-processing gate.
If an alignment is unreliable, its overlay and uncertain context are usable for
review, but discovery is not declared successful. Intersections and multiple
parallel references are represented as alternatives, not a verified road network.

## Native-XY guidance and manual review

The input overlay is a JSON FeatureCollection with **no declared CRS**:

```json
{
  "type": "FeatureCollection",
  "coordinate_frame": "NATIVE_XY",
  "source_sha256": "EXACT_LAZ_SHA256",
  "features": [{
    "type": "Feature",
    "geometry": {"type": "LineString", "coordinates": [[-56000,-89000],[-55980,-89020]]},
    "properties": {"review_status": "PROPOSED_UNREVIEWED"}
  }]
}
```

Use the exact source hash and real coordinates from the overlays, not the example
coordinates above. Invalid/non-overlapping lines, CRS declarations and source-hash
mismatches are refused. For approval, a human adds `review_status: APPROVED`,
`reviewer`, and `reviewed_at`. Approval is not manufactured by the program.

For the pilot review, open the original maximum raster, new normalized raster,
old mask and native-XY overlays in QGIS without assigning a guessed CRS or
reprojecting coordinates. Native JSON/GeoJSON files have a custom coordinate-frame
declaration; software assuming RFC 7946 WGS84 must not transform these coordinates.
They may be saved to a CRS-less GeoPackage for editing.

Draw visible paint strokes as LineStrings or small polygons and pavement extents
as polygons. Each manually checked feature needs:

```json
{
  "kind": "paint",
  "review_status": "MANUALLY_VERIFIED",
  "reviewer": "ACTUAL_REVIEWER",
  "reviewed_at": "ACTUAL_REVIEW_TIME"
}
```

Use `kind: road_boundary` for pavement polygons. Keep the top-level source hash
and `coordinate_frame: NATIVE_XY`. Both paint and boundary samples are required
by the manual QA function; proposed features and empty/non-overlapping paint
samples are refused. Retention is measured only on paint locations with original
observations. A nonexistent return cannot become a missed or invented marking.
The code checks attributed review declarations; it cannot authenticate a person.
There is no human review in the automatic pilot, and no verified marking recall.

The saved-output audit creates one CRS-less review GeoPackage per source, with
reference/search/strict layers and empty `paint_review` / `road_boundary_review`
edit layers. Add `reviewer` / `reviewed_at` fields when approving reference rows.
Export the attributed edits without inventing review status:

```powershell
& $mm run -n roadmarking-ai python scripts/export_corridor_review.py SOURCE_review.gpkg --source Job_20260325_0838_Track01_3 --kind reference --output approved.geojson
& $mm run -n roadmarking-ai python scripts/export_corridor_review.py SOURCE_review.gpkg --source Job_20260325_0838_Track01_3 --kind samples --output reviewed_samples.geojson
```

The exporter refuses CRS assignment and existing export targets. It only exports
already attributed `APPROVED` references or `MANUALLY_VERIFIED` samples; it cannot
approve automatically proposed geometry. A guided input is scoped to its matching
source hash and intersecting pilot windows. `--source` can additionally restrict
propose/diagnose runs to one intact source.

## Outputs and performance scope

Versioned products are under
`data/processed/phase3_1_topodot_inspired/<run_id>/` in the nine requested folders.
Shared immutable point partitions live in `point_cache/<fingerprint>/`; each point
has original XYZ, raw uint16 intensity and its source record index. Points are
stored once per cache, not once per cross-section. A SHA-verified cache allows
geometry experiments without rereading LAZ. Incomplete caches are not appended
over; completed artifacts are hash-checked before reuse.

Reports are under `outputs/reports/phase3_1_topodot_inspired/<run_id>/`.
`latest.json` identifies the latest completed pilot. Per-tile JSON gives numeric
evidence; A–G views share one extent/scale, and difficult profiles have elevation
plots. The complete manifest records the original affine and normalization.

Every tile is 1024² at 0.05 native-unit pixels on the original 768-stride grid.
All overlapping intensity, provenance and classification pixels are compared
exactly. CRS stays unset. Raw/normalized NoData is explicit; no gap filling occurs.
When multiple elevation levels share a pixel, maximum/mean intensity represents
all search observations, not a pavement-height-filtered statistic. Elevated-object
contamination is therefore a reported review concern.

The pilot streams both entire intact files once to obtain strict integrity and
spatial partitions, but rasterizes only selected windows. Per-stage timing and
sampled RSS are reported. The historical ≈315-second full Phase 3 baseline is a
different scope; no speedup is inferred. Overlapping tile sums are explicitly
distinguished from unique survey counts/areas.

## Acceptance and remaining limits

Technical acceptance requires exact source hashes, raw intensity/coordinate
preservation, exact TIFF readback and exact overlap consistency. Reviewed marking
retention and false-positive assessment are separate gates. The wide discovery
context can include roofs, barriers, verges, smooth shoulders and other non-road
objects; its retention rate is not proof of road discovery quality. Strict
geometry cannot determine road semantics from a bright roof or coplanar sidewalk
without additional review. Missing/occluded surfaces remain missing.

The annotations remain geographically unmatched, including the earlier raw-share
inventory. CRS, units, vertical datum and sensor/platform remain unconfirmed.
Full processing and Phase 4/5 remain gated. No pretrained model architecture was
changed, no weights downloaded, no labels generated, and no commit made.
