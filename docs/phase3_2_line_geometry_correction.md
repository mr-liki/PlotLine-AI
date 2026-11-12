# Phase 3.2: finite line geometry and evidence-based continuity

This phase diagnoses the existing Phase 3.1 review packages and adds an opt-in
painted-stripe tracker. The previous reference generator and all previous data
products remain unchanged. These products remain **PROVISIONAL**, not verified
road-marking digitization. No whole-survey processing, AI inference or training
is performed.

## Provenance of the extra lines

The two prior review packages contain 1,031 and 1,212 `reference_candidates`
LineStrings. That is the only populated line layer. `paint_review` is empty;
there is no populated final road-marking layer. Cross-section diagnostics were
JSON profiles and plot overlays, not exported GeoPackage lines.

`stripe_reference.detect_references` calls `_segment` to fit a finite local PCA
fragment, then an endpoint joiner. `qgis_bundle` in
`scripts/audit_phase3_1_topodot_pilot.py` exported each full halo proposal. Every
tile had a 30-native-unit halo, so a reference could legitimately extend beyond
the displayed 51.2-unit tile. **1,669 of 2,243** references extend beyond their
own tile footprint. These are not all invalid road features.

The source GeoPackage FIDs, source tile IDs, local reference IDs, bounds,
generator functions and layer roles are reproducibly recorded in:

- `outputs/reports/phase3_2/layer_inventory.json`
- `outputs/reports/phase3_2/layer_inventory.md`
- `outputs/reports/phase3_2/suspect_features.json`

No screenshots were attached to the Phase 3.2 request. The source layer can be
identified from the actual packages, but a particular brown feature in the
user's QGIS project cannot be assigned a screenshot FID from color alone.

## Actual implementation issues

| Issue | Evidence | Correction |
|---|---|---|
| Halo geometry outside displayed tiles | Export writes the entire reference; no tile clip | Candidate stripes/reference display products are explicitly clipped to selected scope; legacy context remains separately available |
| Gap acceptance does not test paint | Old joiner samples `observed`, including ordinary pavement | Short joins require original bright support plus observed coverage, compatible tangent, class and width |
| Straight approximation through curved evidence | Old PCA fragment has two vertices over its whole supported interval | Trace local ridge centers and graph topology; use evidence-constrained local smoothing |
| Duplicate window-local IDs | Overlapping halos produce separate references | Canonical group tracking uses one shared coordinate/evidence grid; per-tile fragments retain parent IDs |
| Ambiguous line roles | References were the only nonempty line product | Independent reference, candidate-stripe, QA and empty verified layers |
| Polygon stair-steps | Observation footprint uses 1-unit cells; strict candidates use station strips | Keep these explicitly as diagnostic polygons; do not mistake their boundaries for paint or cosmetically smooth them |

No infinite Hough line, unbounded RANSAC fit, or demonstrated affine-origin/
scale error was found. PCA endpoints use finite support percentiles. A finite
line can nevertheless pass through unsupported positions between its endpoints.
The diagnostic records distinguish that problem from tile extent and NoData.

## Independent corrected tracker

`src/roadmarking/preprocessing/stripe_tracking.py` reuses the existing
`local_contrast` operation over all valid source observations. It never requires
strict road acceptance as a prerequisite.

1. Construct local bright ridges from normalized intensity and nearby contrast.
2. Use a one-cell neighborhood for sampling topology; retain original masks.
3. Thin to a graph and trace finite paths, splitting at graph junctions.
4. Anchor path endpoints to original bright-cell centers.
5. Apply local smoothing only where proposed positions stay within measured
   bright support; do not move endpoints/junctions arbitrarily.
6. Measure every path against the original observed/bright evidence.
7. Preserve unsupported or ambiguous paths in a dedicated QA layer. When one
   sampling hole interrupts a long stroke, retain supported runs separately.
8. Permit only short, observed, bright-supported, tangent/width/type-compatible
   joins. All interpolation is explicitly provisional. No stripe is intentionally
   extrapolated through a long unsupported gap.
9. Keep short/dash-like strokes separate. Sequence IDs never add a painted bridge.

Candidate classes are length-based hypotheses: `DASH_OR_SHORT_STROKE_CANDIDATE`
and `CONTINUOUS_OR_FRAGMENT_CANDIDATE`. They are **not verified physical classes**.
Graph branches, coarse sampling, mixed objects, abrupt turns and closely spaced
parallel structures can remain ambiguous. Nothing is automatically approved.

`configs/phase3_2.yaml` records the engineering settings. Initial maximum width
0.8 and length categories inherit prior exploratory settings; they are not
universal physical truths. Bright-cell spacing is measured on each actual raster.
The support radius is derived from that spacing and pixel size; maximum short
interpolation is limited by the configured physical distance and sampling scale.
Every line stores those measurements. Manual reviewed examples are still needed
to calibrate acceptance thresholds and quantify true marking accuracy.

## Controlled pilot and output layout

```powershell
$mm = "$env:USERPROFILE/micromamba/micromamba.exe"
& $mm run -n roadmarking-ai python scripts/run_phase3_2.py --stage inventory
& $mm run -n roadmarking-ai python scripts/run_phase3_2.py
# Optional bounded development group:
& $mm run -n roadmarking-ai python scripts/run_phase3_2.py --group 3_boundary_overlap
```

The final controlled pilot selects eight existing tiles in five groups:
straight and boundary overlaps on Track01_3, its ambiguous smooth blob, an overlap
on Track01_8, and its curved-road case. Original XYZ/intensity/source-record IDs
are read from hash-verified Phase 3.1 spatial partitions. LAZ source hashes are
verified; no full LAZ decoding or complete-survey rasterization is done.

Old and corrected implementations operate on the **same 0.1-native-unit raster
and original point set**. Saved historical FIDs are separately evaluated against
these original observations; their original discovery raster was 0.2 units.
The 0.1 diagnostic grid aligns to the existing 0.05 source lattice. Existing
model-input tiles are not shifted, resampled or overwritten.

Data outputs: `data/processed/phase3_2/<fingerprint>/`.
Reports: `outputs/reports/phase3_2/report.md` and `report.json`.
Per-group JSON and `feature_diagnostics.json` retain numeric evidence and source
identities. All new data artifacts are fingerprinted. Previous packages and
manifests are hash-checked before and after.

## QGIS layer order and styles

Use `metadata/manifest.json` to locate each group package and its GeoTIFFs. Start
with one group at a time. **Do not assign a guessed CRS or transform coordinates.**
All output GeoPackages and rasters retain native XY and no assigned CRS.

Load layers in this order from bottom to top:

1. `rasters/<group>/normalized.tif`: grayscale, minimum 0, maximum 1; keep visible.
   `maximum.tif` preserves raw uint16-range maxima as signed values with -1 NoData.
2. `observed.tif`: inspection layer, normally hidden; 0 missing / 1 measured.
   `count.tif` gives original point support.
3. `wide_marking_search`: blue outline, **no fill**, provisional context.
4. `strict_surface_candidate`: green outline, **no fill**, provisional geometry.
5. `candidate_reference_alignments`: cyan dashed, corridor orientation only.
6. `candidate_painted_stripes`: orange solid, provisional intensity-derived paths.
7. `unsupported_line_qa`: red dotted, normally hidden until inspecting rejects.
8. `verified_road_markings`: empty; reserved for independently accepted features.

`legacy_context_references` uses brown dashed lines and should be toggled for
before/after diagnosis. Its out-of-footprint portions are deliberately retained
and labeled, not silently deleted. QML line styles are under `styles/`; use QGIS
Layer Properties → Symbology → Style → Load Style. Polygon styling above is
documented rather than relying on a random solid fill that hides intensity.
The review package is not a final marking deliverable.

Comparison figures show raw intensity, observed/NoData, point coverage, outline
polygons, old references, corrected stripes and QA separately on the same extent.
Suspect-detail figures highlight the exact source FID in magenta. They are
data-rendered overlays, not claimed screenshots from the user's QGIS project.

## Line QA and continuity

Each candidate has length, nearby bright-supported length, unsupported length,
between-bright-cell/interpolated length, original source observation count and
density, local contrast, outside-observation fraction, search-envelope fraction,
curvature/turn statistics, component count, cross-tile membership, class and review
status. These are diagnostic indicators, not probabilities.

Original point evidence under `source_evidence/` contains exact source XYZ, raw
intensity and source record indices for maximum-intensity observations in bright
cells near the traced path. Proximity and brightness are still not manual paint
truth. No original source point or model-input observation is deleted by tracking.

Canonical candidates are generated once over each selected overlap group, then
clipped to individual tile footprints with shared parent IDs. Exact shared-overlap
geometry is audited. Separate endpoint matching requires type, width, tangent,
distance, bright support and observed coverage. A nearby endpoint alone is never
sufficient. Same-type overlap duplicates may be removed only when almost entirely
covered; unique tails and close parallel paths remain available.

## Manual validation and final-layer gate

No approved new stripe examples are available. `verified_road_markings` is empty.
Review candidate strokes against raw intensity, observed coverage and original
point evidence. A reviewer must explicitly accept the candidate's geometry and
physical class and record their identity and review time in a separate review
artifact. Only those attributed, accepted geometries may be copied into a new
verified deliverable. References, QA objects and polygons must never be bulk-
renamed into final markings. The current pilot never performs that promotion.

## Tests and limits

`tests/test_phase3_2.py` covers finite/no-unbounded geometry, clipping, source
evidence support, straight/curved tracking, short and long gaps, parallel lines,
dash grouping without bridges, common-world overlap geometry, duplicate removal,
tangents, junction splitting, NoData, rejection, observation/file immutability,
QGIS role separation and deterministic processing. The complete existing suite,
healthcheck, Ruff and Black are also run; results are in the report directory.

Lower unsupported length or a cleaner-looking layer does not establish improved
true marking accuracy or recall. Some genuine paint may remain in QA or only in
the preserved raw observations. Bright roofs, barriers and other objects can
still form plausible ridges. The diagnostic resolution and conservative support
rules can fragment sparse measurements; real physical gaps and sampling gaps are
not automatically interchangeable. Curvature is evaluated on known synthetic
truth and visually on real observations, not against unmatched client annotations.

CRS, units, datum and acquisition metadata remain unconfirmed; client annotations
remain geographically unmatched. Full processing, training and Phase 4/5 remain
gated. No weights were downloaded, client files modified, or Git commit made.
