# Phase 3.1: road-corridor extraction correction

Measured results: [phase3_1_report.md](../outputs/reports/phase3_1/phase3_1_report.md)
and [phase3_1_report.json](../outputs/reports/phase3_1/phase3_1_report.json).
Phase 3 outputs, the production Phase 3 manifest and Phase 4 outputs are unchanged.

## Run

```powershell
micromamba run -n roadmarking-ai python scripts/run_phase3_1.py --stage diagnose
micromamba run -n roadmarking-ai python scripts/run_phase3_1.py --stage trial
micromamba run -n roadmarking-ai python scripts/run_phase3_1.py --stage full
micromamba run -n roadmarking-ai python -m pytest --junitxml=outputs/reports/phase3_1/pytest.xml
micromamba run -n roadmarking-ai python -m roadmarking.healthcheck --json outputs/reports/phase3_1/healthcheck.json
micromamba run -n roadmarking-ai python scripts/run_phase3_1.py --stage report
# with a manually reviewed corridor (native XY, GPKG/GeoJSON):
micromamba run -n roadmarking-ai python scripts/run_phase3_1.py --stage full --road-roi reviewed_roi.gpkg
```

Stages:

* `diagnose`: v1 rejection reasons and all-point window figures.
* `trial`: v1 and v2 on the identical Track01_3 problem region.
* `full`: v2 on both intact files.
* `report`: comparisons, figures, proposed ROIs, combined manifest and reports.

Each run is fingerprinted (source hash, settings, refinement criteria, calibration,
ROI hash, code) and written under `data/processed/phase3_1/{rasters,tiles,metadata,road_masks}/<source>/<mode>_<run_id>`.
Proposed ROIs are written to `data/processed/phase3_1/candidate_rois/`, and the combined
manifest to `data/processed/phase3_1/metadata/tile_manifest.json`.

## Why Phase 3 stopped inside the road

Phase 3 (v1) fits one plane to all points in each 1 m cell. A cell is rejected if the plane
RMS is above 4.5 cm, the slope is above 0.20, it has fewer than 8 points, or there is a
center-height step above 0.20 to an accepted neighbor. Components under 100 m² are also
dropped. Point cross-sections show the 12 m candidate is the whole visible carriageway,
between the kerb/embankment and a concrete median barrier.

* **Main cause:** the 1 m cells at the barrier and kerb mix asphalt with barrier or kerb
  points. The whole cell then fails the RMS test, and the 0.5–1.5 m of road beside the
  barrier is lost, including the edge line. RMS is the dominant failing criterion near the
  candidate.
* **Not recoverable:** the opposite carriageway is only seen as sparse patches in the
  barrier's scan shadow. There is no continuous observed surface there to recover.

## Method (`road_method="v2"`, `src/roadmarking/preprocessing/road_surface.py`)

Phase 3 parameters and the v1 code path are unchanged; v1 remains the default.

1. **Continuity breaks:** neighbors whose planes disagree at their shared edge
   (by more than 5 cm, or by a slope change above 0.06) are split. This separates kerbs
   and footpaths that v1's 0.20 m step rule joined.
2. **Corridor core:** components of at least 500 m² that are corridor-shaped
   (length/mean width ≥ 3, mean width ≤ 40). The 40 is an upper bound, not an assumed
   road width. Small or broad components become UNCERTAIN.
3. **Robust edge test:** this needs one extra streaming LAZ pass.
   * Cells up to 2 cells from the core are re-tested point by point. Each point is
     compared with the nearest core cell's plane, extrapolated so that grade and banking
     carry over.
   * Inliers are points with |Δz| ≤ 6 cm. The inliers' own plane must have an RMS of at
     most 3 cm, a slope change of at most 0.06 and an offset of at most 4 cm.
   * A cell needs 8 or more inliers, or 3 or more inside topological holes in the corridor
     (scan gaps, vehicles).
   * Only inliers are selected, so barrier, kerb-top and vehicle points stay out even in
     accepted cells.
4. **Coplanar promotion:** a smaller component joins the corridor only if it touches it
   through verified cells and its median offset from the corridor plane is at most 4 cm.
5. **Uncertain surfaces:** smooth cells up to 30 cells away that stay unconnected are
   UNCERTAIN and are never selected automatically. These could be the opposite
   carriageway, a footpath, a roof or a field.
6. **Evidence categories:** each cell gets one of: no observation, rejected, insufficient,
   uncertain, high, or ROI. A per-cell evidence bit mask is also stored.
   * Files: `surface_confidence_1m.tif` and `refinement_model.npz`, plus a 5 cm
     `road_confidence` layer in every tile.
   * These are evidence categories, not probabilities. No numeric score is produced.

Intensity is never used for selection. v2 reuses each file's Phase 3 intensity
calibration (`calibration_override`), so normalized values stay directly comparable.
Missing pixels stay NoData, and gap fill is disabled.

All parameters are in `configs/default.yaml` → `phase3_1.refinement`.

## Road ROIs (`src/roadmarking/preprocessing/road_roi.py`)

* **Formats:** GeoPackage, Shapefile, GeoJSON or the native JSON format.
* **Geometry:** multiple polygons and holes are supported. Overlaps are unioned and their
  area is reported. Invalid or non-polygon features are refused; nothing is repaired.
* **Frame:**
  * The ROI must overlap the source's native extent.
  * An ROI that declares a CRS is refused for the CRS-less LiDAR unless the caller passes
    `--roi-assert-native-frame`.
  * Coordinates are never changed.
* **Review:** features marked `review_status = PROPOSED_UNREVIEWED` are refused, so an
  automatically generated proposal can never be fed back as approved.
* **Effect:** HIGH or UNCERTAIN cells inside an approved ROI become ROI-confirmed
  (`road_state = 2`); everything outside is excluded. Excluded points and cells are
  reported.

## Outputs and compatibility

* Same global grid as Phase 3: the origin is derived from the source header, with
  0.05 resolution, 1024 px tiles and 256 px overlap (stride 768). New tiles appear only
  on that stride grid. The `road_state` values keep v1 semantics.
* Tile metadata includes `road_extraction_version`.
* The combined manifest also includes, per tile:
  * candidate-evidence percentages
  * a Phase 3 comparison
  * QA status
  * ROI review status
  * road review status
  * `phase4_status = NEEDS_ROAD_REVIEW`
* The Phase 4 preflight accepts it. It does not replace
  `data/processed/phase3/metadata/tile_manifest.json`.

## Reference: Naksha PlotLine

The in-house PlotLine tracer (`D:\lidar_classifer_new\plugins\plotline`, v1.15.1)
finds roads from paint. Its rules were used as a reference, and its traced polylines
serve as independent paint evidence (`scripts/run_plotline_reference.py`). It runs
read-only on the CPU reference rasterizer and checks the LAZ hash before and after.

| PlotLine rule | Phase 3.1 v2 counterpart |
|---|---|
| An edge more than 20 cm above the inward road surface is a kerb or barrier top, not paint | Points more than ±6 cm from the extrapolated corridor plane are outliers, even inside an accepted cell |
| The carriageway must be flat (median plane residual ≤ 3 cm) | Inlier-plane RMS ≤ 3 cm; v1 cell RMS ≤ 4.5 cm |
| A road is bounded by two sustained painted sides with regular lane dashes, ≤ 20 m wide | Not used for selection, because intensity must not drive surface extraction. This is a candidate for future corroboration of UNCERTAIN parallel surfaces, such as the opposite carriageway. |
| Tentative roads must continue a confirmed road | Coplanar promotion only through verified edge cells |
| Long unobserved spans are not filled | Empty cells never produce pixels; no gap fill |

The report's section D gives the share of PlotLine edge-line and dash vertices
(densified to 0.25 m) inside the v1 and v2 corridors.
These are length-based proxies: lines ≥20 m are grouped as edge/solid, and
2.5–12 m lines as dashes; they are not manually verified marking classes.
Track01_3 reuses the earlier scratchpad result. Its source hash was checked
before and after reuse against the Phase 3 source identity; the original
tracing-time hash was not recorded. Track01_8 is traced afresh with hashes
checked before and after tracing.

## Raw data share

`\\192.168.10.6\laz_1_2` was inventoried from headers only (no points decoded):
`outputs/reports/phase3_1/raw_share_header_inventory.json`.

* **Contents:** 266 entries, about 4.6 billion header points, from job groups 0838,
  1014, 1343 and 20251118.
* **Problem entries:** 15 zero-point headers, 12 header errors, 3 empty files and
  2 `.tmp` files.
* **Coordinates:**
  * No header carries a CRS.
  * The combined extent is X −57.5…−42.1 km, Y −118.8…−52.7 km.
  * No file overlaps the Phase 4 annotation area (Y ≈ +166…+177 km).
* **Local copies:** the local `data/raw/pointclouds` files are same-size copies of
  share files.
* **Scope:** Phase 3.1 processed only the two verified intact tracks. Other files need
  Phase 2's strict integrity scan first.

## Tests

`tests/test_phase3_1.py` covers:

* The 12 synthetic scenarios: flat, sloped, embankment with a broad field, an adjacent
  lane behind a barrier, a raised footpath, a flat roof, a scan gap, curved, banked,
  marking at the boundary (with intensity independence), sparse coverage, and a
  disconnected surface.
* ROI validation and clipping.
* Determinism and input immutability.
* An end-to-end v1→v2 pipeline run: namespace separation, identical grid, NoData,
  Phase 4 manifest compatibility, and byte-identical repeat runs.

## Limitations

* Without a reviewed ROI the candidates are automatic and unverified.
* A coplanar hard shoulder or verge cannot be told apart from carriageway by geometry
  alone.
* Boundaries follow 1 m cells (stair-steps), except where point-level inlier selection
  applies.
* CRS, native units and vertical datum are unconfirmed.
* The annotations still do not overlap the LiDAR.
