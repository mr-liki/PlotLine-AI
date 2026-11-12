# Phase 4: human-verified road-marking annotations

## Corrected-source workspace (2026-11-05)

The active workspace is now the corrected-source review pilot, recorded in
`outputs/reports/phase4_verified/latest.json`. Generate it with
`python scripts/prepare_corrected_annotation_project.py`. The raw source is
`\\192.168.10.82\nagendra\NAGENDRA\PROJECTS\NT_895\GIS\Asset project\2026\INPUTS\LAS`.
The four original Road_marking GeoPackages match this source in native XY:
16,520 valid features intersect readable raw footprints, with actual sampled
intensity/centerline alignment checked in RIGHT, LEFT and INTERCHANGE. One source
polygon is invalid; see `phase4_corrected_source_alignment/annotation_reuse_manifest.json`.
No coordinate transform is needed or applied. Earlier forced-fit outputs are excluded.

Open `annotation_projects/road_marking_annotation.qgs` in the latest output root.
Its three bookmarks show bounded point-window samples, not complete production
rasters. Their diagnostic normalization and missing point windows prevent treating
these samples as production training tiles. Complete-file QC and production image
generation remain necessary before dataset release. The old nine-image workspace
below is preserved in report history and uses different geography.

The four original client layers are workspace-owned, read-only reference copies
with unchanged geometry and attributes. Toggle a reference layer to inspect its
placement over the sampled image. Reuse its visible outline by copying the relevant
geometry into an editable draft layer, then check every field and the painted area.
Clip a review copy to the available image extent; preserve its original ID in the
comment and retain the original reference geometry. Do not reuse the old rigid fit.
For a line, verify width and buffered paint shape; do not guess a common width.
Source authoring does not establish pixel-area completeness or human approval.
Only approve after the checklist below. All pixels initially remain unknown/NoData.

Class IDs, review CLI, snapshot/export tools and loss safeguards described below
remain unchanged. Classes 1 and 2 are enabled for the small review pilot; other
client categories remain available in original references pending catalog/visual
adjudication. No real approved masks or train/validation/test splits exist yet.

## Preserved original Phase 3.3 workspace

This workflow starts from Phase 3.3 native 1024-pixel images, with 256-pixel overlap
and unchanged 0.05 native-coordinate-unit sampling. It creates an editable QGIS
workspace, not automatically verified training labels. The mismatched client
GeoPackages are reference material only and are never imported as ground truth.
No authoritative client catalog document was found in this project; existing
code hints remain provisional. No CRS, physical units or paint colour is inferred.

## Open the prepared workspace

Use the project path in [latest.json](../outputs/reports/phase4_verified/latest.json)
and open `annotation_projects/road_marking_annotation.qgs` in installed QGIS 3.44.
The project was generated and reopened using the installed PyQGIS API. It has
nine native-grid intensity images, full-precision source raster references,
observed/valid masks, source footprints and nine spatial bookmarks. Displayed
images are the original clean grayscale PNGs with their original worldfiles.
For safe editing, the project uses byte-identical small grayscale display copies
and nine exact native-resolution normalized GeoTIFF crop copies. QGIS/GDAL can
cache display statistics even when viewing a raster; these workspace-owned copies
keep such caches away from protected source rasters. The canonical source paths,
hashes, original full-precision layers and source-window references remain in the
annotation store metadata. Raw maximum/mean/count rasters are not duplicated.
There are only six independent footprint groups; overlapping tiles are context,
not extra independent examples. More geography is needed before reaching the
20–30-region planning target or a credible held-out test set.

Keep the project and layers in their undefined native XY frame. If QGIS asks for
a CRS, leave it undefined; do not select EPSG:3763 or transform any layer. Use a
bookmark to visit a region rather than zooming to the combined two-track extent.
The project opens at the first tile. Enable the observed mask to distinguish a
measured black pixel from a missing return; enable the valid mask to see padding.
Do not treat interpolation, bright vegetation or reflectors as painted evidence.

## Draw actual paint

1. Select `paint_polygons`, enable editing, zoom until individual pixels are visible,
   and draw tightly around the visible painted area. Preserve character/polygon
   holes with QGIS's Add Ring tool. A bounding rectangle is not an arrow outline.
2. Use one feature/global instance per physical dash or stripe. One continuous
   marking crossing tiles is drawn once, with one stable `global_instance_id`;
   its exported local IDs may differ between tiles. Use `group_id` to relate dashes
   or inscription characters without painting their gaps. Multipart geometry is
   permitted for disconnected visible portions of one physical marking.
3. Choose `visual_class_id`: 1 `longitudinal_solid` or 2 `longitudinal_dash` for the
   current pilot. Check the paired name, source track, source raster/tile anchor,
   geometry type and native coordinate frame. QGIS supplies default UUIDs and
   source lookups, but the reviewer must check them, especially at overlaps.
   Source lookup failures must be corrected in the form before verification.
4. Leave `client_code` empty when its exact functional/catalog meaning is unknown.
   Do not infer a road-edge designation or traffic function from an isolated stripe.
5. Leave the status DRAFT. Record uncertainty, faded paint, gaps, occlusion and
   class ambiguity in the note/comment fields. Do not trace through missing returns
   to make a confident polygon. Check narrow paint and parallel lines independently.
6. Save edits and end editing before running validation, approval or snapshot tools.

`paint_centerlines` retains centerline geometry for later geometric evaluation.
Prefer polygons for segmentation. To convert a line to a paint mask, the reviewer
must measure its width in **native units**, set `marking_width`, explicitly set
`width_verified=1`, inspect the buffered outline with the QGIS buffer expression
`buffer($geometry, "marking_width"/2, 8, 2, 2)` (flat cap, mitre join), and set
`mask_reviewed=1` only when its painted area matches the image. Width is never
defaulted to 0.15 or assumed to be meters. Incorrect or uncertain widths remain
unverified; use a polygon instead. Flat caps preserve individual dash endpoints.

## Review completeness, background and uncertainty

Use `review_areas` independently of paint features:

- `COMPLETE_REGION`: a human has checked **every** target marking inside this area,
  reviewed all its paint features and checked that nothing obvious was omitted.
  Observed non-paint pixels inside an approved region may become background.
- `BACKGROUND`: an explicitly reviewed negative area containing no target paint.
  Do not draw it across actual paint. A region is not negative merely because the
  simple intensity proxy found nothing.
- `UNCERTAIN`: an area that must remain ignored. It overrides paint/background
  labels until adjudicated. Unknown and NoData pixels are never silently background.

All new areas start unverified. Source footprints are a separate planning layer
and do not establish road identity, completeness, background or marking truth.

## Human approval and audit

First draw and classify; then inspect placement, outline, individual dash separation,
grouping, faded edges, roadside objects, missing returns and tile boundaries.
Use a second reviewer for evaluation samples where practical. Ambiguous cases stay
UNCERTAIN or NEEDS_CORRECTION until an attributed decision is made.

Use the existing `roadmarking-ai` environment. Replace paths and IDs below with the
actual workspace and feature values. Validation never approves a feature:

```powershell
python scripts/validate_annotations.py "D:/Plotline-AI/data/processed/phase4_verified/<run>/annotations_draft/road_marking_annotations.gpkg"
```

Copy `metadata/human_review_checklist_TEMPLATE.json` to a feature-specific review
record. Set each item to `true` only after checking it. False is the supplied default.
The complete list covers class, painted outline, placement, instance separation,
grouping, false paint, omissions in the reviewed area, connections and uncertainty.
For an individual feature, the omissions item applies to its reviewed surroundings;
only a separately verified COMPLETE_REGION authorizes background beyond the feature.

```powershell
python scripts/review_annotations.py "<workspace>/annotations_draft/road_marking_annotations.gpkg" --annotation-id "<uuid>" --reviewer-id "<human-reviewer>" --status VERIFIED --checklist "<feature-review.json>"
```

The CLI records reviewer/time, the checklist and a fingerprint of the current source,
geometry, width, class and other annotation attributes. Merely selecting VERIFIED
in QGIS is insufficient: an absent/stale event fails eligibility. Geometry, class or
source edits invalidate approval. Review and edit history retain old geometry and
attributes, including deleted features; audit tables reject updates/deletes.
Statuses are DRAFT, PROPOSED_UNREVIEWED, NEEDS_CORRECTION, VERIFIED, REJECTED, UNCERTAIN.
No software check can establish that a person actually inspected the evidence;
these records establish attribution and consistency, not automatic correctness.

After reviewing and validating, export an immutable **new** snapshot (close QGIS
editing first). The same command imports a workspace snapshot into a new location;
it preserves all native geometry, schema, source identities and review/edit history:

```powershell
python scripts/review_annotations.py "<workspace>/annotations_draft/road_marking_annotations.gpkg" --snapshot-to "<workspace>/annotations_verified/review_001.gpkg"
python scripts/validate_annotations.py "<workspace>/annotations_verified/review_001.gpkg"
python scripts/build_training_dataset.py "<workspace>/annotations_verified/review_001.gpkg" --output "<workspace>/export_001"
python scripts/check_training_dataset.py "<workspace>/export_001"
```

Snapshot destinations and export directories must not already exist. Originals
are not overwritten. Arbitrary client GeoPackages cannot be imported through this
path: they lack this workspace's native source schema and provenance.

## Masks, class identities and loss limitation

The converter reuses existing supersampled area rasterization, hole preservation,
flat-capped line buffering and deterministic global/local instance mappings.
It exports normalized intensity, instance/semantic maps, explicit validity masks,
provenance and colored QA overlays. Original RGB intensity images remain clean.
NoData or padding overrides any polygon claim. Review-only uncertainty overrides
paint/background. Polygon boundary and instance conflicts remain ignored.

Validity category codes are 0 NoData, 1 unknown, 2 verified background, 3 verified
paint, 4 boundary-ignore, 5 conflict-ignore, 6 padding, 7 explicit uncertainty.
Existing legacy state TIFFs retain their six-code convention; canonical category,
observed, valid and ignore TIFFs provide the additional distinction.

Stored visual IDs 1–9 retain the existing taxonomy. The new taxonomy explicitly
records `training_class_id=visual_class_id-1` (0–8) for future model labels; this is
not a silent change to annotation/GIS IDs. Local instance IDs are separate from
both class IDs and stable global UUIDs. Unknown and background are not paint object
instances. Reserved/unlocated classes remain disabled; enabling them later requires
catalog adjudication, real examples and an explicit versioned taxonomy revision.

The installed Transformers processor removes its ignore ID from the object list;
it does **not** prevent BCE/Dice mask loss at unknown or missing pixels within each
remaining binary target. The Hungarian matcher and point-sampled auxiliary losses
also lack this review-validity contract. `pixel_mask` is not a loss-ignore mask.
A fixed-tensor self-test demonstrates the issue without model inference/training.

The export therefore marks partial samples **review-only**, and the stock-target
adapter rejects any sample with unknown/NoData/padding/boundary/conflict pixels.
Sparse real LiDAR is likely to have such pixels even after complete human review.
No valid real training subset is promised until a future loss/matcher/point-sampling
validity strategy is implemented and tested, or genuinely fully supervised samples
can safely be selected. That strategy is deliberately not a model modification here.

Processor tests use `do_resize=False`, categorical instance IDs, uint8 replicated
grayscale, a single 1/255 rescale and standard channel normalization. No smooth
mask interpolation, weights download, model inference, head replacement or training.

## Splits and current delivery

Only verified labels may reach the splitter. The existing spatial splitter handles
overlap and instance guards across both tracks; no continuous-instance exemption is
used by this new workflow. Verified negative samples receive split-only tokens,
never fake paint IDs. Small subsets remain unassigned/pilot-only; 70/15/15 is not
forced. Keep independent test geography untouched during model development.

The current store is empty and no real image/mask pair or split is created. The
separate `quality_control/SYNTHETIC_ONLY` demonstration exercises narrow paint,
dashes, holes and cross-tile identity; it is not client ground truth. The authoritative
delivery state is **ANNOTATION REQUIRED**, with additional loss-validity and class/
coordinate/evaluation blockers recorded in the [report](../outputs/reports/phase4_verified/report.md).
