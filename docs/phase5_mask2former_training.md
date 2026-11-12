# Phase 5: source-QC reuse, complete-point raster pilot and supervised-data gate

The client GIS source is QC-final per the user's explicit confirmation. That
confirmation is recorded against the six source-file hashes in
`outputs/reports/phase5/source_qc_attestation.json`. Original geometry,
classifications and supplied widths are reused. Complete redigitization is not
requested. Invalid or duplicated geometry remains an explicit exception.

The corrected raw source is
`\\192.168.10.82\nagendra\NAGENDRA\PROJECTS\NT_895\GIS\Asset project\2026\INPUTS\LAS`.
The existing annotation path has a separator between `DAILY ACTIVITY REPORT`
and `!!!!!geetha`; the Phase 5 pasted document omitted it. Source configuration
remains in `configs/phase4_network_sources.yaml`. Both shares are read-only.

## Execute the independent preparation stages

Use the existing `roadmarking-ai` environment; do not reinstall dependencies.

```powershell
python scripts/phase5_inventory_sources.py
python scripts/phase5_prepare_production_pilot.py
python scripts/phase5_build_review_package.py
```

The inventory audits all LAS/LAZ headers and three actual record windows per
file. This is sampling, not whole-file certification. A partial fingerprint
includes first/last blocks, size and timestamp and is explicitly not a full-file
hash. Annotation snapshots use full SHA-256 checksums. Source paths, schemas,
width attributes, duplicates, class distributions, CRS and invalid geometries
are retained. Drawing JSON sidecars have also been inspected for approval fields.

The raster pilot fully decodes three selected source files through the existing
strict chunk reader, validates bounds/returns/GPS time, and checks exact point
counts. It uses all valid points from each selected file in its pilot extent.
No bad chunk is partially retained. The remaining sources are not represented
as fully validated. Shared uint16 histogram calibration is fitted once across
the three files, with p1/p99.5, the existing minimum-range safeguard and gamma
0.7. No per-tile normalization, filling, resizing or vector burn-in is applied.

Each group has two 1024-pixel tiles on the global 0.05-native-unit grid, with
768-pixel stride and exact 256-pixel overlap. They retain all observations in
their selected-source windows; no unverified road filter discards faint paint.
All-return context includes surroundings and is not a confirmed road-only mask.
Other intersecting survey footprints are listed but have not been mosaicked.
Complete selected-file coverage does not imply complete all-pass survey coverage.

## Reuse the existing QC geometry

Read `outputs/reports/phase5/annotation_conversion_latest.json` for the current
conversion report and QGIS project. The project contains six full-point image
windows, unchanged read-only client layers and editable derived paint polygons.
Each derived polygon retains the source hash/layer/FID, source ID_SIG and supplied
width. Stable source IDs link one physical feature across tiles. Original Z and
geometry remain in untouched source snapshots; masks operate in native XY.
The original source CRS declaration is preserved as evidence. The editing project
has no guessed CRS or coordinate transformation.

The current 88 paint-area proposals come from QC-final longitudinal centerlines
using their own LARGURA attributes, flat end caps and the existing buffer code.
These are conversion proposals, not invented human approvals. Background outside
paint is unknown until the selected image region's complete labeling scope is
established. NoData stays distinct from observed dark asphalt. Other source
polygons remain visible; uncertain catalog-to-AI mappings have not been invented.
The stable solid/dash IDs remain visual 1/2 and model 0/1.

Use the existing Phase 4 drawing, checklist and review CLI procedure in
`docs/phase4_verified_annotations.md` on the **Phase 5 store** in this project.
Review conversion width/endpoints and the observed region's complete class/instance
coverage; preserve one source global ID across overlapping tiles. Correct dubious
buffer shapes or mark uncertainty. A source-QC statement is not recorded as a new
per-pixel review event for generated buffers. Catalog adjudication is required
before enabling other AI classes. No blanket reannotation of 16,520 features is
needed: this workspace focuses the remaining conversion checks on six windows.

When saving a reviewed snapshot and exporting, always use new local destinations:

```powershell
python scripts/validate_annotations.py "<phase5-store.gpkg>"
python scripts/review_annotations.py "<phase5-store.gpkg>" --snapshot-to "<new-snapshot.gpkg>"
python scripts/build_training_dataset.py "<new-snapshot.gpkg>" --output "<new-export>"
python scripts/phase5_check_dataset.py "<new-export>" --annotation-store "<new-snapshot.gpkg>" --report "<preflight.json>"
```

The review CLI's `--annotation-id`, `--reviewer-id`, `--status VERIFIED` and
completed `--checklist` are used only by the actual reviewer after checking a
feature/region. Merely editing a status field does not approve it. Changes invalidate
existing approval. Source protection and append-only provenance are reused.

## Current material training blocker

The full-point pilot has no wholly observed 1024, 512, 256 or 128 native crop.
See `outputs/reports/phase5/native_crop_coverage.json`; even the best tested
128-pixel window contains missing observations. The six full tiles also lack an
established complete-region background mask. Valid source QC does not make
unmeasured pixels negative training examples.

Installed Transformers 5.22.1 stock Mask2Former loss does not use `pixel_mask` as
an arbitrary loss-ignore map. The Phase 4 fixed-tensor test and fail-closed
processor adapter remain in force. Current masks therefore live under
`conversion_review_only`, with `training_ready=false`. Their semantic paint pixels
visualize the conversion; they are not primary supervised labels. They cannot
be promoted by toggling a manifest flag. The preflight reads actual state,
instance and semantic TIFFs, checks hashes, provenance and spatial separation.
Exit code 2 means a failed data gate, not a CUDA failure.

Before training, either establish fully reviewed paint/background regions with
genuinely fully observed native crops, or implement and validate a loss-validity
strategy covering matching, point selection, BCE/Dice and every auxiliary loss.
The latter must still exclude genuinely unreviewed regions and cannot repair
missing label completeness. Filling missing returns with synthetic values or
ignoring labels only in the processor is not a resolution.

## Model and execution state

`configs/phase5_mask2former.yaml` pins the official Swin-S COCO initialization to
revision `1b7ac418a0f3cb5edfc5037e488d53587b8bb76b` and records future bounded
training settings. The official repository lists safetensors and license `other`:
[official checkpoint](https://huggingface.co/facebook/mask2former-swin-small-coco-instance).
Commercial redistribution/deployment requires a separate rights review.

No approved training pairs or independent splits currently exist. Therefore
weights download, road-class head initialization, GPU forward/backward feasibility,
fine-tuning, evaluation, recognition and trained-checkpoint saving were not
executed. Existing downstream trainer/inference interfaces remain unimplemented;
this delivery does not claim an executable or validated end-to-end training
engine. These dependent stages remain work after the supervision prerequisite
is resolved. The configuration is a recorded starting point, not measured model
feasibility. CUDA health checks do not prove that Swin-S training fits 6 GB.

No trained model or AP/precision/recall/F1/IoU values are available. Phase 6 is not
ready. See `outputs/reports/phase5/report.md` for measured preparation, tests,
exceptions and exact stage status.
