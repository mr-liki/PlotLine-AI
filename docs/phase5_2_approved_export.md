# Phase 5.2: approved outlines and safe-supervision export

The recorded user approval is authoritative for the displayed marking outlines.
It is not awaiting another QGIS review. The approved project is the
`symbols_review_20261105T161125583078Z` project, rather than its earlier
line-only predecessor. The existing attestation is in
`outputs/reports/phase5/20261105T161529533709Z_user_review/annotation_review_attestation.json`.

`scripts/run_phase5_2.py` verifies the exact project and annotation hashes,
historical raster hashes, and equality between displayed rasters and the source
rasters used for export. It exports positive masks from the approved buffered
footprints and original symbol polygons without redrawing, widening, connecting
dashes, changing coordinates, or altering approval records. Source feature IDs,
source hashes, geometry hashes, and the six-window approval scope are reconciled
in the output metadata. Annotation geometries outside that scope are not
promoted to approved training data.

The dataset contains exact grayscale RGB input copies, full-precision intensity
with NoData, instance maps, visual-class maps, pixel-state maps, positive-validity
masks, observation masks, coverage/padding masks, native affines, QA previews,
and provenance. Positive-validity masks are explicitly **not** valid substitutes
for a safe instance-loss criterion. Symbol visual IDs retain the existing
provisional client-code hints; the confirmed model class mapping remains
solid/dash visual 1/2 to model 0/1. No automatic `visual_id - 1` expansion is used.

Example commands in the activated `roadmarking-ai` environment:

```powershell
python scripts/run_phase5_2.py --approval outputs/reports/phase5/20261105T161529533709Z_user_review/annotation_review_attestation.json
python scripts/phase5_2_check_dataset.py data/processed/phase5_training/phase5_2_20261105T163226427054Z --report outputs/reports/phase5_2/dataset_validation.json
```

The export command refuses to replace an existing Phase 5.2 report or dataset.
The validator independently checks actual files, hashes, mask pixel counts,
NoData exclusion, instance identities, class mapping, pixel alignment, crop
eligibility and geographic connected components. It cannot be bypassed by
changing `training_ready`. Exit code 2 is the deliberate supervision gate,
even when dataset integrity passes.

Current outcome: **SUPERVISION_INCOMPLETE**. All 95 outlines are preserved:
78 dashed, 10 solid, five arrow-code and two give-way-code footprints. Six
positive image-mask pairs were generated. Some footprints have no observed
paint pixels; outline count is not the count of usable training instances.

The approval store has no review-area records and the existing attestation does
not establish complete-region background coverage. Observed pixels outside
approved paint remain unknown, not negative training examples. Every native
512/384/256 crop position was checked without resizing; none is fully supervised.
The six tiles form two connected groups after footprint guards and shared global
IDs, rather than six independent samples. No safe split was assigned.

The installed Transformers 5.22.1 criterion was inspected directly, with its
source-file hash recorded. Executable diagnostics show changes to an assumed
ignored pixel affect BCE, Dice and Hungarian matching. Unmatched queries receive
no-object gradients. Auxiliary outputs recurse through the same criterion.
These diagnostics use controlled tensors, not pretrained weights or a model
training run. Tests also cover empty background, unknown regions, multiple
instances, partial objects, padding, missing observations, and stale approval.

Next required evidence concerns **region completeness**, not the already approved
outlines: identify regions in which every target marking is included and observed
non-paint is valid background. Existing client region-coverage/QC records may
establish this; no such evidence is present in the inspected workspace. Once
that scope is established, NoData-aware matching, mask losses and auxiliary
losses still require implementation and validation unless fully observed crops
are available. Masking pixel loss alone does not resolve possible unannotated
objects receiving no-object supervision.

No weights were initialized, optimizer steps executed, checkpoint saved, model
GPU memory measured, or recognition accuracy reported. The remaining trainer
and predictor implementation is dependent work, not an executable training
engine delivered by this export. See `outputs/reports/phase5_2/report.md` and
the JSON audit artifacts for measured results.
