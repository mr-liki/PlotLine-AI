# Phase 5.4: larger supervised sample pool and new client-source review regions

The usable native-512 training pool increased from **10 to 89 crops**, including
**16 verified bright-background-only crops**. The original six-region approval
and completeness evidence was reused; their outlines did not need another review.
The five development-validation crops were preserved byte-for-byte.

This is a sample/coverage increase, not 89 independent geographic regions. The
unique supervised training source IDs remain **six solid and fourteen dash**.
The covered footprint increased from 5,898.24 to 8,683.52 native square units.
The expanded version is:

```text
D:\Plotline-AI\data\processed\phase5_training\phase5_4_20261105T173740792650Z
```

The supplied reference folder contains four GeoPackages and two DGN references,
with 16,521 GIS features in the GeoPackages. Their hashes match the existing
audited snapshots; one original polygon is invalid and is not used as valid
supervision. The network sources remain read-only.

Seven geographically new candidate windows were rasterized from matching actual
LiDAR. All 73,490,887 records in the three selected source files passed strict
integrity checks. The package contains 125 distinct original client feature IDs,
native intensity/observation rasters, supplied-width conversion proposals,
original client geometry, identity/provenance metadata and QA previews.

Open this **new-region** review project:

```text
D:\Plotline-AI\data\processed\phase5_4_candidates\20261105T173740792650Z_attempt2\new_regions_review.qgs
```

It has seven project bookmarks: RIGHT_1, RIGHT_2, RIGHT_3, LEFT_1, LEFT_2, LEFT_3,
and DISTINCT_SOURCE_1. The seven image previews are also available under
`review_previews_verified` beside the project. Orange footprints are conversion
proposals; magenta original symbol references remain visible. All vector layers
are read-only, so the review does not require editing or drawing in QGIS.

Only these **new** windows need their first alignment/conversion and complete
visible-target coverage approval. Existing client source QC remains accepted.
Unobserved areas are not background and cannot be visually certified from a
missing return. New windows share some long source IDs with approved regions;
they cannot be treated as an independent test fold under the current identity
rules. Once approved, preserve the linked geographic group allocation rather
than randomly splitting overlapping crops.

Three additional, disjoint-ID future-test locations are reserved in the versioned
report's `future_test_region_reservations.json`. They are metadata reservations,
not yet image-validated, human-approved, or evaluated test samples. They were not
used for training or confidence calibration.

All baseline false-positive and missed appearances were diagnosed and rendered
into eight error-gallery sheets. A bounded 300-setting calibration search reduced
false-positive appearances **115 to 61** while keeping the same class recall and
12 missed appearances. Duplicate appearances fell from 47 to 35. Mean valid-pixel
foreground IoU increased from .4462 to .5544. These gains are from postprocessing.
The winning setting uses solid confidence .85, dash confidence .25, mask quality
.80, and no mask NMS; the tested NMS options did not win under the recall-preserving
selection rule. No short-dash size suppression or instance joining was added.

Two controlled 200-step experiments used the expanded approved pool, lower
learning rates, class-balanced positives and verified negatives. Full checkpoints
save model, optimizer, cosine scheduler, scaler, step, RNGs and dataset/config
versions. A fresh-process resume at step 100 restored 757 optimizer parameter
states and continued the same schedule. The NoData/unsupported-class criterion
and its invariance tests were preserved. Background-only batches exposed an
FP16 overflow in a zero-valued loss branch; an FP32 reduction and a dedicated
large-FP16-empty-target test fixed it before completed experiments were run.

Neither retraining experiment beat the baseline with the frozen calibrated-F1
selection rule. The selected model therefore retains the actual trained Phase
5.3 weights and adds the measured filtering settings. The complete selected
bundle is:

```text
D:\Plotline-AI\models\mask2former_swin_s\phase5_4_20261105T173740792650Z_attempt2\checkpoint_step0000
```

The experiment-local step zero inherits the 188-step Phase 5.3 model; it is not
random initialization. The sibling step-100/200 bundles retain the additionally
trained models and optimizer moments. `postprocessing.json` is required to obtain
the selected recognition behavior. Fresh-process selected-model reload reproduced
class/mask logits exactly.

Public Toronto-3D and K-Lane sources were assessed separately, but their labels
were not blindly converted into solid/dash painted-area instances. No public
dataset was downloaded or mixed into training. The matched client-source package
is the most directly applicable expansion prepared in this run.

Validation remains development-only: five overlapping crops in one geographic
component and no independent test set. Precision and duplicates remain unsuitable
for unattended vectorization or production. No final 3D digitization occurred.
Final checks: **294 passed, 1 skipped**, Ruff/Black passed, healthcheck passed.

Read the versioned report:

```text
D:\Plotline-AI\outputs\reports\phase5_4\20261105T173740792650Z\report.md
```
