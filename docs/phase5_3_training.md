# Phase 5.3: complete regions, valid supervision and an actual trained pilot

Status: **TRAINED_PROVISIONAL**. An actual pretrained Mask2Former Swin-S was
fine-tuned on CUDA and its saved checkpoint was reloaded in a fresh process.
It is not ready for production or automatic digitization.

The user's Phase 5.3 message establishes complete visible-target coverage in the
six reviewed windows. The region-level attestation is linked to the prior
outline approval and exact annotation/raster hashes in
`outputs/reports/phase5_3/region_completeness_attestation.json`. No further review
of those outlines or regions was requested. All 95 outlines retain their geometry
and identities. Source data and previous outputs remain unchanged.

The versioned supervised export is
`data/processed/phase5_training/phase5_3_20261105T164707129182Z`.
Observed non-paint is verified background only inside the completed regions.
NoData, ambiguous boundaries, padding, and unsupported symbol-class footprints
with a two-native-pixel exclusion margin remain ignored. The model trains on
solid/dash visual IDs 1/2 mapped explicitly to model IDs 0/1. Approved arrow and
give-way outlines remain archived, with no unsupported class silently becoming
background. No original painted footprint was widened or redrawn.

The dataset contains 78 native crops across 512/384/256 sizes. The executed run
used ten 512 training crops from the LEFT/RIGHT connected component and five
512 validation crops from the INTERCHANGE component. Shared global identities
and a ten-native-unit footprint guard prevent cross-split leakage. Within each
split crops overlap; counts are appearances, not independent replicates. There
is one geographic component per split and no separate test group.

`ValidMask2FormerCriterion` implements deterministic, population-weighted
foreground/background sampling on valid native pixels. Hungarian matching,
BCE, Dice and all decoder auxiliary losses exclude ignored locations. Loss
sampling maps directly to the model output grid without mixing ignored native
neighbors. Tests demonstrate unchanged matching/losses after perturbing ignored
logits or targets, zero ignored mask gradients, finite gradients, partial objects,
multiple instances, missing paint observations, padding and empty regions.

Classification is conservative: unmatched queries receive no global no-object
class label on partially valid images. Complete-region coverage and one-to-one
matching instead justify null-mask BCE on valid pixels for unmatched queries.
Queries over excluded symbols or unobserved pixels therefore receive no loss
from those pixels. Fully valid images retain ordinary no-object class CE.
This custom training criterion is separate from the saved model architecture.

Actual warm-start chain:

1. Official pinned COCO Swin-S initialization, two foreground head rows replaced,
   original no-object row retained; no random backbone initialization.
2. Eight-step native-512 technical pilot with finite CUDA forward/backward and
   optimizer updates. Initial default FP16 scaling overflowed and was retained
   as a failed attempt; scale 128 succeeded.
3. A bounded 100-step run visiting all ten available 512 training crops.
4. An 80-step valid-null-mask refinement after its supervision tests passed.

Total: **188 successful optimizer steps**. AdamW, LR 2e-5, weight decay .01,
FP16, microbatch 1, accumulation 1, gradient clipping 1, seed 895. The backbone
was not frozen. Gradient checkpointing was unsupported; native 512 fit on the
RTX 3050 with about 2.50 GiB peak tensor allocation. Smaller fallback crops were
not required. Optimizer moments reset for each bounded warm-start run and are
not saved as resumable optimizer state.

The final trained checkpoint is:

```text
D:\Plotline-AI\models\mask2former_swin_s\phase5_3_valid_null_pilot_native512_80steps\pilot_checkpoint
```

The sibling `processor`, `training_config.json`, `loss_config.json`, model card,
dataset manifest, source provenance, software locks, metrics and recognition
artifacts describe its reproducible scope. Fresh-process reload reproduces
class and mask logits exactly under the recorded deterministic runtime flags.

Actual command-line interfaces, in the activated environment:

```powershell
# Example new bounded run; choose a new destination, never replace a checkpoint.
python scripts/phase5_3_train_worker.py --dataset data/processed/phase5_training/phase5_3_20261105T164707129182Z --initialization models/mask2former_swin_s/phase5_3_valid_null_pilot_native512_80steps/pilot_checkpoint --output models/mask2former_swin_s/new_version --size 512 --steps 8
python scripts/phase5_3_reload.py models/mask2former_swin_s/new_version
python scripts/phase5_3_evaluate.py models/mask2former_swin_s/new_version
```

The installed instance postprocessor uses a hard-coded intermediate 384 grid.
The final inference path instead maps output logits directly to native crop
dimensions. Original RGB inputs are never overwritten with overlays. Predicted
pixel masks, scores, classes, local prediction IDs, tile IDs and affine/world
context are saved in each validation-crop folder. No final GIS vector or 3D
digitization is performed.

All five available native-512 validation crops were evaluated. Final mean
per-crop foreground IoU on valid pixels is **0.4462**. At mask IoU .5 and score
floor .25, solid precision/recall are **0.0448 / 0.5000**; dash precision/recall
are **0.2500 / 0.6538**. There are 115 false-positive and 12 missed instance
appearances, including 47 duplicate detections. These numbers include overlapping
validation crops. Per-crop confidence-truncated AP is reported separately from
pooled full-survey COCO AP. Training-preview results are not independent accuracy.
Validation informed engineering development; no untouched test accuracy is claimed.

The spatial negative-mask refinement improved mean IoU from .3669 to .4462 and
reduced false-positive appearances from 226 to 115, but precision remains too low
for deployment. Additional geographic diversity, duplicate suppression, calibrated
negative-query behavior and a defensible test set remain necessary. Approved
symbol outlines can be used in a later documented class-mapping migration.

Final checks: **290 passed, 1 skipped**, Ruff passed, Black passed for 159 files,
and nine environment health checks passed with the optional network config
check skipped. The skipped model-config network check does not replace the
actual official-weight initialization, CUDA training and checkpoint reload
already executed here. See `outputs/reports/phase5_3/report.md` and `report.json`.
