# Phase 3.3: conservative native-grid model inputs

Run `python scripts/run_phase3_3.py` in the existing `roadmarking-ai` environment.
The runner reads the latest clean-intensity report and nine source raster windows,
plus their existing Phase 3.1 broad search, strict and uncertain masks. It never
opens LAZ files, downloads weights, trains a model or modifies previous products.
Each execution creates a new timestamped directory under `data/processed/phase3_3`.

`road_focused_roi` unions the broad search with strict candidates expanded by a
configurable four native-coordinate-unit margin and uncertain observations.
Without verified non-road exclusions, all observations outside that evidence are
retained for review. The selector rejects only tiles with zero observations;
black measurements, faint paint, junctions and parallel roads remain eligible.
Provisional overlapping masks are unioned; disagreements in raw or normalized
intensity overlaps are errors. This cannot promise background removal where
road identity has not been established.

The benchmark compares the current nine 1024/256 model inputs with globally
aligned 512/128 crops over exactly the same source-pixel union. Context previews
are not model inputs. Source identity plus global row/column gives deterministic
IDs and unique-pixel retention accounting. Source windows are stitched without
resampling and padded outside their union. Separate valid and observed masks
distinguish padding, missing returns and measured dark data. Raw maximum, mean,
normalized intensity and point counts remain in their original GeoTIFFs with
explicit source-window references. No source layers are silently substituted.

Each selected tile has exact grayscale/RGB PNGs, worldfiles, observed/valid TIFFs
and metadata. Normalization remains shared per original source (p1/p99.5),
with the existing fixed gamma 0.7 display transfer. Diagnostic resized panels
and selection outlines live only in `quality_control`; model PNGs have no overlays.
Uncertain CRS and physical units stay unconfirmed. Future predictions can map
through the recorded affine and source grid and merge on global pixel keys;
overlap confidence aggregation remains a future inference-stage concern.

The installed processor is tested with `do_resize=False`; its default shortest
edge resize would change native pixel scale. Processor padding masks do not
replace the supplied valid/observed masks. No model inference measurement is
possible without local checkpoint weights. CPU timings and image dimensions
are not evidence of GPU speed or recognition accuracy.

Measured findings and A–N acceptance discussion:
[report](../outputs/reports/phase3_3/report.md),
[machine-readable results](../outputs/reports/phase3_3/report.json),
[latest output location](../outputs/reports/phase3_3/latest.json).
Status remains provisional pending approved roadway boundaries, marking review,
verified aligned annotations and actual model latency/accuracy evaluation.
Keep the existing 1024 source tiling until those checks justify a change.

Phase 4's original full-layer raster manifest stays intact. The new image crop
manifests provide all `REQUIRED_LAYERS` through native-resolution VRT references
and an explicit all-zero filled mask (no interpolation). Every virtual layer
is read back against the original crop and each manifest passes the existing
Phase 4 metadata preflight. No full-precision source raster is duplicated.
Training remains blocked by the existing source/annotation alignment problem.
