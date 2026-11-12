# Implementation roadmap

Each phase ends with tests and a short report before the next one starts. Phase 1 (setup)
is complete.

| # | Phase | Main modules | Exit criteria |
|---|---|---|---|
| 1 | **Project setup** ✅ | env, configs, interfaces | Health checks + pytest green |
| 2 | **LiDAR inspection** | `io/pointcloud.py`, notebooks, `outputs/reports` | Per-file report: CRS/vertical datum, extents, point count & density (pts/m²), intensity histograms per flight line, classification codes, overlap. Decide `crs.*`, `ground_classification_codes`, chunk size. Add chunked/streamed reading. |
| 3 | **Point-cloud preprocessing** | `preprocessing/*` | Road-surface mask, noise removal and intensity normalization validated visually and against reference GIS. XYZ untouched. |
| 4 | **Intensity image rasterization** | `rasterization/*` | GSD chosen from density analysis. Georeferenced GeoTIFF tiles with overlap. Pixel↔world error < ½ pixel checked against reference vectors. |
| 5 | **Annotation conversion** | `training/dataset.py` | Existing polylines/polygons → instance masks on tiles. Lines buffered by measured width. Class mapping per `classes.yaml`. Train/val/test split by area (no spatial leakage). Visual QA sample. |
| 6 | **Mask2Former fine-tuning** | `models/`, `training/trainer.py` | New head from `model_classes`. Mixed precision + gradient accumulation for 6 GB VRAM. Reproducible seed/config logging, TensorBoard. Per-class AP/IoU on validation. **License verified first.** |
| 7 | **Inference** | `inference/predictor.py`, `postprocessing/*` | Tiled batched inference, tile-overlap merging, mask cleanup. Throughput measured on a full survey. |
| 8 | **Geometric vectorization** | `vectorization/*` | Centerlines (skeleton + curve fit, dashes kept separate), polygons with holes, simplification within tolerance, connectivity checks. |
| 9 | **3D GIS output** | `gis/*` | Classification rules (`rule_families`), widths/dash/gap measurements, ID_GRUPO, ID_HIST matching against reference GIS, Z recovery, schema-validated 3D GPKG. |
| 10 | **Accuracy optimization** | all | Geometric accuracy against manual digitization (Hausdorff/offset, completeness, correctness), error analysis, tuning, profiling large clouds. |

## Open questions to resolve with the client

* The exact semantics of each COD_RST code (M15–M21, M9a) and the catalog rules for the 14
  longitudinal designations.
* The CRS / vertical datum for the deliverable.
* Field types and value domains of the GIS attributes (from the reference GPKG/SHP).
* How historical IDs (ID_HIST) are matched when geometry changes.
* Commercial-use license for the pretrained Mask2Former weights.
