# Architecture

## Design principles

* **Aerial LiDAR first.** The client spec mentions Mobile Laser Scanning, but the real data
  is aerial/drone LiDAR. Expect lower and uneven point density, intensity varying with
  range and incidence angle across flight lines, and overlapping strips. Every density,
  resolution and threshold value stays provisional until real data is inspected.
* **Original coordinates are sacred.** Preprocessing selects points by index masks and
  never rewrites XYZ. Rasters carry an `affine.Affine` transform, so any pixel maps back to
  source-CRS world coordinates. Z comes from real road-surface points (Stage 8).
* **Network classes ≠ GIS classes.** The network predicts a small set of visual classes
  (`configs/classes.yaml: model_classes`). The client designations (14 longitudinal types,
  COD_RST codes, letters/digits) are resolved from those plus measurements, road context
  and catalog rules (`rule_families`).
* **Fail loudly.** No silent CPU fallback, no placeholder algorithms that produce output.
  Stages not built yet raise `NotImplementedError`.

## Stages and modules

| # | Stage | Module | Current state |
|---|---|---|---|
| 1 | Point-cloud input (LAS/LAZ, CRS, metadata) | `io/pointcloud.py` | **Implemented**: headers, in-memory reader, Phase-2 recovery reader and strict Phase-3 chunk iterator |
| 2 | Road surface, noise, intensity normalization | `preprocessing/road_surface.py`, `preprocessing/intensity.py` | **Implemented (Phase 3)**: geometry candidates with ROI/confidence states; shared percentile calibration |
| 3 | BEV intensity raster, adaptive GSD, tiles | `rasterization/rasterizer.py`, `rasterization/tiler.py` | **Implemented (Phase 3)**: sparse observed rasters, masks, conservative optional fill, globally aligned overlapping tiles |
| 4 | Mask2Former Swin-S instance segmentation | `models/mask2former.py`, `inference/predictor.py`, `training/` | Model loader **implemented** (COCO head, eval mode); training/inference are interfaces |
| 5 | Mask cleanup, tile merging | `postprocessing/mask_refinement.py` | Interface only |
| 6 | Centerlines (skeleton, curve fit, dashes) / polygons with holes | `vectorization/lines.py`, `vectorization/polygons.py` | Interface only |
| 7 | GIS classification, attributes, grouping, historical IDs | `gis/attributes.py` | Schemas + validator **implemented**; rules not implemented |
| 8 | Z recovery from road-surface points | `gis/reconstruction_3d.py` | Interface only |
| 9 | 3D GeoPackage export | `gis/export.py`, `io/gis.py` | **Implemented**: schema-validated write, refuses to overwrite by default |

Cross-cutting: `config.py` (YAML loading, includes, env overrides, validation),
`utils/logging.py` (structured / JSON logging), `healthcheck.py`.

## Data flow contracts

```
PointCloud(xyz float64 (N,3), intensity uint16, classification, header{scales, offsets, crs_wkt})
  └─► GeoRaster(image, transform: Affine, crs_wkt, nodata)
        └─► TileWindow(row_off, col_off, h, w, transform)
              └─► InstancePrediction(mask, class_id, score, tile)
                    └─► shapely LineString / Polygon (world XY)
                          └─► LineStringZ / PolygonZ  ─► GeoDataFrame validated by LayerSchema ─► GPKG
```

## Output layers (`configs/gis_schema.yaml`)

* `longitudinal_marks`, LineStringZ: ID_SIG, POSICAO, DESIGNACAO, COD_JAE, LARGURA, TRAÇO,
  ESPAÇO, ID_GRUPO, COMENTARIO, ID_HIST, OBS. Dashes of a broken line are separate
  features that share an ID_GRUPO.
* `other_marks`, PolygonZ (holes allowed): ID_SIG, POSICAO, COD_RST, COD_JAE, ALGARISMO,
  LETRA, LARGURA, ALTURA, AREA_PINTADA, ID_GRUPO, COMENTARIO, ID_HIST, OBS. Each inscription
  character is its own polygon, grouped by ID_GRUPO.

## Model

* Checkpoint `facebook/mask2former-swin-small-coco-instance` (80 COCO classes), used only
  as pretrained initialization.
* The road-marking head (from `model_classes`) is set at fine-tuning time
  (`id2label`/`label2id`, `ignore_mismatched_sizes=True`).
* Intensity rasters are single-channel. Copying to 3 channels versus adding derived
  channels (e.g. height-above-road) is a Phase-5 decision.
* License tag `other`: must be verified before commercial deployment.

## Phase 3 product contract

The strict runner permits only the two verified intact sources and never recovers
corrupt chunks. Geometry candidates are 0 rejected, 1 uncertain, 2 ROI-confirmed smooth
surface. Original coordinates, intensity and source indices are retained. Unknown CRS
stays unset; affine native-grid mapping is preserved. Shared per-source percentile
normalization precedes tiling. Blocks retain nine aligned layers and tiles are exact
slices, with readback QA and SHA-256 completion manifests.

See [phase3_preprocessing.md](phase3_preprocessing.md) for formats, memory budgets,
CLI options and limitations. Phase 4 production readiness requires road-area review
and confirmation of the source/annotation coordinate relationship.
