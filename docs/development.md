# AI Road Marking Digitization

> **Default model (2026-11-12): `v2_realistic_v1_fix_step0500`**, registered with hashes and a
> one-line rollback in [configs/production_model.yaml](../configs/production_model.yaml). Load it with
> `roadmarking.models.production.load_production_model()`. Independent acceptance: V1 F1 solid
> 0.923, dash 0.905 (previous model 0.794 / 0.846) on 56 unused real crops. Two classes only
> (solid, dash); arrows, text and crossings are not recognised.

Automated detection, classification and 3D digitization of road markings from **aerial /
drone LiDAR** point clouds, using **Mask2Former Swin-S** instance segmentation on
LiDAR-intensity images plus geometric vectorization, with output to 3D GeoPackage.

**Primary Phase 3 deliverable: [clean grayscale intensity images](phase3_intensity_images.md).**

Phase 3.3 adds [native-grid model-input optimization](phase3_3_model_input_optimization.md).
Run `python scripts/run_phase3_3.py` for the raster-only 512/1024 comparison.
The measured pilot retains 1024 source tiling; 512 increases workload on the current
regions. Model performance and annotation alignment remain unverified.
`python scripts/run_phase3.py` now exports road-focused images, wider context,
replicated-grayscale RGB inputs and exact-overlap mosaics, with no overlays.
The prior geometry workflow requires `--output-mode legacy`. See the
[image-output report](../outputs/reports/phase3_intensity_images/report.md).

> **Status: Phase 3 preprocessing implemented and executed on the two intact files.**
> See [Phase 3 documentation](phase3_preprocessing.md) and the
> [measured delivery report](../outputs/reports/phase3/phase3_report.md). Outputs retain
> explicit geometric uncertainty; confirmed road areas and source/annotation coordinate
> alignment are required before production Phase 4 annotation conversion.
> No road-marking model training or segmentation has been performed.

Phase 3.1 added an opt-in corridor refinement (`road_method="v2"`) that recovers mixed
road/barrier/kerb edge cells point-by-point and reports per-cell evidence categories.
Outputs are versioned under `data/processed/phase3_1/` and are unreviewed candidates. See
[Phase 3.1 documentation](phase3_1_road_corridor_correction.md) and the
[Phase 3.1 report](../outputs/reports/phase3_1/phase3_1_report.md).

The independent [alignment and cross-section rework](phase3_1_topodot_inspired.md)
separates the marking-search envelope from strict geometric surface candidates.
Run `scripts/run_phase3_1_topodot_inspired.py --mode propose` for the controlled pilot;
versioned results use `phase3_1_topodot_inspired/`. Automatic outputs remain provisional,
and full processing is gated on road-boundary and paint-retention review.

[Phase 3.2](phase3_2_line_geometry_correction.md) diagnoses the old reference
layers and adds finite evidence-based stripe tracking, explicit QA layers and
shared-coordinate tile continuity. Run `scripts/run_phase3_2.py` for its controlled
cached-point pilot; all new line products remain provisional until reviewed.

The new [verified annotation workflow](phase4_verified_annotations.md) prepares
a native-XY QGIS project directly from Phase 3.3. Run
`python scripts/prepare_annotation_project.py`; use the generated project to draw
and explicitly review paint. Current delivery: **ANNOTATION REQUIRED**, with no
real verified labels or training. Partial/NoData samples are excluded from stock
Mask2Former training until loss-validity handling is resolved.

Legacy Phase 4 tooling includes annotation QC, taxonomy, alignment gates, mask
conversion and spatial splits. Its run **stopped at the alignment gate**: the delivered
annotations (EPSG:3763, Y ≈ +170 km) do not overlap any LiDAR raster (no CRS,
Y ≈ −90 km). No training labels were generated. See
[Phase 4 documentation](phase4_dataset_preparation.md) and the
[Phase 4 report](../outputs/reports/phase4/phase4_report.md).

## 1. Project overview

| | |
|---|---|
| Input | LAS/LAZ aerial LiDAR (XYZ + intensity, CRS from headers), existing 2D/3D polylines and polygons, GPKG/SHP reference files |
| Output | 3D polylines (longitudinal marks), 3D polygons (other marks, with holes), client classification codes, GIS attributes, historical IDs, as 3D GeoPackage |
| Priorities | Accuracy → throughput on large clouds → minimal manual work → efficient GPU use → reproducibility → modular, debuggable code |

## 2. Final architecture

```
LAS/LAZ ─► 1 Input ─► 2 Preprocessing ─► 3 BEV intensity raster + tiles
        (laspy/PDAL)  (road surface,     (georeferenced, overlapping,
                       noise, intensity)  pixel↔world affine)
                                               │
   9 Export ◄─ 8 3D reconstruction ◄─ 7 GIS classification ◄─ 6 Vectorization ◄─ 5 Mask post-proc ◄─ 4 Mask2Former Swin-S
  (GPKG, Z)   (Z from road points)   (codes, IDs, widths)   (centerlines /      (cleanup, tile       (instance masks
                                                             polygons+holes)     merging)              + scores)
```

Model: `facebook/mask2former-swin-small-coco-instance` via Hugging Face
`AutoImageProcessor` + `Mask2FormerForUniversalSegmentation`. See
[docs/architecture.md](architecture.md).

## 3. Development environment (as installed and verified)

| Item | Value |
|---|---|
| OS | Windows 11 Pro (build 26300), x86-64 |
| GPU | NVIDIA GeForce RTX 3050, 6 GB, compute 8.6, driver 610.88 |
| Environment manager | micromamba 2.10.0 (user-level, `%USERPROFILE%\micromamba`) |
| Environment | `roadmarking-ai`, Python 3.11.18 |
| PyTorch | 2.14.1+cu130 (CUDA 13.0 runtime, cuDNN 9.24) |
| Transformers | 5.22.1 |

## 4. Required dependencies

* **conda-forge** ([environment.yml](../environment.yml)): Python, numpy, scipy, pandas, GDAL,
  PDAL + python-pdal, laspy + lazrs, rasterio, geopandas, shapely, pyproj, pyogrio, affine,
  scikit-image, pycocotools, pyarrow, and the dev tools.
* **pip** ([requirements-ml.txt](../requirements-ml.txt)): torch / torchvision (official CUDA
  wheels), transformers, accelerate, safetensors, huggingface_hub, timm, einops,
  opencv-python-headless, albumentations, datasets, torchmetrics, tensorboard.
* [constraints-conda.txt](../constraints-conda.txt) stops pip from replacing conda-built
  binaries (numpy, pyarrow, …).
* Exact versions: [environment.lock.txt](../environment.lock.txt) (conda, explicit) and
  [requirements-lock.txt](../requirements-lock.txt) (pip freeze). Full table:
  [docs/requirements.md](../requirements.md).

## 5. Installation

Full steps for this machine: [docs/setup.md](setup.md). Short version (PowerShell):

```powershell
$env:MAMBA_ROOT_PREFIX = "$env:USERPROFILE\micromamba"
$mm = "$env:USERPROFILE\micromamba\micromamba.exe"
& $mm create -y -f environment.yml
& $mm run -n roadmarking-ai python -m pip install -r requirements-ml.txt -c constraints-conda.txt
& $mm run -n roadmarking-ai python -m pip install -e . --no-deps
```

## 6. Activating the environment

```powershell
& "$env:USERPROFILE\micromamba\micromamba.exe" shell hook -s powershell | Out-String | Invoke-Expression
micromamba activate roadmarking-ai
```

Or prefix any command with `micromamba run -n roadmarking-ai`.

## 7. GPU verification

```powershell
python scripts/check_environment.py
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

GPU failure is reported as a failure; nothing silently falls back to CPU
(`runtime.require_gpu: true`).

## 8. Project structure

```
configs/            default.yaml, model.yaml, classes.yaml, gis_schema.yaml, paths.yaml
data/raw/           pointclouds/, annotations/, reference_gis/   (git-ignored)
data/               interim/, processed/, training/, validation/, test/
src/roadmarking/    config, healthcheck, io/, preprocessing/, rasterization/, models/,
                    training/, inference/, postprocessing/, vectorization/, gis/, utils/
scripts/            check_environment.py, check_dependencies.py, smoke_test_model.py
tests/              pytest suite
checkpoints/ outputs/{masks,vectors,gpkg,reports}/ logs/ notebooks/ docs/
```

## 9. Health checks

```powershell
python -m roadmarking.healthcheck                 # offline checks
python -m roadmarking.healthcheck --model-config  # + fetch model config.json (no weights)
python scripts/check_dependencies.py              # versions + pip check
python scripts/smoke_test_model.py --download     # downloads ~280 MB weights, forward pass
```

## 10. Tests

```powershell
python -m pytest                       # whole suite
python -m pytest --cov=roadmarking     # with coverage
$env:ROADMARKING_RUN_NETWORK_TESTS="1"; python -m pytest -m network
ruff check src tests scripts; black --check src tests scripts
```

## 11. Where to put client input data

| Data | Location |
|---|---|
| LAS/LAZ point clouds | `data/raw/pointclouds/` |
| Reference GPKG/SHP (existing digitization, historical IDs) | `data/raw/reference_gis/` |

Everything under `data/` is git-ignored. To keep large data on another drive, set
`ROADMARKING_DATA_ROOT` (see [.env.example](../.env.example)); every path in
`configs/paths.yaml` then resolves against it.

## 12. Where to put training annotations

Manually verified annotations (polylines/polygons or masks) go in `data/raw/annotations/`.
They are converted into image/mask pairs under `data/training|validation|test/` in Phase 4.

## 13. Known limitations

* Strict streaming I/O, geometric candidate extraction, shared intensity calibration,
  sparse rasterization and overlapping tiling are implemented. Later model/training and
  vectorization stages remain interfaces. Geometry candidates require road-area review.
* The COCO checkpoint does not detect road markings. Accuracy is unknown until fine-tuning.
* **License:** the Hub model card lists license `other`. Verify the pretrained-weight
  license and get commercial-use permission before deployment
  (`model.yaml: license.verified_for_commercial_use: false`).
* All processing parameters (CRS, GSD, tile size, thresholds, point density) are
  **provisional** until real client data is inspected.
* The class taxonomy and the meaning of each COD_RST code are provisional and must be
  confirmed against the client catalog.
* 6 GB VRAM limits training batch size and resolution. Expect gradient accumulation,
  mixed precision and modest tile sizes.
* Hugging Face cache symlinks are off on this Windows account (no Developer Mode), so
  cached files take some extra disk space. This is harmless.
* Synthetic LAS/GPKG tests only show that the libraries work, not that the real aerial-LiDAR
  workflow does.

## 14. Next phase

Phase 2 (LiDAR inspection) results: [docs/phase2_lidar_inspection.md](phase2_lidar_inspection.md).
Regenerate the data reports with:

```powershell
micromamba run -n roadmarking-ai python scripts/inspect_lidar.py stats --force
micromamba run -n roadmarking-ai python scripts/inspect_lidar.py report
```

Phase 3 has now been executed on `Track01_3` and `Track01_8` only. The damaged files
remain excluded. Reproduce the bounded representative run and complete runs with:

```powershell
micromamba run -n roadmarking-ai python scripts/run_phase3.py --smoke --previews
micromamba run -n roadmarking-ai python scripts/run_phase3.py --previews --skip-validated
```

Next: **Phase 4 - annotation conversion and alignment**, after road candidate review
and coordinate alignment confirmation. See [docs/phase3_preprocessing.md](phase3_preprocessing.md).
