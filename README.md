# Plotline-AI

**AI road-marking digitization from aerial LiDAR.**
Plotline-AI takes a raw LiDAR point cloud (`.las` / `.laz`) and produces a single GeoPackage
containing every painted road marking as accurate **3D vector geometry** (lane lines as
polylines, arrows, hatching and other area marks as polygons), ready to load into any GIS.

![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-ee4c2c)
![Model](https://img.shields.io/badge/model-Mask2Former%20Swin--S-6f42c1)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20NVIDIA%20GPU-0078d6)
![Tests](https://img.shields.io/badge/tests-509%20passed-brightgreen)

---

## Why

Road-asset inventories need every lane line, arrow and hatch bar drawn as GIS vectors.
Doing that by hand from LiDAR is slow: an operator traces each stripe over an intensity
image, then fixes the heights. Plotline-AI automates the whole chain, from point cloud to
reviewed-quality 3D GeoPackage, in about **4 minutes for a 35-million-point file**.

## Results

Measured against client hand-digitized data on held-out survey tracks:

| Metric | Result |
|---|---|
| Recognition F1, independent acceptance set (56 unseen crops) | **0.923** solid lines, **0.905** dashed lines (previous model: 0.794 / 0.846) |
| Client line length covered | **92.6 – 99.1 %** per track |
| Client area marks found (dense scans) | **124 / 125** on the largest track |
| Area-mark shape match (median IoU) | **0.88 – 0.95** |
| Line offset from painted-stripe centre (median) | **0.4 – 0.5 cm** |
| Heights vs client 3D lines | within a few **millimetres** |

## How it works

```
 LAS / LAZ ─► Validate & read ─► Intensity raster ─► AI recognition ─► Digitize ─► 3D recovery ─► Self-check ─► GeoPackage
 (1+ passes)  (streamed, CRS)    (georeferenced     (Mask2Former     (trace,      (Z from the     (13 QA        (Highway lines,
                                  overlapping tiles)  Swin-S, GPU)     bridge gaps,  road surface)   checks)       Other Marks polygons,
                                                                       fit polygons)                               run metadata)
```

1. **Read**: streams one or more overlapping survey passes (laspy / PDAL), checks CRS
   consistency and corrupt LAZ chunks.
2. **Rasterize**: builds calibrated bird's-eye-view intensity images on a georeferenced grid
   with overlapping tiles, so every pixel maps back to world coordinates.
3. **Recognize**: a fine-tuned **Mask2Former (Swin-Small)** instance-segmentation model finds
   solid and dashed longitudinal markings. Tiles are stitched back into one consistent result.
4. **Digitize**: traces lines along the paint, joins lines across gaps where vehicles hid the
   road (and only there), splits dashes, and fits polygons to area marks (arrows,
   chevrons, hatch bars, give-way frames), using client templates where they match.
5. **Centre on paint**: every 25 cm the brightness profile across each line is measured, and
   the line is moved to the midpoint of the stripe's edges, then smoothed.
6. **Recover 3D**: heights are taken from the LiDAR road surface and only interpolated
   between measured points, never beyond them.
7. **Self-check**: 13 automatic checks (off-centre lines, ends running past the paint,
   overlapping lines, polygons not covering paint, …) clean up or flag problems before export.
8. **Export**: a validated 3D GeoPackage with the client's attribute schema, plus a metadata
   table recording source files, model hash and SHA-256 checksums.

## Output

| Layer | Geometry | Content |
|---|---|---|
| `Highway` | MultiLineString Z | Lane lines on the paint axis. Solid lines are continuous; each dash is its own feature. |
| `Other Marks` | MultiPolygon Z | Arrows, hatch bars, chevrons, colour bands, give-way and framed marks. |
| `plotline_run_metadata` | table | Input files and hashes, model and hash, CRS, fields left for manual completion. |

Every feature is marked as automatically digitized and pending review, so it fits a normal
QA workflow.

## Desktop application

Plotline-AI ships as a Windows desktop app for GIS operators. No scripting is needed:

1. **Browse** for a LAS/LAZ file (and optionally add overlapping passes of the same road,
   which raised line coverage from 36–86 % to 96–100 % on review areas).
2. **Save output as** a `.gpkg`.
3. **Start processing** and follow the progress through each stage. The job can be cancelled
   safely, and input files are never modified.

It is distributed as a self-contained installer (bundled Python, CUDA PyTorch, GDAL, PDAL)
that installs per user without admin rights, and as a plugin for a LiDAR viewer.
See the [User Guide](docs/Plotline-AI_User_Guide.md).

## Tech stack

- **Deep learning**: PyTorch (CUDA), Hugging Face Transformers, Mask2Former Swin-S, timm, albumentations
- **Point clouds & GIS**: laspy / lazrs, PDAL, GDAL, rasterio, GeoPandas, Shapely, pyproj
- **Image & geometry processing**: NumPy, SciPy, scikit-image, OpenCV
- **App & packaging**: Tkinter GUI, Inno Setup installer
- **Quality**: pytest (500+ tests), ruff, black, hash-locked model registry and app baselines

## Repository layout

```
src/roadmarking/
  io/               LAS/LAZ and GIS readers/writers
  preprocessing/    road-surface extraction, noise and intensity calibration
  rasterization/    georeferenced intensity rasters and tiling
  models/           Mask2Former wrapper and production model registry
  training/         fine-tuning pipeline
  inference/        tiled inference, calibration and stitching
  digitization/     line tracing, gap bridging, polygon fitting, 3D surface, QA, export
  app/              desktop pipeline, GUI, paint centring and refinement
  gis/ evaluation/ quality/ ...
src/plotline_ai/    application entry point (python -m plotline_ai)
configs/            pipeline, model, class and GIS-schema configs, model registry, release locks
scripts/            data preparation, training, evaluation and digitization tools
tests/              pytest suite, including regression cases from field review
packaging/          installer and plugin build scripts
plugin/             LiDAR-viewer plugin
docs/               user guide, architecture and per-phase design notes
```

## Getting started (developers)

Requirements: Windows, Python 3.11, an NVIDIA GPU with a current driver, and
[micromamba](https://mamba.readthedocs.io/) or conda.

```powershell
micromamba create -y -f environment.yml
micromamba run -n roadmarking-ai python -m pip install -r requirements-ml.txt -c constraints-conda.txt
micromamba run -n roadmarking-ai python -m pip install -e . --no-deps
```

Check the environment and GPU, then run the tests:

```powershell
micromamba run -n roadmarking-ai python -m roadmarking.healthcheck
micromamba run -n roadmarking-ai python -m pytest
```

Run the app (`Launch_Plotline_AI.bat` does the same) or the command-line pipeline:

```powershell
micromamba run -n roadmarking-ai python -m plotline_ai
micromamba run -n roadmarking-ai python scripts/digitize.py --help
```

> **Note:** the trained model weights and the survey data are not included in this
> repository (client data, and weights are too large for git). The code, configuration and
> tests are complete; to run inference you need your own fine-tuned checkpoint registered in
> [`configs/production_model.yaml`](configs/production_model.yaml).

More detail: [architecture](docs/architecture.md) ·
[development notes and history](docs/development.md) · [setup](docs/setup.md) ·
[training](docs/phase5_mask2former_training.md)

## Limitations

- The recognition model knows two line classes (solid, dashed). Road text, pedestrian
  crossings and stop lines are not produced yet.
- Some client attribute fields (section codes, line subtypes) cannot be derived from the
  point cloud and are left for manual completion.
- Validated on one survey campaign; other sensors or regions may need additional fine-tuning.

## Author

Built by [@mr-liki](https://github.com/mr-liki).
