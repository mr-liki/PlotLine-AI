# Phase 1 Setup Report — 2026-11-04

Every result below comes from commands run on this machine. Machine-readable results:
`outputs/reports/healthcheck.json` (git-ignored, regenerate with
`python -m roadmarking.healthcheck --model-config --json outputs/reports/healthcheck.json`).

## A. Environment

| | |
|---|---|
| Operating system | Windows 11 Pro 10.0.26300, AMD64 (i7-12700F, 64 GB RAM) |
| Free disk | C: 284 GB, D: 334 GB |
| Environment manager | micromamba 2.10.0, user-level (no admin) |
| Environment | `roadmarking-ai` at `%USERPROFILE%\micromamba\envs\roadmarking-ai` |
| Python | 3.11.18 (conda-forge); system Python 3.10 untouched |
| GPU | NVIDIA GeForce RTX 3050, 6.0 GB, compute 8.6, bf16 supported |
| Driver | 610.88 (CUDA UMD 13.3) |
| CUDA available | **Yes**. GPU matmul agrees with CPU (max abs err ≈1.3e-4, fp32/TF32) |
| PyTorch | 2.14.1+cu130 (CUDA runtime 13.0, cuDNN 9.24) |
| Transformers | 5.22.1 |

Pre-existing state: no conda/mamba, no CUDA toolkit, no PyTorch, WSL2 present but not
needed, git 2.56.

## B. Dependency status

* **Installed and verified:** all required packages in groups A–F (full table with
  versions in `docs/requirements.md`). `pip check`: no broken requirements.
* **Optional / deferred:** none deferred. python-pdal works natively on Windows.
* **Failed:** none.
* **Issue found and fixed during setup:** pip pulled in `pyarrow` 26 (via `datasets`), which
  clashed with conda GDAL's Arrow 23 DLLs. Python crashed with the
  *"Entry Point Not Found … parquet.dll"* dialog. Fix: pyarrow 23.0.1 from conda-forge,
  pinned in `constraints-conda.txt`, and `datasets` pinned to 5.0.1 (5.1 needs pyarrow ≥ 24).
  `huggingface_hub` went from 2.4.0 to 1.33.0 as a result, which is still inside
  transformers' supported range. A regression test was added.

## C. Model status

| Item | Status |
|---|---|
| Mask2Former integration configured (`models/mask2former.py`, `configs/model.yaml`) | ✅ |
| `AutoImageProcessor`, `Mask2FormerForUniversalSegmentation` importable | ✅ |
| Model **config** downloaded from the Hub (`config.json`, 80 COCO labels, Swin backbone) | ✅ |
| Pretrained **weights** downloaded | ❌ **Not downloaded** (waiting for your go-ahead; license `other`) |
| Model loaded on GPU | ❌ Not executed |
| Forward-pass test | ❌ Not executed (`python scripts/smoke_test_model.py --download`) |

The COCO checkpoint does not recognise road markings. No accuracy claims are made.

## D. Test results

* `python -m pytest`: **34 passed, 0 failed, 1 skipped**. The skipped test is the network
  test, enabled with `ROADMARKING_RUN_NETWORK_TESTS=1`. The same check ran successfully
  through the health check.
* `python -m roadmarking.healthcheck --model-config`: **9/9 checks OK**.
* `ruff check`: clean. `black --check`: clean.
* Coverage: 67% overall. The uncovered code is mostly interface modules that raise
  `NotImplementedError` by design.
* LAS/LAZ and GPKG tests use tiny synthetic data. They show the libraries work, **not** that
  the real aerial-LiDAR workflow does.

## E. Project structure

```
D:\Plotline-AI
├── README.md  pyproject.toml  environment.yml  environment.lock.txt
├── requirements-ml.txt  requirements-lock.txt  constraints-conda.txt
├── .gitignore  .env.example
├── configs/      default.yaml model.yaml classes.yaml gis_schema.yaml paths.yaml
├── data/         raw/{pointclouds,annotations,reference_gis}/ interim/ processed/
│                 training/ validation/ test/                    (git-ignored)
├── src/roadmarking/
│   ├── __init__.py config.py healthcheck.py py.typed
│   ├── io/              pointcloud.py gis.py
│   ├── preprocessing/   road_surface.py intensity.py
│   ├── rasterization/   rasterizer.py tiler.py
│   ├── models/          mask2former.py
│   ├── training/        dataset.py trainer.py
│   ├── inference/       predictor.py
│   ├── postprocessing/  mask_refinement.py
│   ├── vectorization/   lines.py polygons.py
│   ├── gis/             attributes.py reconstruction_3d.py export.py
│   └── utils/           logging.py
├── scripts/      check_environment.py check_dependencies.py smoke_test_model.py
├── tests/        test_environment.py test_config.py test_geospatial.py test_pointcloud_io.py
├── notebooks/  checkpoints/  logs/
├── outputs/      masks/ vectors/ gpkg/ reports/
└── docs/         architecture.md setup.md requirements.md implementation_roadmap.md setup_report.md
```

## F. Outstanding issues

1. **Pretrained weights not downloaded / forward pass not run.** Waiting for your approval
   (~280 MB to the HF cache, ~1 min).
2. **License:** the Hub license for `facebook/mask2former-swin-small-coco-instance` is
   `other`. Commercial-use permission must be confirmed before deployment.
3. **Provisional parameters:** CRS/vertical datum, GSD, tile size/overlap, thresholds,
   noise and Z-recovery parameters. All depend on inspecting real data.
4. **Class taxonomy:** the model classes and COD_RST meanings are provisional and need
   checking against the client catalog.
5. **6 GB VRAM** will constrain fine-tuning (small batches, fp16/bf16, gradient
   accumulation, maybe smaller tiles).
6. HF cache symlinks are off (Windows Developer Mode not enabled). This only costs some
   extra disk space.
7. One run of the health check printed a stray `{"log_history": []}` line after finishing.
   It did not reproduce in four more runs and is not printed by project code. No effect seen.
8. No git commit has been made yet (repository initialised on `main`).

## G. Next step

The essential checks passed: isolated env, all dependencies, GPU, LAS/LAZ, 3D GPKG, CV
stack, model integration (config level). **The project is ready for Phase 2 — LiDAR
inspection**, once client LAS/LAZ samples are in `data/raw/pointclouds/`. The optional
weight download and forward-pass smoke test can run any time after the license question is
acknowledged.
