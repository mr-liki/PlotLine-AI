# Dependencies — installed and verified versions (2026-11-04)

Source of truth for exact builds: `environment.lock.txt` (conda) and
`requirements-lock.txt` (pip).

## Version-selection rationale

* **torch 2.14.1 + cu130 / torchvision 0.29.1 + cu130.** torchvision 0.29.x requires
  torch ≥ 2.14. The CUDA 13.0 wheels need a driver supporting ≥ 13.0, and this one reports
  13.3. torch 2.15.0 (cu132 only) is newer, but it was passed over for the more established
  2.14 patch release.
* **transformers 5.22.1.** Needs torch ≥ 2.6 and huggingface_hub ≥ 1.31,<3. It still ships
  `Mask2FormerForUniversalSegmentation` and loads the checkpoint config.
* **datasets 5.0.1, not 5.1.0.** 5.1.0 requires pyarrow ≥ 24, but conda-forge GDAL 3.12.3
  is built against Arrow 23. A pip pyarrow alongside conda's libarrow crashes Python on
  Windows (`parquet.dll` entry point). datasets 5.0.1 also caps huggingface_hub < 2.0, so
  hub is 1.33.0. That is still within transformers' range.
* **opencv-python-headless** instead of `opencv-python`. albumentations depends on the
  headless build, and having two cv2 builds causes conflicts.
* **numpy 2.4.6** from conda-forge, held by `constraints-conda.txt`.
* **BLAS/LAPACK = OpenBLAS 0.3.34 (pthreads), not MKL** (changed 2026-11-04).
  conda-forge MKL 2026 depends on LLVM OpenMP, which clashes with the Intel OpenMP runtime
  bundled in the PyTorch wheel: torch together with numpy/scipy linear algebra failed in one
  process. OpenBLAS builds have no OpenMP runtime, so there's no clash.

## Installed versions

| Group | Package | Version | Source | Status |
|---|---|---|---|---|
| Core | python | 3.11.18 | conda-forge | ✅ |
| | numpy | 2.4.6 | conda-forge | ✅ |
| | scipy | 1.17.1 | conda-forge | ✅ |
| | pandas | 3.1.0 | conda-forge | ✅ |
| | pillow | 13.0.0 | conda-forge | ✅ |
| | pyyaml / tqdm / psutil / packaging | 6.0.3 / 4.70.1 / 7.2.2 / 26.3 | conda-forge | ✅ |
| DL / CV | torch | 2.14.1+cu130 | download.pytorch.org | ✅ GPU verified |
| | torchvision | 0.29.1+cu130 | download.pytorch.org | ✅ (CUDA NMS op tested) |
| | transformers | 5.22.1 | PyPI | ✅ |
| | accelerate | 1.15.0 | PyPI | ✅ |
| | safetensors | 0.8.0 | PyPI | ✅ |
| | huggingface_hub | 1.33.0 | PyPI | ✅ |
| | timm | 1.0.31 | PyPI | ✅ |
| | einops | 0.8.2 | PyPI | ✅ |
| | opencv-python-headless | 5.0.0.93 | PyPI | ✅ |
| | scikit-image | 0.26.0 | conda-forge | ✅ |
| LiDAR | laspy | 2.7.0 | conda-forge | ✅ LAS + LAZ round trip |
| | lazrs | 0.8.2 | conda-forge | ✅ (Lazrs, LazrsParallel) |
| | PDAL CLI | 2.10.0 | conda-forge | ✅ |
| | python-pdal | 3.5.3 | conda-forge | ✅ reads LAZ |
| GIS | gdal | 3.12.3 | conda-forge | ✅ GPKG driver |
| | rasterio | 1.4.4 | conda-forge | ✅ GeoTIFF round trip |
| | geopandas | 1.2.0 | conda-forge | ✅ |
| | shapely | 2.2.0 | conda-forge | ✅ 3D geometry |
| | pyproj | 3.7.2 | conda-forge | ✅ |
| | pyogrio | 0.12.1 | conda-forge | ✅ |
| | affine | 3.0.1 | conda-forge | ✅ |
| Training support | datasets | 5.0.1 | PyPI | ✅ |
| | pyarrow | 23.0.1 | conda-forge | ✅ (shared with GDAL) |
| | pycocotools | 2.0.11 | conda-forge | ✅ |
| | torchmetrics | 1.9.0 | PyPI | ✅ |
| | tensorboard | 2.21.0 | PyPI | ✅ |
| | albumentations | 2.0.8 | PyPI | ✅ |
| Dev | pytest / pytest-cov | 9.2.0 / 7.1.0 | conda-forge | ✅ |
| | ruff / black | 0.17.3 / 26.10.1 | conda-forge | ✅ |
| | python-dotenv / ipykernel | 1.2.4 / 7.4.0 | conda-forge | ✅ |

No package failed. No optional package had to be deferred. `pip check`: *No broken
requirements found.*

## Not installed (on purpose)

* CUDA toolkit / nvcc: not needed, since the PyTorch wheels bundle the runtime. Custom CUDA
  kernels (e.g. Mask2Former's optional MSDeformAttn) would need it, but Transformers
  provides a pure-PyTorch path.
* Web frameworks, databases, cloud SDKs, UI libraries: out of scope.
