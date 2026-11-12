# Setup — Windows 11 (native), NVIDIA GPU

This is how the environment was built and verified on the development machine
(Windows 11 Pro build 26300, x86-64, RTX 3050 6 GB, driver 610.88). It runs natively on
Windows; WSL2 was not needed because every dependency, including python-pdal, is available
for win-64 on conda-forge.

## Prerequisites

* NVIDIA driver supporting CUDA ≥ 13.0. `nvidia-smi` reports "CUDA UMD Version" 13.3 here.
  **No CUDA toolkit is needed**: the PyTorch wheels bundle the CUDA runtime.
* Git.
* ~15 GB free disk for the environment, plus space for data and the HF cache.
* No administrator rights are needed. Nothing goes into system Python.

## 1. Install micromamba (user-level, one time)

```powershell
$d = "$env:USERPROFILE\micromamba"
New-Item -ItemType Directory -Force $d | Out-Null
Invoke-WebRequest "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-win-64" -OutFile "$d\micromamba.exe"
& "$d\micromamba.exe" --version     # 2.10.0 used here
```

Set the root prefix in every new shell (or permanently in your user env vars):

```powershell
$env:MAMBA_ROOT_PREFIX = "$env:USERPROFILE\micromamba"
$mm = "$env:USERPROFILE\micromamba\micromamba.exe"
```

## 2. Create the conda-forge environment

```powershell
cd D:\Plotline-AI
& $mm create -y -f environment.yml
```

For a bit-for-bit reproduction, use the explicit lock instead:

```powershell
& $mm create -y -n roadmarking-ai -f environment.lock.txt
```

## 3. Install the ML stack with pip

```powershell
& $mm run -n roadmarking-ai python -m pip install -r requirements-ml.txt -c constraints-conda.txt
& $mm run -n roadmarking-ai python -m pip install -e . --no-deps
```

`constraints-conda.txt` must be passed. It stops pip from replacing conda builds of numpy,
scipy, pillow, shapely and **pyarrow**.

> **Why pyarrow matters on Windows.** GDAL's Parquet driver links to conda's `libarrow`.
> If pip installs its own `pyarrow` wheel (for example as a dependency of `datasets`), the
> two Arrow builds clash. Python then crashes with a dialog saying
> *"Entry Point Not Found … parquet.dll"*. `tests/test_environment.py::
> test_arrow_shared_by_gdal_and_pyarrow` guards against this. For the same reason,
> `datasets` is pinned to 5.0.1 (5.1 needs pyarrow ≥ 24, but GDAL 3.12 uses Arrow 23).

## 4. Activate

```powershell
& $mm shell hook -s powershell | Out-String | Invoke-Expression
micromamba activate roadmarking-ai
```

> **Always run Python through the activated environment** (`micromamba activate
> roadmarking-ai`, or prefix commands with `micromamba run -n roadmarking-ai`). Calling
> `...\envs\roadmarking-ai\python.exe` directly leaves `Library\bin` off `PATH`.
> conda-forge's BLAS loads a further DLL lazily through `PATH`, so the first numpy/scipy
> linear-algebra call then crashes the process (`0xC06D007F`). The health check
> `linear_algebra` and `tests/test_environment.py::test_linear_algebra_backend` detect this.
>
> **BLAS must be OpenBLAS (fixed 2026-11-04).** PyTorch's wheel ships Intel OpenMP
> (`libiomp5md.dll`), while conda-forge MKL 2026 uses LLVM OpenMP (`libomp.dll`). With MKL,
> any process using both torch and numpy/scipy linear algebra failed, in either order
> (`OMP: Error #15`, or `WinError 127` loading `torch\lib\shm.dll`). `environment.yml` now
> pins `libblas`/`libcblas`/`liblapack` to their `*openblas` builds (OpenBLAS 0.3.34,
> pthreads). An environment created before this fix can be repaired in place:
>
> ```powershell
> & $mm install -n roadmarking-ai -c conda-forge "libblas=*=*openblas" "libcblas=*=*openblas" "liblapack=*=*openblas"
> ```
>
> `tests/test_environment.py::test_torch_and_numpy_linalg_share_one_openmp` guards this.
> Do **not** use `KMP_DUPLICATE_LIB_OK=TRUE`; it is documented as unsafe.

## 5. Verify

```powershell
python scripts/check_environment.py      # OS, Python env, GPU (+ GPU matmul vs CPU)
python scripts/check_dependencies.py     # imports, versions, pip check
python -m roadmarking.healthcheck --model-config
python -m pytest
```

Optional: the full model smoke test downloads about 280 MB of weights into the HF cache
(`HF_HOME`, default `%USERPROFILE%\.cache\huggingface`). Check the weight license first:

```powershell
python scripts/smoke_test_model.py --download
```

## 6. Optional settings

* `HF_HUB_DISABLE_SYMLINKS_WARNING=1` turns off the symlink warning. Enabling Windows
  Developer Mode makes the HF cache use symlinks.
* `ROADMARKING_DATA_ROOT` keeps data/outputs on another drive.
* `ROADMARKING_DEVICE=cpu` is for debugging only. CPU is far too slow for real workloads.

## Updating dependencies

1. Change `environment.yml` or `requirements-ml.txt` (pin versions deliberately).
2. Rebuild, run the full verification, then regenerate the locks:

```powershell
& $mm env export -n roadmarking-ai --explicit | Set-Content -Encoding utf8 environment.lock.txt
& $mm run -n roadmarking-ai python -m pip freeze --exclude-editable | Set-Content -Encoding utf8 requirements-lock.txt
```
