"""Environment tests: interpreter, packages, GPU, CV stack, model integration."""

from __future__ import annotations

import os

import pytest

from roadmarking import healthcheck as hc


def test_python_environment() -> None:
    info = hc.check_python()
    assert info["python"].startswith("3.11")
    assert info["isolated_env"]


def test_essential_packages_import_and_pip_check() -> None:
    info = hc.check_packages()
    assert info["failed_essential"] == []


@pytest.mark.gpu
def test_gpu_tensor_ops() -> None:
    import torch

    if not torch.cuda.is_available() and os.environ.get("ROADMARKING_ALLOW_NO_GPU") == "1":
        pytest.skip("CUDA unavailable and ROADMARKING_ALLOW_NO_GPU=1")
    info = hc.check_gpu()  # raises (test fails) when CUDA is unavailable
    assert info["cuda_available"]
    assert info["total_memory_gb"] > 0


def test_torch_torchvision_compatible() -> None:
    import torch
    import torchvision
    from torchvision.ops import nms

    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=torch.float)
    keep = nms(boxes, torch.tensor([0.9, 0.8, 0.7]), 0.5)
    assert keep.tolist() == [0, 2]
    assert torchvision.__version__.split("+")[0]


def test_arrow_shared_by_gdal_and_pyarrow(tmp_path) -> None:
    """GDAL (conda libarrow) and pyarrow/datasets must share one Arrow build.

    Run in a subprocess: a mismatched pip pyarrow wheel crashes the interpreter on Windows
    with "Entry Point Not Found ... parquet.dll" instead of raising an exception.
    """
    import subprocess
    import sys

    code = (
        "from osgeo import gdal, ogr; import pyogrio, geopandas;"
        "import pyarrow as pa, pyarrow.parquet as pq, datasets;"
        f"p = r'{tmp_path / 't.parquet'}';"
        "pq.write_table(pa.table({'a': [1, 2]}), p);"
        "assert pq.read_table(p).num_rows == 2;"
        "assert ogr.GetDriverByName('Parquet') is not None"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr


def test_linear_algebra_backend() -> None:
    """Fails (instead of crashing pytest) when the env is used without activation."""
    assert "OK" in hc.check_linalg()["blas_lapack"]


@pytest.mark.parametrize("torch_first", [True, False], ids=["torch_first", "linalg_first"])
def test_torch_and_numpy_linalg_share_one_openmp(torch_first: bool) -> None:
    """torch and numpy/scipy linear algebra must coexist in one process, in either order.

    With conda-forge MKL (LLVM OpenMP) next to PyTorch's bundled Intel OpenMP this aborted
    with ``OMP: Error #15`` or failed to load ``torch\\lib\\shm.dll``. The environment uses
    OpenBLAS builds of libblas/liblapack to avoid it. Fresh subprocess: a crash must fail
    this test, not kill pytest.
    """
    import subprocess
    import sys

    linalg = (
        "import numpy as np, scipy.linalg as sl; a = np.random.rand(64, 64);"
        "np.linalg.inv(a); sl.eigh(a @ a.T);"
    )
    torch_part = "import torch; torch.linalg.inv(torch.rand(8, 8) + torch.eye(8) * 8);"
    code = (torch_part + linalg if torch_first else linalg + torch_part) + "print('ok')"
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert res.returncode == 0 and "ok" in res.stdout, res.stderr[-2000:]


def test_computer_vision_stack() -> None:
    info = hc.check_cv()
    assert info["contours"] == 1


def test_mask2former_classes_available() -> None:
    info = hc.check_model_imports()
    assert info["checkpoint"] == "facebook/mask2former-swin-small-coco-instance"


def test_resolve_device_does_not_silently_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from roadmarking.models.mask2former import ModelLoadError, resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ModelLoadError):
        resolve_device("cuda:0", require_gpu=True)
    assert resolve_device("cuda:0", require_gpu=False).type == "cpu"


@pytest.mark.network
def test_model_config_download() -> None:
    if os.environ.get("ROADMARKING_RUN_NETWORK_TESTS") != "1":
        pytest.skip("set ROADMARKING_RUN_NETWORK_TESTS=1 to fetch config.json from the Hub")
    info = hc.check_model_config()
    assert info["model_type"] == "mask2former"
