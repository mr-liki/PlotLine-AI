"""Environment health checks (Python env, GPU, LiDAR, GIS, CV, model integration).

Each check returns a :class:`CheckResult`; nothing is reported as passing unless it was
executed. Synthetic data is used only to verify library functionality, never to claim the
real aerial-LiDAR workflow works.

Usage::

    python -m roadmarking.healthcheck            # all offline checks
    python -m roadmarking.healthcheck --model-config   # also fetch model config (network)
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

EXPECTED_ENV_NAME = "roadmarking-ai"

# (import name, distribution name, essential)
PACKAGES: list[tuple[str, str, bool]] = [
    ("numpy", "numpy", True),
    ("scipy", "scipy", True),
    ("pandas", "pandas", True),
    ("PIL", "pillow", True),
    ("yaml", "pyyaml", True),
    ("tqdm", "tqdm", True),
    ("psutil", "psutil", True),
    ("packaging", "packaging", True),
    ("torch", "torch", True),
    ("torchvision", "torchvision", True),
    ("transformers", "transformers", True),
    ("accelerate", "accelerate", True),
    ("safetensors", "safetensors", True),
    ("huggingface_hub", "huggingface_hub", True),
    ("timm", "timm", True),
    ("einops", "einops", True),
    ("cv2", "opencv-python-headless", True),
    ("skimage", "scikit-image", True),
    ("laspy", "laspy", True),
    ("lazrs", "lazrs", True),
    ("pdal", "pdal", False),
    ("osgeo.gdal", "gdal", True),
    ("rasterio", "rasterio", True),
    ("geopandas", "geopandas", True),
    ("shapely", "shapely", True),
    ("pyproj", "pyproj", True),
    ("pyogrio", "pyogrio", True),
    ("affine", "affine", True),
    ("datasets", "datasets", False),
    ("pycocotools", "pycocotools", False),
    ("torchmetrics", "torchmetrics", False),
    ("tensorboard", "tensorboard", False),
    ("albumentations", "albumentations", False),
    ("pytest", "pytest", False),
    ("pytest_cov", "pytest-cov", False),
    ("black", "black", False),
    ("dotenv", "python-dotenv", False),
    ("ipykernel", "ipykernel", False),
]
CLI_TOOLS: list[str] = ["ruff", "pdal"]


def _version_line(text: str, tool: str) -> str:
    """Return the first output line mentioning ``tool`` (e.g. 'pdal 2.10.0 (git-version: …)')."""
    for line in text.splitlines():
        if tool in line.lower():
            return line.strip()
    return text.strip()


@dataclass
class CheckResult:
    name: str
    status: str  # "pass" | "fail" | "skip" | "warn"
    details: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("pass", "skip", "warn")


def _run(name: str, fn: Callable[[], dict[str, Any]]) -> CheckResult:
    try:
        details = fn()
        status = details.pop("_status", "pass")
        return CheckResult(name, status, details, details.pop("_message", ""))
    except Exception as exc:  # report every failure explicitly
        return CheckResult(name, "fail", {}, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- checks
def check_python() -> dict[str, Any]:
    prefix = Path(sys.prefix)
    env_name = os.environ.get("CONDA_DEFAULT_ENV") or prefix.name
    in_env = prefix.name == EXPECTED_ENV_NAME or env_name == EXPECTED_ENV_NAME
    out = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "env_name": env_name,
        "isolated_env": in_env,
    }
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"Python 3.11 required, got {platform.python_version()}")
    if not in_env:
        raise RuntimeError(f"Not running inside '{EXPECTED_ENV_NAME}' (prefix {prefix})")
    return out


def check_packages() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {}
    failed_essential: list[str] = []
    failed_optional: list[str] = []
    for mod, dist, essential in PACKAGES:
        try:
            m = importlib.import_module(mod)
            try:
                versions[dist] = version(dist)
            except PackageNotFoundError:
                versions[dist] = str(getattr(m, "__version__", "unknown"))
        except Exception as exc:
            (failed_essential if essential else failed_optional).append(f"{dist}: {exc}")
    # Command-line tools that have no importable Python module.
    for tool in CLI_TOOLS:
        try:
            res = subprocess.run([tool, "--version"], capture_output=True, text=True, timeout=60)
            versions[tool] = _version_line(res.stdout, tool) if res.returncode == 0 else "error"
            if res.returncode != 0:
                failed_optional.append(f"{tool}: {res.stderr.strip()}")
        except FileNotFoundError:
            failed_optional.append(f"{tool}: not on PATH")
    pip = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    out: dict[str, Any] = {
        "versions": versions,
        "failed_essential": failed_essential,
        "failed_optional": failed_optional,
        "pip_check": pip.stdout.strip() or pip.stderr.strip(),
    }
    if failed_essential or pip.returncode != 0:
        raise RuntimeError(f"essential import failures={failed_essential}; pip check={pip.stdout}")
    if failed_optional:
        out["_status"] = "warn"
    return out


def check_gpu() -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        raise RuntimeError(f"torch.cuda.is_available() is False ({out})")
    props = torch.cuda.get_device_properties(0)
    out.update(
        device_name=props.name,
        total_memory_gb=round(props.total_memory / 1024**3, 2),
        compute_capability=f"{props.major}.{props.minor}",
        cudnn=torch.backends.cudnn.version(),
        bf16_supported=torch.cuda.is_bf16_supported(),
    )
    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    gpu = (a @ b).cpu()
    cpu = a.cpu() @ b.cpu()
    torch.cuda.synchronize()
    max_err = float((gpu - cpu).abs().max())
    out["matmul_max_abs_err_vs_cpu"] = max_err
    if max_err > 1e-1:
        raise RuntimeError(f"GPU matmul disagrees with CPU (max err {max_err})")
    return out


def check_lidar() -> dict[str, Any]:
    import laspy
    import numpy as np

    from roadmarking.io.pointcloud import read_pointcloud

    rng = np.random.default_rng(0)
    n = 1000
    scales = np.array([0.001, 0.001, 0.001])
    offsets = np.array([500000.0, 4000000.0, 100.0])
    xyz = offsets + rng.uniform(0, 50, size=(n, 3))
    xyz = np.round((xyz - offsets) / scales) * scales + offsets  # representable values
    intensity = rng.integers(0, 65535, size=n, dtype=np.uint16)

    out: dict[str, Any] = {"note": "synthetic data; validates libraries only"}
    with tempfile.TemporaryDirectory() as tmp:
        for ext in ("las", "laz"):
            header = laspy.LasHeader(point_format=6, version="1.4")
            header.scales, header.offsets = scales, offsets
            las = laspy.LasData(header)
            las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            las.intensity = intensity
            path = Path(tmp) / f"synthetic.{ext}"
            las.write(path)
            pc = read_pointcloud(path)
            assert np.allclose(pc.xyz, xyz, atol=1e-6), f"{ext}: coordinates changed"
            assert np.array_equal(pc.intensity, intensity), f"{ext}: intensity changed"
            assert np.allclose(pc.header.scales, scales) and np.allclose(pc.header.offsets, offsets)
            out[ext] = "write/read OK"
        out["laz_backends"] = [str(b) for b in laspy.LazBackend.detect_available()]

    try:
        import pdal

        out["python_pdal"] = pdal.__version__
    except Exception as exc:
        out["python_pdal"] = f"unavailable: {exc}"
    try:
        res = subprocess.run(["pdal", "--version"], capture_output=True, text=True, timeout=60)
        out["pdal_cli"] = _version_line(res.stdout, "pdal") if res.returncode == 0 else res.stderr
    except FileNotFoundError:
        out["pdal_cli"] = "not on PATH"
    return out


def check_geospatial() -> dict[str, Any]:
    import geopandas as gpd
    import numpy as np
    import rasterio
    from osgeo import gdal
    from pyproj import Transformer
    from rasterio.transform import from_origin
    from shapely.geometry import LineString, Polygon

    out: dict[str, Any] = {"note": "synthetic data; validates libraries only"}
    t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = t.transform(10.0, 50.0)
    lon, lat = t.transform(x, y, direction="INVERSE")
    assert abs(lon - 10.0) < 1e-9 and abs(lat - 50.0) < 1e-9
    out["crs_roundtrip"] = "EPSG:4326 <-> EPSG:3857 OK (test CRS only, not a project assumption)"

    line = LineString([(0, 0, 10.5), (5, 1, 10.7), (10, 0, 10.9)])
    poly = Polygon(
        [(0, 0, 1.0), (2, 0, 1.1), (2, 2, 1.2), (0, 2, 1.3)],
        holes=[[(0.5, 0.5, 1.15), (1.0, 0.5, 1.15), (1.0, 1.0, 1.15)]],
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "synthetic.gpkg"
        gpd.GeoDataFrame({"ID_SIG": ["a"]}, geometry=[line], crs="EPSG:3857").to_file(
            path, layer="lines", driver="GPKG", engine="pyogrio"
        )
        gpd.GeoDataFrame({"ID_SIG": ["b"]}, geometry=[poly], crs="EPSG:3857").to_file(
            path, layer="polys", driver="GPKG", engine="pyogrio"
        )
        rl = gpd.read_file(path, layer="lines").geometry.iloc[0]
        rp = gpd.read_file(path, layer="polys").geometry.iloc[0]
        assert rl.has_z and list(rl.coords) == list(line.coords), "LineString Z lost"
        assert rp.has_z and rp.equals_exact(poly, 1e-12), "Polygon Z/holes lost"
        assert len(rp.interiors) == 1
        out["gpkg_3d_roundtrip"] = "LineStringZ + PolygonZ (with hole) OK"

        tif = Path(tmp) / "synthetic.tif"
        arr = np.arange(100, dtype=np.uint16).reshape(10, 10)
        transform = from_origin(1000.0, 2000.0, 0.05, 0.05)
        with rasterio.open(
            tif,
            "w",
            driver="GTiff",
            width=10,
            height=10,
            count=1,
            dtype="uint16",
            crs="EPSG:3857",
            transform=transform,
        ) as dst:
            dst.write(arr, 1)
        with rasterio.open(tif) as src:
            assert src.transform == transform and np.array_equal(src.read(1), arr)
            assert src.xy(0, 0) == (1000.025, 1999.975)
        out["georaster_roundtrip"] = "GeoTIFF transform + pixel centre OK"
    out["gdal"] = gdal.__version__
    out["gpkg_driver"] = gdal.GetDriverByName("GPKG") is not None
    return out


def check_linalg() -> dict[str, Any]:
    """numpy/scipy linear algebra in a subprocess (a BLAS DLL failure kills the process).

    conda-forge BLAS on Windows resolves a delay-loaded DLL through PATH; running the env's
    python.exe without activating the environment crashes with 0xC06D007F.
    """
    code = (
        "import numpy as np, scipy.linalg as sl;"
        "a = np.eye(3) * 2.0; np.linalg.inv(a); sl.inv(a); np.linalg.svd(a); print('ok')"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    if res.returncode != 0 or "ok" not in res.stdout:
        raise RuntimeError(
            f"numpy/scipy linear algebra crashed (exit {res.returncode & 0xFFFFFFFF:#x}). "
            "Run Python through the activated environment "
            "(`micromamba activate roadmarking-ai` or `micromamba run -n roadmarking-ai ...`)."
        )
    return {"blas_lapack": "numpy.linalg + scipy.linalg OK"}


def check_cv() -> dict[str, Any]:
    import cv2
    import numpy as np
    from PIL import Image
    from skimage.morphology import skeletonize

    img = np.zeros((64, 64), np.uint8)
    cv2.rectangle(img, (10, 28), (54, 36), 255, -1)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) == 1
    skel = skeletonize(img > 0)
    assert 0 < skel.sum() < (img > 0).sum()
    pil = Image.fromarray(img).convert("RGB")
    assert pil.size == (64, 64) and np.asarray(pil).shape == (64, 64, 3)
    return {"opencv": cv2.__version__, "contours": len(contours), "skeleton_px": int(skel.sum())}


def check_model_imports() -> dict[str, Any]:
    import transformers
    from transformers import (
        AutoImageProcessor,
        Mask2FormerConfig,
        Mask2FormerForUniversalSegmentation,
    )

    from roadmarking.models.mask2former import DEFAULT_CHECKPOINT

    assert AutoImageProcessor and Mask2FormerForUniversalSegmentation and Mask2FormerConfig
    return {
        "transformers": transformers.__version__,
        "checkpoint": DEFAULT_CHECKPOINT,
        "level": "classes importable (no network, no weights)",
    }


def check_model_config() -> dict[str, Any]:
    from roadmarking.models.mask2former import DEFAULT_CHECKPOINT, load_config_only

    cfg = load_config_only(DEFAULT_CHECKPOINT)
    return {
        "level": "config.json downloaded (no weights)",
        "model_type": cfg.model_type,
        "num_labels": cfg.num_labels,
        "backbone": cfg.backbone_config.model_type,
    }


def check_project_config() -> dict[str, Any]:
    from roadmarking.config import EXPECTED_CHECKPOINT, load_config

    cfg = load_config()
    assert cfg.checkpoint == EXPECTED_CHECKPOINT, cfg.checkpoint
    return {
        "config": str(cfg.config_path),
        "root": str(cfg.root),
        "device": cfg.device,
        "provisional_sections": cfg.provisional_sections(),
    }


# --------------------------------------------------------------------------- runner
def run_all(include_model_config: bool = False) -> list[CheckResult]:
    checks: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("python_environment", check_python),
        ("packages", check_packages),
        ("project_config", check_project_config),
        ("gpu", check_gpu),
        ("lidar", check_lidar),
        ("geospatial", check_geospatial),
        ("linear_algebra", check_linalg),
        ("computer_vision", check_cv),
        ("model_imports", check_model_imports),
    ]
    if include_model_config:
        checks.append(("model_config", check_model_config))
    results = [_run(n, f) for n, f in checks]
    if not include_model_config:
        results.append(CheckResult("model_config", "skip", {}, "use --model-config (network)"))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-config",
        action="store_true",
        help="download the model config.json from the Hub (no weights)",
    )
    parser.add_argument("--json", type=Path, help="also write results to this JSON file")
    args = parser.parse_args(argv)

    results = run_all(include_model_config=args.model_config)
    for r in results:
        print(f"[{r.status.upper():4}] {r.name}{' - ' + r.message if r.message else ''}")
        for k, v in r.details.items():
            if k != "versions":
                print(f"         {k}: {v}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps([asdict(r) for r in results], indent=2, default=str), encoding="utf-8"
        )
    failed = [r.name for r in results if not r.ok]
    print(
        f"\n{len(results) - len(failed)}/{len(results)} checks OK"
        + (f"; FAILED: {failed}" if failed else "")
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
