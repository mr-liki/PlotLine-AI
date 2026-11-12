"""LAS/LAZ I/O tests using tiny synthetic files (library verification only).

These do not validate the real aerial-LiDAR workflow; that requires client data.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import laspy
import numpy as np
import pytest
from pyproj import CRS

from roadmarking.io.pointcloud import PointCloudReadError, read_header, read_pointcloud

SCALES = np.array([0.001, 0.001, 0.001])
OFFSETS = np.array([500_000.0, 4_000_000.0, 100.0])


def _write_synthetic(
    path: Path, n: int = 500, crs: CRS | None = None
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    xyz = OFFSETS + rng.uniform(0, 25, size=(n, 3))
    xyz = np.round((xyz - OFFSETS) / SCALES) * SCALES + OFFSETS
    intensity = rng.integers(0, 65535, size=n, dtype=np.uint16)
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales, header.offsets = SCALES, OFFSETS
    if crs is not None:
        header.add_crs(crs)
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.intensity = intensity
    las.classification = np.full(n, 2, dtype=np.uint8)
    las.write(path)
    return xyz, intensity


@pytest.mark.parametrize("ext", ["las", "laz"])
def test_roundtrip_preserves_xyz_intensity_scale_offset(tmp_path: Path, ext: str) -> None:
    path = tmp_path / f"synthetic.{ext}"
    xyz, intensity = _write_synthetic(path)
    pc = read_pointcloud(path)
    assert len(pc) == len(xyz)
    assert pc.xyz.dtype == np.float64
    assert np.allclose(pc.xyz, xyz, atol=1e-6)
    assert np.array_equal(pc.intensity, intensity)
    assert np.allclose(pc.header.scales, SCALES)
    assert np.allclose(pc.header.offsets, OFFSETS)
    assert pc.classification is not None and set(np.unique(pc.classification)) == {2}


def test_laz_backend_available() -> None:
    backends = laspy.LazBackend.detect_available()
    assert backends, "no LAZ backend (lazrs) available"


def test_crs_preserved_in_header(tmp_path: Path) -> None:
    # Arbitrary projected CRS for the test only; no project EPSG is assumed.
    crs = CRS.from_epsg(32629)
    path = tmp_path / "with_crs.las"
    _write_synthetic(path, n=10, crs=crs)
    hdr = read_header(path)
    assert hdr.crs_wkt is not None
    assert CRS.from_wkt(hdr.crs_wkt).equals(crs)
    assert hdr.point_count == 10


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(PointCloudReadError):
        read_pointcloud(tmp_path / "missing.laz")


def test_python_pdal_reads_las(tmp_path: Path) -> None:
    pdal = pytest.importorskip("pdal", reason="python-pdal bindings not installed")
    path = tmp_path / "pdal.laz"
    xyz, intensity = _write_synthetic(path, n=200)
    pipeline = pdal.Reader.las(filename=str(path)).pipeline()
    count = pipeline.execute()
    arr = pipeline.arrays[0]
    assert count == 200
    assert np.allclose(arr["X"], xyz[:, 0], atol=1e-6)
    assert np.array_equal(arr["Intensity"], intensity)


def test_pdal_cli_available() -> None:
    if shutil.which("pdal") is None:
        pytest.skip("pdal CLI not on PATH (activate the roadmarking-ai environment)")
    import subprocess

    res = subprocess.run(["pdal", "--version"], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0 and "pdal" in res.stdout.lower()
