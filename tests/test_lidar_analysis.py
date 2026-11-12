"""Tests for the Phase-2 LiDAR analysis utilities (synthetic data, exact expectations)."""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np
import pytest

from roadmarking.analysis.lidar_stats import (
    GridAccumulator,
    GridSpec,
    collect_file_stats,
    density_summary,
    hist_percentiles,
    intensity_summary,
)
from roadmarking.analysis.sampling import (
    Window,
    bev_raster,
    fill_for_display,
    nn_spacing,
    read_windows,
    robust_scale,
)
from roadmarking.io.pointcloud import (
    ChunkReadReport,
    iter_laz_chunks,
    laz_chunk_size,
    point_validity,
)
from roadmarking.rasterization.rasterizer import pixel_to_world


def _write_grid_las(path: Path, spacing: float = 0.1, size: float = 10.0) -> np.ndarray:
    """Regular grid of points with known spacing; intensity encodes the x column."""
    g = np.arange(0, size, spacing) + spacing / 2
    xx, yy = np.meshgrid(g, g)
    x, y = xx.ravel() + 1000.0, yy.ravel() + 2000.0
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([1000.0, 2000.0, 0.0])
    las = laspy.LasData(header)
    las.x, las.y, las.z = x, y, np.full(x.size, 5.0)
    las.intensity = (np.floor((x - 1000.0) / 1.0) * 100).astype(np.uint16)
    las.return_number = np.ones(x.size, np.uint8)
    las.number_of_returns = np.ones(x.size, np.uint8)
    las.classification = np.where(x < 1005.0, 2, 1).astype(np.uint8)
    las.point_source_id = np.where(y < 2005.0, 7, 8).astype(np.uint16)
    las.write(path)
    return np.column_stack([x, y])


def test_gridspec_covering_and_index() -> None:
    spec = GridSpec.covering(np.array([10.2, 20.7, 0]), np.array([12.9, 23.1, 1]), 1.0)
    assert (spec.x_min, spec.y_max, spec.nx, spec.ny) == (10.0, 24.0, 3, 4)
    row, col = spec.index(np.array([10.2, 12.9]), np.array([23.1, 20.7]))
    assert row.tolist() == [0, 3] and col.tolist() == [0, 2]
    cx, cy = spec.cell_center(np.array([0]), np.array([0]))
    assert (cx[0], cy[0]) == (10.5, 23.5)
    inside = spec.inside(np.array([10.2, 9.9, 13.1, 11.0]), np.array([23.1, 22.0, 22.0, 19.5]))
    assert inside.tolist() == [True, False, False, False]
    with pytest.raises(ValueError):
        GridSpec.covering(np.zeros(3), np.ones(3), 0)


def test_hist_percentiles_exact() -> None:
    data = np.arange(1, 101)  # 1..100
    hist = np.bincount(data, minlength=200)
    p = hist_percentiles(hist, [1, 50, 99, 100])
    assert p == {"p1": 1, "p50": 50, "p99": 99, "p100": 100}


def test_intensity_summary_zero_and_saturation() -> None:
    vals = np.array([0, 0, 10, 20, 65535, 65535, 65535, 30, 40, 50], dtype=np.int64)
    s = intensity_summary(np.bincount(vals, minlength=65536))
    assert s["zero_count"] == 2 and s["max"] == 65535
    assert s["saturated_pct"] == 30.0
    s2 = intensity_summary(np.bincount(np.array([1, 2, 3, 900]), minlength=65536))
    assert s2["saturated_pct"] == 0.0 and s2["at_max_count"] == 1


def test_grid_accumulator_and_density(tmp_path: Path) -> None:
    path = tmp_path / "grid.las"
    _write_grid_las(path, spacing=0.1)  # 100 pts per m²
    st = collect_file_stats(path, grid_cell=1.0)
    assert st.point_count == 10_000 and st.read_error is None
    assert st.grid is not None
    d = density_summary(st.grid)
    assert d["pts_per_m2"]["p50"] == pytest.approx(100.0)
    assert d["implied_spacing_m_at_median"] == pytest.approx(0.1)
    assert st.class_counts[2] == 5000 and st.class_counts[1] == 5000
    assert st.point_source_counts == {7: 5000, 8: 5000}
    assert int(st.grid.single_return.sum()) == 10_000
    assert np.nanmax(st.grid.z_range()) == pytest.approx(0.0, abs=1e-3)


def test_density_summary_respects_mask() -> None:
    acc = GridAccumulator(GridSpec(0.0, 2.0, 1.0, 2, 2))
    acc.add(
        np.array([0.5, 0.5, 1.5]), np.array([1.5, 1.5, 0.5]), np.zeros(3), np.ones(3), np.ones(3)
    )
    mask = np.array([[True, False], [False, False]])
    assert density_summary(acc, mask)["pts_per_m2"]["p50"] == 2.0
    assert density_summary(acc)["cells"] == 2


def test_truncated_laz_is_flagged(tmp_path: Path) -> None:
    src = tmp_path / "ok.laz"
    _write_grid_las(src, spacing=0.05)  # 40 000 points
    data = src.read_bytes()
    bad = tmp_path / "truncated.laz"
    bad.write_bytes(data[: len(data) // 2])
    st = collect_file_stats(bad, grid_cell=1.0)
    assert st.read_error is not None
    assert st.point_count < 40_000


def test_chunk_table_reader_reads_all_points(tmp_path: Path) -> None:
    path = tmp_path / "multi_chunk.laz"
    xy = _write_grid_las(path, spacing=0.04)  # 62 500 points -> 2 LAZ chunks
    assert laz_chunk_size(path) == 50_000
    report = ChunkReadReport(path, 0, 0)
    chunks = list(iter_laz_chunks(path, report))
    assert [len(c) for c in chunks] == [50_000, 12_500]
    assert report.n_chunks == 2 and report.points_read == len(xy)
    assert report.bad_chunks == [] and report.corrupt_chunks == [] and report.dropped_points == 0
    got = np.column_stack(
        [np.concatenate([c.x for c in chunks]), np.concatenate([c.y for c in chunks])]
    )
    assert np.allclose(got, xy, atol=1e-6)


def test_corrupt_chunk_is_cut_and_other_chunks_survive(tmp_path: Path) -> None:
    """Overwrite bytes inside chunk 1 of 3: chunks 0 and 2 must be read completely and
    nothing decoded from the damaged region may be returned."""
    import struct

    import lazrs

    path = tmp_path / "three_chunks.laz"
    xy = _write_grid_las(path, spacing=0.025)  # 160 000 points -> chunks of 50k
    with laspy.open(path) as r:
        hdr = r.header
        vlr = lazrs.LazVlr(next(v for v in hdr.vlrs if v.user_id == "laszip encoded").record_data)
    with path.open("rb") as fh:
        fh.seek(hdr.offset_to_point_data)
        table = lazrs.read_chunk_table(fh, vlr)
    start1 = hdr.offset_to_point_data + 8 + table[0][1]
    data = bytearray(path.read_bytes())
    pos = start1 + table[1][1] // 3
    data[pos : pos + 256] = bytes((i * 37 + 11) % 256 for i in range(256))
    bad = tmp_path / "corrupt.laz"
    bad.write_bytes(bytes(data))
    assert struct.unpack_from("<q", data, hdr.offset_to_point_data)[0] > 0

    report = ChunkReadReport(bad, 0, 0)
    chunks = list(iter_laz_chunks(bad, report))
    assert report.corrupt_chunks or report.bad_chunks
    assert all(c[0] == 1 for c in report.corrupt_chunks + report.bad_chunks)
    assert report.points_read + report.dropped_points == len(xy)
    got = np.column_stack(
        [np.concatenate([c.x for c in chunks]), np.concatenate([c.y for c in chunks])]
    )
    from scipy.spatial import cKDTree

    dist, _ = cKDTree(xy).query(got)
    assert dist.max() <= 1e-3  # every returned point is an original one
    assert len(got) >= 100_000  # chunks 0 and 2 intact


def test_point_validity_rules(tmp_path: Path) -> None:
    path = tmp_path / "v.las"
    _write_grid_las(path, spacing=1.0)
    las = laspy.read(path)
    pts = las.points
    pts.return_number[0] = 2  # rn > nr
    pts.gps_time[1] = 700_000.0  # beyond one GPS week
    ok = point_validity(pts, las.header.mins, las.header.maxs, gps_week_time=True)
    assert not ok[0] and not ok[1] and ok[2:].all()
    ok2 = point_validity(pts, las.header.mins, las.header.maxs, gps_week_time=False)
    assert ok2[1]


def test_chunk_size_is_none_for_uncompressed(tmp_path: Path) -> None:
    path = tmp_path / "plain.las"
    _write_grid_las(path, spacing=0.5)
    assert laz_chunk_size(path) is None
    assert sum(len(c) for c in iter_laz_chunks(path)) == 400


def test_nn_spacing_regular_grid() -> None:
    g = np.arange(0, 5, 0.25)
    xx, yy = np.meshgrid(g, g)
    s = nn_spacing(xx.ravel(), yy.ravel())
    assert s["nn_m"]["p50"] == pytest.approx(0.25)
    dup = nn_spacing(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0]))
    assert dup["coincident_xy_pct"] == pytest.approx(66.667, abs=1e-2)


def test_read_windows_and_bev_georeferencing(tmp_path: Path) -> None:
    path = tmp_path / "grid.laz"
    _write_grid_las(path, spacing=0.1)
    w = Window("w", 1002.0, 2002.0, 1004.0, 2004.0)
    pts = read_windows(path, [w])["w"]
    assert pts["x"].size == 400  # 2 m x 2 m at 100 pts/m²
    assert set(np.unique(pts["point_source_id"])) == {7}
    r = bev_raster(pts["x"], pts["y"], pts["z"], pts["intensity"], w, gsd=0.1)
    assert r.shape == (20, 20) and not r.nodata.any() and r.count.sum() == 400
    xs, ys = pixel_to_world(r.transform, np.array([0]), np.array([0]))
    assert xs[0] == pytest.approx(1002.05) and ys[0] == pytest.approx(2003.95)
    # intensity encodes x in 1 m bands: column 0 (x=1002.05) -> 200, column 19 -> 300
    assert r.i_max[0, 0] == 200 and r.i_max[0, 19] == 300
    r2 = bev_raster(pts["x"], pts["y"], pts["z"], pts["intensity"], w, gsd=0.05)
    assert r2.nodata.mean() == pytest.approx(0.75)  # finer than spacing -> gaps observed


def test_fill_for_display_marks_filled_pixels() -> None:
    img = np.array([[1.0, np.nan, np.nan, np.nan], [1.0, 1.0, 1.0, 1.0]])
    filled, mask = fill_for_display(img, max_distance_px=1)
    assert mask.tolist() == [[False, True, True, True], [False] * 4]
    assert not np.isnan(filled).any()
    far = np.full((1, 5), np.nan)
    far[0, 0] = 2.0
    f2, m2 = fill_for_display(far, max_distance_px=1)
    assert m2.tolist() == [[False, True, False, False, False]] and np.isnan(f2[0, 4])


def test_robust_scale_keeps_nan() -> None:
    img = np.array([0.0, 50.0, 100.0, np.nan])
    s = robust_scale(img, 0, 100)
    assert s[0] == 0 and s[2] == 1 and np.isnan(s[3])
