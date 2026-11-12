"""Phase 3 contracts: source mapping, raised surfaces, exact raster bins and seams."""

import laspy
import numpy as np
import pytest
from affine import Affine
from shapely.geometry import box

from roadmarking.io.pointcloud import (
    PointCloud,
    PointCloudHeader,
    PointCloudReadError,
    StrictReadReport,
    iter_pointcloud_chunks,
)
from roadmarking.preprocessing.intensity import calibrate_histogram, normalize_intensity
from roadmarking.preprocessing.road_surface import RoadCriteria, SurfaceAccumulator
from roadmarking.rasterization.pipeline import BLOCK, read_mosaic, validate_products, write_tif
from roadmarking.rasterization.rasterizer import (
    NODATA,
    GeoRaster,
    RasterAccumulator,
    fill_isolated,
    north_up_transform,
    pixel_to_world,
    rasterize_intensity,
    world_to_pixel,
)
from roadmarking.rasterization.tiler import iter_tiles, sparse_tile_origins, tile_id


def cloud_from(xyz, raw=None):
    xyz = np.asarray(xyz, np.float64)
    header = PointCloudHeader(
        None,
        len(xyz),
        3,
        "1.2",
        np.full(3, 0.001),
        np.zeros(3),
        xyz.min(axis=0),
        xyz.max(axis=0),
        None,
    )
    return PointCloud(xyz, np.full(len(xyz), 1000, np.uint16) if raw is None else raw, None, header)


def surface(x0, x1, y0, y1, height, slope=0):
    x, y = np.meshgrid(np.arange(x0 + 0.1, x1, 0.2), np.arange(y0 + 0.1, y1, 0.2))
    return np.column_stack((x.ravel(), y.ravel(), (height + slope * x).ravel()))


@pytest.mark.parametrize("ext", ["las", "laz"])
@pytest.mark.parametrize("chunk_size", [1, 7, 100])
def test_strict_chunk_source_mapping(tmp_path, ext, chunk_size):
    path = tmp_path / f"source.{ext}"
    h = laspy.LasHeader(point_format=3, version="1.2")
    h.scales = [0.001] * 3
    h.offsets = [-58000, -118000, 30]
    las = laspy.LasData(h)
    las.x = -58000 + np.arange(19) * 0.051
    las.y = -118000 + np.arange(19) * 0.003
    las.z = 30 + np.arange(19) * 0.001
    las.intensity = np.arange(19, dtype=np.uint16) * 123
    las.return_number = np.ones(19, np.uint8)
    las.number_of_returns = np.ones(19, np.uint8)
    las.gps_time = np.full(19, 500.0)
    las.write(path)
    report = StrictReadReport()
    pieces = list(iter_pointcloud_chunks(path, chunk_size, report))
    xyz = np.concatenate([cloud.xyz for _, cloud in pieces])
    assert np.array_equal(xyz, np.column_stack((las.x, las.y, las.z)))
    assert [offset for offset, _ in pieces] == list(range(0, 19, chunk_size))
    assert report.points_read == 19 and report.invalid_points == 0
    assert report.warnings and np.array_equal(pieces[0][1].header.offsets, h.offsets)


@pytest.mark.parametrize("invalid", ["returns", "gps"])
def test_strict_rejects_invalid_without_recovery(tmp_path, invalid):
    las = laspy.LasData(laspy.LasHeader(point_format=3, version="1.2"))
    las.x = [1, 2]
    las.y = [2, 3]
    las.z = [3, 4]
    las.return_number = [1, 1]
    las.number_of_returns = [1, 1]
    las.gps_time = [100, 101]
    if invalid == "returns":
        las.return_number = [1, 2]
    else:
        las.gps_time = [100, np.nan]
    path = tmp_path / "bad.laz"
    las.write(path)
    report = StrictReadReport()
    with pytest.raises(PointCloudReadError, match="no recovery"):
        list(iter_pointcloud_chunks(path, 2, report))
    assert report.invalid_points == 1


@pytest.mark.parametrize("slope,height", [(0, 0), (0.08, 0), (0.04, 10)])
def test_road_geometry_raised_sloped_and_paint(slope, height):
    road = surface(0, 30, 0, 6, height, slope)
    lower_ground = surface(0, 30, 10, 16, -5)
    roof = surface(5, 7, 2, 4, height + 3, slope)
    xyz = np.vstack((road, lower_ground, roof))
    cloud = cloud_from(xyz)
    # Paint intensity changes have no effect on geometry.
    cloud.intensity[::2] = 65535
    acc = SurfaceAccumulator(cloud.header, RoadCriteria(min_component_m2=30))
    acc.add(xyz)
    model = acc.finish(box(-1, -1, 31, 7))
    state = model.classify(xyz)
    assert (state[: len(road)] == 2).mean() > 0.85
    assert not state[len(road) : len(road) + len(lower_ground)].any()
    assert not state[-len(roof) :].any()
    assert np.array_equal(cloud.xyz, xyz)
    acc2 = SurfaceAccumulator(cloud.header, RoadCriteria(min_component_m2=30))
    acc2.add(xyz)
    uncertain = acc2.finish().classify(xyz)
    assert not (uncertain == 2).any()
    assert (uncertain[len(road) : len(road) + len(lower_ground)] == 1).mean() > 0.9


def test_rough_vegetation_and_grid_budget():
    rng = np.random.default_rng(42)
    points = surface(0, 12, 0, 12, 0)
    points[:, 2] += rng.normal(0, 1, len(points))
    cloud = cloud_from(points)
    acc = SurfaceAccumulator(cloud.header, RoadCriteria(min_component_m2=10))
    acc.add(points)
    assert not acc.finish().classify(points).any()
    with pytest.raises(ValueError, match="memory budget"):
        SurfaceAccumulator(cloud.header, RoadCriteria(max_grid_cells=1))


def test_intensity_exact_histogram_saturation_and_gain():
    raw = np.array([0, 100, 100, 200, 65535], np.uint16)
    c = calibrate_histogram(np.bincount(raw, minlength=65536), 0, 100)
    assert c.statistics["saturated_points"] == 1
    assert c.statistics["saturated_pct"] == 20
    assert np.array_equal(raw, [0, 100, 100, 200, 65535])
    assert c.apply(raw).dtype == np.float32
    assert c.apply(raw)[0] == 0 and c.apply(raw)[-1] == 1
    assert normalize_intensity(np.full(5, 100)).max() == 0
    assert normalize_intensity(np.array([100, 101]), "minmax").max() < 0.002
    with pytest.raises(ValueError):
        calibrate_histogram(np.zeros(65536))


@pytest.mark.parametrize("origin", [(-58000.05, -118000.15), (0, 0), (500000, 4000000)])
def test_coordinate_centers_edges_and_negative_local(origin):
    t = north_up_transform(*origin, 0.05)
    c, r = np.array([-1, 0, 1, 1023, 1024]), np.array([-1, 0, 22, 1023, 2048])
    x, y = pixel_to_world(t, c, r)
    cc, rr = world_to_pixel(t, x, y)
    np.testing.assert_array_equal(cc, c)
    np.testing.assert_array_equal(rr, r)
    xx = origin[0] + np.array([-1, 0, 1, 1024]) * 0.05
    yy = origin[1] - np.array([-1, 0, 1, 1024]) * 0.05
    cc, rr = world_to_pixel(t, xx, yy)
    np.testing.assert_array_equal(cc, [-1, 0, 1, 1024])
    np.testing.assert_array_equal(rr, [-1, 0, 1, 1024])


def test_exact_aggregation_order_chunks_duplicates_nodata():
    c, r = np.array([0, 0, 1, 1, 0]), np.array([0, 0, 0, 1, 0])
    raw = np.array([0, 20, 65535, 9, 20], np.uint16)
    cal = calibrate_histogram(np.bincount(raw, minlength=65536))
    a, b = RasterAccumulator((3, 3)), RasterAccumulator((3, 3))
    a.add(c, r, raw)
    for ix in ([4, 2], [3, 0, 1]):
        b.add(c[ix], r[ix], raw[ix])
    for name, value in a.layers(cal).items():
        np.testing.assert_array_equal(value, b.layers(cal)[name])
    layers = a.layers(cal)
    assert layers["count"][0, 0] == 3
    assert layers["maximum"][0, 0] == 20
    assert layers["mean"][0, 0] == pytest.approx(40 / 3)
    assert layers["maximum"][2, 2] == -1 and layers["mean"][2, 2] == NODATA
    assert layers["count"].dtype == np.uint32 and layers["mean"].dtype == np.float32
    assert layers["observed"][2, 2] == 0 and layers["saturation"][0, 1] == 1


def test_low_intensity_zero_is_observed():
    cloud = cloud_from([[0, 0, 0], [0.01, 0.01, 0]], np.array([0, 0], np.uint16))
    result = rasterize_intensity(cloud, 0.05)
    assert (result.image == 0).sum() >= 1
    with pytest.raises(ValueError):
        rasterize_intensity(cloud, 0.05, "median")


def test_isolated_gap_and_large_gaps_and_paint_boundaries():
    image = np.full((9, 9), 0.2, np.float32)
    obs = np.ones((9, 9), bool)
    obs[2, 2] = False
    obs[5:7, 5:7] = False
    original = image.copy()
    image[~obs] = NODATA
    result, provenance = fill_isolated(image, obs, np.ones_like(obs))
    assert provenance.sum() == 1 and provenance[2, 2] == 1
    assert result[2, 2] == pytest.approx(0.2)
    assert (result[5:7, 5:7] == NODATA).all()
    np.testing.assert_array_equal(result[obs], original[obs])
    accepted = np.ones_like(obs)
    accepted[1, 1] = False
    assert not fill_isolated(image, obs, accepted)[1].any()
    image[1, 1] = 0.9
    assert not fill_isolated(image, obs, np.ones_like(obs))[1].any()
    assert not fill_isolated(image, obs, np.ones_like(obs), enabled=False)[1].any()


def test_fill_block_halo_matches_global():
    image = np.full((10, 12), 0.3, np.float32)
    obs = np.ones_like(image, bool)
    obs[4, 5] = False
    image[~obs] = NODATA
    global_fill, global_mask = fill_isolated(image, obs, np.ones_like(obs))
    local_fill, local_mask = fill_isolated(image[:, 4:9], obs[:, 4:9], np.ones_like(obs[:, 4:9]))
    np.testing.assert_array_equal(global_fill[:, 5:8], local_fill[:, 1:4])
    np.testing.assert_array_equal(global_mask[:, 5:8], local_mask[:, 1:4])


def test_tile_coverage_alignment_partial_and_determinism():
    raster = GeoRaster(np.zeros((1200, 1700)), north_up_transform(-50, -100, 0.05), None)
    windows = list(iter_tiles(raster, 1024, 256))
    coverage = np.zeros(raster.image.shape, bool)
    for w in windows:
        coverage[w.row_off : w.row_off + w.height, w.col_off : w.col_off + w.width] = True
        assert w.transform == raster.transform @ Affine.translation(w.col_off, w.row_off)
    assert coverage.all()
    assert windows[1].col_off == 768
    assert windows[-1].width == 164 and windows[-1].height == 432
    blocks = {(0, 0), (10, 10), (10, 11)}
    origins = sparse_tile_origins(blocks)
    assert origins == sparse_tile_origins(reversed(sorted(blocks)))
    assert len(origins) < 20
    assert tile_id("s", 768, 1536) == tile_id("s", 768, 1536)
    with pytest.raises(ValueError):
        list(iter_tiles(raster, 256, 256))


def test_mosaic_readback_and_overlap_verification(tmp_path):
    t = north_up_transform(-58000, -118000, 0.05)
    blocks = {(0, 0), (0, 1)}
    cal = calibrate_histogram(np.bincount([1000], minlength=65536))
    for br, bc in blocks:
        a = RasterAccumulator((BLOCK, BLOCK))
        a.add(np.array([0, 511]), np.array([1, 510]), np.array([0, 65535], np.uint16))
        layers = a.layers(cal)
        layers.update(
            road_state=np.ones((BLOCK, BLOCK), np.uint8),
            filled=layers["normalized"].copy(),
            filled_mask=np.zeros((BLOCK, BLOCK), np.uint8),
        )
        np.savez_compressed(tmp_path / f"b{br}_{bc}.npz", **layers)
    manifest = []
    for row, col in [(0, 0), (0, 768)]:
        w = min(1024, 1024 - col)
        layers = read_mosaic(
            tmp_path,
            blocks,
            row,
            col,
            BLOCK,
            w,
            [
                "maximum",
                "mean",
                "count",
                "observed",
                "normalized",
                "road_state",
                "filled",
                "filled_mask",
            ],
        )
        paths = {}
        for name, image in layers.items():
            file = tmp_path / f"tile{col}_{name}.tif"
            write_tif(file, image, t @ Affine.translation(col, row), None)
            paths[name] = str(file)
        manifest.append(
            dict(
                row_offset=row,
                col_offset=col,
                height=BLOCK,
                width=w,
                layers=paths,
                crs_status="CRS_UNCONFIRMED",
            )
        )
    qc = validate_products(manifest, tmp_path, blocks, t, 4)
    assert qc["passed"] and qc["tile_layers_read_back"] == 16
    assert qc["unique_raster_point_count"] == 4


def test_exact_roi_boundary_and_geometry_chunk_invariance():
    xyz = surface(0, 15, 0, 10, 8, 0.03)
    cloud = cloud_from(xyz)
    roi = box(2.25, 1.25, 12.75, 8.75)
    criteria = RoadCriteria(min_component_m2=20)
    a = SurfaceAccumulator(cloud.header, criteria)
    b = SurfaceAccumulator(cloud.header, criteria)
    a.add(xyz)
    for chunk in np.array_split(xyz[::-1], 13):
        b.add(chunk)
    ma, mb = a.finish(roi), b.finish(roi)
    np.testing.assert_array_equal(ma.classify(xyz), mb.classify(xyz))
    chosen = xyz[ma.classify(xyz) > 0]
    assert (chosen[:, 0] > 2.25).all() and (chosen[:, 0] < 12.75).all()
    assert (chosen[:, 1] > 1.25).all() and (chosen[:, 1] < 8.75).all()
    assert ma.state_at(np.array([2.1]), np.array([3.5]))[0] == 0


def test_pipeline_end_to_end_and_verified_cache(tmp_path, monkeypatch):
    import roadmarking.rasterization.pipeline as pipeline
    from roadmarking.config import load_config

    monkeypatch.setenv("ROADMARKING_DATA_ROOT", str(tmp_path))
    cfg = load_config()
    xyz = surface(-20, 0, -10, 0, 10, 0.04)
    source = "Job_20260325_0838_Track01_3"
    monkeypatch.setitem(pipeline.INTACT, source, len(xyz))
    las = laspy.LasData(laspy.LasHeader(point_format=3, version="1.2"))
    las.header.scales = [0.001] * 3
    las.x, las.y, las.z = xyz.T
    las.intensity = np.full(len(xyz), 5000, np.uint16)
    las.return_number = np.ones(len(xyz), np.uint8)
    las.number_of_returns = np.ones(len(xyz), np.uint8)
    las.gps_time = np.full(len(xyz), 100.0)
    path = tmp_path / f"{source}.laz"
    las.write(path)
    result = pipeline.process_source(cfg, path)
    assert result["qc"]["passed"]
    assert result["selected_points"] == len(xyz)
    assert result["uncertain_points"] == len(xyz)
    assert result["input_valid_points"] == len(xyz)
    assert result["pixels"]["filled"] == 0

    def unexpected_scan(*args, **kwargs):
        raise RuntimeError("Cache was not accepted")

    monkeypatch.setattr(pipeline, "iter_pointcloud_chunks", unexpected_scan)
    cached = pipeline.process_source(cfg, path, skip_validated=True)
    assert cached == result
    manifest = __import__("json").loads(__import__("pathlib").Path(result["manifest"]).read_text())
    artifact = __import__("pathlib").Path(manifest[0]["layers"]["maximum"])
    artifact.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="Cache was not accepted"):
        pipeline.process_source(cfg, path, skip_validated=True)
