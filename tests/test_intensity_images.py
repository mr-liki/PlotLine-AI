"""Raster-only output contracts and known-truth source-to-PNG integration."""

from pathlib import Path

import numpy as np
import pytest
from affine import Affine
from PIL import Image

from roadmarking.preprocessing.intensity import calibrate_histogram
from roadmarking.rasterization.rasterizer import accumulate_intensity_window
from roadmarking.visualization.intensity_render import (
    grayscale_intensity,
    save_intensity_png,
    write_png_worldfile,
)
from roadmarking.visualization.mosaic import stitch_raster_windows


@pytest.mark.parametrize("gamma", [0.7, 1, 1.5])
def test_exact_clean_grayscale_monotone_transfer(gamma):
    normalized = np.linspace(0, 1, 256).reshape(16, 16).astype(np.float32)
    observed = np.ones(normalized.shape, bool)
    before = normalized.copy()
    pixels = grayscale_intensity(normalized, observed, gamma=gamma)
    assert pixels.dtype == np.uint8 and pixels.shape == normalized.shape
    assert np.all(np.diff(pixels.ravel().astype(int)) >= 0)
    assert pixels[0, 0] == 0 and pixels[-1, -1] == 255
    assert np.array_equal(normalized, before)
    assert np.array_equal(pixels, np.rint(255 * normalized.astype(float) ** gamma).astype(np.uint8))


def test_nodata_not_filled_or_confused_with_valid_dark_data():
    normalized = np.array([[0, -9999], [1, 0.5]], np.float32)
    observed = np.array([[1, 0], [1, 1]], np.uint8)
    pixels = grayscale_intensity(normalized, observed)
    assert pixels[0, 0] == pixels[0, 1] == 0
    assert observed[0, 0] == 1 and observed[0, 1] == 0
    assert normalized[0, 1] == -9999
    assert pixels[1, 0] == 255


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf])
def test_invalid_gamma_refused(bad):
    with pytest.raises(ValueError):
        grayscale_intensity(np.ones((3, 3)), np.ones((3, 3)), gamma=bad)


def test_no_overlay_or_rgb_color_can_enter_png(tmp_path):
    image = np.zeros((1024, 1024), np.float32)
    image[100:105, 100:900] = 1
    observed = np.ones(image.shape, bool)
    single = tmp_path / "intensity_model_input.png"
    rgb = tmp_path / "intensity_model_input_rgb.png"
    expected = save_intensity_png(single, image, observed)
    save_intensity_png(rgb, image, observed, rgb=True)
    with Image.open(single) as im:
        assert im.mode == "L" and im.size == (1024, 1024)
        assert np.array_equal(np.asarray(im), expected)
    with Image.open(rgb) as im:
        data = np.asarray(im)
        assert im.mode == "RGB" and data.shape == (1024, 1024, 3)
        assert np.array_equal(data[:, :, 0], data[:, :, 1]) and np.array_equal(
            data[:, :, 1], data[:, :, 2]
        )
        assert np.array_equal(data[:, :, 0], expected)
    # No line is interpolated between separate original painted strokes.
    assert (expected[105:110] == 0).all()


def test_file_calibration_does_not_change_with_tile_histogram():
    a = np.full((50, 50), 0.1, np.float32)
    b = np.full((50, 50), 0.9, np.float32)
    a[:, 0] = b[:, 0] = 0.4
    aa = grayscale_intensity(a, np.ones_like(a, bool))
    bb = grayscale_intensity(b, np.ones_like(b, bool))
    assert np.array_equal(aa[:, 0], bb[:, 0])


def test_worldfile_preserves_pixel_centers_native_xy(tmp_path):
    path = tmp_path / "image.png"
    transform = Affine(0.05, 0, -56000.05, 0, -0.05, -89000.05)
    write_png_worldfile(path, transform)
    a, d, b, e, c, f = map(float, path.with_suffix(".pgw").read_text().split())
    assert (a, b, d, e) == (0.05, 0, 0, -0.05)
    assert (c, f) == pytest.approx(transform * (0.5, 0.5))


def test_mosaic_exact_overlap_and_no_resampling():
    whole = np.arange(30, dtype=np.float32).reshape(5, 6)
    t = Affine(0.05, 0, -56000, 0, -0.05, -89000)
    result, tt, coverage, qa = stitch_raster_windows(
        [(whole[:, :4], t), (whole[:, 2:], t * Affine.translation(2, 0))], missing_value=-9999
    )
    assert np.array_equal(result, whole) and tt == t and coverage.all()
    assert qa["overlap_pixels_compared"] == 10 and not qa["resampled"]


def test_mosaic_refuses_seam_disagreement():
    t = Affine(0.05, 0, 0, 0, -0.05, 1)
    with pytest.raises(ValueError, match="disagree"):
        stitch_raster_windows(
            [(np.ones((3, 3)), t), (np.zeros((3, 3)), t * Affine.translation(2, 0))],
            missing_value=-1,
        )


@pytest.mark.parametrize(
    "transform",
    [
        Affine(0.1, 0, 0, 0, -0.1, 1),
        Affine(0.05, 0, 0.025, 0, -0.05, 1),
        Affine(0.05, 0.01, 0, 0, -0.05, 1),
    ],
)
def test_mosaic_refuses_grid_changes(transform):
    t = Affine(0.05, 0, 0, 0, -0.05, 1)
    with pytest.raises(ValueError):
        stitch_raster_windows(
            [(np.ones((3, 3)), t), (np.ones((3, 3)), transform)], missing_value=-1
        )


def test_all_valid_source_intensities_survive_without_road_mask():
    xyz = np.array([[0.025, 0.975, 3], [0.075, 0.975, 3], [0.075, 0.975, 7], [0.975, 0.025, 9]])
    raw = np.array([4000, 45000, 1000, 6000], np.uint16)
    calibration = calibrate_histogram(np.bincount(raw, minlength=65536))
    layers, keep = accumulate_intensity_window(
        xyz, raw, Affine(0.05, 0, 0, 0, -0.05, 1), (20, 20), calibration
    )
    assert keep.all() and layers["count"].sum() == 4
    assert layers["maximum"][0, 1] == 45000 and layers["mean"][0, 1] == 23000
    assert layers["normalized"][10, 10] == -9999
    assert layers["observed"][19, 19] == 1


def test_mosaic_unprocessed_holes_are_not_claimed_observations():
    t = Affine(0.05, 0, 0, 0, -0.05, 1)
    image, _, covered, _ = stitch_raster_windows(
        [(np.ones((2, 2)), t), (np.ones((2, 2)), t * Affine.translation(4, 0))], missing_value=-9999
    )
    assert not covered[:, 2:4].any() and (image[:, 2:4] == -9999).all()


def test_real_file_pipeline_export_and_immutable_source(tmp_path, monkeypatch):
    import laspy
    import rasterio

    from roadmarking.analysis.corridor_pilot import POINTS
    from roadmarking.io.pointcloud import iter_pointcloud_chunks
    from roadmarking.rasterization.pipeline import INTACT, dump, sha256
    from roadmarking.visualization.intensity_pipeline import run

    source = "Job_20260325_0838_Track01_3"
    path = tmp_path / "data/raw/pointclouds" / f"{source}.laz"
    path.parent.mkdir(parents=True)
    xyz = np.array(
        [[10.025, 10.025, 3], [10.075, 10.025, 3], [10.075, 10.025, 5], [20.025, 20.025, 3]]
    )
    raw = np.array([4000, 45000, 1000, 4500], np.uint16)
    las = laspy.LasData(laspy.LasHeader(point_format=3, version="1.2"))
    las.header.scales = [0.001] * 3
    las.x, las.y, las.z = xyz.T
    las.intensity = raw
    las.return_number = np.ones(4, np.uint8)
    las.number_of_returns = np.ones(4, np.uint8)
    las.gps_time = np.full(4, 100.0)
    las.write(path)
    monkeypatch.setitem(INTACT, source, 4)
    digest = sha256(path)
    records = np.empty(4, POINTS)
    _, cloud = next(iter(iter_pointcloud_chunks(path)))
    records["xyz"], records["intensity"], records["source_index"] = (
        cloud.xyz,
        cloud.intensity,
        np.arange(4),
    )
    cache = tmp_path / "data/processed/phase3_1_topodot_inspired/point_cache/fixture"
    cache.mkdir(parents=True)
    records.tofile(cache / "p_0_0.bin")
    identity = {"boxes": [[-30, -30, 81.2, 81.2]], "partition_m": 64, "source_sha256": digest}
    dump(cache / "completion.json", {"identity": identity})
    stats = {"identity": identity, "artifacts": {"p_0_0.bin": sha256(cache / "p_0_0.bin")}}
    old = tmp_path / "data/processed/phase3_1_topodot_inspired/fixture"
    old_report = tmp_path / "outputs/reports/phase3_1_topodot_inspired/fixture"
    calibration = calibrate_histogram(np.bincount(raw, minlength=65536)).as_dict()
    tile = {
        "tile_id": "fixture_tile",
        "source": source,
        "source_sha256": digest,
        "row_offset": 0,
        "col_offset": 0,
        "affine": [0.05, 0, 0, 0, -0.05, 51.2],
        "normalization": calibration,
    }
    dump(old / "metadata/tile_manifest.json", [tile])
    dump(tmp_path / "data/processed/phase3/metadata/tile_manifest.json", [tile])
    dump(
        old_report / "report.json",
        {
            "run_id": "fixture",
            "output_root": str(old),
            "report_directory": str(old_report),
            "source_streams": {source: stats},
        },
    )
    dump(old_report.parent / "latest.json", {"report": str(old_report / "report.json")})
    (tmp_path / "configs").mkdir()
    settings = Path(__file__).resolve().parents[1] / "configs/phase3_intensity_images.yaml"
    (tmp_path / "configs/phase3_intensity_images.yaml").write_text(settings.read_text())
    # Run identity fingerprints code files; provide real implementation sources.
    for relative in (
        "src/roadmarking/visualization/intensity_render.py",
        "src/roadmarking/visualization/mosaic.py",
        "src/roadmarking/rasterization/rasterizer.py",
        "src/roadmarking/preprocessing/intensity.py",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((settings.parents[1] / relative).read_bytes())
    result = run(tmp_path, tile_ids=["fixture_tile"])
    assert sha256(path) == digest and result["protected_unchanged"]
    entry = result["tiles"][0]
    assert entry["metrics"]["source_points_in_tile"] == 4
    with rasterio.open(entry["rasters"]["maximum"]) as ds:
        assert ds.transform == Affine(*tile["affine"]) and ds.crs is None
        assert ds.read(1)[823, 201] == 45000
    with Image.open(entry["images"]["model_input_rgb"]) as im:
        assert im.size == (1024, 1024) and im.mode == "RGB"
    again = run(tmp_path, tile_ids=["fixture_tile"])
    assert again["run_id"] == result["run_id"]
    Path(entry["images"]["model_input"]).write_bytes(b"corrupted export")
    with pytest.raises(ValueError, match="integrity"):
        run(tmp_path, tile_ids=["fixture_tile"])
