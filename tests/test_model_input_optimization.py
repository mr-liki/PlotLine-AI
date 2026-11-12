"""Native pixel geometry, conservative evidence and source-window contracts."""

import json
from pathlib import Path

import numpy as np
import pytest
from affine import Affine

from roadmarking.analysis.model_input_optimization import processor_probe
from roadmarking.preprocessing.road_focused_roi import road_focused_roi
from roadmarking.rasterization.model_tile_selector import select_model_tile
from roadmarking.rasterization.model_tiler import (
    candidate_origins,
    crop_sources,
    pixel_keys,
    tile_transform,
    unique_pixel_count,
)


@pytest.mark.parametrize(
    "kind",
    [
        "narrow",
        "boundary",
        "edge",
        "faded",
        "dark",
        "sparse",
        "bright_nonroad",
        "junction",
        "parallel",
    ],
)
def test_no_unknown_roadway_evidence_can_be_rejected(kind):
    observed = np.ones((64, 64), bool)
    intensity = np.zeros((64, 64), np.float32)
    search, strict, uncertain = (np.zeros_like(observed) for _ in range(3))
    if kind in {"narrow", "boundary", "edge", "parallel"}:
        intensity[:, -3:] = 1
        if kind == "parallel":
            intensity[:, 8:11] = 1
        search[:, -4:] = True
    elif kind == "faded":
        intensity[:, 30:33] = 0.01
    elif kind == "sparse":
        observed[:] = False
        observed[::8, ::8] = True
    elif kind == "bright_nonroad":
        intensity[8:20, 8:20] = 1
    elif kind == "junction":
        search[25:39, :] = search[:, 25:39] = True
    before = intensity.copy()
    roi = road_focused_roi(observed, search, strict, uncertain)
    result = select_model_tile(
        observed, np.ones_like(observed), search, roi["evidence"], roi["unknown"]
    )
    assert result["decision"] != "SKIP"
    assert roi["roi"][observed].all()
    assert np.array_equal(before, intensity)


def test_empty_is_skipped_but_measured_black_is_retained():
    empty = np.zeros((5, 5), bool)
    valid = ~empty
    assert select_model_tile(empty, valid, empty, empty, empty)["decision"] == "SKIP"
    assert select_model_tile(valid, valid, empty, empty, valid)["decision"] == "KEEP_FOR_REVIEW"
    assert select_model_tile(valid, valid, valid, valid, empty)["decision"] == "KEEP"
    assert (
        select_model_tile(valid, valid, valid, valid, empty, uncertain=valid)["decision"]
        == "KEEP_FOR_REVIEW"
    )
    with pytest.raises(ValueError):
        select_model_tile(valid, empty, empty, empty, empty)


def test_broad_search_and_junction_do_not_shrink_to_strict_margin():
    search = np.zeros((100, 100), bool)
    search[10:90, 5:95] = True
    strict = np.zeros_like(search)
    strict[:, 49:51] = True
    roi = road_focused_roi(search, search, strict, search, resolution=0.05, margin=0.15)
    assert roi["roi"][search].all()
    assert roi["evidence"][:, 46:54].all()


@pytest.mark.parametrize("size,overlap", [(512, 128), (1024, 256)])
def test_exact_crop_padding_affine_overlap_and_unique_coverage(size, overlap):
    array = np.arange(1024**2, dtype=np.float32).reshape(1024, 1024)
    array[:, 510:513] = 65432  # Three-pixel paint must remain three pixels.
    before = array.copy()
    origins = candidate_origins([(0, 0, 1024, 1024)], size, overlap)
    assert origins == candidate_origins([(0, 0, 1024, 1024)], size, overlap)
    base = Affine(0.05, 0, -56000, 0, -0.05, -89000)
    coverage = []
    for row, col in origins:
        crop, valid = crop_sources([(array, 0, 0)], row, col, size, -9999)
        h, w = min(size, 1024 - row), min(size, 1024 - col)
        assert crop.shape == (size, size)
        assert np.array_equal(crop[:h, :w], array[row : row + h, col : col + w])
        assert (crop[~valid] == -9999).all()
        assert valid.sum() == h * w
        transform = tile_transform(base, row, col)
        np.testing.assert_allclose(
            transform @ (0.5, 0.5), base @ (col + 0.5, row + 0.5), rtol=0, atol=1e-9
        )
        np.testing.assert_allclose(
            (~transform) @ (transform @ (33.5, 21.5)), (33.5, 21.5), rtol=0, atol=1e-8
        )
        coverage.append(pixel_keys(valid, row, col))
    assert unique_pixel_count(coverage) == 1024**2
    assert sum(len(c) for c in coverage) > 1024**2
    first, _ = crop_sources([(array, 0, 0)], 0, 0, size)
    second, _ = crop_sources([(array, 0, 0)], 0, size - overlap, size)
    assert np.array_equal(first[:, -overlap:], second[:, :overlap])
    assert np.array_equal(array, before)


def test_inconsistent_intensity_overlap_fails_and_mask_union_is_conservative():
    a = np.zeros((8, 8), np.uint8)
    b = np.ones((8, 8), np.uint8)
    with pytest.raises(ValueError, match="overlap"):
        crop_sources([(a, 0, 0), (b, 0, 0)], 0, 0, 8)
    result, valid = crop_sources([(a, 0, 0), (b, 0, 0)], 0, 0, 8, union=True)
    assert result.all() and valid.all()


@pytest.mark.parametrize("size,overlap", [(0, 0), (512, 512), (512, -1)])
def test_invalid_grid_rejected(size, overlap):
    with pytest.raises(ValueError):
        candidate_origins([(0, 0, 64, 64)], size, overlap)


def test_processor_native_shapes_without_network_or_weights():
    probe = processor_probe()
    assert probe["shapes"]["512"]["native"] == [1, 3, 512, 512]
    assert probe["shapes"]["1024"]["native"] == [1, 3, 1024, 1024]
    assert probe["inference_seconds"] is None


def test_real_pilot_retention_input_protection_and_phase4_manifest_contract():
    root = Path(__file__).resolve().parents[1]
    report_path = root / "outputs/reports/phase3_3/report.json"
    if not report_path.exists():
        pytest.skip("Real-data pilot is not installed in this checkout")
    report = json.loads(report_path.read_text())
    assert report["protected_unchanged"]
    for value in report["comparison"].values():
        for kind in ("observed", "search_observed", "paint_proxy", "high_contrast_proxy"):
            assert value["retention_pct"][kind] == 100
    from roadmarking.rasterization.pipeline import sha256
    from roadmarking.training.preflight import REQUIRED_LAYERS

    phase4 = report["phase4"]
    path = Path(phase4["original_manifest"])
    assert sha256(path) == phase4["original_manifest_sha256"]
    original = json.loads(path.read_text())
    original = original["tiles"] if isinstance(original, dict) else original
    assert original[0]["layers"].keys() >= REQUIRED_LAYERS
    for path, digest in report["protected_hashes"].items():
        assert sha256(Path(path)) == digest
    manifest = json.loads((Path(report["output_root"]) / "metadata/tile_manifest.json").read_text())
    for size, entries in manifest.items():
        from roadmarking.training.preflight import inspect_manifest

        checked = inspect_manifest(
            Path(report["output_root"]) / f"model_tiles_{size}/tile_manifest.json"
        )
        assert checked["metadata_consistent"] and not checked["errors"]
        assert len({entry["tile_id"] for entry in entries}) == len(entries)
        for entry in entries:
            assert entry["width"] == int(size)
            assert entry["source_windows"]
            assert entry["pixel_audit"] == "EXACT_SOURCE_TRANSFER_ALL_CHANNELS"
            assert entry["layers"].keys() >= REQUIRED_LAYERS
