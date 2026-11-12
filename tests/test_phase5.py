"""Phase 5 source integrity, provenance and real data training gates."""

import json
from pathlib import Path

import laspy
import numpy as np
import pytest

from roadmarking.io.source_inventory import discover, fingerprint, inspect_records, safe_header
from roadmarking.training.quality import (
    DatasetNotReadyError,
    inspect_training_dataset,
    require_training_dataset,
)


def make_cloud(path):
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [0.001, 0.001, 0.001]
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = np.arange(20), np.arange(20) + 100, np.ones(20)
    cloud.intensity = np.arange(20, dtype=np.uint16) * 100
    cloud.return_number = np.ones(20, np.uint8)
    cloud.number_of_returns = np.ones(20, np.uint8)
    cloud.gps_time = np.arange(20, dtype=float)
    cloud.write(path)


@pytest.mark.parametrize("suffix", [".las", ".laz"])
def test_readonly_actual_record_sampling_and_native_units(tmp_path, suffix):
    path = tmp_path / ("native" + suffix)
    make_cloud(path)
    before = path.read_bytes()
    assert safe_header(path).point_count == 20
    report = inspect_records(path, sample_points=5)
    assert report["status"] == "SAMPLE_QC_PASS_FULL_SCAN_PENDING"
    assert report["sample_points_passed"] == 15
    assert not report["full_point_integrity_validated"]
    assert report["mins"] == [0, 100, 1]
    assert path.read_bytes() == before
    assert {p.name for p in tmp_path.iterdir()} == {path.name}


def test_corrupt_records_not_accepted_from_good_header(tmp_path):
    path = tmp_path / "invalid.las"
    make_cloud(path)
    cloud = laspy.read(path)
    cloud.return_number = np.zeros(20, np.uint8)
    cloud.write(path)
    report = inspect_records(path, sample_points=5)
    assert report["status"] == "SAMPLE_QC_FAILED"
    assert report["sample_points_passed"] == 0


def test_bounded_header_rejects_huge_offsets(tmp_path):
    path = tmp_path / "invalid.las"
    data = bytearray(227)
    data[:4] = b"LASF"
    data[96:100] = (2**31).to_bytes(4, "little")
    path.write_bytes(data)
    with pytest.raises(ValueError, match="offset"):
        safe_header(path)


def test_inventory_spaces_bangs_and_missing_directory(tmp_path):
    directory = tmp_path / "DAILY ACTIVITY REPORT" / "!!!!!geetha"
    directory.mkdir(parents=True)
    (directory / "source file.gpkg").write_bytes(b"example")
    files = [r for r in discover(tmp_path) if r["kind"] == "file"]
    assert len(files) == 1 and files[0]["extension"] == ".gpkg"
    with pytest.raises(FileNotFoundError):
        discover(tmp_path / "missing")


def test_partial_fingerprint_detects_changed_input_without_claiming_full_hash(tmp_path):
    path = tmp_path / "source.laz"
    path.write_bytes(b"abc")
    first = fingerprint(path)
    path.write_bytes(b"abd")
    assert fingerprint(path)["sha256_partial"] != first["sha256_partial"]
    assert "NOT_FULL_FILE_HASH" in first["method"]


def test_no_labels_cannot_be_bypassed_by_ready_flag(tmp_path):
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata/dataset_manifest.json").write_text(
        json.dumps(
            {
                "training_ready": True,
                "samples": [],
                "source_annotations_qc_final": True,
            }
        )
    )
    report = inspect_training_dataset(tmp_path)
    assert not report["training_ready"]
    assert "NO_APPROVED_IMAGE_MASK_PAIRS" in report["errors"]
    with pytest.raises(DatasetNotReadyError):
        require_training_dataset(tmp_path)
    assert not report["weights_loaded"]


def test_real_full_point_pilot_overlap_and_shared_normalization():
    root = Path(__file__).resolve().parents[1]
    pointer = root / "outputs/reports/phase5/production_pilot_latest.json"
    if not pointer.exists():
        pytest.skip("Phase 5 full-point pilot not installed")
    import rasterio
    from PIL import Image

    result = json.loads(Path(json.loads(pointer.read_text())["report"]).read_text())
    assert len(result["full_source_scans"]) == 3
    assert all(r["full_integrity_passed"] for r in result["full_source_scans"])
    assert result["total_valid_points"] == sum(
        r["expected_points"] for r in result["full_source_scans"]
    )
    assert len({t["normalization"] for t in result["tiles"]}) == 1
    for first, second in zip(result["tiles"][::2], result["tiles"][1::2], strict=True):
        assert second["affine"][2] - first["affine"][2] == pytest.approx(38.4)
        for name in ("maximum", "mean", "normalized", "count", "observed"):
            with rasterio.open(first["layers"][name]) as src:
                a = src.read(1)
                assert src.shape == (1024, 1024)
            with rasterio.open(second["layers"][name]) as src:
                b = src.read(1)
            assert np.array_equal(a[:, 768:], b[:, :256])
        with Image.open(first["images"]["rgb"]) as image:
            rgb = np.asarray(image)
        assert np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
        assert np.array_equal(rgb[:, :, 1], rgb[:, :, 2])


def test_real_qc_source_does_not_authorize_unknown_training_pixels():
    root = Path(__file__).resolve().parents[1]
    pointer = root / "outputs/reports/phase5/annotation_conversion_latest.json"
    if not pointer.exists():
        pytest.skip("Phase 5 conversion review not installed")
    result = json.loads(Path(json.loads(pointer.read_text())["report"]).read_text())
    store = Path(result["annotation_store"])
    review = store.parent.parent / "conversion_review_only"
    report = inspect_training_dataset(review)
    assert result["source_annotations_qc_final"]
    assert not report["training_ready"]
    assert "CONVERSION_REVIEW_MASKS_ARE_NOT_APPROVED_SUPERVISION" in report["errors"]
    assert any("UNKNOWN_NODATA_BOUNDARY_OR_PADDING_PIXELS" in e for e in report["errors"])
    assert result["approved_training_pairs"] == 0
    assert result["derived_draft_paint_areas"] > 0
