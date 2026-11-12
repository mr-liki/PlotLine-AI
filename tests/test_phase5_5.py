"""Approval immutability, geographic allocation and actual new-supervision checks."""

import json
from pathlib import Path

import pytest

from roadmarking.dataset.phase55 import allocate_regions, check_dataset, write_attestation
from roadmarking.rasterization.pipeline import sha256


def region(name, bounds, uid, split=None):
    return {
        "tile_id": name,
        "native_xy_bbox": bounds,
        "source_global_ids": [uid],
        "existing_split": split,
    }


def test_shared_ids_force_same_fold_and_independent_new_region_becomes_validation():
    regions = [
        region("old_train", [0, 0, 10, 10], "a", "train"),
        region("old_val", [100, 0, 110, 10], "b", "val"),
        region("new_linked", [200, 0, 210, 10], "a"),
        region("new_independent", [300, 0, 310, 10], "c"),
    ]
    splits = allocate_regions(regions, [])
    assert splits["assignment"]["new_linked"] == "train"
    assert splits["assignment"]["new_independent"] == "val"


def test_protected_train_validation_bridge_and_future_test_leakage_fail():
    with pytest.raises(ValueError, match="bridges"):
        allocate_regions(
            [region("a", [0, 0, 10, 10], "x", "train"), region("b", [5, 0, 15, 10], "y", "val")], []
        )
    with pytest.raises(ValueError, match="future-test"):
        allocate_regions(
            [region("a", [0, 0, 10, 10], "x", "train")],
            [{"reference_annotation_ids": ["x"], "bounds": [100, 0, 110, 10]}],
        )


def test_attestation_is_write_once_and_hash_bound(tmp_path):
    path = tmp_path / "approval.json"
    digest = write_attestation(path, {"scope": ["A"], "visible_only": True})
    assert sha256(path) == digest
    with pytest.raises(OSError):
        write_attestation(path, {"scope": "expanded without approval"})
    assert json.loads(path.read_text())["scope"] == ["A"]


def test_actual_seven_region_export_has_new_instances_and_protected_validation():
    root = Path(__file__).resolve().parents[1]
    pointer = root / "outputs/reports/phase5_5/latest.json"
    if not pointer.exists():
        pytest.skip("Phase 5.5 approved export not installed")
    dataset = Path(json.loads(pointer.read_text())["dataset"])
    result = check_dataset(dataset)
    assert result["training_ready"]
    assert not result["future_test_regions_used"]
    stats = result["statistics"]
    assert sum(stats["unique_source_instance_ids_by_split_class"]["train"].values()) > 20
    assert stats["crop_counts_by_split"]["val"] == 5
    assert stats["new_supported_ids_with_observed_pixels"] < 125
