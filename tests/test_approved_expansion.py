"""Actual approved expansion preserves roles, exclusions and native target alignment."""

import json
from pathlib import Path

import numpy as np
from scripts.phase5_7_expand_approved import check_expanded


def test_actual_expansion_preserves_approved_roles_and_protected_test():
    root = Path(__file__).resolve().parents[1]
    context = json.loads((root / "outputs/reports/phase5_7_expanded/latest.json").read_text())
    dataset = Path(context["dataset"])
    assert check_expanded(dataset)["training_ready"]
    manifest = json.loads((dataset / "metadata/dataset_manifest.json").read_text())
    approval = json.loads(Path(manifest["approval"]).read_text())
    assert approval["roles"]["NEW_FINAL_TEST_RESERVED"] == "protected_test"
    assert any(c["tile_id"] == "NEW_TRAIN_REVIEW" for c in manifest["crops"])
    assert any(c["tile_id"] == "NEW_DEV_REVIEW" for c in manifest["crops"])
    assert not any(c["tile_id"] == "NEW_FINAL_TEST_RESERVED" for c in manifest["crops"])


def test_actual_new_targets_exclude_nodata_and_keep_native_grayscale():
    root = Path(__file__).resolve().parents[1]
    context = json.loads((root / "outputs/reports/phase5_7_expanded/latest.json").read_text())
    manifest = json.loads((Path(context["dataset"]) / "metadata/dataset_manifest.json").read_text())
    for crop in manifest["crops"]:
        if not crop["tile_id"].startswith("NEW_"):
            continue
        with np.load(crop["file"]) as data:
            assert data["rgb"].shape == (512, 512, 3)
            assert np.array_equal(data["rgb"][:, :, 0], data["rgb"][:, :, 1])
            assert not data["masks"][:, ~data["validity"]].any()
            assert data["masks"].sum((1, 2)).min(initial=4) >= 4
