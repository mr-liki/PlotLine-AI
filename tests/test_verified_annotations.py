"""Synthetic-only human-review fixtures; none are real client annotations."""

import json
import sqlite3
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import pytest
import rasterio
from affine import Affine
from shapely.geometry import LineString, Polygon, box

from roadmarking.dataset.annotation_store import (
    CHECKLIST,
    FIELDS,
    create_store,
    frame_id,
    metadata,
    read_annotations,
    record_review,
    snapshot,
)
from roadmarking.dataset.annotation_validation import validate_store
from roadmarking.dataset.annotation_workspace import (
    ignore_compatibility,
    independent_groups,
    taxonomy,
)
from roadmarking.dataset.labels import LabelFeature
from roadmarking.dataset.verified_export import check_dataset, export_dataset
from roadmarking.dataset.verified_masks import processor_targets, reviewed_masks
from roadmarking.rasterization.pipeline import sha256, write_tif
from roadmarking.visualization.intensity_render import save_intensity_png

ROOT = Path(__file__).resolve().parents[1]
A = Affine(0.05, 0, 1000, 0, -0.05, 2000)


@pytest.fixture
def workspace(tmp_path):
    size = 64
    layers = {}
    observed = np.ones((size, size), np.uint8)
    normalized = np.full((size, size), 0.1, np.float32)
    for name, array in {
        "maximum": np.full((size, size), 5000, np.int32),
        "mean": np.full((size, size), 5000, np.float32),
        "normalized": normalized,
        "observed": observed,
        "count": observed.astype(np.uint32),
        "filled_mask": observed * 0,
    }.items():
        path = tmp_path / "sources" / f"{name}.tif"
        write_tif(path, array, A, None)
        layers[name] = str(path)
    image = tmp_path / "sources/intensity_rgb.png"
    save_intensity_png(image, normalized, observed, rgb=True)
    tile = {
        "tile_id": "SYNTHETIC_TILE",
        "source": "SYNTHETIC_TRACK",
        "source_file": "SYNTHETIC.laz",
        "source_sha256": "SYNTHETIC",
        "height": size,
        "width": size,
        "resolution": 0.05,
        "affine": list(A)[:6],
        "bounds": [1000, 1996.8, 1003.2, 2000],
        "native_xy_bbox": [1000, 1996.8, 1003.2, 2000],
        "crs_status": "CRS_UNCONFIRMED",
        "layers": layers,
        "images": {"rgb": str(image)},
        "masks": {"valid": layers["observed"], "observed": layers["observed"]},
    }
    store = create_store(tmp_path / "annotations.gpkg", [tile], taxonomy(ROOT))
    return store, tile


def append_feature(
    store, tile, *, ident="SYNTHETIC_1", layer="paint_polygons", geometry=None, **changes
):
    record = dict.fromkeys(FIELDS)
    record.update(
        annotation_id=ident,
        global_instance_id="GLOBAL_" + ident,
        source_track=tile["source"],
        source_raster_id=tile["tile_id"],
        source_tile_id=tile["tile_id"],
        visual_class_id=1,
        visual_class_name="longitudinal_solid",
        annotation_source="SYNTHETIC_SELFTEST",
        review_status="DRAFT",
        source_coordinate_frame=frame_id(tile["source"]),
        geometry_type="MultiLineString" if layer == "paint_centerlines" else "MultiPolygon",
        purpose="UNCERTAIN" if layer == "review_areas" else "PAINT",
        width_verified=0,
        mask_reviewed=0,
    )
    record.update(changes)
    geometry = geometry if geometry is not None else box(1000.5, 1998, 1000.65, 1999.5)
    frame = gpd.GeoDataFrame([record], geometry=[geometry], crs=None)
    pyogrio.write_dataframe(
        frame, store, layer=layer, driver="GPKG", append=True, promote_to_multi=True
    )
    return ident


def approve(store, ident):
    return record_review(
        store, ident, "SYNTHETIC_REVIEWER_NOT_CLIENT", "VERIFIED", dict.fromkeys(CHECKLIST, True)
    )


def test_empty_schema_native_coordinates_and_no_automatic_approval(workspace):
    store, tile = workspace
    assert metadata(store)["tiles"][0]["affine"] == list(A)[:6]
    assert read_annotations(store) == []
    assert validate_store(store)["eligible_features"] == 0
    frame = gpd.read_file(store, layer="annotation_regions")
    assert frame.crs is None
    np.testing.assert_allclose(frame.total_bounds, tile["bounds"], rtol=0, atol=1e-9)


def test_drawing_review_stale_edit_and_append_only_history(workspace):
    store, tile = workspace
    ident = append_feature(store, tile)
    assert validate_store(store)["eligible_features"] == 0
    approve(store, ident)
    assert validate_store(store)["eligible_features"] == 1
    with sqlite3.connect(store) as db:
        db.execute("UPDATE paint_polygons SET comment='changed after review'")
        assert db.execute("SELECT COUNT(*) FROM annotation_audit").fetchone()[0] >= 3
        assert (
            db.execute(
                "SELECT old_geometry IS NOT NULL FROM annotation_audit "
                "WHERE action='UPDATE' LIMIT 1"
            ).fetchone()[0]
            == 1
        )
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("DELETE FROM review_events")
    report = validate_store(store)
    assert report["eligible_features"] == 0
    assert "missing_or_stale_human_review_event" in report["records"][0]["errors"]


@pytest.mark.parametrize(
    "status", ["DRAFT", "REJECTED", "UNCERTAIN", "PROPOSED_UNREVIEWED", "NEEDS_CORRECTION"]
)
def test_non_verified_status_never_eligible(workspace, status):
    store, tile = workspace
    append_feature(store, tile, review_status=status)
    assert validate_store(store)["eligible_features"] == 0


def test_status_field_alone_and_incomplete_checklist_do_not_approve(workspace):
    store, tile = workspace
    ident = append_feature(store, tile, review_status="VERIFIED")
    assert validate_store(store)["eligible_features"] == 0
    with pytest.raises(ValueError, match="checklist"):
        record_review(store, ident, "reviewer", "VERIFIED", {})


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"source_coordinate_frame": "EPSG:3763"}, "coordinate_frame_mismatch"),
        ({"visual_class_name": "arrow"}, "class_id_name_mismatch"),
        (
            {"visual_class_id": 3, "visual_class_name": "arrow"},
            "class_not_located_or_catalog_unresolved",
        ),
        ({"source_tile_id": "wrong"}, "unknown_source_raster_or_anchor"),
    ],
)
def test_validation_rejects_wrong_identity_or_unsupported_class(workspace, changes, error):
    store, tile = workspace
    append_feature(store, tile, **changes)
    assert error in validate_store(store)["records"][0]["errors"]


def test_duplicate_geometry_and_global_identity(workspace):
    store, tile = workspace
    append_feature(store, tile)
    append_feature(store, tile, ident="SYNTHETIC_2", global_instance_id="GLOBAL_SYNTHETIC_1")
    report = validate_store(store)
    assert report["duplicate_annotations"] == 2
    assert report["eligible_features"] == 0


def test_centerline_requires_explicit_width_and_paint_area_review(workspace):
    store, tile = workspace
    ident = append_feature(
        store, tile, layer="paint_centerlines", geometry=LineString([(1000.5, 1999), (1002, 1999)])
    )
    with pytest.raises(ValueError, match="width"):
        approve(store, ident)
    with sqlite3.connect(store) as db:
        db.execute(
            "UPDATE paint_centerlines SET marking_width=0.15,width_verified=1,mask_reviewed=1"
        )
    approve(store, ident)
    assert validate_store(store)["eligible_features"] == 1


def test_source_pixel_change_invalidates_review(workspace):
    store, tile = workspace
    ident = append_feature(store, tile)
    approve(store, ident)
    with rasterio.open(tile["layers"]["normalized"], "r+") as ds:
        pixels = ds.read(1)
        pixels[0, 0] = 0.9
        ds.write(pixels, 1)
    report = validate_store(store)
    assert report["global_errors"] and report["eligible_features"] == 0


def test_explicit_background_cannot_cover_known_paint(workspace):
    store, tile = workspace
    append_feature(store, tile)
    ident = append_feature(
        store,
        tile,
        ident="SYNTHETIC_BACKGROUND",
        layer="review_areas",
        geometry=box(1000, 1997, 1002, 2000),
        purpose="BACKGROUND",
    )
    with pytest.raises(ValueError, match="background_overlaps"):
        approve(store, ident)


def test_masks_holes_narrow_instances_and_missing_observations():
    outer = box(1000.5, 1998, 1002.5, 1999.5)
    hole = box(1001, 1998.5, 1002, 1999)
    feats = [
        LabelFeature(
            1, "GLOBAL_HOLE", 1, "polygon", Polygon(outer.exterior.coords, [hole.exterior.coords])
        ),
        LabelFeature(2, "GLOBAL_DASH", 2, "polygon", box(1000.1, 1999.8, 1000.25, 2000)),
    ]
    observed = np.ones((64, 64), bool)
    valid = observed.copy()
    observed[5, 5] = False
    valid[63] = False
    observed[63] = False
    labels, categories = reviewed_masks(A, observed.shape, observed, valid, feats)
    assert labels.instance[25, 30] == 0  # polygon hole
    assert (labels.instance[:4, 2:5] == 2).all()  # exactly three native pixels
    assert categories[5, 5] == 0 and categories[63, 5] == 6
    assert categories[10, 50] == 1  # unreviewed is not background
    assert labels.semantic[5, 5] == 255 and (labels.instance[~observed] == 0).all()


def test_verified_background_uncertainty_and_processor_guard():
    observed = np.ones((64, 64), bool)
    labels, categories = reviewed_masks(
        A,
        observed.shape,
        observed,
        observed,
        [],
        box(1000, 1996.8, 1003.2, 2000),
        box(1001, 1998, 1002, 1999),
    )
    assert categories[10, 10] == 2
    assert categories[25, 30] == 7 and labels.semantic[25, 30] == 255
    rgb = np.zeros((64, 64, 3), np.uint8)
    with pytest.raises(ValueError, match="cannot ignore"):
        processor_targets(rgb, labels)
    processor_targets(rgb, labels, inspection_only=True)


def test_installed_processor_ignore_loss_limitation_without_model_weights():
    evidence = ignore_compatibility()
    assert evidence["unknown_changes_stock_loss"]
    assert evidence["partial_sample_training_gate_rejects"]
    assert evidence["processor_class_labels"] == [0]
    assert not evidence["model_weights_loaded"]


def test_export_snapshot_determinism_identity_and_immutability(workspace, tmp_path):
    store, tile = workspace
    ident = append_feature(store, tile)
    approve(store, ident)
    digest = sha256(store)
    exported = snapshot(store, tmp_path / "verified_snapshot.gpkg")
    assert validate_store(exported)["eligible_features"] == 1
    first = export_dataset(Path(exported), tmp_path / "export_1")
    second = export_dataset(Path(exported), tmp_path / "export_2")
    assert sha256(store) == digest
    assert first["real_image_mask_pairs"] == 1
    assert not first["samples"][0]["stock_loss_eligible"]
    assert first["samples"][0]["instances"][0]["global_id"] == "GLOBAL_SYNTHETIC_1"
    assert first["samples"][0]["instance_id_to_semantic_id"] == {"1": 1}
    assert first["samples"][0]["file_sha256"] == second["samples"][0]["file_sha256"]
    assert check_dataset(tmp_path / "export_1")["integrity_passed"]
    sample = first["samples"][0]
    with rasterio.open(tmp_path / "export_1" / sample["files"]["instance"]) as ds:
        assert ds.transform == A and ds.shape == (64, 64)
    with pytest.raises(FileExistsError):
        snapshot(store, tmp_path / "verified_snapshot.gpkg")


def test_cross_tile_identity_stable_for_two_separate_local_maps():
    observed = np.ones((64, 64), bool)
    feature = LabelFeature(42, "ONE_GLOBAL_DASH", 2, "polygon", box(1002, 1999, 1003, 1999.15))
    left, _ = reviewed_masks(A, observed.shape, observed, observed, [feature])
    right, _ = reviewed_masks(
        A @ Affine.translation(48, 0), observed.shape, observed, observed, [feature]
    )
    assert left.instances[0]["global_id"] == right.instances[0]["global_id"]
    assert np.array_equal(left.instance[:, 48:], right.instance[:, :16])


def test_independent_footprints_include_cross_track_overlap():
    tiles = [
        {"tile_id": "a", "bounds": [0, 0, 10, 10]},
        {"tile_id": "b", "bounds": [5, 5, 15, 15]},
        {"tile_id": "c", "bounds": [100, 100, 110, 110]},
    ]
    assert sorted(map(len, independent_groups(tiles))) == [1, 2]


def test_real_workspace_project_ready_but_not_fake_ground_truth():
    path = ROOT / "outputs/reports/phase4_verified/report.json"
    if not path.exists():
        pytest.skip("Real annotation workspace not installed")
    report = json.loads(path.read_text())
    assert report["status"] == "ANNOTATION REQUIRED"
    assert report["annotations"]["eligible_features"] == 0
    assert report["real_image_mask_pairs"] == 0
    assert not report["mismatched_client_annotations_used_as_ground_truth"]
    assert not report["training_ready"]
    output = Path(report["output_root"])
    qgis = json.loads((output / "metadata/qgis_validation.json").read_text())
    assert qgis["project_reopened"] and not qgis["project_crs_confirmed"]
    assert qgis["bookmark_count"] == len(report["input_regions"])
    assert all(layer["valid"] for layer in qgis["layers"])
    assert all(layer["workspace_owned_source"] for layer in qgis["layers"])
    assert report["protected_unchanged"]
    for filename, digest in report["protected_hashes"].items():
        assert sha256(Path(filename)) == digest
