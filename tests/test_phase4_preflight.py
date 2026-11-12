"""Phase 4 step-1 gates: discovery and honest absence reporting, not label QA."""

import copy
import json

import pytest

from roadmarking.config import load_config
from roadmarking.training.preflight import digest, inspect_manifest, inventory, run_preflight


def test_empty_and_placeholder_directories(tmp_path):
    (tmp_path / ".gitkeep").touch()
    result = inventory(tmp_path)
    assert result["exists"] and result["dataset_count"] == 0
    assert len(result["placeholder_files"]) == 1
    assert not result["other_files"]
    missing = inventory(tmp_path / "missing")
    assert not missing["exists"] and missing["dataset_count"] == 0


@pytest.mark.parametrize(
    "companions,missing",
    [([], [".shx", ".dbf"]), ([".SHX"], [".dbf"]), ([".SHX", ".DBF", ".PRJ"], [])],
)
def test_shapefile_companions_are_case_insensitive(tmp_path, companions, missing):
    folder = tmp_path / "nested"
    folder.mkdir()
    shp = folder / "Roads.SHP"
    shp.write_bytes(b"uninspected shapefile")
    for suffix in companions:
        (folder / ("roads" + suffix)).write_bytes(b"companion")
    original = shp.read_bytes()
    result = inventory(tmp_path)
    assert result["dataset_count"] == 1
    record = result["datasets"][0]
    assert record["missing_required_companions"] == missing
    assert record["prj_present"] == (".PRJ" in companions)
    assert record["status"] == ("INCOMPLETE_SHAPEFILE" if missing else "UNINSPECTED")
    assert record["attribute_fields"] is None
    assert shp.read_bytes() == original


def test_empty_geopackage_orphan_companions_and_unsupported_files(tmp_path):
    (tmp_path / "empty.GPKG").touch()
    (tmp_path / "orphan.dbf").write_bytes(b"metadata")
    (tmp_path / "delivery.zip").write_bytes(b"not automatically extracted")
    result = inventory(tmp_path)
    assert result["datasets"][0]["status"] == "EMPTY_FILE"
    assert len(result["orphan_shapefile_companions"]) == 1
    assert len(result["other_files"]) == 1
    assert result["datasets"][0]["classification_values"] is None


def make_manifest(root):
    root.mkdir(parents=True, exist_ok=True)
    layers = {}
    for name in ("maximum", "mean", "normalized", "count", "observed", "filled_mask"):
        layer = root / f"{name}.tif"
        layer.write_bytes(b"existence-only check, not an actual raster")
        layers[name] = str(layer)
    tile = dict(
        tile_id="tile0",
        height=1024,
        width=1024,
        resolution=0.05,
        affine=[0.05, 0, -58000, 0, -0.05, -118000],
        native_xy_bbox=[-58000, -118051.2, -57948.8, -118000],
        source_file="intact.laz",
        crs_status="CRS_UNCONFIRMED",
        layers=layers,
    )
    path = root / "tile_manifest.json"
    path.write_text(json.dumps([tile]), encoding="utf-8")
    return path, tile


def test_manifest_native_frame_metadata_only(tmp_path):
    path, _ = make_manifest(tmp_path)
    before = digest(path)
    result = inspect_manifest(path)
    assert result["tile_count"] == 1
    assert result["metadata_consistent"] and not result["alignment_verified"]
    assert result["coordinate_units"] == "UNCONFIRMED"
    assert result["crs_statuses"] == {"CRS_UNCONFIRMED": 1}
    assert not result["errors"] and result["existing_layer_files_checked"] == 6
    assert before == digest(path)


@pytest.mark.parametrize("error", ["duplicate", "bounds", "missing_layer"])
def test_manifest_rejects_inconsistencies(tmp_path, error):
    path, tile = make_manifest(tmp_path)
    tiles = [copy.deepcopy(tile)]
    if error == "duplicate":
        tiles.append(tile)
    elif error == "bounds":
        tiles[0]["native_xy_bbox"][0] += 1
    else:
        tiles[0]["layers"]["observed"] = str(tmp_path / "absent.tif")
    path.write_text(json.dumps(tiles), encoding="utf-8")
    result = inspect_manifest(path)
    assert result["errors"] and not result["metadata_consistent"]


def test_missing_manifest_is_explicit(tmp_path):
    result = inspect_manifest(tmp_path / "absent.json")
    assert result["errors"] and not result["exists"]


def test_no_annotation_run_stops_without_dataset_and_preserves_phase3(tmp_path, monkeypatch):
    monkeypatch.setenv("ROADMARKING_DATA_ROOT", str(tmp_path))
    cfg = load_config()
    for key in ("data.raw_annotations", "data.reference_gis"):
        folder = cfg.path(key)
        folder.mkdir(parents=True)
        (folder / ".gitkeep").touch()
    path, _ = make_manifest(cfg.path("data.processed") / "phase3/metadata")
    old_report = cfg.path("outputs.reports") / "phase3/phase3_report.json"
    old_report.parent.mkdir(parents=True)
    old_report.write_text('{"preserve": true}', encoding="utf-8")
    before = {p: digest(p) for p in (path, old_report)}
    report = run_preflight(cfg)
    assert report["status"] == "ANNOTATIONS_NOT_AVAILABLE"
    assert not report["phase4_complete"] and not report["phase5_ready"]
    assert not report["dataset"]["generated"] and report["dataset"]["images"] == 0
    assert not report["splits"]["generated"]
    assert report["alignment"]["displacement_native_units"] is None
    assert report["alignment"]["status"] == "NOT_ASSESSED"
    assert all(digest(p) == sha for p, sha in before.items())
    assert report["phase3_metadata_unchanged"]
    output = cfg.path("data.processed") / "phase4"
    assert (output / "metadata" / report["run_id"] / "availability.json").is_file()
    for name in ("images", "instance_masks", "splits"):
        assert not (output / name).exists()
    assert (cfg.path("outputs.reports") / "phase4/phase4_report.md").is_file()


def test_file_presence_does_not_verify_annotations(tmp_path, monkeypatch):
    monkeypatch.setenv("ROADMARKING_DATA_ROOT", str(tmp_path))
    cfg = load_config()
    raw = cfg.path("data.raw_annotations")
    raw.mkdir(parents=True)
    source = raw / "manual.gpkg"
    source.write_bytes(b"not even a valid geopackage")
    before = source.read_bytes()
    report = run_preflight(cfg)
    assert report["status"] == "ANNOTATION_FILES_REQUIRE_INSPECTION"
    assert not report["dataset"]["generated"] and not report["phase4_complete"]
    assert report["annotation_quality"]["verified_features"] == 0
    assert report["inputs"]["annotations"]["datasets"][0]["status"] == "UNINSPECTED"
    assert source.read_bytes() == before
