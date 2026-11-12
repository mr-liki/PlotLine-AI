"""Configuration loading and validation tests."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from roadmarking.config import (
    EXPECTED_CHECKPOINT,
    PROJECT_ROOT,
    ConfigError,
    load_config,
    validate,
)
from roadmarking.gis.attributes import load_schemas


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_default_config_loads(cfg) -> None:
    assert cfg.checkpoint == EXPECTED_CHECKPOINT
    assert cfg.device.startswith("cuda") or cfg.device == "cpu"
    assert cfg["tiling"]["tile_overlap_px"] < cfg["tiling"]["tile_size_px"]


def test_paths_are_relative_to_root(cfg) -> None:
    p = cfg.path("data.raw_pointclouds")
    assert p.is_relative_to(cfg.root)
    assert cfg.root == PROJECT_ROOT
    with pytest.raises(ConfigError):
        cfg.path("does.not.exist")


def test_no_crs_assumed_and_phase3_resolution_measured(cfg) -> None:
    assert cfg["crs"]["source_epsg"] is None
    assert cfg["rasterization"]["ground_sampling_distance_m"] == 0.05
    assert "rasterization" in cfg.provisional_sections()


def test_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ROADMARKING_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("ROADMARKING_DEVICE", "cpu")
    c = load_config()
    assert c.device == "cpu"
    assert c.path("outputs.gpkg") == (tmp_path / "outputs" / "gpkg").resolve()


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("tiling", "tile_overlap_px", 4096),
        ("inference", "score_threshold", 1.5),
        ("inference", "batch_size", 0),
        ("runtime", "device", "tpu"),
    ],
)
def test_invalid_values_rejected(cfg, section: str, key: str, value: object) -> None:
    raw = copy.deepcopy(cfg.raw)
    raw[section][key] = value
    with pytest.raises(ConfigError):
        validate(raw)


def test_absolute_paths_rejected(tmp_path: Path) -> None:
    src = PROJECT_ROOT / "configs"
    for f in src.glob("*.yaml"):
        (tmp_path / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    paths = yaml.safe_load((tmp_path / "paths.yaml").read_text(encoding="utf-8"))
    paths["logs"] = str(tmp_path.resolve())
    (tmp_path / "paths.yaml").write_text(yaml.safe_dump(paths), encoding="utf-8")
    with pytest.raises(ConfigError, match="relative"):
        load_config(tmp_path / "default.yaml")


def test_class_taxonomy(cfg) -> None:
    classes = cfg["classes"]
    assert len(classes["longitudinal_designations"]) == 14
    codes = {c["code"] for c in classes["other_mark_codes"]}
    expected = {
        "M15",
        "M15a",
        "M15b",
        "M15c",
        "M15d",
        "M15e",
        "M15f",
        "M16",
        "M16a",
        "M16b",
        "M17",
        "M17a",
        "M9a",
        "M20",
        "M21",
    }
    assert codes == expected
    rule_names = set(classes["rule_families"])
    for d in classes["longitudinal_designations"]:
        assert set(d["rules"]) <= rule_names


def test_unknown_model_class_rejected(cfg) -> None:
    raw = copy.deepcopy(cfg.raw)
    raw["classes"]["other_mark_codes"][0]["model_classes"] = ["nonexistent"]
    with pytest.raises(ConfigError):
        validate(raw)


def test_gis_schemas(cfg) -> None:
    schemas = load_schemas(cfg["gis_schema"])
    assert schemas["longitudinal_marks"].geometry_type == "LineStringZ"
    assert schemas["other_marks"].geometry_type == "PolygonZ"
    assert schemas["longitudinal_marks"].field_names == [
        "ID_SIG",
        "POSICAO",
        "DESIGNACAO",
        "COD_JAE",
        "LARGURA",
        "TRAÇO",
        "ESPAÇO",
        "ID_GRUPO",
        "COMENTARIO",
        "ID_HIST",
        "OBS",
    ]
    assert schemas["other_marks"].field_names == [
        "ID_SIG",
        "POSICAO",
        "COD_RST",
        "COD_JAE",
        "ALGARISMO",
        "LETRA",
        "LARGURA",
        "ALTURA",
        "AREA_PINTADA",
        "ID_GRUPO",
        "COMENTARIO",
        "ID_HIST",
        "OBS",
    ]
