"""Configuration loading and validation.

The main file (``configs/default.yaml``) lists sub-configs under ``includes``; each is
loaded into the matching top-level key. Paths in ``paths.yaml`` are resolved against the
project root, or ``ROADMARKING_DATA_ROOT`` when set. No absolute paths are hardcoded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "default.yaml"
EXPECTED_CHECKPOINT = "facebook/mask2former-swin-small-coco-instance"

ENV_DATA_ROOT = "ROADMARKING_DATA_ROOT"
ENV_DEVICE = "ROADMARKING_DEVICE"


class ConfigError(ValueError):
    """Raised when a configuration file is missing or invalid."""


@dataclass(frozen=True)
class Config:
    """Validated, merged project configuration."""

    raw: dict[str, Any]
    config_path: Path
    root: Path
    paths: dict[str, Path] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def device(self) -> str:
        return str(self.raw["runtime"]["device"])

    @property
    def checkpoint(self) -> str:
        return str(self.raw["model"]["model"]["checkpoint"])

    def path(self, key: str) -> Path:
        """Return a resolved path by dotted key, e.g. ``"data.raw_pointclouds"``."""
        try:
            return self.paths[key]
        except KeyError as exc:
            raise ConfigError(f"Unknown path key '{key}'. Known: {sorted(self.paths)}") from exc

    def provisional_sections(self) -> list[str]:
        """List dotted keys of sections flagged ``provisional: true``."""
        found: list[str] = []

        def walk(node: Any, prefix: str) -> None:
            if isinstance(node, dict):
                if node.get("provisional") is True:
                    found.append(prefix or "<root>")
                for k, v in node.items():
                    walk(v, f"{prefix}.{k}" if prefix else str(k))

        walk(self.raw, "")
        return found


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Top-level YAML in {path} must be a mapping")
    return data


def _flatten_paths(node: dict[str, Any], base: Path, prefix: str = "") -> dict[str, Path]:
    out: dict[str, Path] = {}
    for key, value in node.items():
        if key == "root":
            continue
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten_paths(value, base, dotted))
        elif isinstance(value, str):
            p = Path(value)
            if p.is_absolute():
                raise ConfigError(f"Path '{dotted}' must be relative, got absolute: {value}")
            out[dotted] = (base / p).resolve()
        else:
            raise ConfigError(f"Path '{dotted}' must be a string, got {type(value).__name__}")
    return out


def _require(mapping: dict[str, Any], keys: list[str], where: str) -> None:
    missing = [k for k in keys if k not in mapping]
    if missing:
        raise ConfigError(f"Missing keys in {where}: {missing}")


def validate(raw: dict[str, Any]) -> None:
    """Validate structural invariants of the merged configuration."""
    _require(
        raw, ["runtime", "tiling", "inference", "model", "classes", "gis_schema", "paths"], "config"
    )

    device = str(raw["runtime"].get("device", ""))
    if not (device == "cpu" or device.startswith("cuda")):
        raise ConfigError(f"runtime.device must be 'cpu' or 'cuda[:N]', got '{device}'")

    tiling = raw["tiling"]
    size, overlap = tiling.get("tile_size_px"), tiling.get("tile_overlap_px")
    if not isinstance(size, int) or size <= 0:
        raise ConfigError("tiling.tile_size_px must be a positive integer")
    if not isinstance(overlap, int) or not 0 <= overlap < size:
        raise ConfigError("tiling.tile_overlap_px must be an integer in [0, tile_size_px)")

    inf = raw["inference"]
    if not isinstance(inf.get("batch_size"), int) or inf["batch_size"] < 1:
        raise ConfigError("inference.batch_size must be an integer >= 1")
    for key in ("score_threshold", "mask_threshold"):
        val = inf.get(key)
        if not isinstance(val, int | float) or not 0.0 <= val <= 1.0:
            raise ConfigError(f"inference.{key} must be in [0, 1]")

    gsd = raw.get("rasterization", {}).get("ground_sampling_distance_m")
    if gsd is not None and (not isinstance(gsd, int | float) or gsd <= 0):
        raise ConfigError("rasterization.ground_sampling_distance_m must be > 0 or null")

    model = raw["model"].get("model", {})
    _require(model, ["checkpoint", "architecture"], "model.yaml:model")

    names = [c["name"] for c in raw["classes"].get("model_classes", [])]
    ids = [c["id"] for c in raw["classes"].get("model_classes", [])]
    if len(set(names)) != len(names) or len(set(ids)) != len(ids):
        raise ConfigError("classes.yaml: model class names and ids must be unique")
    known = set(names)
    for group in ("longitudinal_designations", "other_mark_codes"):
        for entry in raw["classes"].get(group, []):
            unknown = set(entry.get("model_classes", [])) - known
            if unknown:
                raise ConfigError(f"classes.yaml:{group} {entry} references unknown {unknown}")

    for layer, spec in raw["gis_schema"].get("layers", {}).items():
        _require(spec, ["geometry_type", "fields"], f"gis_schema.yaml:{layer}")
        if not spec["geometry_type"].endswith("Z"):
            raise ConfigError(f"gis_schema.yaml:{layer} geometry must be 3D (…Z)")


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load ``default.yaml`` plus its includes, apply env overrides, and validate."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    config_path = config_path.resolve()
    raw = _read_yaml(config_path)

    for key, rel in (raw.pop("includes", None) or {}).items():
        raw[key] = _read_yaml(config_path.parent / rel)

    if env_device := os.environ.get(ENV_DEVICE):
        raw["runtime"]["device"] = env_device

    validate(raw)

    root_setting = os.environ.get(ENV_DATA_ROOT) or raw["paths"].get("root")
    root = Path(root_setting).resolve() if root_setting else PROJECT_ROOT
    paths = _flatten_paths(raw["paths"], root)
    return Config(raw=raw, config_path=config_path, root=root, paths=paths)
