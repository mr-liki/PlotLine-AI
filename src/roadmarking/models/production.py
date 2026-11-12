"""Load the registered production road-marking model with hash verification.

The registry is ``configs/production_model.yaml``. ``active`` names the default model; every
entry records weight/config/processor SHA-256, class names and V1 thresholds. Loading refuses
any file whose hash differs from the registry, and loads weights strictly.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from roadmarking.config import PROJECT_ROOT

REGISTRY = PROJECT_ROOT / "configs" / "production_model.yaml"


class ProductionModelError(RuntimeError):
    """Registry missing, inconsistent, or a file's hash does not match."""


@dataclass(frozen=True)
class ProductionModel:
    name: str
    weights: Path
    processor: Path
    thresholds_v1: dict
    classes: dict
    entry: dict


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_registry(path: Path = REGISTRY) -> dict:
    if not path.is_file():
        raise ProductionModelError(f"Missing model registry: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data.get("active") not in data.get("models", {}):
        raise ProductionModelError("Registry 'active' does not name a registered model")
    return data


def resolve(name: str | None = None, path: Path = REGISTRY, verify: bool = True) -> ProductionModel:
    reg = read_registry(path)
    name = name or reg["active"]
    if name not in reg["models"]:
        raise ProductionModelError(f"Unknown model {name!r}")
    e = reg["models"][name]
    weights = PROJECT_ROOT / e["weights"]
    processor = PROJECT_ROOT / e["processor"]
    if verify:
        for file, key in (
            (weights / "model.safetensors", "weights_sha256"),
            (weights / "config.json", "config_sha256"),
            (processor / "preprocessor_config.json", "processor_sha256"),
        ):
            if not file.is_file():
                raise ProductionModelError(f"Missing registered file: {file}")
            if _sha256(file) != e[key]:
                raise ProductionModelError(f"Hash mismatch for {file}")
    return ProductionModel(
        name, weights, processor, dict(e["thresholds_v1"]), dict(e["classes"]), e
    )


def load_production_model(name: str | None = None, device: str = "cuda"):
    """Return (model, processor, ProductionModel) for the active (or named) registered model."""
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

    spec = resolve(name)
    model, info = Mask2FormerForUniversalSegmentation.from_pretrained(
        spec.weights, local_files_only=True, output_loading_info=True
    )
    if any(info.values()):
        raise ProductionModelError(f"Non-strict checkpoint load: {info}")
    if {int(k): v for k, v in model.config.id2label.items()} != {
        int(k): v for k, v in spec.classes.items()
    }:
        raise ProductionModelError("Checkpoint classes differ from the registry")
    processor = Mask2FormerImageProcessor.from_pretrained(spec.processor, local_files_only=True)
    return model.to(device).eval(), processor, spec


__all__ = [
    "REGISTRY",
    "ProductionModel",
    "ProductionModelError",
    "load_production_model",
    "read_registry",
    "resolve",
]
