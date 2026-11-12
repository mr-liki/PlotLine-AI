"""Mask2Former Swin-S integration via Hugging Face Transformers.

Phase 1 scope: load the pretrained COCO checkpoint as-is (no head replacement), in eval
mode, on a configurable device. The road-marking class head is configured during the
fine-tuning phase. The COCO checkpoint does NOT recognise road markings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from roadmarking.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_CHECKPOINT = "facebook/mask2former-swin-small-coco-instance"


class ModelLoadError(RuntimeError):
    """Raised when the processor or model cannot be loaded."""


@dataclass(frozen=True)
class ModelSpec:
    """Reproducible description of which checkpoint to load and where."""

    checkpoint: str = DEFAULT_CHECKPOINT
    revision: str = "main"
    device: str = "cuda:0"
    require_gpu: bool = True
    local_files_only: bool = False

    @classmethod
    def from_config(cls, cfg: Any, *, local_files_only: bool = False) -> ModelSpec:
        """Build from a :class:`roadmarking.config.Config`."""
        model = cfg["model"]["model"]
        return cls(
            checkpoint=model["checkpoint"],
            revision=model.get("revision", "main"),
            device=cfg["runtime"]["device"],
            require_gpu=bool(cfg["runtime"].get("require_gpu", True)),
            local_files_only=local_files_only,
        )


@dataclass
class LoadedModel:
    processor: Any
    model: Any
    device: Any
    spec: ModelSpec


def resolve_device(device: str, require_gpu: bool) -> Any:
    """Return a ``torch.device``; raise instead of silently falling back to CPU."""
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        if require_gpu:
            raise ModelLoadError(
                f"Device '{device}' requested but torch.cuda.is_available() is False "
                f"(torch {torch.__version__}, built for CUDA {torch.version.cuda})."
            )
        log.warning("CUDA unavailable; using CPU because require_gpu=False")
        return torch.device("cpu")
    return torch.device(device)


def load_config_only(
    checkpoint: str = DEFAULT_CHECKPOINT, revision: str = "main", local_files_only: bool = False
) -> Any:
    """Load only the model configuration (small JSON, no weights)."""
    from transformers import Mask2FormerConfig

    try:
        return Mask2FormerConfig.from_pretrained(
            checkpoint, revision=revision, local_files_only=local_files_only
        )
    except Exception as exc:
        raise ModelLoadError(f"Could not load config for '{checkpoint}': {exc}") from exc


def load_model(spec: ModelSpec) -> LoadedModel:
    """Load image processor and model weights, move to device, set eval mode.

    This downloads the pretrained weights (~280 MB) unless they are already cached or
    ``spec.local_files_only`` is True.
    """
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    device = resolve_device(spec.device, spec.require_gpu)
    log.info("Loading %s@%s on %s", spec.checkpoint, spec.revision, device)
    try:
        processor = AutoImageProcessor.from_pretrained(
            spec.checkpoint, revision=spec.revision, local_files_only=spec.local_files_only
        )
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            spec.checkpoint, revision=spec.revision, local_files_only=spec.local_files_only
        )
    except Exception as exc:
        raise ModelLoadError(f"Could not load '{spec.checkpoint}': {exc}") from exc

    model.to(device)
    model.eval()
    return LoadedModel(processor=processor, model=model, device=device, spec=spec)
