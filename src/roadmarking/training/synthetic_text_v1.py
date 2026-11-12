"""Version 1 phrase-level targets for the supplied synthetic road-text corpus."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from roadmarking.training.road_text_head import model_class_from_visual


def load_synthetic_text_sample(split_dir: Path, stem: str):
    """Return native intensity and valid-only targets for one generated phrase."""
    metadata = json.loads((split_dir / "metadata" / f"{stem}.json").read_text(encoding="utf-8"))
    if (
        metadata["synthetic"] is not True
        or metadata["split"] != split_dir.name
        or metadata["file"] != f"{stem}.png"
    ):
        raise ValueError("Synthetic sample provenance mismatch")
    rasters = []
    for folder in ("images", "masks", "clean_masks"):
        with Image.open(split_dir / folder / f"{stem}.png") as raster:
            if raster.mode != "L" or raster.size != (512, 512):
                raise ValueError("Expected native grayscale 512 x 512 raster")
            rasters.append(np.array(raster))
    intensity, raw, clean_raw = rasters
    if any(np.any((raster != 0) & (raster != 255)) for raster in (raw, clean_raw)):
        raise ValueError("Expected binary source masks")
    visible, intended = raw != 0, clean_raw != 0
    if (visible & ~intended).any() or bool(visible.any()) != metadata["text_present"]:
        raise ValueError("Visible target disagrees with source geometry or metadata")
    count = int(visible.any())
    return {
        "intensity": torch.from_numpy(intensity).float().div_(255),
        "mask_labels": torch.from_numpy(visible[None, ...][:count]).float(),
        "class_labels": torch.full((count,), model_class_from_visual(9), dtype=torch.long),
        "valid_mask": torch.from_numpy(~(intended & ~visible)),
        "metadata": {
            **metadata,
            "phrase_instance_id": f"{split_dir.name}/{stem}" if count else None,
            "character_instances": None,
            "physical_gsd": None,
        },
    }
