"""Stage 4 — tiled instance-segmentation inference (Phase 6, not implemented)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roadmarking.rasterization.tiler import TileWindow


@dataclass
class InstancePrediction:
    mask: np.ndarray  # (H, W) bool, tile-local
    class_id: int
    score: float
    tile: TileWindow


class Predictor:
    """Runs a fine-tuned Mask2Former over georeferenced tiles."""

    def predict_tile(self, image: np.ndarray, tile: TileWindow) -> list[InstancePrediction]:
        raise NotImplementedError("Inference is implemented in Phase 6 (after fine-tuning).")
