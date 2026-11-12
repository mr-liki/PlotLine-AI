"""Training dataset for Mask2Former fine-tuning (Phase 4/5, not implemented)."""

from __future__ import annotations

from pathlib import Path


class RoadMarkingInstanceDataset:
    """Intensity tiles + instance masks converted from manually verified annotations."""

    def __init__(self, root: Path, split: str) -> None:
        raise NotImplementedError("Annotation conversion / dataset is implemented in Phase 4.")
