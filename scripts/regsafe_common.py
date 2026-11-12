"""Regression-safe generalization sprint: shared paths, frozen-model loading, NoData helpers.

Nothing here modifies the frozen checkpoint, processor, thresholds or production stitching.
"""

from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from stage9d_common import ROOT, dump, frozen_checks, read, sha256, verify_frozen

BASE = ROOT / "outputs/reports/regression_safe_sprint"
MODELS = ROOT / "models/mask2former_swin_s"
GEOMETRY_RUN = ROOT / "outputs/reports/geometry_supervised_sprint/20261112T103243064909Z"
GEOMETRY_MODEL = MODELS / "geometry_sprint_20261112T103243064909Z"
C4_RUN = ROOT / "outputs/reports/stage9e/recipes/20261112T071004863987Z"
C5_RUN = ROOT / "outputs/reports/stage9f/training/20261112T074209709141Z"
C5_MODEL = MODELS / "stage9f_20261112T074209709141Z/C5_s9061"
SYN_PACKAGE = ROOT / "roadway_mixed_500_geometry_annotated/roadway_mixed_500_geometry_annotated"
SYN_RAW = ROOT / "roadway_mixed_raw_test_500"
QUAL_300 = ROOT / "roadway_mixed_raw_test_300_v2"
BOUNDARY_BAND_PX = 8


def frozen_paths():
    checks = frozen_checks()
    verify_frozen(checks)
    model_dir = Path(checks[0]["path"]).parent
    return checks, model_dir, model_dir.parent / "processor"


def load_processor():
    _, _, proc = frozen_paths()
    return Mask2FormerImageProcessor.from_pretrained(proc, local_files_only=True)


def load_model(path, device="cuda"):
    model, info = Mask2FormerForUniversalSegmentation.from_pretrained(
        path, local_files_only=True, output_loading_info=True
    )
    if any(info.values()):
        raise RuntimeError(f"Non-strict checkpoint load: {info}")
    return model.to(device).eval()


def nodata_mask(valid, rgb):
    """Unobserved support: invalid and zero intensity (Stage 9G: matches observation rasters)."""
    return ~valid & (rgb[..., 0] == 0)


def boundary_band(valid, nodata, width=BOUNDARY_BAND_PX):
    """NoData pixels within ``width`` px of observed valid support."""
    return nodata & binary_dilation(valid, iterations=width)


def tensor_stats(values):
    a = np.asarray(values, float)
    if not len(a):
        return None
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)), max=float(a.max()))


__all__ = [
    "BASE",
    "BOUNDARY_BAND_PX",
    "C4_RUN",
    "C5_MODEL",
    "C5_RUN",
    "GEOMETRY_MODEL",
    "GEOMETRY_RUN",
    "MODELS",
    "QUAL_300",
    "ROOT",
    "SYN_PACKAGE",
    "SYN_RAW",
    "boundary_band",
    "dump",
    "frozen_paths",
    "load_model",
    "load_processor",
    "nodata_mask",
    "read",
    "sha256",
    "tensor_stats",
    "torch",
    "verify_frozen",
]
