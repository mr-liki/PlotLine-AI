"""The faster mask IoU and dash de-duplication give exactly the results of the original code
(kept in tests/reference/). Recognition output, and so precision and recall, cannot change."""

import numpy as np
import pytest

from roadmarking.inference.calibration import iou_matrix
from roadmarking.inference.stitching import stitch
from tests.reference.calibration_v1 import iou_matrix as iou_matrix_v1
from tests.reference.stitching_v1 import stitch as stitch_v1


def _blobs(rng, n, shape, max_size=60):
    out = np.zeros((n, *shape), bool)
    for k in range(n):
        h, w = rng.integers(1, max_size, 2)
        r, c = rng.integers(0, shape[0] - h), rng.integers(0, shape[1] - w)
        out[k, r : r + h, c : c + w] = rng.random((h, w)) < 0.8
    return out


@pytest.mark.parametrize("seed", range(6))
def test_iou_matrix_identical(seed):
    rng = np.random.default_rng(seed)
    a = _blobs(rng, int(rng.integers(0, 25)), (128, 128))
    b = _blobs(rng, int(rng.integers(1, 25)), (128, 128))
    valid = rng.random((128, 128)) < 0.9
    assert np.array_equal(iou_matrix(a, b, valid), iou_matrix_v1(a, b, valid))


def _scene(rng, n_tiles=4, per_tile=25):
    """Overlapping 512 px crops on a 896 px raster with solid and dash proposals, many of them
    duplicated across crops (as in production stitching)."""
    windows = {f"T{i}{j}": (i * 384, j * 384) for i in range(2) for j in range(2)}
    shape = (896, 896)
    masks, scores, classes, crop_ids = [], [], [], []
    base = []
    for _ in range(per_tile * n_tiles // 2):
        h, w = (
            (rng.integers(3, 6), rng.integers(40, 200))
            if rng.random() < 0.5
            else (
                rng.integers(30, 120),
                rng.integers(3, 6),
            )
        )
        r, c = rng.integers(0, shape[0] - h), rng.integers(0, shape[1] - w)
        base.append((r, c, h, w, int(rng.random() < 0.5)))
    for tid, (r0, c0) in windows.items():
        for r, c, h, w, cls in base:
            m = np.zeros(shape, bool)
            m[r : r + h, c : c + w] = rng.random((h, w)) < 0.9
            win = np.zeros(shape, bool)
            win[r0 : r0 + 512, c0 : c0 + 512] = True
            m &= win  # a stitched proposal is clipped to its crop
            if m.sum() < 4:
                continue
            masks.append(m)
            scores.append(float(rng.uniform(0.5, 1.0)))
            classes.append(cls)
            crop_ids.append(tid)
    valid = rng.random(shape) < 0.97
    return masks, scores, classes, crop_ids, windows, valid


@pytest.mark.parametrize("seed", range(4))
def test_stitch_identical(seed):
    rng = np.random.default_rng(100 + seed)
    masks, scores, classes, crop_ids, windows, valid = _scene(rng)
    thresholds = {"0": 0.6, "1": 0.7}
    new, audit_new = stitch(masks, scores, classes, crop_ids, windows, valid, thresholds)
    old, audit_old = stitch_v1(masks, scores, classes, crop_ids, windows, valid, thresholds)
    assert audit_new == audit_old
    assert len(new) == len(old)
    for x, y in zip(new, old, strict=True):
        assert x["members"] == y["members"] and x["class"] == y["class"]
        assert x["score"] == y["score"] and np.array_equal(x["mask"], y["mask"])
