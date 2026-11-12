"""Exact lattice stitching: one observed value per pixel, never averaging overlaps."""

import numpy as np
from affine import Affine


def stitch_raster_windows(windows, *, missing_value, dtype=None, max_pixels=20_000_000):
    """Windows are (2D array, affine). All overlap values must be identical.

    Uncovered pixels remain explicitly missing; no resampling, feathering or
    interpolation is performed. Different source frames/calibrations must not mix.
    """
    if not windows:
        raise ValueError("At least one raster window required")
    first = windows[0][1]
    if first.b or first.d or first.a <= 0 or first.e >= 0:
        raise ValueError("North-up native raster lattice required")
    offsets = []
    for image, transform in windows:
        if image.ndim != 2 or (transform.a, transform.b, transform.d, transform.e) != (
            first.a,
            0,
            0,
            first.e,
        ):
            raise ValueError("Different resolution or orientation")
        col = (transform.c - first.c) / first.a
        row = (transform.f - first.f) / first.e
        if abs(col - round(col)) > 1e-7 or abs(row - round(row)) > 1e-7:
            raise ValueError("Raster origins are not on the same pixel lattice")
        offsets.append((round(row), round(col)))
    r0, c0 = min(r for r, _ in offsets), min(c for _, c in offsets)
    height = (
        max(r + image.shape[0] for (image, _), (r, c) in zip(windows, offsets, strict=True)) - r0
    )
    width = (
        max(c + image.shape[1] for (image, _), (r, c) in zip(windows, offsets, strict=True)) - c0
    )
    if height * width > max_pixels:
        raise ValueError("Mosaic exceeds bounded-memory budget")
    result = np.full((height, width), missing_value, dtype=dtype or windows[0][0].dtype)
    covered = np.zeros((height, width), bool)
    checked = 0
    for (image, _), (row, col) in zip(windows, offsets, strict=True):
        rr, cc = row - r0, col - c0
        target = result[rr : rr + image.shape[0], cc : cc + image.shape[1]]
        previous = covered[rr : rr + image.shape[0], cc : cc + image.shape[1]]
        if not np.array_equal(target[previous], image[previous]):
            raise ValueError("Overlapping raster pixels disagree")
        checked += int(previous.sum())
        target[:] = image
        previous[:] = True
    transform = first * Affine.translation(c0, r0)
    return result, transform, covered, {"overlap_pixels_compared": checked, "resampled": False}
