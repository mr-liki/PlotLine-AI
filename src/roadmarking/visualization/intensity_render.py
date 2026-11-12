"""Clean intensity PNG export. No vectors, drawing, labels or plotting APIs.

Missing observations render black, while the separate observed mask distinguishes
them from genuinely measured dark pixels. One fixed transfer curve is shared by
all tiles and contexts; no per-image histogram or brightness adjustment is made.
"""

from pathlib import Path

import numpy as np
from PIL import Image


def grayscale_intensity(normalized, observed, *, gamma=0.7):
    image, valid = np.asarray(normalized), np.asarray(observed, bool)
    if image.ndim != 2 or image.shape != valid.shape:
        raise ValueError("Aligned 2D intensity and observed arrays required")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Positive finite transfer gamma required")
    if not np.isfinite(image[valid]).all():
        raise ValueError("Observed intensity must be finite")
    if ((image[valid] < 0) | (image[valid] > 1)).any():
        raise ValueError("Observed normalized intensity must be in [0,1]")
    result = np.zeros(image.shape, np.uint8)
    result[valid] = np.rint(255 * np.power(image[valid].astype(np.float64), gamma)).astype(np.uint8)
    return result


def save_intensity_png(path, normalized, observed, *, gamma=0.7, rgb=False):
    pixels = grayscale_intensity(normalized, observed, gamma=gamma)
    if rgb:
        pixels = np.repeat(pixels[:, :, None], 3, axis=2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path, format="PNG", compress_level=6)
    with Image.open(path) as image:
        if image.mode != ("RGB" if rgb else "L") or not np.array_equal(np.asarray(image), pixels):
            raise ValueError("PNG readback differs from exact raster transfer")
    return pixels


def write_png_worldfile(path, transform):
    """PNG georeferencing is a separate six-line pixel-center worldfile."""
    center_x, center_y = transform * (0.5, 0.5)
    values = [transform.a, transform.d, transform.b, transform.e, center_x, center_y]
    Path(path).with_suffix(".pgw").write_text(
        "\n".join(f"{v:.15g}" for v in values) + "\n", encoding="ascii"
    )
