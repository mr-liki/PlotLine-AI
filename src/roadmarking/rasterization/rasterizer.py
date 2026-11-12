"""North-up point binning, exact accumulation and conservative gap provenance."""

from dataclasses import dataclass

import numpy as np
from affine import Affine
from scipy import ndimage

NODATA = -9999.0


@dataclass
class GeoRaster:
    image: np.ndarray
    transform: Affine
    crs_wkt: str | None
    nodata: float | int | None = None


def north_up_transform(x_min, y_max, gsd):
    if not np.isfinite([x_min, y_max, gsd]).all() or gsd <= 0:
        raise ValueError("Finite origin and positive gsd required")
    return Affine(gsd, 0, x_min, 0, -gsd, y_max)


def pixel_to_world(transform, cols, rows):
    x, y = transform @ (np.asarray(cols, np.float64) + 0.5, np.asarray(rows, np.float64) + 0.5)
    return np.asarray(x), np.asarray(y)


def world_to_pixel(transform, xs, ys):
    # Subtract local origins before dividing, avoiding cancellation of large inverse terms.
    if transform.b == 0 and transform.d == 0:
        c = (np.asarray(xs, np.float64) - transform.c) / transform.a
        r = (np.asarray(ys, np.float64) - transform.f) / transform.e
    else:
        c, r = ~transform @ (np.asarray(xs, np.float64), np.asarray(ys, np.float64))
    # LAS quantization boundaries can differ by a few floating-point ULPs.
    c = np.where(np.abs(c - np.rint(c)) < 1e-8, np.rint(c), c)
    r = np.where(np.abs(r - np.rint(r)) < 1e-8, np.rint(r), r)
    return np.floor(c).astype(np.int64), np.floor(r).astype(np.int64)


class RasterAccumulator:
    """Bounded block: integer sums are invariant to chunking and point ordering."""

    def __init__(self, shape):
        self.shape = shape
        self.count = np.zeros(shape, np.uint32)
        self.total = np.zeros(shape, np.uint64)
        self.maximum = np.full(shape, -1, np.int32)
        self.saturated = np.zeros(shape, np.uint32)

    def add(self, cols, rows, intensity):
        cols, rows = np.asarray(cols), np.asarray(rows)
        raw = np.asarray(intensity, np.uint16)
        if ((cols < 0) | (cols >= self.shape[1]) | (rows < 0) | (rows >= self.shape[0])).any():
            raise ValueError("Point outside raster block")
        idx = rows * self.shape[1] + cols
        ids, inverse = np.unique(idx, return_inverse=True)
        count = np.bincount(inverse).astype(np.uint64)
        if (self.count.ravel()[ids].astype(np.uint64) + count > np.iinfo(np.uint32).max).any():
            raise OverflowError("Point count exceeds uint32")
        self.count.ravel()[ids] += count.astype(np.uint32)
        self.total.ravel()[ids] += np.bincount(inverse, weights=raw).astype(np.uint64)
        self.saturated.ravel()[ids] += np.bincount(inverse, weights=raw == 65535).astype(np.uint32)
        np.maximum.at(self.maximum.ravel(), idx, raw.astype(np.int32))

    def layers(self, calibration):
        observed = self.count > 0
        mean = np.full(self.shape, NODATA, np.float32)
        np.divide(self.total, self.count, out=mean, where=observed)
        normalized = np.where(observed, calibration.apply(self.maximum), NODATA).astype(np.float32)
        return dict(
            maximum=self.maximum,
            mean=mean,
            count=self.count,
            observed=observed.astype(np.uint8),
            normalized=normalized,
            saturation=self.saturated,
        )


def fill_isolated(image, observed, accepted, *, enabled=True, max_neighbor_range=0.08):
    """Fill only enclosed single missing pixels with eight original observed neighbors.

    All nine pixels must be in confirmed road area. Neighbor contrast range <=0.08
    prevents interpolation across paint/background boundaries. Never uses prior fills.
    Call on a global block with one-pixel halo, then crop, for seam consistency.
    """
    observed, accepted = np.asarray(observed, bool), np.asarray(accepted, bool)
    filled = np.asarray(image).copy()
    provenance = np.zeros(observed.shape, np.uint8)
    if enabled:
        kernel = np.ones((3, 3), np.uint8)
        neighbors = ndimage.convolve(observed.astype(np.uint8), kernel, mode="constant", cval=0)
        interior = ndimage.minimum_filter(
            accepted.astype(np.uint8), size=3, mode="constant", cval=0
        )
        lo = ndimage.minimum_filter(
            np.where(observed, image, np.inf), size=3, mode="constant", cval=np.inf
        )
        hi = ndimage.maximum_filter(
            np.where(observed, image, -np.inf), size=3, mode="constant", cval=-np.inf
        )
        ok = ~observed & (neighbors == 8) & (interior == 1) & (hi - lo <= max_neighbor_range)
        sums = ndimage.convolve(
            np.where(observed, image, 0).astype(np.float64),
            kernel.astype(float),
            mode="constant",
            cval=0,
        )
        filled[ok] = (sums[ok] / 8).astype(filled.dtype)
        provenance[ok] = 1
    return filled, provenance


def rasterize_intensity(cloud, gsd, aggregation="max"):
    if aggregation not in ("max", "mean") or len(cloud) == 0:
        raise ValueError("Nonempty cloud and max/mean aggregation required")
    from roadmarking.preprocessing.intensity import calibrate_histogram

    x0 = np.floor(cloud.xyz[:, 0].min() / gsd) * gsd
    y0 = np.ceil(cloud.xyz[:, 1].max() / gsd) * gsd
    t = north_up_transform(x0, y0, gsd)
    c, r = world_to_pixel(t, cloud.xyz[:, 0], cloud.xyz[:, 1])
    shape = (int(r.max()) + 1, int(c.max()) + 1)
    if np.prod(shape) > 4_000_000:
        raise ValueError("Use sparse block pipeline for large rasters")
    acc = RasterAccumulator(shape)
    acc.add(c, r, cloud.intensity)
    layers = acc.layers(calibrate_histogram(np.bincount(cloud.intensity, minlength=65536)))
    return GeoRaster(
        layers["maximum" if aggregation == "max" else "mean"],
        t,
        cloud.header.crs_wkt,
        -1 if aggregation == "max" else NODATA,
    )


def rasterize_search_corridor(xyz, intensity, transform, shape, calibration, search, strict):
    """Preserve measured intensity independently of pavement classification.

    The two masks are aligned to an existing source grid. Strict is a geometric
    candidate only. This function does not assert manual road confirmation.
    """
    import shapely

    cols, rows = world_to_pixel(transform, xyz[:, 0], xyz[:, 1])
    inside = (cols >= 0) & (rows >= 0) & (cols < shape[1]) & (rows < shape[0])
    keep = inside & shapely.intersects_xy(search, xyz[:, 0], xyz[:, 1])
    acc = RasterAccumulator(shape)
    acc.add(cols[keep], rows[keep], intensity[keep])
    layers = acc.layers(calibration)
    rr, cc = np.indices(shape)
    x, y = pixel_to_world(transform, cc, rr)
    envelope = shapely.intersects_xy(search, x, y)
    strict_mask = shapely.intersects_xy(strict, x, y) & envelope
    layers.update(
        strict_surface_candidate=strict_mask.astype(np.uint8),
        confirmed_road=np.zeros(shape, np.uint8),
        uncertain_road=(envelope & ~strict_mask).astype(np.uint8),
        marking_search=envelope.astype(np.uint8),
        filled_mask=np.zeros(shape, np.uint8),
    )
    return layers, keep


def accumulate_intensity_window(xyz, intensity, transform, shape, calibration):
    """All valid observations on an existing lattice; no road/vector filtering."""
    cols, rows = world_to_pixel(transform, xyz[:, 0], xyz[:, 1])
    keep = (cols >= 0) & (rows >= 0) & (cols < shape[1]) & (rows < shape[0])
    accumulator = RasterAccumulator(shape)
    accumulator.add(cols[keep], rows[keep], np.asarray(intensity)[keep])
    return accumulator.layers(calibration), keep
