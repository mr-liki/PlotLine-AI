"""Exact uint16 histograms and shared robust contrast calibration."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class IntensityCalibration:
    low: float
    high: float
    percentile_low: float
    percentile_high: float
    statistics: dict
    method: str = "file-road-point percentile; linear clipped; no tile recalibration"

    def apply(self, image):
        # Minimum 1024 raw units prevents extreme gain on nearly constant captures.
        return np.clip(
            (np.asarray(image, np.float64) - self.low) / max(self.high - self.low, 1024), 0, 1
        ).astype(np.float32)

    def as_dict(self):
        return asdict(self)


def calibrate_histogram(hist, low=1.0, high=99.5):
    if not 0 <= low < high <= 100:
        raise ValueError("Percentiles must satisfy 0 <= low < high <= 100")
    hist = np.asarray(hist, np.uint64)
    total = int(hist.sum())
    if total == 0:
        raise ValueError("No road-candidate intensity measurements")
    cdf = hist.cumsum()

    def percentile(p):
        return int(np.searchsorted(cdf, max(1, int(np.ceil(p * total / 100)))))

    stats = {f"p{p}": percentile(p) for p in (1, 5, 25, 50, 75, 95, 99, 99.5, 100)}
    stats.update(
        points=total,
        saturated_points=int(hist[-1]),
        saturated_pct=100 * int(hist[-1]) / total,
        mean=float(np.dot(np.arange(65536, dtype=np.float64), hist) / total),
    )
    return IntensityCalibration(float(percentile(low)), float(percentile(high)), low, high, stats)


def normalize_intensity(intensity, method="percentile", low=1.0, high=99.0):
    raw = np.asarray(intensity)
    if method == "none":
        return raw.astype(np.float32)
    if raw.size == 0:
        return raw.astype(np.float32)
    if method == "minmax":
        lo, hi = float(raw.min()), float(raw.max())
    elif method == "percentile":
        if not 0 <= low < high <= 100:
            raise ValueError("Invalid percentiles")
        lo, hi = np.percentile(raw, [low, high])
    else:
        raise ValueError(f"Unknown intensity method: {method}")
    return np.clip((raw.astype(np.float64) - lo) / max(hi - lo, 1024), 0, 1).astype(np.float32)
