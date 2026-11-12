"""Figure helpers for the LiDAR inspection report (matplotlib, Agg backend, PNG output).

Conventions: observed no-data pixels are drawn in a fixed magenta so they can never be
confused with dark asphalt; display-only filled pixels are hatched/flagged separately.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

NODATA_RGB = (1.0, 0.0, 1.0)  # magenta


def _extent(transform, shape) -> list[float]:
    """imshow extent [left, right, bottom, top] in world coordinates."""
    left, top = transform.c, transform.f
    right = left + transform.a * shape[1]
    bottom = top + transform.e * shape[0]
    return [left, right, bottom, top]


def gray_with_nodata(
    scaled: np.ndarray, nodata: np.ndarray, filled: np.ndarray | None = None
) -> np.ndarray:
    """RGB image: grey = observed value, magenta = no data, cyan tint = display fill."""
    g = np.nan_to_num(scaled, nan=0.0)
    rgb = np.dstack([g, g, g])
    rgb[nodata] = NODATA_RGB
    if filled is not None:
        f = filled & nodata
        rgb[f] = np.dstack([g, g, g])[f] * 0.6 + np.array([0.0, 0.4, 0.4])
    return rgb


def save_overview(
    path: Path, panels: list[tuple[str, np.ndarray, dict]], transform, title: str
) -> None:
    """Row of georeferenced map panels sharing the same world extent."""
    n = len(panels)
    shape = panels[0][1].shape[:2]
    aspect = shape[0] / max(shape[1], 1)
    fig_w = 5.0 * n
    fig, axes = plt.subplots(1, n, figsize=(fig_w, max(3.0, 5.0 * aspect) + 0.8), squeeze=False)
    ext = _extent(transform, shape)
    for ax, (name, img, kw) in zip(axes[0], panels, strict=True):
        im = ax.imshow(img, extent=ext, interpolation="nearest", **kw)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=6)
        ax.ticklabel_format(useOffset=False, style="plain")
        if img.ndim == 2:
            fig.colorbar(im, ax=ax, fraction=0.035)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_histograms(path: Path, hists: dict[str, np.ndarray], max_value: int, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    bins = np.arange(0, max_value + 2)
    for name, h in hists.items():
        h = h[: max_value + 1]
        tot = h.sum()
        if tot == 0:
            continue
        axes[0].plot(bins[:-1], h / tot, lw=0.8, label=name)
        axes[1].plot(bins[:-1], np.cumsum(h) / tot, lw=0.8, label=name)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("intensity (raw)")
    axes[0].set_ylabel("fraction of points (log)")
    axes[1].set_xlabel("intensity (raw)")
    axes[1].set_ylabel("cumulative fraction")
    axes[1].legend(fontsize=6)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_density_hist(path: Path, series: dict[str, np.ndarray], title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    hi = max((np.percentile(v, 99.5) for v in series.values() if v.size), default=1.0)
    bins = np.linspace(0, hi, 120)
    for name, v in series.items():
        if v.size:
            ax.hist(v, bins=bins, histtype="step", density=True, label=name)
    ax.set_xlabel("points per m² (1 m cells)")
    ax.set_ylabel("density")
    ax.legend(fontsize=7)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_boxplot(path: Path, groups: dict[str, np.ndarray], ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(groups) + 2), 4.5))
    names = list(groups)
    ax.boxplot([groups[n] for n in names], showfliers=False, whis=(5, 95))
    ax.set_xticks(range(1, len(names) + 1), names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


__all__ = [
    "LogNorm",
    "gray_with_nodata",
    "save_boxplot",
    "save_density_hist",
    "save_histograms",
    "save_overview",
]
