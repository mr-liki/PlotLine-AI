"""Visual QA figures (matplotlib, Agg). Figures display native coordinates unchanged."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.colors import ListedColormap
from shapely.geometry import box
from skimage.segmentation import find_boundaries

from roadmarking.dataset.geometry import line_parts, polygon_parts

STATE_COLORS = ListedColormap(["#ff00ff", "#808080", "#202020", "#2ca02c", "#ffbf00", "#d62728"])


def _extent(tile):
    x0, y0, x1, y1 = tile["native_xy_bbox"]
    return (x0, x1, y0, y1)


def _draw_vector(ax, geom, color):
    geom = shapely.force_2d(geom)
    if geom.geom_type in ("LineString", "MultiLineString"):
        for p in line_parts(geom):
            xs, ys = p.xy
            ax.plot(xs, ys, color=color, lw=0.8)
    else:
        for p in polygon_parts(geom):
            for ring in (p.exterior, *p.interiors):
                xs, ys = ring.xy
                ax.plot(xs, ys, color=color, lw=0.8)


def tile_overlay(path: Path, tile: dict, image: np.ndarray, vectors: list, labels, title: str):
    """Intensity | intensity + original vectors | instances + boundaries | label state."""
    ext = _extent(tile)
    img = np.where(image < -1, np.nan, image)
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    cmap = plt.get_cmap("gray").with_extremes(bad="#ff00ff")
    for ax in axes[:3]:
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1, extent=ext, interpolation="nearest")
    axes[0].set_title("normalized intensity (magenta = no data)")
    for g in vectors:
        _draw_vector(axes[1], g, "#00e5ff")
    axes[1].set_title("original vectors (unmodified)")
    inst = labels.instance
    rng = np.random.default_rng(0)
    colors = rng.uniform(0.2, 1.0, size=(int(inst.max()) + 1, 3))
    rgba = np.zeros((*inst.shape, 4))
    rgba[inst > 0, :3] = colors[inst[inst > 0]]
    rgba[inst > 0, 3] = 0.6
    rgba[find_boundaries(inst, mode="inner")] = [1, 0, 0, 1]
    axes[2].imshow(rgba, extent=ext, interpolation="nearest")
    axes[2].set_title("instance masks (red = boundary)")
    axes[3].imshow(
        labels.state, cmap=STATE_COLORS, vmin=-0.5, vmax=5.5, extent=ext, interpolation="nearest"
    )
    axes[3].set_title(
        "label state + class id\n"
        "magenta no-data | grey unknown | green instance | amber boundary | red conflict",
        fontsize=9,
    )
    for rec in labels.instances:
        if rec["local_id"] and rec["pixels"]:
            rows, cols = np.nonzero(inst == rec["local_id"])
            a = tile["affine"]
            x = a[2] + (cols.mean() + 0.5) * a[0]
            y = a[5] + (rows.mean() + 0.5) * a[4]
            axes[3].text(x, y, str(rec["visual_id"]), color="white", fontsize=7, ha="center")
    for ax in axes:
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_xlabel("native X")
        ax.set_ylabel("native Y")
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.tick_params(labelsize=6)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def extent_overview(path: Path, comparison: dict, features_by_file: dict) -> None:
    """Native-coordinate plot of annotation layers vs raster footprints and LAZ headers."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    for ax_ in (ax, ax2):
        for i, (name, geoms) in enumerate(sorted(features_by_file.items())):
            pts = shapely.get_coordinates(shapely.centroid(geoms))
            ax_.scatter(
                pts[:, 0], pts[:, 1], s=0.3, color=palette[i % 4], label=f"annotation: {name}"
            )
        for name, ext in comparison["lidar_header_extents"].items():
            if ext["valid_header_bounds"]:
                (x0, y0, _), (x1, y1, _) = ext["xyz_min"], ext["xyz_max"]
                xs, ys = box(x0, y0, x1, y1).exterior.xy
                ax_.plot(xs, ys, color="#d62728", lw=1)
                ax_.text(x0, y1, name.replace("Job_", ""), fontsize=6, color="#d62728")
        for b in comparison["raster_bounds_by_source"].values():
            xs, ys = box(*b).exterior.xy
            ax_.fill(xs, ys, color="black", alpha=0.6)
    ax.set_title(
        "All annotation feature centroids, LAZ header extents (red), Phase 3 raster bbox (black)"
    )
    rb = comparison["raster_bounds"]
    pad = 2000
    ax2.set_xlim(rb[0] - pad, rb[2] + pad)
    ax2.set_ylim(rb[1] - pad, rb[3] + pad)
    ax2.set_title("Zoom on Phase 3 raster extent: no annotation coordinates present")
    for ax_ in (ax, ax2):
        ax_.set_aspect("equal")
        ax_.set_xlabel("native X (numbers as stored; frames not assumed equal)")
        ax_.set_ylabel("native Y")
        ax_.ticklabel_format(useOffset=False, style="plain")
    ax.legend(markerscale=20, fontsize=7, loc="lower left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
