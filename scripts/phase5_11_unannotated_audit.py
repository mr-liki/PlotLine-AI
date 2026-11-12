"""Phase 5.11: find bright observed paint-like blobs that NO client outline covers.

Diagnostic for review completeness (approval statement 3). Blobs are candidates for
reviewer attention only; nothing is labelled from them.
"""

import math
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import binary_closing, binary_dilation, label

from phase5_11_common import ROOT, load
from roadmarking.rasterization.pipeline import dump

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(phase="phase5_11"):
    latest = load(ROOT / "outputs/reports" / phase / "latest.json")
    out = Path(latest["report_dir"])
    spec = load(out / "new_region_candidates.json")
    paint = gpd.read_file(spec["candidate_store"], layer="candidate_paint_footprints")
    unsupported = gpd.read_file(spec["candidate_store"], layer="unsupported_marking_regions")
    blobs = []
    for tile in spec["tiles"]:
        name = tile["candidate_id"]
        with rasterio.open(tile["layers"]["normalized"]) as ds:
            gray, transform = ds.read(1).astype(float), ds.transform
        with rasterio.open(tile["layers"]["observed"]) as ds:
            observed = ds.read(1).astype(bool)
        shape = observed.shape
        covered = np.zeros(shape, bool)
        for frame in (
            paint[paint.candidate_id == name],
            unsupported[unsupported.candidate_id == name],
        ):
            if len(frame):
                covered |= rasterize(
                    [(g, 1) for g in frame.geometry], shape, transform=transform
                ).astype(bool)
        annotated = covered & observed & (paint.candidate_id == name).any()
        target = (
            rasterize(
                [(g, 1) for g in paint[paint.candidate_id == name].geometry],
                shape,
                transform=transform,
            ).astype(bool)
            & observed
        )
        threshold = np.percentile(gray[target], 25)
        bright = observed & (gray >= threshold)
        candidate = binary_closing(bright, iterations=2) & ~binary_dilation(covered, iterations=6)
        labels, count = label(candidate, structure=np.ones((3, 3)))
        for k in range(1, count + 1):
            m = labels == k
            area = int(m.sum())
            if area < 150:
                continue
            ys, xs = np.nonzero(m)
            h, w = int(np.ptp(ys)) + 1, int(np.ptp(xs)) + 1
            # Bright fraction of the blob's own observed pixels: paint is dense, foliage sparse.
            density = float(bright[m].mean())
            ring = binary_dilation(m, iterations=8) & ~binary_dilation(m, iterations=3) & observed
            contrast = (
                float(np.median(gray[m & observed]) - np.median(gray[ring])) if ring.any() else 0
            )
            x, y = transform * (xs.mean(), ys.mean())
            blobs.append(
                {
                    "region": name,
                    "pixels": area,
                    "bbox_px": [int(ys.min()), int(xs.min()), int(h), int(w)],
                    "centre_native": [float(x), float(y)],
                    "elongation": float(max(h, w) / max(1, min(h, w))),
                    "bright_density": density,
                    "contrast": contrast,
                    "_m": m,
                    "_gray": gray,
                    "_covered": covered,
                }
            )
        del annotated
    # Paint-like: dense bright return with clear local contrast (foliage/kerbs are sparse).
    keep = [b for b in blobs if b["bright_density"] >= 0.6 and b["contrast"] > 0]
    keep.sort(key=lambda b: -b["pixels"] * b["contrast"])
    shown = keep[:24]
    cols = 4
    rows = max(1, math.ceil(len(shown) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows), squeeze=False)
    for ax, b in zip(axes.flat, shown, strict=False):
        y0, x0, h, w = b["bbox_px"]
        pad = 40
        sl = (slice(max(0, y0 - pad), y0 + h + pad), slice(max(0, x0 - pad), x0 + w + pad))
        g = b["_gray"][sl]
        rgb = np.repeat(np.clip(g, 0, 1)[..., None], 3, -1)
        cov = b["_covered"][sl]
        rgb[cov & ~binary_dilation(~cov)] = (0.2, 0.5, 1.0)
        edge = b["_m"][sl] & ~binary_dilation(~b["_m"][sl])
        rgb[binary_dilation(edge)] = (1.0, 0.9, 0.0)
        ax.imshow(rgb)
        ax.set_title(
            f"{b['region']} @ ({b['centre_native'][0]:.1f}, {b['centre_native'][1]:.1f})"
            f"\n{b['pixels']} px, contrast {b['contrast']:.2f}",
            fontsize=8,
        )
        ax.axis("off")
    for ax in axes.flat[len(shown) :]:
        ax.axis("off")
    fig.suptitle("Unannotated bright observed blobs (yellow) vs client outlines (blue)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    (out / "galleries").mkdir(exist_ok=True)
    fig.savefig(out / "galleries" / "unannotated_bright_blobs.png", dpi=90)
    plt.close(fig)
    rows_out = [{k: v for k, v in b.items() if not k.startswith("_")} for b in keep]
    dump(
        out / "unannotated_paint_audit.json",
        {
            "status": "DIAGNOSTIC_FOR_REVIEW_ONLY",
            "candidates": rows_out,
            "all_blobs_considered": len(blobs),
        },
    )
    for i, b in enumerate(shown):
        print(
            i,
            b["region"],
            b["pixels"],
            round(b["contrast"], 3),
            round(b["elongation"], 1),
            [round(v, 1) for v in b["centre_native"]],
        )


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "phase5_11")
