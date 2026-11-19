"""Overview images and an index table for a batch of digitization runs (review aid only)."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
import rasterio

ROOT = Path(__file__).resolve().parents[1]


def overview(run: Path):
    eng = run / "geopackage/digitization_engineering.gpkg"
    layers = [x[0] for x in pyogrio.list_layers(eng)]
    with rasterio.open(run / "rasters/intensity.tif") as ds:
        img, b = ds.read(1), ds.bounds
    fig, ax = plt.subplots(figsize=(13, 13))
    ax.imshow(img, cmap="gray", extent=[b.left, b.right, b.bottom, b.top], vmin=0, vmax=255)
    if "client_reference_lines_READONLY_COPY" in layers:
        ref = pyogrio.read_dataframe(eng, layer="client_reference_lines_READONLY_COPY")
        for g in ref.geometry:
            for part in getattr(g, "geoms", [g]):
                c = np.asarray(part.coords)
                ax.plot(c[:, 0], c[:, 1], color="yellow", lw=3, alpha=0.45)
    lines = pyogrio.read_dataframe(eng, layer="centerlines_refined_2d")
    for _, r in lines.iterrows():
        c = np.asarray(r.geometry.coords)
        flagged = bool(r.review_flags) and "occlusion_bridged" not in r.review_flags
        color = "red" if r.class_name.endswith("solid") else "cyan"
        ax.plot(c[:, 0], c[:, 1], color="orange" if flagged else color, lw=1.4)
    if "inferred_occlusion_bridges" in layers:
        for g in pyogrio.read_dataframe(eng, layer="inferred_occlusion_bridges").geometry:
            c = np.asarray(g.coords)
            ax.plot(c[:, 0], c[:, 1], color="magenta", lw=3, ls="--")
    if "other_markings_review_2d" in layers:
        for poly in pyogrio.read_dataframe(eng, layer="other_markings_review_2d").geometry:
            x, y = poly.exterior.xy
            ax.plot(x, y, color="lime", lw=1)
    ax.set_xlim(b.left, b.right)
    ax.set_ylim(b.bottom, b.top)
    ax.set_title(
        f"{run.name}: red solid, cyan dash, orange flagged, magenta inferred joins, yellow client"
    )
    fig.savefig(run / "reports/overview.png", dpi=70, bbox_inches="tight")
    plt.close(fig)


def main(runs):
    rows = []
    for run in map(Path, runs):
        q = json.loads((run / "qa/qa_summary.json").read_text())
        v = json.loads((run / "qgis_review/qgis_validation.json").read_text())
        overview(run)
        ref = q["reference"]
        rows.append(
            dict(
                run=run.name,
                solid=q["features"]["solid"],
                dash=q["features"]["dash"],
                final_3d=q["features"]["with_3d"],
                review=q["features"]["review_required"],
                joins=q["occlusion_bridging"]["bridges"],
                inferred_m=round(q["occlusion_bridging"]["inferred_length_m"], 1),
                solid_found=ref["solid"]["recall_length"],
                dash_found=ref["dash"]["recall_length"],
                solid_offset_cm=(
                    None
                    if ref["solid"]["lateral_median"] is None
                    else round(100 * ref["solid"]["lateral_median"], 1)
                ),
                dash_offset_cm=(
                    None
                    if ref["dash"]["lateral_median"] is None
                    else round(100 * ref["dash"]["lateral_median"], 1)
                ),
                z_vs_client_mm=(
                    None
                    if q["z"]["vs_client_z"]["median"] is None
                    else round(1000 * q["z"]["vs_client_z"]["median"], 1)
                ),
                observed=round(q["region"]["observed_fraction"], 2),
                seconds=round(q["performance"]["total_s"]),
                qgis_ok=v["all_valid"],
                other_recall=q.get("other_markings", {}).get("recall"),
                other_precision=q.get("other_markings", {}).get("precision"),
                flags=q["features"]["flags"],
            )
        )
    df = pd.DataFrame(rows)
    print(df.drop(columns="flags").to_string())
    for r in rows:
        print(r["run"], r["flags"])
    return df


if __name__ == "__main__":
    main(sys.argv[1:])
