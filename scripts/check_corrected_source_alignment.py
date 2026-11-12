"""Native-grid point/intensity checks for the corrected survey share; no transforms."""

import json
import time
from pathlib import Path

import geopandas as gpd
import laspy
import matplotlib
import numpy as np
import shapely
from affine import Affine
from PIL import Image
from shapely.geometry import Point, box

from roadmarking.dataset.alignment import measure_line_displacement
from roadmarking.io.pointcloud import point_validity
from roadmarking.preprocessing.intensity import calibrate_histogram
from roadmarking.rasterization.pipeline import dump, write_tif
from roadmarking.rasterization.rasterizer import accumulate_intensity_window
from roadmarking.visualization.intensity_render import save_intensity_png, write_png_worldfile

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def line_parts(geometry):
    if geometry.geom_type == "LineString":
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [part for geom in geometry.geoms for part in line_parts(geom)]
    return []


def main():
    root = Path(__file__).resolve().parents[1]
    latest = json.loads((root / "outputs/reports/phase4_source_alignment/latest.json").read_text())
    audit = json.loads(Path(latest["report"]).read_text())
    if not audit["raw_root"].startswith(r"\\192.168.10.82"):
        raise ValueError("Run the corrected-share header audit first")
    headers = json.loads(
        (Path(audit["report_directory"]) / "raw_header_inventory.json").read_text()
    )
    all_lines = []
    for layer in audit["annotation_layers"]:
        if "Longitudinal" not in Path(layer["path"]).name:
            continue
        frame = gpd.read_file(
            layer["readable_inspection_copy"], layer=layer["layer"], engine="pyogrio"
        )
        all_lines.extend(
            geom
            for geom in shapely.force_2d(frame.geometry.array)
            if geom is not None and not geom.is_empty and geom.is_valid
        )
    tree = shapely.STRtree(all_lines)
    run = time.strftime("%Y%m%dT%H%M%S") + "_native_alignment_check"
    out = root / "data/processed/phase4_source_alignment" / run
    out.mkdir(parents=True)
    entries = []
    choices = [
        ("RIGHT", "Track01_11.laz"),
        ("LEFT", "Track02_11.laz"),
        ("INTERCHANGE", "Track03.laz"),
    ]
    for group, suffix in choices:
        candidates = [
            h
            for h in headers
            if Path(h["path"]).relative_to(Path(audit["raw_root"])).parts[0] == group
            and h["path"].endswith(suffix)
        ]
        header = candidates[0]
        print(
            f"Reading bounded native point sample: {group}/{Path(header['path']).name}", flush=True
        )
        clouds = []
        values = []
        windows = []
        rejected = []
        with laspy.open(
            header["path"], read_evlrs=False, laz_backend=laspy.LazBackend.Lazrs
        ) as reader:
            h = reader.header
            chunk = 50000
            start = max(0, (h.point_count // 2 // chunk) * chunk - 500000)
            for position in range(start, min(h.point_count, start + 1000000), chunk):
                try:
                    reader.seek(position)
                    points = reader.read_points(min(chunk, h.point_count - position))
                    ok = point_validity(
                        points,
                        h.mins,
                        h.maxs,
                        gps_week_time=not bool(h.global_encoding.gps_time_type),
                    )
                    if not len(points) or not ok.all():
                        rejected.append(
                            {
                                "start": position,
                                "reason": "POINT_QC_FAILED",
                                "bad": int((~ok).sum()),
                            }
                        )
                        continue
                    clouds.append(np.column_stack([points.x, points.y, points.z]))
                    values.append(np.asarray(points.intensity).copy())
                    windows.append({"start": position, "count": len(points)})
                except Exception as exc:
                    rejected.append({"start": position, "reason": f"{type(exc).__name__}: {exc}"})
            crs = h.parse_crs()
        if not clouds:
            entries.append(
                {
                    "group": group,
                    "source": header["path"],
                    "status": "NO_QC_PASSED_SAMPLE",
                    "rejected": rejected,
                }
            )
            continue
        xyz = np.concatenate(clouds)
        intensity = np.concatenate(values)
        center = xyz[len(xyz) // 2, :2]
        nearest = all_lines[int(tree.nearest(Point(center)))]
        center = nearest.interpolate(nearest.project(Point(center)))
        resolution = 0.05
        size = 1024
        x0 = np.floor(center.x / resolution) * resolution - size * resolution / 2
        y0 = np.ceil(center.y / resolution) * resolution + size * resolution / 2
        affine = Affine(resolution, 0, x0, 0, -resolution, y0)
        region = box(x0, y0 - size * resolution, x0 + size * resolution, y0)
        calibration = calibrate_histogram(np.bincount(intensity, minlength=65536), low=1, high=99.5)
        layers, keep = accumulate_intensity_window(
            xyz, intensity, affine, (size, size), calibration
        )
        folder = out / group
        folder.mkdir()
        paths = {}
        for name in ("maximum", "mean", "normalized", "count", "observed"):
            path = folder / f"{name}.tif"
            write_tif(path, layers[name], affine, crs)
            paths[name] = str(path)
        image = folder / "diagnostic_clean_intensity.png"
        save_intensity_png(image, layers["normalized"], layers["observed"])
        write_png_worldfile(image, affine)
        clipped = [
            part
            for index in tree.query(region, predicate="intersects")
            for part in line_parts(all_lines[int(index)].intersection(region))
            if part.length > 0.1
        ]
        visible = np.where(layers["observed"], layers["normalized"], np.nan)
        measured = measure_line_displacement(
            clipped, visible, affine, half_window=0.5, spacing=0.5, min_peak=0.1
        )
        extent = (x0, x0 + size * resolution, y0 - size * resolution, y0)
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        with Image.open(image) as source_image:
            pixels = np.asarray(source_image)
        for axis in axes:
            axis.imshow(
                pixels, cmap="gray", vmin=0, vmax=255, extent=extent, interpolation="nearest"
            )
            axis.set_xlim(extent[:2])
            axis.set_ylim(extent[2:])
            axis.ticklabel_format(style="plain", useOffset=False)
        axes[0].set_title("Actual observed intensity; bounded sample")
        axes[1].set_title("Original manual centerlines; NO XY transform")
        for line in clipped:
            coords = np.asarray(line.coords)
            axes[1].plot(coords[:, 0], coords[:, 1], color="#00e5ff", lw=0.7)
        fig.suptitle(f"{group}: native 0.05-unit pixels; " "no shift, rotation, rescale or warp")
        fig.tight_layout()
        preview = folder / "native_annotation_overlay.png"
        fig.savefig(preview, dpi=150)
        plt.close(fig)
        record = {
            "group": group,
            "source": header["path"],
            "status": "NATIVE_SAMPLE_EXPORTED_NO_TRANSFORM",
            "point_windows": windows,
            "rejected_point_windows": rejected,
            "sampled_valid_points": len(xyz),
            "points_in_raster": int(keep.sum()),
            "observed_pixels": int(layers["observed"].sum()),
            "affine": list(affine)[:6],
            "size": size,
            "resolution": resolution,
            "rasters": paths,
            "image": str(image),
            "overlay": str(preview),
            "centerline_parts": len(clipped),
            "lateral_peak_offset_proxy": measured,
            "diagnostic_normalization": calibration.as_dict(),
            "normalization_scope": (
                "bounded per-file sample; diagnostic only, not production calibration"
            ),
            "full_file_integrity_validated": False,
            "training_approved": False,
        }
        dump(folder / "metadata.json", record)
        entries.append(record)
        print(
            json.dumps(
                {"group": group, "observed": record["observed_pixels"], "profiles": measured},
                indent=2,
            ),
            flush=True,
        )
    report = {
        "status": "CORRECTED_SOURCE_NATIVE_ALIGNMENT_PILOT",
        "raw_root": audit["raw_root"],
        "annotation_root": audit["annotation_root"],
        "header_audit": latest["report"],
        "valid_annotation_features": sum(
            layer["valid_features"] for layer in audit["annotation_layers"]
        ),
        "native_header_intersections": audit["annotation_features_intersecting_headers"],
        "entries": entries,
        "coordinate_transform_applied": False,
        "training_started": False,
        "training_gate": (
            "Retained pending pilot review, invalid polygon correction, "
            "source integrity and label/loss validation"
        ),
        "limitations": [
            "Partial point windows, not full file coverage",
            "Offset proxies are not human annotation accuracy",
            "No new human VERIFIED status assigned",
            "No forced-fit annotations used",
        ],
    }
    report_dir = root / "outputs/reports/phase4_corrected_source_alignment"
    report_dir.mkdir(parents=True, exist_ok=True)
    dump(report_dir / "report.json", report)
    dump(
        report_dir / "latest.json",
        {"output_root": str(out), "report": str(report_dir / "report.json")},
    )
    print(out)


if __name__ == "__main__":
    main()
