"""Full-file point QC and complete selected-source raster pilot, never training labels."""

import json
import time
from pathlib import Path

import numpy as np
from affine import Affine
from shapely.geometry import box

from roadmarking.dataset.annotation_workspace import WorkspaceMemoryMonitor
from roadmarking.io.pointcloud import ChunkReadReport, iter_laz_chunks
from roadmarking.io.source_inventory import fingerprint, safe_header
from roadmarking.preprocessing.intensity import calibrate_histogram
from roadmarking.rasterization.pipeline import dump, sha256, write_tif
from roadmarking.rasterization.rasterizer import RasterAccumulator, world_to_pixel
from roadmarking.visualization.intensity_render import save_intensity_png, write_png_worldfile


def run(root):
    started = time.perf_counter()
    old = json.loads(
        (root / "outputs/reports/phase4_corrected_source_alignment/report.json").read_text()
    )
    audit_pointer = root / "outputs/reports/phase5/source_audit_latest.json"
    if not audit_pointer.exists():
        raise ValueError("Run the live Phase 5 source inventory first")
    audit = json.loads(Path(json.loads(audit_pointer.read_text())["report"]).read_text())
    if audit["raw_root"] != old["raw_root"]:
        raise ValueError("Pilot references a different raw root")
    ident = time.strftime("%Y%m%dT%H%M%S") + "_full_point_pilot"
    output = root / "data/processed/phase5_training" / ident
    for folder in (
        "images",
        "instance_masks",
        "observed_masks",
        "metadata",
        "manifests",
        "splits",
        "qa_previews",
        "rasters",
    ):
        (output / folder).mkdir(parents=True, exist_ok=False)
    histogram = np.zeros(65536, np.uint64)
    scans, pending = [], []
    with WorkspaceMemoryMonitor() as memory:
        for entry in old["entries"]:
            source = Path(entry["source"])
            matching = [
                r
                for r in json.loads(
                    (Path(audit["report_directory"]) / "raw_integrity.json").read_text()
                )
                if r["path"] == str(source)
            ]
            if not matching or matching[0]["status"] != "SAMPLE_QC_PASS_FULL_SCAN_PENDING":
                raise ValueError("Selected source did not pass the live sample audit")
            before = fingerprint(source)
            header = safe_header(source)
            # Global 768-pixel lattice, shared by every source/pass. Two overlapping tiles.
            center_x = entry["affine"][2] + 25.6
            center_y = entry["affine"][5] - 25.6
            x0 = np.floor((center_x - 25.6) / 38.4) * 38.4
            y0 = np.ceil((center_y + 25.6) / 38.4) * 38.4
            transform = Affine(0.05, 0, x0, 0, -0.05, y0)
            acc = RasterAccumulator((1024, 1792))
            report = ChunkReadReport(source, 0, 0)
            classes, actual_min, actual_max = {}, np.full(3, np.inf), np.full(3, -np.inf)
            read_seconds, raster_seconds, inside_count = 0.0, 0.0, 0
            iterator = iter(iter_laz_chunks(source, report))
            progress = time.perf_counter()
            print(f"Full source scan: {entry['group']} / {source.name}", flush=True)
            while True:
                tick = time.perf_counter()
                try:
                    points = next(iterator)
                except StopIteration:
                    read_seconds += time.perf_counter() - tick
                    break
                read_seconds += time.perf_counter() - tick
                tick = time.perf_counter()
                xyz = np.column_stack([points.x, points.y, points.z])
                raw = np.asarray(points.intensity)
                histogram += np.bincount(raw, minlength=65536).astype(np.uint64)
                actual_min = np.minimum(actual_min, xyz.min(axis=0))
                actual_max = np.maximum(actual_max, xyz.max(axis=0))
                values, counts = np.unique(np.asarray(points.classification), return_counts=True)
                for value, count in zip(values, counts, strict=True):
                    classes[str(value)] = classes.get(str(value), 0) + int(count)
                cols, rows = world_to_pixel(transform, xyz[:, 0], xyz[:, 1])
                keep = (cols >= 0) & (rows >= 0) & (cols < 1792) & (rows < 1024)
                acc.add(cols[keep], rows[keep], raw[keep])
                inside_count += int(keep.sum())
                raster_seconds += time.perf_counter() - tick
                if time.perf_counter() - progress > 20:
                    print(
                        f"{entry['group']}: {report.points_read:,}/"
                        f"{header.point_count:,} QC points",
                        flush=True,
                    )
                    progress = time.perf_counter()
            after = fingerprint(source)
            if before != after:
                raise ValueError("Source changed during full point scan")
            complete = report.points_read == header.point_count and report.dropped_points == 0
            scan = {
                "path": str(source),
                "group": entry["group"],
                "fingerprint": before,
                "expected_points": int(header.point_count),
                "valid_points": report.points_read,
                "dropped_points": report.dropped_points,
                "bad_chunks": report.bad_chunks,
                "corrupt_chunks": report.corrupt_chunks,
                "full_integrity_passed": complete,
                "actual_mins": actual_min.tolist(),
                "actual_maxs": actual_max.tolist(),
                "classifications": classes,
                "read_and_qc_seconds": read_seconds,
                "accumulation_seconds": raster_seconds,
                "points_in_roi": inside_count,
            }
            scans.append(scan)
            pending.append((entry["group"], transform, acc, scan))
            dump(output / "metadata/full_source_integrity.json", scans)
        calibration = calibrate_histogram(histogram, low=1, high=99.5)
        dump(
            output / "metadata/shared_normalization.json",
            {
                **calibration.as_dict(),
                "scope": "ALL_VALID_POINTS_OF_THREE_FULL_PILOT_FILES",
                "gamma": 0.7,
                "per_tile_recalibration": False,
                "limitation": "Conservative all-return context; no confirmed road-only filter",
            },
        )
        tiles = []
        for group, transform, acc, scan in pending:
            layers = acc.layers(calibration)
            layers.update(
                valid=np.ones(acc.shape, np.uint8), filled_mask=np.zeros(acc.shape, np.uint8)
            )
            for col in (0, 768):
                tick = time.perf_counter()
                tid = f"{group}_r{round(-transform.f / .05)}_c{round(transform.c / .05) + col}"
                affine = transform @ Affine.translation(col, 0)
                rasters = output / "rasters" / tid
                rasters.mkdir()
                paths = {}
                for name, array in layers.items():
                    path = rasters / f"{name}.tif"
                    write_tif(path, array[:, col : col + 1024], affine, None)
                    paths[name] = str(path)
                gray, rgb = output / "images" / f"{tid}.png", output / "images" / f"{tid}_rgb.png"
                normalized = layers["normalized"][:, col : col + 1024]
                observed = layers["observed"][:, col : col + 1024]
                for path, colour in ((gray, False), (rgb, True)):
                    save_intensity_png(path, normalized, observed, gamma=0.7, rgb=colour)
                    write_png_worldfile(path, affine)
                bounds = [affine.c, affine.f - 51.2, affine.c + 51.2, affine.f]
                tile = {
                    "tile_id": tid,
                    "source": group,
                    "raw_source": scan["path"],
                    "source_fingerprint": scan["fingerprint"],
                    "bounds": bounds,
                    "width": 1024,
                    "height": 1024,
                    "affine": list(affine)[:6],
                    "resolution": 0.05,
                    "overlap": 256,
                    "stride": 768,
                    "layers": paths,
                    "masks": {"observed": paths["observed"], "valid": paths["valid"]},
                    "images": {"grayscale": str(gray), "rgb": str(rgb)},
                    "full_selected_source_integrity": scan["full_integrity_passed"],
                    "all_survey_passes_merged": False,
                    "production_scope": "COMPLETE_SELECTED_FILE_WITHIN_PILOT_ROI",
                    "observed_pixels": int(observed.sum()),
                    "unknown_pixels": int(observed.sum()),
                    "nodata_pixels": int((observed == 0).sum()),
                    "verified_background_pixels": 0,
                    "verified_paint_instances": 0,
                    "training_eligible": False,
                    "export_seconds": time.perf_counter() - tick,
                    "normalization": str(output / "metadata/shared_normalization.json"),
                    "image_sha256": sha256(rgb),
                }
                tiles.append(tile)
        # Duplicate observations from other source files are flagged for later mosaic/split audit.
        raw = json.loads((Path(audit["report_directory"]) / "raw_integrity.json").read_text())
        for tile in tiles:
            region = box(*tile["bounds"])
            tile["other_intersecting_source_footprints"] = [
                r["path"]
                for r in raw
                if r.get("mins")
                and r["path"] != tile["raw_source"]
                and box(r["mins"][0], r["mins"][1], r["maxs"][0], r["maxs"][1]).intersects(region)
            ]
        dump(output / "manifests/raster_tiles.json", tiles)
        dump(
            output / "metadata/dataset_manifest.json",
            {
                "status": "ANNOTATION_REVIEW_REQUIRED",
                "samples": [],
                "training_ready": False,
                "real_image_mask_pairs": 0,
                "candidate_raster_tiles": len(tiles),
                "reason": "No complete-region paint/background approvals in supplied sources",
            },
        )
        summary = {
            "output_root": str(output),
            "run_id": ident,
            "source_audit": str(audit_pointer),
            "full_source_scans": scans,
            "tiles": tiles,
            "total_valid_points": sum(x["valid_points"] for x in scans),
            "production_scope": "Three selected full files; two overlapping windows each",
            "entire_network_dataset_processed": False,
            "image_mask_pairs": 0,
            "shared_calibration": calibration.as_dict(),
            "training_started": False,
            "elapsed_seconds": time.perf_counter() - started,
            "sampled_peak_rss_bytes": memory.peak,
        }
    dump(output / "metadata/preparation_report.json", summary)
    dump(
        root / "outputs/reports/phase5/production_pilot_latest.json",
        {"output_root": str(output), "report": str(output / "metadata/preparation_report.json")},
    )
    print(
        json.dumps(
            {
                "output_root": str(output),
                "points": summary["total_valid_points"],
                "tiles": len(tiles),
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            indent=2,
        )
    )
    return summary


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[1])
