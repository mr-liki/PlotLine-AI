"""Bounded real-data clean image delivery, separate from all vector diagnostics."""

# ruff: noqa: E501 -- complete report sentences and artifact descriptions.

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import rasterio
import yaml
from affine import Affine
from PIL import Image
from shapely.geometry import box
from shapely.ops import unary_union

from roadmarking.analysis.corridor_pilot import load_window, read_json
from roadmarking.io.pointcloud import read_header
from roadmarking.preprocessing.intensity import IntensityCalibration
from roadmarking.rasterization.pipeline import INTACT, MemoryMonitor, dump, sha256, write_tif
from roadmarking.rasterization.rasterizer import accumulate_intensity_window
from roadmarking.visualization.intensity_render import (
    grayscale_intensity,
    save_intensity_png,
    write_png_worldfile,
)
from roadmarking.visualization.mosaic import stitch_raster_windows

NAMESPACE = "phase3_intensity_images"
VERSION = "phase3-clean-intensity-v1"
DEFAULT_TILES = [
    "Job_20260325_0838_Track01_3__r0014592_c0014592",
    "Job_20260325_0838_Track01_3__r0014592_c0015360",
    "Job_20260325_0838_Track01_3__r0023808_c0026880",
    "Job_20260325_0838_Track01_3__r0024576_c0027648",
    "Job_20260325_0838_Track01_3__r0017664_c0017664",
    "Job_20260325_0838_Track01_3__r0003072_c0002304",
    "Job_20260325_0838_Track01_8__r0028416_c0016896",
    "Job_20260325_0838_Track01_8__r0028416_c0017664",
    "Job_20260325_0838_Track01_8__r0029952_c0018432",
]
MOSAICS = {
    "Track01_3_straight_section": DEFAULT_TILES[:2],
    "Track01_8_section": DEFAULT_TILES[6:8],
}
RASTER_NAMES = {
    "maximum": "raw_max_intensity.tif",
    "mean": "raw_mean_intensity.tif",
    "normalized": "normalized_intensity.tif",
    "observed": "observed_mask.tif",
    "count": "point_count.tif",
}


def source_stem(value):
    stem = Path(value).stem
    matches = [name for name in INTACT if name == stem or name.endswith(stem)]
    if len(matches) != 1:
        raise ValueError(f"Only approved intact Track01_3 / Track01_8 sources supported: {value}")
    return matches[0]


def _write_rasters(directory, layers, transform):
    paths = {}
    for key, name in RASTER_NAMES.items():
        path = directory / name
        write_tif(path, layers[key], transform, None, config_version=VERSION)
        with rasterio.open(path) as ds:
            if (
                ds.crs is not None
                or ds.transform != transform
                or not np.array_equal(ds.read(1), layers[key])
            ):
                raise ValueError("Raw GeoTIFF readback or source grid mismatch")
        paths[key] = str(path)
    return paths


def _png(path, layers, transform, gamma, *, rgb=False):
    pixels = save_intensity_png(
        path, layers["normalized"], layers["observed"], gamma=gamma, rgb=rgb
    )
    write_png_worldfile(path, transform)
    dump(
        path.with_suffix(".json"),
        {
            "artifact_role": "CLEAN_INTENSITY_IMAGE_NO_OVERLAYS",
            "affine": list(transform)[:6],
            "shape": list(pixels.shape),
            "mode": "RGB_REPLICATED_GRAYSCALE" if rgb else "GRAYSCALE_L",
            "png_nodata_display": 0,
            "nodata_distinguished_by": "observed_mask.tif",
            "display_gamma": gamma,
            "conversion": "round(255 * clip(file_calibrated_intensity,0,1)^gamma), missing=0",
            "crs_status": "CRS_UNCONFIRMED",
            "filled_pixels": 0,
        },
    )
    return str(path)


def _window(cache, stats, transform, shape, calibration, verified):
    bounds = [
        transform.c,
        transform.f + shape[0] * transform.e,
        transform.c + shape[1] * transform.a,
        transform.f,
    ]
    coverage = unary_union([box(*bounds_) for bounds_ in stats["identity"]["boxes"]])
    if not coverage.covers(box(*bounds)):
        raise ValueError("Window exceeds strictly validated cache coverage; refusing false NoData")
    size = stats["identity"]["partition_m"]
    lower = np.floor(np.asarray(bounds[:2]) / size).astype(int)
    upper = np.floor(np.asarray(bounds[2:]) / size).astype(int)
    for x in range(lower[0], upper[0] + 1):
        for y in range(lower[1], upper[1] + 1):
            name = f"p_{x}_{y}.bin"
            if name in stats["artifacts"] and (str(cache), name) not in verified:
                if sha256(cache / name) != stats["artifacts"][name]:
                    raise ValueError("Validated point-cache hash mismatch")
                verified.add((str(cache), name))
    # Tiny query padding avoids losing LAS-quantized boundary records before the
    # exact pixel-index snap. Accumulator cropping still enforces the true window.
    query = np.asarray(bounds) + np.array([-1, -1, 1, 1]) * 1e-6
    points = load_window(cache, query, size)
    layers, keep = accumulate_intensity_window(
        points["xyz"], points["intensity"], transform, shape, calibration
    )
    return layers, points["source_index"][keep].copy()


def _overlaps(tiles):
    checks = []
    for i, a in enumerate(tiles):
        for b in tiles[i + 1 :]:
            if a["source"] != b["source"]:
                continue
            r0, c0 = max(a["row_offset"], b["row_offset"]), max(a["col_offset"], b["col_offset"])
            r1, c1 = min(a["row_offset"] + 1024, b["row_offset"] + 1024), min(
                a["col_offset"] + 1024, b["col_offset"] + 1024
            )
            if r1 <= r0 or c1 <= c0:
                continue
            mismatches = []
            for name in RASTER_NAMES:
                with rasterio.open(a["rasters"][name]) as ds:
                    aa = ds.read(
                        1,
                        window=(
                            (r0 - a["row_offset"], r1 - a["row_offset"]),
                            (c0 - a["col_offset"], c1 - a["col_offset"]),
                        ),
                    )
                with rasterio.open(b["rasters"][name]) as ds:
                    bb = ds.read(
                        1,
                        window=(
                            (r0 - b["row_offset"], r1 - b["row_offset"]),
                            (c0 - b["col_offset"], c1 - b["col_offset"]),
                        ),
                    )
                if not np.array_equal(aa, bb):
                    mismatches.append(name)
            with Image.open(a["images"]["model_input"]) as image:
                aa = np.asarray(image)[
                    r0 - a["row_offset"] : r1 - a["row_offset"],
                    c0 - a["col_offset"] : c1 - a["col_offset"],
                ]
            with Image.open(b["images"]["model_input"]) as image:
                bb = np.asarray(image)[
                    r0 - b["row_offset"] : r1 - b["row_offset"],
                    c0 - b["col_offset"] : c1 - b["col_offset"],
                ]
            if not np.array_equal(aa, bb):
                mismatches.append("model_png")
            checks.append(
                {
                    "a": a["tile_id"],
                    "b": b["tile_id"],
                    "pixels_per_layer": int(aa.size),
                    "layers_checked": 6,
                    "mismatches": mismatches,
                }
            )
    return checks


def run(root, *, sources=None, tile_ids=None, bounds=None, limit=None, gamma=None):
    root = Path(root)
    settings = yaml.safe_load((root / "configs/phase3_intensity_images.yaml").read_text())
    if gamma is not None:
        settings["display_gamma"] = gamma
    if (
        settings["resolution"],
        settings["tile_size"],
        settings["overlap"],
        settings["context_size"],
    ) != (0.05, 1024, 256, 2048) or settings["gap_fill"]:
        raise ValueError("This image delivery keeps the original grid and disables all filling")
    previous = read_json(
        read_json(root / "outputs/reports/phase3_1_topodot_inspired/latest.json")["report"]
    )
    manifest = {
        t["tile_id"]: t
        for t in read_json(Path(previous["output_root"]) / "metadata/tile_manifest.json")
    }
    selected = [manifest[tid] for tid in (tile_ids or DEFAULT_TILES)]
    if sources:
        selected = [t for t in selected if t["source"] in {source_stem(s) for s in sources}]
    if bounds:
        if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            raise ValueError("Ordered native XY bounds required")
        selected = [
            t
            for t in manifest.values()
            if (not sources or t["source"] in {source_stem(s) for s in sources})
            and box(
                t["affine"][2], t["affine"][5] - 51.2, t["affine"][2] + 51.2, t["affine"][5]
            ).intersects(box(*bounds))
        ]
    if limit is not None:
        if limit < 1:
            raise ValueError("Positive tile limit required")
        selected = selected[:limit]
    if not selected:
        raise ValueError("No selected validated pilot tiles")
    code_files = [
        Path(__file__),
        root / "src/roadmarking/visualization/intensity_render.py",
        root / "src/roadmarking/visualization/mosaic.py",
        root / "src/roadmarking/rasterization/rasterizer.py",
        root / "src/roadmarking/preprocessing/intensity.py",
    ]
    identity = {
        "settings": settings,
        "tiles": [t["tile_id"] for t in selected],
        "source_hashes": {t["source"]: t["source_sha256"] for t in selected},
        "normalization": {t["source"]: t["normalization"] for t in selected},
        "code_hash": hashlib.sha256("".join(sha256(p) for p in code_files).encode()).hexdigest(),
    }
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    out = root / "data/processed" / NAMESPACE / run_id
    reports = root / "outputs/reports" / NAMESPACE
    completion = out / "metadata/completion.json"
    if completion.exists():
        saved = read_json(completion)
        if not all(
            Path(p).exists() and sha256(p) == digest for p, digest in saved["artifacts"].items()
        ):
            raise ValueError("Completed clean-image artifact integrity check failed")
        return saved["report"]
    if out.exists():
        raise ValueError("Incomplete image run exists; preserve it and use a changed identity")
    for name in ("rasters", "tiles", "previews", "mosaics", "metadata"):
        (out / name).mkdir(parents=True)
    (reports / "debug").mkdir(parents=True, exist_ok=True)
    source_info, protected = {}, {}
    old_paths = [
        root / "data/processed/phase3/metadata/tile_manifest.json",
        Path(previous["report_directory"]) / "report.json",
        Path(previous["output_root"]) / "metadata/tile_manifest.json",
        *Path(previous["report_directory"]).glob("*review.gpkg"),
    ]
    protected.update({str(p): sha256(p) for p in old_paths})
    for source in identity["source_hashes"]:
        path = root / "data/raw/pointclouds" / f"{source}.laz"
        digest = sha256(path)
        header = read_header(path)
        if digest != identity["source_hashes"][source] or header.point_count != INTACT[source]:
            raise ValueError("Approved intact source identity mismatch")
        protected[str(path)] = digest
        stats = previous["source_streams"][source]
        cache = next(
            p.parent
            for p in (root / "data/processed/phase3_1_topodot_inspired/point_cache").glob(
                "*/completion.json"
            )
            if read_json(p)["identity"] == stats["identity"]
        )
        source_info[source] = {
            "source_file": str(path),
            "source_sha256": digest,
            "previously_strictly_validated_source_points": INTACT[source],
            "strict_reader_reexecuted": False,
            "source_identity_reverified": True,
            "cache": str(cache),
            "stats": stats,
        }
    start = time.perf_counter()
    tiles, contexts = [], {}
    unique_records = {source: [] for source in source_info}
    verified = set()
    with MemoryMonitor() as monitor:
        for tile in selected:
            began = time.perf_counter()
            tid, source = tile["tile_id"], tile["source"]
            info = source_info[source]
            calibration = IntensityCalibration(**tile["normalization"])
            transform = Affine(*tile["affine"])
            layers, ids = _window(
                Path(info["cache"]), info["stats"], transform, (1024, 1024), calibration, verified
            )
            unique_records[source].append(ids)
            paths = _write_rasters(out / "rasters" / tid, layers, transform)
            images = {
                "preview": _png(
                    out / "previews" / tid / "intensity_preview.png",
                    layers,
                    transform,
                    settings["display_gamma"],
                ),
                "model_input": _png(
                    out / "tiles" / tid / "intensity_model_input.png",
                    layers,
                    transform,
                    settings["display_gamma"],
                ),
            }
            if settings["replicated_rgb"]:
                images["model_input_rgb"] = _png(
                    out / "tiles" / tid / "intensity_model_input_rgb.png",
                    layers,
                    transform,
                    settings["display_gamma"],
                    rgb=True,
                )
            observed = layers["observed"] > 0
            pixels = grayscale_intensity(
                layers["normalized"], observed, gamma=settings["display_gamma"]
            )
            stats = tile["normalization"]["statistics"]
            bright = observed & (layers["maximum"] >= stats["p99"])
            background = (
                observed & (layers["maximum"] >= stats["p25"]) & (layers["maximum"] <= stats["p75"])
            )
            metrics = {
                "source_points_in_tile": len(ids),
                "observed_pixels": int(observed.sum()),
                "nodata_pct": float(100 * (~observed).mean()),
                "bright_p99_proxy_pixels": int(bright.sum()),
                "bright_proxy_median_gray": (
                    float(np.median(pixels[bright])) if bright.any() else None
                ),
                "background_p25_p75_median_gray": (
                    float(np.median(pixels[background])) if background.any() else None
                ),
                "observed_saturated_display_pct": (
                    float(100 * (pixels[observed] == 255).mean()) if observed.any() else 0
                ),
                "png_equals_exact_intensity_transfer": True,
                "filled_pixels": 0,
                "vector_inputs_to_renderer": 0,
                "burned_overlays": False,
            }
            # Product B has a larger field of view at the SAME pixel resolution,
            # never an upscaled/blurred copy of the 1024 image.
            context_transform = transform * Affine.translation(-512, -512)
            context_layers, ids = _window(
                Path(info["cache"]),
                info["stats"],
                context_transform,
                (2048, 2048),
                calibration,
                verified,
            )
            unique_records[source].append(ids)
            context_paths = _write_rasters(
                out / "rasters" / "context" / tid, context_layers, context_transform
            )
            images["context_preview"] = _png(
                out / "previews" / tid / "intensity_context_preview.png",
                context_layers,
                context_transform,
                settings["display_gamma"],
            )
            contexts[tid] = {"transform": context_transform, "rasters": context_paths}
            entry = {
                "tile_id": tid,
                "source": source,
                "source_sha256": tile["source_sha256"],
                "row_offset": tile["row_offset"],
                "col_offset": tile["col_offset"],
                "affine": tile["affine"],
                "height": 1024,
                "width": 1024,
                "resolution": 0.05,
                "overlap": 256,
                "stride": 768,
                "rasters": paths,
                "images": images,
                "context_rasters": context_paths,
                "context_affine": list(context_transform)[:6],
                "context_shape": [2048, 2048],
                "normalization": tile["normalization"],
                "display_gamma": settings["display_gamma"],
                "crs_status": "CRS_UNCONFIRMED",
                "artifact_role": "FINAL_CLEAN_INTENSITY_IMAGE_PRODUCT",
                "training_labels_available": False,
                "metrics": metrics,
                "seconds": time.perf_counter() - began,
            }
            tiles.append(entry)
            dump(out / "metadata" / f"{tid}.json", entry)
            print(
                json.dumps(
                    {
                        "clean_image": tid,
                        "observed": metrics["observed_pixels"],
                        "bright_gray": metrics["bright_proxy_median_gray"],
                        "background_gray": metrics["background_p25_p75_median_gray"],
                    }
                ),
                flush=True,
            )
            del layers, context_layers, pixels
        mosaics = []
        for name, tids in MOSAICS.items():
            if not all(tid in contexts for tid in tids):
                continue
            merged = {}
            overlap = {}
            for key in RASTER_NAMES:
                windows = []
                for tid in tids:
                    with rasterio.open(contexts[tid]["rasters"][key]) as ds:
                        windows.append((ds.read(1), ds.transform))
                missing = -1 if key == "maximum" else -9999 if key in ("mean", "normalized") else 0
                merged[key], transform, covered, checks = stitch_raster_windows(
                    windows, missing_value=missing
                )
                overlap[key] = checks
            if not covered.all():
                raise ValueError("Context mosaic has an unprocessed hole")
            paths = _write_rasters(out / "rasters" / "mosaics" / name, merged, transform)
            image_path = _png(
                out / "mosaics" / name / "intensity_context_preview.png",
                merged,
                transform,
                settings["display_gamma"],
            )
            mosaics.append(
                {
                    "name": name,
                    "tile_ids": tids,
                    "shape": list(merged["maximum"].shape),
                    "affine": list(transform)[:6],
                    "rasters": paths,
                    "image": image_path,
                    "overlap_checks": overlap,
                    "resampling": False,
                    "overlays": False,
                }
            )
        overlap = _overlaps(tiles)
        if any(c["mismatches"] for c in overlap):
            raise ValueError("Clean intensity tile seam mismatch")
        for source, record_parts in unique_records.items():
            source_info[source]["unique_source_records_queried"] = len(
                np.unique(np.concatenate(record_parts))
            )
    changed = [path for path, digest in protected.items() if sha256(path) != digest]
    if changed:
        raise ValueError(f"Protected input changed: {changed}")
    inputs = {
        source: {k: v for k, v in info.items() if k not in ("stats", "cache")}
        for source, info in source_info.items()
    }
    report = {
        "status": "PROVISIONAL_PENDING_VISUAL_QA",
        "run_id": run_id,
        "identity": identity,
        "output_root": str(out),
        "inputs": inputs,
        "tiles": tiles,
        "mosaics": mosaics,
        "overlap_checks": overlap,
        "counts": {
            "road_focused_png_previews": len(tiles),
            "grayscale_model_input_images": len(tiles),
            "replicated_rgb_model_input_images": len(tiles) if settings["replicated_rgb"] else 0,
            "wider_context_previews": len(tiles),
            "mosaic_previews": len(mosaics),
            "total_clean_pngs": len(list(out.rglob("*.png"))),
        },
        "protected_hashes": protected,
        "protected_unchanged": True,
        "rendering": {
            "road_focused": "51.2 native-unit source-grid tile; all original context observations preserved",
            "wider_context": "102.4 native-unit field at 0.05 pixels, not enlarged tile imagery",
            "normalization": "Existing whole-file Phase 3 candidate-point p1/p99.5 calibration; fixed linear clipping; no tile histogram normalization",
            "display_gamma": settings["display_gamma"],
            "vectors": False,
            "overlays": False,
            "interpolation": False,
            "nodata_in_geotiff": -9999,
            "nodata_in_maximum": -1,
            "nodata_png_display": 0,
            "observed_mask_required_for_missingness": True,
        },
        "processing_seconds": time.perf_counter() - start,
        "peak_rss_bytes": monitor.peak,
        "full_survey_rasterized": False,
        "training_ready": False,
        "training_blocker": "Matching annotations / reviewed labels remain unavailable; image format is for later training",
        "visual_qa": None,
    }
    dump(out / "metadata/manifest.json", tiles)
    dump(out / "metadata/identity.json", identity)
    dump(
        reports / "debug" / "previous_diagnostic_products.json",
        {
            "role": "DEBUG_REVIEW_ONLY_NOT_FINAL_IMAGES",
            "previous_phase3_1_report": str(Path(previous["report_directory"]) / "report.md"),
            "previous_phase3_2_report": str(root / "outputs/reports/phase3_2/report.md"),
            "note": "No vector, polygon, legend or diagnostic overlay was passed to the clean image renderer.",
        },
    )
    report["data_output_bytes"] = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    artifacts = {str(p): sha256(p) for p in out.rglob("*") if p.is_file()}
    dump(completion, {"report": report, "artifacts": artifacts})
    reports.mkdir(parents=True, exist_ok=True)
    dump(reports / "report.json", report)
    dump(
        reports / "latest.json",
        {"run_id": run_id, "output_root": str(out), "report": str(reports / "report.json")},
    )
    write_report(root, report)
    return report


def write_report(root, report):
    directory = Path(root) / "outputs/reports" / NAMESPACE
    text = [
        "# Phase 3: clean LiDAR intensity image output\n",
        f"**{report['status']}** — run `{report['run_id']}`. Primary products are raster-only images; no final road-marking vectors were generated.\n",
        "## A. Input files processed\n",
        "| Source | Previously strictly validated full-file points | Unique original records queried in this image pilot |",
        "|---|---:|---:|",
    ]
    for source, data in report["inputs"].items():
        text.append(
            f"| {source} | {data['previously_strictly_validated_source_points']:,} | {data['unique_source_records_queried']:,} |"
        )
    text += [
        "\nExisting strict-reader point caches were hash-verified; source LAZ hashes and header counts were checked. The full files were not decoded again. Source counts above are validation provenance, not a claim of full-survey image processing. Previous outputs and original LAZ files are unchanged.\n",
        "## B. Raster generation\n",
        "0.05 native-unit pixels (provisional 5 cm), 1024² tiles, 256-pixel overlap / 768 stride, original source affines. Product A is a 51.2-unit road-focused window, Product B a 102.4-unit context window at the same resolution. No strict-road mask clips the observations. Raw maximum, raw mean, linear normalized intensity, observed mask and point count are separate GeoTIFFs. CRS remains unset.\n",
        f"Normalization reuses each source's whole-file Phase 3 p1/p99.5 geometry-candidate calibration. PNG = round(255 × normalized^{report['rendering']['display_gamma']}); missing observations = black. This fixed monotone transfer is identical across neighboring tiles, previews, RGB copies and mosaics. It changes contrast only, never raw values or spatial geometry. No per-tile histogram adjustment, smoothing, filling or paint interpolation.\n",
        "## C. Final image outputs\n",
        f"Counts: {report['counts']}. Primary folder: `{report['output_root']}`. Per-tile paths and geospatial metadata: metadata/manifest.json. PNG worldfiles and JSON metadata are separate from the image pixels. RGB images contain three identical intensity channels.\n",
        "| Tile | Clean image | Context image | Bright p99 proxy median gray / background p25–p75 median gray | NoData % |",
        "|---|---|---|---|---:|",
    ]
    for tile in report["tiles"]:
        metric = tile["metrics"]
        text.append(
            f"| {tile['tile_id']} | {tile['images']['preview']} | {tile['images']['context_preview']} | {metric['bright_proxy_median_gray']} / {metric['background_p25_p75_median_gray']} | {metric['nodata_pct']:.1f} |"
        )
    text += [
        "\n## D. Visual and numerical quality\n",
        f"Visual QA: {report.get('visual_qa')}. Every PNG is checked against the exact intensity-to-grayscale equation and decoded again; L/RGB modes have no palette or artificial colors. All raw GeoTIFF layers read back exactly. Overlap checks: {report['overlap_checks']}. Context mosaics use exact overlap equality with no averaging/resampling. NoData stays explicit in TIFF; black PNG pixels can also be genuinely dark observations, so observed_mask.tif is authoritative. Bright-intensity statistics are proxies, not verified paint labels. Real missing observations remain visible as gaps.\n",
        "## E. Comparison with old outputs\n",
        "Previous multi-panel figures and GeoPackages were debugging/review products. Their vectors were not part of the source intensity TIFFs. The new primary PNG writer uses only numerical intensity and observed arrays; it has no vector or drawing arguments, no axes, titles, legends, outlines or labels. No debug map is exported as a model-input image. Earlier diagnostics remain available separately through debug/previous_diagnostic_products.json.\n",
        "## F. Tests and performance\n",
        f"Image processing: {report['processing_seconds']:.1f} s; sampled peak RSS {report['peak_rss_bytes']/2**20:.1f} MiB; data output {report['data_output_bytes']/2**20:.1f} MiB. Cached-window scope differs from whole-file preprocessing; no full-survey speedup claimed. Validation: {report.get('validation','Pending final validation.json')}.\n",
        "## G. Final status\n",
        f"**{report['status']}** for the clean image-export scope. No geometric road-boundary or marking accuracy is asserted. Images have the requested input format, but a labeled Mask2Former training dataset is still not ready: client annotations remain geographically unmatched. No model architecture change, weights download, training, new marking vectors or Git commit.\n",
    ]
    (directory / "report.md").write_text("\n".join(text), encoding="utf-8")
