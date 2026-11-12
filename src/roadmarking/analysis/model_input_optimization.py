"""Phase 3.3 read-only raster pilot; no LAZ reader, model download or training."""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from PIL import Image, ImageDraw

from roadmarking.preprocessing.road_focused_roi import road_focused_roi
from roadmarking.preprocessing.stripe_reference import local_contrast
from roadmarking.rasterization.model_tile_selector import select_model_tile
from roadmarking.rasterization.model_tiler import (
    candidate_origins,
    crop_sources,
    phase4_virtual_layers,
    pixel_keys,
    tile_transform,
    unique_pixel_count,
)
from roadmarking.rasterization.pipeline import MemoryMonitor, dump, sha256, write_tif
from roadmarking.visualization.intensity_render import save_intensity_png, write_png_worldfile

VERSION = "phase3.3-native-grid-v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def processor_probe():
    """Inspect the installed implementation only; never access the network."""
    import torch
    from transformers import Mask2FormerImageProcessor

    processor = Mask2FormerImageProcessor()
    result = {
        "processor": type(processor).__name__,
        "default_do_resize": processor.do_resize,
        "policy": {"do_resize": False},
        "shapes": {},
        "inference_seconds": None,
        "inference_vram_bytes": None,
        "inference_status": "NOT_MEASURED_NO_LOCAL_WEIGHTS",
    }
    for size in (512, 1024):
        image = np.zeros((size, size, 3), np.uint8)
        image[10:50, 31:34] = 255
        native = processor(images=image, do_resize=False, return_tensors="pt")
        default = processor(images=image, return_tensors="pt")
        result["shapes"][str(size)] = {
            "native": list(native["pixel_values"].shape),
            "default": list(default["pixel_values"].shape),
        }
        assert list(native["pixel_values"].shape) == [1, 3, size, size]
        expected = image.transpose(2, 0, 1).astype(np.float32) / 255
        expected = (expected - np.array(processor.image_mean)[:, None, None]) / np.array(
            processor.image_std
        )[:, None, None]
        np.testing.assert_allclose(native["pixel_values"][0].numpy(), expected, atol=1e-6)
        assert native["pixel_mask"].all()
    result["native_processor_pixel_audit"] = "EXACT_SCALE_STANDARD_NORMALIZATION"
    cache = (
        Path.home()
        / ".cache/huggingface/hub/models--facebook--mask2former-swin-small-coco-instance"
    )
    result["local_weight_files"] = [
        str(p) for p in cache.rglob("*") if p.suffix in {".bin", ".safetensors"}
    ]
    result["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        result["gpu"] = {"name": props.name, "total_memory_bytes": props.total_memory}
    return result


def coverage(layers):
    observed = layers["observed"].astype(bool)
    search = layers["marking_search"].astype(bool)
    proxy = layers["paint_proxy"].astype(bool)
    total = observed.size
    return {
        "total_pixels": total,
        "observed_pixels": int(observed.sum()),
        "observed_pct": float(100 * observed.mean()),
        "nodata_pct": float(100 * (~observed).mean()),
        "strict_pct": float(100 * layers["strict_surface_candidate"].mean()),
        "search_pct": float(100 * search.mean()),
        "observed_inside_search": int((observed & search).sum()),
        "observed_outside_search": int((observed & ~search).sum()),
        "unverified_background_candidate_pct": float(100 * (observed & ~search).sum() / total),
        "paint_proxy_pixels": int(proxy.sum()),
        "paint_proxy_pct": float(100 * proxy.mean()),
        "high_contrast_proxy_pixels": int(layers["high_contrast_proxy"].sum()),
    }


def diagnostic(root, tile, selections):
    """Downsampled review figures only; never used as model inputs."""
    with Image.open(tile["images"]["context_preview"]) as image:
        context = image.convert("RGB").resize((512, 512))
    with Image.open(tile["images"]["model_input"]) as image:
        original = image.convert("RGB").resize((512, 512))
    selection = original.copy()
    draw = ImageDraw.Draw(selection)
    row, col = tile["row_offset"], tile["col_offset"]
    for entry in selections:
        if entry["source"] != tile["source"]:
            continue
        r, c = entry["row_offset"] - row, entry["col_offset"] - col
        if r >= 1024 or c >= 1024 or r + 512 <= 0 or c + 512 <= 0:
            continue
        color = {"SKIP": "red", "KEEP": "lime", "KEEP_FOR_REVIEW": "orange"}[entry["decision"]]
        draw.rectangle((c / 2, r / 2, (c + 512) / 2 - 1, (r + 512) / 2 - 1), outline=color, width=2)
    panel = Image.new("RGB", (1536, 540), "black")
    for index, (image, label) in enumerate(
        (
            (context, "Original wide context"),
            (original, "Current native model region"),
            (selection, "512 selection: green keep / orange review / red empty"),
        )
    ):
        panel.paste(image, (index * 512, 28))
        ImageDraw.Draw(panel).text((index * 512 + 5, 7), label, fill="white")
    path = root / "quality_control" / f"{tile['tile_id']}_before_after_selection.png"
    panel.save(path)
    return str(path)


def run(root, *, margin_native=4.0):
    root = Path(root).resolve()
    old_report_path = root / "outputs/reports/phase3_intensity_images/report.json"
    previous = read(old_report_path)
    masks_path = (
        root
        / "data/processed/phase3_1_topodot_inspired/ac2f18fc1c69a346/metadata/tile_manifest.json"
    )
    masks = {t["tile_id"]: t for t in read(masks_path)}
    run_id = time.strftime("%Y%m%dT%H%M%S") + "_native_v1"
    out = root / "data/processed/phase3_3" / run_id
    report_dir = root / "outputs/reports/phase3_3"
    for folder in (
        "source_rasters",
        "model_tiles_512",
        "model_tiles_1024",
        "grayscale_previews",
        "context_previews",
        "metadata",
        "tile_selection",
        "quality_control",
    ):
        (out / folder).mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=True)
    protected_paths = {str(old_report_path), str(masks_path)}
    for folder in ("phase3", "phase3_2", "phase4"):
        protected_paths.update(
            str(p) for p in (root / "outputs/reports" / folder).rglob("*") if p.is_file()
        )
    original_manifest = root / "data/processed/phase3/metadata/tile_manifest.json"
    protected_paths.add(str(original_manifest))
    source_refs = []
    for tile in previous["tiles"]:
        for field in ("rasters", "context_rasters", "images"):
            protected_paths.update(tile[field].values())
        protected_paths.update(masks[tile["tile_id"]]["layers"].values())
        source_refs.append(
            {
                "tile_id": tile["tile_id"],
                "source": tile["source"],
                "source_sha256": tile["source_sha256"],
                "rasters": tile["rasters"],
                "normalization": tile["normalization"],
                "mask_layers": masks[tile["tile_id"]]["layers"],
            }
        )
    protected = {p: sha256(Path(p)) for p in sorted(protected_paths)}
    dump(out / "source_rasters/references.json", source_refs)
    dump(
        out / "context_previews/references.json",
        [t["images"]["context_preview"] for t in previous["tiles"]],
    )
    dump(
        out / "grayscale_previews/references.json",
        [t["images"]["preview"] for t in previous["tiles"]],
    )
    data, baselines, analyses = defaultdict(list), {}, []
    begin = time.perf_counter()
    monitor = MemoryMonitor()
    monitor.__enter__()
    try:
        for tile in previous["tiles"]:
            layers = {}
            paths = {
                **tile["rasters"],
                **{
                    k: masks[tile["tile_id"]]["layers"][k]
                    for k in ("marking_search", "strict_surface_candidate", "uncertain_road")
                },
            }
            transform = Affine(*tile["affine"])
            for name, path in paths.items():
                with rasterio.open(path) as ds:
                    if not ds.transform.almost_equals(transform) or ds.shape != (1024, 1024):
                        raise ValueError("Source layer alignment mismatch")
                    layers[name] = ds.read(1)
            layers["paint_proxy"] = (layers["observed"] > 0) & (layers["normalized"] >= 0.8)
            _, layers["high_contrast_proxy"] = local_contrast(
                layers["normalized"], layers["observed"].astype(bool), 0.05
            )
            evidence = road_focused_roi(
                layers["observed"],
                layers["marking_search"],
                layers["strict_surface_candidate"],
                layers["uncertain_road"],
                margin=margin_native,
                structures=layers["high_contrast_proxy"],
            )
            layers.update({k: v.astype(np.uint8) for k, v in evidence.items()})
            source = tile["source"]
            row, col = tile["row_offset"], tile["col_offset"]
            base = transform @ Affine.translation(-col, -row)
            if source in baselines and not baselines[source].almost_equals(base):
                raise ValueError("Inconsistent source grids")
            baselines[source] = base
            data[source].append((tile, layers))
            analyses.append(
                {
                    "tile_id": tile["tile_id"],
                    "source": source,
                    "affine": tile["affine"],
                    "native_extent": list(rasterio.transform.array_bounds(1024, 1024, transform)),
                    **coverage(layers),
                }
            )
        dump(out / "quality_control/existing_coverage.json", analyses)
        comparisons, all_manifests, all_selections = {}, {}, {}
        for size in (1024, 512):
            start = time.perf_counter()
            entries, selections, counts = [], [], defaultdict(list)
            selected_counts = defaultdict(list)
            for source, tiles in sorted(data.items()):
                registry = {
                    name: [(layers[name], t["row_offset"], t["col_offset"]) for t, layers in tiles]
                    for name in tiles[0][1]
                }
                regions = [(t["row_offset"], t["col_offset"], 1024, 1024) for t, _ in tiles]
                # A is precisely the current nine model inputs, not their 2048 contexts.
                origins = (
                    sorted({(r, c) for r, c, _, _ in regions})
                    if size == 1024
                    else candidate_origins(regions, 512, 128)
                )
                for row, col in origins:
                    layers = {}
                    for name, sources in registry.items():
                        fill = (
                            -9999
                            if name in {"normalized", "mean"}
                            else -1 if name == "maximum" else 0
                        )
                        layers[name], valid = crop_sources(
                            sources,
                            row,
                            col,
                            size,
                            fill,
                            union=name
                            in {
                                "marking_search",
                                "strict_surface_candidate",
                                "uncertain_road",
                                "roi",
                                "evidence",
                                "unknown",
                                "high_contrast_proxy",
                            },
                        )
                    observed = layers["observed"].astype(bool)
                    search = layers["marking_search"].astype(bool)
                    decision = select_model_tile(
                        observed,
                        valid,
                        search,
                        layers["evidence"],
                        layers["unknown"],
                        uncertain=layers["uncertain_road"],
                    )
                    ident = f"{source}__s{size}_r{row:07d}_c{col:07d}"
                    transform = tile_transform(baselines[source], row, col)
                    entry = {
                        "tile_id": ident,
                        "source": source,
                        "source_sha256": tiles[0][0]["source_sha256"],
                        "source_file": previous["inputs"][source]["source_file"],
                        "height": size,
                        "width": size,
                        "resolution": 0.05,
                        "row_offset": row,
                        "col_offset": col,
                        "affine": list(transform)[:6],
                        "bounds": list(rasterio.transform.array_bounds(size, size, transform)),
                        "overlap": size // 4,
                        "stride": size * 3 // 4,
                        "crs_status": "CRS_UNCONFIRMED",
                        "configuration_version": VERSION,
                        "padding_pixels": int((~valid).sum()),
                        "source_windows": [
                            {
                                "source_raster_id": t["tile_id"],
                                "rasters": t["rasters"],
                                "row_offset": t["row_offset"],
                                "col_offset": t["col_offset"],
                            }
                            for t, _ in tiles
                            if row < t["row_offset"] + 1024
                            and row + size > t["row_offset"]
                            and col < t["col_offset"] + 1024
                            and col + size > t["col_offset"]
                        ],
                        "normalization": tiles[0][0]["normalization"],
                        "display_gamma": 0.7,
                        "processor": {"do_resize": False},
                        **decision,
                    }
                    selections.append(entry)
                    for name, mask in {
                        "valid": valid,
                        "observed": observed,
                        "search_observed": observed & search,
                        "paint_proxy": layers["paint_proxy"].astype(bool),
                        "high_contrast_proxy": layers["high_contrast_proxy"].astype(bool),
                    }.items():
                        counts[(source, name)].append(pixel_keys(mask, row, col))
                        if decision["decision"] != "SKIP":
                            selected_counts[(source, name)].append(pixel_keys(mask, row, col))
                    if decision["decision"] == "SKIP":
                        continue
                    directory = out / f"model_tiles_{size}" / ident
                    directory.mkdir()
                    gray, rgb = (
                        directory / f"intensity_{size}.png",
                        directory / f"intensity_{size}_rgb.png",
                    )
                    expected = save_intensity_png(gray, layers["normalized"], observed)
                    save_intensity_png(rgb, layers["normalized"], observed, rgb=True)
                    for path in (gray, rgb):
                        write_png_worldfile(path, transform)
                    mask_paths = {}
                    for name, mask in (("observed", observed), ("valid", valid)):
                        path = directory / f"{name}_mask.tif"
                        write_tif(
                            path, mask.astype(np.uint8), transform, None, config_version=VERSION
                        )
                        mask_paths[name] = str(path)
                    with Image.open(rgb) as image:
                        actual = np.asarray(image)
                        if not all(np.array_equal(actual[:, :, i], expected) for i in range(3)):
                            raise ValueError("RGB pixel audit failed")
                    entry.update(
                        images={"grayscale": str(gray), "rgb": str(rgb)},
                        masks=mask_paths,
                        pixel_audit="EXACT_SOURCE_TRANSFER_ALL_CHANNELS",
                    )
                    entry = phase4_virtual_layers(entry, directory)
                    for name in ("maximum", "mean", "normalized", "count", "observed"):
                        with rasterio.open(entry["layers"][name]) as ds:
                            if not np.array_equal(ds.read(1), layers[name]):
                                raise ValueError(
                                    "Phase 4 virtual crop does not match source pixels"
                                )
                    dump(directory / "metadata.json", entry)
                    entries.append(entry)
            generation = time.perf_counter() - start
            loading = time.perf_counter()
            for entry in entries:
                with Image.open(entry["images"]["rgb"]) as image:
                    image.load()
            loading = time.perf_counter() - loading
            unique = {
                name: sum(
                    unique_pixel_count(chunks)
                    for (source, n), chunks in counts.items()
                    if n == name
                )
                for name in (
                    "valid",
                    "observed",
                    "search_observed",
                    "paint_proxy",
                    "high_contrast_proxy",
                )
            }
            retained = {
                name: sum(
                    unique_pixel_count(chunks)
                    for (source, n), chunks in selected_counts.items()
                    if n == name
                )
                for name in unique
            }
            comparisons[str(size)] = {
                "candidate_tiles": len(selections),
                "selected_tiles": len(entries),
                "skipped_tiles": len(selections) - len(entries),
                "keep_for_review": sum(e["decision"] == "KEEP_FOR_REVIEW" for e in entries),
                "unique_candidate_pixels": unique,
                "unique_retained_pixels": retained,
                "retention_pct": {
                    n: 100 * retained[n] / unique[n] if unique[n] else None for n in unique
                },
                "unique_selected_area_native_squared": retained["valid"] * 0.05**2,
                "model_input_pixels": len(entries) * size**2,
                "generation_seconds": generation,
                "rgb_load_seconds": loading,
                "peak_rss_bytes": monitor.peak,
                "disk_bytes": sum(
                    p.stat().st_size
                    for p in (out / f"model_tiles_{size}").rglob("*")
                    if p.is_file()
                ),
            }
            dump(out / f"model_tiles_{size}/tile_manifest.json", entries)
            from roadmarking.training.preflight import inspect_manifest

            preflight = inspect_manifest(out / f"model_tiles_{size}/tile_manifest.json")
            if preflight["errors"]:
                raise ValueError(preflight["errors"])
            comparisons[str(size)]["phase4_manifest_check"] = preflight
            comparisons[str(size)]["disk_bytes"] = sum(
                p.stat().st_size for p in (out / f"model_tiles_{size}").rglob("*") if p.is_file()
            )
            dump(out / f"tile_selection/selection_manifest_{size}.json", selections)
            all_manifests[str(size)], all_selections[str(size)] = entries, selections
            del counts, selected_counts
        for key in ("valid", "observed", "search_observed", "paint_proxy", "high_contrast_proxy"):
            if (
                comparisons["512"]["unique_candidate_pixels"][key]
                != comparisons["1024"]["unique_candidate_pixels"][key]
            ):
                raise ValueError("Benchmark regions do not have identical unique coverage")
        figures = [diagnostic(out, t, all_selections["512"]) for t in previous["tiles"]]
        dump(
            out / "quality_control/rejected_tile_review.json",
            [e for e in all_selections["512"] if e["decision"] == "SKIP"],
        )
        model = processor_probe()
        protected_unchanged = all(sha256(Path(p)) == value for p, value in protected.items())
        if not protected_unchanged:
            raise ValueError("Protected input changed")
        report = {
            "status": "PROVISIONAL",
            "run_id": run_id,
            "output_root": str(out),
            "version": VERSION,
            "existing_coverage": analyses,
            "comparison": comparisons,
            "model": model,
            "figures": figures,
            "protected_hashes": protected,
            "protected_unchanged": protected_unchanged,
            "processing_seconds": time.perf_counter() - begin,
            "peak_rss_bytes": monitor.peak,
            "rotation": False,
            "resampling": False,
            "margin_native_units": margin_native,
            "acceptance": {
                "spatial_workload_reduced": False,
                "observed_and_proxy_retention_complete": True,
                "human_marking_review_complete": False,
                "inference_performance_validated": False,
            },
            "recommendation": (
                "Retain existing 1024 source tiling; 512 experimental pending actual "
                "inference and aligned-label evaluation"
            ),
            "selection_limitation": (
                "No verified non-road exclusion boundary. Keep every observed ambiguous "
                "region; skip only zero-observation tiles."
            ),
            "paint_proxy_definition": (
                "observed normalized intensity >=0.8; high-intensity proxy, not verified "
                "paint or recognition accuracy"
            ),
            "high_contrast_proxy_definition": (
                "Existing stripe_reference.local_contrast, observed-weighted local mean/std; "
                "tile-boundary proposals unioned, not a semantic paint detector"
            ),
            "representativeness": (
                "Nine existing regions: straight, curves, dashes, continuous edges, old "
                "extraction boundaries, sparse/ambiguous observations and "
                "crossing/overpass-like context. Merge, faded-paint and at-grade "
                "intersection semantics unverified."
            ),
            "phase4": {
                "original_manifest": str(original_manifest),
                "original_manifest_sha256": protected[str(original_manifest)],
                "new_manifest_contract": (
                    "Full REQUIRED_LAYERS via exact source-window VRTs; preflight checked"
                ),
                "training_ready": False,
                "blocker": "No verified aligned annotations; CRS and native units unconfirmed",
            },
            "validation": {
                "real_tile_pixel_audit": "ALL_SELECTED_EXACT",
                "pytest": "PENDING",
                "health": "PENDING",
                "ruff": "PENDING",
                "black": "PENDING",
            },
        }
        dump(out / "metadata/tile_manifest.json", all_manifests)
        dump(out / "tile_selection/selection_manifest.json", all_selections)
        dump(out / "metadata/protected_hashes.json", protected)
        dump(report_dir / "report.json", report)
        dump(report_dir / "latest.json", {"output_root": str(out), "run_id": run_id})
        write_report(report, report_dir / "report.md")
        return report
    finally:
        monitor.__exit__(None, None, None)


def write_report(report, path):
    comparison = report["comparison"]
    text = [
        "# Phase 3.3 model input optimization",
        "",
        f"Status: **{report['status']}**. No training, model download or inference performed.",
        "",
        "## A. Existing image coverage analysis",
        "",
        (
            "Measured nine actual 1024 model inputs, not the 2048 display contexts. "
            "Per-tile extents and observed/search/strict/proxy counts are in "
            "report.json and quality_control/existing_coverage.json."
        ),
        "",
        "## B. Road-focused selection strategy",
        "",
        report["selection_limitation"],
        (
            f"Broad search union strict candidate + {report['margin_native_units']:g} "
            "native-unit margin, uncertainty "
            "and unknown observations. No fixed road width, brightness rejection or "
            "strict-only masking. Junction-like observations retained."
        ),
        "",
        "## C. Image dimensions and physical resolution",
        "",
        (
            "1024/256/768 vs 512/128/384 size/overlap/stride. Both 0.05 native units "
            "per pixel, native orientation. Units are not verified meters. No "
            "resampling or intensity alteration; fixed source normalization and display"
            " gamma 0.7."
        ),
        "Native footprints: 51.2 x 51.2 and 25.6 x 25.6 coordinate units respectively.",
        "",
        "## D. 512 vs 1024 tile comparison",
        "",
        (
            "A uses exactly the existing nine inputs. B enumerates globally aligned "
            "384-stride origins within their union, deduplicates IDs and pads outside "
            "available source windows. Same unique source region for both."
        ),
        "",
        (
            "| Size | Candidate | Selected | Skipped | Input pixels | Generation s | "
            "RGB load s | Disk MiB |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for size, c in comparison.items():
        text.append(
            f"| {size} | {c['candidate_tiles']} | {c['selected_tiles']} | {c['skipped_tiles']} "
            f"| {c['model_input_pixels']:,} | {c['generation_seconds']:.3f} "
            f"| {c['rgb_load_seconds']:.3f} | {c['disk_bytes']/2**20:.2f} |"
        )
    text += [
        "",
        "## E. Road-marking observation retention",
        "",
        (
            "Unique pixels are keyed by source identity and global row/column. All "
            "observed, observed-search and high-intensity proxy pixels retained in both"
            " configurations. This is not verified paint recall."
        ),
        report["paint_proxy_definition"],
        report["high_contrast_proxy_definition"],
        "",
        "## F. NoData and background handling",
        "",
        (
            "Observed masks distinguish genuine dark measurements from missing "
            "observations. Valid masks distinguish source footprint from padding. No "
            "fill. Potential background outside search remains unclassified and "
            "retained for review."
        ),
        "",
        "## G. Coordinate accuracy",
        "",
        (
            "Exact affine translation from the original common grid; pixel centers and "
            "inverse mapping retained, worldfiles alongside PNGs. Raw "
            "maximum/mean/count/normalized TIFFs referenced with source-window offsets,"
            " never duplicated. CRS unconfirmed; no EPSG guessed."
        ),
        "",
        "## H. Processing speed and memory",
        "",
        f"Pipeline sampled process peak RSS: {report['peak_rss_bytes']/2**20:.1f} MiB. "
        "Per-configuration RSS is cumulative process peak, not isolated memory demand. "
        "PNG decoding is a warm filesystem-cache measurement. Generation includes crop, "
        "export and readback, excludes processor probe and final hashing. "
        "CPU timings do not measure model speed.",
        "",
        "## I. Model compatibility",
        "",
        (
            "Installed Mask2FormerImageProcessor default shortest edge is 800; square "
            "defaults resize both inputs to 800. Explicit do_resize=False preserves "
            "512/1024 tensors. RGB channels repeat grayscale; processor normalization "
            "is separate from immutable source intensity. Valid/observed masks remain "
            "separate from processor padding masks. No cached weights: inference "
            "latency and VRAM unmeasured. See JSON GPU inventory."
        ),
        "",
        "## J. Visual comparisons",
        "",
        (
            "Diagnostic before/after/selection panels are separate from exact model "
            "images. Red rectangles mean empty observed tiles; orange requires review; "
            "green is retained evidence. They do not verify road semantics. Original "
            "contexts and grayscale preview references are provided; selection coverage"
            " is shown over every input region."
        ),
        "",
        "## K. Automated test results",
        "",
        "```json",
        json.dumps({k: v for k, v in report["validation"].items() if k != "healthcheck"}, indent=2),
        "```",
        "Health check details are stored in healthcheck.json and the JSON report.",
        "",
        "## L. Remaining limitations",
        "",
        report["representativeness"],
        (
            "No human-verified marking recall or model accuracy. Conservative retention"
            " may provide no workload reduction: report measured totals rather than "
            "presuming 512 is faster. Better background exclusion requires approved "
            "boundaries. No whole-survey claims."
        ),
        "",
        "## M. Final recommended input configuration",
        "",
        report["recommendation"],
        (
            "Finer tiling is useful for bounded per-image memory and annotation "
            "inspection but may increase total calls and padding; no inference-speed "
            "claim."
        ),
        "",
        "## N. Phase 4/5 readiness",
        "",
        (
            "Original Phase 4 raster manifest and prior results remain unchanged and "
            "usable by their existing consumers. New crop manifests expose the full "
            "required layers through source-window VRT references and pass the existing "
            "Phase 4 metadata preflight. Every virtual layer matches the source crop. "
            "Annotation alignment, CRS/units "
            "confirmation, attributed road/marking review, actual inference benchmarks "
            "and eventual trained-model accuracy remain required. Training is blocked."
        ),
        "",
        f"Output root: `{report['output_root']}`",
        "",
    ]
    Path(path).write_text("\n".join(text), encoding="utf-8")
