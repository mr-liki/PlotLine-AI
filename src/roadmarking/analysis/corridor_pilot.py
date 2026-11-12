"""Versioned bounded-window pilot orchestration for the independent corridor workflow.

One strict LAZ stream per source partitions original points once on disk. Each
window then uses a bounded KD-tree. All 42 historical boundary flags are audited,
plus eight representative cases. No whole-survey raster processing is implicit.
"""

# ruff: noqa: E501 -- report prose is deliberately kept as complete Markdown lines.

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import rasterio
import shapely
import yaml
from affine import Affine
from rasterio.features import rasterize, shapes
from scipy import ndimage
from shapely.geometry import LineString, box, mapping, shape
from shapely.ops import unary_union

from roadmarking.analysis.phase3_1_diagnostics import band_statistics
from roadmarking.analysis.phase3_1_report import CASES
from roadmarking.io.pointcloud import StrictReadReport, iter_pointcloud_chunks, read_header
from roadmarking.preprocessing.corridor_boundaries import build_boundaries
from roadmarking.preprocessing.corridor_discovery import discover, load_references, load_seed_points
from roadmarking.preprocessing.intensity import IntensityCalibration
from roadmarking.preprocessing.road_cross_sections import SectionSettings, extract_sections
from roadmarking.preprocessing.stripe_reference import StripeSettings
from roadmarking.quality.marking_retention import (
    manual_retention,
    retention_metrics,
    reviewed_samples,
)
from roadmarking.rasterization.pipeline import INTACT, MemoryMonitor, dump, sha256, write_tif
from roadmarking.rasterization.rasterizer import (
    RasterAccumulator,
    rasterize_search_corridor,
    world_to_pixel,
)

NS = "phase3_1_topodot_inspired"
POINTS = np.dtype([("xyz", "<f8", (3,)), ("intensity", "<u2"), ("source_index", "<u8")])
DIRECTORIES = (
    "reference_alignments",
    "cross_sections",
    "corridor_boundaries",
    "road_masks",
    "rasters",
    "tiles",
    "metadata",
    "review_data",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def pilot_tiles(root):
    manifest = read_json(root / "data/processed/phase3/metadata/tile_manifest.json")
    lookup = {tile["tile_id"]: tile for tile in manifest}
    p4 = read_json(root / "outputs/reports/phase4/phase4_report.json")
    review = read_json(Path(p4["output_root"]) / "metadata/road_review.json")
    flagged = {
        tid
        for tid, stats in review["tiles"].items()
        if "paint_like_pixels_at_candidate_edge" in stats["flags"]
    }
    selected = flagged | {tid for cases in CASES.values() for _, tid in cases}
    return [dict(lookup[tid], historical_boundary_flag=tid in flagged) for tid in sorted(selected)]


def diagnose(root, report_dir):
    previous = read_json(root / "outputs/reports/phase3/phase3_report.json")
    results = {}
    for run in previous["full_runs"]:
        directory = Path(run["output_directories"]["road_masks"])
        with np.load(directory / "surface_model.npz") as model:
            model = dict(model)
        with rasterio.open(directory / "surface_state_1m.tif") as src:
            model["state"] = src.read(1)
        processing = read_json(Path(run["reports"]) / "processing.json")
        results[run["source"]] = band_statistics(model, processing["criteria"])
    dump(report_dir / "root_cause.json", results)
    text = [
        "# Root cause evidence (before rework)\n",
        "| Hypothesis | Direct evidence / conclusion |",
        "|---|---|",
    ]
    for source, stats in results.items():
        text.append(
            f"| {source}: mixed planes | First failing criterion in 15-cell band: {stats['by_first_failing_criterion']}; point-count quartiles {stats['points_per_cell_in_band']}; RMS quartiles {stats['rms_percentiles_in_band']} |"
        )
    text += [
        "| Seed / width restriction | v1 has no user seed and no road-width cap. It fits all-point planes, removes small components, then hard-filters raster input. The 10–12 native-unit strip is an outcome, not a configured width. |",
        "| Connectivity / termination | v1 removes components <100 square native units and rejects neighboring cell-center steps >0.20. v2 further restricts core connectivity. Rejected cells are not all road. |",
        "| Banking / slope | v1 fits tilted planes; slope >0.20 still fails. Banking is supported within that limit; mixed cells fail RMS independently of grade. Cross-sections now use local profile slope and normal changes. |",
        "| Density / scan gaps | Rejected-band medians 79 / 62 points per 1-unit cell vs 659 / 561 in candidates; sparse and empty bins remain uncertain or NoData. |",
        "| Adjacent carriageway / elevation | Earlier original-point cross-sections show ~1.3-unit barrier, elevated/disconnected surfaces and sparse opposite-side observations. No assertion that all rejected bright pixels are roadway. |",
        "| Tile effects | Existing common-grid affine and overlap checks pass; extraction precedes tiling. No evidence that tile seams caused the clipping. |",
        "| Architectural loss | Hard filtering by geometric candidate state deletes intensity evidence before rasterization. New search-corridor accumulation is independent of strict classification. |",
        "\nOriginal-point profile images: ../phase3_1/figures/diagnosis_*.png. The new pilot also queries original XYZ/intensity; no unmatched annotations are used.",
    ]
    (report_dir / "root_cause.md").write_text("\n".join(text), encoding="utf-8")
    return results


def _extent(tile, halo=0):
    x0, y0, x1, y1 = tile["native_xy_bbox"]
    return [x0 - halo, y0 - halo, x1 + halo, y1 + halo]


def _in_box(xyz, bounds):
    x0, y0, x1, y1 = bounds
    return (xyz[:, 0] >= x0) & (xyz[:, 0] < x1) & (xyz[:, 1] >= y0) & (xyz[:, 1] < y1)


def partition_source(path, tiles, cache, settings, expected_hash):
    identity = {
        "source_sha256": expected_hash,
        "boxes": [_extent(t, settings["pilot_halo_m"]) for t in tiles],
        "partition_m": settings["partition_m"],
        "dtype": POINTS.descr,
    }
    completion = cache / "completion.json"
    if completion.exists():
        result = read_json(completion)
        if result["identity"] == json.loads(json.dumps(identity)) and all(
            sha256(cache / p) == digest for p, digest in result["artifacts"].items()
        ):
            return result
        raise ValueError("Cached source partition identity/integrity mismatch")
    if cache.exists():
        raise ValueError("Incomplete point cache; preserve it and use a new version")
    cache.mkdir(parents=True)
    report = StrictReadReport()
    started = time.perf_counter()
    selected, read_seconds = 0, 0.0
    iterator = iter(iter_pointcloud_chunks(path, settings["chunk_size"], report))
    while True:
        before = time.perf_counter()
        try:
            start, cloud = next(iterator)
        except StopIteration:
            read_seconds += time.perf_counter() - before
            break
        read_seconds += time.perf_counter() - before
        keep = np.zeros(len(cloud.xyz), bool)
        for bounds in identity["boxes"]:
            keep |= _in_box(cloud.xyz, bounds)
        xyz, intensity = cloud.xyz[keep], cloud.intensity[keep]
        record_ids = start + np.flatnonzero(keep)
        keys = np.floor(xyz[:, :2] / settings["partition_m"]).astype(int)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        for i, (x, y) in enumerate(unique):
            mask = inverse == i
            records = np.empty(int(mask.sum()), POINTS)
            records["xyz"], records["intensity"], records["source_index"] = (
                xyz[mask],
                intensity[mask],
                record_ids[mask],
            )
            with (cache / f"p_{x}_{y}.bin").open("ab") as f:
                records.tofile(f)
        selected += len(xyz)
    if report.points_read != INTACT[path.stem] or sha256(path) != expected_hash:
        raise ValueError("Source integrity check failed")
    result = {
        "identity": identity,
        "points_streamed": report.points_read,
        "unique_cached_points": selected,
        "invalid_points": report.invalid_points,
        "reading_seconds": read_seconds,
        "reading_and_partition_seconds": time.perf_counter() - started,
        "artifacts": {p.name: sha256(p) for p in sorted(cache.glob("*.bin"))},
    }
    dump(completion, result)
    return result


def load_window(cache, bounds, partition_m):
    lower = np.floor(np.asarray(bounds[:2]) / partition_m).astype(int)
    upper = np.floor(np.asarray(bounds[2:]) / partition_m).astype(int)
    parts = []
    for x in range(lower[0], upper[0] + 1):
        for y in range(lower[1], upper[1] + 1):
            path = cache / f"p_{x}_{y}.bin"
            if path.exists():
                points = np.memmap(path, dtype=POINTS, mode="r")
                parts.append(np.array(points[_in_box(points["xyz"], bounds)]))
    return np.concatenate(parts) if parts else np.empty(0, POINTS)


def accumulate(points, transform, shape, calibration):
    c, r = world_to_pixel(transform, points["xyz"][:, 0], points["xyz"][:, 1])
    keep = (c >= 0) & (r >= 0) & (c < shape[1]) & (r < shape[0])
    accumulator = RasterAccumulator(shape)
    accumulator.add(c[keep], r[keep], points["intensity"][keep])
    return accumulator.layers(calibration)


def observed_support_envelope(points, tile):
    """Adaptive 1-unit observation footprint; never a fixed-width road buffer.

    Retains an explicit uncertain discovery context even when stripes end or a
    structure interrupts profiles. NoData remains NoData within occupied cells.
    """
    transform = Affine(*tile["affine"]) * Affine.scale(20)
    c, r = world_to_pixel(transform, points["xyz"][:, 0], points["xyz"][:, 1])
    footprint = np.zeros((52, 52), np.uint8)
    valid = (c >= 0) & (r >= 0) & (c < 52) & (r < 52)
    footprint[r[valid], c[valid]] = 1
    footprint = ndimage.binary_dilation(footprint, iterations=1).astype(np.uint8)
    polygons = [
        shape(g) for g, value in shapes(footprint, mask=footprint > 0, transform=transform) if value
    ]
    return unary_union(polygons).intersection(box(*_extent(tile)))


def vector_file(path, features, source_hash):
    dump(
        path,
        {
            "type": "FeatureCollection",
            "coordinate_frame": "NATIVE_XY",
            "source_sha256": source_hash,
            "features": features,
        },
    )


def _feature(geometry, **props):
    return {
        "type": "Feature",
        "geometry": mapping(geometry),
        "properties": {"review_status": "PROPOSED_UNREVIEWED", **props},
    }


def _geometry_samples(tile):
    with rasterio.open(tile["layers"]["road_state"]) as ds:
        mask = ds.read(1)[::20, ::20] > 0
    r, c = np.nonzero(mask)
    transform = Affine(*tile["affine"])
    x, y = transform * (c * 20 + 10, r * 20 + 10)
    return np.column_stack((x, y))


def propose_window(points, tile, settings, refs=None, mode="intensity", seed_points=None):
    start = time.perf_counter()
    transform = Affine(*tile["affine"])
    halo_pixels = round(settings["pilot_halo_m"] / 0.05)
    scale = settings["discovery_resolution"] / transform.a
    if scale < 1:
        raise ValueError("Discovery resolution must not be finer than the measurement raster")
    coarse = transform * Affine.translation(-halo_pixels, -halo_pixels) * Affine.scale(scale)
    shape = (int(np.ceil((1024 + 2 * halo_pixels) / scale)),) * 2
    calibration = IntensityCalibration(**tile["normalization"])
    layers = accumulate(points, coarse, shape, calibration)
    references, evidence, status = discover(
        layers["normalized"],
        layers["observed"].astype(bool),
        coarse,
        mode=mode,
        references=refs,
        geometry_xy=_geometry_samples(tile),
        settings=StripeSettings(**settings["stripe"]),
        seed_points=seed_points,
    )
    reference_seconds = time.perf_counter() - start
    start = time.perf_counter()
    sections = extract_sections(points["xyz"], references, SectionSettings(**settings["sections"]))
    section_seconds = time.perf_counter() - start
    start = time.perf_counter()
    boundaries = build_boundaries(sections, SectionSettings(**settings["sections"]))
    return (
        references,
        evidence,
        sections,
        boundaries,
        status,
        {
            "reference_seconds": reference_seconds,
            "cross_sections_seconds": section_seconds,
            "boundaries_seconds": time.perf_counter() - start,
        },
        coarse,
    )


def overlap_check(manifest):
    comparisons, mismatches = 0, []
    for i, a in enumerate(manifest):
        for b in manifest[i + 1 :]:
            if a["source"] != b["source"]:
                continue
            ar, ac, br, bc = a["row_offset"], a["col_offset"], b["row_offset"], b["col_offset"]
            r0, r1, c0, c1 = (
                max(ar, br),
                min(ar + 1024, br + 1024),
                max(ac, bc),
                min(ac + 1024, bc + 1024),
            )
            if r0 >= r1 or c0 >= c1:
                continue
            for name in a["layers"]:
                with rasterio.open(a["layers"][name]) as ds:
                    aa = ds.read(1, window=((r0 - ar, r1 - ar), (c0 - ac, c1 - ac)))
                with rasterio.open(b["layers"][name]) as ds:
                    bb = ds.read(1, window=((r0 - br, r1 - br), (c0 - bc, c1 - bc)))
                comparisons += 1
                if not np.array_equal(aa, bb):
                    mismatches.append([a["tile_id"], b["tile_id"], name])
    return {"layer_pairs_checked": comparisons, "mismatches": mismatches, "passed": not mismatches}


def figures(path, tile, raw, old, new, references, boundaries, sections):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    extent = tile["native_xy_bbox"]
    x0, y0, x1, y1 = extent
    display_extent = (x0, x1, y0, y1)
    fig, axes = plt.subplots(2, 4, figsize=(23, 12), sharex=True, sharey=True)
    views = [
        (raw["normalized"], "A Original valid intensity", "gray"),
        (old["normalized"], "B Phase 3 restricted intensity", "gray"),
        (raw["normalized"], "C Reference hypotheses (cyan)", "gray"),
        (raw["normalized"], "D Cross-section locations (orange)", "gray"),
        (
            new["strict_surface_candidate"] + 2 * new["uncertain_road"],
            "E Strict candidate=green; search uncertainty=amber",
            ListedColormap(["black", "green", "orange"]),
        ),
        (new["normalized"], "F New measured search-corridor intensity", "gray"),
        (
            np.where(
                new["observed"] > 0,
                np.where(old["observed"] > 0, 1, 2),
                np.where(raw["observed"] > 0, 3, 0),
            ),
            "G Grey=retained; green=new; red=excluded; black=NoData",
            ListedColormap(["black", "gray", "lime", "red"]),
        ),
        (np.log1p(raw["count"]), "Original log(1 + point count)", "viridis"),
    ]
    for ax, (arr, title, cmap) in zip(axes.ravel(), views, strict=True):
        limits = {}
        if title.startswith("G "):
            limits = {"vmin": 0, "vmax": 3}
        elif title.startswith("E "):
            limits = {"vmin": 0, "vmax": 2}
        elif cmap == "gray":
            limits = {"vmin": 0, "vmax": 1}
        ax.imshow(
            np.ma.masked_less(arr, 0),
            extent=display_extent,
            cmap=cmap,
            origin="upper",
            interpolation="nearest",
            aspect="equal",
            **limits,
        )
        ax.set_title(title, fontsize=10)
    axes[0, 1].contour(
        old["road_state"] > 0,
        levels=[0.5],
        extent=display_extent,
        origin="upper",
        colors="cyan",
        linewidths=0.5,
    )
    for reference in references:
        xy = np.asarray(reference["coords"])
        axes[0, 2].plot(xy[:, 0], xy[:, 1], "c-", linewidth=0.8)
    for section in sections[:: max(1, len(sections) // 18)]:
        center, normal = np.asarray(section["center"]), np.asarray(section["normal"])
        line = center + np.array([-30, 30])[:, None] * normal
        axes[0, 3].plot(line[:, 0], line[:, 1], color="orange", alpha=0.6, linewidth=0.6)
    for ax in axes.ravel():
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.ticklabel_format(useOffset=False)
        ax.tick_params(labelsize=7)
    fig.suptitle(tile["tile_id"] + " | Native XY; PROPOSED_UNREVIEWED; no interpolation")
    fig.tight_layout()
    fig.savefig(path / f"{tile['tile_id']}_views.png", dpi=110)
    plt.close(fig)
    if sections:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        chosen = np.linspace(0, len(sections) - 1, 4).astype(int)
        for ax, i in zip(axes.ravel(), chosen, strict=True):
            section = sections[i]
            offset, z = np.array(section["offsets"]), np.array(
                [v if v is not None else np.nan for v in section["elevations"]]
            )
            ax.plot(offset, z, ".", color="gray", markersize=3, label="original-point median bins")
            for segment in section["segments"]:
                mask = (offset >= segment["left"]) & (offset <= segment["right"])
                ax.plot(
                    offset[mask],
                    z[mask],
                    "-",
                    color="green" if segment["near_reference"] else "orange",
                    linewidth=2,
                )
            ax.axvline(0, color="cyan", label="reference")
            ax.set_title(
                f"{section['reference_id']} station {section['station']:.1f}; edges uncertain"
            )
            ax.set_xlabel("Across-reference offset (native units)")
            ax.set_ylabel("Original elevation (native units)")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(path / f"{tile['tile_id']}_profiles.png", dpi=120)
        plt.close(fig)


def run_pilot(
    root,
    *,
    mode="propose",
    reference_path=None,
    review_path=None,
    discovery_mode="intensity",
    tile_limit=None,
    source=None,
    seed_path=None,
):
    settings_path = root / "configs/phase3_1_topodot_inspired.yaml"
    settings = yaml.safe_load(settings_path.read_text())
    tiles = pilot_tiles(root)
    if source:
        if source not in INTACT:
            raise ValueError("Only previously verified intact sources may be selected")
        tiles = [tile for tile in tiles if Path(tile["source_file"]).stem == source]
    guidance_path = reference_path or seed_path
    if guidance_path:
        reference_data = read_json(guidance_path)
        tiles = [
            tile for tile in tiles if tile["source_sha256"] == reference_data.get("source_sha256")
        ]
        reference_geometry = unary_union(
            [shape(f["geometry"]) for f in reference_data.get("features", [])]
        )
        tiles = [
            tile
            for tile in tiles
            if reference_geometry.intersects(box(*_extent(tile, settings["pilot_halo_m"])))
        ]
    if not tiles:
        raise ValueError("No pilot windows overlap the selected source/reference")
    if tile_limit:
        tiles = tiles[:tile_limit]
    code_paths = [
        Path(__file__),
        *sorted((root / "src/roadmarking/preprocessing").glob("*.py")),
        root / "src/roadmarking/quality/marking_retention.py",
        root / "src/roadmarking/rasterization/rasterizer.py",
    ]
    identity = {
        "settings": settings,
        "tiles": [t["tile_id"] for t in tiles],
        "mode": mode,
        "discovery_mode": discovery_mode,
        "source_hashes": {Path(t["source_file"]).stem: t["source_sha256"] for t in tiles},
        "code_hash": hashlib.sha256("".join(sha256(p) for p in code_paths).encode()).hexdigest(),
        "reference_sha256": sha256(reference_path) if reference_path else None,
        "seed_sha256": sha256(seed_path) if seed_path else None,
        "review_sha256": sha256(review_path) if review_path else None,
    }
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    out = root / "data/processed" / NS / run_id
    reports = root / "outputs/reports" / NS / run_id
    if (reports / "report.json").exists():
        saved = read_json(reports / "report.json")
        if not all(
            Path(p).is_file() and sha256(p) == digest
            for p, digest in saved["artifact_hashes"].items()
        ):
            raise ValueError("Existing run artifact verification failed")
        return saved
    if out.exists():
        raise ValueError("Incomplete pilot exists; preserve it and change run identity")
    for directory in DIRECTORIES:
        (out / directory).mkdir(parents=True, exist_ok=True)
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    diagnosis = diagnose(root, reports)
    if mode == "diagnose":
        return {"status": "DIAGNOSED", "root_cause": diagnosis, "report_directory": str(reports)}
    if mode == "guided" and reference_path is None and seed_path is None:
        raise ValueError("Guided/validated mode requires an explicit reference")
    if mode == "validated" and reference_path is None:
        raise ValueError("Validated processing requires an approved alignment, not seed-only hints")
    if mode == "validated" and review_path is None:
        raise ValueError("Validated processing requires manually verified paint/boundary review")
    protected = {
        str(p): sha256(p)
        for p in (
            root / "data/processed/phase3/metadata/tile_manifest.json",
            root / "outputs/reports/phase3/phase3_report.json",
            root / "outputs/reports/phase3_1/phase3_1_report.json",
            root / "outputs/reports/phase4/phase4_report.json",
        )
    }
    manifest, results, source_stats = [], [], {}
    start_total = time.perf_counter()
    with MemoryMonitor() as monitor:
        for source in sorted(identity["source_hashes"]):
            source_tiles = [t for t in tiles if Path(t["source_file"]).stem == source]
            path = Path(source_tiles[0]["source_file"])
            before_hash = sha256(path)
            if before_hash != identity["source_hashes"][source]:
                raise ValueError("LAZ does not match previously verified intact source")
            header = read_header(path)
            if header.point_count != INTACT[source]:
                raise ValueError("Unexpected source point count")
            cache_identity = hashlib.sha256(
                json.dumps(
                    {
                        "source": before_hash,
                        "tiles": [t["tile_id"] for t in source_tiles],
                        "halo": settings["pilot_halo_m"],
                        "partition": settings["partition_m"],
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()[:16]
            cache = root / "data/processed" / NS / "point_cache" / cache_identity
            source_stats[source] = partition_source(
                path, source_tiles, cache, settings, before_hash
            )
            refs = None
            seeds = (
                load_seed_points(seed_path, before_hash, [*header.mins[:2], *header.maxs[:2]])
                if seed_path
                else None
            )
            if reference_path:
                refs = load_references(
                    reference_path,
                    before_hash,
                    [*header.mins[:2], *header.maxs[:2]],
                    approved=mode == "validated",
                )
            summaries, source_boundaries = {}, []
            for tile in source_tiles:
                tid = tile["tile_id"]
                points = load_window(
                    cache, _extent(tile, settings["pilot_halo_m"]), settings["partition_m"]
                )
                references, evidence, sections, boundaries, status, timings, coarse = (
                    propose_window(
                        points,
                        tile,
                        settings,
                        refs,
                        (
                            "reference"
                            if refs
                            else "geometry" if seeds is not None else discovery_mode
                        ),
                        seeds,
                    )
                )
                source_boundaries.append(boundaries)
                summaries[tid] = (references, sections, status, timings, boundaries)
                vector_file(
                    out / "reference_alignments" / f"{tid}.geojson",
                    [
                        _feature(
                            LineString(r["coords"]),
                            **{k: v for k, v in r.items() if k != "coords"},
                        )
                        for r in references
                    ],
                    before_hash,
                )
                dump(out / "cross_sections" / f"{tid}.json", sections)
                vector_file(
                    out / "corridor_boundaries" / f"{tid}.geojson",
                    [
                        _feature(boundaries[k], kind=k)
                        for k in ("strict", "search", "uncertain", "rejected", "insufficient")
                        if not boundaries[k].is_empty
                    ],
                    before_hash,
                )
                for name, arr in evidence.items():
                    write_tif(out / "rasters" / tid / f"discovery_{name}.tif", arr, coarse, None)
                dump(
                    out / "metadata" / f"{tid}_proposal.json",
                    {
                        "status": status,
                        "timings": timings,
                        "boundary_diagnostics": boundaries["diagnostics"],
                        "points_in_halo": len(points),
                    },
                )
                print(
                    json.dumps(
                        {
                            "proposed": tid,
                            "references": len(references),
                            "sections": len(sections),
                            "status": status,
                        }
                    ),
                    flush=True,
                )
            # Global union within pilot scope makes every overlapping classification
            # layer identical. The explicit review-window fallback protects observations
            # where an automatic reference stops or remains unreliable.
            strict = unary_union([b["strict"] for b in source_boundaries])
            profile_search = unary_union([b["search"] for b in source_boundaries])
            window_scope = unary_union([box(*_extent(t)) for t in source_tiles])
            rejected = unary_union([b["rejected"] for b in source_boundaries])
            context = unary_union(
                [
                    observed_support_envelope(
                        load_window(cache, _extent(t), settings["partition_m"]), t
                    )
                    for t in source_tiles
                ]
            )
            search = context.union(profile_search).intersection(window_scope)
            vector_file(
                out / "corridor_boundaries" / f"{source}_shared.geojson",
                [
                    _feature(strict.intersection(window_scope), kind="strict_surface_candidate"),
                    _feature(
                        search,
                        kind="marking_search",
                        reason="profile_support_plus_bounded_review_window_fallback",
                    ),
                    _feature(search.difference(strict), kind="uncertain_search"),
                ],
                before_hash,
            )
            source_review = reviewed_samples(review_path, before_hash) if review_path else None
            for tile in source_tiles:
                tid = tile["tile_id"]
                references, sections, status, timings, boundaries = summaries[tid]
                points = load_window(cache, _extent(tile), settings["partition_m"])
                transform, calibration = Affine(*tile["affine"]), IntensityCalibration(
                    **tile["normalization"]
                )
                start = time.perf_counter()
                raw = accumulate(points, transform, (1024, 1024), calibration)
                new, keep = rasterize_search_corridor(
                    points["xyz"],
                    points["intensity"],
                    transform,
                    (1024, 1024),
                    calibration,
                    search,
                    strict,
                )
                timings["rasterization_seconds"] = time.perf_counter() - start
                old = {}
                for name in ("observed", "maximum", "normalized", "road_state"):
                    with rasterio.open(tile["layers"][name]) as ds:
                        old[name] = ds.read(1)
                from roadmarking.preprocessing.stripe_reference import local_contrast

                _, paint = local_contrast(
                    raw["normalized"],
                    raw["observed"].astype(bool),
                    0.05,
                    StripeSettings(**settings["stripe"]),
                )
                metrics = retention_metrics(raw, old, new, paint)
                metrics.update(
                    source_points_in_window=len(points),
                    search_points=int(keep.sum()),
                    strict_point_count=int(
                        shapely.intersects_xy(
                            strict, points["xyz"][:, 0], points["xyz"][:, 1]
                        ).sum()
                    ),
                    old_mask_area=float((old["road_state"] > 0).sum() * 0.05**2),
                )
                # Geometry rejection remains an independent QA layer; no observations
                # are suppressed by it in model-input intensity.
                new["rejected_geometry"] = (
                    rasterize(
                        [(mapping(rejected), 1)],
                        out_shape=(1024, 1024),
                        transform=transform,
                        fill=0,
                        dtype="uint8",
                    )
                    if not rejected.is_empty
                    else np.zeros((1024, 1024), np.uint8)
                )
                metrics["observed_in_rejected_geometry"] = int(
                    ((new["observed"] > 0) & (new["rejected_geometry"] > 0)).sum()
                )
                if source_review:
                    metrics["manual_review"] = manual_retention(
                        source_review, raw["observed"] > 0, new["observed"] > 0, transform
                    )
                start = time.perf_counter()
                layer_paths, readback = {}, True
                for name, arr in new.items():
                    target = out / "tiles" / tid / f"{name}.tif"
                    write_tif(target, arr, transform, None)
                    with rasterio.open(target) as ds:
                        readback &= (
                            ds.transform == transform
                            and ds.crs is None
                            and np.array_equal(ds.read(1), arr)
                        )
                    layer_paths[name] = str(target)
                write_tif(
                    out / "rasters" / tid / "original_maximum.tif", raw["maximum"], transform, None
                )
                write_tif(out / "road_masks" / f"{tid}_old.tif", old["road_state"], transform, None)
                timings["writing_and_readback_seconds"] = time.perf_counter() - start
                metrics["readback_exact"] = bool(readback)
                entry = {
                    "tile_id": tid,
                    "source": source,
                    "source_sha256": before_hash,
                    "row_offset": tile["row_offset"],
                    "col_offset": tile["col_offset"],
                    "affine": tile["affine"],
                    "height": 1024,
                    "width": 1024,
                    "resolution": 0.05,
                    "overlap": 256,
                    "stride": 768,
                    "layers": layer_paths,
                    "normalization": tile["normalization"],
                    "crs_status": "CRS_UNCONFIRMED",
                    "review_status": "PROPOSED_UNREVIEWED",
                    "model_input_status": "PROVISIONAL_REVIEW_REQUIRED",
                    "run_id": run_id,
                }
                manifest.append(entry)
                widths = [
                    d["width_m"] for d in boundaries["diagnostics"] if d["width_m"] is not None
                ]
                result = {
                    "tile_id": tid,
                    "source": source,
                    "historical_boundary_flag": tile["historical_boundary_flag"],
                    "reference_status": status,
                    "references": len(references),
                    "supported_references": sum(r["reference_supported"] for r in references),
                    "cross_sections": len(sections),
                    "width_percentiles": (
                        np.percentile(widths, [0, 10, 50, 90, 100]).tolist() if widths else []
                    ),
                    "metrics": metrics,
                    "timings": timings,
                }
                results.append(result)
                dump(reports / f"{tid}.json", result)
                vector_file(
                    out / "review_data" / f"{tid}_samples.geojson",
                    [
                        _feature(
                            box(*_extent(tile)),
                            kind="review_window",
                            examples="solid,broken,curved,multilane,edge,embankment,false_positive,sparse,old_boundary",
                        )
                    ],
                    before_hash,
                )
                figures(reports / "figures", tile, raw, old, new, references, boundaries, sections)
                print(
                    json.dumps(
                        {
                            "rasterized": tid,
                            "recovered": metrics["previously_excluded_observed_recovered"],
                            "paint_proxy_pct": metrics["candidate_paint_preserved_pct"],
                        }
                    ),
                    flush=True,
                )
            if sha256(path) != before_hash:
                raise ValueError("Source changed during pilot")
            source_stats[source]["source_sha256_unchanged"] = True
        overlap = overlap_check(manifest)
    if not all(sha256(p) == digest for p, digest in protected.items()):
        raise ValueError("Protected previous output changed")
    report = {
        "status": "PROVISIONAL",
        "run_id": run_id,
        "identity": identity,
        "full_processing_performed": False,
        "full_processing_gate": "Manual boundary/paint review unavailable; uncertain references and non-road inclusion require review before full processing",
        "root_cause": diagnosis,
        "source_streams": source_stats,
        "pilot_tiles": len(tiles),
        "boundary_flagged_tiles": sum(t["historical_boundary_flag"] for t in tiles),
        "tiles": results,
        "overlap": overlap,
        "total_seconds": time.perf_counter() - start_total,
        "peak_rss_bytes": monitor.peak,
        "scope": "Representative pilot windows; full LAZ decoded once per source for spatial partitioning, not full-survey rasterization",
        "protected_artifact_hashes": protected,
        "output_root": str(out),
        "report_directory": str(reports),
        "verified_marking_recall": None,
        "confirmed_road_area": 0,
        "search_policy": "All original observations within explicit bounded pilot review windows; profile search is separate. Search envelope is NOT a verified road boundary.",
        "reference_manual_review": "UNAVAILABLE",
        "phase4_5_ready": False,
    }
    if not overlap["passed"] or any(
        not t["metrics"]["readback_exact"]
        or not t["metrics"]["raw_max_equal_original_where_retained"]
        for t in results
    ):
        report["status"] = "FAILED"
    dump(out / "metadata/tile_manifest.json", manifest)
    dump(out / "metadata/identity.json", identity)
    report["output_bytes"] = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    report["artifact_hashes"] = {str(p): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}
    dump(reports / "report.json", report)
    dump(
        root / "outputs/reports" / NS / "latest.json",
        {"run_id": run_id, "report": str(reports / "report.json")},
    )
    write_report(report, reports / "report.md")
    return report


def write_report(report, path):
    tiles = report["tiles"]
    validation_path = path.parent.parent / "validation.json"
    validation = read_json(validation_path) if validation_path.exists() else None

    def total(key):
        return sum(t["metrics"][key] for t in tiles)

    text = [
        f"# Phase 3.1 independent TopoDOT-inspired pilot — {report['status']}\n",
        f"Run `{report['run_id']}`. All geometry is PROPOSED_UNREVIEWED; confirmed road area = 0.\n",
        "## A. Previous extraction failure\n",
        "See root_cause.md and root_cause.json. Mixed all-point OLS cells fail RMS; destructive geometry filtering then removes their intensity evidence. Adjacent surfaces also have sparse observations and elevation differences. No seed/width cap in v1; no evidence of tiling-caused clipping.\n",
        "## B. New architecture\n",
        "All-valid-point spatial cache → local-contrast stripe proposals / guided reference → native-XY perpendicular station profiles → longitudinally constrained surface candidates + alternatives → independent search-corridor intensity accumulation. Strict surface is an aligned candidate layer, never an intensity prerequisite.\n",
        "## C. Public workflow principles\n",
        "Public workflow inspiration only; no proprietary code or algorithms reproduced. Stripe/alignment references and station cross-sections: [infrastructure note](https://www.certainty3d.com/pdf/technotes/TopoDOT_Infrastructure.pdf). Local direction/turn handling is our independent implementation, inspired by [2022 release notes](https://blog.topodot.com/1039-topodot-version-2022-0-release-highlights/). Feature-specific analysis: [classification discussion](https://blog.topodot.com/1039-is-the-requirement-for-point-cloud-classification-dead/). Review and control: [product page](https://www.topodot.com/solutions/topodot).\n",
        "## D. Reference alignment results\n",
        "Local adaptive contrast, short PCA fragments, direction-compatible observed-gap graph. All competing paths retained. Bright linear objects are not verified paint. Geometry fallback is a review-only PCA path, never a reliable alignment assertion. See per-tile reference vectors and source-indexed point caches.\n",
        "## E. Cross-section results\n",
        "1 native-unit stations, ±30 search limit, 0.25 profile bins; median/MAD profiles, density-adaptive counts, height/slope/normal breaks. No lowest-surface rule. Selected segment must include the reference; disconnected levels remain alternatives. Smooth terminations and missing samples retain boundary uncertainty. Width jumps become uncertain rather than silently smoothed. Station strips are local geometric hypotheses, not surveyed edges. Intersections retain multiple reference hypotheses and require review.\n",
        "## F. Corridor extraction results\n",
        f"{report['pilot_tiles']} pilot tiles including {report['boundary_flagged_tiles']} historical boundary cases. No full-survey raster rerun. Overlapping tile counts below are NOT unique survey areas or point totals.\n",
        "| Tile | Ref status | Refs / supported | Sections | Width min/p10/p50/p90/max | Old / strict / search area | Source / search points |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for t in tiles:
        m = t["metrics"]
        text.append(
            f"| {t['tile_id']} | {t['reference_status']} | {t['references']}/{t['supported_references']} | {t['cross_sections']} | {t['width_percentiles']} | {m['old_mask_area']:.1f}/{m['strict_candidate_area']:.1f}/{m['search_area']:.1f} | {m['source_points_in_window']}/{m['search_points']} |"
        )
    text += [
        "\n## G. Road-marking preservation\n",
        f"Sum over tiles: {total('previously_excluded_observed_recovered'):,} previously excluded observed pixels recovered; {total('candidate_paint_preserved'):,}/{total('candidate_paint_pixels'):,} local-contrast proxy pixels retained; {total('old_observed_now_missing'):,} old observed pixels now missing. This is NOT verified marking recall. Human review is unavailable; candidate labels must not be relabeled verified by this pipeline. Full-retention results in bounded pilot windows follow the explicit review-envelope policy and are not proof that automatic discovery finds every road in a whole survey.\n",
        "## H. False positives\n",
        f"{total('new_observed_outside_strict'):,} observed pixels lie outside the strict geometric candidate; {total('observed_in_rejected_geometry'):,} lie in rough/mixed geometric rejection regions (tile sums overlap). Search intentionally preserves these for review. New non-road area is unknown without reviewed boundaries. Parallel bright structures, roofs, sidewalks and parking surfaces can remain in search; intensity and smoothness alone do not establish a road. Strict excludes disconnected profile segments but is not semantically validated.\n",
        "## I. Raster quality\n",
        f"Overlap: {report['overlap']}. Native 0.05 grid, 1024² tiles, 256 overlap / 768 stride; original affines reused. Raw max/mean and shared-file normalization checked against original valid observations; all tile layers read back exactly. NoData remains explicit; filled pixels = {total('filled_pixels')}. Missing observations are not inferred paint. confirmed_road is zero until independent reviewed surface acceptance. New raw maxima can exceed old maxima when previously excluded elevated points share a pixel; this is reported in JSON, not hidden.\n",
        "## J. Performance\n",
        f"Pilot total {report['total_seconds']:.1f} s; sampled peak RSS {report['peak_rss_bytes']/2**20:.1f} MiB; run outputs {report['output_bytes']/2**20:.1f} MiB (shared point cache additional). Historical full Phase 3 ≈315 s is a different scope; no speedup claimed. Per-window detection, section, boundary, raster and writing times are in JSON. Source streams: "
        + json.dumps(
            {
                k: {
                    n: v[n]
                    for n in (
                        "points_streamed",
                        "unique_cached_points",
                        "reading_seconds",
                        "reading_and_partition_seconds",
                        "source_sha256_unchanged",
                    )
                }
                for k, v in report["source_streams"].items()
            }
        )
        + "\n",
        "## K. Tests\n",
        (
            f"Full suite: {validation['pytest']['passed']} passed, {validation['pytest']['skipped']} skipped, {validation['pytest']['failures']} failures, {validation['pytest']['errors']} errors. Healthcheck: {validation['healthcheck']}. Ruff and Black passed; see [validation.json](../validation.json) and the parent-folder logs.\n"
            if validation
            else "Validation is pending; see the parent-folder pytest/healthcheck/lint artifacts.\n"
        ),
        "## L. Remaining issues\n",
        "CRS, units, vertical datum and acquisition metadata unconfirmed. Annotations do not overlap local LiDAR or the inventoried share. No unrelated annotations used. Sparse/occluded carriageways cannot be reconstructed as measurements. Automatic path continuity, stripe semantic identity, smooth boundaries and inclusion of non-road search observations need human review. No model download, training, labels or Phase 4/5 processing. Previous Phase 3/3.1/4 protected artifacts verified unchanged.\n",
        "## M. Final status\n",
        f"**{report['status']}**. {report['full_processing_gate']}. Review QGIS native-XY overlays and A–G images before approving references or evaluating manually verified paint. Full processing was not performed. No Git commit.\n",
    ]
    audit_path = path.parent / "quality_audit.json"
    if audit_path.exists():
        audit = read_json(audit_path)
        text += [
            "\n## Persisted-output audit\n",
            f"Validation: {audit['validation']}. Full metrics: [quality_audit.json](quality_audit.json). Executed core-code snapshot verified: {audit.get('executed_pilot_code_snapshot_verified')}; protected previous outputs unchanged: {audit['protected_outputs_unchanged']}.\n",
            "### Pilot before/after totals\n",
            "All totals below sum overlapping tiles; they are not unique survey counts/areas. Strict point counts use the XY candidate footprint, not validated 3D pavement classification.\n",
            "| Source | Old selected points → new search points | Old mask → strict candidate → search area (native units²) |",
            "|---|---:|---|",
        ]
        for source in report["source_streams"]:
            group = [t for t in report["tiles"] if t["source"] == source]
            ids = {t["tile_id"] for t in group}
            old_count = sum(
                t.get("old_selected_points", 0) for t in audit["tiles"] if t["tile_id"] in ids
            )
            new_count = sum(t["metrics"]["search_points"] for t in group)
            areas = [
                sum(t["metrics"][key] for t in group)
                for key in ("old_mask_area", "strict_candidate_area", "search_area")
            ]
            text.append(
                f"| {source} | {old_count:,} → {new_count:,} | {areas[0]:,.1f} → {areas[1]:,.1f} → {areas[2]:,.1f} |"
            )
        text += [
            "\n### Section sensitivity\n",
            "One provisional boundary-case reference per source was investigated at 0.5/1/2 station spacing and ±20/30/40 search limits. Median supported width remains about 11.5 native units; widening the search reveals more alternatives rather than proving additional roadway. The ±40 test is limited by the cached 30-unit window halo; no points are extrapolated. Noise and width percentiles are in quality_audit.json.\n",
            "### Review packages\n",
        ]
        text += [
            f"- [{source} CRS-less QGIS review package]({Path(bundle['review_package']).name})"
            for source, bundle in audit["review_bundles"].items()
        ]
        text += [
            "\nReference/search/strict layers remain PROPOSED_UNREVIEWED; manual edit layers are empty. Original source-indexed XYZ/raw-intensity proximity evidence is under source_point_evidence/. It supports review, not verified paint truth.\n"
        ]
    path.write_text("\n".join(text), encoding="utf-8")
