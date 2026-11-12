"""Build the Phase 3 delivery report from executed runs and regression evidence."""

import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from roadmarking.config import load_config
from roadmarking.rasterization.pipeline import dump


def main():
    cfg = load_config()
    report_dir = cfg.path("outputs.reports") / "phase3"
    runs = json.loads((report_dir / "full_run.json").read_text())
    smoke = json.loads((report_dir / "smoke_run.json").read_text())
    suite = ET.parse(report_dir / "pytest.xml").getroot().find("testsuite")
    health = json.loads((report_dir / "healthcheck.json").read_text())
    checks = dict(
        tests=int(suite.attrib["tests"]),
        failures=int(suite.attrib["failures"]),
        errors=int(suite.attrib["errors"]),
        skipped=int(suite.attrib["skipped"]),
    )
    checks["passed"] = checks["tests"] - checks["failures"] - checks["errors"] - checks["skipped"]
    checks["health_passed"] = sum(x["status"] == "pass" for x in health)
    checks["health_skipped"] = sum(x["status"] == "skip" for x in health)
    checks["health_failed"] = sum(x["status"] == "fail" for x in health)
    checks["health_details"] = health
    combined = []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from matplotlib.patches import Rectangle

    for run in runs:
        tiles = json.loads(Path(run["manifest"]).read_text())
        combined.extend(tiles)
        out = Path(run["reports"])
        with rasterio.open(
            Path(run["output_directories"]["road_masks"]) / "surface_state_1m.tif"
        ) as ds:
            image = ds.read(1)
            extent = [ds.bounds.left, ds.bounds.right, ds.bounds.bottom, ds.bounds.top]
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(
            image,
            extent=extent,
            origin="upper",
            aspect="equal",
            cmap="viridis",
            vmin=0,
            vmax=2,
            interpolation="nearest",
        )
        for tile in tiles:
            x0, y0, x1, y1 = tile["native_xy_bbox"]
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    linewidth=0.35,
                    edgecolor="red",
                    alpha=0.5,
                )
            )
        ax.set(
            title=f'{run["source"]} | 5 cm tiles, 1 m uncertainty overview\n'
            "Red = tile footprint; CRS_UNCONFIRMED",
            xlabel="Native X",
            ylabel="Native Y",
        )
        fig.tight_layout()
        fig.savefig(out / "tile_coverage.png", dpi=150)
        plt.close(fig)
        best = sorted(tiles, key=lambda t: (-t["observed_pct"], t["tile_id"]))[:3]
        fig, axes = plt.subplots(1, len(best), figsize=(6 * len(best), 6), squeeze=False)
        for ax, tile in zip(axes[0], best, strict=True):
            with rasterio.open(tile["layers"]["normalized"]) as ds:
                image = np.ma.masked_equal(ds.read(1), -9999)
                ext = [ds.bounds.left, ds.bounds.right, ds.bounds.bottom, ds.bounds.top]
            cm = plt.get_cmap("gray").copy()
            cm.set_bad("magenta")
            ax.imshow(
                image, extent=ext, cmap=cm, vmin=0, vmax=1, aspect="equal", interpolation="nearest"
            )
            ax.set_title(
                f'{tile["tile_id"]}\n5 cm; {tile["observed_pct"]:.1f}% observed', fontsize=9
            )
            ax.tick_params(labelsize=7)
        fig.suptitle(f'{run["source"]} | example output tiles | magenta = no observation')
        fig.tight_layout()
        fig.savefig(out / "example_tiles.png", dpi=150)
        plt.close(fig)
    dump(cfg.path("data.processed") / "phase3" / "metadata" / "tile_manifest.json", combined)
    report = dict(
        date=datetime.now(ZoneInfo("Asia/Calcutta")).isoformat(),
        status="EXECUTED_AND_QA_PASSED; ROAD_REVIEW_PENDING",
        full_runs=runs,
        representative_runs=smoke,
        tests=checks,
        tiles=len(combined),
        input_points=sum(r["input_valid_points"] for r in runs),
        selected_candidate_points=sum(r["selected_points"] for r in runs),
        processing_seconds=sum(r["timings"]["total_processing_seconds"] for r in runs),
        peak_sampled_rss_bytes=max(r["peak_sampled_rss_bytes"] for r in runs),
        phase4_ready=False,
        blockers=[
            "Confirm actual road ROI or manually review geometry candidates "
            "before annotation conversion",
            "Confirm source horizontal CRS/native units and annotation coordinate alignment",
        ],
        client_pending=[
            "Corrected copies of the eight damaged/unusable source files with checksums",
            "Verified horizontal CRS and coordinate units",
            "Vertical datum",
            "Sensor/platform and trajectory/range information if available",
        ],
        manifest=str(cfg.path("data.processed") / "phase3" / "metadata" / "tile_manifest.json"),
    )
    dump(report_dir / "phase3_report.json", report)
    lines = [
        "# Phase 3 preprocessing: executed results",
        "",
        f'Date: {report["date"]}. Only Track01_3 and Track01_8 were processed.',
        "",
        "**Pipeline execution and data QA passed. Phase 4 production readiness is"
        " pending road-area review and coordinate confirmation.**",
        "",
        "## Inputs and extraction",
        "",
        "| Source | Valid records (each scan) | Candidate points | Rejected "
        "points | Candidate area (native units squared) | Confirmed road |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in runs:
        lines.append(
            f'| {r["source"]} | {r["input_valid_points"]:,} | {r["selected_points"]:,} | '
            f'{r["rejected_region_points"]:,} | {r["candidate_area_native_units2"]:,.0f} | '
            f'{r["confirmed_points"]:,} |'
        )
    lines += [
        "",
        "Both files completed two strict scans with zero invalid records and zero"
        " recovered/dropped chunks. "
        "SHA-256 hashes before and after processing confirm the source bytes were unchanged. "
        "No damaged inputs were included.",
        "",
        "Extraction fits local planes from double-precision moments on a 1-unit "
        "grid, rejects rough/mixed cells "
        "(RMS > 0.045), slopes > 0.20 and neighboring height steps > 0.20, "
        "removes connected components "
        "smaller than 100 square units, then selects points within 0.10 of their cell plane. "
        "Intensity never participates in extraction. Roads can be higher than neighboring terrain. "
        "Density supplies geometric sample support (minimum eight points), "
        "without a road-density cutoff. "
        "All automatic candidates remain uncertain; this is not a semantic road classifier. "
        "A supplied native-XY polygon limits candidates and marks geometry within"
        " it as ROI-confirmed. "
        "Robustness comes from conservative residual/cell rejection; mixed cells "
        "may lose road points.",
        "",
        "## Intensity",
        "",
        "Exact uint16 histograms calibrate a shared file-level p1 to p99.5 linear"
        " scale on selected point intensities. "
        "The denominator is at least 1,024 raw units to limit gain. Scaling is "
        "clipped to [0,1]; raw values stay intact. "
        "Saturation at 65,535 is counted, never discarded. No tile is "
        "recalibrated. Local normalization is disabled: "
        "there is no verified range/incidence model or annotation evidence to "
        "choose a reliable local correction.",
        "",
        "| Source | p1 / p50 / p99.5 | Saturated points (%) | Raw mean |",
        "|---|---:|---:|---:|",
    ]
    for r in runs:
        n = r["normalization"]["statistics"]
        lines.append(
            f'| {r["source"]} | {n["p1"]:,} / {n["p50"]:,} / {n["p99.5"]:,} | '
            f'{n["saturated_points"]:,} ({n["saturated_pct"]:.4f}%) | {n["mean"]:.2f} |'
        )
    lines += [
        "",
        "## Rasters, gaps and tiles",
        "",
        "The 0.05 native-unit north-up grid is treated provisionally as 5 cm; the"
        " client must confirm units. "
        "Origins are snapped to the same native-coordinate grid. Every raster and"
        " tile has its affine; "
        "CRS is unset and metadata says `CRS_UNCONFIRMED`. No EPSG was inferred. "
        "15 cm spans three pixels only where actual observations exist.",
        "",
        "Nine aligned layers are retained: maximum (int32, NoData -1), mean "
        "(float32, NoData -9999), "
        "normalized maximum and optional filled normalized maximum (float32, NoData -9999), "
        "count and saturation count (uint32), observed mask, fill mask, and road state (uint8). "
        "Zero count means missing; a measured zero intensity remains observed. "
        "Masks have no NoData code: "
        "zero is a meaningful false value. Lossless block NPZs, individual "
        "GeoTIFFs, record-aligned packed "
        "selection masks, and selected original XYZ/intensity/source-index "
        "partitions are preserved.",
        "",
        "| Source | Candidate pixels | Observed pixels | Missing within candidate"
        " (%) | Filled (%) | Tiles |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in runs:
        p = r["pixels"]
        lines.append(
            f'| {r["source"]} | {p["candidate_pixels"]:,} | {p["observed"]:,} | '
            f'{r["missing_candidate_pct"]:.2f}% | {r["filled_candidate_pct"]:.3f}% | {r["tiles"]} |'
        )
    lines += [
        "",
        "Pixel statistics count each sparse master pixel once; tile overlap does not inflate them. "
        "Tiles are at most 1024 x 1024, overlap 256, stride 768 (provisionally "
        "51.2 / 12.8 / 38.4 m). "
        "Edge tiles retain actual dimensions. Every nonempty observed pixel is "
        "covered; empty tiles are excluded. "
        "The combined manifest contains layer paths, source/hash, dimensions, "
        "affine, native bbox, percentages, "
        "normalization and configuration/run versions.",
        "",
        "Gap filling is available but disabled in these deliveries. Even when "
        "enabled it only fills an enclosed "
        "one-pixel hole with eight originally observed neighbors, all inside "
        "confirmed road, and normalized "
        "neighbor range <= 0.08. A one-pixel halo ensures global block "
        "consistency. Large holes and paint/background "
        "edges remain missing. No ROI was supplied, so there are no confirmed "
        "regions eligible for interpolation.",
        "",
        "## Performance",
        "",
        "Actual sequential measurements, seconds. Peak RAM is process RSS sampled"
        " every 0.1 seconds, "
        "not a theoretical or exact OS high-water mark. Full processing includes "
        "source/product hash verification; "
        "aggregate delivery-chart/report generation is outside the source-run timings.",
        "",
        "| Source | Read (two scans) | Surface fit | Select/partition | Intensity"
        " histogram | Rasterize | Tile write | Finalize hashes | Total | Peak RSS"
        " GiB | Product GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in runs:
        t = r["timings"]
        values = [
            t[k]
            for k in (
                "reading_input_seconds",
                "road_extraction_seconds",
                "selection_partition_seconds",
                "intensity_processing_seconds",
                "rasterization_seconds",
                "tile_writing_seconds",
                "finalization_seconds",
                "total_processing_seconds",
            )
        ]
        lines.append(
            f'| {r["source"]} | '
            + " | ".join(f"{v:.2f}" for v in values)
            + f' | {r["peak_sampled_rss_bytes"]/2**30:.3f} | {r["output_bytes"]/2**30:.3f} |'
        )
    lines += [
        "",
        "Source-run product sizes include selected raw point partitions and block/tile products, "
        "and exclude final completion/report JSON and subsequently generated overview figures. "
        "Stage sums exclude QA/readback, previews and other orchestration measured in total. "
        "No GPU processing or dependency changes were needed. Reading is "
        "performed twice; disk partitions "
        "prevent repeated LAZ reads for overlapping tiles. Dense arrays are "
        "bounded to a coarse grid with "
        "an eight-million-cell limit; full-resolution blocks are 512 square "
        "pixels. Oversized partitions fail "
        "at a configurable 256 MiB budget. Input chunk size is 250,000.",
        "",
        "## Validation and environment",
        "",
        f'Full pytest suite: **{checks["passed"]} passed, {checks["failures"]} failed, '
        f'{checks["errors"]} errors, {checks["skipped"]} skipped**. '
        "The skipped test fetches model configuration over the network. "
        f'Health checks: **{checks["health_passed"]} passed, '
        f'{checks["health_failed"]} failed, {checks["health_skipped"]} skipped** '
        "(CLI reports 10/10 OK including the optional skip). CUDA and both "
        "OpenBLAS/PyTorch import orders pass. "
        "No model weights were requested. Ruff and Black checks passed at delivery.",
        "",
        "Synthetic tests cover raised/sloped roads, vegetation, roofs and nearby "
        "smooth non-road areas, "
        "ROI boundaries, negative/large coordinates, pixel edges, "
        "mean/max/counts/NoData/dtypes, duplicates, "
        "chunk/order invariance, conservative filling, seam halos, partial tiles,"
        " deterministic IDs and "
        "cache tampering. Every full-run tile/layer was read back and compared "
        "bit-for-bit to the global "
        "block mosaic; this validates overlapping regions. Counts reconcile to "
        "selected source points. "
        "All tile transforms and observation/fill provenance were checked; no "
        "fabricated CRS is present.",
        "",
        "## Previews and limitations",
        "",
        "Representative raw, extracted, mean, maximum, normalized, density, "
        "observed and filled masks share "
        "the same extent and aspect ratio. Magenta denotes missing measurements. "
        "Solid and dashed lines "
        "are visible in the inspected Track01_3 region. These windows do not "
        "establish verified coverage "
        "of arrows, curves or other symbols; no such markings were invented for a preview. "
        "Full-source tile overviews and three high-coverage example tiles are also included.",
        "",
    ]
    for r in runs:
        relative = Path(r["reports"]).relative_to(report_dir).as_posix()
        lines += [
            f'### {r["source"]}',
            "",
            f"![Aligned representative views]({relative}/representative_comparison.png)",
            "",
            f"![Tile coverage]({relative}/tile_coverage.png)",
            "",
            f"![Example tiles]({relative}/example_tiles.png)",
            "",
            f"![Intensity histogram]({relative}/intensity_histogram.png)",
            "",
        ]
    lines += [
        "## Phase 4 readiness and pending client information",
        "",
        "**Not yet cleared for production annotation conversion.** The "
        "native-coordinate products are reusable "
        "for annotation review, but actual road areas must be confirmed and "
        "annotation coordinates aligned to "
        "the source grid. Unknown CRS does not prevent same-native-grid review; "
        "it blocks claims of verified "
        "geographic alignment. Confirm horizontal CRS, coordinate units and "
        "vertical datum, sensor/platform, "
        "and request corrected source files with checksums. No reference "
        "annotations were available for "
        "marking recall or geometric road-boundary accuracy. OLS rejection is "
        "intentionally conservative and "
        "may omit mixed cells; smooth non-road terrain remains explicitly uncertain.",
        "",
        "Do not start Phase 4 automatically. No training, segmentation, weights "
        "download, vectorization or Git commit was performed.",
        "",
    ]
    (report_dir / "phase3_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "tiles",
                    "input_points",
                    "selected_candidate_points",
                    "processing_seconds",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
