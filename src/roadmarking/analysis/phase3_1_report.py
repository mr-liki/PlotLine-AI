"""Phase 3.1 report: trial and full before/after comparisons, manifest, ROI proposals."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from roadmarking.analysis import phase3_1_compare as cmp
from roadmarking.rasterization.pipeline import dump, sha256

IST = timezone(timedelta(hours=5, minutes=30))

# Representative windows (Phase 3 tile ids) chosen from Phase 4 review + diagnostics.
CASES = {
    "Job_20260325_0838_Track01_3": [
        ("straight_roadway", "Job_20260325_0838_Track01_3__r0014592_c0014592"),
        ("lane_dashes_at_boundary", "Job_20260325_0838_Track01_3__r0023808_c0026880"),
        ("adjacent_lanes_behind_barrier", "Job_20260325_0838_Track01_3__r0005376_c0004608"),
        ("embankment_and_fragmented_edge", "Job_20260325_0838_Track01_3__r0018432_c0019968"),
        ("smooth_non_road_blob", "Job_20260325_0838_Track01_3__r0017664_c0017664"),
    ],
    "Job_20260325_0838_Track01_8": [
        ("lane_dashes_at_boundary", "Job_20260325_0838_Track01_8__r0036096_c0021504"),
        ("road_curve", "Job_20260325_0838_Track01_8__r0029952_c0018432"),
        ("sparse_scanning", "Job_20260325_0838_Track01_8__r0026880_c0016896"),
    ],
}


def _pytest(path: Path):
    if not path.is_file():
        return None
    suites = list(ET.parse(path).getroot().iter("testsuite"))
    t = {
        k: sum(int(s.attrib.get(k, 0)) for s in suites)
        for k in ("tests", "failures", "errors", "skipped")
    }
    t["passed"] = t["tests"] - t["failures"] - t["errors"] - t["skipped"]
    return t


def _thresholds(run):
    stats = run["result"]["normalization"]["statistics"]
    return {"ge_p95": int(stats["p95"]), "ge_p99": int(stats["p99"])}


def compare_pair(before, after, fig_dir: Path, cases, label: str) -> dict:
    thr = _thresholds(before)  # identical calibration in both runs
    out = dict(
        cells=cmp.cell_comparison(before, after),
        pixels=cmp.pixel_comparison(before, after),
        grid=cmp.grid_identity(before, after),
        points=dict(
            selected_before=before["result"]["selected_points"],
            selected_after=after["result"]["selected_points"],
            bright_selected_before=cmp.bright_points(before["histogram"], thr),
            bright_selected_after=cmp.bright_points(after["histogram"], thr),
            bright_thresholds_raw=thr,
        ),
        windows={},
    )
    tiles = {t["tile_id"]: t for t in before["manifest"]}
    tiles.update({t["tile_id"]: t for t in after["manifest"]})
    for name, tid in cases:
        if tid not in tiles:
            continue
        out["windows"][name] = cmp.window_figure(
            fig_dir / f"{label}_{name}.png", name.replace("_", " "), before, after, tiles[tid]
        )
    return out


def edge_tile_markings(before, after, edge_ids) -> dict:
    """Phase 4 'paint at candidate edge' tiles, re-measured on the v2 rasters."""
    from roadmarking.dataset.road_review import tile_statistics

    mb = {t["tile_id"]: t for t in before["manifest"]}
    ma = {t["tile_id"]: t for t in after["manifest"]}
    rows = []
    for tid in edge_ids:
        if tid in mb and tid in ma:
            sb, sa = tile_statistics(mb[tid]), tile_statistics(ma[tid])
            rows.append(
                dict(
                    tile_id=tid,
                    edge_band_bright_pct_before=sb["bright_in_edge_band_pct"],
                    edge_band_bright_pct_after=sa["bright_in_edge_band_pct"],
                    flagged_before="paint_like_pixels_at_candidate_edge" in sb["flags"],
                    flagged_after="paint_like_pixels_at_candidate_edge" in sa["flags"],
                    observed_pct_before=sb["observed_pct"],
                    observed_pct_after=sa["observed_pct"],
                )
            )
    return dict(
        tiles=rows,
        still_flagged=sum(r["flagged_after"] for r in rows),
        compared=len(rows),
    )


def combined_manifest(cfg, full: dict, phase3: dict) -> list[dict]:
    p3 = {
        t["tile_id"]: t
        for t in json.loads(
            (cfg.path("data.processed") / "phase3/metadata/tile_manifest.json").read_text()
        )
    }
    out = []
    for run in full.values():
        for t in run["manifest"]:
            t = dict(t)
            old = p3.get(t["tile_id"])
            t["phase3_comparison"] = dict(
                phase3_tile_present=old is not None,
                observed_pct_phase3=old["observed_pct"] if old else None,
                observed_pct_delta=(t["observed_pct"] - old["observed_pct"]) if old else None,
                road_surface_coverage_pct_phase3=old["road_surface_coverage_pct"] if old else None,
            )
            t["candidate_confidence_pct"] = _confidence_pct(t)
            t["qa_status"] = (
                "READBACK_AND_OVERLAP_VERIFIED" if run["result"]["qc"]["passed"] else "QA_FAILED"
            )
            roi = run["result"].get("roi")
            t["roi_review_status"] = roi["review_status"] if roi else "NO_APPROVED_ROI"
            t["road_review_status"] = (
                "AUTOMATIC_CANDIDATE_REQUIRES_REVIEW" if not roi else "ROI_CLIPPED"
            )
            t["phase4_status"] = "NEEDS_ROAD_REVIEW" if not roi else "ROI_REVIEWED_CANDIDATE"
            t["processing_configuration"] = dict(
                road_method=run["result"]["road_method"],
                refinement=run["result"]["refinement"]["criteria"],
                normalization="Phase 3 calibration reused (fixed for comparability)",
            )
            out.append(t)
    return sorted(out, key=lambda t: t["tile_id"])


def _confidence_pct(tile) -> dict:
    import rasterio

    with rasterio.open(tile["layers"]["road_confidence"]) as src:
        conf = src.read(1)
    counts = np.bincount(conf.ravel(), minlength=6)
    from roadmarking.preprocessing.road_surface import CONFIDENCE_NAMES

    return {CONFIDENCE_NAMES[i]: 100 * float(c) / conf.size for i, c in enumerate(counts)}


def build_report(cfg, root: Path, runs: dict, phase3_runs: dict) -> dict:
    reports = cfg.path("outputs.reports") / "phase3_1"
    figs = reports / "figures"
    processed = cfg.path("data.processed") / "phase3_1"
    report: dict = dict(
        date=datetime.now(IST).isoformat(), stage="Phase 3.1 road-corridor correction"
    )
    if (reports / "diagnostics.json").is_file():
        report["diagnostics"] = json.loads(
            (reports / "diagnostics.json").read_text(encoding="utf-8")
        )
    phase4 = json.loads(
        (cfg.path("outputs.reports") / "phase4/phase4_report.json").read_text(encoding="utf-8")
    )
    edge_ids = []
    review = Path(phase4["output_root"]) / "metadata/road_review.json"
    if review.is_file():
        tiles = json.loads(review.read_text())["tiles"]
        edge_ids = sorted(
            t for t, s in tiles.items() if "paint_like_pixels_at_candidate_edge" in s["flags"]
        )

    if "trial" in runs:
        tb = cmp.load_run(Path(runs["trial"]["v1"]["processing"]))
        ta = cmp.load_run(Path(runs["trial"]["v2"]["processing"]))
        report["trial"] = dict(
            bounds=runs["trial"]["bounds"],
            comparison=compare_pair(tb, ta, figs, CASES["Job_20260325_0838_Track01_3"], "trial"),
            timings_v1=tb["result"]["timings"],
            timings_v2=ta["result"]["timings"],
            peak_rss_v1=tb["result"]["peak_sampled_rss_bytes"],
            peak_rss_v2=ta["result"]["peak_sampled_rss_bytes"],
            refinement=ta["result"]["refinement"],
        )
    full = {}
    report["full"] = {}
    for source, proc in runs.get("full", {}).items():
        before = cmp.load_run(Path(phase3_runs[source]["reports"]) / "processing.json")
        after = cmp.load_run(Path(proc))
        full[source] = after
        comp = compare_pair(before, after, figs, CASES.get(source, []), source.split("_")[-1])
        comp["edge_tiles"] = edge_tile_markings(
            before, after, [t for t in edge_ids if t.startswith(source)]
        )
        comp["removed_cells"] = cmp.removed_cell_analysis(before, after)
        ref_lines = reports / "reference_plotline" / f"{source}.json"
        if ref_lines.is_file():
            data = json.loads(ref_lines.read_text(encoding="utf-8"))
            comp["plotline_paint_support"] = dict(
                plotline_version=data.get("plotline_version"),
                polylines=len(data["lines"]),
                support=cmp.paint_line_support(data["lines"], before, after),
            )
        cmp.overview_figure(figs / f"{source}_overview_confidence.png", source, before, after)
        roi = cmp.proposed_roi(
            processed / "candidate_rois" / f"{source}_proposed_road_corridor.gpkg", source, after
        )
        b, a = before["result"], after["result"]
        report["full"][source] = dict(
            comparison=comp,
            refinement=a["refinement"],
            proposed_roi=roi,
            performance=dict(
                phase3_timings=b["timings"],
                phase3_1_timings=a["timings"],
                phase3_peak_rss=b["peak_sampled_rss_bytes"],
                phase3_1_peak_rss=a["peak_sampled_rss_bytes"],
                phase3_output_bytes=b.get("output_bytes"),
                phase3_1_output_bytes=a.get("output_bytes"),
                input_bytes=(cfg.path("data.raw_pointclouds") / f"{source}.laz").stat().st_size,
            ),
            qc=a["qc"],
            input_valid_points=a["input_valid_points"],
            run_id=a["run_id"],
            source_sha256_unchanged=sha256(cfg.path("data.raw_pointclouds") / f"{source}.laz")
            == before["manifest"][0]["source_sha256"],
        )
    if full:
        manifest = combined_manifest(cfg, full, phase3_runs)
        dump(processed / "metadata/tile_manifest.json", manifest)
        from roadmarking.training.preflight import inspect_manifest

        check = inspect_manifest(processed / "metadata/tile_manifest.json")
        report["manifest"] = dict(
            path=str(processed / "metadata/tile_manifest.json"),
            tiles=len(manifest),
            phase4_preflight_compatible=bool(check.get("metadata_consistent")),
            phase4_preflight_errors=check["errors"],
            review_statuses=dict(Counter(t["road_review_status"] for t in manifest)),
            replaces_phase3_manifest=False,
        )
    share = reports / "raw_share_header_inventory.json"
    if share.is_file():
        recs = json.loads(share.read_text(encoding="utf-8"))
        ok = [r for r in recs if r["status"] == "HEADER_OK" and r["points"] > 0]
        ann = (-38506.7, 166202.1, 5068.0, 177217.3)  # Phase 4 annotation bbox (EPSG:3763)
        report["raw_share"] = dict(
            location=r"\\192.168.10.6\laz_1_2",
            entries=len(recs),
            readable_with_points=len(ok),
            zero_point_headers=sum(r["status"] == "HEADER_OK" and r["points"] == 0 for r in recs),
            header_errors=sum(r["status"].startswith("HEADER_ERROR") for r in recs),
            empty_files=sum(r["status"] == "EMPTY" for r in recs),
            temporary_files=sum(".tmp" in r["name"] for r in recs),
            header_points=sum(r["points"] for r in ok),
            crs_values=sorted({str(r["crs"]) for r in ok}),
            extent_xyz=[
                min(r["mins"][0] for r in ok),
                min(r["mins"][1] for r in ok),
                min(r["mins"][2] for r in ok),
                max(r["maxs"][0] for r in ok),
                max(r["maxs"][1] for r in ok),
                max(r["maxs"][2] for r in ok),
            ],
            files_overlapping_annotation_bbox=[
                r["name"]
                for r in ok
                if not (
                    r["maxs"][0] < ann[0]
                    or r["mins"][0] > ann[2]
                    or r["maxs"][1] < ann[1]
                    or r["mins"][1] > ann[3]
                )
            ],
            job_groups=dict(Counter(r["name"].split("_Track")[0] for r in recs)),
            method="header-only read; no points decoded; integrity not verified",
        )
    report["validation"] = dict(
        pytest=_pytest(reports / "pytest.xml"),
        healthcheck=(
            dict(
                Counter(r["status"] for r in json.loads((reports / "healthcheck.json").read_text()))
            )
            if (reports / "healthcheck.json").is_file()
            else None
        ),
    )
    report["phase3_hashes"] = {
        str(p): sha256(p)
        for p in (
            cfg.path("data.processed") / "phase3/metadata/tile_manifest.json",
            cfg.path("outputs.reports") / "phase3/phase3_report.json",
            cfg.path("outputs.reports") / "phase4/phase4_report.json",
        )
    }
    report["status"] = "CANDIDATE_OUTPUTS_GENERATED_REQUIRE_ROAD_REVIEW" if full else "TRIAL_ONLY"
    report["phase4_reuse_ready"] = False
    dump(reports / "phase3_1_report.json", report)
    from roadmarking.analysis.phase3_1_markdown import write_markdown

    write_markdown(report, reports / "phase3_1_report.md")
    return report
