"""Phase 4 JSON + Markdown report (sections A-H of the Phase 4 request)."""

# ruff: noqa: E501  (long Markdown sentences are kept on one line for readability)

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def _pytest(report_dir: Path) -> dict | None:
    xml = report_dir / "pytest.xml"
    if not xml.is_file():
        return None
    suites = list(ET.parse(xml).getroot().iter("testsuite"))
    t = {
        k: sum(int(s.attrib.get(k, 0)) for s in suites)
        for k in ("tests", "failures", "errors", "skipped")
    }
    t["passed"] = t["tests"] - t["failures"] - t["errors"] - t["skipped"]
    return t


def _health(report_dir: Path) -> dict | None:
    path = report_dir / "healthcheck.json"
    if not path.is_file():
        return None
    return dict(Counter(r["status"] for r in json.loads(path.read_text(encoding="utf-8"))))


def _fmt(v, nd=2):
    return (
        "n/a"
        if v is None
        else f"{v:,.{nd}f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else str(v)
    )


def write_reports(report: dict, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    report = dict(report)
    report["validation"] = {"pytest": _pytest(report_dir), "healthcheck": _health(report_dir)}
    (report_dir / "phase4_report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8"
    )
    a = report["alignment"]
    cmp_ = a["comparison"]
    q = report["annotation_quality"]
    rr = report["road_review"]
    perf = report["performance"]
    lines = []
    w = lines.append
    w("# Phase 4 report: annotation alignment and dataset preparation\n")
    w(f"Run `{report['run_id']}` — {report['date']}\n")
    w(f"**Status: {report['status']}**\n")
    w(
        "**The training dataset is NOT ready for Mask2Former fine-tuning.** No production "
        "training labels were generated because none of the delivered annotations overlap the "
        "Phase 3 LiDAR rasters in any tested coordinate interpretation.\n"
    )

    w("## A. Input summary\n")
    w(
        "| File | Layer | Features | Geometry | Declared CRS | Z | Invalid | Empty | Missing IDs | Dup IDs | Unknown/missing class |"
    )
    w("|---|---|---:|---|---|---|---:|---:|---:|---:|---:|")
    for f in report["inputs"]["files"]:
        for lay in f.get("layers", []):
            unk = sum(lay["unknown_or_missing_class_codes"].values())
            w(
                f"| {f['name']} | {lay['layer']} | {lay['feature_count']:,} | {lay['declared_geometry_type']} | "
                f"{lay['declared_crs']} | {'yes' if lay['z_available'] else 'no'} | {lay['invalid_geometries']} | "
                f"{lay['empty_geometries']} | {lay['missing_identifiers']} | {lay['duplicate_identifiers']} | {unk} |"
            )
    w("")
    for f in report["inputs"]["files"]:
        for lay in f.get("layers", []):
            ext = ", ".join(f"{v:,.1f}" for v in lay["extent"])
            w(
                f"- **{f['name']}** — extent ({ext}); Z {lay['z_range'][0]:.1f}–{lay['z_range'][1]:.1f}; "
                f"multipart {lay['multipart_features']}; polygons with holes {lay['polygons_with_holes']}; "
                f"clockwise exterior rings {lay['clockwise_exterior_rings']}; class field `{lay['classification_field']}` "
                f"values: {json.dumps(lay['classification_values'], ensure_ascii=False)}"
            )
            for d in lay["invalid_details"]:
                w(f"  - invalid fid {d['fid']}: {d['reason']}")
    w("")
    for o in report["inputs"]["other_files"]:
        w(
            f"- `{Path(o['path']).name}`: **{o['status']}** — {o.get('error') or ''} "
            "(MicroStation DGN v8; the installed GDAL has no DGNv8 driver; not converted, no dependency added)."
        )
    w(
        "\nAll attributes are retained verbatim per feature (`attributes` JSON in the run "
        "metadata). No shapefiles were delivered. Original files were hashed before/after: "
        f"unchanged = {report['integrity']['client_annotation_files_unchanged']}.\n"
    )

    w("## B. Alignment results\n")
    w(
        f"Gate status: **{a['status']}**. Transformation applied: {a['transformation_applied']}. "
        f"CRS assigned: {a['crs_assigned']}.\n"
    )
    rb = cmp_["raster_bounds"]
    w(
        f"- Phase 3 rasters: {', '.join(cmp_['raster_crs_status'])}; resolution {cmp_['raster_resolution']}; "
        f"bounds X {rb[0]:,.1f}…{rb[2]:,.1f}, Y {rb[1]:,.1f}…{rb[3]:,.1f}; {cmp_['raster_axis']}."
    )
    w(
        f"- Features intersecting any of the 280 raster tiles: **{cmp_['feature_overlap']['features_intersecting_tiles']} "
        f"of {cmp_['feature_overlap']['features_total']:,}**.\n"
    )
    w(
        "| Annotation file | Declared CRS | X range | Y range | Z range | Gap to raster footprint | Any axis swap/sign flip overlaps? | Overlaps any of 10 LAZ headers? |"
    )
    w("|---|---|---|---|---|---:|---|---|")
    for name, f in cmp_["annotation_files"].items():
        b = f["bounds"]
        w(
            f"| {name} | {', '.join(f['declared_crs'])} | {b[0]:,.0f}…{b[2]:,.0f} | {b[1]:,.0f}…{b[3]:,.0f} | "
            f"{f['z_range'][0]:.0f}…{f['z_range'][1]:.0f} | {f['gap_to_raster_footprint_native']:,.0f} | "
            f"{'yes' if any(f['axis_hypotheses_bbox_intersect'].values()) else 'no'} | "
            f"{'yes' if any(f['bbox_intersects_any_lidar_header'].values()) else 'no'} |"
        )
    w("")
    lz = [v for v in cmp_["lidar_header_extents"].values() if v["valid_header_bounds"]]
    zmin = min(v["xyz_min"][2] for v in lz)
    zmax = max(v["xyz_max"][2] for v in lz)
    w(
        f"LiDAR header Z for all readable LAZ files: {zmin:.1f}…{zmax:.1f}; no LAZ header carries a CRS. "
        "The annotation layers declare EPSG:3763 (metres). The nearest annotation is ≈255 km "
        "(native units) from the nearest raster tile; the annotated corridor is ~43 km × 11 km "
        "while the 10 LAZ headers together span ~13 km × 32 km. Pure axis swaps and sign flips "
        "were tested as diagnostics only; none produces overlap. No shift, rotation, scale or "
        "reprojection was applied or estimated — establishing the relationship would require "
        "inventing a transformation.\n"
    )
    w(
        "Displacement measurement (perpendicular intensity-profile offsets of centrelines, "
        "implemented and validated on synthetic data) could not be run: there are **no shared "
        "features** to measure. Measured alignment error on client data: **not measurable**.\n"
    )
    w(
        "Overlay: `outputs/reports/phase4/alignment_extent_overview.png` (annotation centroids, "
        "LAZ header boxes and Phase 3 raster bboxes in stored coordinates). Per-section overlays "
        "of solid/broken lines, curves, arrows and polygons on intensity tiles are impossible "
        "because no annotation falls inside any tile.\n"
    )

    w("## C. Annotation quality\n")
    sc = q["status_counts"]
    w("| Status | Features |\n|---|---:|")
    for k, v in sc.items():
        w(f"| {k} | {v:,} |")
    w(
        "\nNo feature is VERIFIED: verification requires a passed alignment gate and a human "
        "review record (`--review-csv`), neither of which exists. INCOMPLETE = missing class "
        "code (`COD_RST` 'NULL', OBS 'Code not avilable'); UNUSABLE = invalid geometry.\n"
    )
    w("| QC reason | Features |\n|---|---:|")
    for k, v in sorted(q["reason_counts"].items(), key=lambda kv: -kv[1]):
        w(f"| {k} | {v:,} |")
    w("")
    w(
        "Visual-class distribution (from attributes, unverified): "
        + ", ".join(
            f"{k} {v:,}"
            for k, v in sorted(q["visual_class_distribution"].items(), key=lambda kv: -kv[1])
        )
        + ".\n"
    )
    for kind, dist in q["client_code_distribution"].items():
        w(
            f"Client codes ({kind}): "
            + ", ".join(f"{k}: {v:,}" for k, v in sorted(dist.items(), key=lambda kv: -kv[1]))
            + ".\n"
        )
    w(
        "Per file: "
        + "; ".join(
            f"{n}: {d['features']:,} features, {d['duplicates_flagged']} duplicate-flagged"
            for n, d in q["by_file"].items()
        )
        + ".\n"
    )
    lz_ = q["line_length_m"]
    w(
        f"Line lengths (m): median {lz_['median']:.2f}, p95 {lz_['p95']:.2f}, max {lz_['max']:,.1f} "
        "(single edge-line features up to ~40 km). Vertex Z minus `ELEVATION` attribute: "
        f"median {q['z_minus_elevation_attr']['median']:.3f} m, p5/p95 {q['z_minus_elevation_attr']['p5']:.2f}/"
        f"{q['z_minus_elevation_attr']['p95']:.2f} m, extremes {q['z_minus_elevation_attr']['min']:.1f}/"
        f"{q['z_minus_elevation_attr']['max']:.1f} m.\n"
    )

    w("## D. Dataset generation\n")
    d = report["dataset"]
    w(
        f"Generated: **{d['generated']}**. Images {d['images']}; instances {d['instances']}; "
        f"polygon-derived {d['polygon_instances']}; polyline-derived {d['polyline_instances']}. "
        "No masks by class, no ignored-region statistics: nothing was rasterized from client "
        "vectors. The converter (polygons with holes, flat-capped width buffers of centrelines, "
        "4× supersampled coverage, boundary/conflict/no-data/unknown ignore states, stable "
        "global IDs across overlapping tiles) passed its synthetic self-test: "
        f"{report['converter_selftest']['checks']} — see `outputs/reports/phase4/converter_selftest/` "
        "(figures are titled SYNTHETIC).\n"
    )

    w("## E. Data splitting\n")
    if report["splits"]:
        s = report["splits"]
        for k, v in s["splits"].items():
            w(f"- {k}: {v['tiles']} tiles, {v['unique_instances']} instances")
        w(f"- leakage: {s['leakage']}")
    else:
        w(
            "Not generated (no labelled tiles). The implemented method orders tiles along each "
            "corridor, assigns contiguous blocks to train/val/test, removes guard tiles whose "
            "footprint overlaps another split or that share a (non-long) instance, and reports "
            "insufficient val/test sets instead of forcing 70/15/15. With the current Phase 3 "
            "extent (two non-overlapping ~2 km corridors, 141 + 139 tiles) a meaningful "
            "independent test set would need annotations covering both tracks.\n"
        )

    w("## Road-surface candidate review (step 3)\n")
    w(f"Tiles reviewed: {rr['tiles_reviewed']}. Flag counts: {json.dumps(rr['flag_counts'])}.\n")
    w(
        "| Source | Candidate 1 m cells | Components | Cells outside largest | Flat rejected cells ≤15 m (regions ≥100 m²) | Bright rejected cells ≤5 m |"
    )
    w("|---|---:|---:|---:|---:|---:|")
    for name, s in rr["sources"].items():
        if s and s.get("comparable"):
            w(
                f"| {name} | {s['candidate_cells']:,} | {s['candidate_components']} | {s['cells_outside_largest_component']:,} | "
                f"{s['flat_rejected_cells_within_15m']:,} ({s['flat_rejected_cells_in_regions_ge_100m2']:,}) | {s['bright_rejected_cells_within_5m']:,} |"
            )
    w(
        "\nSee the narrative findings and recommendations in "
        "`docs/phase4_dataset_preparation.md` and figures in `outputs/reports/phase4/road_review/`.\n"
    )

    w("## F. Quality checks\n")
    v = report["validation"]
    if v["pytest"]:
        t = v["pytest"]
        w(
            f"- Pytest: {t['passed']} passed, {t['failures']} failed, {t['errors']} errors, {t['skipped']} skipped (`outputs/reports/phase4/pytest.xml`)."
        )
    if v["healthcheck"]:
        w(f"- Health check: {v['healthcheck']}.")
    w(f"- Converter self-test passed: {report['converter_selftest']['passed']}.")
    w(
        f"- Client annotation files unchanged: {report['integrity']['client_annotation_files_unchanged']}; "
        f"Phase 3 metadata unchanged: {report['integrity']['phase3_metadata_unchanged']}."
    )
    w("- Visual overlays of client annotations on intensity: not possible (no overlap).\n")

    w("## G. Performance\n")
    w("| Step | Seconds |\n|---|---:|")
    for k, s in perf["timings_seconds"].items():
        w(f"| {k} | {s:.2f} |")
    w(
        f"\nTotal {perf['total_seconds']:.1f} s; peak working set {perf['peak_working_set_bytes'] / 2**20:,.0f} MiB "
        f"(OS process peak incl. imports); run output {perf['output_bytes'] / 2**20:.1f} MiB. "
        "GPU not used; no LAZ reread. Main costs: reading 280 × 3 tile layers for road review "
        "and pairwise duplicate/overlap checks (STRtree-filtered).\n"
    )

    w("## H. Phase 5 readiness\n")
    w("**Not ready.** Blockers:\n")
    rc = q["reason_counts"]
    counts = {
        "dash": rc.get("dash_length_mismatch", 0),
        "dups": rc.get("duplicate_in_other_file", 0) + rc.get("duplicate_in_same_file", 0),
        "overlap": rc.get("polygon_overlap", 0),
        "z": q.get("z_mismatch_gt_1m", 0),
        "invalid": rc.get("invalid_geometry", 0),
        "nocode": rc.get("missing_class_code", 0),
    }
    for i, b in enumerate(BLOCKERS, 1):
        w(f"{i}. {b.format(**counts)}")
    w(
        "\nNo model training, weight download, architecture change, dependency change or Git commit occurred.\n"
    )
    (report_dir / "phase4_report.md").write_text("\n".join(lines), encoding="utf-8")


BLOCKERS = [
    "Coordinate-frame mismatch: annotations (declared EPSG:3763, Y ≈ +166…+177 km) and all 10 LiDAR "
    "files (no CRS; Y ≈ −118…−86 km) do not overlap. The client must either deliver annotations "
    "for the surveyed LiDAR area, or LiDAR covering the annotated corridor, or state the exact CRS "
    "of the LiDAR and any documented transformation between the two.",
    "LiDAR CRS, native units and vertical datum remain unconfirmed (annotation Z 75–371 vs LiDAR Z 13–86).",
    "DGN v8 deliveries cannot be read with the installed GDAL; ask for GPKG/SHP exports if they contain additional data.",
    "Phase 3 road candidates cover a narrow strip and exclude flat, adjacent road-like surface; "
    "lane markings lie on the candidate edge. A confirmed road ROI (or relaxed selection) is needed before labels.",
    "Client catalog confirmation for COD_RST visual groupings (M15–M21, M9a), 'Linha mista' (code 4 vs 5) and the "
    "missing Portuguese labels for codes 8, 10, 11, 12; {nocode} polygons have no class code.",
    "Attribute/geometry inconsistencies need review: {dash:,} dash lengths disagree with TRAÇO by >25 %; {dups} duplicate flags "
    "between/within files; {overlap} overlapping polygons; {z} features with vertex Z ≠ ELEVATION by >1 m; {invalid} invalid geometry.",
    "Human review records (VERIFIED) and reviewed-completeness regions are needed; without them no pixel can be background.",
]
