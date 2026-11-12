"""Phase 3.1 Markdown report, sections A-J."""

# ruff: noqa: E501  (long Markdown sentences are kept on one line)

from __future__ import annotations

from pathlib import Path


def _n(v, nd=0):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    return f"{v:,}"


def _mib(v):
    return "n/a" if v is None else f"{v / 2**20:,.0f} MiB"


def write_markdown(report: dict, path: Path) -> None:
    lines: list[str] = []
    w = lines.append
    full = report.get("full", {})
    diag = report.get("diagnostics", {})
    w("# Phase 3.1 report: road-corridor extraction correction\n")
    w(f"{report['date']} — status **{report['status']}**\n")
    w(
        "**The refined candidates are automatic geometric evidence, not verified road boundaries. "
        "No ROI was approved, no training labels were generated, and the dataset is not yet usable for "
        "Mask2Former training because the annotations and LiDAR still do not overlap (Phase 4 blocker).**\n"
    )

    w("## A. Root cause analysis\n")
    w(
        "Phase 3 (v1) fits one OLS plane to *all* points of each 1 m cell and accepts the cell only if the "
        "plane RMS ≤ 0.045, slope ≤ 0.20, ≥ 8 points, no >0.20 centre-height step to an accepted neighbour, "
        "and the connected component is ≥ 100 m². Intensity is not used.\n"
    )
    for source, d in diag.items():
        if source == "seconds":
            continue
        b = d["band"]
        reasons = ", ".join(f"{k}: {_n(v)}" for k, v in b["by_first_failing_criterion"].items())
        w(
            f"- **{source}** — occupied rejected cells within {b['band_cells']} m of the candidate: "
            f"{_n(b['rejected_occupied_cells_in_band'])}; first failing criterion: {reasons}. "
            f"Plane RMS quartiles there {', '.join(f'{x:.3f}' for x in b['rms_percentiles_in_band'])} vs "
            f"{', '.join(f'{x:.3f}' for x in b['rms_percentiles_candidates'])} inside candidates; points per cell "
            f"{', '.join(_n(float(x)) for x in b['points_per_cell_in_band'])} vs "
            f"{', '.join(_n(float(x)) for x in b['points_per_cell_candidates'])}."
        )
    w("")
    w("Point cross-sections (`figures/diagnosis_*.png`) show what these numbers mean physically:\n")
    w(
        "1. **Mixed edge cells.** The ~12 m candidate is the full visible carriageway between the kerb/embankment and a ~1.3 m concrete median barrier. The last 0.5–1.5 m of asphalt (with the edge line) shares 1 m cells with barrier or kerb points, so the whole-cell RMS fails and the paint next to the barrier is dropped. This is the main reason for the 'dashes at the boundary' finding of Phase 4 (RMS is the dominant failing criterion)."
    )
    w(
        "2. **Occlusion, not exclusion, beyond the barrier.** The opposite carriageway appears only as sparse, disconnected patches in the barrier's scan shadow. v1 accepted some as separate small components; most are rejected for few points or small area. There is no continuous observed surface to recover there."
    )
    w("3. **Embankments and verges** fail the slope/RMS tests correctly.")
    w(
        "4. **Kerb joins.** The centre-height step limit (0.20) can join 10–15 cm kerbs/footpaths to the road when the step falls on a cell boundary."
    )
    w(
        "5. **Missing pixels** (~1/3 of candidate pixels) are the 5 cm sampling pattern of a dense-centre / sparse-edge swath, not extraction failures; they are kept as NoData.\n"
    )

    w("## B. Algorithm changes (road_method `v2`)\n")
    crit = (
        next(iter(full.values()))["refinement"]["criteria"]
        if full
        else report.get("trial", {}).get("refinement", {}).get("criteria", {})
    )
    w(
        "Phase 3 parameters are unchanged; v2 adds a second geometric stage (one extra streaming pass):\n"
    )
    w(
        f"1. **Continuity breaks** — 4-neighbour candidate pairs whose planes, evaluated at the shared cell edge, differ by > {crit.get('continuity_step_m')} or whose slopes differ by > {crit.get('max_slope_change')} are split (kerb/footpath joins)."
    )
    w(
        f"2. **Corridor core** — remaining components ≥ {crit.get('core_min_area_m2')} m² with length/mean-width ≥ {crit.get('corridor_min_elongation')} and mean width ≤ {crit.get('corridor_max_mean_width_m')} (an upper bound, not an assumed road width). Smaller or broad components become UNCERTAIN."
    )
    w(
        f"3. **Robust edge test** — cells within {crit.get('edge_buffer_cells')} cells of the core are re-tested point-by-point against the plane of the nearest core cell (extrapolated, so grade and banking carry over). Inliers: |Δz| ≤ {crit.get('inlier_band_m')}. A cell passes with ≥ {crit.get('edge_min_inliers')} inliers (≥ {crit.get('enclosed_min_inliers')} inside topological holes / when ≥ {crit.get('enclosed_min_core_neighbors')} of 8 neighbours are core), inlier-plane RMS ≤ {crit.get('edge_max_rms_m')}, slope change ≤ {crit.get('max_slope_change')}, offset ≤ {crit.get('max_offset_m')}. Only inlier points are selected, so barrier/kerb/vehicle points stay excluded even in an accepted cell."
    )
    w(
        f"4. **Coplanar promotion** — a smaller component is promoted only if it touches the corridor through verified cells and its median plane offset from the corridor plane ≤ {crit.get('max_offset_m')}."
    )
    w(
        f"5. **Parallel/unconnected smooth surfaces** within {crit.get('parallel_search_cells')} cells that pass the strict v1 cell tests are UNCERTAIN (possible opposite carriageway, footpath, roof, field) and are *not* selected (`include_uncertain: {crit.get('include_uncertain')}`)."
    )
    w(
        "6. **Evidence categories per 1 m cell** (`road_confidence` raster, codes 0–5): no observation, rejected, insufficient observations, uncertain (requires review), high-confidence geometric candidate, inside approved ROI; plus a per-cell evidence bit mask. These are categories of algorithm evidence, not calibrated probabilities; no numeric score is produced."
    )
    w(
        "7. **ROI support** (`roadmarking.preprocessing.road_roi`): GeoPackage/Shapefile/GeoJSON/native JSON, multiple polygons, holes, overlaps (reported), invalid geometry refused (never repaired), ROI must overlap the source extent; a CRS-declaring ROI is refused for a CRS-less LiDAR unless explicitly asserted to be native; `PROPOSED_UNREVIEWED` features are refused. Points/cells excluded by the ROI are reported."
    )
    w(
        "8. **Intensity** is never used for selection. For comparability v2 reuses each file's Phase 3 calibration (no recalibration, no contrast change).\n"
    )

    w("## C. Road coverage comparison\n")
    if "trial" in report:
        t = report["trial"]["comparison"]
        w(
            f"Representative trial (Track01_3 problem region, bounds {', '.join(_n(float(x), 2) for x in report['trial']['bounds'])}; identical points and bounds for v1 and v2):\n"
        )
        _coverage_table(w, {"trial region": t})
    if full:
        w(
            "Full intact files (v1 = existing Phase 3 outputs, v2 = Phase 3.1; identical inputs and grids):\n"
        )
        _coverage_table(w, {s: d["comparison"] for s, d in full.items()})
        for s, d in full.items():
            c = d["comparison"]["cells"]
            w(
                f"- **{s}**: recovered cells by evidence {_nonzero(c.get('recovered_by_evidence', {}))}; removed (now UNCERTAIN) by evidence {_nonzero(c.get('removed_by_evidence', {}))}. Share of non-inlier (above-band) points inside recovered cells: {_n(c.get('recovered_cell_outlier_share'), 2)} — i.e. recovered cells are mixed cells whose barrier/kerb points remain excluded."
            )
        w("")
    w(
        "These are coverage observations. Without verified road boundaries they are **not** recall or precision measurements.\n"
    )

    w("## D. Marking preservation\n")
    for label, d in [("trial region", report.get("trial", {}).get("comparison"))] + [
        (s, v["comparison"]) for s, v in full.items()
    ]:
        if not d:
            continue
        p = d["points"]
        b0, b1 = p["bright_selected_before"], p["bright_selected_after"]
        w(
            f"- **{label}**: selected points with raw intensity ≥ file p95 ({_n(p['bright_thresholds_raw']['ge_p95'])}): {_n(b0['ge_p95'])} → {_n(b1['ge_p95'])}; ≥ p99 ({_n(p['bright_thresholds_raw']['ge_p99'])}): {_n(b0['ge_p99'])} → {_n(b1['ge_p99'])}."
        )
        if "edge_tiles" in d:
            e = d["edge_tiles"]
            w(
                f"  Phase 4 'paint at candidate edge' tiles re-measured: {e['compared']} compared, {e['still_flagged']} still flagged after v2."
            )
        pl = d.get("plotline_paint_support")
        if pl:
            for kind, s in pl["support"].items():
                if s.get("lines"):
                    w(
                        f"  Independent PlotLine {pl['plotline_version']} traces ({kind}, {s['lines']} lines, {s['vertices_0_25m']:,} vertices at 0.25 m): inside v1 corridor {s['inside_v1_pct']:.1f} % → inside v2 corridor {s['inside_v2_pct']:.1f} % (v2-only {s['v2_only_pct']:.1f} %, v1-only {s['v1_only_pct']:.1f} %, v2 UNCERTAIN {s['inside_v2_uncertain_pct']:.1f} %)."
                    )
        for name, win in d.get("windows", {}).items():
            w(
                f"  - {name} (`{win['tile_id']}`): observed pixels {_n(win['observed_before'])} → {_n(win['observed_after'])}, bright (normalized ≥ 0.5) pixels {_n(win['bright_pixels_before'])} → {_n(win['bright_pixels_after'])}."
            )
    w(
        "\nBright points are a proxy for paint, not verified markings. Visual evidence: `outputs/reports/phase3_1/figures/` (before/after per case). No marking was synthesised; all added pixels are original measurements.\n"
    )

    w(
        "PlotLine groups are length-based proxies (≥20 m edge/solid, 2.5–12 m dash), not verified marking classes. Track01_3 reuses an earlier trace: its source hash was checked on reuse, but the original tracing-time hash is unavailable. Track01_8 was traced with source hashes checked before and after.\n"
    )
    w("## E. False-positive assessment\n")
    w(
        "- Recovered cells are confined to ≤ 2 m from the corridor core and contain only points within ±6 cm of the extrapolated corridor plane; barrier, kerb-top and vehicle points in those cells stay excluded (see outlier shares above)."
    )
    w(
        "- Possible residual false positives: barrier/kerb *foot* points within 6 cm of the road plane; a flat verge or hard shoulder continuous with the carriageway is indistinguishable from road by geometry alone and is included if coplanar."
    )
    w(
        "- v1 candidates that were disconnected or broad (sparse opposite-carriageway patches, the smooth blob in Track01_3, broad fields) are now UNCERTAIN and excluded from the automatic mask — this *reduces* automatic coverage there and may remove some genuine road; they are delivered as review polygons instead."
    )
    for s, d in full.items():
        r = d["comparison"].get("removed_cells") or {}
        if r.get("removed_cells"):
            dist = r["distance_to_core_cells_percentiles_10_50_90"]
            off = r["abs_plane_offset_percentiles_10_50_90"]
            near = r["removed_within_4_cells_abs_offset_percentiles_10_50_90"] or [0, 0, 0]
            w(
                f"- **{s}** — v1 cells no longer auto-selected: {_n(r['removed_cells'])}. Distance to corridor core (10/50/90 %): {dist[0]:.1f} / {dist[1]:.1f} / {dist[2]:.1f} cells; |plane offset| from the corridor surface: {off[0]:.2f} / {off[1]:.2f} / {off[2]:.2f}. Most are therefore other surface levels (opposite carriageway behind the barrier, verge, slopes), not coplanar carriageway. {_n(r['removed_within_4_cells'])} lie within 4 cells of the core with offsets {near[0]:.3f} / {near[1]:.3f} / {near[2]:.3f} — a slight shoulder/verge drop-off beyond the 4 cm coplanarity limit; some of these may be genuine paved shoulder."
            )
    w(
        "- No field/roof/footpath surface is automatically included in the synthetic tests. Visual inspection of the real-data case figures (straight, dashes at boundary, barrier, embankment, blob, curve, sparse) shows recovered pixels along the carriageway edges only, but this is not a verification against a reviewed ROI.\n"
    )

    w("## F. Raster and tile validation\n")
    for s, d in full.items():
        g = d["comparison"]["grid"]
        q = d["qc"]
        w(
            f"- **{s}**: tiles {g['tiles_before']} → {g['tiles_after']} ({len(g['new_tiles'])} new on the same global stride grid, {len(g['dropped_tiles'])} no longer containing selected points); common tiles {g['common_tiles']} with identical affines/dimensions: {g['identical_grid']}. Readback QC: {q['passed']} ({_n(q['tile_layers_read_back'])} tile layers read back exactly against master blocks, overlap consistency {q['exact_overlap_consistency']}, all observations covered {q['all_observations_covered']}). Source SHA-256 unchanged: {d['source_sha256_unchanged']}."
        )
    if "manifest" in report:
        m = report["manifest"]
        w(
            f"- Combined manifest `{m['path']}`: {m['tiles']} tiles; Phase 4 preflight compatible: {m['phase4_preflight_compatible']} {m['phase4_preflight_errors'] or ''}; review status {m['review_statuses']}; Phase 3 production manifest replaced: {m['replaces_phase3_manifest']}."
        )
    w(
        "- The 5 cm `road_confidence` tile layer exists only on raster blocks containing selected points; the complete per-cell evidence map is `surface_confidence_1m.tif` plus `refinement_model.npz` in each run's road_masks folder."
    )
    w(
        "- Grid origin, 0.05 resolution, 1024 px tiles, 256 px overlap (stride 768) and pixel-centre convention are inherited unchanged from Phase 3; CRS remains unassigned (`CRS_UNCONFIRMED`); missing pixels remain NoData; gap fill disabled.\n"
    )

    w("## G. Runtime and memory\n")
    w(
        "| Source | Phase 3 total s | Phase 3.1 total s | Read s (3.1) | Refinement s | Raster s (3 → 3.1) | Tiles s (3 → 3.1) | Peak RSS 3 → 3.1 | Output 3 → 3.1 |"
    )
    w("|---|---:|---:|---:|---:|---:|---:|---|---|")
    for s, d in full.items():
        p = d["performance"]
        a, b = p["phase3_timings"], p["phase3_1_timings"]
        w(
            f"| {s} | {a['total_processing_seconds']:.0f} | {b['total_processing_seconds']:.0f} | {b['reading_input_seconds']:.0f} | {b.get('road_refinement_seconds', 0):.1f} | {a['rasterization_seconds']:.0f} → {b['rasterization_seconds']:.0f} | {a['tile_writing_seconds']:.0f} → {b['tile_writing_seconds']:.0f} | {_mib(p['phase3_peak_rss'])} → {_mib(p['phase3_1_peak_rss'])} | {_mib(p['phase3_output_bytes'])} → {_mib(p['phase3_1_output_bytes'])} |"
        )
    w(
        "\nThe extra cost is one additional streaming read of the LAZ (refinement pass); the refinement arithmetic itself takes seconds. Peak RSS is sampled process RSS. No speed-up is claimed. Input LAZ sizes: "
        + ", ".join(f"{s}: {_mib(d['performance']['input_bytes'])}" for s, d in full.items())
        + ".\n"
    )

    w("## H. Test results\n")
    v = report.get("validation", {})
    if v.get("pytest"):
        t = v["pytest"]
        w(
            f"- Full suite: **{t['passed']} passed, {t['failures']} failed, {t['errors']} errors, {t['skipped']} skipped** (`outputs/reports/phase3_1/pytest.xml`)."
        )
    if v.get("healthcheck"):
        w(f"- Health check: {v['healthcheck']}.")
    w(
        "- New tests (`tests/test_phase3_1.py`): flat, sloped (6/12 %), embankment with broad field, adjacent lane behind a barrier, raised footpath, flat roof, scan gap, curved and banked road, marking at the boundary (and intensity independence), sparse coverage, disconnected smooth surface, ROI holes/overlaps/invalid/proposed/frame checks, ROI clipping, determinism and input immutability, and an end-to-end v1→v2 pipeline run (namespace separation, grid identity, NoData, Phase 4 manifest compatibility, byte-identical repeat)."
    )
    w("- Ruff and Black were run on all changed files.\n")

    w("## I. Remaining limitations\n")
    w(
        "- Automatic candidates are not verified road: no approved ROI exists. Proposed corridor and uncertain-candidate polygons (`data/processed/phase3_1/candidate_rois/*.gpkg`, `review_status = PROPOSED_UNREVIEWED`) need manual review."
    )
    w(
        "- The opposite carriageway is largely unobserved (barrier occlusion); it cannot be recovered without additional scans and is not manufactured."
    )
    w(
        "- 1 m cell decisions remain stair-stepped at boundaries; point-level inlier selection limits this for recovered cells only."
    )
    w("- Coplanar hard shoulders/verges cannot be separated from carriageway by geometry alone.")
    w(
        "- Horizontal CRS, native units (5 cm provisional) and vertical datum are unconfirmed; no EPSG was assigned."
    )
    w(
        "- Client data issues from Phase 4 remain: annotations do not overlap the LiDAR; DGN v8 unreadable; taxonomy and attribute inconsistencies; damaged LAZ files were not processed.\n"
    )

    rs = report.get("raw_share")
    if rs:
        e = rs["extent_xyz"]
        w("### Raw data share\n")
        w(
            f"`{rs['location']}` (header-only inventory, `raw_share_header_inventory.json`): {rs['entries']} entries in job groups {rs['job_groups']}; {rs['readable_with_points']} readable files with points ({rs['header_points']:,} header points); {rs['zero_point_headers']} zero-point headers, {rs['header_errors']} header errors, {rs['empty_files']} empty files, {rs['temporary_files']} temporary files. CRS in headers: {rs['crs_values']}. Combined extent X {e[0]:,.0f}…{e[3]:,.0f}, Y {e[1]:,.0f}…{e[4]:,.0f}, Z {e[2]:.0f}…{e[5]:.0f}. Files overlapping the annotation bounding box: {len(rs['files_overlapping_annotation_bbox'])}. The local `data/raw/pointclouds` files are same-size copies of share files. Only the two verified intact tracks were processed in Phase 3.1; the remaining files need the Phase 2 strict integrity scan first.\n"
        )
    w("## J. Phase 4 reuse readiness\n")
    w(
        "The Phase 3.1 rasters are on the identical global grid, carry the same layers plus `road_confidence`, pass exact readback/overlap QC, and the combined manifest is accepted by the Phase 4 preflight with `phase4_status = NEEDS_ROAD_REVIEW`. They are **technically suitable for annotation alignment and mask generation once matching, verified annotations exist and the corridor (or an ROI) has been reviewed**. They are **not** ready now: there are no annotations covering this LiDAR, and the road candidates are unreviewed. Phase 4/5 were not started.\n"
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _nonzero(d: dict) -> str:
    items = [f"{k} {v:,}" for k, v in d.items() if v]
    return ", ".join(items) or "none"


def _coverage_table(w, data: dict) -> None:
    w(
        "| Scope | HIGH area v1 → v2 (native units²) | Selected points v1 → v2 | Recovered / removed cells | Observed 5 cm pixels v1 → v2 (added / removed) | Missing in candidate % | Components | Holes | Boundary/area |"
    )
    w("|---|---|---|---|---|---|---|---|---|")
    for name, d in data.items():
        c, p, pts = d["cells"], d["pixels"], d["points"]
        w(
            f"| {name} | {c['before']['area']:,.0f} → {c['after']['area']:,.0f} | {pts['selected_before']:,} → {pts['selected_after']:,} | {c['recovered_cells']:,} / {c['removed_cells']:,} | {p['observed_pixels_before']:,} → {p['observed_pixels_after']:,} (+{p['observed_pixels_added']:,} / −{p['observed_pixels_removed']:,}) | {p['missing_in_candidate_pct_before']:.1f} → {p['missing_in_candidate_pct_after']:.1f} | {c['before']['components']} → {c['after']['components']} | {c['before']['interior_holes']} → {c['after']['interior_holes']} | {c['before']['boundary_per_area']:.3f} → {c['after']['boundary_per_area']:.3f} |"
        )
    w("")
