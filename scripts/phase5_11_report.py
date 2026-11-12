"""Phase 5.11 report: audit, verified regions, dataset bound, gallery and approval checklist."""

import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import rasterio
from PIL import Image
from rasterio.features import rasterize
from shapely.geometry import box

from phase5_11_common import ROOT, client_features, load, reserved_geography
from roadmarking.rasterization.pipeline import dump, sha256

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def gallery(spec, path, title="Phase 5.11 new training candidates"):
    tiles = spec["tiles"]
    store = spec["candidate_store"]
    paint = gpd.read_file(store, layer="candidate_paint_footprints")
    try:
        unsupported = gpd.read_file(store, layer="unsupported_marking_regions")
    except Exception:
        unsupported = gpd.GeoDataFrame(geometry=[])
    cols = 4
    rows = math.ceil(len(tiles) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5.9 * rows), squeeze=False)
    for ax, tile in zip(axes.flat, tiles, strict=False):
        gray = np.asarray(Image.open(tile["images"]["grayscale"]).convert("L"), float) / 255
        with rasterio.open(tile["layers"]["observed"]) as ds:
            transform = ds.transform
        base = np.repeat(gray[..., None], 3, -1)
        for frame, value, colour in (
            (
                paint[(paint.candidate_id == tile["candidate_id"]) & (paint.visual_id == 1)],
                1,
                (1, 0.2, 0.1),
            ),
            (
                paint[(paint.candidate_id == tile["candidate_id"]) & (paint.visual_id == 2)],
                1,
                (0.1, 0.5, 1),
            ),
            (
                (
                    unsupported[unsupported.candidate_id == tile["candidate_id"]]
                    if len(unsupported)
                    else unsupported
                ),
                1,
                (1, 0, 1),
            ),
        ):
            if not len(frame):
                continue
            mask = rasterize(
                [(g, value) for g in frame.geometry], gray.shape, transform=transform
            ).astype(bool)
            edge = mask & ~np.roll(mask, 1, 0) | mask & ~np.roll(mask, 1, 1)
            base[edge | (mask & (np.indices(mask.shape).sum(0) % 3 == 0))] = colour
        ax.imshow(base)
        ax.set_title(
            f"{tile['candidate_id']} [{tile['priority_tag']}]\n"
            f"{tile['observed_solid_ids']} solid / {tile['observed_dash_ids']} dash "
            f"observed; obs {tile['observed_fraction']:.0%}; {tile['acquisition']}",
            fontsize=8,
        )
        ax.axis("off")
    for ax in axes.flat[len(tiles) :]:
        ax.axis("off")
    fig.suptitle(
        f"{title} — client outlines on actual intensity "
        "(red solid, blue dash, magenta unsupported). PENDING approval.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.5)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    latest = load(ROOT / "outputs/reports/phase5_11/latest.json")
    out = Path(latest["report_dir"])
    audit = load(out / "training_coverage_audit.json")
    ranking = load(out / "candidate_ranking.json")
    spec = load(out / "new_region_candidates.json")
    scans = load(out / "source_integrity.json")
    qgis = (
        load(out / "qgis_review_package.json")
        if (out / "qgis_review_package.json").exists()
        else None
    )
    (out / "galleries").mkdir(exist_ok=True)
    gallery(spec, out / "galleries" / "new_training_candidates_overview.png")

    # ---------------- catalogue upper bound (NOT point verified)
    features, _, raw_audit = client_features(check_network=False)
    regions, extra, _ = reserved_geography()
    reserved_ids = set(extra)
    train_ids = set()
    for r in regions:
        (train_ids if r["role"] == "train" else reserved_ids).update(r["ids"])
    raw = [
        r
        for r in load(Path(raw_audit["report_directory"]) / "raw_integrity.json")
        if "backup" not in r["path"].lower()
    ]
    headers = [box(r["mins"][0], r["mins"][1], r["maxs"][0], r["maxs"][1]) for r in raw]
    eligible = Counter()
    solid_length = 0.0
    for f in features:
        if f["visual_id"] not in (1, 2) or f["global_id"] in reserved_ids | train_ids:
            continue
        c = f["geometry"].centroid
        if min(r["box"].distance(c) for r in regions) < 500:
            continue
        if not any(h.contains(c) for h in headers):
            continue
        eligible["solid" if f["visual_id"] == 1 else "dash"] += 1
        if f["visual_id"] == 1:
            solid_length += f["centerline"].length
    verification = spec["verification"]
    new = spec["new_unique_observed_identities"]
    existing = audit["summary"]
    totals = {
        "existing_unique_training_identities": existing["unique_training_identities"],
        "new_unique_observed_identities_in_package": new["total"],
        "total_if_all_packaged_regions_approved": existing["unique_training_identities"]
        + new["total"],
        "solid_total_if_approved": existing["unique_solid_identities"] + new["solid"],
        "dash_total_if_approved": existing["unique_dash_identities"] + new["dash"],
    }
    tiles = spec["tiles"]
    acquisitions = Counter(t["acquisition"] for t in tiles)
    tags = Counter(t["priority_tag"] for t in tiles)
    result = {
        "phase": "5.11",
        "decision": "REAL_DATA_REVIEW_REQUIRED",
        "run_id": latest["run_id"],
        "training_coverage_summary": existing,
        "weaknesses": audit["weaknesses"],
        "acquisition_priority": audit["acquisition_priority"],
        "discovery": {
            "candidate_windows_considered": ranking["candidate_windows_considered"],
            "windows_point_verified": verification["windows_point_verified"],
            "windows_meaningful": verification["windows_meaningful"],
            "windows_packaged": verification["windows_packaged"],
            "priority_tags": dict(tags),
            "acquisitions": dict(acquisitions),
            "failed_observation_check": verification["failed_observation_check"],
        },
        "source_integrity": {
            "files_decoded": len(scans),
            "all_full_point_integrity_passed": all(
                s["full_source_integrity_passed"] for s in scans
            ),
            "points_decoded": int(sum(s["actual_points"] for s in scans)),
            "decode_seconds": float(sum(s["seconds"] for s in scans)),
        },
        "dataset_size": {
            **totals,
            "catalogue_upper_bound_not_point_verified": {
                "solid_source_ids": eligible["solid"],
                "dash_source_ids": eligible["dash"],
                "solid_centerline_length_native": solid_length,
                "definition": "client IDs not in any existing split, >=500 from all existing "
                "footprints, centroid inside a non-backup LAZ header box. Header boxes "
                "over-claim coverage; actual observed yield is lower.",
            },
        },
        "qgis_package": qgis,
        "training_performed": False,
        "protected_final_test_used": False,
        "existing_approvals_modified": False,
    }
    pytest_xml = ROOT / "outputs/reports/phase5_11_pytest.xml"
    if pytest_xml.exists():
        suite = ET.parse(pytest_xml).getroot()
        suite = suite.find("testsuite") if suite.tag == "testsuites" else suite
        result["validation"] = {"pytest": dict(suite.attrib)}
        for name in ("phase5_11_ruff.log", "phase5_11_black.log", "phase5_11_healthcheck.json"):
            p = ROOT / "outputs/reports" / name
            if p.exists():
                result["validation"][name] = sha256(p)
    dump(out / "report.json", result)

    # ---------------- approval checklist
    lines = [
        "# Phase 5.11 — first-approval checklist for NEW training regions",
        "",
        f"QGIS project: `{qgis['project'] if qgis else 'n/a'}` (bookmarks: one per region).",
        "Layers: red = candidate solid paint, blue = candidate dash paint, magenta = "
        "unsupported symbols/hatching (always excluded), yellow = review footprint. Toggle "
        "the observed/NoData masks to see missing LiDAR returns (never treated as background).",
        "",
        "For each region, reply with **APPROVE**, **REJECT**, or **APPROVE WITH NOTES**. "
        "Approval confirms all four statements:",
        "",
        "1. Painted marking geometry is correctly positioned.",
        "2. Solid/dash classes are correct.",
        "3. Every visible target road marking inside the yellow footprint is represented.",
        "4. Observed non-target pixels inside the footprint may be treated as background.",
        "",
        "| Region | Priority | Acquisition / source | Observed solid / dash IDs | Observed % | "
        "Reply |",
        "|---|---|---|---|---:|---|",
    ]
    for t in tiles:
        lines.append(
            f"| {t['candidate_id']} | {t['priority_tag']} | {t['acquisition']} / "
            f"{Path(t['source']).name} | {t['observed_solid_ids']} / {t['observed_dash_ids']} | "
            f"{t['observed_fraction']:.0%} | |"
        )
    lines += [
        "",
        "A rejected region is simply left out; it is never repaired from model output. "
        "Approval stays limited to the documented yellow footprint.",
        "",
    ]
    (out / "review_checklist.md").write_text("\n".join(lines), encoding="utf-8")
    (Path(spec["output"]) / "review_checklist.md").write_text("\n".join(lines), encoding="utf-8")

    # ---------------- markdown report
    s = existing
    md = [
        "# Phase 5.11: targeted real LiDAR training-data expansion",
        "",
        "**Decision: REAL_DATA_REVIEW_REQUIRED.** The new training regions are packaged for "
        "first approval. No new region is used for supervision, export or training until "
        "it is explicitly approved. No training was run in this phase, the selected "
        "checkpoint is unchanged, and the protected final test was not read.",
        "",
        "## Stage 1: existing training coverage (unique physical identities)",
        "",
        f"- {s['unique_training_identities']} unique identities: "
        f"{s['unique_solid_identities']} solid, {s['unique_dash_identities']} dash. These "
        f"come from {s['crop_appearances']} overlapping crop appearances plus "
        f"{s['background_only_training_crops']} background-only crops.",
        "- Unique observed paint pixels: solid "
        f"{s['unique_observed_paint_pixels_by_class']['solid']}, "
        f"dash {s['unique_observed_paint_pixels_by_class']['dash']}.",
        f"- Solid shape: {s['solid_shape_counts']}. Edge lines: {s['solid_edge_lines']}. "
        f"Merge/diverge lines: {s['merge_or_interchange_lines']['merge_designations']}. "
        "Interchange-layer identities: "
        f"{s['merge_or_interchange_lines']['from_interchange_layer']}.",
        f"- Narrow/faded proxy: {s['narrow_or_faded_proxy']['count']} identities. "
        f"Definition: {s['narrow_or_faded_proxy']['definition']}.",
        f"- {s['training_regions']} training regions fall into {s['section_count']} geographic "
        f"sections (500-unit clustering). They come from {s['distinct_source_files']} source "
        f"files, all in one acquisition job: `{s['source_acquisitions']}`.",
        f"- Hard negatives: {s['verified_hard_negative_scope']}.",
        "",
        "Weaknesses consistent with the Phase 5.9/5.10 failures:",
        *[f"- {w}" for w in audit["weaknesses"]],
        "",
        "Data-acquisition priority:",
        *[f"- {p}" for p in audit["acquisition_priority"]],
        "",
        "## Stage 2–3: discovery and actual point verification",
        "",
        f"- {ranking['candidate_windows_considered']} candidate 51.2×51.2 windows were "
        "proposed from client content. They exclude any window within 500 units of an "
        "existing training, development, benchmark, protected-test or held-out footprint, "
        "and any window containing a source ID from a non-training split.",
        f"- {verification['windows_point_verified']} spatially independent windows (≥500 "
        "units apart) were chosen with category quotas. Each assigned LAZ file was fully "
        "decoded, and every window was required to have ≥10% observed pixels, ≥64 observed "
        "paint pixels and ≥3 observed IDs. Header boxes alone were not trusted: if the "
        "first file lacked observations, the next file was tried, with one source per "
        "window and never merged.",
        f"- {verification['windows_meaningful']} windows passed and "
        f"{verification['windows_packaged']} were packaged. Priority mix: {dict(tags)}. "
        f"Acquisitions: {dict(acquisitions)}.",
        f"- Source integrity: {result['source_integrity']}. Files with corrupt LAZ chunks "
        f"or point-count mismatch were rejected outright, never partially used: "
        f"{[Path(x['source']).name for x in scans if not x['full_source_integrity_passed']]}. "
        "Imagery uses the frozen shared normalization, the native 0.05 grid and gamma .7. "
        "Observed and NoData masks are kept, and no observations were invented.",
        "",
        "| Region | Priority | Source | Solid/Dash observed IDs | Observed % | Max turn °/50 | "
        "Merge lines | Shares IDs with training |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for t in tiles:
        md.append(
            f"| {t['candidate_id']} | {t['priority_tag']} | {Path(t['source']).name} | "
            f"{t['observed_solid_ids']}/{t['observed_dash_ids']} | {t['observed_fraction']:.0%} | "
            f"{t['content_stats']['max_turning_deg_per_50']:.1f} | "
            f"{t['content_stats']['merge_lines']} | {len(t['shares_ids_with_existing_training'])} |"
        )
    ub = result["dataset_size"]["catalogue_upper_bound_not_point_verified"]
    md += [
        "",
        "## Dataset size",
        "",
        "- If every packaged region is approved: "
        f"{totals['total_if_all_packaged_regions_approved']} unique identities "
        f"({totals['solid_total_if_approved']} solid, "
        f"{totals['dash_total_if_approved']} dash). That is "
        f"+{new['total']} new observed identities ({new['solid']} solid, {new['dash']} dash).",
        f"- Catalogue upper bound (not point-verified): {ub['solid_source_ids']} solid and "
        f"{ub['dash_source_ids']} dash source IDs remain eligible, with "
        f"{ub['solid_centerline_length_native'] / 1000:.1f} km of solid centerline. Solid IDs "
        "are long features, so one solid ID can yield many geographically distinct segments. "
        "The real limit is review effort and actual observed coverage, not the catalogue.",
        "",
        "## Stage 4: review package",
        "",
        f"- QGIS: `{qgis['project'] if qgis else 'not built'}`. Validation: `{qgis}`.",
        "- Overview gallery: `galleries/new_training_candidates_overview.png`. Per-region "
        "QGIS renders are in `review_previews/`.",
        "- Approval checklist: `review_checklist.md`, with the four required confirmations "
        "for each region.",
        "",
        "## Stages 5–10: not started (by design)",
        "",
        "Export, training and evaluation need the explicit first approval above. Next steps "
        "once regions are approved:",
        "- Export with the tested NoData-aware pipeline (`reviewed_masks`). Exclude "
        "unsupported symbols with a margin, ambiguous boundaries and NoData, and keep dash "
        "and solid identities.",
        "- Run one bounded, predeclared warm-start run from the step-50 checkpoint, using "
        "curve/solid/dash-weighted crop sampling.",
        "- Evaluate the raw model and post-processing separately on LEGACY dev and "
        "PHASE58_POSITIVE_DEV_REVIEW, keeping the protected final test sealed.",
        "",
        f"Validation: `{result.get('validation')}`",
        "",
    ]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(out / "report.md"),
                "dataset_size": result["dataset_size"],
                "discovery": {
                    k: v for k, v in result["discovery"].items() if k != "failed_observation_check"
                },
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
