"""Phase 5.12: geometry/raster integrity verification, contact sheet, checklist and report."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from shapely.geometry import box

from phase5_11_common import ROOT, client_features, load
from phase5_11_report import gallery
from roadmarking.rasterization.pipeline import dump, sha256


def verify(spec):
    checks = {}
    store = spec["candidate_store"]
    checks["candidate_store_hash_unchanged"] = sha256(store) == spec["candidate_store_sha256"]
    paint = gpd.read_file(store, layer="candidate_paint_footprints")
    features = {f["global_id"]: f for f in client_features(check_network=False)[0]}
    tiles = {t["candidate_id"]: t for t in spec["tiles"]}
    id_ok = geom_ok = inside_ok = True
    worst = 0.0
    for _, r in paint.iterrows():
        id_ok &= hashlib.sha256(r.source_identity.encode()).hexdigest() == r.global_id
        f = features[r.global_id]
        region = box(*tiles[r.candidate_id]["bounds"])
        expected = f["geometry"].intersection(region)
        diff = r.geometry.symmetric_difference(expected).area / max(1e-9, expected.area)
        worst = max(worst, diff)
        geom_ok &= diff < 1e-6 and r.width == f["width"] and int(r.visual_id) == f["visual_id"]
        inside_ok &= r.geometry.difference(region.buffer(1e-6)).area < 1e-9
    checks.update(
        stable_ids_recomputed_from_source_identity=bool(id_ok),
        painted_outlines_equal_recorded_width_buffer=bool(geom_ok),
        worst_relative_outline_difference=float(worst),
        outlines_inside_review_footprints=bool(inside_ok),
        one_row_per_source_feature_per_region=not paint.duplicated(
            ["candidate_id", "global_id"]
        ).any(),
        individual_dashes_not_merged=bool(
            (paint[paint.visual_id == 2].groupby(["candidate_id", "global_id"]).size() == 1).all()
        ),
    )
    raster_ok = True
    for t in spec["tiles"]:
        with rasterio.open(t["layers"]["observed"]) as ds:
            observed = ds.read(1).astype(bool)
            raster_ok &= np.allclose(tuple(ds.transform)[:6], t["affine"]) and ds.shape == (
                1024,
                1024,
            )
        with rasterio.open(t["layers"]["nodata"]) as ds:
            raster_ok &= bool(np.array_equal(ds.read(1).astype(bool), ~observed))
        png = Image.open(t["images"]["grayscale"])
        raster_ok &= (
            png.size == (1024, 1024) and Path(t["images"]["grayscale"]).with_suffix(".pgw").exists()
        )
        raster_ok &= abs(observed.mean() - t["observed_fraction"]) < 1e-9
    checks["rasters_native_affine_and_explicit_nodata"] = bool(raster_ok)
    checks["all_sources_full_point_integrity"] = all(
        s["full_source_integrity_passed"] for s in spec["source_integrity"]
    )
    checks["no_shared_ids_with_dev_benchmark_test"] = all(
        not t["shared_ids_with_dev_benchmark_or_test"] for t in spec["tiles"]
    )
    checks["guard_500_from_all_non_training_roles"] = all(
        d >= 500
        for t in spec["tiles"]
        for role, d in t["minimum_distance_by_role_native"].items()
        if not role.startswith("train")
    )
    checks["no_region_training_eligible"] = not any(t["training_eligible"] for t in spec["tiles"])
    return checks


def main():
    latest = load(ROOT / "outputs/reports/phase5_12/latest.json")
    out = Path(latest["report_dir"])
    spec = load(out / "new_region_candidates.json")
    qgis = load(out / "qgis_review_package.json")
    checks = verify(spec)
    dump(out / "geometry_integrity.json", checks)
    assert all(v for k, v in checks.items() if k != "worst_relative_outline_difference"), checks
    (out / "galleries").mkdir(exist_ok=True)
    gallery(
        spec,
        out / "galleries" / "reserve_regions_contact_sheet.png",
        title="Phase 5.12 reserve regions",
    )
    flags = load(out / "unannotated_paint_audit.json")["candidates"]
    tiles = spec["tiles"]
    rows = []
    for t in tiles:
        rows.append(
            {
                "region": t["candidate_id"],
                "reserve_of": t["reserve_of"],
                "priority": t["priority_tag"],
                "source": Path(t["source"]).name,
                "observed_solid_ids": t["observed_solid_ids"],
                "observed_dash_ids": t["observed_dash_ids"],
                "new_solid_ids": t["new_observed_solid_ids"],
                "new_dash_ids": t["new_observed_dash_ids"],
                "observed_paint_pixels": t["observed_paint_pixels"],
                "observation_coverage": t["observed_fraction"],
                "supported": t["supported_classes_in_footprint"],
                "unsupported": t["unsupported_codes_in_footprint"],
                "targets_without_observations": len(t["targets_without_observed_pixels"]),
                "target_paint_unobserved_fraction": t["target_paint_area_unobserved_fraction"],
                "min_distance_non_training": min(
                    d
                    for r, d in t["minimum_distance_by_role_native"].items()
                    if not r.startswith("train")
                ),
                "nearest_non_training_role": min(
                    (
                        (d, r)
                        for r, d in t["minimum_distance_by_role_native"].items()
                        if not r.startswith("train")
                    )
                )[1],
                "shared_ids_with_training": len(t["shares_ids_with_existing_training"]),
                "audit_flags": sum(f["region"] == t["candidate_id"] for f in flags),
            }
        )
    dump(out / "per_region_statistics.json", rows)
    new = spec["new_unique_observed_identities"]
    lines = [
        "# Phase 5.12 — review checklist (10 reserve regions)",
        "",
        f"QGIS project: `{qgis['project']}`. There is one bookmark per region, P512_01 to "
        "P512_10.",
        "Layers: red outlines are candidate solid paint, blue are candidate dash paint, magenta "
        "are unsupported symbols (always excluded), and yellow is the review footprint. The "
        "client layers are read-only.",
        "Two editable layers are provided under **EDITABLE reviewer layers**:",
        "- `Reviewer: draw MISSING paint / problem areas here`: draw a polygon around any "
        "visible marking that has no outline, or any wrong outline.",
        "- `Reviewer: per-outline decision`: optionally set `review_decision` or "
        "`reviewer_note` on individual outlines.",
        "",
        "For each region, reply **APPROVE**, **REJECT**, or **APPROVE WITH NOTES**. Approval "
        "confirms both:",
        "1. The painted-area geometry and solid/dash classes are correct.",
        "2. Every visible target marking inside the observed part of the footprint is "
        "represented. Anything you drew in the missing-paint layer is excluded as an "
        "ignore zone and never becomes background.",
        "",
        "Point-cloud integrity verification is **not** ground-truth approval. Nothing is "
        "approved yet.",
        "",
        "| Region | Priority | Source | Solid/Dash observed | New solid/dash IDs | Coverage | "
        "Audit flags | Reply |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['region']} | {r['priority']} | {r['source']} | "
            f"{r['observed_solid_ids']}/{r['observed_dash_ids']} | "
            f"{r['new_solid_ids']}/{r['new_dash_ids']} | {r['observation_coverage']:.0%} | "
            f"{r['audit_flags']} | |"
        )
    lines += [
        "",
        "Spots worth a specific look (bright observed returns with no outline; these "
        "may be paint, kerbs, barriers or objects):",
        "",
    ]
    for f in flags:
        lines.append(
            f"- {f['region']} at native ({f['centre_native'][0]:.1f}, "
            f"{f['centre_native'][1]:.1f}): {f['pixels']} px, elongation "
            f"{f['elongation']:.1f}"
        )
    lines.append("")
    (out / "review_checklist.md").write_text("\n".join(lines), encoding="utf-8")
    (Path(spec["output"]) / "review_checklist.md").write_text("\n".join(lines), encoding="utf-8")

    validation = {}
    xml = ROOT / "outputs/reports/phase5_12_pytest.xml"
    if xml.exists():
        suite = ET.parse(xml).getroot()
        suite = suite.find("testsuite") if suite.tag == "testsuites" else suite
        validation["pytest"] = {
            k: suite.attrib[k] for k in ("tests", "failures", "errors", "skipped")
        }
        for name in ("phase5_12_ruff.log", "phase5_12_black.log", "phase5_12_healthcheck.json"):
            if (ROOT / "outputs/reports" / name).exists():
                validation[name] = sha256(ROOT / "outputs/reports" / name)
    md = [
        "# Phase 5.12: reserve-region review package",
        "",
        "**Decision: REAL_DATA_REVIEW_REQUIRED.** The 10 reserve windows are packaged and "
        "waiting for your first approval. No region is training-eligible. No training was "
        "run, no thresholds were changed, the default model is unchanged, and the protected "
        "final test was not read.",
        "",
        "## What was done",
        "",
        "- The 10 windows that passed actual-point verification in Phase 5.11 (P511_17 to "
        "P511_26) were re-rendered from their single recorded source file. That meant 9 LAZ "
        "files and no share-wide search. Every file passed strict full-point and fingerprint "
        f"integrity again ({sum(s['actual_points'] for s in spec['source_integrity']):,} "
        "points).",
        "- Imagery uses the native 0.05 grid with exact affines, the frozen shared "
        "normalization and gamma .7. Observed and explicit NoData masks are written "
        "(NoData is exactly the complement of observed).",
        "- Painted outlines come from client centerlines buffered by their recorded LARGURA "
        "width. Every row keeps its stable source ID, client ID_SIG and ID_GRUPO marking "
        "group. Individual dashes are separate rows.",
        "- The 16 approved Phase 5.11 regions, their dataset, checkpoints, evaluations and "
        "all reservations are untouched.",
        "",
        "## Geometry and integrity checks",
        "",
        f"`{json.dumps(checks)}`",
        "",
        "## Per-region statistics",
        "",
        "| Region | Priority | Source | Solid/Dash observed | New S/D | Paint px | Coverage | "
        "Target paint unobserved | Targets w/o returns | Unsupported | Nearest non-train role "
        "(dist) |",
        "|---|---|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['region']} | {r['priority']} | {r['source']} | "
            f"{r['observed_solid_ids']}/{r['observed_dash_ids']} | "
            f"{r['new_solid_ids']}/{r['new_dash_ids']} | {r['observed_paint_pixels']} | "
            f"{r['observation_coverage']:.0%} | {r['target_paint_unobserved_fraction']:.0%} | "
            f"{r['targets_without_observations']} | {r['unsupported'] or '-'} | "
            f"{r['nearest_non_training_role']} ({r['min_distance_non_training']:.0f}) |"
        )
    md += [
        "",
        "## Important findings",
        "",
        f"- **New identities if all 10 are approved: {new['solid']} solid and {new['dash']} "
        "dash.** Every solid in these windows is a segment of a long edge-line feature that "
        "already appears in training. These windows still add new road sections of straight "
        "edge lines, which addresses the legacy regression, but they add no new unique solid "
        "identities. They contribute mainly new dashes, merge areas and straight-edge "
        "exposure.",
        "- Observation is one-sided. Typically 56–76% of the outlined target paint area has "
        "no LiDAR returns, because only one carriageway was scanned. Those pixels are NoData "
        "and are never background. Targets with no returns at all are listed per region and "
        "cannot be supervised.",
        "- Leakage risk: there are no shared IDs with any development, benchmark or "
        "protected-test region. **P512_02 is 502 native units from PHASE58_POSITIVE_DEV_REVIEW**, "
        "just over the 500 guard. It is allowed under the rule, but if you prefer a wider "
        "margin for the development area, reject it. P512_09 and P512_10 are 517 apart and "
        "come from the same source file, so they are nearby sections rather than "
        "independent ones.",
        "- The missed-paint audit flagged a few unoutlined bright returns. Most look like "
        "kerbs, barriers or objects, but in P512_04 there is a long thin line between two "
        "outlined lines that you should check specifically. See "
        "`galleries/unannotated_bright_blobs.png` and the checklist.",
        "",
        "## Deliverables",
        "",
        f"- QGIS project with 10 bookmarks and 2 editable reviewer layers: `{qgis['project']}`",
        "- Contact sheet: `galleries/reserve_regions_contact_sheet.png`",
        "- Per-region statistics: `per_region_statistics.json`. Source and geometry "
        "provenance: `new_region_candidates.json`, `source_integrity.json`, "
        "`geometry_integrity.json`",
        "- Review instructions: `review_checklist.md`",
        "",
        "After approval, the next step is to export these regions with the Phase 5.11 export "
        "rules and run one controlled experiment with the same protocol and budget, compared "
        "on both development areas.",
        "",
        f"Validation: `{validation}`",
        "",
    ]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    dump(
        out / "report.json",
        {
            "decision": "REAL_DATA_REVIEW_REQUIRED",
            "checks": checks,
            "per_region": rows,
            "new_unique_observed_identities": new,
            "qgis": qgis,
            "validation": validation,
            "training_performed": False,
            "protected_final_test_used": False,
        },
    )
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
