"""Seal Stage 8B review, correction and restricted-population audit evidence."""

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path

from phase5_11_common import ROOT, load
from roadmarking.rasterization.pipeline import dump

OUT = ROOT / "outputs/reports/stage8b"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    approval = load(OUT / "approval_manifest.json")
    assert all(sha(path) == expected for path, expected in approval["sha256"].items())
    correction = load(OUT / "single_dash_geometry_nodata_audit.json")
    independence = load(OUT / "independence_audit.json")
    coverage = load(OUT / "source_integrity_and_coverage.json")
    inventory = load(OUT / "acquisition_support/resolved_inventory.json")
    assert inventory["mixed_identity_union_mismatches"] == 0
    assert inventory["unresolved_acquisition_attribution"] == 0
    assert coverage["source_integrity_passed_for_all_selected_regions"]
    assert all(r["guard_ge_500"] for r in independence["footprint_separation"])
    assert all(r["guard_ge_500"] for r in independence["candidate_pair_separation"])
    pytest_summary = re.search(
        r"(\d+) passed, (\d+) skipped", (OUT / "pytest.log").read_text(encoding="utf-8")
    )
    assert pytest_summary
    assert "All checks passed!" in (OUT / "ruff.log").read_text(encoding="utf-8")
    assert "left unchanged" in (OUT / "black.log").read_text(encoding="utf-8")
    assert "10/10 checks OK" in (OUT / "healthcheck.log").read_text(encoding="utf-8")
    assert load(OUT / "artifact_validation.json")["all_passed"]
    earlier_excluded = load(
        ROOT / "outputs/reports/stage8/8_3_solid_discovery/solid_inventory.json"
    )["summary"]["excluded_integrity_failed_sources"]
    excluded = sorted(
        set(earlier_excluded) | set(inventory["stage8_integrity_failed_sources_excluded"])
    )
    dump(
        OUT / "source_integrity_findings.json",
        {
            "selected_region_sources": coverage["sources"],
            "all_selected_region_sources_valid": True,
            "mixed_acquisition_scans": len(inventory["scans"]),
            "new_integrity_failures": inventory["new_integrity_failures"],
            "all_known_integrity_failed_sources_excluded": excluded,
            "mixed_identity_union_mismatches": inventory["mixed_identity_union_mismatches"],
            "unresolved_acquisition_attribution": 0,
            "fingerprint_limitation": "First/last MiB hash plus size/mtime binds reused full-decode integrity; not a full-file hash.",  # noqa: E501
        },
    )
    populations = {}
    for name, kind in (("T7", "dash_focused"), ("S8", "unseen_solid")):
        members = [
            r
            for r in independence["populations"]
            if r["cohort"] == name and r["population_candidate"] == kind and r["identity_eligible"]
        ]
        populations[name] = {
            "role": kind,
            "protected": True,
            "annotation_review_accepted": True,
            "source_integrity_validated": True,
            "member_ids": sorted({r["global_id"] for r in members}),
            "member_appearances": members,
            "physical_solid_groups": sorted(
                {r["physical_solid_group"] for r in members if r["physical_solid_group"]}
            ),
            "excluded_shared_or_out_of_scope_solids": independence["summary"][name][
                "excluded_shared_or_out_of_scope_solid_ids"
            ],
            "excluded_paint_policy": "Ignore for this population; never verified negative background or unseen-solid true positives.",  # noqa: E501
            "test_execution_authorized": False,
        }
    decision = "INDEPENDENCE_RESTRICTIONS_REMAIN"
    dump(
        OUT / "evaluation_eligibility.json",
        {
            "decision": decision,
            "populations": populations,
            "approval_manifest_sha256": sha(OUT / "approval_manifest.json"),
            "coverage_audit_sha256": sha(OUT / "source_integrity_and_coverage.json"),
            "independence_audit_sha256": sha(OUT / "independence_audit.json"),
            "mixed_acquisition_resolution_sha256": sha(
                OUT / "acquisition_support/resolved_inventory.json"
            ),
            "fully_independent_combined_benchmark": False,
            "test_inference_or_metrics_computed": False,
            "remaining_restrictions": [
                "T7 shared solids are seen identities and outside its unseen-solid population.",
                "S8 shared/context solids are excluded; five novel client solid IDs form four metadata-derived physical-line groups.",  # noqa: E501
                "All selected regions are from the trained acquisition; no unseen-acquisition claim.",  # noqa: E501
                "Training/development have three shared physical-line groups under the refined continuation screen.",  # noqa: E501
                "Physical grouping is an explicit geometric screen, not exhaustive human certification of every cross-ID continuation.",  # noqa: E501
                "Seal population-specific scoring/ignore rules before any future explicitly authorized test execution. No test inference in Stage 8B.",  # noqa: E501
            ],
        },
    )
    confirmed = dict(correction)
    confirmed.update(
        {
            "approval": "USER_CONFIRMED_SINGLE_MARK_CORRECTION",
            "approval_manifest_sha256": sha(OUT / "approval_manifest.json"),
            "confirmed_geometry": str(OUT / "single_dash_confirmed.gpkg"),
            "original_packages_unchanged": True,
            "effective_geometry_recipe": "Immutable original T7 candidate geometry + one confirmed additive dash; exclude 11 uncertain boundary pixels. No evaluation dataset or test inference produced.",  # noqa: E501
        }
    )
    dump(OUT / "confirmed_geometry_nodata_audit.json", confirmed)
    minimum_prior = min(
        r["distance_to_prior_role_footprints"] for r in independence["footprint_separation"]
    )
    minimum_pairs = min(r["distance"] for r in independence["candidate_pair_separation"])
    lines = [
        "# Stage 8B — accepted review, one confirmed dash, restricted evaluation populations",
        "",
        f"Final decision: **{decision}**",
        "",
        "## Human review and the one correction",
        "",
        "The user reviewed all eight T7 and both S8 regions, accepted all other reviewed geometry/classifications, and explicitly answered **Confirm this single-mark correction** to the localized T7_05 before/after image. The approval manifest binds original candidate stores, reviewer stores, reference geometry, QGIS projects/previews, the screenshot, the one proposed/confirmed patch and its QA evidence. No original package, approved geometry or source client file was overwritten.",  # noqa: E501
        "",
        "The missing dash is in **T7_05**, on **Job_20250708_0834_Track01_50.laz**. Screenshot matching used 21 inliers with median residual 0.271 source pixel. Approximate screenshot coordinates were used only for localization; the annotation comes from native source observations.",  # noqa: E501
        f"Native centroid: **({correction['native_centroid'][0]:.6f}, {correction['native_centroid'][1]:.6f})**. Native bounds: `{correction['native_bounds']}`. CRS remains unconfirmed; no EPSG, latitude/longitude or metre conversion was invented.",  # noqa: E501
        "One local identity, **stage8b_T7_05_missing_dash_001**, class dash (visual class 2 / model class 1), with client ID null. No existing client geometry intersects the traced pixels; no client feature ID was guessed.",  # noqa: E501
        "",
        "## Geometry and NoData audit",
        "",
        "The confirmed MultiPolygon contains **248 observed paint pixels**, area **0.62 native units²**, and **zero unobserved pixels**. The oriented envelope contains 108 NoData pixels, which remain outside the paint geometry. Fragment grouping did not fill the exported polygon. The single identity is preserved across observed fragments. Eleven uncertain boundary pixels are locally ignored, rather than called verified background. Photometric tracing is confined to this annotation and changes no AI threshold, score, normalization or stitching rule.",  # noqa: E501
        "",
        "Focused QA: [before/after](single_dash_before_after.png). Confirmed annotation: `single_dash_confirmed.gpkg`. Separate project: `T7_05_confirmed_correction.qgs`. Original/proposed evidence is retained. The prior proposal audit remains immutable; its confirmed status is recorded separately in confirmed_geometry_nodata_audit.json.",  # noqa: E501
        "",
        "## Source integrity and acquisition attribution",
        "",
        f"All ten selected regions reconcile their per-ID observed-pixel counts. Their {len(coverage['sources'])} unique source files retain the fingerprint of a successful full decode (actual points equal expected, zero dropped). Fingerprints combine first/last-MiB SHA-256, size and mtime and are explicitly **not full-file hashes**. Existing raw source files were read only.",  # noqa: E501
        f"Mixed-acquisition uncertainty is resolved with {len(inventory['scans'])} cached full-source decodes: **{len(inventory['new_integrity_failures'])} new integrity failures**, **0 unresolved attributions**, and **0 mixed-identity union mismatches** against Stage 8. All reused verified source fingerprints are unchanged. The 14 known failed Stage-8 sources remain excluded. No partial decode contributes support.",  # noqa: E501
        "",
        "| Acquisition | Solid IDs with observed paint | IDs supported >=10 units | Observed paint pixels | Supported 1-unit bins |",  # noqa: E501
        "|---|---:|---:|---:|---:|",
    ]
    for job, row in inventory["summary"].items():
        lines.append(
            f"| {job} | {row['solid_ids_with_any_observed_paint']} | {row['solid_ids_supported_length_ge_10']} | {row['observed_paint_pixels']} | {row['supported_length']} |"  # noqa: E501
        )
    lines += [
        "",
        "Acquisition rows are observations of identities, not independent identities. Overlapping acquisitions and repeated centreline bins must not be summed as globally distinct paint. Detailed per-ID/acquisition counts and source/cache provenance are in acquisition_support/resolved_inventory.json.",  # noqa: E501
        "",
        "## Training/development/test independence",
        "",
        f"Minimum candidate-to-prior-role footprint distance is **{minimum_prior:.3f} native units**; minimum distance between any two candidate windows is **{minimum_pairs:.3f}**. All pass the 500-unit guard. Mark-ID and physical-line checks use role metadata and original client geometry, not prediction scores or test-based checkpoint selection.",  # noqa: E501
        "The refined continuation graph rasterizes only local 4×4-unit client-geometry neighborhoods and applies unchanged link_audit.physical_lines rules, then takes transitive closure across IDs. It distinguishes close parallel paint from continuations, unlike Stage 8's broader proximity exclusion. The existing training/development sets share three physical-line groups under this screen despite direct ID disjointness and >500-unit geographic separation. These dependencies remain disclosed; no split was silently reassigned.",  # noqa: E501
        "",
        "| Cohort | Primary population | Eligible client/local IDs | Physical solid groups | Shared-solid policy |",  # noqa: E501
        "|---|---|---:|---:|---|",
        "| T7 | Protected dash-focused candidate | 58 | n/a | Shared solids excluded from unseen-solid counts |",  # noqa: E501
        "| S8 | Protected unseen-solid candidate | 5 | 4 | Two shared/context solid IDs excluded |",
        "",
        "T7 now has 57 existing observed dash identities plus the one confirmed local dash. S8's five identity-eligible solid IDs each have 49–52 supported one-unit bins in their reviewed windows. Two client IDs belong to the same metadata-derived physical-line group, so five client IDs are not five statistically independent physical lines. Exact member IDs and per-region support are bound in evaluation_eligibility.json and source_integrity_and_coverage.json. All ten windows use the trained July acquisition: no unseen-acquisition generalization claim is made.",  # noqa: E501
        "Do not merge these cohorts into a fully independent solid-and-dash benchmark. Keep their targets, ignore masks, denominators and dependence groups separate. Excluded shared/context paint must not become verified negative background. Geometry acceptance does not authorize evaluation or remove identity restrictions.",  # noqa: E501
        "",
        "## Verification and protected-test boundary",
        "",
        f"Regression: **{pytest_summary.group(1)} passed, {pytest_summary.group(2)} skipped**; **Ruff and Black pass**; **healthcheck 10/10**. artifact_validation.json checks approval hashes, exact observed support, original-geometry preservation, ignore/paint disjointness and one-dash identity. ai_preservation.json verifies frozen weights, default weights and production stitching against sealed hashes.",  # noqa: E501
        "No training/fine-tuning, model inference, test metrics, threshold tuning, checkpoint selection, score recalibration, normalization change, stitching change, promotion, production digitization or commit occurred. Work on protected data was limited to the explicitly requested review/correction and source/identity audits. No protected final test was consumed for model evaluation or selection.",  # noqa: E501
        "",
        "**Stop:** annotation review and the single correction are confirmed. Mixed-acquisition attribution is resolved. **INDEPENDENCE_RESTRICTIONS_REMAIN**; any future test must retain the declared cohort/instance exclusions and receive explicit execution authorization after its scoring protocol is sealed.",  # noqa: E501
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    dump(
        OUT / "final_decision.json",
        {
            "decision": decision,
            "single_mark_confirmation_received": True,
            "geometry_approval_manifest": "approval_manifest.json",
            "mixed_acquisition_uncertainty_resolved": True,
            "selected_source_integrity_passed": True,
            "protected_test_consumed": False,
        },
    )
    sources = [*ROOT.glob("scripts/stage8b*.py")]
    dump(
        OUT / "source_manifest.json",
        {"files": [{"path": str(p), "sha256": sha(p)} for p in sources]},
    )
    files = [
        {"path": p.relative_to(OUT).as_posix(), "sha256": sha(p), "size": p.stat().st_size}
        for p in sorted(OUT.rglob("*"))
        if p.is_file() and p.name != "package_manifest.json"
    ]
    dump(
        OUT / "package_manifest.json",
        {
            "schema_version": 1,
            "package_version": datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"),
            "decision": decision,
            "files": files,
        },
    )
    print(f"Stage 8B sealed: {decision}; {len(files)} files")


if __name__ == "__main__":
    main()
