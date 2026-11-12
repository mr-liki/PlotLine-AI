"""Validate and seal Stage 9 evidence; stop at first review of new regions only."""

import csv
import hashlib
import re
import shutil
from itertools import combinations
from pathlib import Path

import pyogrio
import rasterio
from shapely.geometry import box

from phase5_11_common import ROOT, load
from roadmarking.rasterization.pipeline import dump
from stage9_common import BASE, context, identity_group, out_dir


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = out_dir()
    ctx = context()
    baseline = load(out / "preservation_baseline.json")
    assert all(sha(p) == expected for p, expected in baseline["protected_approval_inputs"].items())
    assert all(sha(p) == expected for p, expected in baseline["prior_manifests"].items())
    ai = load(ROOT / "outputs/reports/stage8b/ai_preservation.json")
    assert all(sha(row["path"]) == row["expected_sha256"] for row in ai["checks"])
    old_masks = load(ROOT / "outputs/reports/stage8b/source_integrity_and_coverage.json")["regions"]
    mask_hashes = {r["region"]: r["observed_raster_sha256"] for r in old_masks}
    for manifest in (
        ROOT / "outputs/reports/stage7_new_test/20261107T051050944913Z/new_region_candidates.json",
        Path(load(ROOT / "outputs/reports/stage8_new_solid_review/latest.json")["report_dir"])
        / "new_region_candidates.json",
    ):
        for tile in load(manifest)["tiles"]:
            assert sha(tile["layers"]["observed"]) == mask_hashes[tile["candidate_id"]]
    historic = load(ROOT / "outputs/reports/stage7/package_manifest.json")["files"]
    preserved_scores = {}
    for path, expected in historic.items():
        if path.endswith(
            ("7F_stage3_link_breakdown.json", "7H_results.json", "baseline_results.json")
        ):
            assert sha(ROOT / path) == expected
            preserved_scores[path] = expected
    spec = load(out / "new_region_candidates.json")
    qgis = load(out / "qgis_review_package.json")
    assert qgis["all_layers_valid"] and qgis["all_client_and_candidate_layers_read_only"]
    data = Path(spec["output"])
    assert not list(data.rglob("*.npz")), "No supervised target/crop arrays may be exported"
    assert all(
        s["full_source_integrity_passed"]
        and s["expected_points"] == s["actual_points"]
        and s["dropped_points"] == 0
        for s in spec["source_integrity"]
    )
    train_groups = {
        identity_group(ctx, u)
        for r in ctx["used"]
        if r["role"].startswith("train")
        for u in r["ids"]
    }
    dev_groups = {
        identity_group(ctx, u) for r in ctx["used"] if "dev" in r["role"] for u in r["ids"]
    }
    for tile in spec["tiles"]:
        geometry = box(*tile["bounds"])
        assert all(geometry.intersection(r["box"]).area == 0 for r in ctx["used"])
        assert (
            tile["training_eligible"] is False
            and tile["approval_status"] == "PENDING_FIRST_HUMAN_REVIEW"
        )
        group_set = set(tile["physical_groups"])
        assert not group_set & ctx["protected_groups"]
        if tile["proposed_role"] == "development_candidate":
            assert (
                not group_set & train_groups
                and tile["split_audit"]["distance_to_training_native"] >= 500
            )
        else:
            assert (
                not group_set & dev_groups
                and tile["split_audit"]["distance_to_development_native"] >= 500
            )
        assert tile["split_audit"]["distance_to_protected_and_historical_native"] >= 500
        with rasterio.open(tile["layers"]["observed"]) as ds:
            observed = ds.read(1)
            assert ds.crs is None
            assert abs(float(observed.mean()) - tile["observed_fraction"]) < 1e-8
    for a, b in combinations(spec["tiles"], 2):
        assert box(*a["bounds"]).distance(box(*b["bounds"])) >= 500
        if a["proposed_role"] != b["proposed_role"]:
            assert not (set(a["physical_groups"]) & set(b["physical_groups"]))
    reviewers = pyogrio.read_dataframe(spec["review_edits_store"], layer="reviewer_outline_check")
    assert reviewers["review_decision"].fillna("").eq("").all()
    audit = load(out / "physical_line_audit.json")
    assert len(audit["exact_three_groups"]) == 3
    assert all(
        any(p["path"] for p in g["paths_using_only_supported_edges"])
        for g in audit["exact_three_groups"]
    )
    validation = {
        "all_passed": True,
        "protected_approval_hashes_unchanged": len(baseline["protected_approval_inputs"]),
        "ten_protected_observed_masks_unchanged": True,
        "sealed_AI_artifacts_match": ai["checks"],
        "historical_scores_unchanged": preserved_scores,
        "new_regions_do_not_overlap_previously_approved_footprints": True,
        "opposite_roles_and_protected_footprints_guard_ge_500_native": True,
        "physical_group_disjoint_new_roles": True,
        "new_candidate_geometries_not_approved_or_exported_as_supervised_arrays": True,
        "all_three_dependencies_have_supported_geometry_paths": True,
    }
    dump(out / "artifact_validation.json", validation)
    inventory = load(out / "solid_acquisition_inventory.json")
    matrix = load(out / "opportunity_matrix.json")
    root_logs = BASE
    for name in (
        "audit.log",
        "opportunities.log",
        "review.log",
        "plan.log",
        "qgis.log",
        "pytest.log",
        "healthcheck.log",
        "healthcheck.json",
        "ruff.log",
        "black.log",
    ):
        if (root_logs / name).exists():
            shutil.copyfile(root_logs / name, out / name)
    tests = re.search(
        r"(\d+) passed, (\d+) skipped", (out / "pytest.log").read_text(encoding="utf-8")
    )
    assert tests and "All checks passed!" in (out / "ruff.log").read_text(encoding="utf-8")
    assert "left unchanged" in (out / "black.log").read_text(encoding="utf-8")
    assert "10/10 checks OK" in (out / "healthcheck.log").read_text(encoding="utf-8")
    rows = []
    for t in spec["tiles"]:
        stats = t["support_statistics"]
        values = stats["per_id"]
        rows.append(
            {
                "region": t["candidate_id"],
                "proposed_role": t["proposed_role"],
                "acquisition": t["acquisition"],
                "observed_paint_pixels": t["observed_paint_pixels"],
                "solid_ids": sum(v["class"] == "solid" for v in values),
                "dash_ids": sum(v["class"] == "dash" for v in values),
                "observed_new_solid_groups_ge_10": len(
                    {
                        v["physical_group"]
                        for v in values
                        if v["class"] == "solid"
                        and v["physical_group"] in t["new_solid_groups"]
                        and v["supported_length_native_bins"] >= 10
                    }
                ),
                "eligible_nonoverlap_crops_after_approval": stats[
                    "eligible_nonoverlap_crops_after_approval"
                ],
                "distance_to_training_native": t["split_audit"]["distance_to_training_native"],
                "distance_to_development_native": t["split_audit"][
                    "distance_to_development_native"
                ],
                "physical_groups": ";".join(t["physical_groups"]),
                "class_ids": ";".join(v["global_id"] for v in values),
                "intensity_and_contrast": str(
                    [
                        {
                            k: v[k]
                            for k in (
                                "global_id",
                                "median_paint_normalized_intensity",
                                "contrast_normalized",
                                "client_width_native",
                            )
                        }
                        for v in values
                    ]
                ),
                "first_human_approval_required": True,
            }
        )
    with (out / "verified_opportunity_matrix.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    protected = load(ROOT / "outputs/reports/stage8b/evaluation_eligibility.json")
    dump(
        out / "protected_test_preservation.json",
        {
            "T7": protected["populations"]["T7"],
            "S8": protected["populations"]["S8"],
            "approval_manifest_sha256": baseline["approval_manifest_sha256"],
            "geometry_and_mask_preservation_validated": True,
            "new_test_inference": False,
            "test_annotation_review_repeated": False,
            "policy": "Preserve exact approved exclusions and T7_05 confirmed dash plus uncertain-boundary ignore. Shared solids excluded from unseen-solid scoring remain ignored paint, never negative background. No test-to-train reassignment.",  # noqa: E501
        },
    )
    status = "NEW_REAL_DATA_REVIEW_REQUIRED"
    lines = [
        "# Stage 9 — physical-line independence and recognition readiness",
        "",
        f"Final status: **{status}**",
        "",
        "No training, test inference, model selection, calibration/stitching change, digitization, role reassignment or commit. The frozen candidate/default and all prior approvals, geometry, observations and historical scores remain unchanged.",  # noqa: E501
        "",
        "## 1. Physical-line leakage audit",
        "",
        f"Audited **{len(audit['regions'])} role-tagged regions**, **{len(audit['physical_groups'])} physical solid groups** and **{len(audit['edge_evidence'])} nearby geometry pairs**. Direct training/development feature-ID overlap is **zero**. Nonetheless three connected physical groups link approved training to **NEW_POSITIVE_DEV / PHASE58_POSITIVE_DEV_REVIEW**, across Job_20250708_0834 and Job_20250708_1218. Geographic training/development footprint separation is 501.554 native units; it does not imply marking independence.",  # noqa: E501
        "",
        "Graph starting point: Stage-8B continuation rule on authoritative client geometry. Stage 9 adds exact endpoint positions/gaps, tangent angles, centreline intersection lengths, duplicate/paint-overlap evidence and supported shortest paths. Close parallel markings are reported separately. Ambiguous local joins remain qualified/conservatively grouped; no approved test exclusion is relaxed.",  # noqa: E501
        "",
        "| Group | Client IDs in component | Actual training target IDs | Actual dev target IDs | Observed affected dev paint px | Strong paths |",  # noqa: E501
        "|---|---:|---:|---:|---:|---:|",
    ]
    for g in audit["exact_three_groups"]:
        lines.append(
            f"| {g['reported_group']} | {len(g['member_ids'])} | {len(g['observed_training_pixel_evidence'])} | {len(g['observed_development_pixel_evidence'])} | {sum(r['observed_paint_pixels'] for r in g['observed_development_pixel_evidence'])} | {sum(bool(p['path']) for p in g['paths_using_only_supported_edges'])} |"  # noqa: E501
        )
    lines += [
        "",
        "Full exact IDs, source-layer/FID identities, affected regions, acquisitions and every path/edge are in physical_line_audit.json. Geometry maps and shortest paths are in continuation_evidence/group_1.png through group_3.png.",  # noqa: E501
    ]
    for g in audit["exact_three_groups"]:
        lines += [
            "",
            f"### Group {g['reported_group']}",
            "Training IDs: " + ", ".join(g["training_ids"]),
            "Development IDs: " + ", ".join(g["development_ids"]),
            "Evidence classes: " + str(g["evidence_classification_counts"]),
        ]
    lines += [
        "",
        "### Metric qualification and acquisition/footprint distinctions",
        "The four affected development solid IDs account for **4,870 / 6,415** observed solid-paint pixels in NEW_POSITIVE_DEV. This is a label-support/dependency audit, not a recomputed performance score. Its full-solid/continuity and any pooled pixel/instance results including those targets cannot establish wholly unseen-line development generalization. Historical numbers remain byte-for-byte unchanged. LEGACY_POSITIVE_DEV and S6_01 have no training-group target overlap under this graph, but remain repeatedly used selection evidence.",  # noqa: E501
        "The model-specific, SHA-matched training manifests show default exposure to at least groups 1 and 3, and frozen-candidate exposure to all three. Parent/warm-start exposure is not erased by future dataset reassignment. Detailed provenance is in the audit.",  # noqa: E501
        "Direct ID leakage, physical-group sharing, acquisition equality, footprint intersection area and native distance are separate fields for every cross-role region pair. FUTURE_TEST_1/2/3 and consumed NEW_FINAL_TEST_RESERVED remain historical, not untouched tests. Context sharing in T7/S8 is not allowed to inflate their eligible unseen populations.",  # noqa: E501
        "",
        "## 2. Group-safe split proposal",
        "Preserve existing roles. For future development protocols, explicitly exclude the four dependent solid targets and their declared uncertainty/context masks from unseen-line metrics; do not turn that paint into verified background. This leaves **45 historical target identities (10 solid IDs in 9 physical groups, 35 dash IDs)** with 12,230 observed solid pixels and 4,172 dash pixels. These are metadata counts, not fresh scores or newly independent samples.",  # noqa: E501
        "Add the three new development regions only after first geometry and complete-visible-marking review. Reserve N9_04 for training, not development. Opposite proposed roles share no marking/physical groups and are separated by >=500 native units; all are >=500 from protected/historical footprints, and none overlaps any already approved footprint. Same-role proximity is disclosed: some new windows extend an existing development corridor and do not represent new acquisition domains.",  # noqa: E501
        "If instead retiring the three groups from future training is desired, split_proposal.json lists every affected approved training region. This requires a separate explicit decision and clean-origin future training; moving roles cannot make current weights forget prior exposure. No reassignment was applied.",  # noqa: E501
        "",
        "## 3. Independent solids and acquisition diversity",
        "The verified support inventory covers the 255 Stage-8 screened solids; the full client inventory has 578. Unmeasured/excluded client solids are not declared absent or unobservable. Solid shape, width, edge/merge designation, exact acquisition support and distances are in solid_acquisition_inventory.json.",  # noqa: E501
        "",
        "| Acquisition | Supported IDs >=10 | Physical groups | Previously unused groups | Groups repeated in another acquisition | Groups exclusive in verified inventory |",  # noqa: E501
        "|---|---:|---:|---:|---:|---:|",
    ]
    for job, s in inventory["acquisition_summary"].items():
        lines.append(
            f"| {job} | {s['supported_client_ids']} | {s['supported_physical_groups']} | {s['previously_unused_supported_groups']} | {s['groups_also_observed_in_other_acquisitions']} | {len(s['groups_unique_to_this_acquisition_in_verified_inventory'])} |"  # noqa: E501
        )
    lines += [
        "",
        "All seven supported September physical groups also have July observations. September offers different acquisition conditions, not seven new geographic physical lines. Its six previously unused groups are new relative to current role assignments, while still repeated across acquisitions. July-1218 has four acquisition-exclusive groups in this screened inventory; their geometry is not assumed to satisfy geographic guards or observed-window approval. Counts are not additive independent examples.",  # noqa: E501
        "",
        "## 4. New real-data opportunity matrix and review package",
        f"Metadata search considered **{matrix['summary']['metadata_windows_considered']}** windows. Header coverage was only a screen; every packaged window passed whole-source integrity and actual observed new-solid support >=10 native-unit bins, with observed dash paint. Failed sources were excluded; alternate verified acquisitions were tried when a header-covered window lacked adequate paint. Three validated raster sets were reused after fingerprint verification when replacing an overlapping draft window.",  # noqa: E501
        "No new supervised crop/target arrays were exported. Candidate masks were transient coverage calculations only. Client painted-area candidates are review evidence, not approved labels. Actual raw grayscale, NoData/observed masks, unsupported marks, source and physical IDs, proposed roles and four named bookmarks are in the QGIS project.",  # noqa: E501
        "",
        "| New region | Proposed role | Acquisition | Observed paint px | Solid / dash IDs | New supported solid groups | Non-overlap crop candidates |",  # noqa: E501
        "|---|---|---|---:|---|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['region']} | {r['proposed_role']} | {r['acquisition']} | {r['observed_paint_pixels']} | {r['solid_ids']} / {r['dash_ids']} | {r['observed_new_solid_groups_ge_10']} | {r['eligible_nonoverlap_crops_after_approval']} |"  # noqa: E501
        )
    lines += [
        "",
        "Full matrix: opportunity_matrix.json (unverified windows retain null window paint/crop counts); verified rows: verified_opportunity_matrix.csv, including intensity/contrast and native client widths. Crop counts are possible exposures after approval, not independent examples. Thin/faded proxies and bright non-road pixels require human classification/completeness review; they are not automatically verified negatives.",  # noqa: E501
        f"QGIS: **{qgis['project']}**. Source/client/candidate layers are read-only; reviewer decision/problem layers are editable. **First review is requested only for N9_01–N9_04.** No repeat review of T7, S8 or other previously approved regions.",  # noqa: E501
        "",
        "## 5. Protected cohorts and recognition readiness",
        "T7 stays protected for **58 dash identities**, including the confirmed T7_05 patch and its 11-pixel uncertain-boundary ignore. S8 stays protected for **5 solid IDs / 4 physical groups**. Exact geometry, class mapping, source hashes, observations and exclusion rules remain unchanged. Shared solid paint is excluded from unseen-solid scoring and never negative background. No protected model predictions, images or metrics were used for candidate/model selection; only role/identity metadata and preservation hashes were consulted.",  # noqa: E501
        "Real painted-pixel and individual-dash measurement definitions exist, but dependable performance is not established. Complete solid-line recognition and continuity through NoData remain unsupported. Thin/faded strata, merge/split diversity and observed roadside hard negatives are not sufficiently approved/stratified. New development adds only three new solid groups in the trained July acquisition; broader road/acquisition generalization remains unproven. September training would no longer be an unseen acquisition condition. See recognition_readiness.json for separate evidence and limitations for all seven reliability dimensions.",  # noqa: E501
        "",
        "## 6. Next controlled experiment — proposal only",
        "After first approval and split/ignore-rule sealing, compare one real-data expansion against the unchanged frozen candidate with fixed Mask2Former Swin-S. Three seeds (895/896/897), at most 400 steps, evaluations at 200/400, conservative step-200 default with a predeclared class/region-balanced development-only rule. Log actual observed paint exposure by physical group/class/acquisition and cap dominant long-line groups. Fixed thresholds, NMS, stitching and normalization; no recalibration, synthetic mixing, self-training or repeated Stage-7 2×2. No protected-test selection or inference. Predeclare numerical reliability targets, sufficient group/stratum coverage and dependence-aware uncertainty before any future run; no experiment is authorized now. Detailed conditions, failure/stopping rules and calibration-stability assessment are in next_experiment_proposal.json.",  # noqa: E501
        "",
        "## 7. Verification and final gate",
        f"**{tests.group(1)} passed, {tests.group(2)} skipped; Ruff and Black pass; healthcheck 10/10.** artifact_validation.json verifies the 33 approval-bound inputs, all ten observed-support masks, sealed AI hashes, historical score hashes, role/group guards, no old-footprint overlap and no supervised label export. External new review assets are hash-bound in review_assets_manifest.json.",  # noqa: E501
        f"**{status}**. Review only the four new regions and decide their proposed roles after geometry/complete-visible-marking acceptance. No old approval or model was changed. Stop before supervision, training, protected-test consumption or digitization.",  # noqa: E501
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    review_files = [
        {"path": str(p), "size": p.stat().st_size, "sha256": sha(p)}
        for p in sorted(data.rglob("*"))
        if p.is_file()
    ]
    dump(
        out / "review_assets_manifest.json",
        {
            "files": review_files,
            "first_human_approval_required": [t["candidate_id"] for t in spec["tiles"]],
        },
    )
    dump(
        out / "source_manifest.json",
        {"files": [{"path": str(p), "sha256": sha(p)} for p in ROOT.glob("scripts/stage9*.py")]},
    )
    files = [
        {"path": p.relative_to(out).as_posix(), "size": p.stat().st_size, "sha256": sha(p)}
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "package_manifest.json"
    ]
    dump(
        out / "package_manifest.json", {"version": out.name, "final_status": status, "files": files}
    )
    dump(
        BASE / "latest.json",
        {
            **load(BASE / "latest.json"),
            "status": status,
            "report": str(out / "report.md"),
            "review_project": qgis["project"],
        },
    )
    (BASE / "report.md").write_text(
        f"# Stage 9\n\nStatus: **{status}**\n\n[Versioned report]({out.name}/report.md)\n\n[New-regions QGIS project]({qgis['project']})\n\nOnly N9_01-N9_04 need first human review. The ten T7/S8 regions remain approved and protected. No training or test inference.\n",  # noqa: E501
        encoding="utf-8",
    )
    print(
        f"Stage 9 sealed: {status}; {len(files)} report files, {len(review_files)} new review assets"  # noqa: E501
    )


if __name__ == "__main__":
    main()
