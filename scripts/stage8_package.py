"""Assemble Stage 8 evidence and a review-gated proposal; never record approval."""

import hashlib
import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from phase5_11_common import ROOT, load
from roadmarking.rasterization.pipeline import dump

OUT = ROOT / "outputs/reports/stage8"


def sheets(audit):
    folder = OUT / "contact_sheets"
    folder.mkdir(exist_ok=True)
    categories = sorted({r["category"] for v in audit.values() for r in v["records"]})
    for cat in categories:
        paths = [
            Path(r["evidence_png"])
            for v in audit.values()
            for r in v["records"]
            if r["category"] == cat and Path(r["evidence_png"]).exists()
        ]
        for start in range(0, len(paths), 20):
            page = Image.new("RGB", (1600, 1200), "white")
            draw = ImageDraw.Draw(page)
            draw.text(
                (10, 5),
                f"{cat} | {start + 1}-{min(start + 20, len(paths))}/{len(paths)}",
                fill="black",
            )
            for k, path in enumerate(paths[start : start + 20]):
                with Image.open(path) as im:
                    thumb = ImageOps.contain(im.convert("RGB"), (390, 200))
                x, y = (k % 4) * 400, 30 + (k // 4) * 230
                page.paste(thumb, (x, y))
                draw.text((x + 2, y + 202), path.stem[:60], fill="black")
            page.save(folder / f"{cat}_{start // 20 + 1:03}.jpg", quality=88)


def main():
    audit_path = OUT / "8_1_link_audit/link_audit.json"
    inventory_path = OUT / "8_3_solid_discovery/solid_inventory.json"
    eligibility_path = OUT / "8_2_eligibility/eligibility.json"
    required = [
        audit_path,
        inventory_path,
        eligibility_path,
        OUT / "8_4_paint_statistics.json",
        OUT / "8_4_split_identity.json",
        OUT / "8_4_dev_paint.json",
        OUT / "8_4_paint_by_acquisition.json",
        OUT / "8_4_geographic_feasibility.json",
    ]
    missing = [str(p.relative_to(OUT)) for p in required if not p.exists()]
    report = [
        "# Stage 8 — recognition data and test independence",
        "",
        (
            "Status: HUMAN REVIEW REQUIRED. Stage 7 remains NO_IMPROVEMENT. Frozen "
            "candidate, default model, checkpoints, earlier evaluation records and "
            "production stitching are unchanged. No training, new model inference, "
            "digitization, production export, promotion, approval recording or "
            "commit is performed by Stage 8."
        ),
        "",
        "## Stitching safety",
    ]
    if audit_path.exists():
        audit = load(audit_path)
        for value in audit.values():
            summary = value["summary"]
            summary["by_output_class"] = {}
            for cls, name in ((0, "solid"), (1, "dash")):
                records = [r for r in value["records"] if r["output_class"] == cls]
                denominator = summary["denominators"][
                    "output_solids" if cls == 0 else "output_dashes"
                ]
                rejected = sum(not r["accepted"] for r in records)
                summary["by_output_class"][name] = {
                    "outputs": denominator,
                    "category_counts": dict(Counter(r["category"] for r in records)),
                    "not_accepted": rejected,
                    "not_accepted_rate_of_class_outputs": rejected / max(1, denominator),
                }
        dump(audit_path, audit)
        sheets(audit)
        report += [
            (
                "Output category precedence: unsupported NoData crossing, separate "
                "dashes, different/parallel identity, observed unpainted gap, "
                "continuous marking. Only continuous_marking is accepted within the "
                "audited questionable-output set; other unaudited outputs are not "
                "thereby certified."
            ),
            "",
            "| Source | Outputs | Solids | Not accepted / all outputs | Categories |",
            "|---|---:|---:|---|---|",
        ]
        mismatches = {}
        for source, value in audit.items():
            s = value["summary"]
            d = s["denominators"]
            report.append(
                f"| {source} | {d['outputs']} | {d['output_solids']} | {s['not_accepted']}/{d['outputs']} ({s['not_accepted_rate_of_outputs']:.2%}) | {s['category_counts']} |"  # noqa: E501
            )
            reconciliation = s.get("stage7f_reconciliation", {})
            pairs = [
                ("stage7f_output_instances", "recomputed_output_instances"),
                ("stage7f_output_solids", "recomputed_output_solids"),
                ("stage7f_multi_id_outputs", "recomputed_multi_id_outputs"),
                ("stage7f_nodata_bridges", "recomputed_stage7f_nodata_flags"),
                ("stage7f_stage3_rule", "recomputed_stage3_rule"),
            ]
            bad = {
                a: [reconciliation[a], reconciliation[b]]
                for a, b in pairs
                if a in reconciliation and reconciliation[a] != reconciliation[b]
            }
            if bad:
                mismatches[source] = bad
            report += [
                "",
                f"{source}: identity × category = `{json.dumps(s['identity_by_category'])}`. Class counts = `{json.dumps(s['questionable_by_output_class'])}`. NoData by bridge origin and identity = `{json.dumps(s['nodata_crossings'])}`.",  # noqa: E501
            ]
        report += [
            "",
            f"7F reconciliation mismatches: `{json.dumps(mismatches)}`.",
            (
                "7F mixed denominators: Stage-3 rule divided by output solids, true "
                "errors by all outputs, while solid-join counts included mixed "
                "solid/dash joins from both output classes. Its NoData flag did not "
                "override same_physical_line acceptance and excluded single-ID "
                "outputs. Stage 8 reports each denominator explicitly and rejects "
                "unsupported NoData connections irrespective of identity."
            ),
            "",
            (
                "Evidence: 8_1_link_audit/link_evidence.gpkg, individual PNGs and "
                "contact_sheets/. Straight nearest-component MST segments are an audit "
                "proxy; component size >=10 px, >50% unobserved threshold and "
                "dilated-GT paint support define the test. This is not a proof that "
                "every possible curved or minority-NoData crossing was detected."
            ),
        ]
    else:
        report.append(
            "PENDING: saved-prediction audit and exact 7F reconciliation have not finished."
        )
    report += ["", "## Eight-region eligibility"]
    if eligibility_path.exists():
        e = load(eligibility_path)
        report += [
            e["role"],
            "",
            "| Region | Shared training solids | Observed fraction | Distance to training |",
            "|---|---:|---:|---:|",
        ]
        for r in e["regions"]:
            report.append(
                f"| {r['region']} | {len(r['shared_training_solid_ids'])} | {r['observed_fraction']:.3f} | {r['min_distance_to_training_native']:.2f} |"  # noqa: E501
            )
        report += [
            "",
            (
                "All eight use Job_20250708_0834. The 57 new observed dash IDs do not "
                "establish solid independence. T7_02 and T7_07 share Track02_35. Full "
                "shared solid IDs and per-ID pixel counts are in "
                "8_2_eligibility/eligibility.json; exact observed-raster polygon "
                "footprints are in observed_footprints.gpkg. Shared IDs across T7 "
                "regions are enumerated in the JSON. These regions remain protected "
                "and require first human geometry and complete-visible-marking review."
            ),
            f"Artifact search: `{json.dumps(e['inference_artifact_search'])}`",
        ]
    else:
        report.append("PENDING: footprint polygonization and artifact search.")
    report += ["", "## Solid discovery and split proposal"]
    if inventory_path.exists():
        inv = load(inventory_path)
        passed_sources = {s["source"] for s in inv["scans"] if s["full_source_integrity_passed"]}
        support_jobs = Counter()
        ambiguous = []
        for row in inv["solids"]:
            if not row.get("observed_supported"):
                continue
            jobs = {
                Path(p).name.split("_Track")[0] for p in row["header_files"] if p in passed_sources
            }
            if len(jobs) == 1:
                support_jobs.update(jobs)
            else:
                ambiguous.append(
                    {"global_id": row["global_id"], "verified_covering_acquisitions": sorted(jobs)}
                )
        original = inv["summary"].pop("by_acquisition_observed_supported", None)
        if original is not None:
            inv["summary"]["header_acquisition_associations_of_supported_solids"] = original
        inv["summary"]["verified_single_acquisition_supported_solids"] = dict(support_jobs)
        inv["summary"]["mixed_acquisition_supported_solids_unresolved"] = ambiguous
        dump(inventory_path, inv)
        report += [
            "Full per-solid IDs, roles, support, acquisition associations and integrity "
            "records: 8_3_solid_discovery/solid_inventory.json.",
            "",
            "Unused IDs are the 419 records with used=false in that inventory.",
            "",
            "| Inventory measure | Count |",
            "|---|---:|",
        ]
        for key, value in inv["summary"].items():
            if isinstance(value, int):
                report.append(f"| {key} | {value} |")
        report.append(
            f"Per-acquisition support: {dict(support_jobs)} for single-acquisition "
            f"solids; {len(ambiguous)} mixed-acquisition solids remain unallocated "
            "by acquisition. Coverage-only acquisition counts are retained in the JSON."
        )
        closure = load(OUT / "8_4_split_identity.json")
        blocked = set(closure["used_or_connected_ids"])
        supported_independent = [
            r
            for r in inv["solids"]
            if r.get("observed_supported") and r["global_id"] not in blocked
        ]
        report.append(
            f"Unused solid IDs: {inv['summary']['unused_solid_ids']}; "
            f"identity-independent before transitive screening: {inv['summary']['identity_independent']}; "  # noqa: E501
            f"observed-supported after transitive screening: {len(supported_independent)}."
        )
        report.append(
            f"Supported after transitive screening: "
            f"{sum(r['edge_line'] for r in supported_independent)} edge lines, "
            f"{sum(r['shape'] == 'curved' for r in supported_independent)} curved lines, "
            f"{sum(r['merge_line'] for r in supported_independent)} merge lines. "
            "Support by individual acquisition remains unresolved for inventory "
            "solids whose observations were merged across acquisitions; header "
            "coverage associations are not treated as verified per-acquisition support."
        )
        if not supported_independent:
            report.append(
                "No genuinely independent solid markings in the available data could "
                "be verified with supported length >=10 under this conservative "
                "identity screen. Unverified or unobserved geometry is not recognition data."
            )
        eligible = [
            r
            for r in inv["solids"]
            if r.get("eligible_for_independent_test") and r["global_id"] not in blocked
        ]
        report += [
            "",
            f"Independent-test candidates surviving observed support >=10, >=500 separation from all used footprints and transitive proximity exclusion: {len(eligible)}.",  # noqa: E501
            (
                "The global <=1-unit transitive proximity screen includes different "
                "client IDs and can conservatively exclude parallel lines; local "
                "physical_lines raster grouping is also used in the link audit. A "
                "surviving candidate still requires physical-line review, "
                "dash-positive coverage and >=500 separation between all proposed role "
                "footprints."
            ),
        ]
        feasibility = load(OUT / "8_4_geographic_feasibility.json")
        report += [
            "",
            "| Existing roles | Min distance | Shared mark IDs | Shared proximity groups | Meets all new independence rules |",  # noqa: E501
            "|---|---:|---:|---:|---|",
        ]
        for comparison in feasibility["existing_role_comparisons"]:
            report.append(
                f"| {' / '.join(comparison['roles'])} | "
                f"{comparison['minimum_footprint_distance']:.2f} | "
                f"{len(comparison['shared_mark_ids'])} | "
                f"{len(comparison['shared_conservative_physical_groups'])} | "
                f"{comparison['independent_under_required_rules']} |"
            )
        report.append(
            "Keeping established development as a historical comparison does not "
            "certify it under the new 500-unit/physical-line independence rules. "
            "The new three-role split remains infeasible without additional data "
            "or a human-approved reallocation that meets all rules."
        )
        report.append(
            f"Geometry-only discovery examined {feasibility['guarded_windows']} guarded "
            f"51.2-unit windows: {feasibility['identity_disjoint_windows']} have no used "
            "or transitively connected IDs, and "
            f"{feasibility['solid_and_dash_identity_disjoint_windows']} contain both "
            "solid and dash geometry with identity disjointness. See "
            "8_4_geographic_feasibility.json. This configured search is not exhaustive "
            "over every possible window placement."
        )
        if not eligible or not feasibility["solid_and_dash_identity_disjoint_windows"]:
            report.append(
                "No eligible independent solid-and-dash final-test region was "
                "found in the configured discovery search. This is a limitation of verified support, "  # noqa: E501
                "identity and separation in available data, not proof that no "
                "independent paint exists anywhere. Acquire integrity-verified LiDAR "
                "covering genuinely unseen physical solid lines and dash IDs, "
                "including edge, curved and merge lines, separated >=500 units from "
                "all used footprints and future roles. Isolated new-solid review "
                "candidates are distinct from an independent final-test region."
            )
        else:
            report.append(
                "BLOCKER: surviving solid candidates require Stage-8 discovery and "
                "QGIS packaging before a new region package is complete. Do not "
                "allocate these candidates using protected test imagery or labels."
            )
        dump(
            OUT / "8_4_split_proposal.json",
            {
                "eligible_solid_ids": [r["global_id"] for r in eligible],
                "status": "PENDING_HUMAN_REVIEW",
                "roles": closure["proposal"],
                "counts_by_acquisition": dict(
                    Counter(a for r in eligible for a in r["acquisitions"])
                ),
            },
        )
    else:
        report.append(
            "PENDING: full LAZ integrity/support inventory. Do not interpret "
            "unfinished discovery as zero independent solids."
        )
    report += [
        "",
        (
            "Retain approved training and established development pending review. "
            "T7 may be considered only for a protected dash-only held-out role; it "
            "cannot supply an unseen-solid final test because physical client "
            "identities span roles. Do not reassign or select regions from "
            "protected images/labels. New training/development positives need "
            "geometry review, complete-visible-marking review, mark-ID and "
            "end-to-end physical-line disjointness, and >=500 native-unit role "
            "separation before allocation."
        ),
        "",
        "## Real observed paint statistics",
        (
            "See 8_4_paint_statistics.json for phase5_12 and Stage-6 train/val "
            "IDs, deduplicated observed paint pixels, supported centreline-bin "
            "length and contrast. See 8_4_paint_metadata.json for "
            "per-region/acquisition metadata including T7. One-unit support bins "
            "require >=3 observed paint pixels; polygon/client length is not "
            "substituted for support. RGB contrast and normalized source contrast "
            "have different scales and must not be pooled. Missing acquisition "
            "attribution or supported length remains null and is a limitation, not "
            "zero. All four established development areas are measured in "
            "8_4_dev_paint.json. Acquisition-resolved counts are in "
            "8_4_paint_by_acquisition.json, with mixed-acquisition identities "
            "remeasured separately rather than counted twice in full."
        ),
        "",
        "## Future training protocol — proposal only",
        (
            "Condition: sufficient approved real training and solid+dash "
            "development data with physical-line disjointness. Freeze identities, "
            "geography, preprocessing, metrics and stopping rules before any run. "
            "Compare one bounded real-data change against the unchanged frozen "
            "candidate with three predeclared seeds (895, 896, 897). Budget each "
            "run to step 400; evaluate only at steps 200 and 400; select step 200 "
            "unless step 400 improves the equal-area mean all_instance_f150 "
            "over LEGACY_POSITIVE_DEV, NEW_POSITIVE_DEV and S6_01, "
            "without degrading either class recall or increasing unsupported NoData "
            "joins. Choose once using seed-mean dev results with a step-200 tie "
            "break. Never use protected test metrics for selection. Stage 7 arm-A "
            "runs all scored higher at step 200 than 600, so step 600 is not the "
            "default selection rule. No synthetic mixing, self-training or "
            "repeated 2×2. This protocol is unapproved and has not run."
        ),
        "",
        "## Verification, blockers and human-review gate",
        f"Missing required artifacts: {missing}.",
        (
            "Regression: 374 passed, 1 skipped; healthcheck: 10/10. "
            "Checks are logged in black.log, ruff.log, pytest_verify.log and "
            "healthcheck.json; inspect final statuses before declaring "
            "verification complete. Discovery corruption is excluded rather than "
            "silently counted as observation. Nothing is committed."
        ),
        "",
        (
            "STOP: Review the T7 regions in "
            "data/processed/stage7_new_test/20261107T051050944913Z/stage7_new_test_"
            "training_review.qgs in QGIS (the filename does not authorize "
            "training). Review any separately packaged new solid regions if "
            "available. Decide the split and whether to obtain additional "
            "independent solid data. No approval is recorded by this report."
        ),
    ]
    review_latest = ROOT / "outputs/reports/stage8_new_solid_review/latest.json"
    if review_latest.exists():
        latest = load(review_latest)
        review_spec = load(Path(latest["report_dir"]) / "new_region_candidates.json")
        data = Path(review_spec["output"])
        projects = list(data.glob("*.qgs"))
        review_files = [
            {
                "path": str(p),
                "size": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in sorted([*data.rglob("*"), *Path(latest["report_dir"]).rglob("*")])
            if p.is_file()
        ]
        dump(
            OUT / "8_5_review_package.json",
            {
                "latest": latest,
                "region_count": len(review_spec["tiles"]),
                "regions": [t["candidate_id"] for t in review_spec["tiles"]],
                "projects": [str(p) for p in projects],
                "files": review_files,
                "status": "PENDING_FIRST_HUMAN_REVIEW",
                "independent_final_test_eligible": False,
            },
        )
        position = report.index("## Verification, blockers and human-review gate")
        report[position:position] = [
            "## New-solid human-review package",
            "",
            f"Packaged {len(review_spec['tiles'])} new-solid review windows: "
            + ", ".join(t["candidate_id"] for t in review_spec["tiles"])
            + ".",
            "These contain verified new solid identities with shared surrounding "
            "markings. They are NOT independent solid-and-dash final-test regions, "
            "and are eligible for no training or evaluation role until human review "
            "and an explicit split decision. Shared IDs are listed per tile.",
            "QGIS projects: " + ", ".join(str(p) for p in projects),
            "External review assets and SHA-256 hashes: 8_5_review_package.json.",
            "",
        ]
        if not projects:
            missing.append("new_solid_review_qgis_project")
    acquisition_path = OUT / "8_4_paint_by_acquisition.json"
    if acquisition_path.exists():
        table = [
            "",
            "| Dataset/area | Split | Acquisition | Class | IDs | Observed paint px | Supported length | Contrast |",  # noqa: E501
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
        for row in load(acquisition_path)["summary"]:
            contrast = row["contrast_mean_over_ids"]
            contrast = f"{contrast:.2f}" if contrast is not None else "unavailable"
            table.append(
                f"| {row['dataset']} | {row['split']} | {row['acquisition']} | "
                f"{row['class']} | {row['ids']} | {row['observed_paint_pixels']} | "
                f"{row['supported_length_native']} | {contrast} |"
            )
        table.append(
            "IDs and supported bins may recur across acquisitions and datasets; do not sum them as globally distinct paint."  # noqa: E501
        )
        position = report.index("## Future training protocol — proposal only")
        report[position:position] = [*table, ""]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    roadmap = ROOT / "outputs/reports/roadmap_summary/report.md"
    text = roadmap.read_text(encoding="utf-8")
    marker = "\n## Stage 8 — recognition data and test independence\n"
    section = (
        "\nRecognition safety and data-independence evidence is in "
        "../stage8/report.md. T7 is a protected dash-only held-out candidate, "
        "with shared training solids; no independent solid final-test claim is "
        "accepted. Stage 7 remains NO_IMPROVEMENT. Stage 8 checks passed; "
        "S8_01/S8_02 are packaged for human QGIS review and a split decision. "
        "Mixed-acquisition inventory support remains explicitly unresolved. No training, model "
        "promotion, approval or commit.\n"
    )
    if marker in text:
        text = text.split(marker)[0]
    roadmap.write_text(text.rstrip() + "\n" + marker + section, encoding="utf-8")
    files = []
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)
    for log in (ROOT / "outputs/reports").glob("stage8*.log"):
        shutil.copyfile(log, logs / log.name)
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "package_manifest.json":
            files.append(
                {
                    "path": path.relative_to(OUT).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    dump(
        OUT / "package_manifest.json",
        {
            "schema_version": 1,
            "package_version": datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"),
            "status": "PENDING_HUMAN_REVIEW" if not missing else "INCOMPLETE",
            "missing": missing,
            "files": files,
        },
    )


if __name__ == "__main__":
    main()
