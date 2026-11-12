"""Stage 6 review package: integrity checks, contact sheet, missed-paint audit, checklist."""

from pathlib import Path

from shapely.geometry import box

import phase5_11_unannotated_audit
from phase5_11_common import ROOT, load, reserved_geography
from phase5_11_report import gallery
from phase5_12_report import verify
from roadmarking.rasterization.pipeline import dump


def main():
    latest = load(ROOT / "outputs/reports/stage6_review/latest.json")
    out = Path(latest["report_dir"])
    spec = load(out / "new_region_candidates.json")
    regions, extra, _ = reserved_geography()
    reserved_ids = set(extra)
    for r in regions:
        if not r["role"].startswith("train"):
            reserved_ids.update(r["ids"])
    for t in spec["tiles"]:
        t["shared_ids_with_dev_benchmark_or_test"] = sorted(
            set(t["source_global_ids"]) & reserved_ids
        )
        t["minimum_distance_by_role_native"] = {}
        for r in regions:
            d = box(*t["bounds"]).distance(r["box"])
            t["minimum_distance_by_role_native"][r["role"]] = min(
                d, t["minimum_distance_by_role_native"].get(r["role"], d)
            )
    used = {t["source"] for t in spec["tiles"]}
    rejected = sorted(
        {
            Path(x["source"]).name
            for x in spec["source_integrity"]
            if not x["full_source_integrity_passed"]
        }
    )
    spec["source_integrity"] = [x for x in spec["source_integrity"] if x["source"] in used]
    checks = verify(spec)
    checks["rejected_corrupt_sources_not_used"] = rejected
    dump(out / "geometry_integrity.json", checks)
    assert all(
        v
        for k, v in checks.items()
        if k not in ("worst_relative_outline_difference", "rejected_corrupt_sources_not_used")
    ), checks
    (out / "galleries").mkdir(exist_ok=True)
    gallery(spec, out / "galleries" / "stage6_contact_sheet.png", title="Stage 6 diverse regions")
    phase5_11_unannotated_audit.main("stage6_review")
    flags = load(out / "unannotated_paint_audit.json")["candidates"]
    qgis = load(out / "qgis_review_package.json")
    lines = [
        "# Stage 6 — review checklist (10 diversity-ranked regions)",
        "",
        f"QGIS project: `{qgis['project']}`, with bookmarks S6_01 to S6_10. Layers are as "
        "before: red solid, blue dash, magenta unsupported, yellow footprint. The client layers "
        "are read-only, and the two reviewer layers can be edited.",
        "",
        "For each region, reply **APPROVE**, **REJECT** or **APPROVE WITH NOTES**. Approving "
        "confirms (1) the outlines and classes are correct and (2) every visible target "
        "marking in the observed area is represented. Draw any missed paint in the "
        "missing-paint layer, and it will be excluded as an ignore zone.",
        "",
        f"**{spec['fresh_development_candidate']} is proposed as a FRESH DEVELOPMENT region.** "
        "It is the most distant from all training, from a different acquisition, and never "
        "used for any selection. Please confirm whether it should be held out as development "
        "(recommended) or used for training.",
        "",
        "| Region | Priority | Acquisition / source | Observed solid/dash | New solid/dash | "
        "Coverage | Distance to training | Proposed role | Audit flags | Reply |",
        "|---|---|---|---|---|---:|---:|---|---:|---|",
    ]
    for t in spec["tiles"]:
        n_flags = sum(f["region"] == t["candidate_id"] for f in flags)
        lines.append(
            f"| {t['candidate_id']} | {t['priority_tag']} | {t['acquisition']} / "
            f"{Path(t['source']).name} | {t['observed_solid_ids']}/{t['observed_dash_ids']} | "
            f"{t['new_observed_solid_ids']}/{t['new_observed_dash_ids']} | "
            f"{t['observed_fraction']:.0%} | {t['min_distance_to_training_native']:.0f} | "
            f"{t['proposed_role']} | {n_flags} | |"
        )
    lines += ["", "Unoutlined bright returns worth a look (paint, kerb, barrier or object):", ""]
    lines += [
        f"- {f['region']} at ({f['centre_native'][0]:.1f}, {f['centre_native'][1]:.1f}): "
        f"{f['pixels']} px, elongation {f['elongation']:.1f}"
        for f in flags
    ]
    (out / "review_checklist.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    dump(
        out / "report.json",
        {
            "stage": 6,
            "status": "REAL_DATA_REVIEW_REQUIRED",
            "checks": checks,
            "qgis": qgis,
            "new_unique_observed_identities": spec["new_unique_observed_identities"],
            "fresh_development_candidate": spec["fresh_development_candidate"],
            "candidates_considered": spec["candidates_considered"],
        },
    )
    print(checks)


if __name__ == "__main__":
    main()
