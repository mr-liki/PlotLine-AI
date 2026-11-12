"""Search metadata and validated support for new, role-reserved review windows."""

from collections import Counter
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import box

from phase5_11_common import ROOT, load
from roadmarking.rasterization.pipeline import dump
from stage9_common import context, identity_group, out_dir


def main():
    out = out_dir()
    ctx = context()
    used = ctx["used"]
    train = [r for r in used if r["role"].startswith("train")]
    dev = [r for r in used if "dev" in r["role"]]
    other = [r for r in used if not r["role"].startswith("train") and "dev" not in r["role"]]
    used_ids = set(ctx["extra"]).union(*(r["ids"] for r in used))
    used_groups = {identity_group(ctx, uid) for uid in used_ids}
    train_groups = {identity_group(ctx, uid) for r in train for uid in r["ids"]}
    dev_groups = {identity_group(ctx, uid) for r in dev for uid in r["ids"]}
    observed = [
        r
        for r in ctx["support"]["per_id_acquisition"]
        if r["supported_length"] >= 10 and identity_group(ctx, r["global_id"]) not in used_groups
    ]
    seed_ids = sorted({r["global_id"] for r in observed})
    raw = load(Path(ctx["source"]["report_directory"]) / "raw_integrity.json")
    passed = {
        s["source"]
        for s in load(ROOT / "outputs/reports/stage8/8_3_solid_discovery/solid_inventory.json")[
            "scans"
        ]
        if s["full_source_integrity_passed"]
    }
    sources = {r["path"]: r for r in raw if r["path"] in passed}
    tree = shapely.STRtree([f["geometry"] for f in ctx["features"]])
    rows, selected = [], []
    seen = set()
    for uid in seed_ids:
        feature = ctx["by_id"][uid]
        for d in np.arange(0, feature["centerline"].length, 25):
            point = feature["centerline"].interpolate(d)
            key = (round(point.x / 10), round(point.y / 10))
            if key in seen:
                continue
            seen.add(key)
            region = box(point.x - 25.6, point.y - 25.6, point.x + 25.6, point.y + 25.6)
            train_dist = min(region.distance(r["box"]) for r in train)
            dev_dist = min(region.distance(r["box"]) for r in dev)
            protected_dist = min(region.distance(r["box"]) for r in other)
            hits = [ctx["features"][int(i)] for i in tree.query(region, predicate="intersects")]
            groups = {identity_group(ctx, f["global_id"]) for f in hits if f["visual_id"] in (1, 2)}
            novel_solids = [
                f
                for f in hits
                if f["visual_id"] == 1 and identity_group(ctx, f["global_id"]) not in used_groups
            ]
            novel_dashes = [
                f for f in hits if f["visual_id"] == 2 and f["global_id"] not in used_ids
            ]
            covering = [
                r
                for r in sources.values()
                if r["mins"][0] <= point.x - 25.6
                and r["maxs"][0] >= point.x + 25.6
                and r["mins"][1] <= point.y - 25.6
                and r["maxs"][1] >= point.y + 25.6
            ]
            training_ok = (
                train_dist >= 1
                and protected_dist >= 500
                and dev_dist >= 500
                and not (groups & dev_groups)
                and not (groups & ctx["protected_groups"])
            )
            dev_ok = (
                dev_dist >= 1
                and protected_dist >= 500
                and train_dist >= 500
                and not (groups & train_groups)
                and not (groups & ctx["protected_groups"])
                and bool(novel_dashes)
            )
            dev_scoped = (
                protected_dist >= 500
                and train_dist >= 500
                and bool(novel_solids)
                and bool(novel_dashes)
            )
            row = {
                "opportunity_id": f"O9_{len(rows)+1:05d}",
                "seed_solid_id": uid,
                "bounds": list(region.bounds),
                "physical_groups": sorted(groups),
                "new_solid_ids": sorted(f["global_id"] for f in novel_solids),
                "new_solid_groups": sorted(
                    {identity_group(ctx, f["global_id"]) for f in novel_solids}
                ),
                "new_dash_ids": sorted(f["global_id"] for f in novel_dashes),
                "distance_to_training_native": train_dist,
                "distance_to_development_native": dev_dist,
                "distance_to_protected_and_historical_native": protected_dist,
                "shared_training_groups": sorted(groups & train_groups),
                "shared_development_groups": sorted(groups & dev_groups),
                "covering_integrity_passed_sources": [r["path"] for r in covering],
                "source_acquisitions": sorted(
                    {Path(r["path"]).name.split("_Track")[0] for r in covering}
                ),
                "training_role_candidate": training_ok and bool(covering),
                "development_role_candidate": dev_ok and bool(covering),
                "development_primary_only_with_context_exclusions": dev_scoped and bool(covering),
                "observed_seed_support_by_acquisition_whole_identity": [
                    r for r in observed if r["global_id"] == uid
                ],
                "actual_window_observed_paint_pixels": None,
                "eligible_nonoverlap_crops": None,
                "intensity_characteristics": "Not measured until source-backed raster verification; no faint/thin appearance claim from header coverage.",  # noqa: E501
                "human_approval": "FIRST_REVIEW_REQUIRED_IF_PACKAGED",
                "supervised_labels_created": False,
            }
            rows.append(row)
    # Reserve development first; select a small number of spatial/group-disjoint sections.
    options = sorted(
        [r for r in rows if r["development_role_candidate"]],
        key=lambda r: (
            "Job_20250924_1354" not in r["source_acquisitions"],
            "Job_20250708_1218" not in r["source_acquisitions"],
            -len(r["new_dash_ids"]),
            -len(r["new_solid_groups"]),
        ),
    )
    options += sorted(
        [r for r in rows if r["training_role_candidate"]],
        key=lambda r: (
            "Job_20250924_1354" not in r["source_acquisitions"],
            "Job_20250708_1218" not in r["source_acquisitions"],
            -len(r["new_solid_groups"]),
            -len(r["new_dash_ids"]),
        ),
    )
    for row in options:
        role = (
            "development_candidate" if row["development_role_candidate"] else "training_candidate"
        )
        region = box(*row["bounds"])
        if any(region.distance(box(*s["bounds"])) < 500 for s in selected):
            continue
        if any(set(row["new_solid_groups"]) & set(s["new_solid_groups"]) for s in selected):
            continue
        if any(
            set(row["physical_groups"]) & set(s["physical_groups"])
            for s in selected
            if s["proposed_role"] != role
        ):
            continue
        selected.append({**row, "proposed_role": role})
        if len(selected) >= 4:
            break
    summary = {
        "metadata_windows_considered": len(rows),
        "training_candidates": sum(r["training_role_candidate"] for r in rows),
        "strict_development_candidates": sum(r["development_role_candidate"] for r in rows),
        "development_with_context_exclusions": sum(
            r["development_primary_only_with_context_exclusions"] for r in rows
        ),
        "selected_for_source_backed_review_verification": len(selected),
        "selected_roles": dict(Counter(r["proposed_role"] for r in selected)),
    }
    dump(
        out / "opportunity_matrix.json",
        {
            "summary": summary,
            "opportunities": rows,
            "selected": selected,
            "rules": "51.2-unit metadata windows seeded by observed-supported, unused physical solid groups. Maintain >=500 native-unit separation from other role footprints and protected/historical regions; full window group exclusions for strict role candidates. Training may adjoin current training. No target labels or model outputs generated. Actual window support and crop counts remain unverified until rendering. Each road section is the sampling unit, not each overlapping crop.",  # noqa: E501
        },
    )
    print(summary)


if __name__ == "__main__":
    main()
