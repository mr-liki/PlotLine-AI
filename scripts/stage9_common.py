"""Stage 9 metadata helpers; never read predictions or change dataset roles."""

from datetime import UTC, datetime
from pathlib import Path

from phase5_11_common import ROOT, client_features, load, reserved_geography
from roadmarking.rasterization.pipeline import dump
from stage8_solid_discovery import used_geography

BASE = ROOT / "outputs/reports/stage9"


def prepare():
    BASE.mkdir(parents=True, exist_ok=True)
    previous = load(BASE / "latest.json")["report_dir"] if (BASE / "latest.json").exists() else None
    run = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = BASE / run
    out.mkdir()
    dump(
        BASE / "latest.json",
        {
            "run_id": run,
            "report_dir": str(out),
            "status": "IN_PROGRESS",
            "previous_report_dir": previous,
        },
    )
    return out


def out_dir():
    return Path(load(BASE / "latest.json")["report_dir"])


def context():
    features, layers, source = client_features(check_network=False)
    by_id = {f["global_id"]: f for f in features}
    audit = load(ROOT / "outputs/reports/stage8b/independence_audit.json")
    groups = {u: min(g) for g in audit["physical_groups"] for u in g}
    used, extra = used_geography()
    mapping = load(ROOT / "outputs/reports/stage8/8_4_paint_by_acquisition.json")[
        "tile_acquisition_mapping"
    ]
    for region in reserved_geography()[0]:
        if region.get("source"):
            mapping[region["id"]] = Path(region["source"]).name.split("_Track")[0]
    from shapely.geometry import box

    s8 = load(
        Path(load(ROOT / "outputs/reports/stage8_new_solid_review/latest.json")["report_dir"])
        / "new_region_candidates.json"
    )
    for tile in s8["tiles"]:
        used.append(
            {
                "id": tile["candidate_id"],
                "role": "protected_S8",
                "ids": set(tile["source_global_ids"]),
                "box": box(*tile["bounds"]),
            }
        )
        mapping[tile["candidate_id"]] = tile["acquisition"]
    for row in used:
        if row["id"].startswith("T7_"):
            row["role"] = "protected_T7"
        if row["id"] == "NEW_FINAL_TEST_RESERVED":
            row["role"] = "historical_consumed_test"
        row["acquisition"] = mapping.get(row["id"], "NOT_IN_REGION_METADATA")
    support = load(ROOT / "outputs/reports/stage8b/acquisition_support/resolved_inventory.json")
    protected = load(ROOT / "outputs/reports/stage8b/evaluation_eligibility.json")
    protected_ids = set(protected["populations"]["T7"]["member_ids"]) | set(
        protected["populations"]["S8"]["member_ids"]
    )
    protected_groups = {groups[u] for u in protected_ids if u in groups}
    return {
        "features": features,
        "by_id": by_id,
        "layers": layers,
        "source": source,
        "audit": audit,
        "groups": groups,
        "used": used,
        "extra": extra,
        "support": support,
        "protected": protected,
        "protected_ids": protected_ids,
        "protected_groups": protected_groups,
    }


def identity_group(ctx, uid):
    return ctx["groups"].get(uid, uid)
