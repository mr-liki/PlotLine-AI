("Geometry-only discovery feasibility; no protected image/label files " "are opened.")

import numpy as np
import shapely
from shapely.geometry import box

from phase5_11_common import ROOT, client_features, load
from roadmarking.rasterization.pipeline import dump
from stage8_solid_discovery import used_geography


def main():
    features = client_features(check_network=False)[0]
    used, extra = used_geography()
    blocked = set(
        load(ROOT / "outputs/reports/stage8/8_4_split_identity.json")["used_or_connected_ids"]
    )
    used_ids = extra.union(*(r["ids"] for r in used)) | blocked
    tree = shapely.STRtree([f["geometry"] for f in features])
    candidates = []
    for f in features:
        if f["visual_id"] != 1 or f["global_id"] in used_ids:
            continue
        if f["centerline"].length < 10:
            continue
        for d in np.arange(0, f["centerline"].length + 1, 25):
            p = f["centerline"].interpolate(d)
            window = box(p.x - 25.6, p.y - 25.6, p.x + 25.6, p.y + 25.6)
            distance = min(window.distance(r["box"]) for r in used)
            if distance < 500:
                continue
            hits = [features[int(j)] for j in tree.query(window, predicate="intersects")]
            shared = sorted({h["global_id"] for h in hits} & used_ids)
            candidates.append(
                {
                    "seed_solid_id": f["global_id"],
                    "bounds": list(window.bounds),
                    "distance_to_all_used_footprints": distance,
                    "solid_ids": sorted(h["global_id"] for h in hits if h["visual_id"] == 1),
                    "dash_ids": sorted(h["global_id"] for h in hits if h["visual_id"] == 2),
                    "used_or_connected_ids": shared,
                    "identity_disjoint": not shared,
                    "solid_and_dash_geometry": any(h["visual_id"] == 2 for h in hits),
                }
            )
    viable = [c for c in candidates if c["identity_disjoint"] and c["solid_and_dash_geometry"]]
    role_groups = {
        "training": [r for r in used if r["role"].startswith("train")],
        "development": [r for r in used if "dev" in r["role"]],
        "T7_protected_candidate": [r for r in used if r["id"].startswith("T7_")],
    }
    group_map = {
        uid: k
        for k, ids in enumerate(
            load(ROOT / "outputs/reports/stage8/8_4_split_identity.json")["groups"]
        )
        for uid in ids
    }
    comparisons = []
    for left, right in (
        ("training", "development"),
        ("training", "T7_protected_candidate"),
        ("development", "T7_protected_candidate"),
    ):
        a, b = role_groups[left], role_groups[right]
        aid = set().union(*(r["ids"] for r in a))
        bid = set().union(*(r["ids"] for r in b))
        ag = {group_map[u] for u in aid if u in group_map}
        bg = {group_map[u] for u in bid if u in group_map}
        minimum = min(x["box"].distance(y["box"]) for x in a for y in b)
        comparisons.append(
            {
                "roles": [left, right],
                "minimum_footprint_distance": minimum,
                "separation_ge_500": minimum >= 500,
                "shared_mark_ids": sorted(aid & bid),
                "shared_conservative_physical_groups": sorted(ag & bg),
                "independent_under_required_rules": minimum >= 500
                and not (aid & bid)
                and not (ag & bg),
            }
        )
    dump(
        ROOT / "outputs/reports/stage8/8_4_geographic_feasibility.json",
        {
            "method": (
                "51.2-native-unit windows, centres every 25 units along unused solid "
                "client centrelines; >=500 units from every used role including "
                "protected T7 footprints. Complete geometry intersection must have no "
                "used/proximity-connected IDs. Uses client geometry/role metadata, no "
                "protected raster images, target arrays, model outputs or metrics. Not "
                "an exhaustive search over all possible window placements."
            ),
            "guarded_windows": len(candidates),
            "identity_disjoint_windows": sum(c["identity_disjoint"] for c in candidates),
            "solid_and_dash_identity_disjoint_windows": len(viable),
            "windows": candidates,
            "existing_role_comparisons": comparisons,
            "status": "geometry_only_support_and_human_review_required",
        },
    )
    print(f"{len(candidates)} guarded windows; {len(viable)} solid+dash identity-disjoint windows")


if __name__ == "__main__":
    main()
