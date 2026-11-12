("Conservative transitive client-line identity audit, independent of " "image selection.")

from collections import defaultdict

import shapely

from phase5_11_common import ROOT, client_features
from roadmarking.rasterization.pipeline import dump
from stage8_solid_discovery import used_geography


def main():
    solids = [f for f in client_features(check_network=False)[0] if f["visual_id"] == 1]
    used, extra = used_geography()
    used_ids = extra.union(*(r["ids"] for r in used))
    parent = list(range(len(solids)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tree = shapely.STRtree([f["geometry"] for f in solids])
    edges = []
    for i, f in enumerate(solids):
        for j in tree.query(f["geometry"], predicate="dwithin", distance=1.0):
            j = int(j)
            if j <= i:
                continue
            # Conservative exclusion also catches overlapping/parallel paint within 1 unit.
            # These groups are an exclusion screen, not an assertion of identical paint.
            parent[find(j)] = find(i)
            edges.append([f["global_id"], solids[j]["global_id"]])
    groups = defaultdict(list)
    for i, f in enumerate(solids):
        groups[find(i)].append(f["global_id"])
    blocked = set()
    for ids in groups.values():
        if used_ids.intersection(ids):
            blocked.update(ids)
    roles = {r["id"]: {"role": r["role"], "ids": sorted(r["ids"])} for r in used}
    dump(
        ROOT / "outputs/reports/stage8/8_4_split_identity.json",
        {
            "method": (
                "Global client painted geometry proximity <=1 native unit, transitive "
                "closure including used and unused IDs. Conservative exclusion screen; "
                "local raster physical_lines is used in the connection audit. This "
                "screen can exclude nearby parallel lines and cannot establish an "
                "absence of physically independent paint."
            ),
            "groups": list(groups.values()),
            "edges": edges,
            "used_or_connected_ids": sorted(blocked),
            "independence_screen_survivors": sorted(
                f["global_id"] for f in solids if f["global_id"] not in blocked
            ),
            "roles": roles,
            "proposal": {
                "training": (
                    "Existing approved training only; new regions require human geometry "
                    "and complete-visible review."
                ),
                "development": (
                    "Retain S6_01 and existing development; require additional approved "
                    "solid+dash positives disjoint from future training at physical-line "
                    "level."
                ),
                "protected_final_test": (
                    "T7 remains protected dash-only candidate; independent solid+dash test "
                    "requires unseen physical solid lines and unseen dash IDs, >=500 "
                    "native units from every other role."
                ),
                "status": (
                    "No allocation authorized; combine support inventory with this "
                    "transitive exclusion and human review before proposing coordinates."
                ),
            },
        },
    )
    print(
        f"{len(solids)} solids; {len(groups)} proximity groups; {len(blocked)} used or transitively connected"  # noqa: E501
    )


if __name__ == "__main__":
    main()
