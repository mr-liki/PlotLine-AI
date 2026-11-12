"""Role/identity checks only; no test prediction, checkpoint or metric selection."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import shapely
from affine import Affine
from rasterio.features import rasterize
from shapely.ops import nearest_points

from phase5_11_common import ROOT, client_features, load
from roadmarking.evaluation.link_audit import physical_lines
from roadmarking.rasterization.pipeline import dump
from stage8_solid_discovery import used_geography

OUT = ROOT / "outputs/reports/stage8b"


def main():
    features = client_features(check_network=False)[0]
    solids = [f for f in features if f["visual_id"] == 1]
    parent = list(range(len(solids)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tree = shapely.STRtree([f["geometry"] for f in solids])
    edges = []
    for i, a in enumerate(solids):
        for j in tree.query(a["geometry"], predicate="dwithin", distance=1.0):
            j = int(j)
            if j <= i:
                continue
            b = solids[j]
            pa, pb = nearest_points(a["geometry"], b["geometry"])
            x, y = (pa.x + pb.x) / 2, (pa.y + pb.y) / 2
            affine = Affine(0.05, 0, x - 2, 0, -0.05, y + 2)
            masks = np.stack(
                [
                    rasterize(
                        [(f["geometry"], 1)], out_shape=(80, 80), transform=affine, dtype="uint8"
                    ).astype(bool)
                    for f in (a, b)
                ]
            )
            groups = physical_lines(masks, np.array([0, 0]))
            continuation = groups[0] == groups[1]
            if continuation:
                parent[find(j)] = find(i)
            edges.append(
                {
                    "ids": [a["global_id"], b["global_id"]],
                    "local_physical_lines_continuation": continuation,
                    "native_distance": a["geometry"].distance(b["geometry"]),
                }
            )
    groups = defaultdict(list)
    for i, f in enumerate(solids):
        groups[find(i)].append(f["global_id"])
    group_map = {uid: key for key, ids in groups.items() for uid in ids}
    used, extra = used_geography()
    prior = [r for r in used if not r["id"].startswith("T7_")]
    prior_ids = extra.union(*(r["ids"] for r in prior))
    prior_groups = {group_map[u] for u in prior_ids if u in group_map}
    t7_spec = load(
        ROOT / "outputs/reports/stage7_new_test/20261107T051050944913Z/new_region_candidates.json"
    )
    s8_spec = load(
        Path(load(ROOT / "outputs/reports/stage8_new_solid_review/latest.json")["report_dir"])
        / "new_region_candidates.json"
    )
    t7_ids = {p["global_id"] for t in t7_spec["tiles"] for p in t["observed_ids"]}
    t7_groups = {group_map[u] for u in t7_ids if u in group_map}
    populations = []
    for cohort, spec in (("T7", t7_spec), ("S8", s8_spec)):
        for t in spec["tiles"]:
            for p in t["observed_ids"]:
                uid = p["global_id"]
                novel_id = uid not in prior_ids
                novel_physical = p["class"] != "solid" or group_map[uid] not in prior_groups
                disjoint_t7 = cohort == "T7" or (
                    uid not in t7_ids and (p["class"] != "solid" or group_map[uid] not in t7_groups)
                )
                populations.append(
                    {
                        "cohort": cohort,
                        "region": t["candidate_id"],
                        **p,
                        "unseen_id_against_prior_roles": novel_id,
                        "unseen_solid_physical_group_against_prior_roles": novel_physical,
                        "disjoint_from_T7": disjoint_t7,
                        "population_candidate": (
                            "dash_focused"
                            if cohort == "T7" and p["class"] == "dash"
                            else (
                                "unseen_solid"
                                if cohort == "S8" and p["class"] == "solid"
                                else "out_of_primary_population"
                            )
                        ),
                        "identity_eligible": novel_id and novel_physical and disjoint_t7,
                    }
                )
    populations.append(
        {
            "cohort": "T7",
            "region": "T7_05",
            "global_id": "stage8b_T7_05_missing_dash_001",
            "class": "dash",
            "observed_pixels": 248,
            "unseen_id_against_prior_roles": True,
            "unseen_solid_physical_group_against_prior_roles": True,
            "disjoint_from_T7": True,
            "population_candidate": "dash_focused",
            "identity_eligible": True,
            "identity_basis": "New local observed-paint annotation, no intersecting client feature; no client ID invented.",  # noqa: E501
        }
    )
    windows = [("T7", t) for t in t7_spec["tiles"]] + [("S8", t) for t in s8_spec["tiles"]]
    distances = []
    for _cohort, tile in windows:
        geom = shapely.box(*tile["bounds"])
        distance = min(geom.distance(r["box"]) for r in prior)
        distances.append(
            {
                "region": tile["candidate_id"],
                "distance_to_prior_role_footprints": distance,
                "guard_ge_500": distance >= 500,
            }
        )
    pairs = []
    for i, (ca, a) in enumerate(windows):
        for cb, b in windows[i + 1 :]:
            distance = shapely.box(*a["bounds"]).distance(shapely.box(*b["bounds"]))
            shared = sorted(set(a["source_global_ids"]) & set(b["source_global_ids"]))
            pairs.append(
                {
                    "regions": [a["candidate_id"], b["candidate_id"]],
                    "cohorts": [ca, cb],
                    "distance": distance,
                    "guard_ge_500": distance >= 500,
                    "shared_mark_ids": shared,
                }
            )
    for population in populations:
        uid = population["global_id"]
        population["physical_solid_group"] = (
            min(groups[group_map[uid]]) if uid in group_map else None
        )
    role_sets = {
        "training": [r for r in prior if r["role"].startswith("train")],
        "development": [r for r in prior if "dev" in r["role"]],
    }
    training_ids = set().union(*(r["ids"] for r in role_sets["training"]))
    development_ids = set().union(*(r["ids"] for r in role_sets["development"]))
    training_groups = {group_map[u] for u in training_ids if u in group_map}
    development_groups = {group_map[u] for u in development_ids if u in group_map}
    train_dev = {
        "minimum_native_distance": min(
            a["box"].distance(b["box"])
            for a in role_sets["training"]
            for b in role_sets["development"]
        ),
        "shared_mark_ids": sorted(training_ids & development_ids),
        "shared_physical_solid_groups": [
            groups[key] for key in sorted(training_groups & development_groups)
        ],
    }
    summary = {}
    for cohort, primary in (("T7", "dash_focused"), ("S8", "unseen_solid")):
        eligible = [
            p
            for p in populations
            if p["cohort"] == cohort
            and p["population_candidate"] == primary
            and p["identity_eligible"]
        ]
        summary[cohort] = {
            "population": primary,
            "eligible_unique_ids": len({p["global_id"] for p in eligible}),
            "eligible_physical_solid_groups": len(
                {
                    p["physical_solid_group"]
                    for p in eligible
                    if p["physical_solid_group"] is not None
                }
            ),
            "eligible_id_appearances": len(eligible),
            "excluded_shared_or_out_of_scope_solid_ids": sorted(
                {
                    p["global_id"]
                    for p in populations
                    if p["cohort"] == cohort
                    and p["class"] == "solid"
                    and (cohort == "T7" or not p["identity_eligible"])
                }
            ),
        }
    dump(
        OUT / "independence_audit.json",
        {
            "method": "Global client solid pairs within 1 native unit; 4x4-unit local 0.05-grid rasterization and unchanged link_audit.physical_lines continuation rule; transitive grouping across client IDs. Geometry/identity metadata only; test rasters and labels are not used for selection. This is a documented identity screen, not human certification of every cross-ID physical continuation.",  # noqa: E501
            "physical_groups": list(groups.values()),
            "pair_audit": edges,
            "populations": populations,
            "summary": summary,
            "footprint_separation": distances,
            "candidate_pair_separation": pairs,
            "policy": "T7 is dash-focused; shared solids ignored for unseen-solid metrics and never treated as negative background. S8 is a candidate cohort for identified unseen solid instances only; shared-context markings must be excluded from that population. Do not pool into a fully independent benchmark or run inference before explicit test-use authorization.",  # noqa: E501
            "overall_decision": "INDEPENDENCE_RESTRICTIONS_REMAIN",
        },
    )
    result = load(OUT / "independence_audit.json")
    result["training_development_comparison"] = train_dev
    dump(OUT / "independence_audit.json", result)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
