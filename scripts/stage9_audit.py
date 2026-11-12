"""Explain the three Stage-8B group overlaps and all role-level dependencies."""

import hashlib
from collections import Counter, defaultdict, deque
from itertools import combinations, pairwise
from pathlib import Path

import matplotlib
import numpy as np
from shapely.geometry import Point
from shapely.ops import nearest_points

from phase5_11_common import ROOT, load
from roadmarking.rasterization.pipeline import dump
from stage9_common import context, identity_group, prepare

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def endpoints(line):
    parts = list(line.geoms) if line.geom_type == "MultiLineString" else [line]
    result = []
    for part in parts:
        for end in (False, True):
            point = Point(part.coords[-1 if end else 0])
            inner = part.interpolate(max(0, part.length - 1) if end else min(1, part.length))
            vector = np.array([inner.x - point.x, inner.y - point.y])
            vector /= max(1e-9, np.linalg.norm(vector))
            result.append((point, vector))
    return result


def edge_evidence(a, b):
    ea, eb = endpoints(a["centerline"]), endpoints(b["centerline"])
    pa, ua, pb, ub = min(
        ((p, u, q, v) for p, u in ea for q, v in eb), key=lambda pair: pair[0].distance(pair[2])
    )
    gap = pa.distance(pb)
    angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(ua, ub)), 0, 1))))
    overlap = a["geometry"].intersection(b["geometry"]).area / max(
        1e-9, min(a["geometry"].area, b["geometry"].area)
    )
    same_direction = float(np.dot(ua, ub))
    intersection = a["centerline"].intersection(b["centerline"]).length
    if overlap >= 0.8:
        kind = "coincident_or_duplicate_representation"
    elif gap <= 1.0 and angle <= 10 and same_direction < -0.9:
        kind = "end_to_end_continuation_supported"
    elif intersection > 1.0:
        kind = "overlapping_same_line_segment_supported"
    else:
        kind = "ambiguous_local_join_requires_qualification"
    return {
        "endpoint_gap_native": gap,
        "endpoint_angle_degrees": angle,
        "endpoint_inward_direction_dot": same_direction,
        "paint_overlap_fraction_of_smaller": overlap,
        "centreline_intersection_length_native": intersection,
        "endpoint_coordinates": [[pa.x, pa.y], [pb.x, pb.y]],
        "classification": kind,
    }


def shortest_path(graph, start, end):
    queue = deque([(start, [start])])
    seen = {start}
    while queue:
        node, path = queue.popleft()
        if node == end:
            return path
        for neighbor in sorted(graph[node]):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, [*path, neighbor]))
    return []


def distinct_pair_evidence(a, b):
    pa, pb = nearest_points(a["centerline"], b["centerline"])

    def tangent(line, point):
        parts = list(line.geoms) if line.geom_type == "MultiLineString" else [line]
        part = min(parts, key=lambda p: p.distance(point))
        along = part.project(point)
        p, q = part.interpolate(max(0, along - 1)), part.interpolate(min(part.length, along + 1))
        vector = np.array([q.x - p.x, q.y - p.y])
        return vector / max(1e-9, np.linalg.norm(vector))

    ua, ub = tangent(a["centerline"], pa), tangent(b["centerline"], pb)
    angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(ua, ub)), 0, 1))))
    disp = np.array([pb.x - pa.x, pb.y - pa.y])
    lateral = abs(float(disp[0] * ua[1] - disp[1] * ua[0]))
    parallel = angle <= 10 and lateral > 0.2
    return {
        "classification": (
            "parallel_physically_distinct"
            if parallel
            else "close_distinct_or_unresolved_not_continuation"
        ),
        "local_tangent_angle_degrees": angle,
        "local_lateral_separation_native": lateral,
        "closest_centreline_coordinates": [[pa.x, pa.y], [pb.x, pb.y]],
    }


def main():
    out = prepare()
    ctx = context()
    used = ctx["used"]
    train = [r for r in used if r["role"].startswith("train")]
    dev = [r for r in used if "dev" in r["role"]]
    train_ids, dev_ids = set().union(*(r["ids"] for r in train)), set().union(
        *(r["ids"] for r in dev)
    )
    graph = defaultdict(set)
    edges = []
    for row in ctx["audit"]["pair_audit"]:
        a, b = row["ids"]
        evidence = (
            edge_evidence(ctx["by_id"][a], ctx["by_id"][b])
            if row["local_physical_lines_continuation"]
            else distinct_pair_evidence(ctx["by_id"][a], ctx["by_id"][b])
        )
        if row["local_physical_lines_continuation"]:
            graph[a].add(b)
            graph[b].add(a)
        edges.append({**row, **evidence})
    edge_map = {tuple(sorted(row["ids"])): row for row in edges}
    details = []
    maps = out / "continuation_evidence"
    maps.mkdir()
    for number, ids in enumerate(
        ctx["audit"]["training_development_comparison"]["shared_physical_solid_groups"], 1
    ):
        training, development = sorted(set(ids) & train_ids), sorted(set(ids) & dev_ids)
        paths = []
        for a in training:
            for b in development:
                path = shortest_path(graph, a, b)
                paths.append(
                    {
                        "training_id": a,
                        "development_id": b,
                        "path": path,
                        "edges": [edge_map[tuple(sorted((x, y)))] for x, y in pairwise(path)],
                        "direct_id_leakage": a == b,
                    }
                )
        members = []
        for uid in ids:
            f = ctx["by_id"][uid]
            members.append(
                {
                    "global_id": uid,
                    "source_identity": f["source_identity"],
                    "layer": f["layer"],
                    "designation": f["designation"],
                    "client_length_native": f["centerline"].length,
                    "regions": [
                        {"region": r["id"], "role": r["role"], "acquisition": r["acquisition"]}
                        for r in used
                        if uid in r["ids"]
                    ],
                    "observations_by_acquisition": [
                        r for r in ctx["support"]["per_id_acquisition"] if r["global_id"] == uid
                    ],
                }
            )
        group_edges = [
            e
            for e in edges
            if e["local_physical_lines_continuation"] and set(e["ids"]).issubset(ids)
        ]
        strong_graph = defaultdict(set)
        for edge in group_edges:
            if edge["classification"] != "ambiguous_local_join_requires_qualification":
                a, b = edge["ids"]
                strong_graph[a].add(b)
                strong_graph[b].add(a)
        supported_paths = [
            {"training_id": a, "development_id": b, "path": shortest_path(strong_graph, a, b)}
            for a in training
            for b in development
        ]
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        for uid in ids:
            f = ctx["by_id"][uid]
            color = "tab:red" if uid in train_ids else "tab:blue" if uid in dev_ids else "grey"
            parts = (
                list(f["centerline"].geoms)
                if f["centerline"].geom_type == "MultiLineString"
                else [f["centerline"]]
            )
            for part in parts:
                xy = np.array(part.coords)
                axes[0].plot(xy[:, 0], xy[:, 1], color=color, alpha=0.65)
        for r in train + dev:
            if not (r["ids"] & set(ids)):
                continue
            x, y = r["box"].exterior.xy
            axes[0].plot(x, y, color="red" if r in train else "blue", linewidth=2)
            axes[0].text(r["box"].centroid.x, r["box"].centroid.y, r["id"], fontsize=6)
        chosen = min(paths, key=lambda p: len(p["path"]))
        for i, uid in enumerate(chosen["path"]):
            axes[1].scatter(
                i, 0, color="red" if uid in train_ids else "blue" if uid in dev_ids else "grey"
            )
            axes[1].text(i, 0.05, uid[:12], rotation=60, fontsize=7)
            if i:
                axes[1].plot([i - 1, i], [0, 0], color="black")
        axes[1].set_ylim(-0.2, 0.9)
        axes[1].set_title("Shortest training-to-development ID path; full IDs in JSON")
        axes[0].set_title("Authoritative client centreline evidence; red=train, blue=dev")
        axes[0].set_aspect("equal", adjustable="datalim")
        fig.suptitle(
            f"Stage 8B overlap group {number}: {len(ids)} client IDs; native CRS unconfirmed"
        )
        fig.tight_layout()
        figure = maps / f"group_{number}.png"
        fig.savefig(figure, dpi=140)
        plt.close(fig)
        details.append(
            {
                "reported_group": number,
                "group_id": min(ids),
                "member_ids": ids,
                "training_ids": training,
                "development_ids": development,
                "members": members,
                "paths": paths,
                "paths_using_only_supported_edges": supported_paths,
                "group_edges": group_edges,
                "evidence_classification_counts": dict(
                    Counter(e["classification"] for e in group_edges)
                ),
                "qualified_as_dependency": True,
                "policy": "Retain exclusion under original Stage-8B graph; ambiguous connections are disclosed, not used to relax approved test identities.",  # noqa: E501
                "figure": str(figure),
            }
        )
    region_pairs = []
    for a, b in combinations(used, 2):
        if a["role"] == b["role"]:
            continue
        ga = {identity_group(ctx, uid) for uid in a["ids"]}
        gb = {identity_group(ctx, uid) for uid in b["ids"]}
        region_pairs.append(
            {
                "regions": [a["id"], b["id"]],
                "roles": [a["role"], b["role"]],
                "shared_feature_ids": sorted(a["ids"] & b["ids"]),
                "shared_physical_groups": sorted(ga & gb),
                "same_acquisition": (
                    a["acquisition"] == b["acquisition"]
                    if "NOT_IN_REGION_METADATA" not in (a["acquisition"], b["acquisition"])
                    else None
                ),
                "acquisitions": [a["acquisition"], b["acquisition"]],
                "footprint_overlap_area_native2": a["box"].intersection(b["box"]).area,
                "distance_native": a["box"].distance(b["box"]),
            }
        )
    inventory = []
    used_ids = set(ctx["extra"]).union(*(r["ids"] for r in used))
    used_groups = {identity_group(ctx, uid) for uid in used_ids}
    for row in ctx["support"]["per_id_acquisition"]:
        uid = row["global_id"]
        f = ctx["by_id"][uid]
        inventory.append(
            {
                **row,
                "physical_group": identity_group(ctx, uid),
                "previously_used_group": identity_group(ctx, uid) in used_groups,
                "protected_population_group": identity_group(ctx, uid) in ctx["protected_groups"],
                "edge_line": f["designation"] in {"Guia direita", "Guia esquerda"},
                "width_native": f["width"],
                "designation": f["designation"],
                "distance_to_any_used_footprint_native": min(
                    f["geometry"].distance(r["box"]) for r in used
                ),
            }
        )
    jobs = {}
    for job in sorted({r["acquisition"] for r in inventory if r["acquisition"]}):
        rows = [r for r in inventory if r["acquisition"] == job and r["supported_length"] >= 10]
        new = [r for r in rows if not r["previously_used_group"]]
        other_jobs = {
            r["physical_group"]
            for r in inventory
            if r["acquisition"] != job and r["supported_length"] >= 10
        }
        jobs[job] = {
            "supported_client_ids": len(rows),
            "supported_physical_groups": len({r["physical_group"] for r in rows}),
            "previously_unused_supported_groups": len({r["physical_group"] for r in new}),
            "groups_also_observed_in_other_acquisitions": len(
                {r["physical_group"] for r in rows} & other_jobs
            ),
            "groups_unique_to_this_acquisition_in_verified_inventory": sorted(
                {r["physical_group"] for r in rows} - other_jobs
            ),
            "unused_groups_at_least_500_from_all_used_footprints": len(
                {
                    r["physical_group"]
                    for r in new
                    if r["distance_to_any_used_footprint_native"] >= 500
                }
            ),
        }
    dump(
        out / "physical_line_audit.json",
        {
            "graph_basis": "Stage-8B authoritative geometry and documented local continuation graph, enriched with endpoint/overlap evidence; ambiguous joins retained conservatively. No role changes or prediction-based joins.",  # noqa: E501
            "physical_groups": ctx["audit"]["physical_groups"],
            "edge_evidence": edges,
            "regions": [
                {
                    "region": r["id"],
                    "role": r["role"],
                    "bounds": list(r["box"].bounds),
                    "ids": sorted(r["ids"]),
                    "acquisition": r["acquisition"],
                }
                for r in used
            ],
            "region_pairs": region_pairs,
            "exact_three_groups": details,
            "direct_train_dev_id_leakage": sorted(train_ids & dev_ids),
            "historical_metric_qualification": "Existing LEGACY_POSITIVE_DEV, NEW_POSITIVE_DEV and S6_01 results remain unchanged. Identify affected regions/IDs in the three groups; pooled solid/continuity and class-agnostic pixel/instance metrics using them cannot establish fully unseen-physical-line development generalization. Repeated acquisition observations and overlapping crops are not independent samples. Historical benchmarks and consumed NEW_FINAL_TEST_RESERVED remain historical, not untouched tests.",  # noqa: E501
        },
    )
    dump(
        out / "solid_acquisition_inventory.json",
        {
            "per_id_acquisition": inventory,
            "acquisition_summary": jobs,
            "coverage_limit": "Reuses validated support for the 255 Stage-8 screened solid candidates. Other client solids are metadata-only/unmeasured, not asserted absent or unobserved. Repeated acquisitions count observations, not new physical identities.",  # noqa: E501
        },
    )
    protected_files = load(ROOT / "outputs/reports/stage8b/approval_manifest.json")["sha256"]
    assert all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha
        for p, sha in protected_files.items()
    )
    dump(
        out / "preservation_baseline.json",
        {
            "protected_approval_inputs": protected_files,
            "approval_manifest_sha256": hashlib.sha256(
                (ROOT / "outputs/reports/stage8b/approval_manifest.json").read_bytes()
            ).hexdigest(),
            "prior_manifests": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    ROOT / "outputs/reports/stage8/package_manifest.json",
                    ROOT / "outputs/reports/stage8b/package_manifest.json",
                ]
            },
        },
    )
    print(
        f"Audited {len(used)} role-tagged regions, {len(edges)} geometry connections and exactly {len(details)} training/dev overlap groups"  # noqa: E501
    )
    print(jobs)


if __name__ == "__main__":
    main()
