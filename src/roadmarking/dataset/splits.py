"""Spatial train/val/test splitting without overlap or instance leakage.

Tiles are ordered along each survey corridor (principal axis of tile centres), cut into
contiguous blocks, and blocks are assigned to splits as contiguous runs. Any tile whose
footprint overlaps (or lies within ``guard`` native units of) a tile of another split, or
that contains an instance already used by another split, becomes a guard tile and is
excluded from all splits. Proportions are targets, never forced.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from shapely import STRtree
from shapely.geometry import box

SPLITS = ("train", "val", "test")


def _order_along_track(tiles: list[dict]) -> list[dict]:
    centres = np.array(
        [
            [
                (t["native_xy_bbox"][0] + t["native_xy_bbox"][2]) / 2,
                (t["native_xy_bbox"][1] + t["native_xy_bbox"][3]) / 2,
            ]
            for t in tiles
        ]
    )
    if len(tiles) < 2:
        return list(tiles)
    centred = centres - centres.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axis = vt[0] if vt[0][np.argmax(np.abs(vt[0]))] > 0 else -vt[0]  # deterministic sign
    proj = centred @ axis
    order = np.lexsort((np.array([t["tile_id"] for t in tiles]), proj))
    return [tiles[i] for i in order]


def spatial_split(
    tiles: list[dict],
    tile_instances: dict[str, set[int]],
    tile_classes: dict[str, Counter],
    ratios=(0.70, 0.15, 0.15),
    block_tiles: int = 6,
    guard: float = 0.0,
    min_instances_per_split: int = 50,
    long_instances: set[int] | None = None,
) -> dict:
    """``long_instances``: continuous features (e.g. a 40 km edge line) that cannot fit one
    split. They are not used for the instance-sharing guard; footprint guards ensure no
    painted pixel of them appears in two splits, and they are reported as segmented."""
    long_instances = long_instances or set()
    labelled = [t for t in tiles if tile_instances.get(t["tile_id"])]
    by_source = defaultdict(list)
    for t in labelled:
        by_source[t["source_file"]].append(t)
    blocks = []
    for source in sorted(by_source):
        ordered = _order_along_track(by_source[source])
        for i in range(0, len(ordered), block_tiles):
            blocks.append(ordered[i : i + block_tiles])
    weights = [sum(len(tile_instances[t["tile_id"]]) for t in b) for b in blocks]
    total = sum(weights) or 1
    cuts = np.cumsum(ratios)[:-1] * total
    assignment: dict[str, str] = {}
    running = 0
    for block, weight in zip(blocks, weights, strict=True):
        centre = running + weight / 2
        split = SPLITS[int(np.searchsorted(cuts, centre, side="right"))]
        for t in block:
            assignment[t["tile_id"]] = split
        running += weight

    # Guard pass 1: overlapping footprints across splits (train keeps priority).
    priority = {"train": 0, "val": 1, "test": 2}
    boxes = {t["tile_id"]: box(*t["native_xy_bbox"]).buffer(guard / 2) for t in labelled}
    ids = list(boxes)
    tree = STRtree([boxes[i] for i in ids])
    guards: dict[str, str] = {}
    for a, b in zip(*tree.query([boxes[i] for i in ids], predicate="intersects"), strict=True):
        ta, tb = ids[a], ids[b]
        if ta == tb or ta in guards or tb in guards:
            continue
        sa, sb = assignment[ta], assignment[tb]
        if sa != sb and boxes[ta].intersection(boxes[tb]).area > 0:
            loser = ta if priority[sa] > priority[sb] else tb
            guards[loser] = "footprint_overlap_with_other_split"
    # Guard pass 2: instance sharing across splits.
    owner: dict[int, str] = {}
    for tid in sorted(assignment, key=lambda t: (priority[assignment[t]], t)):
        if tid in guards:
            continue
        inst = tile_instances[tid] - long_instances
        split = assignment[tid]
        if any(owner.get(i, split) != split for i in inst):
            guards[tid] = "shares_instance_with_other_split"
            continue
        for i in inst:
            owner[i] = split
    final = {tid: s for tid, s in assignment.items() if tid not in guards}

    summary = {}
    for split in SPLITS:
        members = sorted(t for t, s in final.items() if s == split)
        inst = set().union(*(tile_instances[t] for t in members)) if members else set()
        classes = Counter()
        for t in members:
            classes.update(tile_classes.get(t, Counter()))
        summary[split] = {
            "tiles": len(members),
            "unique_instances": len(inst),
            "instance_tile_appearances_by_class": dict(classes),
            "tile_ids": members,
        }
    insufficient = [
        s for s in ("val", "test") if summary[s]["unique_instances"] < min_instances_per_split
    ]
    return {
        "method": "along-track contiguous blocks per source + overlap/instance guard tiles",
        "ratios_target": list(ratios),
        "block_tiles": block_tiles,
        "guard_native_units": guard,
        "assignment": final,
        "guard_tiles": guards,
        "splits": summary,
        "insufficient_splits": insufficient,
        "leakage": check_leakage(final, labelled, tile_instances, long_instances),
    }


def check_leakage(
    assignment: dict[str, str], tiles: list[dict], tile_instances, long_instances=frozenset()
) -> dict:
    lookup = {t["tile_id"]: t for t in tiles if t["tile_id"] in assignment}
    ids = sorted(lookup)
    boxes = [box(*lookup[i]["native_xy_bbox"]) for i in ids]
    overlaps = 0
    if boxes:
        tree = STRtree(boxes)
        for a, b in zip(*tree.query(boxes, predicate="intersects"), strict=True):
            if (
                a < b
                and assignment[ids[a]] != assignment[ids[b]]
                and boxes[a].intersection(boxes[b]).area > 0
            ):
                overlaps += 1
    seen: dict[int, str] = {}
    shared = set()
    for tid in ids:
        for inst in tile_instances.get(tid, ()):
            if seen.setdefault(inst, assignment[tid]) != assignment[tid]:
                shared.add(inst)
    segmented = shared & set(long_instances)
    leaked = shared - set(long_instances)
    return {
        "cross_split_overlapping_tile_pairs": overlaps,
        "instances_in_multiple_splits": len(leaked),
        "long_instances_segmented_across_splits": len(segmented),
        "passed": overlaps == 0 and not leaked,
    }
