"""Reusable pipeline stages shared by ``scripts/digitize.py`` and the development loop.

* ``recognize_cached``: registered-model recognition, stored on disk under a key built from
  the image content, the model hash and the thresholds, so a region whose raster has not
  changed is never sent through the GPU twice (the frozen recognition contract is unchanged);
* ``piece_features``: stitched instances -> observed fragments -> one centerline per fragment
  (the seeds that ``line_tracing`` assembles into physical lines).
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from shapely.geometry import LineString

from roadmarking.digitization.centerline import fragment_centerline
from roadmarking.digitization.fragments import (
    bright_paint,
    continuity_relations,
    extract_fragments,
    merge_evidence_links,
)
from roadmarking.digitization.qa import pixel_consistency, smoothness_mm
from roadmarking.digitization.raster_tiles import pixel_to_xy
from roadmarking.models.production import resolve

MIN_LENGTH_M = {0: 0.5, 1: 0.5}
SMOOTH_WINDOW_PX = {0: 60.0, 1: 30.0}  # 3 m solid / 1.5 m dash local-regression windows
GROUP_GAP_PX = 10
WIDE_M, MIN_ELONGATION = 0.45, 3.0


def _pack(instances):
    out = []
    for inst in instances:
        m = inst["mask"]
        ys, xs = np.nonzero(m)
        box = (
            (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
            if len(ys)
            else (0, 0, 0, 0)
        )
        crop = m[box[0] : box[1], box[2] : box[3]]
        rec = {k: v for k, v in inst.items() if k != "mask"}
        rec.update(_box=box, _shape=crop.shape, _bits=np.packbits(crop), _full=m.shape)
        out.append(rec)
    return out


def _unpack(packed):
    out = []
    for rec in packed:
        m = np.zeros(rec["_full"], bool)
        r0, r1, c0, c1 = rec["_box"]
        n = rec["_shape"][0] * rec["_shape"][1]
        m[r0:r1, c0:c1] = np.unpackbits(rec["_bits"])[:n].astype(bool).reshape(rec["_shape"])
        inst = {k: v for k, v in rec.items() if not k.startswith("_")}
        inst["mask"] = m
        out.append(inst)
    return out


def recognize_cached(model_name, gray, observed, grid, cache_dir=None, device="cuda", bundle=None):
    """Return (instances, diagnostics, registered model spec, loaded_from_cache).

    ``bundle`` = (model, processor, spec) already loaded (and hash-verified) by the caller, so a
    job over many blocks loads the registered model once."""
    spec = bundle[2] if bundle is not None else resolve(model_name, verify=cache_dir is None)
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(gray).tobytes())
    h.update(np.packbits(observed).tobytes())
    h.update(
        json.dumps(
            dict(
                weights=spec.entry["weights_sha256"],
                thresholds=spec.thresholds_v1,
                grid=[grid.x0, grid.y0, grid.rows, grid.cols, grid.cell, grid.tile, grid.stride],
            ),
            sort_keys=True,
        ).encode()
    )
    store = Path(cache_dir) / "recognition" / f"{h.hexdigest()[:24]}.pkl" if cache_dir else None
    if store is not None and store.exists():
        with store.open("rb") as f:
            packed, diag = pickle.load(f)
        return _unpack(packed), diag, spec, True
    from roadmarking.digitization.recognition import recognize
    from roadmarking.models.production import load_production_model

    if bundle is not None:
        model, processor, spec = bundle
    else:
        model, processor, spec = load_production_model(model_name, device)
    instances, diag = recognize(model, processor, gray, observed, grid, spec.thresholds_v1, device)
    if bundle is None:
        del model
    if store is not None:
        store.parent.mkdir(parents=True, exist_ok=True)
        tmp = store.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump((_pack(instances), diag), f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(store)
    return instances, diag, spec, False


def to_xy(grid, rc, offset=(0, 0)):
    rc = np.asarray(rc, float)
    x, y = pixel_to_xy(grid, rc[:, 0] + offset[0], rc[:, 1] + offset[1])
    return np.column_stack([x, y])


def piece_features(instances, gray, observed, grid):
    """One centerline per observed fragment. Returns dict(features, relations, diagnostics,
    rejected, rejected_lines, fragments)."""
    frags, rejected, fdiag = extract_fragments(instances, observed)
    relations = continuity_relations(frags, observed, bright=bright_paint(gray, observed))
    n_before = len(frags)
    frags = merge_evidence_links(frags, relations, observed.shape)
    gparent = {f["fragment_id"]: f["fragment_id"] for f in frags}
    alias = {m: f["fragment_id"] for f in frags for m in f["merged_fragments"]}

    def gfind(i):
        while gparent[i] != i:
            gparent[i] = gparent[gparent[i]]
            i = gparent[i]
        return i

    for rr in relations:
        if rr["kind"] == "possible_continuation" and rr["gap_px"] <= GROUP_GAP_PX:
            a_, b_ = gfind(alias[rr["from_fragment"]]), gfind(alias[rr["to_fragment"]])
            if a_ != b_:
                gparent[a_] = b_
    line_group = {fid: f"G{gfind(fid)}" for fid in gparent}
    fdiag["evidence_merges"] = n_before - len(frags)
    features, rejected_lines = [], []
    for fr in frags:
        cl = fragment_centerline(
            fr["connect_mask"], fr["mask"], smooth_window_px=SMOOTH_WINDOW_PX[fr["cls"]]
        )
        if cl is None or len(cl["refined"]) < 2:
            rejected_lines.append(dict(fragment_id=fr["fragment_id"], reason="no_centerline"))
            continue
        xy = to_xy(grid, cl["refined"], fr["offset"])
        raw_xy = to_xy(grid, cl["raw"], fr["offset"])
        line = LineString(xy)
        length = line.length
        width_m = cl["width_px"] * grid.cell
        flags = list(cl["flags"])
        if length < MIN_LENGTH_M[fr["cls"]]:
            rejected_lines.append(
                dict(
                    fragment_id=fr["fragment_id"], reason="shorter_than_min_length", length_m=length
                )
            )
            continue
        if width_m > WIDE_M or length / max(width_m, 1e-6) < MIN_ELONGATION:
            flags.append("possible_unsupported_marking_shape")
        if fr.get("evidence_bridge_px"):
            flags.append("joined_by_paint_evidence")
        if fr["class_conflict_px"]:
            flags.append("class_conflict")
        pc = pixel_consistency(
            cl["refined"], cl["raw"], fr["connect_mask"], fr["mask"], cl["width_px"]
        )
        pc.update(smoothness_mm(xy))
        features.append(
            dict(
                mark_id=len(features) + 1,
                fragment_id=fr["fragment_id"],
                instance_id=fr["instance_id"],
                class_id=fr["cls"],
                class_name="longitudinal_solid" if fr["cls"] == 0 else "longitudinal_dash",
                score=fr["score"],
                geometry_2d=line,
                raw_2d=LineString(raw_xy) if len(raw_xy) > 1 else None,
                length_m=length,
                width_m=width_m,
                flags=flags,
                tiles=sorted({p["tile"] for p in fr["provenance"]}),
                queries=[f"{p['tile']}:{p['query']}" for p in fr["provenance"]],
                sampling_bridge_px=fr["sampling_bridge_px"],
                instance_ids=fr["instance_ids"],
                provisional_line_group=line_group.get(fr["fragment_id"], f"G{fr['fragment_id']}"),
                duplicate_observations=fr["duplicate_observations"],
                **pc,
            )
        )
    return dict(
        features=features,
        relations=relations,
        diagnostics=fdiag,
        rejected=rejected,
        rejected_lines=rejected_lines,
        fragments=frags,
    )


def deck_pixels(grid, low_height, top_height, min_clearance_m=2.0, where=None):
    """Pixels where a second road-like surface (bridge deck) lies >= ``min_clearance_m``
    above the lowest surface. ``where`` (bool raster) limits the test to those pixels; the
    result elsewhere is False (callers AND it with the top surface's observed cells)."""
    out = np.zeros((grid.rows, grid.cols), bool)
    if where is None:
        where = np.ones((grid.rows, grid.cols), bool)
    rr, cc = np.nonzero(where)
    for k in range(0, len(rr), 2_000_000):
        r, c = rr[k : k + 2_000_000], cc[k : k + 2_000_000]
        x, y = pixel_to_xy(grid, r, c)
        diff = top_height.at(x, y) - low_height.at(x, y)
        out[r, c] = np.nan_to_num(diff, nan=0.0) >= min_clearance_m
    return out


def build_levels(las_paths, grid, cache_dir, model_name=None, deck_min_px=20000, bundle=None):
    """Rasterize every overlapping source on the lowest road surface and, where a structure
    crosses above it, on the deck surface; run (cached) recognition per level.

    Returns (levels, diagnostics, spec). Each level: dict(name, gray, observed, solid, dash,
    instances, height, points, sources, surface)."""
    from roadmarking.digitization.raster_tiles import rasterize_region
    from roadmarking.digitization.surface import Surface

    extracts = Path(cache_dir) / "extracts"
    low = rasterize_region(las_paths, grid, cache_dir=extracts, surface="low")
    top = rasterize_region(las_paths, grid, cache_dir=extracts, surface="top")
    deck = deck_pixels(grid, low["height"], top["height"], where=top["observed"])
    levels, diags, spec = [], {}, None
    for name, ras, keep in (("road", low, None), ("deck", top, deck)):
        if keep is not None:
            if keep.sum() < deck_min_px:
                break
            from scipy.ndimage import binary_dilation as _dil

            keep = _dil(keep, iterations=10) & ras["observed"]
            ras = dict(ras, gray=np.where(keep, ras["gray"], 0).astype(np.uint8), observed=keep)
            p = ras["points"]
            zp = ras["height"].at(p[:, 0], p[:, 1])
            lowz = low["height"].at(p[:, 0], p[:, 1])
            above = np.nan_to_num(zp - lowz, nan=0.0) >= 1.0
            ras["points"] = p[above]
            ras["point_source"] = ras["point_source"][above]
        instances, diag, spec, cached = recognize_cached(
            model_name, ras["gray"], ras["observed"], grid, cache_dir, bundle=bundle
        )
        solid, dash = np.zeros(ras["gray"].shape, bool), np.zeros(ras["gray"].shape, bool)
        for inst in instances:
            (solid if inst["class"] == 0 else dash)[...] |= inst["mask"]
        diags[name] = dict(diag, from_cache=cached, observed_px=int(ras["observed"].sum()))
        levels.append(
            dict(
                name=name,
                gray=ras["gray"],
                observed=ras["observed"],
                solid=solid,
                dash=dash,
                instances=instances,
                height=ras["height"],
                points=ras["points"],
                sources=ras["sources"],
                surface=Surface(ras["points"], ras["point_source"]),
                raster=ras,
            )
        )
    return levels, diags, spec


def level_pieces(levels, grid):
    """Seed pieces of every level, tagged with ``layer``."""
    out = []
    for k, lv in enumerate(levels):
        res = piece_features(lv["instances"], lv["gray"], lv["observed"], grid)
        for f in res["features"]:
            f["layer"] = k
            f["fragment_id"] = f"L{k}_{f['fragment_id']}"
        out += res["features"]
    return out


def on_road_fraction(line, surface, samples=20):
    """Share of points along ``line`` with a fitted road surface under them."""
    xy = np.asarray(line.coords)[:, :2]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    t = np.linspace(0, s[-1], min(samples, max(3, int(s[-1]) + 1)))
    pts = np.column_stack([np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])])
    return float(np.mean([surface.z_at(x, y)["ok"] for x, y in pts]))


def _merges_into(seed, traced, max_angle_deg=15.0, max_gap_m=0.35):
    """A leftover strip is a missed converging line only if it runs along a traced line
    (direction within ``max_angle_deg``) and touches or merges into it (<= ``max_gap_m``)."""
    c = np.asarray(seed.coords)
    t = c[-1] - c[0]
    t = t / max(np.linalg.norm(t), 1e-9)
    for f in traced:
        h = f["geometry_2d"]
        if h.distance(seed) > max_gap_m:
            continue
        d = h.project(seed.centroid)
        a = np.asarray(h.interpolate(max(d - 1, 0)).coords[0])[:2]
        b = np.asarray(h.interpolate(min(d + 1, h.length)).coords[0])[:2]
        u = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        if abs(float(u @ t)) >= np.cos(np.radians(max_angle_deg)):
            return True
    return False


def trace_features(features, levels, grid, *, paint_delta=40):
    """Assemble seed pieces into traced physical lines.

    ``levels``: list of dict(gray, observed, solid, dash, height, surface, name), one per
    height level; ``features[i]["layer"]`` names the level a piece was recognised on (default
    0). Returns (features, audit dict). Seeds on wide blobs (not line-like) or without road
    surface under them are passed through untouched and stay flagged.
    """
    from roadmarking.digitization.line_tracing import Layer, seed_pieces, trace_lines

    layers, seeds = [], []
    for k, lv in enumerate(levels):
        bright = bright_paint(lv["gray"], lv["observed"], delta=paint_delta)
        model = lv["solid"] | lv["dash"]
        layers.append(
            Layer(bright, lv["observed"], model, lv.get("height"), lv.get("name", str(k)))
        )
        for xy in seed_pieces(bright, model, grid):
            line = LineString(xy)
            if on_road_fraction(line, lv["surface"]) < 0.5:
                continue
            seeds.append(
                dict(
                    fragment_id=f"paint_L{k}_{len(seeds)}",
                    instance_id=None,
                    class_id=-1,
                    class_name="unknown",
                    score=0.0,
                    geometry_2d=line,
                    raw_2d=None,
                    length_m=line.length,
                    width_m=np.nan,
                    flags=["paint_only_detection"],
                    tiles=[],
                    queries=[],
                    sampling_bridge_px=0,
                    instance_ids=[],
                    provisional_line_group=f"paint_L{k}_{len(seeds)}",
                    duplicate_observations=0,
                    containment=1.0,
                    paint_coverage=1.0,
                    skeleton_displacement_px=0.0,
                    jitter_rms_mm=0.0,
                    jitter_max_mm=0.0,
                    layer=k,
                )
            )
    eligible, held = [], []
    for f in features:
        f.setdefault("layer", 0)
        wide = "possible_unsupported_marking_shape" in f["flags"]
        off_road = on_road_fraction(f["geometry_2d"], levels[f["layer"]]["surface"]) < 0.5
        if off_road:
            f["_held"] = "not_traced (no road surface: off-road candidate)"
        elif wide:
            f["_held"] = "not_traced (wide paint shape)"
        (held if wide or off_road else eligible).append(f)
    pieces = eligible + seeds
    for k, f in enumerate(pieces, start=1):
        f["mark_id"] = k
    traced, chain_audit = trace_lines(pieces, grid, layers=layers)
    # Second pass: where two stripes converge closely their paint merges and recognition often
    # reports one line only. A thin elongated strip of paint left beside the traced lines is a
    # missed line: trace it as well (it is then joined at the junction below).
    from rasterio.features import rasterize

    from roadmarking.digitization.line_tracing import drop_overlaps

    second = []
    if traced:
        taken = rasterize(
            [f["geometry_2d"].buffer(0.12) for f in traced],
            out_shape=layers[0].bright.shape,
            transform=grid.affine,
        ).astype(bool)
        for k, lv in enumerate(levels):
            for xy in seed_pieces(layers[k].bright, taken, grid, min_length_m=2.0):
                line = LineString(xy)
                if not _merges_into(line, traced) or on_road_fraction(line, lv["surface"]) < 0.5:
                    continue
                second.append(
                    dict(
                        pieces[0],
                        fragment_id=f"leftover_L{k}_{len(second)}",
                        instance_id=None,
                        class_id=-1,
                        class_name="unknown",
                        geometry_2d=line,
                        length_m=line.length,
                        flags=["paint_only_detection", "second_pass_leftover_paint"],
                        tiles=[],
                        queries=[],
                        instance_ids=[],
                        layer=k,
                        mark_id=len(pieces) + len(second) + 1,
                    )
                )
    if second:
        extra, extra_audit = trace_lines(second, grid, layers=layers)
        traced, _, _ = drop_overlaps(traced + extra)
        chain_audit += extra_audit
    # Junctions: converging child lines are extended to meet their parent (shared vertex).
    from scipy.ndimage import binary_dilation

    from roadmarking.digitization.junctions import apply_joins
    from roadmarking.digitization.raster_tiles import xy_to_pixel

    paint_any = binary_dilation(np.any([lay.paint for lay in layers], axis=0), iterations=1)

    def paint_at(xy):
        r, c = xy_to_pixel(grid, xy[:, 0], xy[:, 1])
        r, c = np.rint(r).astype(int), np.rint(c).astype(int)
        ok = (r >= 0) & (c >= 0) & (r < grid.rows) & (c < grid.cols)
        out = np.zeros(len(xy), bool)
        out[ok] = paint_any[r[ok], c[ok]]
        return out

    junction_joins = apply_joins(traced, paint_at)
    union = None
    if traced:
        from shapely.ops import unary_union

        union = unary_union([f["geometry_2d"] for f in traced]).buffer(0.15)
    out = list(traced)
    dropped = 0
    for f in held:
        if union is not None and f["geometry_2d"].intersection(union).length > 0.5 * f["length_m"]:
            dropped += 1
            continue
        out.append(
            dict(
                f,
                track_id=None,
                class_rule="model",
                dash_pattern=None,
                gap_next_m=None,
                paint_traced_m=0.0,
                paint_fraction=float(f.get("paint_coverage", np.nan)),
                z_hint=None,
                inferred_segments=[],
                inferred_labels=[],
                inferred_length_m=0.0,
                merged_marks=[f["mark_id"]],
                flags=sorted(set(f["flags"]) | {f.pop("_held")}),
            )
        )
    for k, f in enumerate(out, start=1):
        f["mark_id"] = k
    audit = dict(
        chain=chain_audit,
        seed_pieces=len(eligible),
        paint_only_seeds=len(seeds),
        held_untraced=len(held),
        held_dropped_as_duplicates=dropped,
        traced=len(traced),
        second_pass_seeds=len(second),
        junction_joins=junction_joins,
    )
    return out, audit


__all__ = [
    "build_levels",
    "deck_pixels",
    "level_pieces",
    "on_road_fraction",
    "piece_features",
    "recognize_cached",
    "to_xy",
    "trace_features",
]
