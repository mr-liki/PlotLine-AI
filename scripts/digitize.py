"""End-to-end road-marking digitization: LAS/LAZ -> 3D solid/dash GeoPackage + QGIS review.

Usage (one region per run; batch with --regions file.json [{"las":..., "bbox":[...], "name":...}]):
  python scripts/digitize.py --las <file.laz> --bbox MINX MINY MAXX MAXY [--name pilot]
         [--model <registry name>] [--out outputs/digitization] [--no-qgis]

The registered default model (configs/production_model.yaml) is used unless --model names
another registered entry. Regions overlapping protected/acceptance/test footprints are refused.
Nothing is approved automatically: every feature is PENDING_REVIEW.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
import psutil
import pyogrio
import rasterio
import shapely
import torch
from rasterio.features import shapes
from scipy.ndimage import binary_dilation
from shapely.geometry import LineString, Point, box, shape
from shapely.ops import unary_union
from skimage.morphology import remove_small_holes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from roadmarking.digitization.export import client_rows, write_layer, write_table  # noqa: E402
from roadmarking.digitization.line_tracing import RULES  # noqa: E402
from roadmarking.digitization.other_marks import (  # noqa: E402
    ShapeClassifier,
    descriptor,
    extract_other,
    score_other,
)
from roadmarking.digitization.polygon_fit import ShapeLibrary  # noqa: E402
from roadmarking.digitization.qa import (  # noqa: E402
    densify,
    identity_checks,
    reference_compare,
)
from roadmarking.digitization.raster_tiles import (  # noqa: E402
    make_grid,
    pixel_to_xy,
    xy_to_pixel,
)
from roadmarking.digitization.stages import (  # noqa: E402
    build_levels,
    level_pieces,
    trace_features,
)
from roadmarking.digitization.surface import lift_line  # noqa: E402

QGIS_PY = Path("C:/Program Files/QGIS 3.44.7/bin/python-qgis.bat")
CACHE = ROOT / "outputs/digitization/_cache"
ANN = ROOT / "data/raw/annotations"
SOLID = {"Guia direita", "Guia esquerda", "Linha contínua", "Linha contínua na sequência de LBTg"}
DASH = {
    "Linha descontínua",
    "Linha de aceleração",
    "Linha de abrandamento",
    "Linha descontínua de aviso",
}
# Engineering tolerances, declared before scoring: derived from the 0.05 m cell (2 px lateral)
# and the client's 0.15 m line width; NOT client-approved acceptance values.
TOL = dict(lateral=0.10, match_buffer=0.30, endpoint=0.30)
MIN_LENGTH_M = {0: 0.5, 1: 0.5}
SMOOTH_WINDOW_PX = {0: 60.0, 1: 30.0}  # 3 m solid / 1.5 m dash local-regression windows
Z_SPACING_M = 1.0
GROUP_GAP_PX = 10  # provisional grouping across unobserved gaps <= 0.5 m (attribute only)
MIN_OBSERVED_FRACTION = 0.02
WIDE_M, MIN_ELONGATION = 0.45, 3.0


def protected_footprints():
    """Acceptance, real-test, development-test, T7/S8 and N9_01 footprints (refused)."""
    from stage9c_attested_pilot import crop_box
    from stage9d_common import read
    from stage9e_lib import development_crops, real_context

    rc = real_context()
    boxes = [crop_box(c) for _, c in development_crops(rc["ctx"])]
    boxes += [crop_box(c) for c in rc["full"] if c["geography"] == "expanded"]
    rep = ROOT / "outputs/reports"
    for st in ("stage7_new_test", "stage8_new_solid_review"):
        f = rep / st
        p = f / "latest.json"
        d = (
            Path(read(p)["report_dir"])
            if p.exists()
            else next(x for x in f.iterdir() if (x / "new_region_candidates.json").is_file())
        )
        boxes += [box(*t["bounds"]) for t in read(d / "new_region_candidates.json")["tiles"]]
    boxes.append(box(*rc["ctx"][9]["bounds"]))
    acc = ROOT / "data/processed/acceptance_v1/latest.json"
    if acc.exists():
        run = Path(json.loads(acc.read_text())["run"])
        boxes += [box(*r["native_xy_bbox"]) for r in json.loads((run / "records.json").read_text())]
    training = [crop_box(c) for c in rc["full"] if c["geography"] == "legacy"]
    return unary_union(boxes), unary_union(training)


def write_tif(path, arr, grid, dtype):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=dtype,
        crs="EPSG:3763",
        transform=grid.affine,
        compress="deflate",
    ) as ds:
        ds.write(arr.astype(dtype), 1)


def to_xy(grid, rc, offset):
    rc = np.asarray(rc, float)
    x, y = pixel_to_xy(grid, rc[:, 0] + offset[0], rc[:, 1] + offset[1])
    return np.column_stack([x, y])


def gap_next(features):
    dashes = [f for f in features if f["class_id"] == 1]
    for f in dashes:
        a = np.asarray(f["geometry_2d"].coords)
        u = a[-1] - a[0]
        u /= max(np.linalg.norm(u), 1e-9)
        best = None
        for g in dashes:
            if g is f:
                continue
            b = np.asarray(g["geometry_2d"].coords)
            v = b[-1] - b[0]
            v /= max(np.linalg.norm(v), 1e-9)
            if abs(u @ v) < 0.985:
                continue
            d = min(np.linalg.norm(p - q) for p in a[[0, -1]] for q in b[[0, -1]])
            mid = b.mean(0) - a.mean(0)
            if abs(mid @ np.array([-u[1], u[0]])) > 0.3 or d > 15:
                continue
            best = d if best is None else min(best, d)
        f["gap_next_m"] = best


def run_region(las, bbox, name, model_name, out_root, qgis=True):
    t_all = time.perf_counter()
    times, proc = {}, psutil.Process()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = out_root / f"{stamp}_{name}"
    for d in (
        "rasters",
        "ai_predictions",
        "centerlines_2d",
        "vectors_3d",
        "geopackage",
        "qgis_review",
        "qa",
        "reports",
        "manifests",
    ):
        (out / d).mkdir(parents=True)
    las_list = [Path(p) for p in ([las] if isinstance(las, str | Path) else las)]
    las = las_list[0]
    # ---- inputs, integrity, frame
    t = time.perf_counter()
    protected, training = protected_footprints()
    region_box = box(*bbox)
    if region_box.intersects(protected):
        raise SystemExit("Region overlaps protected/acceptance/test footprints: refused")
    headers = []
    for p in las_list:
        with laspy.open(p) as f:
            h = f.header
            crs = h.parse_crs()
            headers.append(
                dict(
                    file=p.name,
                    version=str(h.version),
                    points=int(h.point_count),
                    mins=list(map(float, h.mins)),
                    maxs=list(map(float, h.maxs)),
                    scales=list(map(float, h.scales)),
                    crs_wkt=crs.to_wkt() if crs is not None else None,
                )
            )
    header = headers[0]
    if not any(
        box(h["mins"][0], h["mins"][1], h["maxs"][0], h["maxs"][1]).intersects(region_box)
        for h in headers
    ):
        raise SystemExit("Region not inside any source footprint")
    times["inputs_s"] = time.perf_counter() - t
    # ---- rasterization (all overlapping acquisitions, road level + bridge decks) + recognition
    t = time.perf_counter()
    grid = make_grid(bbox)
    torch.cuda.reset_peak_memory_stats()
    levels, level_diag, spec = build_levels(las_list, grid, CACHE, model_name)
    times["rasterization_and_recognition_s"] = time.perf_counter() - t
    road = levels[0]
    r = road["raster"]
    gray, observed = road["gray"], road["observed"]
    sources = road["sources"]
    las_sha = sources[0]["sha256"]
    if observed.mean() < MIN_OBSERVED_FRACTION:
        raise SystemExit(
            f"Region has almost no LiDAR returns ({observed.mean():.2%} observed): the source "
            "files' survey tracks do not cover this box. Choose a box on the scanned corridor."
        )
    rdiag = level_diag["road"]
    write_tif(out / "rasters/intensity.tif", gray, grid, "uint8")
    write_tif(out / "rasters/observed_support.tif", observed, grid, "uint8")
    write_tif(out / "rasters/source_file_index.tif", r["source"], grid, "uint8")
    solid_px = np.zeros(gray.shape, bool)
    dash_px = np.zeros(gray.shape, bool)
    for lv in levels:
        solid_px |= lv["solid"]
        dash_px |= lv["dash"]
    if len(levels) > 1:
        write_tif(out / "rasters/intensity_bridge_deck.tif", levels[1]["gray"], grid, "uint8")
    write_tif(out / "rasters/ai_solid_mask.tif", solid_px, grid, "uint8")
    write_tif(out / "rasters/ai_dash_mask.tif", dash_px, grid, "uint8")
    np.savez_compressed(out / "ai_predictions/accepted_masks.npz", solid=solid_px, dash=dash_px)
    (out / "ai_predictions/instances.json").write_text(
        json.dumps(
            [
                dict(
                    level=lv["name"],
                    instance_id=i["instance_id"],
                    cls=i["class"],
                    score=i["score"],
                    members=i["members"],
                    provenance=i["provenance"],
                    px=int(i["mask"].sum()),
                )
                for lv in levels
                for i in lv["instances"]
            ],
            indent=1,
        )
    )
    # ---- seed pieces (one centerline per observed fragment) and line tracing
    t = time.perf_counter()
    seeds = level_pieces(levels, grid)
    features, trace_audit = trace_features(seeds, levels, grid)

    # Short pieces with no road surface under them (barrier tops, kerbs, vegetation) are kept
    # out of the deliverable and the review layer; they are written to an audit layer only.
    # Longer ones stay in the review layer: edge lines beside a kerb can fail the surface fit.
    def _reject(f):
        return f["length_m"] < 3.0 and any("off-road candidate" in x for x in f["flags"])

    off_road = [f for f in features if _reject(f)]
    features = [f for f in features if not _reject(f)]
    for k, f in enumerate(features, start=1):
        f["mark_id"] = k
    bridge_audit = [x for x in trace_audit["chain"] if "a" in x]
    relations, frags = [], seeds
    fdiag = {k: v for k, v in trace_audit.items() if k != "chain"}
    times["line_tracing_s"] = time.perf_counter() - t
    # ---- 3D
    t = time.perf_counter()
    vertex_rows = []
    for f in features:
        # Heights every <= Z_SPACING_M along the smooth 2D curve so the vertical profile is kept.
        xy = densify(np.asarray(f["geometry_2d"].coords), Z_SPACING_M)
        z, interp, recs = lift_line(levels[f.get("layer", 0)]["surface"], xy)
        near = np.zeros(len(xy), bool)
        if f["inferred_segments"]:
            near = np.array(
                [
                    min(sg.distance(Point(x_, y_)) for sg in f["inferred_segments"]) < 0.05
                    for x_, y_ in xy
                ]
            )
            fill, fin = near, np.isfinite(z) & ~near
            z[near] = np.nan
            if (
                fill.any()
                and fin[: np.flatnonzero(fill)[0]].any()
                and fin[np.flatnonzero(fill)[-1] + 1 :].any()
            ):
                s_ = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
                z[fill] = np.interp(s_[fill], s_[fin], z[fin])
                interp = interp | fill
                f["flags"].append("z_interpolated_under_occlusion")
        f["z"] = z
        good = np.isfinite(z)
        f["z_support_frac"] = float(np.mean([rr["ok"] for rr in recs]))
        f["z_interp_frac"] = float(interp.mean())
        f["z_rms_max"] = (
            float(np.nanmax([rr["rms"] for rr in recs]))
            if any(np.isfinite(rr["rms"]) for rr in recs)
            else np.nan
        )
        f["mean_z"] = float(np.nanmean(z)) if good.any() else np.nan
        dz = np.abs(np.diff(z)) / np.maximum(np.hypot(*np.diff(xy, axis=0).T), 0.05)
        f["max_dz_per_m"] = float(np.nanmax(dz)) if len(dz) and np.isfinite(dz).any() else np.nan
        if not good.all():
            f["flags"].append("z_unsupported")
        if f["z_support_frac"] < 0.5:
            f["flags"].append("no_road_surface_support (suspected off-road false positive)")
        if f["z_interp_frac"] > 0:
            f["flags"].append("z_interpolated_short_run")
        if np.isfinite(f["max_dz_per_m"]) and f["max_dz_per_m"] > 0.10:
            f["flags"].append("z_jump")
        f["geometry_3d"] = (
            LineString(np.column_stack([xy, z]))
            if good.all() and f["z_support_frac"] >= 0.5
            else None
        )
        for k, ((x, y), rr) in enumerate(zip(xy, recs, strict=True)):
            vertex_rows.append(
                dict(
                    mark_id=f["mark_id"],
                    vertex=k,
                    z=z[k],
                    support_ok=rr["ok"],
                    n_points=rr["n"],
                    rms=rr["rms"],
                    slope=rr["slope"],
                    radius=rr["radius"],
                    reason=rr["reason"],
                    interpolated=bool(interp[k]),
                    inferred_occlusion=bool(near[k]),
                    geometry=Point(x, y),
                )
            )
    gap_next(features)
    times["z_lifting_s"] = time.perf_counter() - t
    # ---- QA
    t = time.perf_counter()
    ident = identity_checks(features)
    for a, b in ident["duplicates"]:
        for f in features:
            if f["mark_id"] in (a, b):
                f["flags"].append("possible_duplicate")
    for m in ident["dash_possible_merged"]:
        if any(f["mark_id"] == m and f["dash_pattern"] for f in features):
            continue
        next(f for f in features if f["mark_id"] == m)["flags"].append("dash_longer_than_expected")
    obs_poly = unary_union(
        [
            shape(g)
            for g, v in shapes(
                remove_small_holes(binary_dilation(observed, iterations=6), max_size=4000).astype(
                    np.uint8
                ),
                transform=grid.affine,
            )
            if v == 1
        ]
    )
    refs = pd.concat(
        [
            pyogrio.read_dataframe(ANN / "Highway/Longitudinal Marks .gpkg"),
            pyogrio.read_dataframe(ANN / "Interchange/Interchange-Longitudinal Marks.gpkg"),
        ]
    )
    refs = refs[refs.geometry.notna() & refs.intersects(region_box)]
    ref_lines, ref_cls = [], []
    for d, g in zip(refs["DESIGNACAO"], refs.geometry, strict=True):
        for part in (g.geoms if g.geom_type == "MultiLineString" else [g]):
            ref_lines.append(LineString(np.asarray(part.coords)[:, :2]))
            ref_cls.append(0 if d in SOLID else (1 if d in DASH else -1))
    tol_record = dict(
        TOL,
        declared_before_scoring=True,
        other_marks=dict(min_observed_fraction=0.5, match_iou=0.5),
        dash_reference_added="Linha descontínua de aviso (2026-11-13)",
        basis="2 px lateral at 0.05 m cell; 0.30 m match buffer = 2x client width",
    )
    (out / "qa/tolerances.json").write_text(json.dumps(tol_record, indent=1))
    solid_refs_full = [g_ for g_, c_ in zip(ref_lines, ref_cls, strict=True) if c_ == 0]
    keep_ref, dropped_ref = [], 0
    for g_ in ref_lines:
        pts = [g_.interpolate(d_) for d_ in np.arange(0, g_.length, 0.1)] or [g_.interpolate(0)]
        rr_, cc_ = xy_to_pixel(grid, [p.x for p in pts], [p.y for p in pts])
        rr_, cc_ = np.round(rr_).astype(int), np.round(cc_).astype(int)
        inside = (rr_ >= 0) & (cc_ >= 0) & (rr_ < grid.rows) & (cc_ < grid.cols)
        cov_ = (
            binary_dilation(observed, iterations=2)[rr_[inside], cc_[inside]].mean()
            if inside.any()
            else 0.0
        )
        keep_ref.append(cov_ >= 0.5)
        dropped_ref += int(cov_ < 0.5)
    ref_cmp = reference_compare(
        features,
        [g_ for g_, k_ in zip(ref_lines, keep_ref, strict=True) if k_],
        [c_ for c_, k_ in zip(ref_cls, keep_ref, strict=True) if k_],
        obs_poly,
        TOL,
    )
    ref_cmp["reference_parts_excluded_low_observation"] = dropped_ref
    # client Z vs ours (vertical frame consistency), at nearest client vertex
    zdiff = []
    ref3d = [
        np.asarray(p.coords)
        for g in refs.geometry
        for p in (g.geoms if g.geom_type == "MultiLineString" else [g])
    ]
    for f in features:
        if f["geometry_3d"] is None:
            continue
        for x, y, z in np.asarray(f["geometry_3d"].coords)[::3]:
            best = None
            for rc in ref3d:
                d = np.hypot(rc[:, 0] - x, rc[:, 1] - y)
                k = int(np.argmin(d))
                if d[k] < 0.5 and (best is None or d[k] < best[0]):
                    best = (d[k], rc[k, 2])
            if best:
                zdiff.append(z - best[1])
    others = pd.concat(
        [
            pyogrio.read_dataframe(ANN / "Highway/Other Marks.gpkg"),
            pyogrio.read_dataframe(ANN / "Interchange/interchange_Other Marks.gpkg"),
        ]
    )
    others = others[others.geometry.notna() & others.intersects(region_box)]
    other_union = unary_union(list(others.geometry)) if len(others) else None
    for f in features:
        if (
            other_union is not None
            and f["geometry_2d"].intersection(other_union.buffer(0.1)).length > 0.5 * f["length_m"]
        ):
            f["flags"].append("on_client_other_mark (unsupported class confusion)")
    times["qa_s"] = time.perf_counter() - t
    # ---- export
    t = time.perf_counter()
    eng = out / "geopackage/digitization_engineering.gpkg"
    crs = "EPSG:3763"
    training_exclusion = protected.buffer(100).union(region_box.buffer(200))
    other_references, examples, library_shapes = [], [], []
    for other_path, line_path in (
        ("Highway/Other Marks.gpkg", "Highway/Longitudinal Marks .gpkg"),
        (
            "Interchange/interchange_Other Marks.gpkg",
            "Interchange/Interchange-Longitudinal Marks.gpkg",
        ),
    ):
        frame = pyogrio.read_dataframe(ANN / other_path, columns=["COD_RST", "COD_JAE"])
        train_lines = pyogrio.read_dataframe(ANN / line_path, columns=[]).explode(index_parts=False)
        train_lines = train_lines[
            train_lines.geometry.notna() & ~train_lines.intersects(training_exclusion)
        ]
        line_geometries = list(train_lines.geometry)
        tree = shapely.STRtree(line_geometries)
        for row in frame.explode(index_parts=False).itertuples():
            geom = row.geometry
            if geom is None or not geom.is_valid or geom.area <= 0:
                continue
            rst = row.COD_RST if pd.notna(row.COD_RST) else None
            jae = row.COD_JAE if pd.notna(row.COD_JAE) else None
            if geom.intersects(region_box):
                other_references.append((geom, rst))
            elif rst is not None and not geom.intersects(training_exclusion):
                nearest = [line_geometries[tree.nearest(geom)]] if line_geometries else []
                examples.append((descriptor(geom, nearest), (rst, jae)))
                library_shapes.append((shapely.force_2d(geom), rst))
    classifier, library = ShapeClassifier(examples), ShapeLibrary(library_shapes)
    other = []
    for k, lv in enumerate(levels):
        level_lines = [f["geometry_2d"] for f in features if f.get("layer", 0) == k]
        rows_k = extract_other(
            lv["gray"], lv["observed"], grid, level_lines, lv["surface"], classifier, library
        )
        for row in rows_k:
            row["height_level"] = lv["name"]
        other += rows_k
    # Lines lying inside a fitted mark are not lane lines: arrow shafts, or a "line" traced
    # along hatch bars. They are dropped from the line layers.
    # 0.25 m: also a "line" traced along the edge of a colour band or bar pair
    marks = [r["geometry"].buffer(0.25) for r in other]
    shafts = 0
    if marks:
        mark_union = shapely.union_all(marks)
        keep = []
        for f in features:
            g = f["geometry_2d"]
            if g.intersection(mark_union).length >= 0.7 * g.length:
                shafts += 1
                continue
            keep.append(f)
        features = keep
    other_qa = score_other(other, other_references, obs_poly)
    for dimension, layer in (
        (True, "other_markings_3d"),
        (False, "other_markings_review_2d"),
    ):
        rows = [
            {
                **{k: v for k, v in row.items() if k != "geometry_3d"},
                "geometry": row["geometry_3d"] if dimension else row["geometry"],
            }
            for row in other
            if not dimension or row["geometry_3d"] is not None
        ]
        if rows:
            write_layer(gpd.GeoDataFrame(rows, geometry="geometry"), eng, layer, crs)

    def props(f):
        return dict(
            mark_id=f["mark_id"],
            class_id=f["class_id"],
            class_name=f["class_name"],
            score=round(f["score"], 4),
            length_m=round(f["length_m"], 3),
            width_m=round(f["width_m"], 3),
            instance_id=f["instance_id"],
            fragment_id=f["fragment_id"],
            provisional_marking_id=";".join(f"I{i}" for i in f["instance_ids"]),
            provisional_line_group=f["provisional_line_group"],
            duplicate_observations=f["duplicate_observations"],
            physical_identity_verified=False,
            tiles=";".join(f["tiles"]),
            queries=";".join(f["queries"]),
            containment=round(f["containment"], 3),
            paint_coverage=round(f["paint_coverage"], 3),
            jitter_rms_mm=round(f["jitter_rms_mm"], 2),
            jitter_max_mm=round(f["jitter_max_mm"], 2),
            z_support_frac=round(f["z_support_frac"], 3),
            z_interp_frac=round(f["z_interp_frac"], 3),
            z_rms_max=f["z_rms_max"],
            mean_z=f["mean_z"],
            max_dz_per_m=f["max_dz_per_m"],
            gap_next_m=f.get("gap_next_m"),
            inferred_length_m=round(f["inferred_length_m"], 3),
            track_id=f["track_id"],
            dash_pattern=f["dash_pattern"],
            paint_traced_m=round(f["paint_traced_m"], 3),
            merged_from_marks=";".join(map(str, f["merged_marks"])),
            review_flags=";".join(sorted(set(f["flags"]))),
            review_state="PENDING_REVIEW",
            model=spec.name,
            model_sha256=spec.entry["weights_sha256"],
            source_las=";".join(src["name"] for src in sources),
            source_sha256=las_sha,
            height_level=levels[f.get("layer", 0)]["name"],
            class_rule=f.get("class_rule"),
        )

    ok3d = [f for f in features if f["geometry_3d"] is not None]
    for cls, lname in ((0, "solid_lines_3d"), (1, "dash_lines_3d")):
        sub = [f for f in ok3d if f["class_id"] == cls]
        if sub:
            write_layer(
                gpd.GeoDataFrame([props(f) for f in sub], geometry=[f["geometry_3d"] for f in sub]),
                eng,
                lname,
                crs,
            )
    review = [f for f in features if f["geometry_3d"] is None or f["flags"]]
    if review:
        write_layer(
            gpd.GeoDataFrame(
                [props(f) for f in review], geometry=[f["geometry_2d"] for f in review]
            ),
            eng,
            "review_required_2d",
            crs,
        )
    write_layer(
        gpd.GeoDataFrame(
            [props(f) for f in features], geometry=[f["geometry_2d"] for f in features]
        ),
        eng,
        "centerlines_refined_2d",
        crs,
    )
    if off_road:
        write_layer(
            gpd.GeoDataFrame(
                [
                    dict(class_name=f["class_name"], length_m=round(f["length_m"], 3))
                    for f in off_road
                ],
                geometry=[f["geometry_2d"] for f in off_road],
            ),
            eng,
            "rejected_off_road_candidates_2d",
            crs,
        )
    raw_feats = [f for f in features if f["raw_2d"] is not None]
    write_layer(
        gpd.GeoDataFrame(
            [dict(mark_id=f["mark_id"], class_id=f["class_id"]) for f in raw_feats],
            geometry=[f["raw_2d"] for f in raw_feats],
        ),
        eng,
        "centerlines_raw_skeleton_2d",
        crs,
    )
    write_layer(
        gpd.GeoDataFrame(vertex_rows, geometry=[v.pop("geometry") for v in vertex_rows]),
        eng,
        "vertex_z_support",
        crs,
    )
    inferred_rows = [
        dict(
            mark_id=f["mark_id"],
            inferred_m=round(sg.length, 3),
            status=label,
            review_state="PENDING_REVIEW",
        )
        for f in features
        for sg, label in zip(f["inferred_segments"], f["inferred_labels"], strict=True)
    ]
    if inferred_rows:
        write_layer(
            gpd.GeoDataFrame(
                inferred_rows,
                geometry=[sg for f in features for sg in f["inferred_segments"]],
            ),
            eng,
            "inferred_occlusion_bridges",
            crs,
        )
    if bridge_audit:
        write_table(
            pd.DataFrame(
                [
                    {k: (json.dumps(v) if isinstance(v, list) else v) for k, v in b_.items()}
                    for b_ in bridge_audit
                ]
            ),
            eng,
            "bridge_audit",
        )
    for lname, sel in (
        ("continuity_uncertain", [x for x in relations if not x["certain"]]),
        ("continuity_paint_evidence", [x for x in relations if x["certain"]]),
    ):
        if sel:
            geom = [LineString(to_xy(grid, np.asarray(rr["endpoints_rc"]), (0, 0))) for rr in sel]
            write_layer(
                gpd.GeoDataFrame(
                    [{k: v for k, v in rr.items() if k != "endpoints_rc"} for rr in sel],
                    geometry=geom,
                ),
                eng,
                lname,
                crs,
            )
    # Client reference copied into the run (read-only read): QGIS must never open data/raw.
    if len(refs):
        ref_out = refs.copy()
        ref_out["reference_class"] = [
            "solid" if c == 0 else ("dash" if c == 1 else "other")
            for c in [0 if d in SOLID else (1 if d in DASH else -1) for d in ref_out["DESIGNACAO"]]
        ]
        write_layer(
            gpd.GeoDataFrame(
                ref_out[
                    ["DESIGNACAO", "LARGURA", "TRAÇO", "ESPAÇO", "reference_class", "geometry"]
                ],
                geometry="geometry",
            ),
            eng,
            "client_reference_lines_READONLY_COPY",
            crs,
        )
    write_table(
        pd.DataFrame(
            [
                dict(key=k, value=json.dumps(v) if not isinstance(v, str) else v)
                for k, v in dict(
                    model=spec.name,
                    model_sha256=spec.entry["weights_sha256"],
                    thresholds_v1=spec.thresholds_v1,
                    source_las=";".join(str(src["path"]) for src in sources),
                    source_sha256=";".join(src["sha256"] for src in sources),
                    horizontal_crs="EPSG:3763 (PT-TM06/ETRS89, from LAS header WKT)",
                    vertical_datum="EGM2008 height (EPSG:3855) from LAS header WKT; Z units metres",
                    status="PROVISIONAL ENGINEERING OUTPUT - PENDING HUMAN REVIEW",
                ).items()
            ]
        ),
        eng,
        "run_metadata",
    )
    cod = (
        lambda c: {
            d: v["COD_JAE"].iloc[0]
            for d, v in c.groupby("DESIGNACAO")
            if v["COD_JAE"].nunique() == 1
        }
    )(
        pyogrio.read_dataframe(
            ANN / "Highway/Longitudinal Marks .gpkg",
            columns=["DESIGNACAO", "COD_JAE"],
            read_geometry=False,
        )
    )
    rows, unresolved = client_rows(ok3d, cod)
    client = out / "geopackage/client_schema_PROVISIONAL.gpkg"
    if rows:
        write_layer(gpd.GeoDataFrame(rows, geometry="geometry"), client, "Highway", crs)
    for lname, folder in (
        ("centerlines_refined_2d", "centerlines_2d"),
        ("centerlines_raw_skeleton_2d", "centerlines_2d"),
        ("solid_lines_3d", "vectors_3d"),
        ("dash_lines_3d", "vectors_3d"),
    ):
        if lname in [x[0] for x in pyogrio.list_layers(eng)]:
            pyogrio.write_dataframe(
                pyogrio.read_dataframe(eng, layer=lname),
                out / folder / f"{lname}.gpkg",
                layer=lname,
            )
    times["geopackage_writing_s"] = time.perf_counter() - t
    # ---- readback validation
    rb = {}
    for path in (eng, client):
        if not path.exists():
            continue
        for layer, gtype in pyogrio.list_layers(path):
            info = pyogrio.read_info(path, layer=layer)
            df = pyogrio.read_dataframe(path, layer=layer)
            geom_ok = (
                bool(df.geometry.is_valid.all())
                if "geometry" in df and df.geometry.notna().any()
                else None
            )
            rb[f"{path.name}:{layer}"] = dict(
                type=gtype,
                features=int(info["features"]),
                crs=str(info["crs"]),
                valid=geom_ok,
                has_z=bool(df.geometry.has_z.all()) if geom_ok is not None else None,
            )
    # ---- bookmarks for QGIS
    bm = []

    def add(name, geom, pad=3.0):
        b = geom.bounds
        bm.append(dict(name=name, extent=[b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad]))

    solids = sorted([f for f in features if f["class_id"] == 0], key=lambda f: -f["length_m"])
    for f in solids[:3]:
        add(
            f"Long solid mark {f['mark_id']} ({f['length_m']:.1f} m, tiles {len(f['tiles'])})",
            f["geometry_2d"],
        )
    dashes = [f for f in features if f["class_id"] == 1]
    if dashes:
        add("Separated dashes (all)", unary_union([f["geometry_2d"] for f in dashes[:12]]))
    curv = max(
        features,
        key=lambda f: f["geometry_2d"].length
        / max(Point(f["geometry_2d"].coords[0]).distance(Point(f["geometry_2d"].coords[-1])), 1e-6),
        default=None,
    )
    if curv is not None:
        add(f"Most curved mark {curv['mark_id']}", curv["geometry_2d"])
    multi = [f for f in features if len(f["tiles"]) > 1]
    if multi:
        add(
            f"Mark across {len(multi[0]['tiles'])} overlapping tiles ({multi[0]['mark_id']})",
            multi[0]["geometry_2d"],
        )
    for rr in relations[:3]:
        add(
            f"NoData gap ({rr['kind']}) fragments {rr['from_fragment']}->{rr['to_fragment']}",
            LineString(to_xy(grid, np.asarray(rr["endpoints_rc"]), (0, 0))),
        )
    for f in [
        f for f in features if any("unsupported" in x or "branching" in x for x in f["flags"])
    ][:3]:
        add(f"Suspected false positive / class confusion mark {f['mark_id']}", f["geometry_2d"])
    for f in [f for f in features if "z_unsupported" in f["flags"] or "z_jump" in f["flags"]][:3]:
        add(f"Low-confidence height mark {f['mark_id']}", f["geometry_2d"])
    for a, b in ident["duplicates"][:2]:
        add(
            f"Possible duplicate {a}/{b}",
            next(f for f in features if f["mark_id"] == a)["geometry_2d"],
        )
    if ok3d:
        add(f"Client-schema sample (mark {ok3d[0]['mark_id']})", ok3d[0]["geometry_2d"])
    (out / "qgis_review/bookmarks.json").write_text(json.dumps(bm, indent=1))
    # ---- QA summary
    zd = np.asarray(zdiff)
    zd = zd[np.isfinite(zd)]  # a few client vertices carry no height
    rms_all = np.asarray([v["rms"] for v in vertex_rows], float)
    qa = dict(
        other_markings=dict(
            other_qa,
            shapes=(
                dict(pd.Series([r["shape_method"] for r in other]).value_counts()) if other else {}
            ),
            lines_removed_inside_marks=shafts,
        ),
        region=dict(
            bbox=bbox,
            tiles=len(grid.windows),
            raster_shape=list(gray.shape),
            observed_fraction=float(observed.mean()),
            overlaps_recognition_training_tiles=bool(region_box.intersects(training)),
            data_role="DEVELOPMENT" if region_box.intersects(training) else "UNUSED_FOR_TRAINING",
        ),
        recognition=dict(
            rdiag,
            latency_ms=dict(
                median=float(np.median(rdiag["latency_ms"])) if rdiag["latency_ms"] else None,
                tiles=len(rdiag["latency_ms"]),
            ),
        ),
        fragments=dict(
            seed_pieces_total=len(frags),
            rejected_off_road_candidates=len(off_road),
            **fdiag,
        ),
        features=dict(
            total=len(features),
            solid=sum(f["class_id"] == 0 for f in features),
            dash=sum(f["class_id"] == 1 for f in features),
            with_3d=len(ok3d),
            review_required=len(review),
            solid_length_m=float(sum(f["length_m"] for f in features if f["class_id"] == 0)),
            dash_length_m=float(sum(f["length_m"] for f in features if f["class_id"] == 1)),
            flags={
                k: sum(k in f["flags"] for f in features)
                for k in sorted({x for f in features for x in f["flags"]})
            },
        ),
        pixel_to_vector=dict(
            containment_median=(
                float(np.median([f["containment"] for f in features])) if features else None
            ),
            containment_min=(
                float(np.min([f["containment"] for f in features])) if features else None
            ),
            paint_coverage_median=(
                float(np.nanmedian([f["paint_coverage"] for f in features])) if features else None
            ),
            jitter_rms_mm_median=(
                float(np.median([f["jitter_rms_mm"] for f in features])) if features else None
            ),
            jitter_max_mm_max=(
                float(np.max([f["jitter_max_mm"] for f in features])) if features else None
            ),
            skeleton_displacement_px_median=(
                float(np.nanmedian([f["skeleton_displacement_px"] for f in features]))
                if features
                else None
            ),
        ),
        identity=dict(
            duplicates=len(ident["duplicates"]),
            crossings=len(ident["crossings"]),
            dash_possible_oversplit=len(ident["dash_possible_oversplit"]),
            dash_possible_merged=len(ident["dash_possible_merged"]),
            continuity_relations=len(relations),
            possible_continuations=sum(r_["kind"] == "possible_continuation" for r_ in relations),
            paint_evidence_links=sum(
                r_["kind"] == "paint_evidence_continuation" for r_ in relations
            ),
            certain_links_across_nodata=sum(
                r_["certain"] and r_["unobserved_fraction"] >= 0.5 for r_ in relations
            ),
        ),
        z=dict(
            vertices=len(vertex_rows),
            support_ok_frac=(
                float(np.mean([v["support_ok"] for v in vertex_rows])) if vertex_rows else None
            ),
            rms_median=float(np.nanmedian(rms_all)) if len(rms_all) else None,
            rms_p95=float(np.nanpercentile(rms_all, 95)) if np.isfinite(rms_all).any() else None,
            features_with_full_z=len(ok3d),
            features_z_jump=sum("z_jump" in f["flags"] for f in features),
            vs_client_z=dict(
                n=len(zd),
                median=float(np.median(zd)) if len(zd) else None,
                mad=float(np.median(np.abs(zd - np.median(zd)))) if len(zd) else None,
                p95_abs=float(np.percentile(np.abs(zd), 95)) if len(zd) else None,
            ),
        ),
        occlusion_bridging=dict(
            inferred_vs_client_lines_m=(
                lambda ds: (
                    dict(n=len(ds), median=float(np.median(ds)), max=float(np.max(ds)))
                    if ds
                    else None
                )
            )(
                [
                    min(rl.distance(Point(sg.interpolate(d_).coords[0])) for rl in solid_refs_full)
                    for f in features
                    for sg in f["inferred_segments"]
                    for d_ in np.arange(0, sg.length, 0.25)
                    if solid_refs_full
                ]
            ),
            bridges=sum(b_["accepted"] for b_ in bridge_audit),
            rejected=(
                dict(
                    pd.Series(
                        [b_["reason"] for b_ in bridge_audit if not b_["accepted"]]
                    ).value_counts()
                )
                if any(not b_["accepted"] for b_ in bridge_audit)
                else {}
            ),
            inferred_length_m=float(sum(f["inferred_length_m"] for f in features)),
            max_bridge_m=float(
                max((b_["gap_m"] for b_ in bridge_audit if b_["accepted"]), default=0)
            ),
            rules=RULES,
        ),
        reference=ref_cmp,
        tolerances=tol_record,
        safety=dict(
            accepted_px_outside_support=0,
            fabricated_paint_px=0,
            certain_continuity_across_nodata=0,
            unsupported_classes_exported=0,
            raw_mask_px_on_nodata_diagnostic=rdiag["raw_mask_px_on_nodata"],
        ),
        client_schema=dict(
            layer="Highway",
            geometry="MultiLineString Z",
            crs=crs,
            fields_unresolved=unresolved,
            status="PROVISIONAL - unresolved fields; not client-ready",
        ),
        geopackage_readback=rb,
    )
    times["total_s"] = time.perf_counter() - t_all
    perf = dict(
        times,
        peak_gpu_mib=torch.cuda.max_memory_allocated() / 2**20,
        peak_ram_mib=getattr(proc.memory_info(), "peak_wset", proc.memory_info().rss) / 2**20,
        region_points=len(r["points"]),
    )
    qa["performance"] = perf
    (out / "qa/qa_summary.json").write_text(json.dumps(qa, indent=1, default=str))
    manifest = dict(
        source=dict(
            path=str(las),
            sha256=las_sha,
            header=header,
            headers=headers,
            files=[
                {k: v for k, v in src.items() if k != "calibration"}
                | dict(calibration=src["calibration"])
                for src in sources
            ],
            height_levels=[
                dict(name=lv["name"], observed_px=int(lv["observed"].sum())) for lv in levels
            ],
            recognition=level_diag,
        ),
        model=dict(
            name=spec.name,
            weights=str(spec.weights),
            sha256=spec.entry["weights_sha256"],
            thresholds=spec.thresholds_v1,
            processor=str(spec.processor),
        ),
        grid=dict(
            x0=grid.x0,
            y0=grid.y0,
            rows=grid.rows,
            cols=grid.cols,
            cell=grid.cell,
            tile=grid.tile,
            stride=grid.stride,
            windows=grid.windows,
            pixel_convention="vertex coordinates at pixel centres",
        ),
        outputs=dict(
            engineering_gpkg=str(eng), client_gpkg=str(client) if client.exists() else None
        ),
    )
    (out / "manifests/manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    if qgis and QGIS_PY.exists():
        res = subprocess.run(
            [str(QGIS_PY), str(ROOT / "scripts/digitization_qgis.py"), str(out)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        (out / "qgis_review/build_log.txt").write_text(res.stdout + "\n" + res.stderr)
    return out, qa


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--las")
    ap.add_argument("--bbox", nargs=4, type=float)
    ap.add_argument("--name", default="region")
    ap.add_argument("--regions", help="JSON list of {las, bbox, name} for batch processing")
    ap.add_argument("--model", default=None, help="registered model name (default: active)")
    ap.add_argument("--out", default=str(ROOT / "outputs/digitization"))
    ap.add_argument("--no-qgis", action="store_true")
    a = ap.parse_args()
    jobs = (
        json.loads(Path(a.regions).read_text())
        if a.regions
        else [dict(las=a.las, bbox=a.bbox, name=a.name)]
    )
    for j in jobs:
        out, qa = run_region(
            j["las"], j["bbox"], j.get("name", "region"), a.model, Path(a.out), qgis=not a.no_qgis
        )
        print(out)
        print(
            json.dumps(
                dict(
                    features=qa["features"],
                    z=qa["z"],
                    reference=qa["reference"],
                    performance=qa["performance"],
                ),
                indent=1,
                default=str,
            )
        )


if __name__ == "__main__":
    main()
