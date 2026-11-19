"""Digitization software tests on synthetic fixtures with exact answers.

These verify geometry/3D/IO behaviour only; they are not evidence of road recognition accuracy.
"""

import geopandas as gpd
import numpy as np
import pyogrio
from shapely.geometry import LineString

from roadmarking.digitization.centerline import fragment_centerline
from roadmarking.digitization.export import snap
from roadmarking.digitization.fragments import continuity_relations, extract_fragments
from roadmarking.digitization.qa import identity_checks
from roadmarking.digitization.raster_tiles import make_grid, pixel_to_xy, xy_to_pixel
from roadmarking.digitization.surface import Surface, lift_line


def test_grid_pixel_centres_round_trip():
    g = make_grid((100.0, 200.0, 150.0, 260.0))
    assert g.x0 == 100.0 and g.y0 == 260.0 and g.rows >= 1200 and g.cols >= 1000
    x, y = pixel_to_xy(g, [0, 10], [0, 20])
    assert np.allclose(x, [100.025, 101.025]) and np.allclose(y, [259.975, 259.475])
    r, c = xy_to_pixel(g, x, y)
    assert np.allclose(r, [0, 10]) and np.allclose(c, [0, 20])
    rr, cc = zip(*g.windows.values(), strict=True)
    assert min(rr) == 0 and sorted(set(np.diff(sorted(set(cc))))) == [384]


def test_straight_centerline_is_exact_and_full_length():
    m = np.zeros((40, 120), bool)
    m[19:22, 10:110] = True  # centre row 20, 100 px long
    cl = fragment_centerline(m, m)
    p = cl["refined"]
    assert np.abs(p[:, 0] - 20).max() < 0.15
    assert abs((p[:, 1].max() - p[:, 1].min()) - 99) < 2.5
    assert len(p) <= 4 and cl["flags"] == []


def test_curved_centerline_follows_arc():
    yy, xx = np.mgrid[0:140, 0:140]
    r = np.hypot(yy - 130, xx - 10)
    m = (np.abs(r - 100) <= 1.5) & (yy < 125) & (xx > 15)
    cl = fragment_centerline(m, m)
    d = np.hypot(cl["refined"][:, 0] - 130, cl["refined"][:, 1] - 10)
    assert np.abs(d - 100).max() < 0.3 and len(cl["refined"]) > 4


def test_short_dash_preserved_and_branch_flagged():
    d = np.zeros((20, 30), bool)
    d[9:12, 8:20] = True
    assert fragment_centerline(d, d) is not None
    t = np.zeros((60, 60), bool)
    t[29:32, 5:55] = True
    t[5:30, 29:32] = True
    assert "branching" in fragment_centerline(t, t)["flags"]


def inst(mask, cls, k):
    return dict(
        mask=mask,
        **{"class": cls},
        score=0.9,
        instance_id=k,
        members=[k],
        provenance=[dict(tile="T", query=k)],
    )


def test_fragments_split_dashes_bridge_only_sampling_holes():
    obs = np.ones((60, 200), bool)
    m = np.zeros_like(obs)
    m[10:13, 10:60] = True
    m[10:13, 70:120] = True  # observed 10 px gap: two dashes
    frags, _, _ = extract_fragments([inst(m, 1, 0)], obs)
    assert len(frags) == 2
    s = np.zeros_like(obs)
    s[30:33, 10:190] = True
    obs2 = obs.copy()
    obs2[30:33, 100] = False  # one-pixel sampling hole (unobserved)
    s2 = s & obs2
    frags, _, _ = extract_fragments([inst(s2, 0, 0)], obs2)
    assert len(frags) == 1 and frags[0]["sampling_bridge_px"] > 0
    assert frags[0]["mask"].sum() == s2.sum()  # no paint created


def test_large_nodata_gap_is_relation_not_merge_and_parallels_stay_apart():
    obs = np.ones((60, 300), bool)
    obs[:, 140:170] = False
    s = np.zeros_like(obs)
    s[20:23, 10:290] = True
    s &= obs
    p = np.zeros_like(obs)
    p[40:43, 10:130] = True  # parallel line 20 px away
    frags, _, _ = extract_fragments([inst(s, 0, 0), inst(p, 0, 1)], obs)
    assert len(frags) == 3
    rel = continuity_relations(frags, obs)
    assert any(r["kind"] == "possible_continuation" and r["certain"] is False for r in rel)
    obs3 = np.ones_like(obs)
    q = np.zeros_like(obs)
    q[20:23, 10:140] = True
    q[20:23, 170:290] = True
    frags3, _, _ = extract_fragments([inst(q, 0, 0)], obs3)
    assert {r["kind"] for r in continuity_relations(frags3, obs3)} == {"observed_break"}


def test_surface_plane_robust_to_elevated_objects_and_missing_support():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 10, (20000, 2))
    z = 100 + 0.02 * xy[:, 0] - 0.03 * xy[:, 1] + rng.normal(0, 0.005, len(xy))
    car = (np.abs(xy[:, 0] - 5) < 1) & (np.abs(xy[:, 1] - 5) < 1) & (rng.random(len(xy)) < 0.3)
    z[car] += 1.5
    s = Surface(np.column_stack([xy, z]))
    r = s.z_at(5.0, 5.0)
    assert r["ok"] and abs(r["z"] - (100 + 0.1 - 0.15)) < 0.01
    far = s.z_at(50.0, 50.0)
    assert not far["ok"] and np.isnan(far["z"])
    line = np.array([[2, 2], [2.4, 2], [2.8, 2], [30, 2], [31, 2]], float)
    zz, interp, _ = lift_line(s, line)
    assert np.isfinite(zz[:3]).all() and np.isnan(zz[3:]).all() and not interp.any()


def test_geopackage_3d_round_trip(tmp_path):
    g = LineString([(1.0, 2.0, 100.25), (3.0, 4.0, 100.5)])
    p = tmp_path / "t.gpkg"
    pyogrio.write_dataframe(
        gpd.GeoDataFrame({"mark_id": [1]}, geometry=[g], crs="EPSG:3763"), p, layer="solid_lines_3d"
    )
    back = pyogrio.read_dataframe(p, layer="solid_lines_3d")
    assert back.geometry.has_z.all() and np.allclose(
        np.asarray(back.geometry[0].coords), np.asarray(g.coords)
    )
    assert pyogrio.read_info(p, layer="solid_lines_3d")["crs"] == "EPSG:3763"


def test_duplicate_detection_and_snapping():
    a = dict(mark_id=1, class_id=0, length_m=10.0, geometry_2d=LineString([(0, 0), (10, 0)]))
    b = dict(
        mark_id=2, class_id=0, length_m=9.0, geometry_2d=LineString([(0.5, 0.02), (9.5, 0.02)])
    )
    assert identity_checks([a, b])["duplicates"] == [(1, 2)]
    assert snap(0.16, (0.15, 0.30)) == 0.15 and snap(0.6, (0.15, 0.30)) is None


def test_overlapping_tile_observations_reconcile_to_one_fragment():
    obs = np.ones((40, 300), bool)
    a = np.zeros_like(obs)
    a[20:23, 10:200] = True  # seen by crop A
    b = np.zeros_like(obs)
    b[20:23, 120:290] = True  # same line seen by crop B, overlapping
    frags, _, _ = extract_fragments([inst(a, 0, 0), inst(b, 0, 1)], obs)
    assert len(frags) == 1 and frags[0]["duplicate_observations"] == 1
    assert frags[0]["instance_ids"] == [0, 1] and frags[0]["mask"].sum() == (a | b).sum()


def test_paint_evidence_joins_only_bright_observed_gaps():
    from roadmarking.digitization.fragments import bright_paint, merge_evidence_links

    obs = np.ones((60, 200), bool)
    gray = np.full(obs.shape, 60, np.uint8)
    m = np.zeros_like(obs)
    m[30:33, 10:95] = True
    m[30:33, 99:190] = True  # 4 px mask gap
    gray[30:33, 10:190] = 230  # LiDAR shows paint across the gap
    frags, _, _ = extract_fragments([inst(m, 0, 0)], obs, sampling_gap_px=1)
    assert len(frags) == 2
    rel = continuity_relations(frags, obs, bright=bright_paint(gray, obs))
    assert [r["kind"] for r in rel] == ["paint_evidence_continuation"]
    merged = merge_evidence_links(frags, rel, obs.shape)
    assert len(merged) == 1 and merged[0]["mask"].sum() == m.sum()  # no paint created
    gray[30:33, 95:99] = 60  # gap observed but not paint-bright
    rel = continuity_relations(frags, obs, bright=bright_paint(gray, obs))
    assert [r["kind"] for r in rel] == ["observed_break"]
    assert len(merge_evidence_links(frags, rel, obs.shape)) == 2


def test_jagged_paint_gives_smooth_straight_centerline():
    from roadmarking.digitization.qa import smoothness_mm

    rng = np.random.default_rng(3)
    m = np.zeros((40, 300), bool)
    for c in range(10, 290):
        top = 19 + rng.integers(-1, 2)
        bot = 22 + rng.integers(-1, 2)
        m[top:bot, c] = True
    m[(np.add.outer(np.arange(40), np.arange(300)) % 9) == 0] = False  # diagonal scan stripes
    cl = fragment_centerline(m, m)
    p = cl["refined"]
    assert np.abs(p[:, 0] - 20.0).max() < 0.7
    j = smoothness_mm(p[:, ::-1] * 0.05)
    raw = smoothness_mm(cl["centroid"][:, ::-1] * 0.05)
    assert j["jitter_max_mm"] < 3.0 and raw["jitter_max_mm"] > 5 * j["jitter_max_mm"]


def _bridge(lines, nodata_x=None, dark_x=None):
    from roadmarking.digitization.gap_bridging import bridge_solids

    def observed_at(xy):
        xy = np.asarray(xy)
        return (
            ~((xy[:, 0] > nodata_x[0]) & (xy[:, 0] < nodata_x[1]))
            if nodata_x
            else np.ones(len(xy), bool)
        )

    def bright_at(xy):
        xy = np.asarray(xy)
        return (
            ~((xy[:, 0] > dark_x[0]) & (xy[:, 0] < dark_x[1])) if dark_x else np.ones(len(xy), bool)
        )

    return bridge_solids(lines, observed_at, bright_at)


def test_occluded_gap_is_bridged_and_marked_inferred():
    a = LineString([(0, 0), (20, 0.2)])
    b = LineString([(32, 0.32), (60, 0.6)])  # same line, 12 m hidden by a truck
    chains, audit = _bridge([a, b], nodata_x=(20.2, 31.8))
    assert len(chains) == 1 and sorted(chains[0]["members"]) == [0, 1]
    inf = chains[0]["inferred"][0]
    assert abs(inf.length - 12.0) < 0.1 and LineString(chains[0]["coords"]).length > 59.9
    assert any(r["accepted"] and r["evidence"] == "occlusion" for r in audit)


def test_observed_unpainted_gap_and_parallel_line_are_not_bridged():
    a = LineString([(0, 0), (20, 0)])
    b = LineString([(32, 0), (60, 0)])
    chains, audit = _bridge([a, b], dark_x=(20.2, 31.8))  # road seen, no paint: real break
    assert len(chains) == 2 and audit[0]["reason"] == "observed_break"
    c = LineString([(32, 1.5), (60, 1.5)])  # different lane line, 1.5 m sideways
    chains, _ = _bridge([a, c], nodata_x=(20.2, 31.8))
    assert len(chains) == 2


def test_curved_bridge_is_tangent_continuous():
    r = 400.0  # highway-scale curve; 15 m occlusion gap
    th1 = np.linspace(0, 0.075, 30)
    th2 = np.linspace(0.1125, 0.2, 30)
    a = LineString(np.column_stack([r * np.sin(th1), r - r * np.cos(th1)]))
    b = LineString(np.column_stack([r * np.sin(th2), r - r * np.cos(th2)]))
    chains, _ = _bridge([a, b], nodata_x=(r * np.sin(0.075) + 0.1, r * np.sin(0.1125) - 0.1))
    assert len(chains) == 1
    p = chains[0]["inferred"][0]
    d = np.abs(np.hypot(np.asarray(p.coords)[:, 0], np.asarray(p.coords)[:, 1] - r) - r)
    assert d.max() < 0.05  # bridge follows the arc


def test_short_scan_gap_bridged_unless_unpainted_road_seen():
    a = LineString([(0, 0), (10, 0)])
    b = LineString([(10.15, 0), (20, 0)])  # 15 cm gap, observed both sides
    chains, _ = _bridge([a, b])
    assert len(chains) == 1
    chains, audit = _bridge([a, b], dark_x=(10.01, 10.14))
    assert len(chains) == 2 and audit[0]["reason"] == "observed_break"


def test_partly_hidden_partly_painted_gap_is_bridged_with_evidence():
    a = LineString([(0, 0), (10, 0)])
    b = LineString([(12, 0), (20, 0)])  # 2 m gap: first half hidden, second half bright paint
    chains, audit = _bridge([a, b], nodata_x=(10.0, 11.0))
    acc = [r for r in audit if r["accepted"]]
    assert len(chains) == 1 and acc[0]["evidence"] == "occlusion_and_visible_paint"


def test_tiny_gap_with_small_sideways_step_is_bridged_but_offset_line_is_not():
    a = LineString([(0, 0), (10, 0)])
    b = LineString([(10.1, 0.03), (20, 0.03)])  # 10 cm gap, 3 cm step: same line
    assert len(_bridge([a, b])[0]) == 1
    c = LineString([(10.1, 0.30), (20, 0.30)])  # 30 cm sideways: a different line
    assert len(_bridge([a, c])[0]) == 2


def test_dash_rhythm_misread_as_solid_is_never_chained():
    pieces = [LineString([(x, 0), (x + 4.0, 0)]) for x in np.arange(0, 70, 14.0)]  # 4 m / 10 m
    chains, _ = _bridge(pieces, nodata_x=(-1, 100))  # gaps unobserved (sparse scan)
    assert len(chains) == len(pieces)


def _trace_fixture(intervals, paint_intervals, hidden=None, *, road_support=None):
    from roadmarking.digitization.line_tracing import trace_lines
    from roadmarking.digitization.raster_tiles import Grid

    grid = Grid(0, 1.025, 41, 1400)
    observed = np.ones((grid.rows, grid.cols), bool)
    if hidden:
        observed[:, round(hidden[0] * 20) : round(hidden[1] * 20)] = False
    paint = np.zeros_like(observed)
    for a, b in paint_intervals:
        paint[19:22, round(a * 20) : round(b * 20)] = True
    pieces = [
        dict(
            geometry_2d=LineString([(a, 0), (b, 0)]),
            length_m=b - a,
            class_id=cls,
            mark_id=i,
            flags=[],
            tiles=[],
            queries=[],
            instance_ids=[],
        )
        for i, (a, b, cls) in enumerate(intervals, 1)
    ]
    return trace_lines(
        pieces, grid, paint, observed, np.zeros_like(paint), road_support=road_support
    )[0]


def test_trace_duplicate_classes_and_visible_missing_paint():
    result = _trace_fixture([(1, 10, 0), (2, 9, 1), (14, 25, 0)], [(1, 25)])
    assert len(result) == 1
    line = result[0]
    assert line["class_id"] == 0 and line["length_m"] > 23
    assert line["paint_traced_m"] > 23 and not line["inferred_segments"]
    assert line["jitter_max_mm"] < 10


def test_trace_hidden_and_dark_gaps():
    pieces = [(1, 10, 0), (14, 25, 0)]
    paint = [(1, 10), (14, 25)]
    hidden = _trace_fixture(pieces, paint, hidden=(10, 14))
    assert len(hidden) == 1 and hidden[0]["inferred_labels"] == ["INFERRED_UNDER_OCCLUSION"]
    dark = _trace_fixture(pieces, paint)
    assert len(dark) == 2 and all(not line["inferred_segments"] for line in dark)


def test_trace_dash_rhythms():
    for size, gap, pattern in [(4, 10, "normal"), (10, 4, "aviso")]:
        intervals = [(1, 1 + size), (1 + size + gap, 1 + 2 * size + gap)]
        result = _trace_fixture([(a, b, 0) for a, b in intervals], intervals)
        assert len(result) == 2
        assert all(f["class_id"] == 1 and f["dash_pattern"] == pattern for f in result)


def test_other_marking_classifier_and_surface_support():
    from shapely.geometry import box

    from roadmarking.digitization.other_marks import ShapeClassifier, descriptor, extract_other
    from roadmarking.digitization.raster_tiles import Grid

    class FlatRoad:
        def z_at(self, x, y):
            return dict(ok=x < 4, z=10.0)

    grid = Grid(0, 5, 100, 160)
    image = np.zeros((100, 160), np.uint8)
    image[30:50, 20:50] = 220
    image[30:50, 100:130] = 220
    sample = descriptor(box(1, 2.5, 2.5, 3.5), [])
    model = ShapeClassifier([(sample, ("M20", "Bp")), (sample + 0.02, ("M20", "Bp"))])
    rows = extract_other(
        image, np.ones_like(image, bool), grid, [], FlatRoad(), model, max_line_distance_m=None
    )
    assert len(rows) == 1
    assert rows[0]["COD_RST"] == "M20" and rows[0]["geometry_3d"].has_z


def test_solid_extension_resumes_after_sparse_observed_paint():
    result = _trace_fixture([(1, 8, 0)], [(1, 8), (8.25, 17), (18, 22)])
    assert len(result) == 1
    assert 16.8 < result[0]["geometry_2d"].bounds[2] < 17.2
    assert not result[0]["inferred_segments"]


def test_obstruction_returns_do_not_split_a_solid_line():
    pieces = [(1, 10, 0), (14, 25, 0)]
    paint = [(1, 10), (14, 25)]

    def road_support(points):
        return (points[:, 0] < 10) | (points[:, 0] >= 14)

    result = _trace_fixture(pieces, paint, road_support=road_support)
    assert len(result) == 1
    assert result[0]["inferred_labels"] == ["INFERRED_UNDER_OCCLUSION"]
    assert result[0]["inferred_length_m"] > 3.8


def test_converging_stripes_stay_separate():
    from roadmarking.digitization.line_tracing import cluster

    straight = LineString([(0, 0), (20, 0)])
    taper = LineString([(0, 1), (19, 0), (20, 0)])
    assert len(cluster([straight, taper])) == 2


def test_overlapping_fragments_on_one_stripe_merge():
    from roadmarking.digitization.line_tracing import cluster

    paths = cluster([LineString([(0, 0), (10, 0)]), LineString([(6, 0), (18, 0)])])
    assert len(paths) == 1
    assert paths[0][0].length == 18


def test_junction_shared_approach_is_drawn_once():
    from roadmarking.digitization.line_tracing import trim_shared_approach

    trunk = LineString([(0, 0), (20, 0)])
    branch = LineString([(0, 0), (6, 0), (12, 2)])
    trimmed = trim_shared_approach(branch, [trunk])
    assert trimmed.coords[0][0] >= 6
    assert trunk.distance(trimmed.boundary.geoms[0]) == 0
    assert trimmed.coords[-1] == branch.coords[-1]
    parallel = LineString([(0, 0.3), (12, 0.3)])
    assert trim_shared_approach(parallel, [trunk]).equals(parallel)


def test_fuse_sources_fills_cells_from_other_acquisition():
    from roadmarking.digitization.raster_tiles import fuse_sources

    a = np.zeros((20, 20), np.float32)
    b = np.ones((20, 20), np.float32)
    ca = np.zeros((20, 20), np.uint32)
    cb = np.zeros((20, 20), np.uint32)
    ca[:, :10] = 5  # file A scanned the left half densely
    cb[:, 5:] = 1  # file B the right part, sparsely
    fused, observed, source = fuse_sources([a, b], [ca, cb], density_window=3)
    assert observed.all()
    assert (source[:, :4] == 0).all() and (source[:, 12:] == 1).all()
    assert (fused[:, 12:] == 1).all()


def test_surface_band_keeps_road_under_bridge_and_deck_separately():
    from roadmarking.digitization.raster_tiles import surface_band

    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 4000)
    y = rng.uniform(0, 10, 4000)
    z = np.where(rng.random(4000) < 0.5, 100.0, 107.0) + rng.normal(0, 0.01, 4000)
    low, height = surface_band(x, y, z)
    assert low[z < 101].all() and not low[z > 106].any()
    assert abs(np.nanmedian(height.z) - 100) < 0.05
    top, _ = surface_band(x, y, z, top=True)
    assert top[z > 106].all() and not top[z < 101].any()


def test_hairpin_skeleton_is_split():
    from roadmarking.digitization.line_tracing import split_reversals

    hairpin = LineString([(0, 0), (10, 0), (10.1, 0.15), (10, 0.3), (0, 0.3)])
    parts = split_reversals(hairpin)
    assert len(parts) == 2 and all(p.length > 8 for p in parts)
    assert len(split_reversals(LineString([(0, 0), (20, 0.5)]))) == 1


def test_converging_lines_are_drawn_once_and_snapped():
    from roadmarking.digitization.line_tracing import drop_overlaps

    def feat(coords):
        g = LineString(coords)
        return dict(geometry_2d=g, length_m=g.length, flags=[])

    trunk = feat([(0, 0), (40, 0)])
    branch = feat([(0, 0.05), (15, 0.05), (30, 3)])
    kept, dropped, trimmed = drop_overlaps([branch, trunk])
    assert dropped == 0 and trimmed == 1
    joined = next(f for f in kept if "joined_to_merging_line" in f["flags"])
    start = joined["geometry_2d"].coords[0]
    assert trunk["geometry_2d"].distance(shapely_point(start)) < 1e-6
    assert joined["jitter_max_mm"] < 30
    duplicate = feat([(5, 0.03), (12, 0.03)])
    assert drop_overlaps([duplicate, trunk])[1] == 1
    crossing = feat([(20, -10), (20, 10)])
    assert drop_overlaps([crossing, trunk])[2] == 0


def shapely_point(xy):
    from shapely.geometry import Point

    return Point(xy[:2])


def test_short_visible_gaps_make_dashes():
    intervals = [(1, 6), (7.5, 12.5), (14, 19), (20.5, 25.5)]
    result = _trace_fixture([(a, b, 0) for a, b in intervals], intervals)
    assert len(result) == 4 and all(f["class_id"] == 1 for f in result)


def _pixelate(poly, cell=0.05, noise=0.0, seed=0):
    """Raster outline of ``poly`` (like a paint blob from the intensity image)."""
    from rasterio.features import rasterize, shapes
    from rasterio.transform import from_origin
    from shapely.geometry import shape

    x0, y0, x1, y1 = poly.bounds
    tr = from_origin(x0 - 0.5, y1 + 0.5, cell, cell)
    h, w = int((y1 - y0 + 1) / cell) + 1, int((x1 - x0 + 1) / cell) + 1
    m = rasterize([poly], out_shape=(h, w), transform=tr).astype(bool)
    if noise:
        rng = np.random.default_rng(seed)
        m ^= rng.random(m.shape) < noise
    blobs = [shape(g) for g, v in shapes(m.astype(np.uint8), mask=m, transform=tr)]
    return max(blobs, key=lambda g: g.area)


def test_straight_edge_fit_gives_clean_rectangle():
    from shapely import affinity
    from shapely.geometry import box

    from roadmarking.digitization.polygon_fit import fit_straight, iou

    bar = affinity.rotate(box(0, 0, 3.3, 0.55), 23, origin="centroid")
    blob = _pixelate(bar)
    assert len(blob.exterior.coords) > 20  # stair-stepped pixel outline
    poly, corners, score = fit_straight(blob)
    assert corners == 4 and score > 0.85 and iou(poly, bar) > 0.9


def test_arrow_fitted_with_client_shape():
    from shapely import affinity
    from shapely.geometry import Polygon

    from roadmarking.digitization.polygon_fit import ShapeLibrary, fit_template, iou

    arrow = Polygon(
        [(0, -0.1), (5.5, -0.1), (5.5, -0.5), (7.5, 0), (5.5, 0.5), (5.5, 0.1), (0, 0.1)]
    )
    library = ShapeLibrary(
        [(affinity.rotate(arrow, 70), "M15"), (affinity.scale(arrow, 1.0, 3.0), "M16a")]
    )
    placed = affinity.translate(affinity.rotate(arrow, -35, origin="centroid"), 100, 200)
    poly, code, score = fit_template(_pixelate(placed), library)
    assert code == "M15" and score > 0.7
    assert len(poly.exterior.coords) == len(arrow.exterior.coords)
    assert iou(poly, placed) > 0.85


def test_frame_is_one_polygon_with_a_hole():
    from shapely import affinity
    from shapely.geometry import box

    from roadmarking.digitization.polygon_fit import fit_straight

    frame = affinity.rotate(box(0, 0, 4, 1.4).difference(box(0.2, 0.2, 3.8, 1.2)), 15)
    poly, corners, score = fit_straight(_pixelate(frame))
    assert corners == 4 and len(poly.interiors) == 1
    assert len(poly.interiors[0].coords) - 1 == 4
    assert score > 0.8


def test_converging_child_joins_parent_with_shared_vertex():
    from shapely.geometry import Point

    from roadmarking.digitization.junctions import apply_joins

    parent = LineString([(0, 0), (60, 0)])
    # child converges at ~7 degrees and stops 0.35 m short of the parent
    child = LineString([(55, 3.4), (30, 0.35)])
    parallel = LineString([(10, 0.5), (40, 0.5)])
    feats = [
        dict(geometry_2d=parent, class_id=0, flags=[], length_m=60.0),
        dict(geometry_2d=child, class_id=0, flags=[], length_m=child.length),
        dict(geometry_2d=parallel, class_id=0, flags=[], length_m=30.0),
    ]
    n = apply_joins(feats, paint_at=lambda xy: np.ones(len(xy), bool))
    assert n == 1
    joined = feats[1]["geometry_2d"]
    tip = Point(joined.coords[-1])
    assert parent.distance(tip) < 1e-6
    assert any(Point(c).distance(tip) < 1e-6 for c in feats[0]["geometry_2d"].coords)
    assert feats[2]["geometry_2d"].equals(parallel)
    assert "joined_at_junction" in feats[1]["flags"]


def test_long_junction_join_needs_paint():
    from roadmarking.digitization.junctions import apply_joins

    parent = LineString([(0, 0), (80, 0)])
    child = LineString([(70, 2.0), (40, 0.5)])  # ~10 m to go at a shallow angle
    feats = [
        dict(geometry_2d=parent, class_id=0, flags=[], length_m=80.0),
        dict(geometry_2d=child, class_id=0, flags=[], length_m=child.length),
    ]
    assert apply_joins(feats, paint_at=lambda xy: np.zeros(len(xy), bool)) == 0


def test_touching_hatch_bars_split_into_one_polygon_per_bar():
    import shapely
    from shapely.geometry import Polygon, box

    from roadmarking.digitization.other_marks import split_touching_bars

    bars = [
        Polygon([(x, 0), (x + 0.45, 0), (x + 1.45, 2), (x + 1.0, 2)]) for x in (0, 1.5, 3.0, 4.5)
    ]
    border = box(-0.2, -0.2, 6.2, 0.0)  # an untraced border line touching every bar
    comb = shapely.union_all([*bars, border])
    parts = split_touching_bars(comb)
    assert parts is not None and len(parts) == 4
    for bar in bars:
        best = max(parts, key=lambda p: p.intersection(bar).area)
        assert best.intersection(bar).area / bar.area > 0.8
    # One solid mark (a band, an arrow) is never split.
    assert split_touching_bars(box(0, 0, 4, 0.5)) is None
