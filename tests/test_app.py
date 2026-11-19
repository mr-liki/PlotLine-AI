"""Plotline-AI application: validation, CRS handling, block stitching, GeoPackage export and
verification, cancellation, GUI start-up and (opt-in) a real end-to-end run."""

import os
import threading
from pathlib import Path

import laspy
import numpy as np
import pytest
import shapely
from shapely.affinity import translate as shapely_translate
from shapely.geometry import LineString, Polygon

from roadmarking.app import pipeline as P  # noqa: N812

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "outputs/app_pilot/pilot_sample_240m.laz"


def _tiny_las(path, n=2000, crs=True):
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [-20000, 168000, 0]
    if crs:
        import pyproj

        header.add_crs(pyproj.CRS.from_epsg(3763))
    las = laspy.LasData(header)
    rng = np.random.default_rng(0)
    las.x = -20000 + rng.uniform(0, 20, n)
    las.y = 168000 + rng.uniform(0, 20, n)
    las.z = 250 + rng.uniform(0, 0.05, n)
    las.intensity = rng.integers(100, 5000, n).astype(np.uint16)
    las.write(path)
    return path


def test_validation_rejects_bad_inputs(tmp_path):
    out = tmp_path / "out.gpkg"
    with pytest.raises(P.JobError, match="not found"):
        P.validate_input(tmp_path / "missing.laz", out, tmp_path)
    txt = tmp_path / "points.txt"
    txt.write_text("1 2 3")
    with pytest.raises(P.JobError, match=r"\.las or \.laz"):
        P.validate_input(txt, out, tmp_path)
    bad = tmp_path / "corrupt.laz"
    bad.write_bytes(os.urandom(4096))
    with pytest.raises(P.JobError, match="cannot be opened"):
        P.validate_input(bad, out, tmp_path)
    good = _tiny_las(tmp_path / "tiny.las")
    with pytest.raises(P.JobError, match=r"\.gpkg"):
        P.validate_input(good, tmp_path / "out.shp", tmp_path)
    hdr, crs, _text, warning = P.validate_input(good, out, tmp_path)
    assert hdr["points"] == 2000 and crs == "EPSG:3763" and warning is None


def test_crs_never_guessed():
    import pyproj

    assert P.resolve_crs(None)[0] is None and "no coordinate" in P.resolve_crs(None)[2]
    compound = pyproj.CRS.from_user_input("EPSG:3763+3855")
    assert P.resolve_crs(compound)[0] == "EPSG:3763"
    other, text, _ = P.resolve_crs(pyproj.CRS.from_epsg(32629))
    assert other != "EPSG:3763" and "UTM" in text


def _line(coords, cls=0, block=0):
    g = LineString(coords)
    return dict(
        geometry_2d=LineString(np.asarray(coords)[:, :2]),
        geometry_3d=g,
        class_id=cls,
        length_m=g.length,
        flags=[],
        block=block,
        mean_z=float(np.mean(np.asarray(coords)[:, 2])),
    )


def test_lines_cut_at_block_border_are_stitched():
    a = _line([(0, 0, 10), (96, 0.5, 10.2)], block=0)
    b = _line([(96.02, 0.5, 10.2), (150, 1.0, 10.5)], block=1)
    dash_a = _line([(94, 5, 10), (96, 5, 10)], cls=1, block=0)
    dash_b = _line([(96, 5, 10), (98, 5, 10)], cls=1, block=1)
    other_class = _line([(96.01, 0.52, 10.2), (100, 0.6, 10.2)], cls=1, block=2)
    out = P.stitch_lines([a, b, dash_a, dash_b, other_class])
    solids = [f for f in out if f["class_id"] == 0]
    assert len(solids) == 1 and abs(solids[0]["length_m"] - 150.0) < 0.1
    assert sum(f["class_id"] == 1 for f in out) == 2  # one dash rebuilt + unrelated piece


def _rows():
    line = LineString([(-19990, 168300, 268.1), (-19980, 168301, 268.2)])
    poly = Polygon([(-19985, 168310, 268), (-19983, 168310, 268), (-19983, 168311, 268)])
    lines = [
        dict(
            geometry_3d=line,
            class_id=1,
            width_m=0.15,
            length_m=line.length,
            gap_next_m=10.0,
            mark_id=1,
            mean_z=268.15,
            dash_pattern="normal",
            flags=[],
        )
    ]
    polys = [dict(geometry=Polygon(np.asarray(poly.exterior.coords)[:, :2]), geometry_3d=poly)]
    polys[0].update(COD_RST="M21", COD_JAE=None, review_flags="")
    return lines, polys


def test_export_matches_client_layout_and_verifies(tmp_path):
    lines, polys = _rows()
    line_rows, _ = P.client_line_rows(lines, {"Linha descontínua": "LBT"})
    poly_rows = P.client_poly_rows(polys)
    out = tmp_path / "result.gpkg"
    P.write_output(out, line_rows, poly_rows, "EPSG:3763", dict(model="test"))
    bounds = ([-20000, 168000, 200], [-19900, 168400, 300])
    assert P.verify_output(out, "EPSG:3763", bounds, 1, 1) == []
    assert line_rows[0]["DESIGNACAO"] == "Linha descontínua"
    assert line_rows[0]["COMENTARIO"] == "Ativo novo"
    far = ([0, 0, 0], [1, 1, 1])  # wrong footprint: the check must fail
    assert any("outside" in p for p in P.verify_output(out, "EPSG:3763", far, 1, 1))


def test_cancel_leaves_no_output(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    src = _tiny_las(tmp_path / "tiny.las")
    out = tmp_path / "tiny_digitized.gpkg"
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(P.Cancelled):
        P.run_job(src, out, cancel=cancel)
    assert not out.exists()
    assert not list((tmp_path / "appdata/Plotline-AI/jobs").glob("*"))  # temp work removed


def test_existing_output_needs_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    src = _tiny_las(tmp_path / "tiny.las")
    out = tmp_path / "exists.gpkg"
    out.write_bytes(b"keep")
    with pytest.raises(P.JobError, match="already exists"):
        P.run_job(src, out)
    assert out.read_bytes() == b"keep"


def test_gui_starts_and_enables_start_only_with_valid_paths(tmp_path):
    tk = pytest.importorskip("tkinter")
    from roadmarking.app.gui import App

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    app = App(root)
    assert str(app.start_btn["state"]) == "disabled"
    src = _tiny_las(tmp_path / "tiny.las")
    app.input_var.set(str(src))
    app.output_var.set(str(tmp_path / "tiny_digitized.gpkg"))
    app.refresh()
    assert str(app.start_btn["state"]) == "normal"
    root.destroy()


@pytest.mark.skipif(
    not (SAMPLE.exists() and os.environ.get("PLOTLINE_E2E") == "1"),
    reason="set PLOTLINE_E2E=1 (needs the real sample LAZ and a CUDA GPU; ~3 min)",
)
def test_end_to_end_real_sample(tmp_path, monkeypatch):
    import pyogrio

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    out = tmp_path / "sample_digitized.gpkg"
    stages = []
    res = P.run_job(SAMPLE, out, lambda s, f, m: stages.append(s))
    assert out.exists() and res.lines > 0
    assert sorted(set(stages)) == sorted(s[0] for s in P.STAGES)
    assert [p.name for p in tmp_path.iterdir() if p.name != "appdata"] == [out.name]
    layers = dict(pyogrio.list_layers(out))
    assert layers["Highway"] == "MultiLineString Z" and "Other Marks" in layers
    # Regression guard: none of the mistakes found in review may come back on the sample
    # (unjoined vehicle gaps, ends past paint, junction polygons, overlaps, excess vertices).
    import json

    checks = json.loads(Path(res.report).read_text())["self_check"]
    assert all(v["count"] == 0 for v in checks.values()), checks


def test_verification_releases_the_file(tmp_path):
    lines, polys = _rows()
    out = tmp_path / "result.gpkg"
    P.write_output(
        out, P.client_line_rows(lines, {})[0], P.client_poly_rows(polys), "EPSG:3763", {}
    )
    P.verify_output(out, "EPSG:3763", ([-20000, 168000, 200], [-19900, 168400, 300]), 1, 1)
    out.unlink()  # fails on Windows if a handle is still open
    assert not out.exists()


def test_simplify_3d_keeps_shape_with_few_vertices():
    straight = np.column_stack([np.linspace(0, 100, 101), np.zeros(101), np.linspace(10, 11, 101)])
    assert len(P.simplify_3d(straight, [])) == 2
    t = np.linspace(0, np.pi / 4, 400)  # curve of radius 200 m, 157 m long
    arc = np.column_stack([200 * np.sin(t), 200 * (1 - np.cos(t)), 50 + 0.01 * t])
    s = P.simplify_3d(arc, [150])
    assert 10 < len(s) < 60 and any(np.allclose(v, arc[150]) for v in s)
    line = LineString(s)
    assert max(line.distance(LineString(arc[k : k + 2])) for k in range(0, 399, 7)) < 0.011


def test_line_end_trimmed_to_paint_but_junction_end_kept():
    from roadmarking.digitization.raster_tiles import Grid

    grid = Grid(0, 1.025, 41, 600)
    paint = np.zeros((41, 600), bool)
    paint[19:22, 20:400] = True  # paint from x = 1 m to 20 m
    line = LineString([(1, 0), (25, 0)])  # runs 5 m past the paint
    trimmed = P.trim_to_paint(line, paint, grid)
    assert abs(trimmed.coords[-1][0] - 20) < 0.15
    junction = LineString([(25, -3), (25, 3)])  # end lies on another line: keep it
    assert P.trim_to_paint(line, paint, grid, [junction]).coords[-1][0] == 25


def test_merged_line_paint_polygon_dropped_hatch_bar_kept():
    from shapely.geometry import box as sbox

    lines = [LineString([(0, 0), (60, 0)]), LineString([(0, 0.6), (60, 0.6)])]
    merged = dict(geometry=sbox(20, -0.1, 30, 0.7))  # paint of two converging lines
    bar = dict(geometry=Polygon([(40, 0), (41.5, 2), (42, 2), (40.5, 0)]))  # hatch bar
    kept, dropped = P.drop_line_paint_polygons([merged, bar], lines)
    assert dropped == 1 and kept == [bar]


def _solid(x0, x1, block, n=20):
    g = LineString([(x, 0.0, 100.0) for x in np.linspace(x0, x1, n)])
    return dict(
        class_id=0,
        geometry_3d=g,
        geometry_2d=LineString(np.asarray(g.coords)[:, :2]),
        length_m=x1 - x0,
        mean_z=100.0,
        flags=[],
        block=block,
    )


def test_line_joined_across_vehicle_gap_but_not_across_visible_break():
    # Road seen everywhere along y=0 except x in [40, 52] (a vehicle shadow).
    xs = np.arange(0, 100, P.COVER_CELL_M) + 0.01
    seen = np.unique(
        P.cell_keys(xs[(xs < 40) | (xs > 52)], np.zeros(np.sum((xs < 40) | (xs > 52))))
    )
    paint = seen
    hidden = [_solid(0, 40, 0), _solid(52, 99, 1)]
    out, n = P.bridge_occlusions(hidden, seen, paint)
    assert n == 1 and len(out) == 1
    g = out[0]["geometry_3d"]
    assert g.has_z and abs(g.length - 99) < 0.01
    assert any("INFERRED_UNDER_OCCLUSION" in f for f in out[0]["flags"])
    # The same gap with visible unpainted road is a real break: never joined.
    all_seen = np.unique(P.cell_keys(xs, np.zeros(len(xs))))
    out, n = P.bridge_occlusions([_solid(0, 40, 0), _solid(52, 99, 1)], all_seen, paint)
    assert n == 0 and len(out) == 2
    # Offset (different lane) lines are not joined.
    other = _solid(52, 99, 1)
    other["geometry_3d"] = LineString([(x, 1.0, 100.0) for x in np.linspace(52, 99, 20)])
    other["geometry_2d"] = LineString([(x, 1.0) for x in np.linspace(52, 99, 20)])
    out, n = P.bridge_occlusions([_solid(0, 40, 0), other], seen, paint)
    assert n == 0


def _keys(xs, ys):
    return np.unique(P.cell_keys(np.asarray(xs, float), np.asarray(ys, float)))


def test_self_check_flags_each_known_mistake_and_passes_clean_output():
    xs = np.arange(-10, 110, P.COVER_CELL_M) + 0.01
    road = (xs < 40) | (xs > 52)  # vehicle shadow at x 40..52
    seen = _keys(xs[road], np.zeros(road.sum()))
    paint = _keys(
        xs[road & (xs >= 0) & (xs <= 99)], np.zeros((road & (xs >= 0) & (xs <= 99)).sum())
    )
    clean = [_solid(0, 99, 0, n=2)]  # straight line: two vertices
    res = P.self_check(clean, [], seen, paint)
    assert all(v["count"] == 0 for v in res.values()), res

    # Mistake 1: line left broken under a vehicle.
    res = P.self_check([_solid(0, 40, 0), _solid(52, 99, 1)], [], seen, paint)
    assert res["unjoined_hidden_gaps"]["count"] == 1
    assert abs(res["unjoined_hidden_gaps"]["where"][0][0] - 46) < 1

    # Mistake 2: line running past the end of its paint (visible road beyond x = 99).
    res = P.self_check([_solid(0, 105, 0)], [], seen, paint)
    assert res["ends_past_paint"]["count"] == 1

    # Mistake 3: merged junction paint exported as a polygon along a long line.
    poly = dict(geometry=Polygon([(20, -0.1), (24, -0.1), (24, 0.1), (20, 0.1)]))
    res = P.self_check(clean, [poly], seen, paint)
    assert res["junction_paint_polygons"]["count"] == 1

    # Mistake 4: the same line drawn twice.
    res = P.self_check([_solid(0, 99, 0), _solid(10, 60, 1)], [], seen, paint)
    assert res["overlapping_lines"]["count"] >= 1

    # Mistake 5: a straight line exported with many unnecessary vertices.
    dense = _solid(0, 99, 0)
    g = LineString([(x, 0.0, 100.0) for x in np.linspace(0, 99, 400)])
    dense.update(geometry_3d=g, geometry_2d=LineString(np.asarray(g.coords)[:, :2]))
    res = P.self_check([dense], [], seen, paint)
    assert res["excess_vertices"]["count"] == 1

    # Mistake 6: a line on visible road with no paint under it.
    ghost = _solid(0, 30, 0, n=2)
    for k in ("geometry_3d", "geometry_2d"):
        ghost[k] = shapely_translate(ghost[k], yoff=3.0)
    res = P.self_check([ghost], [], np.union1d(seen, _keys(xs, np.full(len(xs), 3.0))), paint)
    assert res["lines_without_visible_paint"]["count"] == 1

    # Mistake 7: a short line traced across hatch bars.
    bars = [
        dict(geometry=Polygon([(x, -1), (x + 0.4, -1), (x + 1.4, 1), (x + 1.0, 1)]))
        for x in (20, 22, 24)
    ]
    across = _solid(20, 25, 0, n=2)
    res = P.self_check([_solid(0, 99, 0, n=2), across], bars, seen, paint)
    assert res["lines_through_marks"]["count"] == 1

    # Mistake 8: one polygon drawn around a hatch zone (bars and the road between them).
    zx = np.arange(60, 70, 0.1) + 0.01
    zy = np.arange(2, 6, 0.1) + 0.01
    gx, gy = np.meshgrid(zx, zy)
    zone_seen = np.union1d(seen, _keys(gx.ravel(), gy.ravel()))
    bar_x = gx.ravel()[(gx.ravel() % 2.0) < 0.5]
    bar_y = gy.ravel()[(gx.ravel() % 2.0) < 0.5]
    zone_paint = np.union1d(paint, _keys(bar_x, bar_y))
    outline = dict(geometry=Polygon([(60, 2), (70, 2), (70, 6), (60, 6)]))
    bar = dict(geometry=Polygon([(62, 2), (62.4, 2), (62.4, 6), (62, 6)]))
    res = P.self_check(clean, [outline, bar], zone_seen, zone_paint)
    assert res["zone_outline_polygons"]["count"] == 1

    # Mistake 9: the paint of an untraced border line exported as a thin polygon.
    strip = dict(geometry=Polygon([(30, -0.1), (34, -0.1), (34, 0.1), (30, 0.1)]))
    res = P.self_check(clean, [strip], seen, paint)
    assert res["line_paint_polygons"]["count"] == 1
    hatch_bar = Polygon([(30, -1), (30.45, -1), (30.45, 1.5), (30, 1.5)])
    assert not P.is_line_paint_polygon(hatch_bar)

    # Mistake 10: speckle on the verge beyond the edge line exported as a mark.
    edge = _solid(0, 99, 0, n=2)
    lane = _solid(0, 99, 0, n=2)
    for k in ("geometry_3d", "geometry_2d"):
        lane[k] = shapely_translate(lane[k], yoff=3.5)
    verge = dict(geometry=Polygon([(40, -4.5), (44, -4.5), (44, -3.8), (40, -3.8)]))
    arrow = dict(geometry=Polygon([(60, 1.2), (65, 1.2), (65, 2.3), (60, 2.3)]))  # in the lane
    res = P.self_check([edge, lane], [verge, arrow], seen, paint)
    assert res["off_lane_polygons"]["count"] == 1
    assert abs(res["off_lane_polygons"]["where"][0][1] + 4.15) < 0.2


def test_short_line_through_hatch_bars_is_not_a_lane_line():
    bars = [Polygon([(x, -1), (x + 0.4, -1), (x + 1.4, 1), (x + 1.0, 1)]) for x in (0, 2, 4)]
    across = LineString([(0, 0), (5, 0)])  # traced through three bars
    assert P.runs_through_marks(across, bars)
    beside = LineString([(0, 2), (5, 2)])  # a real line next to the hatch zone
    assert not P.runs_through_marks(beside, bars)
    long_line = LineString([(-20, 0), (30, 0)])  # long lines are never dropped by this rule
    assert not P.runs_through_marks(long_line, bars)


def test_paint_support_counts_only_observed_stations():
    from roadmarking.digitization.raster_tiles import make_grid

    grid = make_grid((0.0, 0.0, 10.0, 2.0))
    cell = grid.cell
    c0 = round((0.0 - grid.x0) / cell)
    observed = np.zeros((grid.rows, grid.cols), bool)
    observed[:, c0 : c0 + int(8 / cell)] = True  # last 2 m hidden
    paint = np.zeros_like(observed)
    row = round((grid.y0 - 1.0) / cell - 0.5)
    paint[row - 1 : row + 2, c0 : c0 + int(4 / cell)] = True  # paint on the first 4 m only
    line = LineString([(0.0, 1.0), (10.0, 1.0)])
    sup = P.paint_support(line, paint, observed, grid)
    assert 0.45 < sup < 0.55  # 4 of the 8 observed metres
    assert P.paint_support(line, paint, np.zeros_like(observed), grid) is None


def test_dash_beside_continuous_line_becomes_a_line_not_a_polygon():
    from roadmarking.app.recover import dash_beside_line

    solid = LineString([(-50, 0), (50, 0)])
    merged = Polygon([(0, -0.1), (3.3, -0.1), (3.3, 0.55), (0, 0.5)])  # dash + line paint
    got = dash_beside_line(merged, [solid])
    assert got is not None
    dash, width = got
    assert abs(dash.length - 3.3) < 0.2 and 0.2 < dash.centroid.y < 0.45 and width <= 0.45
    arrow = Polygon([(0, 1.0), (5, 1.0), (5, 2.2), (0, 2.2)])  # a mark in the lane
    assert dash_beside_line(arrow, [solid]) is None
    across = Polygon([(0, -0.5), (0.4, -0.5), (0.4, 2.5), (0, 2.5)])  # bar across the line
    assert dash_beside_line(across, [solid]) is None


def test_mark_cut_back_at_the_lane_line_it_touches():
    edge = LineString([(-50, 0), (50, 0)])
    arrow = Polygon([(0, -0.4), (5, -0.4), (5, 1.5), (0, 1.5)])  # fitted over the line paint
    cut = P.clip_at_lines(arrow, [edge])
    assert cut.bounds[1] >= 0.09 and cut.area > 0.6 * arrow.area
    assert P.clip_at_lines(Polygon([(0, 1), (5, 1), (5, 2), (0, 2)]), [edge]).area == 5.0


def test_short_line_inside_an_arrow_is_dropped():
    arrow = Polygon([(0, 1), (5, 1), (5, 2.2), (0, 2.2)])
    inside = LineString([(1, 1.6), (3.5, 1.4)])
    assert P.runs_through_marks(inside, [arrow])
    assert not P.runs_through_marks(LineString([(-20, 0), (30, 0)]), [arrow])


def test_line_end_running_into_an_arrow_is_cut_back_and_arrow_not_cut_by_it():
    arrow = Polygon([(10, 0.4), (15, 0.4), (15, 1.6), (10, 1.6)])
    hook = LineString([(-30, 0), (9, 0), (12, 1.0)])  # edge line, then into the arrow
    t = P.trim_ends_in_marks(hook, [arrow])
    assert t is not None and t.length < hook.length
    assert t.distance(arrow) >= 0.2
    assert P.clip_at_lines(arrow, [hook]).equals(arrow)  # a line ending in the mark cuts nothing


def _paint_raster(bbox, centre_fn, half_width):
    """Intensity raster with a painted stripe of ``half_width`` around y = centre_fn(x)."""
    from roadmarking.digitization.raster_tiles import make_grid, pixel_to_xy

    grid = make_grid(bbox)
    rr, cc = np.mgrid[0 : grid.rows, 0 : grid.cols]
    x, y = pixel_to_xy(grid, rr, cc)
    gray = np.where(np.abs(y - centre_fn(x)) <= half_width, 190.0, 50.0)
    gray += np.random.default_rng(0).normal(0, 6, gray.shape)
    return grid, gray, np.ones_like(gray, bool)


def test_line_moved_to_the_centre_of_a_wide_stripe():
    from roadmarking.app.centring import recentre

    grid, gray, obs = _paint_raster((0.0, 0.0, 30.0, 4.0), lambda x: 2.0 + 0 * x, 0.25)
    edge_line = LineString([(1, 2.18), (29, 2.18)])  # on the stripe's upper edge (+18 cm)
    new, _, _ = recentre(edge_line, gray, obs, grid)
    ys = np.asarray(new.coords)[:, 1]
    assert np.max(np.abs(ys - 2.0)) < 0.015


def test_line_follows_the_paint_on_a_curve_and_junction_end_stays():
    from roadmarking.app.centring import recentre

    radius = 60.0  # tight curve: centre (15, 2 + radius)

    def arc(x):
        return 2.0 + radius - np.sqrt(np.maximum(radius**2 - (x - 15.0) ** 2, 0))

    grid, gray, obs = _paint_raster((0.0, 0.0, 30.0, 6.0), arc, 0.08)
    xs = np.linspace(1, 29, 15)
    chord = LineString(np.column_stack([xs, arc(xs) + 0.06]))  # smoothed: 6 cm off the paint
    new, _, _ = recentre(chord, gray, obs, grid)
    c = np.asarray(new.coords)
    assert np.median(np.abs(c[:, 1] - arc(c[:, 0]))) < 0.01
    assert np.max(np.abs(c[3:-3, 1] - arc(c[3:-3, 0]))) < 0.02
    kept, _, _ = recentre(chord, gray, obs, grid, fixed_ends=(True, False))
    assert np.allclose(kept.coords[0], chord.coords[0])  # junction end not moved


def test_off_centre_stretch_reported_and_centred_line_clean():
    from roadmarking.app.centring import off_centre_stretches

    grid, gray, obs = _paint_raster((0.0, 0.0, 30.0, 4.0), lambda x: 2.0 + 0 * x, 0.08)
    assert off_centre_stretches(LineString([(1, 2.0), (29, 2.0)]), gray, obs, grid) == []
    off = off_centre_stretches(LineString([(1, 2.05), (29, 2.05)]), gray, obs, grid)
    assert off and sum(s[2] for s in off) > 20 and abs(off[0][3] + 0.05) < 0.01


def test_gui_after_a_failed_job_a_new_file_can_be_started(tmp_path):
    tk = pytest.importorskip("tkinter")
    from roadmarking.app.gui import App

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    app = App(root)
    bad = _tiny_las(tmp_path / "Track01_7.las")
    app.set_input(str(bad))
    assert app.output_var.get().endswith("Track01_7_digitized.gpkg")
    # a job fails (e.g. corrupt LAZ chunks)
    app.worker = object()
    app.events.put(("error", "1.1% of the points could not be decoded"))
    app.poll()
    assert app.worker is None and str(app.start_btn["state"]) == "normal"
    assert "could not be decoded" in app.result_text.get("1.0", "end")
    # the user picks another file: the output follows it and the panel is cleared
    good = _tiny_las(tmp_path / "Track01_8.las")
    app.set_input(str(good))
    assert app.output_var.get().endswith("Track01_8_digitized.gpkg")
    assert app.result_text.get("1.0", "end").strip() == ""
    assert app.percent_var.get() == "0 %" and str(app.start_btn["state"]) == "normal"
    # an output chosen by the user is kept when the input changes
    app.output_var.set(str(tmp_path / "my_name.gpkg"))
    app.output_auto = False
    app.set_input(str(bad))
    assert app.output_var.get().endswith("my_name.gpkg")
    root.destroy()


def test_polygon_fitted_to_its_paint():
    from shapely import affinity

    from roadmarking.app.refine import fit_to_paint

    truth = Polygon([(10, 2), (14, 2), (14, 3.2), (10, 3.2)])
    grid, gray, _obs = _paint_raster((5.0, 0.0, 20.0, 6.0), lambda x: 0 * x - 100, 0.0)
    from rasterio.features import rasterize

    paint = rasterize([truth], out_shape=gray.shape, transform=grid.affine).astype(bool)
    off = affinity.rotate(affinity.translate(truth, 0.1, -0.08), 3.0)  # template slightly off
    new, before, after = fit_to_paint(off, paint, grid)
    assert after > before + 0.05 and after > 0.93
    assert new.symmetric_difference(truth).area < 0.35 * off.symmetric_difference(truth).area


def test_lines_inside_marks_dropped_over_the_whole_file():
    frame = Polygon(
        [(0, 0), (8, 1.5), (8, -1.5)], [[(1.5, 0), (7.4, 1.0), (7.4, -1.0)]]
    )  # give-way frame (painted border with hole)
    along_border = _solid(0.4, 7.6, 0, n=2)
    along_border["geometry_2d"] = LineString([(0.3, 0.02), (7.7, 1.38)])
    lane = _solid(-20, 40, 0, n=2)
    lane["geometry_2d"] = LineString([(-20, -3), (40, -3)])
    kept, dropped = P.drop_lines_in_marks([along_border, lane], [dict(geometry=frame)])
    assert dropped == 1 and kept[0] is lane


@pytest.mark.skipif(
    os.environ.get("PLOTLINE_REPORTED_CASES") != "1",
    reason="set PLOTLINE_REPORTED_CASES=1 (re-runs every reported file: ~4 min each, NAS + GPU)",
)
def test_every_reported_mistake_stays_fixed(tmp_path, monkeypatch):
    """Re-run the app on every file the user reported a mistake on and check every recorded
    place (tests/reference/reported_cases.json). Run before each release."""
    import json
    import subprocess
    import sys

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    share = Path(
        os.environ.get(
            "PLOTLINE_LAS_DIR", "//192.168.10.81/e/New folder/Converted_laz_file_format_laz_1.2"
        )
    )
    cases = json.loads((ROOT / "tests/reference/reported_cases.json").read_text(encoding="utf-8"))
    outs = []
    for stem in sorted({c["file"] for c in cases["cases"]}):
        las = share / f"{stem}.laz"
        if not las.exists():
            pytest.skip(f"{las} not reachable")
        out = tmp_path / f"{stem}_digitized.gpkg"
        P.run_job(las, out)
        outs.append(str(out))
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_reported_cases.py"), *outs],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stdout


def test_polygon_edges_snapped_to_the_paint_outline_and_frame_hole():
    from shapely.geometry import Polygon as Poly

    from roadmarking.app.refine import snap_edges
    from roadmarking.digitization.raster_tiles import make_grid, pixel_to_xy

    outer = [(0.0, 0.0), (8.0, 1.6), (8.0, -1.6)]
    hole = [(1.6, 0.0), (7.6, 1.15), (7.6, -1.15)]
    truth = Poly(outer, [hole])  # painted give-way frame
    grid = make_grid((-1.0, -3.0, 10.0, 3.0))
    rr, cc = np.mgrid[0 : grid.rows, 0 : grid.cols]
    x, y = pixel_to_xy(grid, rr, cc)
    painted = shapely.contains_xy(truth, x, y)
    gray = np.where(painted, 190.0, 50.0) + np.random.default_rng(0).normal(0, 6, x.shape)
    obs = np.ones_like(painted)
    # fitted frame: outline 8 cm too big, hole 8 cm too small (unpainted road inside the paint)
    fitted = Poly(
        Poly(outer).buffer(0.08, join_style="mitre").exterior.coords,
        [Poly(hole).buffer(-0.08, join_style="mitre").exterior.coords],
    )
    new, moved = snap_edges(fitted, gray, obs, grid)
    assert moved >= 5
    assert new.symmetric_difference(truth).area < 0.3 * fitted.symmetric_difference(truth).area
    assert Poly(new.exterior).hausdorff_distance(Poly(outer)) < 0.06
    assert Poly(new.interiors[0]).hausdorff_distance(Poly(hole)) < 0.06


def test_give_way_frame_exported_as_outer_and_inner_solid_polygons():
    from shapely.geometry import Polygon as Poly

    outer = [(0, 0, 10.0), (8, 1.6, 10.1), (8, -1.6, 10.1)]
    hole = [(1.6, 0, 10.02), (7.6, 1.15, 10.09), (7.6, -1.15, 10.09)]
    frame = dict(
        geometry=Poly([p[:2] for p in outer], [[p[:2] for p in hole]]),
        geometry_3d=Poly(outer, [hole]),
        COD_RST="M9a",
        COD_JAE=None,
        review_flags="outline_mark",
    )
    arrow = dict(
        geometry=Poly([(20, 0), (25, 0), (25, 1), (20, 1)]),
        geometry_3d=Poly([(20, 0, 10), (25, 0, 10), (25, 1, 10), (20, 1, 10)]),
        COD_RST="M15",
        COD_JAE=None,
    )
    out = P.split_frames([frame, arrow])
    assert len(out) == 3 and out[2] is arrow
    o, i = out[0], out[1]
    assert not o["geometry"].interiors and not i["geometry"].interiors
    assert abs(o["geometry"].area - Poly([p[:2] for p in outer]).area) < 1e-9
    assert abs(i["geometry"].area - Poly([p[:2] for p in hole]).area) < 1e-9
    assert o["geometry_3d"].has_z and i["geometry_3d"].has_z
    assert o["COD_RST"] == i["COD_RST"] == "M9a"
    assert "frame_outer" in o["review_flags"] and "frame_inner" in i["review_flags"]
    rows = P.client_poly_rows(out)
    assert [r["COD_RST"] for r in rows] == ["M9a", "M9a", "M15"]


def test_line_crossing_between_two_side_by_side_stripes_is_found():
    from roadmarking.app.centring import gap_crossings
    from roadmarking.digitization.raster_tiles import make_grid, pixel_to_xy

    grid = make_grid((0.0, 0.0, 40.0, 4.0))
    rr, cc = np.mgrid[0 : grid.rows, 0 : grid.cols]
    _x, y = pixel_to_xy(grid, rr, cc)
    stripes = (np.abs(y - 2.0) <= 0.08) | (np.abs(y - 2.32) <= 0.08)  # 32 cm apart, never merge
    gray = np.where(stripes, 190.0, 50.0) + np.random.default_rng(0).normal(0, 5, y.shape)
    obs = np.ones_like(stripes)
    # traced line: on the lower stripe, jumps across the gap at x 18-21, then on the upper one
    jumper = LineString([(1, 2.0), (18, 2.0), (21, 2.32), (39, 2.32)])
    gaps = gap_crossings(jumper, gray, obs, grid)
    assert len(gaps) == 1 and 16.5 < gaps[0][0] < 19.5 and 18.5 < gaps[0][1] < 21.5
    assert gap_crossings(LineString([(1, 2.0), (39, 2.0)]), gray, obs, grid) == []
    assert gap_crossings(LineString([(1, 2.32), (39, 2.32)]), gray, obs, grid) == []


def test_thin_strip_touching_a_line_is_line_paint_but_a_mark_beside_it_is_not():
    line = LineString([(0, 0), (50, 0)])
    strip = Polygon([(20, 0.1), (20.9, 0.1), (20.9, 0.34), (20, 0.34)])  # 0.9 x 0.24, 10 cm off
    assert P.touches_line_paint(strip, [line])
    bar_nearby = Polygon([(20, 0.6), (22, 0.6), (22, 0.85), (20, 0.85)])  # clear of the line
    assert not P.touches_line_paint(bar_nearby, [line])
    arrow = Polygon([(20, 0.1), (25, 0.1), (25, 1.3), (20, 1.3)])  # wide mark next to the line
    assert not P.touches_line_paint(arrow, [line])


def test_scan_gap_across_a_mark_is_bridged_but_bare_road_is_not():
    from roadmarking.digitization.raster_tiles import make_grid, pixel_to_xy

    grid = make_grid((0.0, 0.0, 6.0, 3.0))
    rr, cc = np.mgrid[0 : grid.rows, 0 : grid.cols]
    x, y = pixel_to_xy(grid, rr, cc)
    bar = (np.abs(y - 1.5) <= 0.3) & (x >= 1) & (x <= 5)
    gray = np.where(bar, 200.0, 50.0)
    observed = np.ones_like(bar)
    observed[:, (x[0] > 2.9) & (x[0] < 3.1)] = False  # 20 cm unscanned strip across the bar
    gray[:, (x[0] > 3.9) & (x[0] < 4.1)] = 50.0  # 20 cm of scanned bare road across the bar
    m = P.mark_paint_raster(dict(gray=gray, observed=observed))
    inner = (np.abs(y - 1.5) <= 0.2) & ~observed & (x >= 1) & (x <= 5)
    assert m[inner].all()  # unscanned gap inside the mark: paint
    assert not m[bar & observed & (np.abs(x - 4.0) < 0.08)].any()  # scanned bare road: not paint


def test_host_paused_while_running_and_always_resumed(tmp_path):
    import subprocess
    import sys

    import psutil

    from roadmarking.app import host_pause

    host = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        assert host_pause.pause(host.pid)
        assert psutil.Process(host.pid).status() == psutil.STATUS_STOPPED
        assert host_pause.resume(host.pid)
        assert psutil.Process(host.pid).status() != psutil.STATUS_STOPPED
        # Plotline-AI dies while the host is paused: the watchdog resumes the host
        fake_plotline = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2)"])
        host_pause.pause(host.pid)
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        watch = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "roadmarking.app.host_pause",
                "--watch",
                str(fake_plotline.pid),
                str(host.pid),
            ],
            env=env,
        )
        watch.wait(timeout=30)
        assert psutil.Process(host.pid).status() != psutil.STATUS_STOPPED
        assert not host_pause.pause(os.getpid())  # never pauses itself
    finally:
        host_pause.resume(host.pid)
        host.kill()


def _sparse_faint_raster(bbox, centre_y, half_width, share, seed=1):
    """Far edge of a scan: a faint stripe seen in only ``share`` of the cells, plus single bright
    returns scattered beside it."""
    from roadmarking.digitization.raster_tiles import make_grid, pixel_to_xy

    rng = np.random.default_rng(seed)
    grid = make_grid(bbox)
    rr, cc = np.mgrid[0 : grid.rows, 0 : grid.cols]
    _, y = pixel_to_xy(grid, rr, cc)
    gray = np.where(np.abs(y - centre_y) <= half_width, 95.0, 50.0)
    gray += rng.normal(0, 10, gray.shape)
    specks = rng.random(gray.shape) < 0.01
    gray[specks] = 160.0  # isolated bright returns
    obs = rng.random(gray.shape) < share
    return grid, gray, obs


def test_line_on_faint_sparse_paint_is_centred_and_clean():
    """Reported: on faded paint at the far edge of the scan the line drifted a few cm from the
    centre (Track01_14). Sparse cross-sections are averaged over a longer stretch and smoothed
    slightly, so single bright returns do not set the stripe edges."""
    from roadmarking.app.centring import off_centre_stretches, recentre

    grid, gray, obs = _sparse_faint_raster((0.0, 0.0, 40.0, 4.0), 2.0, 0.07, share=0.3)
    line = LineString([(1, 2.04), (39, 2.04)])  # 4 cm off the paint centre
    new, _, share = recentre(line, gray, obs, grid)
    ys = np.asarray(new.coords)[:, 1]
    assert share is not None and share > 0.8
    assert np.percentile(np.abs(ys - 2.0), 90) < 0.015
    assert off_centre_stretches(new, gray, obs, grid) == []


def test_line_along_a_bright_band_is_not_a_stripe():
    """Reported: a line along a roadside barrier base (a wide bright band, darker road on one
    side only, a few dark pixels at the posts) passed the stripe check, which compared the axis
    with the *darkest* pixel on each side. Compared with the typical road on each side it is not
    a stripe; a real lane line still is."""
    from roadmarking.digitization.raster_tiles import make_grid, pixel_to_xy

    rng = np.random.default_rng(2)
    grid = make_grid((0.0, 0.0, 30.0, 6.0))
    rr, cc = np.mgrid[0 : grid.rows, 0 : grid.cols]
    x, y = pixel_to_xy(grid, rr, cc)
    gray = np.full(x.shape, 50.0)
    gray[(y >= 3.0) & (y <= 4.3)] = 85.0  # wide light band (barrier base / verge)
    gray[(y >= 4.2) & (y <= 4.3)] = 100.0  # slightly brighter rim the AI line follows
    posts = (np.abs(((x + 0.5) % 2.0) - 0.5) < 0.05) & (y > 3.6) & (y < 4.0)
    gray[posts] = 40.0  # dark feet of the posts inside the band
    gray[np.abs(y - 1.5) <= 0.07] = 120.0  # a real lane line
    gray += rng.normal(0, 4, gray.shape)
    obs = np.ones_like(gray, bool)
    band_line = LineString([(1, 4.25), (29, 4.25)])
    lane_line = LineString([(1, 1.5), (29, 1.5)])
    assert P.stripe_support(band_line, gray, obs, grid) >= P.MIN_STRIPE_SUPPORT  # old check
    typ = P.stripe_support(band_line, gray, obs, grid, typical_side=True)
    assert typ < P.MIN_TYPICAL_STRIPE_SUPPORT
    assert P.stripe_support(lane_line, gray, obs, grid, typical_side=True) > 0.95
