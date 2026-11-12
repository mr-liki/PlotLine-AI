"""Phase 3.1 corridor refinement: synthetic ground truth, ROI rules, grid/version safety."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pytest
import rasterio
from shapely.geometry import Polygon, box, mapping

from roadmarking.io.pointcloud import PointCloud, PointCloudHeader
from roadmarking.preprocessing.road_roi import RoadRoiError, load_road_roi
from roadmarking.preprocessing.road_surface import (
    CONF_HIGH,
    CONF_INSUFFICIENT,
    CONF_NO_OBSERVATION,
    CONF_ROI_CONFIRMED,
    CONF_UNCERTAIN,
    CorridorRefiner,
    RefinementCriteria,
    RoadCriteria,
    SurfaceAccumulator,
)

RNG = np.random.default_rng(7)
V1 = RoadCriteria(min_component_m2=30)
V2 = RefinementCriteria(core_min_area_m2=200)


def header_for(xyz):
    return PointCloudHeader(
        None,
        len(xyz),
        3,
        "1.2",
        np.full(3, 0.001),
        np.zeros(3),
        xyz.min(axis=0),
        xyz.max(axis=0),
        None,
    )


def grid(x0, x1, y0, y1, z, spacing=0.2, noise=0.004):
    x, y = np.meshgrid(
        np.arange(x0 + spacing / 2, x1, spacing), np.arange(y0 + spacing / 2, y1, spacing)
    )
    x, y = x.ravel(), y.ravel()
    zz = z(x, y) if callable(z) else np.full(x.shape, float(z))
    return np.column_stack((x, y, zz + RNG.normal(0, noise, x.shape)))


def run(parts, roi=None, v1=V1, v2=V2):
    xyz = np.vstack(list(parts.values()))
    acc = SurfaceAccumulator(header_for(xyz), v1)
    acc.add(xyz)
    base = acc.finish(None)
    ref = CorridorRefiner(base, v2)
    for chunk in np.array_split(xyz, 5):  # streaming, chunk-invariant
        ref.add(chunk)
    model = ref.finish(roi)
    out = {}
    start = 0
    v1_state = base.classify(xyz)
    v2_state = model.classify(xyz)
    for name, pts in parts.items():
        sl = slice(start, start + len(pts))
        out[name] = (v1_state[sl] > 0, v2_state[sl] > 0)
        start += len(pts)
    return model, out


def frac(mask):
    return float(mask.mean()) if len(mask) else 0.0


def test_1_flat_road():
    model, out = run({"road": grid(0, 60, 0, 10, 5.0)})
    assert frac(out["road"][1]) > 0.95
    assert (model.confidence == CONF_HIGH).sum() >= 0.9 * 600


@pytest.mark.parametrize("grade", [0.06, 0.12])
def test_2_sloped_road(grade):
    _, out = run({"road": grid(0, 60, 0, 10, lambda x, y: 20 + grade * x)})
    assert frac(out["road"][1]) > 0.95


def test_3_elevated_embankment_does_not_take_slopes_or_ground():
    road = grid(0, 60, 10, 20, 5.0)
    left = grid(0, 60, 0, 10, lambda x, y: 5.0 - 0.5 * (10 - y))
    right = grid(0, 60, 20, 30, lambda x, y: 5.0 - 0.5 * (y - 20))
    ground = grid(0, 60, 30, 70, 0.0)  # broad flat field beside the embankment
    model, out = run({"road": road, "left": left, "right": right, "ground": ground})
    assert frac(out["road"][1]) > 0.95
    assert frac(out["ground"][0]) > 0.9  # v1 accepted the field as a candidate
    r, c = model.spec.index(np.array([30.0]), np.array([50.0]))
    assert model.confidence[r, c][0] == CONF_UNCERTAIN
    # Only points within the inlier band at the crest may join; nothing far down-slope.
    assert frac(out["left"][1]) < 0.05 and frac(out["right"][1]) < 0.05
    assert not out["ground"][1].any()


def barrier_scene():
    road = grid(0, 60, 0, 12, 5.0)
    road = road[(road[:, 1] < 8.4) | (road[:, 1] > 8.8)]  # barrier footprint has no road
    barrier = grid(0, 60, 8.4, 8.8, 5.8, spacing=0.1)
    paint = grid(0, 60, 8.05, 8.2, 5.0, spacing=0.05)  # edge line in the mixed cell
    return road, barrier, paint


def test_4_adjacent_lane_beyond_mixed_barrier_cell_and_10_marking_at_boundary():
    road, barrier, paint = barrier_scene()
    _model, out = run({"road": road, "barrier": barrier, "paint": paint})
    v1_paint, v2_paint = out["paint"]
    assert frac(v1_paint) < 0.2  # v1 loses the whole mixed cell
    assert frac(v2_paint) > 0.95  # v2 keeps the painted surface points
    assert not out["barrier"][1].any()  # barrier points stay excluded
    road_mixed = road[(road[:, 1] >= 8) & (road[:, 1] < 9)]
    assert len(road_mixed)
    assert frac(out["road"][1]) > frac(out["road"][0])


def test_10_marking_result_does_not_depend_on_intensity():
    road, barrier, paint = barrier_scene()
    xyz = np.vstack((road, barrier, paint))
    states = []
    for raw in (np.full(len(xyz), 1000, np.uint16), np.full(len(xyz), 65535, np.uint16)):
        cloud = PointCloud(xyz, raw, None, header_for(xyz))
        acc = SurfaceAccumulator(cloud.header, V1)
        acc.add(cloud.xyz)
        ref = CorridorRefiner(acc.finish(), V2)
        ref.add(cloud.xyz)
        states.append(ref.finish().classify(cloud.xyz))
    assert np.array_equal(*states)


def test_5_raised_sidewalk_is_not_automatic_road():
    road = grid(0, 60, 0, 12.5, 5.0)
    sidewalk = grid(0, 60, 12.5, 15.0, 5.15)  # 15 cm kerb inside a 1 m cell
    _, out = run({"road": road, "sidewalk": sidewalk})
    assert frac(out["road"][1]) > 0.95
    assert not out["sidewalk"][1].any()
    kerb_side_road = out["road"][1][road[:, 1] > 12]
    assert frac(kerb_side_road) > 0.9  # road surface up to the kerb is recovered


def test_6_flat_roof_near_road_is_uncertain_not_selected():
    road = grid(0, 60, 0, 10, 5.0)
    roof = grid(20, 32, 16, 28, 9.0)
    model, out = run({"road": road, "roof": roof})
    assert not out["roof"][1].any()
    r, c = model.spec.index(np.array([26.0]), np.array([22.0]))
    assert model.confidence[r, c][0] == CONF_UNCERTAIN


def test_7_scan_gap_is_not_manufactured():
    road = grid(0, 150, 0, 10, 5.0)
    road = road[(road[:, 0] < 70) | (road[:, 0] > 75)]
    model, out = run({"road": road})
    assert frac(out["road"][1]) > 0.95
    x = np.full(5, 72.5)
    y = np.linspace(1, 9, 5)
    assert (model.state_at(x, y) == 0).all()
    assert (model.confidence_at(x, y) == CONF_NO_OBSERVATION).all()


def test_8_curved_and_9_banked_road():
    t = RNG.uniform(0, np.pi / 2, 60000)
    rad = RNG.uniform(36, 44, 60000)
    x, y = rad * np.cos(t), rad * np.sin(t)
    z = 10 + 0.06 * (rad - 40) + 0.02 * t * 40  # 6 % banking + gentle grade
    curve = np.column_stack((x, y, z + RNG.normal(0, 0.004, len(x))))
    _, out = run({"curve": curve})
    assert frac(out["curve"][1]) > 0.93


def test_11_sparse_cells_inside_corridor_recovered_edge_sparse_stays_insufficient():
    road = grid(0, 60, 0, 10, 5.0)
    hole = (road[:, 0] > 20) & (road[:, 0] < 24) & (road[:, 1] > 3) & (road[:, 1] < 7)
    sparse = road[hole][::5]  # ~5 points per 1 m cell (< v1 min_points 8)
    road = road[~hole]
    edge_sparse = grid(0, 60, 10, 11, 5.0)[::10]
    model, out = run({"road": road, "sparse": sparse, "edge": edge_sparse})
    assert frac(out["sparse"][0]) < 0.1
    assert frac(out["sparse"][1]) > 0.9
    r, c = model.spec.index(np.array([30.5]), np.array([10.5]))
    assert model.confidence[r, c][0] == CONF_INSUFFICIENT


def test_12_uncertain_disconnected_surface_and_vegetation():
    road = grid(0, 60, 0, 10, 5.0)
    veg = grid(0, 60, 10, 18, 5.0)
    veg[:, 2] += RNG.normal(0, 0.8, len(veg))
    patch = grid(20, 35, 18, 30, 5.1)
    model, out = run({"road": road, "veg": veg, "patch": patch})
    assert not out["patch"][1].any() and not out["veg"][1].any()
    r, c = model.spec.index(np.array([27.5]), np.array([24.5]))
    assert model.confidence[r, c][0] == CONF_UNCERTAIN


def test_roi_clipping_v2_marks_supported_cells_only():
    road = grid(0, 60, 0, 10, 5.0)
    roof = grid(20, 32, 16, 28, 9.0)
    roi = box(10, -1, 40, 30)
    model, out = run({"road": road, "roof": roof}, roi=roi)
    chosen = road[out["road"][1]]
    assert (chosen[:, 0] > 10).all() and (chosen[:, 0] < 40).all()
    assert (model.state[model.state > 0] == 2).all()
    r, c = model.spec.index(np.array([26.0]), np.array([22.0]))
    assert model.confidence[r, c][0] == CONF_ROI_CONFIRMED  # roof only if reviewer drew it
    r, c = model.spec.index(np.array([5.0]), np.array([5.0]))
    assert model.state[r, c] == 0


def test_refinement_is_deterministic_and_inputs_immutable():
    road, barrier, paint = barrier_scene()
    xyz = np.vstack((road, barrier, paint))
    copy = xyz.copy()
    results = []
    for order in (np.arange(len(xyz)), RNG.permutation(len(xyz))):
        acc = SurfaceAccumulator(header_for(xyz), V1)
        acc.add(xyz[order])
        ref = CorridorRefiner(acc.finish(), V2)
        for chunk in np.array_split(xyz[order], 7):
            ref.add(chunk)
        results.append(ref.finish())
    assert np.array_equal(results[0].confidence, results[1].confidence)
    assert np.array_equal(results[0].classify(xyz), results[1].classify(xyz))
    assert np.array_equal(xyz, copy)


# ----------------------------------------------------------------------------- ROI files
class H:
    crs_wkt = None
    mins = np.array([0.0, 0.0, 0.0])
    maxs = np.array([100.0, 100.0, 10.0])


def _geojson(tmp_path, feats, crs=None):
    doc = {"type": "FeatureCollection", "features": feats}
    if crs:
        doc["crs"] = crs
    p = tmp_path / "roi.geojson"
    p.write_text(json.dumps(doc))
    return p


def _feat(geom, **props):
    return {"type": "Feature", "geometry": mapping(geom), "properties": props}


def test_roi_holes_overlaps_and_multiple_polygons(tmp_path):
    outer = Polygon(
        [(0, 0), (50, 0), (50, 50), (0, 50)], [[(10, 10), (20, 10), (20, 20), (10, 20)]]
    )
    p = _geojson(
        tmp_path,
        [
            _feat(outer, review_status="APPROVED"),
            _feat(box(40, 0, 60, 10), review_status="APPROVED"),
        ],
    )
    roi, rep = load_road_roi(p, H)
    assert rep["features"] == 2 and rep["holes"] == 1
    assert rep["overlapping_area_native_units2"] == pytest.approx(100.0)
    assert rep["review_status"] == "APPROVED" and not rep["coordinates_modified"]
    assert not roi.contains(box(12, 12, 13, 13))


@pytest.mark.parametrize(
    "geom,props,match",
    [
        (Polygon([(0, 0), (10, 10), (10, 0), (0, 10)]), {}, "invalid"),
        (box(0, 0, 10, 10), {"review_status": "PROPOSED_UNREVIEWED"}, "not an approved"),
        (box(500, 500, 510, 510), {}, "does not overlap"),
    ],
)
def test_roi_rejections(tmp_path, geom, props, match):
    with pytest.raises(RoadRoiError, match=match):
        load_road_roi(_geojson(tmp_path, [_feat(geom, **props)]), H)


def test_roi_crs_mismatch_requires_explicit_native_assertion(tmp_path):
    gdf = gpd.GeoDataFrame(
        {"review_status": ["APPROVED"]}, geometry=[box(0, 0, 10, 10)], crs="EPSG:3763"
    )
    p = tmp_path / "roi.gpkg"
    gdf.to_file(p, layer="roi", driver="GPKG")
    with pytest.raises(RoadRoiError, match="CRS"):
        load_road_roi(p, H)
    roi, rep = load_road_roi(p, H, assert_native_frame=True)
    assert rep["native_frame_assertion"] and roi.area == 100


# ----------------------------------------------------------------------------- pipeline
def _write_laz(path, xyz, intensity):
    las = laspy.LasData(laspy.LasHeader(point_format=3, version="1.2"))
    las.header.scales = [0.001] * 3
    las.header.offsets = [float(np.floor(xyz[:, 0].min())), float(np.floor(xyz[:, 1].min())), 0.0]
    las.x, las.y, las.z = xyz.T
    las.intensity = intensity
    las.return_number = np.ones(len(xyz), np.uint8)
    las.number_of_returns = np.ones(len(xyz), np.uint8)
    las.gps_time = np.full(len(xyz), 100.0)
    las.write(path)


def test_pipeline_v2_namespace_grid_layers_and_determinism(tmp_path, monkeypatch):
    import roadmarking.rasterization.pipeline as pipeline
    from roadmarking.config import load_config

    monkeypatch.setenv("ROADMARKING_DATA_ROOT", str(tmp_path))
    cfg = load_config()
    road, barrier, paint = barrier_scene()
    xyz = np.vstack((road, barrier, paint)) + np.array([-56000.0, -89000.0, 0.0])
    intensity = np.r_[
        np.full(len(road), 4000), np.full(len(barrier), 3000), np.full(len(paint), 40000)
    ].astype(np.uint16)
    source = "Job_20260325_0838_Track01_3"
    monkeypatch.setitem(pipeline.INTACT, source, len(xyz))
    path = tmp_path / f"{source}.laz"
    _write_laz(path, xyz, intensity)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    cfg.raw["phase3"]["road_surface"]["min_component_m2"] = 30.0
    r1 = pipeline.process_source(cfg, path)
    calib = r1["normalization"]
    kw = dict(
        namespace="phase3_1",
        road_method="v2",
        refinement={"core_min_area_m2": 200.0},
        calibration_override=calib,
    )
    r2 = pipeline.process_source(cfg, path, **kw)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert r1["qc"]["passed"] and r2["qc"]["passed"]
    assert "phase3_1" in r2["manifest"] and "phase3_1" not in r1["manifest"]
    assert r2["selected_points"] > r1["selected_points"]
    assert r2["normalization"] == calib
    m1 = {t["tile_id"]: t for t in json.loads(Path(r1["manifest"]).read_text())}
    m2 = {t["tile_id"]: t for t in json.loads(Path(r2["manifest"]).read_text())}
    for tid in set(m1) & set(m2):
        assert m1[tid]["affine"] == m2[tid]["affine"]  # same global grid
    t = next(iter(m2.values()))
    assert t["road_extraction_version"] == "v2" and "road_confidence" in t["layers"]
    with (
        rasterio.open(t["layers"]["road_confidence"]) as a,
        rasterio.open(t["layers"]["count"]) as b,
    ):
        assert a.transform == b.transform and a.crs is None
        conf, count = a.read(1), b.read(1)
    assert set(np.unique(conf)) <= set(range(6))
    with rasterio.open(t["layers"]["normalized"]) as n:
        norm = n.read(1)
    assert (norm[count == 0] == -9999).all()  # NoData preserved, nothing filled
    # Phase 4 can consume the v2 manifest (all required layers, native north-up grid).
    from roadmarking.training.preflight import inspect_manifest

    check = inspect_manifest(Path(r2["manifest"]))
    assert check["metadata_consistent"] and not check["errors"]
    assert all(t["phase4_status"] == "NEEDS_ROAD_REVIEW" for t in m2.values())
    # Phase 3 namespace untouched by the v2 run.
    assert not any("phase3_1" in str(p) for p in (tmp_path / "data/processed/phase3").rglob("*"))
    # Determinism: a fresh namespace run reproduces identical layer bytes.
    r3 = pipeline.process_source(cfg, path, **{**kw, "namespace": "phase3_1_repeat"})
    m3 = {t["tile_id"]: t for t in json.loads(Path(r3["manifest"]).read_text())}
    assert set(m3) == set(m2)
    for tid in m2:
        for name in ("road_confidence", "count", "normalized"):
            assert _read(m2[tid]["layers"][name]) == _read(m3[tid]["layers"][name])


def _read(path):
    with rasterio.open(path) as src:
        return src.read(1).tobytes()
