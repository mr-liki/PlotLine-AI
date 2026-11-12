"""Phase 4 dataset tooling: coordinates, masks, QC statuses, splits, end-to-end gates."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from affine import Affine
from shapely.geometry import LineString, MultiPolygon, Polygon, box

from roadmarking.dataset import alignment as al
from roadmarking.dataset import synthetic
from roadmarking.dataset.annotations import assign_status, normalize_features
from roadmarking.dataset.geometry import (
    coverage_fraction,
    painted_area_from_centerline,
    pixel_center_to_world,
    pixel_index,
    tile_affine,
    world_to_pixel,
)
from roadmarking.dataset.labels import (
    BACKGROUND,
    BOUNDARY,
    CONFLICT,
    INSTANCE,
    NO_DATA,
    UNKNOWN,
    LabelFeature,
    build_tile_labels,
    instance_id_to_semantic_id,
)
from roadmarking.dataset.pipeline import converter_selftest, run_phase4
from roadmarking.dataset.splits import check_leakage, spatial_split
from roadmarking.dataset.taxonomy import Taxonomy
from roadmarking.dataset.writer import write_sample

ROOT = Path(__file__).resolve().parents[1]
TAX = Taxonomy.load(ROOT / "configs/phase4_taxonomy.yaml")
A = Affine(0.05, 0, 1000.0, 0, -0.05, 2000.0)


# ----------------------------------------------------------------- coordinates
def test_world_pixel_roundtrip_and_centre_convention():
    rows, cols = np.array([0, 5, 1023]), np.array([0, 7, 1023])
    x, y = pixel_center_to_world(A, rows, cols)
    assert np.allclose(x, 1000.0 + (cols + 0.5) * 0.05)
    assert np.allclose(y, 2000.0 - (rows + 0.5) * 0.05)
    r, c = world_to_pixel(A, x, y)
    assert np.allclose(r, rows + 0.5) and np.allclose(c, cols + 0.5)
    pr, pc = pixel_index(A, x, y)
    assert (pr == rows).all() and (pc == cols).all()
    # the top-left corner is pixel (0,0); a point 1e-9 inside stays in pixel 0
    assert tuple(np.ravel(pixel_index(A, 1000.0 + 1e-9, 2000.0 - 1e-9))) == (0, 0)


def test_polygon_rasterization_no_half_pixel_shift():
    # square exactly covering pixels rows 10-19, cols 20-29
    sq = box(1000 + 20 * 0.05, 2000 - 20 * 0.05, 1000 + 30 * 0.05, 2000 - 10 * 0.05)
    cov, (r0, r1, c0, c1) = coverage_fraction(sq, A, (64, 64), supersample=4)
    full = np.zeros((64, 64))
    full[r0:r1, c0:c1] = cov
    assert full[10:20, 20:30].min() == 1.0
    assert full.sum() == pytest.approx(100.0)


def test_polygon_holes_preserved_and_area_consistent():
    outer = box(1000.5, 1998.0, 1001.5, 1999.0)
    hole = box(1000.8, 1998.3, 1001.2, 1998.7)
    poly = Polygon(outer.exterior.coords, [hole.exterior.coords])
    cov, win = coverage_fraction(poly, A, (64, 64), supersample=8)
    assert cov.sum() * 0.05**2 == pytest.approx(poly.area, rel=0.02)
    r, c = world_to_pixel(A, 1001.0, 1998.5)
    assert cov[int(r) - win[0], int(c) - win[2]] == 0


def test_centreline_buffer_width_flat_caps_and_curves():
    line = LineString([(1000.0, 1999.0, 7.0), (1002.0, 1999.0, 7.0)])
    area = painted_area_from_centerline(line, 0.15)
    assert area.area == pytest.approx(2.0 * 0.15, rel=1e-6)  # flat caps: no extension
    minx, _, maxx, _ = area.bounds
    assert (minx, maxx) == pytest.approx((1000.0, 1002.0))
    t = np.linspace(0, np.pi / 2, 50)
    arc = LineString(np.c_[1000 + 3 * np.cos(t), 1995 + 3 * np.sin(t)])
    assert painted_area_from_centerline(arc, 0.3).area == pytest.approx(arc.length * 0.3, rel=0.01)
    with pytest.raises(ValueError):
        painted_area_from_centerline(line, float("nan"))


# ----------------------------------------------------------------- labels
def _labels(feats, observed=None, completeness=None, shape=(256, 256)):
    obs = np.ones(shape, np.uint8) if observed is None else observed
    return build_tile_labels(A, shape, obs, feats, completeness=completeness)


def test_dashes_and_parallel_lines_stay_separate_instances():
    feats = [
        LabelFeature(1, "a", 2, "line", LineString([(1001, 1998), (1003, 1998)]), 0.15),
        LabelFeature(2, "b", 2, "line", LineString([(1004, 1998), (1006, 1998)]), 0.15),
        LabelFeature(3, "c", 1, "line", LineString([(1001, 1997.725), (1006, 1997.725)]), 0.15),
    ]
    lab = _labels(feats)
    ids = [r["local_id"] for r in lab.instances]
    assert ids == [1, 2, 3]
    assert all(r["pixels"] > 0 for r in lab.instances)
    assert set(np.unique(lab.instance)) == {0, 1, 2, 3}
    assert (lab.state == CONFLICT).sum() == 0
    # 0.15 m wide line on a 5 cm grid: 3 pixel rows per column
    cols = np.nonzero((lab.instance == 3).any(axis=0))[0]
    assert ((lab.instance == 3)[:, cols[5]]).sum() == 3


def test_semantic_vs_instance_ids_and_ignore_policy():
    feats = [LabelFeature(10, "p", 5, "polygon", box(1001, 1997, 1002, 1998))]
    obs = np.ones((256, 256), np.uint8)
    obs[:, 200:] = 0
    lab = _labels(feats, observed=obs)
    assert instance_id_to_semantic_id(lab) == {1: 5}
    assert lab.semantic[lab.instance == 1].tolist() == [5] * int((lab.instance == 1).sum())
    # no completeness region: nothing becomes background; unknown and no-data are ignored
    assert (lab.state == BACKGROUND).sum() == 0
    assert (lab.semantic == 0).sum() == 0
    assert (lab.state[:, 200:] == NO_DATA).all()
    assert (lab.semantic[lab.state == UNKNOWN] == 255).all()
    assert lab.instance.shape == lab.semantic.shape == lab.state.shape == (256, 256)
    lab2 = _labels(feats, observed=obs, completeness=box(1000, 1990, 1012.8, 2000))
    assert (lab2.semantic[(lab2.state == BACKGROUND)] == 0).all()
    assert (lab2.state == BACKGROUND).sum() > 0
    assert (lab2.state[:, 200:] == NO_DATA).all()  # unobserved never becomes background


def test_overlapping_instances_become_conflict_ignore_not_merged():
    feats = [
        LabelFeature(1, "a", 3, "polygon", box(1001, 1997, 1002, 1998)),
        LabelFeature(2, "b", 3, "polygon", box(1001.5, 1997, 1002.5, 1998)),
    ]
    lab = _labels(feats)
    assert (lab.state == CONFLICT).sum() == 10 * 20
    assert (lab.semantic[lab.state == CONFLICT] == 255).all()
    assert {1, 2} <= set(np.unique(lab.instance))


def test_partial_boundary_pixels_are_ignored():
    feats = [LabelFeature(1, "a", 3, "polygon", box(1001.01, 1997.01, 1002.01, 1998.01))]
    lab = _labels(feats)
    assert (lab.state == BOUNDARY).sum() > 0
    assert (lab.state == INSTANCE).sum() > 0


def test_unverified_centreline_width_is_rejected():
    feat = LabelFeature(1, "a", 1, "line", LineString([(1001, 1998), (1003, 1998)]), None)
    with pytest.raises(ValueError):
        _labels([feat])


def test_cross_tile_identity_and_selftest():
    feats = synthetic.features()
    left, right = synthetic.make_tile("L", 0, 0), synthetic.make_tile("R", 0, 192)
    labs = {}
    for t in (left, right):
        _, obs = synthetic.paint_raster(t, feats)
        labs[t["tile_id"]] = build_tile_labels(tile_affine(t), (256, 256), obs, feats)
    to_uid = {
        k: {r["local_id"]: r["instance_uid"] for r in v.instances if r["local_id"]}
        for k, v in labs.items()
    }
    a = np.vectorize(lambda x: to_uid["L"].get(x, 0))(labs["L"].instance[:, 192:])
    b = np.vectorize(lambda x: to_uid["R"].get(x, 0))(labs["R"].instance[:, :64])
    assert np.array_equal(a, b)
    crosser = max(f.instance_uid for f in feats if f.visual_id == 7)
    assert crosser in to_uid["L"].values() and crosser in to_uid["R"].values()


def test_converter_selftest_writes_figures(tmp_path):
    res = converter_selftest(tmp_path)
    assert res["passed"], res["checks"]
    assert len(list(tmp_path.glob("SYNTHETIC_*.png"))) == 2


def test_writer_roundtrip_dimensions_nodata_and_filled_band(tmp_path):
    tile = synthetic.make_tile("W", 0, 0)
    feats = synthetic.features()
    image, obs = synthetic.paint_raster(tile, feats)
    lab = build_tile_labels(tile_affine(tile), (256, 256), obs, feats)
    filled = np.zeros_like(obs)
    filled[0, 0] = 1
    tile["layers"] = {"normalized": "x"}
    rec = write_sample(
        tmp_path, tile, {"normalized": image, "observed": obs, "filled_mask": filled}, lab, {}
    )
    with (
        rasterio.open(tmp_path / rec["files"]["image"]) as img,
        rasterio.open(tmp_path / rec["files"]["instance"]) as ins,
    ):
        assert img.shape == ins.shape == (256, 256)
        assert img.transform == ins.transform == tile_affine(tile)
        assert img.crs is None and img.nodata == -9999.0
        band = img.read()
        assert np.array_equal(band[0], image) and (band[0] == -9999).sum() == (obs == 0).sum()
        assert (
            band[2][0, 0] == 1 and band[2].sum() == 1
        )  # interpolated pixels remain distinguishable
        assert np.array_equal(ins.read(1), lab.instance)
    meta = json.loads((tmp_path / "metadata/tiles/W.json").read_text())
    assert set(meta["instance_id_to_semantic_id"].values()) <= set(TAX.visual.values())


# ----------------------------------------------------------------- alignment
@pytest.mark.parametrize("shift", [0.0, 0.10, -0.15])
def test_displacement_measured_from_intensity(shift):
    tile = synthetic.make_tile("D", 0, 0)
    line = LineString([(1001, 1997), (1011, 1997)])
    feat = LabelFeature(1, "l", 1, "line", line, 0.15)
    image, _ = synthetic.paint_raster(tile, [feat], shift=(0.0, shift), missing=0.1)
    img = np.where(image < -1, np.nan, image).astype(float)
    m = al.measure_line_displacement([line], img, tile_affine(tile))
    assert m["samples"] > 10
    assert m["median_signed"] == pytest.approx(shift, abs=0.03)


def test_axis_hypotheses_and_disjoint_frames():
    hyp = al.axis_hypotheses((-38000, 166000, 5000, 177000), (-57000, -98000, -48000, -88000))
    assert not any(hyp.values())
    assert al.axis_hypotheses((10, 20, 11, 21), (20, 10, 21, 11))["swap_xy"]
    tiles = [synthetic.make_tile("T", 0, 0)]
    far = gpd.GeoDataFrame(
        {"source_file": ["f"], "declared_crs": ["X"]}, geometry=[LineString([(0, 0, 1), (1, 1, 1)])]
    )
    assert al.feature_overlap(far, tiles)["features_intersecting_tiles"] == 0


# ----------------------------------------------------------------- taxonomy and QC
def test_taxonomy_levels():
    d = TAX.classify_line({"DESIGNACAO": "Linha descontínua", "TRAÇO": "4.0"})
    assert (d.visual_class, d.client_code) == ("longitudinal_dash", "2")
    d = TAX.classify_line({"DESIGNACAO": "Guia direita", "TRAÇO": None})
    assert (d.visual_class, d.client_code) == ("longitudinal_solid", "13")
    d = TAX.classify_line({"DESIGNACAO": "Linha mista", "TRAÇO": "4.0"})
    assert d.client_code is None and "client_code_ambiguous" in d.flags
    assert "unknown_designation" in TAX.classify_line({"DESIGNACAO": "Other"}).flags
    assert TAX.classify_polygon({"COD_RST": "M15d"}).visual_class == "arrow"
    assert "missing_class_code" in TAX.classify_polygon({"COD_RST": "NULL"}).flags
    assert "unknown_class_code" in TAX.classify_polygon({"COD_RST": "M99"}).flags
    assert TAX.classify_polygon({"COD_RST": None, "LETRA": "A"}).visual_class == "character"


def _write_gpkg(path, rows, geoms, layer):
    gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:3763").to_file(
        path, layer=layer, driver="GPKG"
    )


def test_feature_qc_statuses_and_original_files_untouched(tmp_path):
    lines = tmp_path / "lines.gpkg"
    polys = tmp_path / "polys.gpkg"
    _write_gpkg(
        lines,
        [
            {
                "ID_SIG": 1.0,
                "DESIGNACAO": "Linha descontínua",
                "LARGURA": 0.15,
                "TRAÇO": "4.0",
                "ESPAÇO": "10.0",
                "ELEVATION": "5",
            },
            {
                "ID_SIG": 2.0,
                "DESIGNACAO": "Linha contínua",
                "LARGURA": None,
                "TRAÇO": None,
                "ESPAÇO": None,
                "ELEVATION": "5",
            },
        ],
        [LineString([(0, 0, 5), (4, 0, 5)]), LineString([(0, 5, 5), (50, 5, 5)])],
        "L",
    )
    bowtie = Polygon([(0, 0, 1), (1, 1, 1), (1, 0, 1), (0, 1, 1)])
    _write_gpkg(
        polys,
        [
            {"ID_SIG": 1, "COD_RST": "M15", "LARGURA": None},
            {"ID_SIG": 2, "COD_RST": "NULL", "LARGURA": None},
            {"ID_SIG": 3, "COD_RST": "M17", "LARGURA": "0.3"},
            {"ID_SIG": 4, "COD_RST": "M17", "LARGURA": "0.3"},
        ],
        [MultiPolygon([box(10, 10, 11, 11)]), box(20, 20, 21, 21), bowtie, box(30, 30, 31, 31)],
        "P",
    )
    before = {p: p.read_bytes() for p in (lines, polys)}
    df = normalize_features([lines, polys], TAX, (0.05, 1.0))
    assert df["global_id"].is_unique and list(df["instance_uid"]) == list(range(1, 7))
    assign_status(df, alignment_passed=False, reviews={})
    status = dict(zip(df["global_id"], df["qc_status"], strict=True))
    assert status["lines.gpkg|L|fid=1"] == "NEEDS_REVIEW"
    assert status["lines.gpkg|L|fid=2"] == "INCOMPLETE"  # no width
    assert status["polys.gpkg|P|fid=2"] == "INCOMPLETE"  # NULL code
    assert status["polys.gpkg|P|fid=3"] == "UNUSABLE"  # self-intersection
    assert "VERIFIED" not in status.values()
    # explicit review cannot verify without the alignment gate
    reviews = {g: "VERIFIED" for g in df["global_id"]}
    assign_status(df, alignment_passed=False, reviews=reviews)
    assert "VERIFIED" not in set(df["qc_status"])
    assign_status(df, alignment_passed=True, reviews=reviews)
    assert (
        dict(zip(df["global_id"], df["qc_status"], strict=True))["lines.gpkg|L|fid=1"] == "VERIFIED"
    )
    assert (
        dict(zip(df["global_id"], df["qc_status"], strict=True))["polys.gpkg|P|fid=3"] == "UNUSABLE"
    )
    assert all(p.read_bytes() == b for p, b in before.items())
    attrs = json.loads(df.loc[df["global_id"] == "lines.gpkg|L|fid=1", "attributes"].iloc[0])
    assert attrs["ESPAÇO"] == "10.0"  # unknown/extra attributes retained verbatim


def test_duplicates_across_files_are_flagged(tmp_path):
    for name in ("a.gpkg", "b.gpkg"):
        _write_gpkg(tmp_path / name, [{"ID_SIG": 1, "COD_RST": "M15"}], [box(0, 0, 1, 1)], "P")
    df = normalize_features([tmp_path / "a.gpkg", tmp_path / "b.gpkg"], TAX, (0.05, 1.0))
    assert all("duplicate_in_other_file" in r for r in df["qc_reasons"])
    assert df["duplicate_of"].isna().tolist() == [True, False]
    assert df["duplicate_of"].iloc[1] == "a.gpkg|P|fid=1"


# ----------------------------------------------------------------- splits
def _strip(n=30):
    return [synthetic.make_tile(f"t{i:02d}", 0, i * 192) for i in range(n)]  # 64-px overlaps


def test_spatial_split_no_overlap_no_shared_instances():
    tiles = _strip()
    inst = {t["tile_id"]: {i, i + 1} for i, t in enumerate(tiles)}  # each instance spans 2 tiles
    classes = {t["tile_id"]: Counter({1: 2}) for t in tiles}
    res = spatial_split(tiles, inst, classes, block_tiles=3, min_instances_per_split=2)
    assert res["leakage"]["passed"]
    assert res["guard_tiles"]
    assert {s for s in res["assignment"].values()} == {"train", "val", "test"}
    assert res["splits"]["train"]["tiles"] > res["splits"]["test"]["tiles"]
    # an injected leak is detected
    bad = dict(res["assignment"])
    t0 = next(t for t, s in bad.items() if s == "train")
    bad[t0] = "test"
    assert not check_leakage(bad, tiles, inst)["passed"]


def test_long_instances_are_segmented_not_leaked_and_small_sets_reported():
    tiles = _strip(12)
    inst = {t["tile_id"]: {999} for t in tiles}  # one continuous edge line everywhere
    res = spatial_split(
        tiles, inst, {}, block_tiles=2, min_instances_per_split=50, long_instances={999}
    )
    assert res["leakage"]["passed"]
    assert res["leakage"]["long_instances_segmented_across_splits"] == 1
    assert set(res["insufficient_splits"]) == {"val", "test"}


# ----------------------------------------------------------------- end to end
class _Cfg:
    def __init__(self, root):
        self.root = root

    def path(self, key):
        return {
            "outputs.reports": self.root / "outputs/reports",
            "data.processed": self.root / "data/processed",
            "data.raw_annotations": self.root / "data/raw/annotations",
            "data.reference_gis": self.root / "data/raw/reference_gis",
        }[key]


def _project(tmp_path, offset=(0.0, 0.0), n_tiles=24):
    feats_by_tile, manifest = [], []
    tiles_dir = tmp_path / "data/processed/phase3/tiles/synthetic/full_x"
    rows, geoms = [], []
    for i in range(n_tiles):
        t = synthetic.make_tile(f"synthetic__c{i:03d}", 0, i * 192)
        x0 = t["native_xy_bbox"][0]
        ln = LineString([(x0 + 1, 1998, 3.0), (x0 + 9, 1998, 3.0)])
        arr = box(x0 + 2, 1994, x0 + 4, 1995)
        rows += [
            {
                "ID_SIG": float(2 * i),
                "DESIGNACAO": "Linha descontínua",
                "LARGURA": 0.15,
                "TRAÇO": "8.0",
                "ESPAÇO": "4.0",
                "ELEVATION": "3",
            }
        ]
        geoms.append(ln)
        feats = [LabelFeature(1, "l", 2, "line", ln, 0.15), LabelFeature(2, "p", 3, "polygon", arr)]
        image, obs = synthetic.paint_raster(t, feats, shift=offset, seed=i, missing=0.1)
        d = tiles_dir / t["tile_id"]
        d.mkdir(parents=True)
        layers = {}
        prof = {
            "driver": "GTiff",
            "height": 256,
            "width": 256,
            "count": 1,
            "transform": tile_affine(t),
        }
        for name, arr_, dt in (
            ("normalized", image, "float32"),
            ("observed", obs, "uint8"),
            ("count", obs, "uint32"),
            ("filled_mask", np.zeros_like(obs), "uint8"),
            ("road_state", np.ones_like(obs), "uint8"),
        ):
            p = d / f"{name}.tif"
            with rasterio.open(p, "w", dtype=dt, **prof) as dst:
                dst.write(arr_.astype(dt), 1)
            layers[name] = str(p)
        t["layers"] = layers
        manifest.append(t)
        feats_by_tile.append(arr)
    meta = tmp_path / "data/processed/phase3/metadata"
    meta.mkdir(parents=True)
    (meta / "tile_manifest.json").write_text(json.dumps(manifest))
    ann = tmp_path / "data/raw/annotations"
    ann.mkdir(parents=True)
    (tmp_path / "data/raw/reference_gis").mkdir(parents=True)
    gpd.GeoDataFrame(rows, geometry=geoms).to_file(ann / "lines.gpkg", layer="L", driver="GPKG")
    gpd.GeoDataFrame(
        [{"ID_SIG": i, "COD_RST": "M15"} for i in range(n_tiles)], geometry=feats_by_tile
    ).to_file(ann / "other.gpkg", layer="P", driver="GPKG")
    (tmp_path / "outputs/reports/phase3").mkdir(parents=True)
    (tmp_path / "outputs/reports/phase3/phase3_report.json").write_text("{}")
    (tmp_path / "outputs/reports/lidar_cache").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/phase4_taxonomy.yaml").write_text(
        (ROOT / "configs/phase4_taxonomy.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    review = tmp_path / "review.csv"
    with open(review, "w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["global_id", "status"])
        for i in range(n_tiles):
            w.writerow([f"lines.gpkg|L|fid={i + 1}", "VERIFIED"])
            w.writerow([f"other.gpkg|P|fid={i + 1}", "VERIFIED"])
    return review


OPTS = {"min_alignment_samples": 50, "min_instances_per_split": 2, "split_block_tiles": 3}


def test_end_to_end_aligned_dataset(tmp_path):
    review = _project(tmp_path)
    rep = run_phase4(_Cfg(tmp_path), tmp_path, review, OPTS)
    assert rep["alignment"]["status"] == "PASSED_SHARED_NATIVE_FRAME"
    assert rep["alignment"]["displacement"]["median_abs"] < 0.05
    assert rep["dataset"]["generated"] and rep["dataset"]["images"] == 24
    assert rep["dataset"]["polygon_instances"] > 0 and rep["dataset"]["polyline_instances"] > 0
    assert rep["splits"]["leakage"]["passed"]
    out = Path(rep["output_root"])
    manifest = json.loads((out / "metadata/dataset_manifest.json").read_text())
    for rec in manifest:
        with (
            rasterio.open(out / rec["files"]["image"]) as a,
            rasterio.open(out / rec["files"]["instance"]) as b,
        ):
            assert a.shape == b.shape and a.transform == b.transform
            assert b.read(1).max() >= 1
    assert rep["integrity"]["client_annotation_files_unchanged"]


def test_end_to_end_misaligned_annotations_produce_no_labels(tmp_path):
    review = _project(tmp_path, offset=(0.0, 0.3))
    rep = run_phase4(_Cfg(tmp_path), tmp_path, review, OPTS)
    assert rep["alignment"]["status"] == "FAILED_DISPLACEMENT_ABOVE_TOLERANCE"
    assert not rep["dataset"]["generated"]
    assert not any((Path(rep["output_root"]) / "instance_masks").glob("*"))
    assert rep["annotation_quality"]["status_counts"]["VERIFIED"] == 0


def test_end_to_end_disjoint_frames_stop_at_gate(tmp_path):
    review = _project(tmp_path)
    ann = tmp_path / "data/raw/annotations"
    for p in ann.glob("*.gpkg"):
        p.unlink()
    gpd.GeoDataFrame(
        [{"ID_SIG": 1, "COD_RST": "M15"}],
        geometry=[box(-38000, 170000, -37999, 170001)],
        crs="EPSG:3763",
    ).to_file(ann / "far.gpkg", layer="P", driver="GPKG")
    rep = run_phase4(_Cfg(tmp_path), tmp_path, review, OPTS)
    assert rep["alignment"]["status"] == "BLOCKED_NO_SPATIAL_OVERLAP"
    assert rep["alignment"]["transformation_applied"] is False
    assert not rep["dataset"]["generated"] and rep["phase5_ready"] is False
