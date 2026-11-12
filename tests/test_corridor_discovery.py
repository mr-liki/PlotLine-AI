"""Known-truth architectural tests: measurement retention and honest uncertainty."""

import json

import numpy as np
import pytest
from affine import Affine
from shapely.geometry import LineString, box, mapping

from roadmarking.preprocessing.corridor_boundaries import build_boundaries
from roadmarking.preprocessing.corridor_discovery import discover, load_references
from roadmarking.preprocessing.intensity import calibrate_histogram
from roadmarking.preprocessing.road_cross_sections import (
    SectionSettings,
    extract_sections,
    stations,
)
from roadmarking.preprocessing.stripe_reference import StripeSettings, detect_references
from roadmarking.quality.marking_retention import manual_retention, reviewed_samples
from roadmarking.rasterization.rasterizer import rasterize_search_corridor, world_to_pixel


def road(kind="straight"):
    x, y = np.meshgrid(np.arange(0, 24.01, 0.10), np.arange(-12, 12.01, 0.10))
    bend = 0.008 * x * x if kind == "curved" else np.zeros_like(x)
    yy = y + bend
    z = 5 + 0.03 * x + 0.04 * y
    if kind == "banked":
        z = 5 + 0.03 * x + 0.15 * y
    if kind == "embankment":
        z += np.maximum(np.abs(y) - 8, 0) * 0.6
    if kind in ("sidewalk", "levels"):
        z += (y > 6) * (0.16 if kind == "sidewalk" else 2.0)
    if kind == "width_change":
        z += (np.abs(y) > (5 + 0.2 * x)) * 0.5
    if kind == "roof":
        z += 8
    xyz = np.column_stack((x.ravel(), yy.ravel(), z.ravel()))
    if kind == "gap":
        xyz = xyz[(xyz[:, 0] < 10) | (xyz[:, 0] > 13)]
    coords = np.column_stack(
        (
            np.arange(1, 24, 1),
            0.008 * np.arange(1, 24, 1) ** 2 if kind == "curved" else np.zeros(23),
        )
    )
    return xyz, [{"id": "known_reference", "coords": coords.tolist(), "reference_supported": True}]


@pytest.mark.parametrize(
    "kind",
    [
        "straight",
        "curved",
        "embankment",
        "banked",
        "multilane",
        "gap",
        "sidewalk",
        "width_change",
        "levels",
        "roof",
        "intersection",
    ],
)
def test_profiles_known_geometry_and_status(kind):
    xyz, refs = road(kind)
    if kind == "intersection":
        refs += [{"id": "crossing", "coords": [[12, -10], [12, 10]], "reference_supported": True}]
    cfg = SectionSettings(half_width_m=14, spacing_m=2)
    profiles = extract_sections(xyz, refs, cfg)
    result = build_boundaries(profiles, cfg)
    assert profiles and result["review_status"] == "PROPOSED_UNREVIEWED"
    assert not result["strict"].is_empty
    assert all(p["review_status"] == "PROPOSED_UNREVIEWED" for p in profiles)
    if kind in ("levels", "sidewalk"):
        assert result["strict"].intersection(box(2, 7, 22, 11)).area == 0
        assert not result["uncertain"].intersection(box(2, 7, 22, 11)).is_empty
    if kind == "gap":
        assert result["strict"].intersection(box(10.8, -2, 12.2, 2)).is_empty
    if kind == "width_change":
        widths = [d["width_m"] for d in result["diagnostics"] if d["width_m"]]
        assert max(widths) - min(widths) > 2
    if kind == "roof":
        # Smoothness + a provisional line cannot establish road semantics.
        assert "smooth_edges_not_semantically_verified" in result["diagnostics"][0]["uncertainty"]


def stripe_image(broken=False, curved=False, parallel=False):
    image = np.full((160, 400), 0.1, np.float32)
    for c in range(10, 390):
        if broken and c % 60 >= 35:
            continue
        row = 70 + round(0.0002 * (c - 200) ** 2) if curved else 70
        image[row : row + 2, c] = 0.95
        if parallel:
            image[110:112, c] = 0.95
    return image, np.ones(image.shape, bool), Affine(0.1, 0, 0, 0, -0.1, 16)


@pytest.mark.parametrize("kind", ["solid", "broken", "curved", "parallel_nonroad"])
def test_independent_stripe_proposals(kind):
    image, observed, transform = stripe_image(
        kind == "broken", kind == "curved", kind == "parallel_nonroad"
    )
    proposals, evidence = detect_references(image, observed, transform)
    assert proposals and evidence["candidate_paint"].sum() > 100
    assert all(p["review_status"] == "PROPOSED_UNREVIEWED" for p in proposals)
    assert any(p["length_m"] >= 8 for p in proposals)
    if kind == "parallel_nonroad":
        centers = [np.mean(np.asarray(p["coords"])[:, 1]) for p in proposals]
        assert max(centers) - min(centers) > 2


def test_large_unobserved_gap_not_bridged():
    image, observed, transform = stripe_image()
    observed[:, 150:250] = False
    proposals, _ = detect_references(image, observed, transform, StripeSettings(max_gap_m=15))
    assert not any(
        min(np.asarray(p["coords"])[:, 0]) < 15 and max(np.asarray(p["coords"])[:, 0]) > 25
        for p in proposals
    )


def test_cross_section_coordinate_accuracy_native_xy():
    distances, centers, tangents = stations([[-57000, -89000], [-56997, -88996]], 1)
    assert np.allclose(distances, np.arange(6))
    assert np.allclose(centers[-1], [-56997, -88996])
    normal = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    assert np.allclose((normal * tangents).sum(axis=1), 0)
    assert np.allclose(np.linalg.norm(normal, axis=1), 1)


def test_measurements_outside_strict_survive_and_raw_preserved():
    xyz = np.array([[0.02, 0.98, 3], [0.07, 0.98, 3], [0.07, 0.98, 9], [0.97, 0.03, 3]])
    intensity = np.array([20, 40000, 100, 65535], np.uint16)
    calibration = calibrate_histogram(np.bincount(intensity, minlength=65536))
    transform = Affine(0.05, 0, 0, 0, -0.05, 1)
    before = xyz.copy()
    layers, kept = rasterize_search_corridor(
        xyz, intensity, transform, (20, 20), calibration, box(0, 0, 1, 1), box(0, 0, 0.05, 1)
    )
    assert kept.all() and np.array_equal(xyz, before)
    assert layers["maximum"][0, 1] == 40000
    assert layers["mean"][0, 1] == 20050
    assert layers["count"][0, 1] == 2
    assert layers["uncertain_road"][0, 1] == 1
    assert layers["confirmed_road"].sum() == 0
    assert layers["normalized"][5, 5] == -9999
    assert layers["filled_mask"].sum() == 0


def test_overlapping_grids_exact_and_deterministic():
    rng = np.random.default_rng(12)
    xyz = rng.uniform([0, 0, 2], [90, 51, 3], (10000, 3))
    raw = rng.integers(0, 65536, len(xyz), dtype=np.uint16)
    calibration = calibrate_histogram(np.bincount(raw, minlength=65536))
    transform = Affine(0.05, 0, 0, 0, -0.05, 51.2)
    a, _ = rasterize_search_corridor(
        xyz, raw, transform, (1024, 1024), calibration, box(0, 0, 100, 52), box(0, 0, 100, 20)
    )
    b, _ = rasterize_search_corridor(
        xyz,
        raw,
        transform * Affine.translation(768, 0),
        (1024, 1024),
        calibration,
        box(0, 0, 100, 52),
        box(0, 0, 100, 20),
    )
    repeat, _ = rasterize_search_corridor(
        xyz[::-1],
        raw[::-1],
        transform,
        (1024, 1024),
        calibration,
        box(0, 0, 100, 52),
        box(0, 0, 100, 20),
    )
    for name in a:
        assert np.array_equal(a[name][:, 768:], b[name][:, :256])
        assert np.array_equal(a[name], repeat[name])
    c, r = world_to_pixel(transform, xyz[:, 0], xyz[:, 1])
    assert np.all((xyz[:, 0] - (c + 0.5) * 0.05) <= 0.025 + 1e-10)
    assert np.isfinite(r).all()


def test_no_reliable_reference_requires_review():
    image = np.ones((100, 100), np.float32) * 0.1
    lines, _, status = discover(
        image,
        np.ones_like(image, bool),
        Affine(0.2, 0, 0, 0, -0.2, 20),
        geometry_xy=np.column_stack((np.arange(20), np.zeros(20))),
    )
    assert status == "REVIEW_REQUIRED_NO_RELIABLE_REFERENCE"
    assert lines and not any(p["reference_supported"] for p in lines)


def test_native_reference_approval_gate(tmp_path):
    path = tmp_path / "alignment.json"
    data = {
        "type": "FeatureCollection",
        "coordinate_frame": "NATIVE_XY",
        "source_sha256": "source",
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(LineString([(1, 1), (8, 1)])),
                "properties": {"review_status": "PROPOSED_UNREVIEWED"},
            }
        ],
    }
    path.write_text(json.dumps(data))
    assert load_references(path, "source", [0, 0, 10, 10])[0]["reference_supported"]
    with pytest.raises(ValueError, match="approved"):
        load_references(path, "source", [0, 0, 10, 10], approved=True)
    with pytest.raises(ValueError, match="hash"):
        load_references(path, "other", [0, 0, 10, 10])
    data["features"][0]["properties"].update(
        review_status="APPROVED", reviewer="human", reviewed_at="2026-11-04"
    )
    path.write_text(json.dumps(data))
    assert load_references(path, "source", [0, 0, 10, 10], approved=True)
    data["crs"] = {"name": "EPSG:3763"}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="NATIVE_XY"):
        load_references(path, "source", [0, 0, 10, 10])


def test_manual_truth_cannot_be_automatic_and_recall_only_visible(tmp_path):
    path = tmp_path / "review.json"
    props = {"review_status": "PROPOSED_UNREVIEWED", "kind": "paint"}
    paint = {
        "type": "Feature",
        "geometry": mapping(LineString([(1, 1), (3, 1)])),
        "properties": props,
    }
    boundary = {
        "type": "Feature",
        "geometry": mapping(box(0, 0, 4, 4)),
        "properties": dict(props, kind="road_boundary"),
    }
    data = {
        "coordinate_frame": "NATIVE_XY",
        "source_sha256": "source",
        "features": [paint, boundary],
    }
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="verified"):
        reviewed_samples(path, "source")
    for feature in data["features"]:
        feature["properties"].update(
            review_status="MANUALLY_VERIFIED", reviewer="human", reviewed_at="2026-11-04"
        )
    path.write_text(json.dumps(data))
    features = reviewed_samples(path, "source")
    observed = np.ones((4, 4), bool)
    retained = observed.copy()
    retained[3, 2] = False
    measured = manual_retention(features, observed, retained, Affine(1, 0, 0, 0, -1, 4))
    assert measured["observed_paint_retention"] < 1


def test_uncertain_width_jump_not_silently_smoothed():
    xyz, refs = road("straight")
    cfg = SectionSettings(half_width_m=14)
    sections = extract_sections(xyz, refs, cfg)
    sections[1]["segments"][0]["right"] += 5
    result = build_boundaries(sections, cfg)
    assert result["diagnostics"][1]["width_m"] is None
    assert (
        "width_or_station_discontinuity_preserved_as_uncertain"
        in result["diagnostics"][1]["uncertainty"]
    )


def test_strict_stream_partition_source_immutability_and_cache(tmp_path, monkeypatch):
    import laspy

    from roadmarking.analysis.corridor_pilot import load_window, partition_source
    from roadmarking.rasterization.pipeline import INTACT, sha256

    xyz, _ = road()
    xyz = xyz[::10]
    source = "Job_20260325_0838_Track01_3"
    path = tmp_path / f"{source}.laz"
    data = laspy.LasData(laspy.LasHeader(point_format=3, version="1.2"))
    data.header.scales = [0.001] * 3
    data.x, data.y, data.z = xyz.T
    data.intensity = np.arange(len(xyz), dtype=np.uint16)
    data.return_number = np.ones(len(xyz), np.uint8)
    data.number_of_returns = np.ones(len(xyz), np.uint8)
    data.gps_time = np.full(len(xyz), 100.0)
    data.write(path)
    before = sha256(path)
    monkeypatch.setitem(INTACT, source, len(xyz))
    settings = {"pilot_halo_m": 0, "partition_m": 8, "chunk_size": 1000}
    tiles = [{"native_xy_bbox": [0, -10, 20, 10]}]
    cache = tmp_path / "cache"
    result = partition_source(path, tiles, cache, settings, before)
    assert sha256(path) == before and result["invalid_points"] == 0
    records = load_window(cache, [0, -10, 20, 10], 8)
    assert len(records) == result["unique_cached_points"]
    assert len(np.unique(records["source_index"])) == len(records)
    repeated = partition_source(path, tiles, cache, settings, before)
    assert repeated == json.loads(json.dumps(result))
    partition = next(cache.glob("*.bin"))
    with partition.open("r+b") as f:
        f.write(b"broken")
    with pytest.raises(ValueError, match="integrity"):
        partition_source(path, tiles, cache, settings, before)


def test_reference_mode_requires_line_and_geometry_fallback_is_review_only():
    image = np.ones((20, 20), np.float32)
    observed = np.ones_like(image, bool)
    transform = Affine(0.2, 0, 0, 0, -0.2, 4)
    with pytest.raises(ValueError, match="supplied"):
        discover(image, observed, transform, mode="reference")
    lines, _, status = discover(
        image,
        observed,
        transform,
        mode="geometry",
        geometry_xy=np.column_stack((np.arange(20), np.zeros(20))),
    )
    assert status.startswith("REVIEW_REQUIRED")
    assert not lines[0]["reference_supported"]


def test_local_contrast_acquisition_gain_invariance():
    # Shared-file calibrated contrast is used, not a raw global fixed threshold.
    from roadmarking.preprocessing.stripe_reference import local_contrast

    image, observed, _ = stripe_image()
    _, bright_a = local_contrast(image, observed, 0.1)
    _, bright_b = local_contrast(0.5 * image, observed, 0.1)
    assert (bright_a & bright_b).sum() / bright_a.sum() > 0.95


def test_old_boundary_stripe_retained_when_strict_classification_empty():
    xyz = np.column_stack((np.arange(0, 5, 0.05) + 0.025, np.full(100, 2.025), np.full(100, 3)))
    raw = np.full(100, 45000, np.uint16)
    calibration = calibrate_histogram(np.bincount(raw, minlength=65536))
    layers, keep = rasterize_search_corridor(
        xyz,
        raw,
        Affine(0.05, 0, 0, 0, -0.05, 5),
        (100, 100),
        calibration,
        box(0, 0, 5, 5),
        box(0, 0, 5, 1),
    )
    assert keep.all() and layers["observed"].sum() == 100
    assert layers["strict_surface_candidate"][layers["observed"] > 0].sum() == 0
    assert layers["maximum"][layers["observed"] > 0].min() == 45000


@pytest.mark.parametrize("approved", [False, True])
def test_qgis_export_never_manufactures_approval(tmp_path, monkeypatch, approved):
    import importlib.util
    from pathlib import Path

    import geopandas as gpd

    from roadmarking.rasterization.pipeline import sha256

    script = Path(__file__).resolve().parents[1] / "scripts/export_corridor_review.py"
    spec = importlib.util.spec_from_file_location("corridor_review_export_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = "Job_20260325_0838_Track01_3"
    original = tmp_path / f"{source}.laz"
    original.write_bytes(b"immutable source identity fixture")
    digest = sha256(original)
    monkeypatch.setattr(
        module,
        "read_json",
        lambda _: [
            {
                "source_file": str(original),
                "source_sha256": digest,
                "native_xy_bbox": [0, 0, 10, 10],
            }
        ],
    )
    package = tmp_path / "review.gpkg"
    gpd.GeoDataFrame(
        [
            {
                "review_status": "APPROVED" if approved else "PROPOSED_UNREVIEWED",
                "reviewer": "reviewer" if approved else "",
                "reviewed_at": "2026-11-04" if approved else "",
                "geometry": LineString([(1, 1), (8, 1)]),
            }
        ],
        crs=None,
    ).to_file(package, layer="reference_candidates", driver="GPKG")
    before = sha256(package)
    output = tmp_path / "reference.json"
    args = [str(package), "--source", source, "--kind", "reference", "--output", str(output)]
    if approved:
        module.main(args)
        assert load_references(output, digest, [0, 0, 10, 10], approved=True)
        with pytest.raises(ValueError, match="overwrite"):
            module.main(args)
    else:
        with pytest.raises(ValueError, match="approved/verified"):
            module.main(args)
        assert not output.exists()
    assert sha256(original) == digest and sha256(package) == before


def test_native_seed_hint_is_not_a_verified_alignment(tmp_path):
    from roadmarking.preprocessing.corridor_discovery import load_seed_points

    path = tmp_path / "seeds.geojson"
    path.write_text(
        json.dumps(
            {
                "coordinate_frame": "NATIVE_XY",
                "source_sha256": "source",
                "features": [{"geometry": {"type": "Point", "coordinates": [5, 0]}}],
            }
        )
    )
    seeds = load_seed_points(path, "source", [0, -1, 100, 1])
    image = np.ones((20, 20), np.float32) * 0.1
    lines, _, status = discover(
        image,
        np.ones_like(image, bool),
        Affine(0.2, 0, 0, 0, -0.2, 4),
        mode="geometry",
        geometry_xy=np.column_stack((np.arange(100), np.zeros(100))),
        seed_points=seeds,
    )
    assert status.startswith("REVIEW_REQUIRED") and not lines[0]["reference_supported"]
    assert np.asarray(lines[0]["coords"])[:, 0].max() <= 20
    with pytest.raises(ValueError, match="hash"):
        load_seed_points(path, "different", [0, -1, 100, 1])
