"""Finite evidence, type separation and real world-coordinate continuity contracts."""

import copy

import numpy as np
import pytest
from affine import Affine
from shapely.geometry import LineString, box

from roadmarking.preprocessing.stripe_tracking import (
    LineEvidence,
    TrackingSettings,
    _candidate,
    clip_candidates,
    graph_paths,
    line_quality,
    stitch_candidates,
    tangent,
    thin,
    trace_stripes,
)


def scene(kind="straight"):
    image = np.full((160, 420), 0.1, np.float32)
    for c in range(20, 400):
        row = 60 + round(10 * np.sin(c / 90)) if kind == "curve" else 60
        if kind == "dash" and c % 70 >= 40:
            continue
        if kind == "long_gap" and 160 < c < 240:
            continue
        if kind == "short_gap" and 200 < c < 202:
            continue
        image[row : row + 2, c] = 0.95
        if kind == "parallel":
            image[100:102, c] = 0.95
    observed = np.ones(image.shape, bool)
    if kind == "nodata":
        observed[:, 180:220] = False
        image[:, 180:220] = -9999
    return LineEvidence(image, observed, Affine(0.1, 0, -56000, 0, -0.1, -89000))


@pytest.mark.parametrize(
    "kind", ["straight", "curve", "parallel", "dash", "long_gap", "short_gap", "nodata"]
)
def test_finite_measured_tracking(kind):
    evidence = scene(kind)
    before = evidence.image.copy()
    candidates, qa, _ = trace_stripes(evidence)
    assert candidates or qa
    assert np.array_equal(evidence.image, before)
    for candidate in candidates + qa:
        xy = np.asarray(candidate["coords"])
        assert np.isfinite(xy).all()
        assert xy[:, 0].min() >= -56000 and xy[:, 0].max() <= -55958
        assert candidate["review_status"] == "PROPOSED_UNREVIEWED"
        assert candidate["role"] == "CANDIDATE_PAINTED_STRIPE"
    if kind == "straight":
        assert candidates and max(c["quality"]["total_length"] for c in candidates) > 30
    if kind == "curve":
        long = max(candidates + qa, key=lambda c: c["quality"]["total_length"])
        assert np.ptp(np.asarray(long["coords"])[:, 1]) > 1
        xy = np.asarray(long["coords"])
        column = (xy[:, 0] + 56000) / 0.1 - 0.5
        true_center = -89000 - (60 + np.rint(10 * np.sin(column / 90)) + 1) * 0.1
        assert np.percentile(np.abs(xy[:, 1] - true_center), 95) < 0.2
    if kind == "parallel":
        means = [np.mean(np.asarray(c["coords"])[:, 1]) for c in candidates]
        assert max(means) - min(means) > 3
    if kind in ("long_gap", "nodata"):
        assert not any(
            LineString(c["coords"]).bounds[0] < -55984
            and LineString(c["coords"]).bounds[2] > -55976
            for c in candidates
        )


def test_local_tangent_and_junction_topology():
    assert np.allclose(tangent([[0, 0], [1, 1], [2, 2]]), [2**-0.5, 2**-0.5])
    assert np.allclose(tangent([[0, 0], [1, 1], [2, 2]], True), [2**-0.5, 2**-0.5])
    mask = np.zeros((21, 21), bool)
    mask[10, 2:19] = True
    mask[2:11, 10] = True
    paths = graph_paths(thin(mask))
    assert len(paths) >= 3
    assert all(not (path[0] == (10, 2) and path[-1] == (2, 10)) for path in paths)


def known_candidate(coords, evidence):
    return _candidate(coords, evidence, TrackingSettings(), 0.2)


def test_short_supported_join_and_long_unpainted_gap_rejected():
    evidence = scene()
    y = -89006.05
    a = known_candidate([[-55996, y], [-55990, y]], evidence)
    b = known_candidate([[-55989.8, y], [-55984, y]], evidence)
    merged, details = stitch_candidates([a, b], evidence)
    assert len(merged) == 1 and len(details["joins"]) == 1
    assert details["joins"][0]["status"] == "SHORT_PAINT_SUPPORTED_INTERPOLATION"
    c = known_candidate([[-55978, y], [-55972, y]], evidence)
    separated, _ = stitch_candidates([a, c], evidence)
    assert len(separated) == 2


def test_observed_bare_pavement_is_not_paint_support():
    evidence = scene("long_gap")
    quality = line_quality([[-55985, -89006.05], [-55975, -89006.05]], evidence)
    assert quality["outside_observed_fraction"] == 0
    assert quality["unsupported_length"] > 5


def test_dash_group_does_not_add_painted_geometry():
    evidence = scene()
    y = -89006.05
    a = known_candidate([[-55996, y], [-55991, y]], evidence)
    b = known_candidate([[-55988, y], [-55983, y]], evidence)
    before = [copy.deepcopy(c["coords"]) for c in (a, b)]
    output, details = stitch_candidates([a, b], evidence)
    assert len(output) == 2 and details["dash_groups"] == 1
    assert {c["dash_group_id"] for c in output} == {"dash_group_0000"}
    assert sorted(c["coords"] for c in output) == sorted(before)


def test_duplicates_removed_parallel_lines_preserved():
    evidence = scene("parallel")
    a = known_candidate([[-55996, -89006.05], [-55984, -89006.05]], evidence)
    duplicate = copy.deepcopy(a)
    duplicate["id"] = "other_tile_id"
    b = known_candidate([[-55996, -89010.05], [-55984, -89010.05]], evidence)
    output, details = stitch_candidates([a, duplicate, b], evidence)
    assert len(output) == 2 and len(details["duplicate_ids"]) == 1


def test_world_coordinate_tile_clipping_has_common_parent():
    evidence = scene()
    candidate = known_candidate([[-55996, -89006.05], [-55964, -89006.05]], evidence)
    first = clip_candidates([candidate], box(-56000, -89016, -55975, -89000), evidence)
    second = clip_candidates([candidate], box(-55980, -89016, -55958, -89000), evidence)
    assert first[0]["parent_id"] == second[0]["parent_id"]
    shared = LineString(first[0]["coords"]).intersection(LineString(second[0]["coords"]))
    assert shared.length == pytest.approx(5)
    assert LineString(first[0]["coords"]).bounds[2] <= -55975


def test_unsupported_geometry_flagged_and_no_final_label():
    evidence = scene()
    candidate = known_candidate([[-55995, -89003], [-55970, -89003]], evidence)
    assert "INSUFFICIENT_PAINT_SUPPORT" in candidate["flags"]
    assert candidate["quality"]["supported_length"] == 0
    assert candidate["role"] != "VERIFIED_ROAD_MARKING"


def test_deterministic_immutable_tracking_and_explicit_nodata():
    evidence = scene("curve")
    before = evidence.image.tobytes(), evidence.observed.tobytes()
    a = trace_stripes(evidence)
    b = trace_stripes(evidence)
    assert a == b
    assert (evidence.image.tobytes(), evidence.observed.tobytes()) == before
    empty = LineEvidence(
        np.zeros((20, 20)), np.zeros((20, 20), bool), Affine(0.1, 0, 0, 0, -0.1, 2)
    )
    assert trace_stripes(empty)[0] == []


@pytest.mark.parametrize("coords", [[[np.nan, 0], [1, 1]], [[0, 0], [0, 0]]])
def test_invalid_or_zero_geometry_never_becomes_candidate(coords):
    assert known_candidate(coords, scene()) is None


def test_sampling_hole_preserves_supported_stroke_parts():
    evidence = scene()
    evidence.observed[:, 210] = False
    candidates, qa, _ = trace_stripes(evidence)
    assert candidates and qa
    assert sum(c["quality"]["supported_length"] for c in candidates) > 20
    assert all(c["quality"]["outside_observed_fraction"] == 0 for c in candidates)


def test_no_join_across_nodata_even_when_endpoints_are_close():
    evidence = scene()
    evidence.observed[:, 100] = False
    y = -89006.05
    a = known_candidate([[-55996, y], [-55990.05, y]], evidence)
    b = known_candidate([[-55989.85, y], [-55984, y]], evidence)
    output, stats = stitch_candidates([a, b], evidence)
    assert len(output) == 2 and not stats["joins"]


def test_qgis_layer_roles_and_empty_verified_output(tmp_path):
    import geopandas as gpd
    import pyogrio

    from roadmarking.analysis.phase3_2_lines import _frame, _line_row

    evidence = scene()
    candidate = known_candidate([[-55996, -89006.05], [-55984, -89006.05]], evidence)
    package = tmp_path / "native_review.gpkg"
    row = _line_row(candidate)
    _frame([row]).to_file(package, layer="candidate_painted_stripes", driver="GPKG")
    _frame([dict(row, role="PROVISIONAL_CORRIDOR_REFERENCE_NOT_MARKING")]).to_file(
        package, layer="candidate_reference_alignments", driver="GPKG"
    )
    _frame([]).to_file(
        package, layer="verified_road_markings", driver="GPKG", geometry_type="LineString"
    )
    assert len(pyogrio.list_layers(package)) == 3
    for layer, _ in pyogrio.list_layers(package):
        frame = gpd.read_file(package, layer=layer)
        assert frame.crs is None
        if layer == "verified_road_markings":
            assert frame.empty
        else:
            assert set(frame.review_status) == {"PROPOSED_UNREVIEWED"}


def test_file_backed_original_intensity_immutable(tmp_path):
    from roadmarking.rasterization.pipeline import sha256

    original = scene().image
    path = tmp_path / "original.npy"
    np.save(path, original)
    digest = sha256(path)
    image = np.load(path, mmap_mode="r")
    evidence = LineEvidence(
        image, np.ones(image.shape, bool), Affine(0.1, 0, -56000, 0, -0.1, -89000)
    )
    candidates, _, _ = trace_stripes(evidence)
    assert candidates and sha256(path) == digest
