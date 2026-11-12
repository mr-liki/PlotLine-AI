"""End-to-end test of the Phase-2 inspection pipeline on a small synthetic survey.

The synthetic data only exercises the code path (caching, report, previews); it says
nothing about the client data.
"""

from __future__ import annotations

import json
from pathlib import Path

import laspy
import numpy as np
import pytest

from roadmarking.analysis.pipeline import compute_stats_cache, write_reports
from roadmarking.config import load_config


def _synthetic_road(path: Path) -> None:
    """80 m x 80 m flat 'road' at 400 pts/m² with a bright 15 cm stripe and a 'tree'."""
    rng = np.random.default_rng(1)
    n = 80 * 80 * 400
    x = rng.uniform(0, 80, n)
    y = rng.uniform(0, 80, n)
    z = 10.0 + rng.normal(0, 0.005, n)
    inten = rng.normal(6000, 500, n).clip(1, 65535)
    stripe = np.abs(x - 40.0) < 0.075
    inten[stripe] = 40000
    nr = np.ones(n, np.uint8)
    rn = np.ones(n, np.uint8)
    tree = (x > 5) & (x < 15) & (y > 5) & (y < 15)
    z[tree] += rng.uniform(0, 6, tree.sum())
    nr[tree], rn[tree] = 2, rng.integers(1, 3, tree.sum())
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([0.0, 0.0, 0.0])
    las = laspy.LasData(header)
    las.x, las.y, las.z = x - 50_000, y - 90_000, z
    las.intensity = inten.astype(np.uint16)
    las.return_number, las.number_of_returns = rn, nr
    las.gps_time = np.sort(rng.uniform(1000, 1100, n))
    las.write(path)


@pytest.fixture
def survey(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ROADMARKING_DATA_ROOT", str(tmp_path))
    cfg = load_config()
    raw = cfg.path("data.raw_pointclouds")
    raw.mkdir(parents=True)
    _synthetic_road(raw / "synthetic.laz")
    (raw / "empty.laz").write_bytes(b"")
    return cfg, sorted(raw.glob("*.laz"))


def test_pipeline_end_to_end(survey) -> None:
    cfg, files = survey
    before = {f: f.read_bytes() for f in files}
    records = compute_stats_cache(cfg, files)
    by = {r["file"]: r for r in records}
    assert "error" in by["empty.laz"]
    assert by["synthetic.laz"]["complete"] is True
    assert by["synthetic.laz"]["class_counts"] == {0: 80 * 80 * 400}

    js, md = write_reports(cfg, files)
    result = json.loads(js.read_text(encoding="utf-8"))
    f = next(x for x in result["files"] if x["file"] == "synthetic.laz")
    road = f["density"]["road_candidate_cells"]["pts_per_m2"]["p50"]
    assert 380 <= road <= 420  # measured on road cells, not the bbox
    assert f["road_surface"]["criteria_pass_pct_of_occupied_cells"]["candidate"] < 100
    assert f["previews"], "expected at least one preview window"
    for p in f["previews"]:
        g = p["gsd"]["0.05"]
        # 400 pts/m² = 1 pt per 5 cm pixel on average: random sampling leaves e^-1 = 36.8 %
        # empty. Pixels outside the survey must not be counted as road no-data.
        assert 33.0 < g["road_nodata_pct"] < 41.0
        assert abs(g["road_nodata_pct"] - g["road_nodata_pct_poisson_expected"]) < 3.0
        g10 = p["gsd"]["0.1"]  # 4 pts per 10 cm pixel: e^-4 = 1.8 %
        assert g10["road_nodata_pct"] < 4.0
    samples = [s for s in f["density_samples"] if "nn_spacing_road" in s]
    assert samples and 0.01 < samples[0]["nn_spacing_road"]["nn_m"]["p50"] < 0.05
    t = result["survey_totals"]
    assert t["files_usable"] == 1 and t["files_total"] == 2
    assert t["valid_points_total"] == 80 * 80 * 400
    assert f["acquisition"]["gps_duration_s"] == pytest.approx(100, abs=0.5)
    assert t["tiles_touching_corridor"]["0.05m_1024px_overlap128px"] >= 1
    assert "Acquisition geometry" in md.read_text(encoding="utf-8")
    prev = cfg.path("outputs.reports") / "lidar_previews"
    assert (prev / "intensity_histograms.png").exists()
    assert list(prev.glob("*_gsd050mm_imax.tif"))
    text = md.read_text(encoding="utf-8")
    assert "synthetic.laz" in text and "NOT usable" in text

    # Inputs are untouched.
    assert all(f.read_bytes() == b for f, b in before.items())

    import rasterio

    tif = next(prev.glob("*_gsd050mm_imax.tif"))
    with rasterio.open(tif) as src:
        assert src.transform.a == pytest.approx(0.05) and src.crs is None
