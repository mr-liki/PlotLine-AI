"""Phase 3.1: road-corridor refinement trial, full runs, comparisons and report.

Stages (run in order, each reusing earlier versioned outputs):
  trial  - Track01_3 problem region: v1 and v2 on identical points/bounds (phase3_1 ns)
  full   - v2 on both intact sources (Phase 3 intensity calibration kept fixed)
  report - comparisons vs Phase 3, figures, proposed ROIs, combined manifest, report

Phase 3 outputs are read only. Nothing here trains, downloads or labels anything.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from roadmarking.config import load_config
from roadmarking.rasterization.pipeline import INTACT, dump, process_source

ROOT = Path(__file__).resolve().parents[1]
NS = "phase3_1"
TRIAL_SOURCE = "Job_20260325_0838_Track01_3"
# Phase 4 problem tiles (paint at candidate edge, fragmentation, smooth blob) in Track01_3.
TRIAL_TILES = [
    "Job_20260325_0838_Track01_3__r0017664_c0017664",
    "Job_20260325_0838_Track01_3__r0018432_c0019968",
    "Job_20260325_0838_Track01_3__r0020736_c0023040",
    "Job_20260325_0838_Track01_3__r0023808_c0026880",
]


def phase3_runs(cfg) -> dict:
    rep = json.loads((cfg.path("outputs.reports") / "phase3/phase3_report.json").read_text())
    return {r["source"]: r for r in rep["full_runs"]}


def calibration(run: dict) -> dict:
    meta = json.loads((Path(run["output_directories"]["metadata"]) / "source.json").read_text())
    return meta["normalization"]


def trial_bounds(cfg) -> list[float]:
    tiles = {
        t["tile_id"]: t
        for t in json.loads(
            (cfg.path("data.processed") / "phase3/metadata/tile_manifest.json").read_text()
        )
    }
    boxes = [tiles[t]["native_xy_bbox"] for t in TRIAL_TILES]
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--stage", choices=["diagnose", "trial", "full", "report", "all"], default="all")
    p.add_argument("--road-roi", type=Path, help="Approved native-XY road ROI (GPKG/GeoJSON)")
    p.add_argument("--roi-assert-native-frame", action="store_true")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    refinement = cfg.raw.get("phase3_1", {}).get("refinement", {})
    p3 = phase3_runs(cfg)
    reports = cfg.path("outputs.reports") / NS
    runs_file = reports / "runs.json"
    runs = json.loads(runs_file.read_text()) if runs_file.is_file() else {}
    common = dict(
        namespace=NS,
        refinement=refinement,
        roi_path=args.road_roi,
        roi_assert_native_frame=args.roi_assert_native_frame,
        gap_fill=False,
    )
    if args.stage in ("diagnose", "all"):
        from roadmarking.analysis.phase3_1_diagnostics import run_diagnostics
        from roadmarking.analysis.phase3_1_report import CASES

        t = time.perf_counter()
        diag = run_diagnostics(cfg, p3, CASES, reports / "figures")
        diag["seconds"] = time.perf_counter() - t
        dump(reports / "diagnostics.json", diag)
        print(
            json.dumps(
                {
                    "diagnose": {
                        k: v["band"]["by_first_failing_criterion"]
                        for k, v in diag.items()
                        if k != "seconds"
                    }
                }
            )
        )
    if args.stage in ("trial", "all"):
        bounds = trial_bounds(cfg)
        path = cfg.path("data.raw_pointclouds") / f"{TRIAL_SOURCE}.laz"
        cal = calibration(p3[TRIAL_SOURCE])
        trial = {"bounds": bounds}
        for method in ("v1", "v2"):
            t = time.perf_counter()
            res = process_source(
                cfg, path, bounds=bounds, road_method=method, calibration_override=cal, **common
            )
            trial[method] = dict(
                processing=str(Path(res["reports"]) / "processing.json"),
                wall_seconds=time.perf_counter() - t,
            )
            print(
                json.dumps(
                    {
                        "trial": method,
                        "selected": res["selected_points"],
                        "tiles": res["tiles"],
                        "qc": res["qc"]["passed"],
                    }
                )
            )
        runs["trial"] = trial
        dump(runs_file, runs)
    if args.stage in ("full", "all"):
        runs.setdefault("full", {})
        for source in INTACT:
            path = cfg.path("data.raw_pointclouds") / f"{source}.laz"
            res = process_source(
                cfg,
                path,
                road_method="v2",
                calibration_override=calibration(p3[source]),
                **common,
            )
            runs["full"][source] = str(Path(res["reports"]) / "processing.json")
            print(
                json.dumps(
                    {
                        "full": source,
                        "selected": res["selected_points"],
                        "tiles": res["tiles"],
                        "qc": res["qc"]["passed"],
                    }
                )
            )
            dump(runs_file, runs)
    if args.stage in ("report", "all"):
        from roadmarking.analysis.phase3_1_report import build_report

        report = build_report(cfg, ROOT, runs, p3)
        print(json.dumps({k: report[k] for k in ("status", "phase4_reuse_ready")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
