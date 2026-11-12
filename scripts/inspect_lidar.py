"""Phase 2: read-only inspection of the real LAS/LAZ files in data/raw/pointclouds.

Stages (run in order; results are cached under outputs/reports/lidar_cache/):

    python scripts/inspect_lidar.py stats [--force]  # stream every file once: stats + 1 m grid
    python scripts/inspect_lidar.py report           # density/intensity/road summaries, previews

Input files are opened read-only and never modified.
"""

from __future__ import annotations

import argparse
import sys

from roadmarking.analysis.pipeline import compute_stats_cache, write_reports
from roadmarking.config import load_config
from roadmarking.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["stats", "report"])
    parser.add_argument("--grid-cell", type=float, default=1.0)
    parser.add_argument("--force", action="store_true", help="recompute cached results")
    args = parser.parse_args()
    setup_logging("INFO")
    cfg = load_config()
    files = sorted(cfg.path("data.raw_pointclouds").glob("*.la[sz]"))
    if not files:
        print("No LAS/LAZ files found in data/raw/pointclouds — nothing to inspect.")
        return 1
    if args.stage == "stats":
        for rec in compute_stats_cache(cfg, files, args.grid_cell, args.force):
            status = rec.get("error") or rec.get("read_error") or "complete"
            print(f"{rec['file']}: {rec.get('streamed_point_count', 0):,} valid points — {status}")
    else:
        js, md = write_reports(cfg, files)
        print(f"wrote {js}\nwrote {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
