"""Controlled independent corridor pilot; never implicitly processes entire surveys."""

import argparse
import json
from pathlib import Path

from roadmarking.analysis.corridor_pilot import read_json, run_pilot, write_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("diagnose", "propose", "guided", "validated", "compare"),
        default="propose",
    )
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--seed-points",
        type=Path,
        help="Native-XY Point/MultiPoint hints; proposed paths require review",
    )
    parser.add_argument("--manual-review", type=Path)
    parser.add_argument(
        "--source", help="Optional intact source stem (Track01_3 or Track01_8 full filename stem)"
    )
    parser.add_argument("--discovery-mode", choices=("intensity", "geometry"), default="intensity")
    parser.add_argument(
        "--tile-limit",
        type=int,
        help="Small development pilot only; omitted audits all 42 boundary flags",
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if args.mode == "compare":
        latest = read_json(root / "outputs/reports/phase3_1_topodot_inspired/latest.json")
        report = read_json(latest["report"])
        write_report(report, Path(latest["report"]).with_suffix(".md"))
    else:
        report = run_pilot(
            root,
            mode=args.mode,
            reference_path=args.reference,
            review_path=args.manual_review,
            discovery_mode=args.discovery_mode,
            tile_limit=args.tile_limit,
            source=args.source,
            seed_path=args.seed_points,
        )
    print(
        json.dumps(
            {
                k: report.get(k)
                for k in (
                    "status",
                    "run_id",
                    "pilot_tiles",
                    "full_processing_performed",
                    "report_directory",
                )
            }
        )
    )
    return 1 if report["status"] == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
