"""Run Phase 4: annotation inspection, alignment gate, road review and (gated) labels.

Exit codes: 0 = dataset generated; 2 = stopped at a documented gate (no labels from
unverified inputs); 1 = error. Use --preflight-only for the original availability gate.
"""

import argparse
import json
from pathlib import Path

from roadmarking.config import load_config
from roadmarking.dataset.pipeline import run_phase4
from roadmarking.dataset.report import write_reports
from roadmarking.training.preflight import run_preflight

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--review-csv", type=Path, default=None, help="Human review records: global_id,status"
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.preflight_only:
        report = run_preflight(cfg)
        print(json.dumps({k: report[k] for k in ("status", "annotation_file_count")}))
        return 2
    preflight = run_preflight(cfg)
    if preflight["annotation_file_count"] == 0:
        print(json.dumps({"status": preflight["status"]}))
        return 2
    report = run_phase4(cfg, ROOT, args.review_csv)
    write_reports(report, cfg.path("outputs.reports") / "phase4")
    print(
        json.dumps({k: report[k] for k in ("status", "run_id", "phase4_complete", "phase5_ready")})
    )
    return 0 if report["phase4_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
