"""Validate approved-positive export integrity separately from training eligibility."""

import argparse
import json
from pathlib import Path

from roadmarking.dataset.approved_export import validate_approved_export
from roadmarking.rasterization.pipeline import dump

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = validate_approved_export(args.dataset)
    if args.report:
        dump(args.report, result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["training_ready"] else 2)
