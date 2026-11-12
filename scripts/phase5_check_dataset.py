"""Fail-closed supervised-data preflight. Does not load model weights."""

import argparse
import json
from pathlib import Path

from roadmarking.rasterization.pipeline import dump
from roadmarking.training.quality import inspect_training_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--guard", type=float, default=10)
    parser.add_argument("--annotation-store", type=Path)
    args = parser.parse_args()
    report = inspect_training_dataset(
        args.dataset, annotation_store=args.annotation_store, guard=args.guard
    )
    if args.report:
        dump(args.report, report)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["training_ready"] else 2)
