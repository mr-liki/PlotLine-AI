"""Inspect a native annotation workspace without approving any feature."""

import argparse
import json

from roadmarking.dataset.annotation_validation import validate_store

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations")
    args = parser.parse_args()
    result = validate_store(args.annotations)
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if result["invalid_features"] or result["global_errors"] else 0)
