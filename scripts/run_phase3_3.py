"""Run Phase 3.3 using existing validated raster products only."""

import argparse
from pathlib import Path

from roadmarking.analysis.model_input_optimization import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--margin-native", type=float, default=4.0)
    args = parser.parse_args()
    result = run(Path(__file__).resolve().parents[1], margin_native=args.margin_native)
    print(result["output_root"])
    print(result["comparison"])
