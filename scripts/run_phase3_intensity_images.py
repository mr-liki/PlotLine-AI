"""Generate clean grayscale images from the approved real-data pilot windows."""

import argparse
import json

from roadmarking.config import load_config
from roadmarking.visualization.intensity_pipeline import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--tile-id", action="append")
    parser.add_argument("--bounds", nargs=4, type=float)
    parser.add_argument("--tile-limit", type=int)
    parser.add_argument("--gamma", type=float)
    args = parser.parse_args()
    cfg = load_config(args.config)
    report = run(
        cfg.root,
        sources=args.sources,
        tile_ids=args.tile_id,
        bounds=args.bounds,
        limit=args.tile_limit,
        gamma=args.gamma,
    )
    print(json.dumps({k: report[k] for k in ("status", "run_id", "output_root", "counts")}))


if __name__ == "__main__":
    main()
