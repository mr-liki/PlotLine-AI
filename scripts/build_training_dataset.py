"""Export only attributed verified annotations; partial masks remain review-only."""

import argparse
from pathlib import Path

from roadmarking.dataset.verified_export import export_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export_dataset(args.annotations, args.output)
    print(result["status"])
    print("Real image-mask pairs:", result["real_image_mask_pairs"])
