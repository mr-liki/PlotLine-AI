"""Check dataset image/mask hashes and exact native-grid alignment."""

import argparse
import json
from pathlib import Path

from roadmarking.dataset.verified_export import check_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    print(json.dumps(check_dataset(parser.parse_args().dataset), indent=2))
