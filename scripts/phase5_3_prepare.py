"""Record region completeness and export supervised native-resolution crops."""

import json
from datetime import UTC, datetime
from pathlib import Path

from roadmarking.dataset.complete_regions import check_complete_dataset, prepare_complete_regions
from roadmarking.rasterization.pipeline import dump

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    source = json.loads((root / "outputs/reports/phase5_2/final_export.json").read_text())[
        "dataset"
    ]
    output = (
        root
        / "data/processed/phase5_training"
        / ("phase5_3_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    )
    report_dir = root / "outputs/reports/phase5_3"
    manifest = prepare_complete_regions(source, output, report_dir)
    check = check_complete_dataset(output)
    dump(report_dir / "dataset_latest.json", {"dataset": str(output)})
    dump(report_dir / "dataset_readiness.json", check)
    print(
        json.dumps(
            {
                "dataset": str(output),
                "readiness": check,
                "verified_background_pixels": manifest["verified_background_pixels"],
            },
            indent=2,
        )
    )
