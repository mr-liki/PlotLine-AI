"""Record Phase 5.5 user approval and export the seven additional supervised regions."""

import json
from datetime import UTC, datetime
from pathlib import Path

from roadmarking.dataset.phase55 import check_dataset, prepare
from roadmarking.rasterization.pipeline import dump

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    version = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    reports = root / "outputs/reports/phase5_5" / version
    output = root / "data/processed/phase5_training" / ("phase5_5_" + version)
    manifest = prepare(root, output, reports)
    readiness = check_dataset(output)
    dump(reports / "dataset_readiness.json", readiness)
    dump(
        root / "outputs/reports/phase5_5/latest.json",
        {"run_id": version, "report_dir": str(reports), "dataset": str(output)},
    )
    print(
        json.dumps(
            {
                "dataset": str(output),
                "statistics": manifest["statistics"],
                "allocation": manifest["region_allocation"],
            },
            indent=2,
        )
    )
