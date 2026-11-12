"""Attach real validation evidence to the Phase 3.3 measured report."""

import json
from pathlib import Path

from PIL import Image, ImageDraw

from roadmarking.analysis.model_input_optimization import write_report
from roadmarking.rasterization.pipeline import dump, sha256
from roadmarking.training.preflight import validation_evidence


def main():
    root = Path(__file__).resolve().parents[1]
    directory = root / "outputs/reports/phase3_3"
    report = json.loads((directory / "report.json").read_text())
    evidence = validation_evidence(directory)
    report["validation"].update(evidence)
    for tool in ("ruff", "black"):
        report["validation"][tool] = (directory / f"{tool}.txt").read_text().strip()
    report["validation"].pop("health", None)
    health = evidence["healthcheck"]
    if any(check["status"] not in {"pass", "skip"} for check in health):
        raise ValueError("Health check did not pass")
    report["validation"]["health_summary"] = {
        status: sum(check["status"] == status for check in health) for status in ("pass", "skip")
    }
    if not evidence["pytest"]["failures"] and not evidence["pytest"]["errors"]:
        report["validation"]["test_gate"] = "PASSED"
    else:
        raise ValueError("Pytest did not pass")
    if not all(sha256(Path(path)) == digest for path, digest in report["protected_hashes"].items()):
        raise ValueError("Protected input changed during validation")
    output = Path(report["output_root"])
    overview = Image.new("RGB", (1536, 568), "black")
    ImageDraw.Draw(overview).text(
        (8, 8), "Diagnostic coverage overview; resized for review only", fill="white"
    )
    for index, filename in enumerate(report["figures"][:9]):
        with Image.open(filename) as image:
            overview.paste(image.resize((512, 180)), ((index % 3) * 512, 28 + (index // 3) * 180))
    overview_path = output / "quality_control/tile_coverage_overview.png"
    overview.save(overview_path)
    rejection = overview.copy()
    count = report["comparison"]["512"]["skipped_tiles"]
    ImageDraw.Draw(rejection).rectangle((0, 0, 1536, 27), fill="black")
    ImageDraw.Draw(rejection).text(
        (8, 8),
        f"Rejected tile review: {count} skipped; orange windows retained for review",
        fill="white",
    )
    rejection_path = output / "quality_control/rejected_tile_review_map.png"
    rejection.save(rejection_path)
    report["coverage_overview"] = str(overview_path)
    report["rejected_tile_map"] = str(rejection_path)
    dump(directory / "report.json", report)
    write_report(report, directory / "report.md")
    artifacts = {
        str(p): sha256(p) for p in output.rglob("*") if p.is_file() and p.name != "completion.json"
    }
    dump(
        output / "metadata/completion.json",
        {
            "status": "COMPLETE_PROVISIONAL",
            "report": str(directory / "report.json"),
            "artifacts": artifacts,
        },
    )
    print(report["validation"])


if __name__ == "__main__":
    main()
