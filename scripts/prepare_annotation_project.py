"""Prepare the native-XY QGIS annotation workspace from Phase 3.3."""

from pathlib import Path

from roadmarking.dataset.annotation_workspace import prepare

if __name__ == "__main__":
    report = prepare(Path(__file__).resolve().parents[1])
    print(report["qgis_project"])
    print(report["status"])
