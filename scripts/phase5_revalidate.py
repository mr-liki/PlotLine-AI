"""Recheck live sources and supervision without replacing prior Phase 5 outputs."""

import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from roadmarking.io.source_inventory import discover, fingerprint
from roadmarking.rasterization.pipeline import dump, sha256
from roadmarking.training.quality import inspect_training_dataset


def main():
    root = Path(__file__).resolve().parents[1]
    previous = root / "outputs/reports/phase5"
    out = previous / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "_revalidation")
    out.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((root / "configs/phase4_network_sources.yaml").read_text())
    inventories = {}
    literal = (
        r"\\Nast\naksha\DAILY ACTIVITY REPORT!!!!!geetha\2026\NT895\2026"
        r"\Highway inventory project - A4\Road_marking"
    )
    for name, path in {
        "annotations_literal_request": literal,
        "annotations_existing_config": config["annotation_root"],
        "lidar": config["candidate_raw_roots"][0],
    }.items():
        try:
            inventories[name] = {"root": path, "entries": discover(path), "accessible": True}
        except OSError as exc:
            inventories[name] = {"root": path, "accessible": False, "error": str(exc)}
    dump(out / "file_inventory.json", inventories)
    old = json.loads(
        Path(json.loads((previous / "source_audit_latest.json").read_text())["report"]).read_text()
    )
    comparisons = []
    for path, expected in old["source_annotation_hashes"].items():
        try:
            actual = sha256(Path(path))
            comparisons.append({"path": path, "unchanged": actual == expected, "sha256": actual})
        except OSError as exc:
            comparisons.append({"path": path, "unchanged": False, "error": str(exc)})
    conversion = json.loads(
        Path(
            json.loads((previous / "annotation_conversion_latest.json").read_text())["report"]
        ).read_text()
    )
    pilot = json.loads(
        Path(
            json.loads((previous / "production_pilot_latest.json").read_text())["report"]
        ).read_text()
    )
    # Recheck selected raw sources against their recorded partial fingerprints.
    raw_checks = []
    old_raw = json.loads((Path(old["report_directory"]) / "raw_integrity.json").read_text())
    for record in old_raw:
        try:
            current = fingerprint(record["path"])
            raw_checks.append(
                {
                    "path": record["path"],
                    "unchanged": current == record.get("fingerprint"),
                    "fingerprint": current,
                }
            )
        except (OSError, ValueError) as exc:
            raw_checks.append({"path": record["path"], "unchanged": False, "error": str(exc)})
    dump(out / "raw_fingerprint_comparison.json", raw_checks)
    gate = inspect_training_dataset(
        Path(pilot["output_root"]), annotation_store=conversion["annotation_store"]
    )
    dump(out / "training_gate.json", gate)
    sources_ok = all(x["unchanged"] for x in comparisons + raw_checks)
    status = "ANNOTATION_REVIEW_REQUIRED" if sources_ok else "BLOCKED"
    report = {
        "status": status,
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "annotation_hash_comparison": comparisons,
        "raw_files_fingerprinted": len(raw_checks),
        "raw_fingerprints_unchanged": sum(x["unchanged"] for x in raw_checks),
        "fingerprint_limit": "Partial fingerprints do not certify complete point integrity",
        "training_gate": gate,
        "prior_report": str(previous / "report.json"),
        "qgis_project": conversion["qgis_project"],
        "trained_checkpoint": None,
        "next_action": conversion["next_action"],
    }
    dump(out / "report.json", report)
    (out / "report.md").write_text(
        f"# Phase 5 live revalidation\n\nStatus: **{status}**\n\n"
        f"Annotation hashes unchanged: {sum(x['unchanged'] for x in comparisons)}/"
        f"{len(comparisons)}. Raw partial fingerprints unchanged: "
        f"{report['raw_fingerprints_unchanged']}/{len(raw_checks)}.\n\n"
        f"Supervised-data gate:\n\n```json\n{json.dumps(gate, indent=2)}\n```\n\n"
        f"Next action: {conversion['next_action']}.\n\n"
        f"QGIS review project: `{conversion['qgis_project']}`.\n\n"
        "No training or evaluation was performed. Previous phase outputs were preserved. "
        "See the prior Phase 5 report for preparation measurements; they were not rerun here.\n",
        encoding="utf-8",
    )
    print(out, flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
