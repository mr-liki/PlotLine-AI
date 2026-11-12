"""Finalize measured annotation-workspace validation; never approves labels."""

import json
from pathlib import Path

import numpy as np
import rasterio

from roadmarking.dataset.annotation_validation import validate_store
from roadmarking.dataset.annotation_workspace import write_report
from roadmarking.rasterization.pipeline import dump, sha256
from roadmarking.training.preflight import validation_evidence


def main():
    root = Path(__file__).resolve().parents[1]
    directory = root / "outputs/reports/phase4_verified"
    report = json.loads((directory / "report.json").read_text())
    evidence = validation_evidence(directory)
    if evidence["pytest"]["failures"] or evidence["pytest"]["errors"]:
        raise ValueError("Tests failed")
    if any(c["status"] not in {"pass", "skip"} for c in evidence["healthcheck"]):
        raise ValueError("Health checks failed")
    report["validation"] = {
        **evidence,
        "health_summary": {
            status: sum(c["status"] == status for c in evidence["healthcheck"])
            for status in ("pass", "skip")
        },
    }
    for name in ("ruff", "black"):
        report["validation"][name] = (directory / f"{name}.txt").read_text(encoding="utf-8").strip()
    report["annotations"] = validate_store(report["annotation_store"])
    report["protected_unchanged"] = all(
        sha256(Path(p)) == h for p, h in report["protected_hashes"].items()
    )
    if not report["protected_unchanged"]:
        raise ValueError("Protected prior output changed")
    output = Path(report["output_root"])
    qgis = json.loads((output / "metadata/qgis_validation.json").read_text())
    report["qgis_validation"] = qgis
    spec = json.loads((output / "annotation_projects/qgis_spec.json").read_text())
    from roadmarking.dataset.annotation_store import metadata

    canonical = {t["tile_id"]: t for t in metadata(report["annotation_store"])["tiles"]}
    for tile in spec["tiles"]:
        original = canonical[tile["tile_id"]]
        with (
            rasterio.open(tile["layers"]["normalized"]) as local,
            rasterio.open(original["layers"]["normalized"]) as source,
        ):
            if (
                not np.array_equal(local.read(1), source.read(1))
                or local.transform != source.transform
            ):
                raise ValueError("QGIS precision copy differs from original pixels/grid")
        if sha256(Path(tile["images"]["grayscale"])) != sha256(
            Path(original["images"]["grayscale"])
        ):
            raise ValueError("QGIS display pixels differ from original")
    report["qgis_owned_copy_pixel_checks"] = "ALL_NINE_EXACT_GRAYSCALE_AND_NORMALIZED_NATIVE_CROPS"
    audits = []
    for latest in ("phase3_3", "phase3_intensity_images"):
        old_output = Path(
            json.loads((root / f"outputs/reports/{latest}/latest.json").read_text())["output_root"]
        )
        old_hashes = json.loads((old_output / "metadata/completion.json").read_text())["artifacts"]
        extra = [
            str(p)
            for p in old_output.rglob("*")
            if p.is_file() and str(p) not in old_hashes and p.name != "completion.json"
        ]
        mismatches = [p for p, h in old_hashes.items() if sha256(Path(p)) != h]
        if extra or mismatches:
            raise ValueError("Previous product directory changed")
        audits.append({"namespace": latest, "artifact_hash_mismatches": 0, "unexpected_files": 0})
    report["prior_product_directory_audits"] = audits
    report["visual_review"] = {
        "rendered_project_inspected": True,
        "real_annotations_verified_by_agent": False,
        "preview": str(output / "quality_control/qgis_workspace_preview.png"),
    }
    report["implementation_incident"] = {
        "qgis_statistics_cache": (
            "Detected during initial project validation; nine VRT references restored "
            "byte-for-byte to recorded hashes, 30 newly created PAM sidecars relocated "
            "to recovery evidence. Final QGIS project uses owned display artifacts and "
            "nine small native full-precision crop copies to prevent delegated source caching. "
            "Prior source pixels were not altered."
        ),
        "remaining_prior_artifact_hash_mismatches": 0,
    }
    dump(directory / "report.json", report)
    write_report(report, directory / "report.md")
    artifacts = {
        str(p): sha256(p) for p in output.rglob("*") if p.is_file() and p.name != "completion.json"
    }
    dump(
        output / "metadata/completion.json",
        {
            "status": "ANNOTATION REQUIRED",
            "report": str(directory / "report.json"),
            "artifacts": artifacts,
            "mutable_annotation_store": report["annotation_store"],
            "hash_scope": "Delivery snapshot; human edits are expected and audited after delivery",
        },
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "tests": evidence["pytest"],
                "health": report["validation"]["health_summary"],
                "qgis_layers": len(qgis["layers"]),
                "protected_unchanged": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
