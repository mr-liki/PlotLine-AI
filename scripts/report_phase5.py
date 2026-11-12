"""Assemble the Phase 5 report from executed preparation and supervision audits."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from roadmarking.rasterization.pipeline import dump, sha256
from roadmarking.training.quality import inspect_training_dataset


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "outputs/reports/phase5"

    def linked(filename):
        pointer = json.loads((folder / filename).read_text())
        return json.loads(Path(pointer["report"]).read_text())

    source = linked("source_audit_latest.json")
    raster = linked("production_pilot_latest.json")
    conversion = linked("annotation_conversion_latest.json")
    output = Path(raster["output_root"])
    qc = json.loads((folder / "source_qc_attestation.json").read_text())
    model = json.loads((folder / "model_provenance.json").read_text())
    crops = json.loads((folder / "native_crop_coverage.json").read_text())
    gate = inspect_training_dataset(output, annotation_store=conversion["annotation_store"])
    dump(folder / "training_gate.json", gate)
    changed = [
        p for p, digest in source["source_annotation_hashes"].items() if sha256(Path(p)) != digest
    ]
    if changed:
        raise ValueError("Original annotations changed since source audit")
    tests = {}
    for filename in ("pytest.xml", "pytest_qc_gate.xml"):
        path = folder / filename
        if path.exists():
            suite = ET.parse(path).getroot().find("testsuite")
            attrs = suite.attrib
            tests[filename] = {k: int(attrs[k]) for k in ("tests", "failures", "errors", "skipped")}
            tests[filename]["seconds"] = float(attrs["time"])
    health_file = folder / "healthcheck.json"
    health = json.loads(health_file.read_text()) if health_file.exists() else []
    report = {
        "status": "ANNOTATION_REVIEW_REQUIRED",
        "phase6_ready": False,
        "source_annotations_qc_final": True,
        "source_qc_attestation": qc,
        "source_report": source,
        "raster_report": raster,
        "conversion_report": conversion,
        "dataset_gate": gate,
        "native_crop_coverage": crops,
        "model_provenance": model,
        "training_configuration": str(root / "configs/phase5_mask2former.yaml"),
        "model_initialized": False,
        "training_started": False,
        "inference_started": False,
        "trained_checkpoint": None,
        "recognition_metrics": None,
        "model_gpu_peak_bytes": None,
        "model_head_initialization_test": "NOT_RUN_DATASET_GATE_FAILED",
        "forward_backward_test": "NOT_RUN_NO_APPROVED_PAIR",
        "model_save_reload_test": "NOT_RUN_NO_TRAINED_MODEL",
        "training_evaluation_inference_implementation": "DOWNSTREAM_INTERFACES_NOT_IMPLEMENTED",
        "tests": tests,
        "health": {
            name: sum(r["status"] == name for r in health) for name in ("pass", "skip", "fail")
        },
        "source_annotation_hash_mismatches": len(changed),
        "visual_checks": {
            "qgis_render_and_representative_right_left_interchange_overlays_inspected": True,
            "observations": "Native placement supported; NoData and large LEFT offset tail visible",
            "human_generated_buffer_approval_invented": False,
        },
        "execution_exceptions_resolved": [
            "Initial conversion failed at offset helper signature; corrected "
            "and rerun in a new workspace",
            "Lint long lines/import order/exception naming corrected before final checks",
        ],
        "remaining_work": [
            "Focused paint-area conversion checks, complete reviewed "
            "region/background scope and unresolved visual mappings",
            "Validated loss exclusion for NoData/boundary/unknown pixels or "
            "genuinely fully supervised observed native crops",
            "Further source-point validation and all-pass mosaicking before "
            "complete-area expansion",
            "Independent approved train/validation/test geography and "
            "global-instance leakage audit",
            "Implement and execute gated model initialization, memory test, "
            "training, metrics and held-out inference",
        ],
    }
    dump(folder / "report.json", report)
    per_tile = "\n".join(
        f"| {t['tile_id']} | {t['observed_pixels']:,} | {t['nodata_pixels']:,} |"
        for t in raster["tiles"]
    )
    scans = "\n".join(
        f"| {s['group']} | {s['valid_points']:,} | {s['dropped_points']} | "
        f"{s['read_and_qc_seconds']:.2f} | {s['accumulation_seconds']:.2f} |"
        for s in raster["full_source_scans"]
    )
    counts_by_size = {
        size: sum(
            r["fully_observed_candidate_windows"] for r in crops if r["native_crop_size"] == size
        )
        for size in (1024, 512, 256, 128)
    }
    coverage = "\n".join(
        f"| {size} | "
        f"{counts_by_size[size]} | "
        f"{max(r['best_observed_fraction'] for r in crops if r['native_crop_size'] == size):.4%} |"
        for size in (1024, 512, 256, 128)
    )
    sections = {
        "A. Input sources": f"Raw: `{source['raw_root']}`\n\n"
        f"Annotations: `{source['annotation_root']}`\n\n"
        "Live inventory found 173 LAS/LAZ files, four annotation "
        "GeoPackages and two DGN files. "
        "The separator typo in the pasted annotation path was resolved to "
        "the existing user-confirmed directory. "
        "All processing is read-only on the network shares; derived files "
        "are local. See file_inventory.json "
        "under the versioned source-audit report directory for every entry,"
        " timestamp and size.",
        "B. Source integrity": f"All 173 files passed three actual-record sample windows: "
        f"{source['sampled_valid_points']:,} sampled points. This is not full-file certification. "
        "Three complete pilot files passed strict chunk decoding, bounds, returns, GPS and counts: "
        f"{raster['total_valid_points']:,} points, zero dropped. "
        "The other 170 remain full-scan pending. "
        "Partial first/last-block fingerprints are labeled as partial; "
        "annotation hashes are full SHA-256. "
        "Potential duplicate fingerprints are not proof of full-file byte identity.\n\n"
        "| Source | Full valid points | Dropped | Read + point QC seconds |"
        " Accumulation/statistics seconds |\n"
        "|---|---:|---:|---:|---:|\n" + scans,
        "C. Coordinate matching": "Independent live audit: 16,521 features; "
        "16,520 geometrically valid; "
        "all valid features intersect raw footprints. No transform was "
        "applied. Original client QC is accepted "
        "per the user confirmation, with source hashes attached. Native "
        "overlays and intensity peak proxies "
        "support placement; median absolute offsets span approximately "
        "0.018–0.039 native units. LEFT has "
        "a p90 offset near 0.475 requiring conversion QA. These proxies are"
        " not model accuracy or a complete "
        "human pixel review. Source CRS declarations are evidence, not an "
        "independently confirmed absolute datum.",
        "D. Roadway intensity generation": "Six clean 1024×1024 windows; "
        "0.05 native-unit sampling, "
        "256 overlap and 768 stride on a global lattice. Complete "
        "selected-file points are used in each ROI. "
        "One shared p1/p99.5 calibration across all three full files; gamma"
        " 0.7, grayscale replicated into RGB. "
        "Mean, maximum, count, normalized, observed, valid and "
        "fill-provenance layers are retained. No filling "
        "or resampling. All-return conservative context is retained; this "
        "is not certified road-only extraction. "
        "Other intersecting passes are listed, not merged. Whole-network "
        "raster expansion has not occurred.\n\n"
        "| Tile | Observed pixels | NoData pixels |\n|---|---:|---:|\n" + per_tile,
        "E. Annotation verification": "Source geometry/classifications/attributes are QC-final "
        "according "
        "to the user's explicit message. No blanket reannotation is "
        "requested. Original width attributes "
        "include 0.15 and 0.30 for lines. One invalid polygon and 50 "
        "duplicate-geometry excess records "
        "are explicitly reported. Existing polygons remain available; "
        "unresolved AI catalog mappings are "
        "not silently approved. 88 longitudinal paint-area proposals reuse "
        "source widths, stable IDs and "
        "the existing flat-cap buffer. New conversion outlines are "
        "separately tagged PROPOSED_UNREVIEWED; "
        "no new human checklist events were invented.",
        "F. Dataset quality": "Six attributed image/mask conversion-review pairs were exported "
        "with "
        "instance/semantic/state/category masks and native affine mapping. "
        "They are not approved primary "
        "training pairs. Foreground source identity is retained across tile"
        " boundaries, and polygon holes "
        "are preserved by the existing converter. Verified background "
        "pixels: zero in the new crops. "
        "Observed pixels outside derived paint remain unknown. NoData "
        "overrides all polygon claims. "
        "Approved training images and background-only images: zero. Missing"
        " classes are not synthesized. "
        "Class 1 solid/2 dash retain model IDs 0/1; other categories remain"
        " catalog/visual-scope pending.\n\n"
        "Every native crop position was checked for wholly observed windows:\n\n"
        "| Native crop | Fully observed windows | Best observed fraction |\n"
        "|---:|---:|---:|\n" + coverage,
        "G. Spatial splits": "No train/validation/test split is assigned without approved samples. "
        "Overlap, repeated passes and shared global IDs must stay together."
        " Existing spatial splitter "
        "is retained, with a proposed 10-native-unit guard and independent "
        "checks in the new preflight. "
        "Six crops cannot be assumed to form six independent survey "
        "regions. 70/15/15 has not been forced.",
        "H. Model initialization": f"Fixed checkpoint: `{model['repository']}` at revision "
        f"`{model.get('revision')}`. Official metadata and local cache checked: no cached weights. "
        "Safetensors exists in the official repository; license is `other`."
        " Separate rights review is needed "
        "before commercial redistribution/deployment. Weights were not "
        "downloaded or initialized because "
        "the supervised-data gate failed. No COCO head is presented as a "
        "road-marking model.\n\n"
        "[Official model "
        "repository](https://huggingface.co/facebook/mask2former-swin-small-coco-instance)",
        "I. GPU feasibility": "CUDA environment checks passed on RTX 3050, 6 GB. No model "
        "forward/backward memory test was run without an approved pair. "
        "Model VRAM feasibility is NOT "
        "MEASURED. Configuration records 1024 then native 512/256 fallback,"
        " batch 1, FP16 and accumulation; "
        "these are proposed settings, not measured successful model runs.",
        "J. Fine-tuning": "NOT RUN. Zero epochs, no optimizer steps, no loss curves and no "
        "trained weights. "
        "A reproducible bounded starting configuration is saved, but the "
        "existing downstream trainer and "
        "inference interfaces remain unimplemented. This delivery completes"
        " executed preparation through "
        "the supervision gate, not an end-to-end training engine.",
        "K. Recognition performance": "NOT EVALUATED for every class. "
        "AP/AP50/AP75, precision, recall, "
        "F1 and IoU are null because no trained model or approved held-out "
        "dataset exists. No accuracy, "
        "missed-instance or duplicate-prediction counts have been fabricated.",
        "L. Visual results": f"Open `{conversion['qgis_project']}`. "
        "Six bookmarks contain full-point "
        "images, original QC-final annotation references and 88 editable "
        "draft conversion areas. All GIS "
        "display inputs are owned copies. QGIS project was saved/reopened "
        "with valid layers and no guessed "
        "CRS. QA previews show original intensity, original vectors, "
        "derived instance masks and label states. "
        "Representative RIGHT/LEFT/INTERCHANGE overlays and the QGIS render"
        " were inspected. "
        "Ground-truth-versus-prediction/error figures are unavailable "
        "because inference was not run.",
        "M. Processing speed": f"Source audit: {source['elapsed_seconds']:.2f}s. "
        "Complete-source pilot: "
        f"{raster['elapsed_seconds']:.2f}s; sampled peak process-tree RSS "
        f"{raster['sampled_peak_rss_bytes'] / 2**20:.1f} MiB. "
        "Per-source read/QC and raster/statistics "
        "timings are in section B; per-tile export timings are in "
        "report.json. Accumulation includes "
        "histogram/class/bounds statistics. Training step time, model "
        "latency, GPU peak and throughput "
        "are NOT MEASURED and must not be confused with these CPU/network "
        "preparation times.",
        "N. Best checkpoint": "NONE. No trained checkpoint exists; no save/reload claim is made. "
        "Prior checkpoints were not modified.",
        "O. Remaining problems": "Stock Mask2Former losses do not ignore arbitrary NoData/unknown "
        "pixels via pixel_mask. Even native 128 crops have missing returns;"
        " verified region/background "
        "scope and new buffer conversion checks are also incomplete. "
        "Resolve these with genuinely fully "
        "supervised observed regions or a validated "
        "matching/point-sampling/BCE/Dice/auxiliary validity "
        "strategy. Filling missing returns or setting all unannotated "
        "pixels to background is not a fix. "
        "Further tasks: missing catalog classes, invalid/duplicate "
        "geometry, full remaining-source QC, "
        "other-pass mosaicking, independent splits and subsequent model "
        "implementation/execution. "
        "The initial conversion helper-call error was fixed and the package"
        " regenerated in a new version; "
        "failed workspace artifacts were retained. No source-QC "
        "confirmation was discarded.",
        "P. Phase 6 readiness": "NOT READY. Final Phase 5 status: **ANNOTATION_REVIEW_REQUIRED** "
        "for conversion/region supervision, while original client source QC"
        " is accepted. Exact next "
        "action: inspect the six-window QGIS conversion package, establish "
        "complete observed-region "
        "paint/background supervision and validate NoData loss exclusion "
        "before model training. "
        "No final polyline/polygon digitization or Git commit was performed.",
    }
    text = "# Phase 5 execution report\n\n" + "\n\n".join(
        f"## {title}\n\n{body}" for title, body in sections.items()
    )
    text += "\n\n## Automated checks\n\n" + json.dumps(
        {"pytest": tests, "health": report["health"]}, indent=2
    )
    text += "\n\nSee validation_results.json for final Ruff/Black and additional checks. "
    text += "No source annotation hash mismatches were found.\n"
    (folder / "report.md").write_text(text, encoding="utf-8")
    dump(
        folder / "latest.json",
        {
            "output_root": str(output),
            "report": str(folder / "report.json"),
            "qgis_project": conversion["qgis_project"],
        },
    )
    print(report["status"])


if __name__ == "__main__":
    main()
