"""Phase 4 availability gate. Never generates labels from missing/unverified inputs."""

from __future__ import annotations

import hashlib
import json
import math
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psutil

REQUIRED_LAYERS = {"maximum", "mean", "normalized", "count", "observed", "filled_mask"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root: Path) -> dict:
    """Recursively locate GIS files and incomplete Shapefile deliveries, read-only.

    Discovering a filename does not establish valid geometry, a label, or alignment.
    SHP/SHX/DBF are required; PRJ is reported separately and never guessed.
    """
    files = sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else []
    by_name = {(str(p.parent), p.name.casefold()): p for p in files}
    datasets = []
    for path in files:
        ext = path.suffix.casefold()
        if ext not in {".gpkg", ".shp"}:
            continue
        record = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "format": ext[1:],
            "status": "UNINSPECTED",
            "layers": None,
            "feature_count": None,
            "geometry_types": None,
            "declared_crs": None,
            "extent": None,
            "has_z": None,
            "attribute_fields": None,
            "classification_values": None,
            "invalid_geometries": None,
            "empty_geometries": None,
            "missing_ids": None,
            "duplicate_ids": None,
            "unknown_class_codes": None,
        }
        if ext == ".shp":
            missing = [
                suffix
                for suffix in (".shp", ".shx", ".dbf")
                if (str(path.parent), (path.stem + suffix).casefold()) not in by_name
            ]
            record["missing_required_companions"] = missing
            record["prj_present"] = (str(path.parent), (path.stem + ".prj").casefold()) in by_name
            if missing:
                record["status"] = "INCOMPLETE_SHAPEFILE"
        if record["size_bytes"] == 0:
            record["status"] = "EMPTY_FILE"
        datasets.append(record)
    orphans = [
        str(p)
        for p in files
        if p.suffix.casefold() in {".dbf", ".shx", ".prj"}
        and (str(p.parent), (p.stem + ".shp").casefold()) not in by_name
    ]
    return {
        "directory": str(root),
        "exists": root.is_dir(),
        "datasets": datasets,
        "dataset_count": len(datasets),
        "orphan_shapefile_companions": orphans,
        "other_files": [
            str(p)
            for p in files
            if p.suffix.casefold() not in {".gpkg", ".shp", ".shx", ".dbf", ".prj", ".cpg"}
            and p.name != ".gitkeep"
        ],
        "placeholder_files": [str(p) for p in files if p.name == ".gitkeep"],
    }


def inspect_manifest(path: Path) -> dict:
    """Inspect native grid metadata only; this is not vector/raster alignment QA."""
    result = {"path": str(path), "exists": path.is_file(), "errors": []}
    if not path.is_file():
        result["errors"].append("Phase 3 tile manifest is missing")
        return result
    result["sha256"] = digest(path)
    try:
        tiles = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(tiles, list) or not tiles:
            raise ValueError("Expected a nonempty tile list")
        ids = set()
        sources = Counter()
        dimensions = Counter()
        statuses = Counter()
        bounds_by_source = {}
        checked_layers = 0
        for tile in tiles:
            tid = tile["tile_id"]
            if tid in ids:
                raise ValueError(f"Duplicate tile ID: {tid}")
            ids.add(tid)
            h, w, res = tile["height"], tile["width"], tile["resolution"]
            a, b, c, d, e, f = tile["affine"]
            values = [a, b, c, d, e, f, res, *tile["native_xy_bbox"]]
            if not all(math.isfinite(v) for v in values) or h <= 0 or w <= 0 or res <= 0:
                raise ValueError(f"Invalid grid values: {tid}")
            if b != 0 or d != 0 or not math.isclose(a, res) or not math.isclose(e, -res):
                raise ValueError(f"Expected native north-up square grid: {tid}")
            expected = [c, f - h * res, c + w * res, f]
            if not all(
                math.isclose(x, y, rel_tol=0, abs_tol=1e-8)
                for x, y in zip(expected, tile["native_xy_bbox"], strict=True)
            ):
                raise ValueError(f"Affine/bounds mismatch: {tid}")
            layers = tile["layers"]
            if not REQUIRED_LAYERS.issubset(layers):
                raise ValueError(f"Missing layer entries: {tid}")
            for filename in layers.values():
                if not Path(filename).is_file():
                    raise ValueError(f"Missing raster: {filename}")
                checked_layers += 1
            source = tile["source_file"]
            sources[source] += 1
            dimensions[f"{w}x{h}"] += 1
            statuses[tile["crs_status"]] += 1
            old = bounds_by_source.setdefault(source, expected.copy())
            old[:] = [
                min(old[0], expected[0]),
                min(old[1], expected[1]),
                max(old[2], expected[2]),
                max(old[3], expected[3]),
            ]
        result.update(
            tile_count=len(tiles),
            sources=dict(sources),
            dimensions=dict(dimensions),
            crs_statuses=dict(statuses),
            native_bounds_by_source=bounds_by_source,
            existing_layer_files_checked=checked_layers,
            metadata_consistent=True,
            alignment_verified=False,
            coordinate_units="UNCONFIRMED",
        )
    except (KeyError, TypeError, ValueError, OSError) as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        result["metadata_consistent"] = False
    return result


def validation_evidence(report_dir: Path) -> dict:
    evidence = {}
    xml = report_dir / "pytest.xml"
    if xml.is_file():
        suites = list(ET.parse(xml).getroot().iter("testsuite"))
        totals = {
            key: sum(int(s.attrib.get(key, 0)) for s in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
        totals["passed"] = (
            totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
        )
        evidence["pytest"] = totals
    health = report_dir / "healthcheck.json"
    if health.is_file():
        evidence["healthcheck"] = json.loads(health.read_text(encoding="utf-8"))
    return evidence


def run_preflight(cfg) -> dict:
    """Write an honest stopped-stage report, without creating dataset artifacts."""
    started = time.perf_counter()
    report_dir = cfg.path("outputs.reports") / "phase4"
    protected = [
        cfg.path("data.processed") / "phase3/metadata/tile_manifest.json",
        cfg.path("outputs.reports") / "phase3/phase3_report.json",
    ]
    before = {str(p): digest(p) for p in protected if p.is_file()}
    inputs = {
        "annotations": inventory(cfg.path("data.raw_annotations")),
        "reference_gis": inventory(cfg.path("data.reference_gis")),
    }
    count = sum(x["dataset_count"] for x in inputs.values())
    state = "ANNOTATIONS_NOT_AVAILABLE" if count == 0 else "ANNOTATION_FILES_REQUIRE_INSPECTION"
    grid = inspect_manifest(protected[0])
    blockers = (
        ["No GPKG or SHP annotations were found in either configured input directory"]
        if count == 0
        else ["Located files still require layer/geometry/attribute and alignment inspection"]
    )
    blockers.extend(grid["errors"])
    memory = psutil.Process().memory_info()
    identity = {
        "inputs": inputs,
        "phase3_manifest_sha256": grid.get("sha256"),
        "preflight_code_sha256": digest(Path(__file__)),
    }
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    result = {
        "date": datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat(),
        "stage": "Phase 4 step 1: annotation data availability",
        "status": state,
        "run_id": run_id,
        "inputs": inputs,
        "annotation_file_count": count,
        "phase3": grid,
        "alignment": {
            "status": "NOT_ASSESSED",
            "overlap": None,
            "displacement_native_units": None,
            "displacement_cm": None,
            "transformation_applied": False,
        },
        "annotation_quality": {
            "status": "NOT_ASSESSED",
            "verified_features": 0,
            "rejected_features": None,
            "uncertain_features": None,
            "class_distribution": None,
        },
        "dataset": {
            "generated": False,
            "images": 0,
            "instances": 0,
            "polygon_masks": 0,
            "polyline_masks": 0,
            "ignore_policy": "No labels generated; unknown is not background",
        },
        "splits": {
            "generated": False,
            "training_images": 0,
            "validation_images": 0,
            "test_images": 0,
            "leakage_check": "NOT_APPLICABLE_NO_DATASET",
        },
        "qa": {
            "overlays": "NOT_GENERATED_NO_VERIFIED_ANNOTATIONS",
            "mask_integrity": "NOT_APPLICABLE_NO_MASKS",
        },
        "validation": validation_evidence(report_dir),
        "blockers": blockers,
        "phase5_ready": False,
        "phase4_complete": False,
        "stopped_before": (
            "Annotation alignment, road review, taxonomy approval " "and mask generation"
        ),
        "working_set_bytes": memory.rss,
        "peak_working_set_bytes": getattr(memory, "peak_wset", None),
        "memory_measurement": ("OS process working set and peak working set, " "including imports"),
        "preflight_seconds": time.perf_counter() - started,
        "phase3_input_hashes": before,
        "next_inputs_required": [
            "Manual GPKG or complete SHP/SHX/DBF annotation files",
            "Any declared CRS/coordinate units and native-frame evidence",
            "Reliable marking widths for longitudinal centerlines",
            "Feature identifiers, class fields, and completeness/review evidence",
        ],
    }
    if any(digest(Path(p)) != sha for p, sha in before.items()):
        raise RuntimeError("Phase 3 metadata changed during the read-only preflight")
    result["phase3_metadata_unchanged"] = True
    destination = cfg.path("data.processed") / "phase4/metadata" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, allow_nan=False)
    (destination / "availability.json").write_text(payload, encoding="utf-8")
    (report_dir / "phase4_report.json").write_text(payload, encoding="utf-8")
    validation_summary = []
    test_results = result["validation"].get("pytest")
    if test_results:
        validation_summary.append(
            f"Pytest: {test_results['passed']} passed, {test_results['failures']} failed, "
            f"{test_results['errors']} errors, {test_results['skipped']} skipped."
        )
    if "healthcheck" in result["validation"]:
        health_counts = Counter(r["status"] for r in result["validation"]["healthcheck"])
        validation_summary.append(
            f"Health checks: {health_counts['pass']} passed, {health_counts['fail']} failed, "
            f"{health_counts['skip']} skipped."
        )
    lines = [
        "# Phase 4 data-availability report",
        "",
        f"Status: **{state}**.",
        "",
        "Phase 4 is not complete and the dataset is not ready for Phase 5.",
        "",
        "## Input summary",
        "",
        "| Location | Directory exists | GPKG/SHP files | Orphan companions |",
        "|---|---|---:|---:|",
    ]
    for name, record in inputs.items():
        lines.append(
            f'| {name} | {record["exists"]} | {record["dataset_count"]} | '
            f'{len(record["orphan_shapefile_companions"])} |'
        )
    lines += ["", "Searched recursively at these exact configured locations:", ""]
    lines += [f'- `{r["directory"]}`' for r in inputs.values()]
    lines += [
        "",
        "Placeholder `.gitkeep` files are not annotation data. "
        "Layers, feature counts, geometry types, Z availability, declared CRS, "
        "attributes, identifiers, classification values and geometry quality "
        "have not been assessed because no usable annotations were inspected.",
        "",
        "## Raster context",
        "",
        f'Phase 3 manifest tiles: {grid.get("tile_count", "not established")}. '
        f'Existing raster layer files checked: {grid.get("existing_layer_files_checked", 0)}. '
        "Affine/bounding-box checks inspect metadata, not vector alignment. "
        "Native coordinate units and CRS remain unconfirmed. "
        "The exact manifest hash and per-source native bounds are in the JSON report. "
        "Earlier Phase 3 metadata hashes are unchanged; "
        "raster files were only checked for existence.",
        "",
        "## Alignment and annotation quality",
        "",
        "No overlay or independent displacement measurement is possible without "
        "manual vectors. No shift, rotation, scale change, reprojection or EPSG "
        "assignment was applied. Road-candidate verification and class mapping "
        "approval remain pending. Existing taxonomy hints remain provisional.",
        "",
        "## Dataset and splits",
        "",
        "Generated images: 0; instances: 0; polygon masks: 0; polyline masks: 0. "
        "Training/validation/test images: 0/0/0. No training manifest, background "
        "labels or split manifest was created. Unknown pixels were not converted "
        "to negative labels. Mask integrity, spatial leakage and visual mask QA "
        "are not applicable until verified inputs exist.",
        "",
        "## Validation",
        "",
        " ".join(validation_summary) or "Regression checks have not been recorded yet.",
        "",
        "Pytest and health-check results are recorded above when executed. "
        "Synthetic preflight tests do not verify real annotation alignment.",
        "",
        "## Performance",
        "",
        f'Preflight inspection: {result["preflight_seconds"]:.4f} seconds '
        "(excludes writing this report). No dataset bytes were generated. "
        f"Working set: {memory.rss:,} bytes. "
        f'OS peak working set: {result["peak_working_set_bytes"]} bytes. '
        "This includes imported libraries. No LAZ was reread and no GPU work "
        "was performed. With no annotation data, there is no measured mask "
        "rasterization or dataset throughput bottleneck.",
        "",
        "## Blocking requirements",
        "",
    ]
    lines += [f"- {item}" for item in blockers]
    lines += ["", "Provide the following before resuming annotation inspection:", ""]
    lines += [f"- {item}" for item in result["next_inputs_required"]]
    lines += [
        "",
        "The request explicitly requires stopping before mask generation "
        "when annotations are unavailable. The current implementation is the "
        "availability gate only; converters, datasets and spatial splits have "
        "not been implemented or validated in this blocked run. No training, "
        "weights download, model change, dependency change or Git commit occurred.",
        "",
    ]
    (report_dir / "phase4_report.md").write_text("\n".join(lines), encoding="utf-8")
    return result
