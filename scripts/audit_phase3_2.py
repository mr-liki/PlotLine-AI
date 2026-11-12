"""Audit persisted Phase 3.2 coordinates, exports, metadata and validation evidence."""

# ruff: noqa: E501 -- complete audit-report sentences.

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import rasterio
from osgeo import ogr
from shapely.geometry import LineString, box
from shapely.ops import unary_union

from roadmarking.analysis.corridor_pilot import read_json
from roadmarking.analysis.phase3_2_lines import write_report
from roadmarking.rasterization.pipeline import dump, sha256


def audit(root):
    began = time.perf_counter()
    directory = root / "outputs/reports/phase3_2"
    report = read_json(directory / "report.json")
    out = Path(report["output_root"])
    if not out.resolve().is_relative_to((root / "data/processed/phase3_2").resolve()):
        raise ValueError("Audit target outside Phase 3.2 output scope")
    manifest = read_json(out / "metadata/manifest.json")
    previous = read_json(
        read_json(root / "outputs/reports/phase3_1_topodot_inspired/latest.json")["report"]
    )
    old = {
        t["tile_id"]: t
        for t in read_json(Path(previous["output_root"]) / "metadata/tile_manifest.json")
    }
    exports, overlaps = [], []
    for entry in manifest:
        group = entry["group"]
        package = Path(entry["vector_package"])
        candidates = read_json(out / "metadata" / f"{group}_candidates.json")
        # Synchronize derived cross-tile membership metadata. Geometry and FIDs
        # are untouched; membership is calculated after the initial export.
        dataset = ogr.Open(str(package), 1)
        memberships = {c["id"]: c["quality"]["cross_tile_joins"] for c in candidates}
        for table in ("candidate_painted_stripes", "candidate_reference_alignments"):
            before_geometry = (
                gpd.read_file(package, layer=table, fid_as_index=True).geometry.to_wkb().to_dict()
            )
            layer = dataset.GetLayerByName(table)
            for feature in layer:
                identifier = feature.GetField("id")
                if identifier in memberships:
                    feature.SetField("cross_tile_joins", memberships[identifier])
                    if layer.SetFeature(feature) != 0:
                        raise ValueError("Could not synchronize derived membership metadata")
            layer.SyncToDisk()
            layer = None
            after_geometry = (
                gpd.read_file(package, layer=table, fid_as_index=True).geometry.to_wkb().to_dict()
            )
            if before_geometry != after_geometry:
                raise ValueError("Geometry/FID changed during metadata synchronization")
        dataset = None
        scope = unary_union(
            [
                box(
                    old[tid]["affine"][2],
                    old[tid]["affine"][5] - 51.2,
                    old[tid]["affine"][2] + 51.2,
                    old[tid]["affine"][5],
                )
                for tid in entry["tile_ids"]
            ]
        )
        for layer, _ in pyogrio.list_layers(package):
            info = pyogrio.read_info(package, layer=layer)
            if info["crs"] is not None:
                raise ValueError("Unexpected assigned CRS")
            if layer == "verified_road_markings" and info["features"]:
                raise ValueError("No human-reviewed final lines are authorized")
            exports.append(
                {"group": group, "layer": layer, "features": info["features"], "crs": None}
            )
        frame = gpd.read_file(package, layer="candidate_painted_stripes")
        if len(frame) and (frame.geometry.difference(scope).length > 1e-7).any():
            raise ValueError("Stripe geometry extends beyond explicit display scope")
        if len(frame) and set(frame.review_status) != {"PROPOSED_UNREVIEWED"}:
            raise ValueError("Unexpected automatic approval")
        fragments = {
            tid: read_json(out / "metadata" / f"{tid}_stripe_fragments.json")
            for tid in entry["tile_ids"]
        }
        for i, a in enumerate(entry["tile_ids"]):
            ta = old[a]["affine"]
            for b in entry["tile_ids"][i + 1 :]:
                tb = old[b]["affine"]
                shared = box(ta[2], ta[5] - 51.2, ta[2] + 51.2, ta[5]).intersection(
                    box(tb[2], tb[5] - 51.2, tb[2] + 51.2, tb[5])
                )
                checked, failures = 0, []
                for candidate in candidates:
                    aa = unary_union(
                        [
                            LineString(c["coords"])
                            for c in fragments[a]
                            if c["parent_id"] == candidate["id"]
                        ]
                    ).intersection(shared)
                    bb = unary_union(
                        [
                            LineString(c["coords"])
                            for c in fragments[b]
                            if c["parent_id"] == candidate["id"]
                        ]
                    ).intersection(shared)
                    if aa.length < 1e-7 and bb.length < 1e-7:
                        continue
                    checked += 1
                    if aa.is_empty or bb.is_empty or aa.hausdorff_distance(bb) > 1e-7:
                        failures.append(candidate["id"])
                overlaps.append(
                    {
                        "group": group,
                        "tile_a": a,
                        "tile_b": b,
                        "shared_parent_geometries_checked": checked,
                        "failures": failures,
                    }
                )
        for path in entry["raster_layers"].values():
            with rasterio.open(path) as ds:
                before = ds.read(1)
                transform = ds.transform
                if ds.crs is not None:
                    raise ValueError("Unexpected raster CRS")
            # The shared writer's old producer tag is replaced for these NEW
            # Phase 3.2 files; no old phase raster or source pixels are changed.
            with rasterio.open(path, "r+") as ds:
                ds.update_tags(
                    CONFIG_VERSION="phase3.2-evidence-stripe-tracking", RUN_ID=report["run_id"]
                )
            with rasterio.open(path) as ds:
                if ds.transform != transform or not np.array_equal(ds.read(1), before):
                    raise ValueError("Pixel/affine changed while updating producer metadata")
    root_xml = ET.parse(directory / "pytest.xml").getroot()
    suites = list(root_xml.iter("testsuite"))
    tests = {
        k: sum(int(s.attrib.get(k, 0)) for s in suites)
        for k in ("tests", "failures", "errors", "skipped")
    }
    tests["passed"] = tests["tests"] - tests["failures"] - tests["errors"] - tests["skipped"]
    health = read_json(directory / "healthcheck.json")
    validation = {
        "pytest": tests,
        "healthcheck": {
            status: sum(item["status"] == status for item in health)
            for status in ("pass", "skip", "fail")
        },
        "ruff": (directory / "ruff.log").read_text().strip(),
        "black": (directory / "black.log").read_text().strip(),
    }
    dump(directory / "validation.json", validation)
    changed = [p for p, digest in report["protected_hashes"].items() if sha256(p) != digest]
    audit_result = {
        "export_layers": exports,
        "overlap_geometry_checks": overlaps,
        "protected_mismatches": changed,
        "raster_metadata_update_preserved_pixels": True,
        "reviewed_final_lines": 0,
        "audit_seconds": time.perf_counter() - began,
    }
    if changed or any(item["failures"] for item in overlaps):
        report["status"] = "FAILED"
    dump(directory / "export_audit.json", audit_result)
    report["export_audit"] = audit_result
    report["validation"] = validation
    report["artifact_hashes"] = {str(p): sha256(p) for p in out.rglob("*") if p.is_file()}
    dump(directory / "report.json", report)
    write_report(root, report)
    with (directory / "report.md").open("a", encoding="utf-8") as f:
        f.write("\n## Persisted export audit\n\n")
        f.write(
            f"Exact shared-parent overlap checks: {overlaps}. Protected input mismatches: {changed}. All review layers/raster CRS unset; verified line layers empty. Producer metadata was corrected only on new Phase 3.2 TIFFs and pixel arrays/affines were checked unchanged. Full details: [export_audit.json](export_audit.json).\n"
        )
    print(
        json.dumps(
            {
                "status": report["status"],
                "validation": validation,
                "overlap_checks": overlaps,
                "protected_mismatches": changed,
            }
        )
    )


if __name__ == "__main__":
    audit(Path(__file__).resolve().parents[1])
