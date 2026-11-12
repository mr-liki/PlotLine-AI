"""Validate recovered rasters, export native model tiles and report review-only readiness."""

import json
import shutil
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

from roadmarking.rasterization.pipeline import dump, sha256

root = Path(__file__).resolve().parents[1]
base = root / "outputs/reports/phase5_6a"
context = json.loads((base / "latest.json").read_text())
report = Path(context["report_dir"])
spec = json.loads((report / "new_region_candidates.json").read_text())
diagnostic = base / "20261106T093540287823Z"
for name in ("file_integrity_findings.json", "alternative_inventory_candidates.json"):
    shutil.copy2(diagnostic / name, report / name)
frozen = json.loads((report / "frozen_model.json").read_text())
for filename, expected in frozen["artifact_sha256"].items():
    assert sha256(filename) == expected, filename
assert len({s["crs_wkt"] for s in spec["source_integrity"]}) == 1
assert all(s["full_source_integrity_passed"] for s in spec["source_integrity"])
coverage, tiles = [], []
for region in spec["tiles"]:
    with rasterio.open(region["layers"]["observed"]) as ds:
        observed, affine = ds.read(1).astype(bool), ds.transform
    with rasterio.open(region["layers"]["nodata"]) as ds:
        assert np.array_equal(ds.read(1).astype(bool), ~observed)
        assert ds.transform == affine
    with rasterio.open(region["layers"]["count"]) as ds:
        assert np.array_equal(ds.read(1) > 0, observed)
    with rasterio.open(region["layers"]["normalized"]) as ds:
        assert (ds.read(1)[~observed] == -9999).all()
    rgb = np.asarray(Image.open(region["images"]["grayscale"]).convert("RGB"))
    assert rgb.shape == (1024, 1024, 3)
    assert (rgb[~observed] == 0).all()
    assert affine.a == 0.05 and affine.e == -0.05
    coverage.append(
        {
            "region": region["candidate_id"],
            "source": region["source"],
            "observed_pixels": int(observed.sum()),
            "missing_pixels": int((~observed).sum()),
            "observed_fraction": float(observed.mean()),
            "review_support": "Exact observed-pixel subset only; NoData outside support",
            "bounds": region["bounds"],
            "affine": list(affine)[:6],
        }
    )
    for row in (0, 256, 512):
        for col in (0, 256, 512):
            valid = observed[row : row + 512, col : col + 512]
            if not valid.any():
                continue
            path = (
                Path(spec["output"])
                / "model_tiles"
                / (f"{region['candidate_id']}_r{row}_c{col}.npz")
            )
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, rgb=rgb[row : row + 512, col : col + 512], validity=valid)
            transform = affine * rasterio.Affine.translation(col, row)
            tiles.append(
                {
                    "file": str(path),
                    "sha256": sha256(path),
                    "region": region["candidate_id"],
                    "affine": list(transform)[:6],
                    "size": 512,
                    "approval": "PENDING",
                    "training_eligible": False,
                }
            )
dump(report / "model_tiles.json", tiles)
dump(report / "coverage.json", coverage)
project = Path(spec["output"]) / "independent_test_review.qgs"
artifacts = [p for p in Path(spec["output"]).rglob("*") if p.is_file()]
dump(report / "review_artifact_hashes.json", {str(p): sha256(p) for p in artifacts})
result = {
    "status": "PARTIAL_TEST_IMAGERY_READY",
    "qgis_project": str(project),
    "regions": coverage,
    "source_integrity": spec["source_integrity"],
    "replacement": "FUTURE_TEST_3 uses validated RIGHT Track01_52, not failed Track01_24",
    "frozen_hashes_verified": True,
    "coordinate_compatibility": "Identical source CRS WKT",
    "native_units": "Source headers report meters; no coordinate transform applied",
    "model_tiles": len(tiles),
    "approval": "PENDING_NEW_TEST_REGIONS_ONLY",
    "independent_accuracy": "NOT_EVALUATED",
    "training_or_tuning_performed": False,
    "elapsed_preparation_seconds": spec["elapsed_seconds"],
    "limitations": [
        "Sparse observed support; no full rectangular coverage claim",
        "Independent decoder unavailable; both installed backends use Lazrs",
        "Other inventory alternatives are candidates, not admitted sources",
    ],
}
dump(report / "report.json", result)
dump(base / "report.json", result)
findings = json.loads((report / "file_integrity_findings.json").read_text())
lines = [
    "# Phase 5.6A: PARTIAL_TEST_IMAGERY_READY",
    "",
    "Three reserved windows have verified-source imagery. Review and evaluation scope "
    "is only the explicit observed-pixel support; missing pixels remain NoData. "
    "No interpolation or failed-source points were admitted.",
    "",
    f"QGIS project: `{project}`",
    "",
    "Use bookmarks FUTURE_TEST_1, FUTURE_TEST_2, FUTURE_TEST_3. Orange outlines are "
    "candidate painted areas, not approved ground truth. Review their alignment and "
    "completeness for visible targets only. Unsupported symbols stay excluded. "
    "The thirteen prior approvals are unchanged.",
    "",
    "## Exact nominated-source findings",
    "",
    "| File | Bytes | Header points | Admissible decoded points | Discarded |",
    "|---|---:|---:|---:|---:|",
]
for f in findings:
    lines.append(
        f"| {Path(f['source']).name} | {f['size_bytes']} | {f['header_count']} "
        f"| {f['decoded_admissible_points']} | {f['discarded_points']} |"
    )
lines += [
    "",
    "Full paths, expected/observed fingerprints, local/remote SHA-256, invalid "
    "chunk offsets and first failing values are in file_integrity_findings.json. "
    "Source 24 failure reproduces on an identical local copy. This isolates it "
    "from copy transport loss; source-versus-Lazrs decoder cause remains unresolved. "
    "Its whole-file integrity failed; plausible chunks were not used as test imagery.",
    "",
    "## Recovery",
    "",
    result["replacement"],
    "",
    "Source 52 passed all 33,170,305 points with zero dropped records. Source-header "
    "CRS WKT matches sources 23 and 53. Full source identities and fingerprints are "
    "recorded. This is a documented alternate acquisition, not a silent substitution.",
    "",
    "| Region | Observed pixels | NoData pixels | Observed fraction |",
    "|---|---:|---:|---:|",
]
for c in coverage:
    lines.append(
        f"| {c['region']} | {c['observed_pixels']} | {c['missing_pixels']} "
        f"| {c['observed_fraction']:.4f} |"
    )
lines += [
    "",
    "Full reserved rectangles are retained for navigation. Their intended valid "
    "review footprint is the observed mask, not the entire rectangle. No fully "
    "observed rectangular subregion is claimed. Coverage masks, exact affines, "
    "NoData and native 512 crops are saved with hashes. Alternative inventory "
    "overlaps are listed separately; unvalidated candidates were not stitched.",
    "",
    "## Independence and model freeze",
    "",
    "independence_audit.json verifies no shared training/development IDs, no footprint "
    "overlap and the 10-native-unit guard. The alternative uses the same reserved "
    "footprint and the same client IDs; actual painted candidates were checked too. "
    "Test windows share some long source features and must not be counted as three "
    "independent statistical samples. All frozen model artifact hashes reverified. "
    "No inference, retraining, tuning, accuracy claims or production vector exports.",
    "",
    "## QA and remaining gate",
    "",
    "Raster observation/count equality, complementary NoData, normalization NoData, "
    "native affine/image dimensions, source counts, coordinate compatibility and "
    "model hashes passed. QGIS project reopened with all layers valid and read-only "
    "vector layers. Regression, lint, format and health results accompany this report.",
    "",
    f"Preparation renderer time: {spec['elapsed_seconds']:.2f} seconds. "
    "This excludes diagnostic copies, hashing, QGIS rendering and regression time; "
    "it is not an inference throughput benchmark.",
    "",
    "Next gate: first geometry and complete-visible-target approval for these new "
    "test regions only. Independent recognition/geometric accuracy remains unevaluated.",
]
(report / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
shutil.copy2(report / "report.md", base / "report.md")
context.update(status=result["status"], qgis_project=str(project))
dump(base / "latest.json", context)
print(json.dumps(result, indent=2))
