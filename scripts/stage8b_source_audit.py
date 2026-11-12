"""Verify source provenance and observed support without evaluating any model."""

import hashlib
from pathlib import Path

import numpy as np
import rasterio
import shapely
from affine import Affine
from shapely.geometry import box

from phase5_11_common import ROOT, client_features, load
from roadmarking.dataset.labels import LabelFeature
from roadmarking.dataset.verified_masks import reviewed_masks
from roadmarking.io.source_inventory import fingerprint
from roadmarking.rasterization.pipeline import dump

OUT = ROOT / "outputs/reports/stage8b"


def main():
    features = client_features(check_network=False)[0]
    tree = shapely.STRtree([f["geometry"] for f in features])
    sources, regions = {}, []
    manifests = [
        ROOT / "outputs/reports/stage7_new_test/20261107T051050944913Z/new_region_candidates.json",
        Path(load(ROOT / "outputs/reports/stage8_new_solid_review/latest.json")["report_dir"])
        / "new_region_candidates.json",
    ]
    for manifest in manifests:
        spec = load(manifest)
        scans = load(manifest.parent / "source_integrity.json")
        for tile in spec["tiles"]:
            source = tile["source"]
            passed = [
                s for s in scans if s["source"] == source and s["full_source_integrity_passed"]
            ]
            assert passed and all(
                s["expected_points"] == s["actual_points"] and s["dropped_points"] == 0
                for s in passed
            )
            scan = passed[-1]
            current = fingerprint(Path(source))
            assert current == scan["fingerprint"], source
            sources[source] = {
                "current_fingerprint": current,
                "full_decode_integrity_record": scan,
                "fingerprint_unchanged": True,
                "fingerprint_is_full_file_hash": False,
            }
            with rasterio.open(tile["layers"]["observed"]) as ds:
                observed = ds.read(1).astype(bool)
                affine = ds.transform
            assert affine.almost_equals(Affine(*tile["affine"]))
            geometry = box(*tile["bounds"])
            hits = tree.query(geometry, predicate="intersects")
            fs = [
                LabelFeature(
                    i + 1,
                    features[j]["global_id"],
                    features[j]["visual_id"],
                    "polygon",
                    features[j]["geometry"].intersection(geometry),
                )
                for i, j in enumerate(hits)
                if features[j]["visual_id"] in (1, 2)
            ]
            labels, _ = reviewed_masks(
                affine, observed.shape, observed, np.ones_like(observed), fs, completeness=geometry
            )
            counts = {}
            per_id = []
            for entry in labels.instances:
                if not entry["local_id"]:
                    continue
                mask = (labels.instance == entry["local_id"]) & observed
                pixels = int(mask.sum())
                counts[entry["global_id"]] = pixels
                if pixels < 4:
                    continue
                f = features[
                    next(int(j) for j in hits if features[j]["global_id"] == entry["global_id"])
                ]
                row, col = np.nonzero(mask)
                x, y = affine * (col + 0.5, row + 0.5)
                along = shapely.line_locate_point(f["centerline"], shapely.points(x, y))
                supported_bins = int(
                    np.count_nonzero(np.bincount(np.floor(along).astype(int)) >= 3)
                )
                per_id.append(
                    {
                        "global_id": entry["global_id"],
                        "class": "solid" if entry["visual_id"] == 1 else "dash",
                        "observed_pixels": pixels,
                        "supported_length_native_bins": supported_bins,
                    }
                )
            discrepancies = [
                p
                for p in tile["observed_ids"]
                if counts.get(p["global_id"], 0) != p["observed_pixels"]
            ]
            assert not discrepancies, f"Observed paint metadata mismatch in {tile['candidate_id']}"
            regions.append(
                {
                    "region": tile["candidate_id"],
                    "source": source,
                    "observed_pixels": int(observed.sum()),
                    "observed_fraction": float(observed.mean()),
                    "metadata_pixel_counts_reconciled": True,
                    "observed_raster_sha256": hashlib.sha256(
                        Path(tile["layers"]["observed"]).read_bytes()
                    ).hexdigest(),
                    "per_id_support": per_id,
                }
            )
            print(
                f"{tile['candidate_id']}: source fingerprint and per-ID observation counts match",
                flush=True,
            )
    patch = load(OUT / "single_dash_geometry_nodata_audit.json")
    dump(
        OUT / "source_integrity_and_coverage.json",
        {
            "sources": sources,
            "regions": regions,
            "confirmed_patch": {
                "region": "T7_05",
                "local_mark_id": patch["local_mark_id"],
                "observed_paint_pixels": patch["observed_paint_pixels"],
                "unobserved_paint_pixels": 0,
            },
            "source_integrity_passed_for_all_selected_regions": True,
            "method": "Reuse full-decode integrity only after exact partial-SHA256/size/mtime fingerprint match. Reconcile existing per-ID observed-pixel statistics from source geometry and observed raster; no model execution or evaluation metrics. Source fingerprint is explicitly not a full-file content hash.",  # noqa: E501
        },
    )


if __name__ == "__main__":
    main()
