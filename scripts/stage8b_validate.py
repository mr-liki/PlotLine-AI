"""Artifact-level regression checks for the confirmed single-mark amendment."""

import hashlib
from pathlib import Path

import numpy as np
import pyogrio
import rasterio
from rasterio.features import rasterize

from phase5_11_common import ROOT, load
from roadmarking.rasterization.pipeline import dump

OUT = ROOT / "outputs/reports/stage8b"


def main():
    approval = load(OUT / "approval_manifest.json")
    hashes = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
        for path, expected in approval["sha256"].items()
    }
    assert all(hashes.values()), "Approved evidence or original geometry changed"
    proposed = pyogrio.read_dataframe(
        OUT / "single_dash_proposed.gpkg", layer="proposed_dash_observed_paint"
    )
    final = pyogrio.read_dataframe(
        OUT / "single_dash_confirmed.gpkg", layer="confirmed_dash_observed_paint"
    )
    assert len(final) == len(proposed) == 1
    assert final.geometry.iloc[0].equals(proposed.geometry.iloc[0])
    assert final.iloc[0]["class_name"] == "dash" and final.iloc[0]["model_class"] == 1
    assert final.iloc[0]["client_global_id"] is None
    tile = load(OUT / "missing_mark_location.json")["winner"]["tile"]
    with rasterio.open(tile["layers"]["observed"]) as ds:
        observed, affine = ds.read(1).astype(bool), ds.transform
    with np.load(OUT / "single_dash_support.npz") as data:
        paint = data["proposed_paint"]
    reconstructed = rasterize(
        [(final.geometry.iloc[0], 1)], out_shape=observed.shape, transform=affine, dtype="uint8"
    ).astype(bool)
    assert np.array_equal(reconstructed, paint)
    assert int(paint.sum()) == 248 and not (paint & ~observed).any()
    assert abs(final.geometry.iloc[0].area - paint.sum() * 0.05**2) < 1e-8
    fringe = pyogrio.read_dataframe(
        OUT / "single_dash_confirmed.gpkg", layer="localized_uncertain_boundary_ignore"
    )
    fringe_mask = rasterize(
        [(g, 1) for g in fringe.geometry], out_shape=observed.shape, transform=affine, dtype="uint8"
    ).astype(bool)
    assert int(fringe_mask.sum()) == 11 and not (fringe_mask & paint).any()
    assert not (fringe_mask & ~observed).any()
    ai = load(OUT / "ai_preservation.json")
    assert all(
        hashlib.sha256(Path(r["path"]).read_bytes()).hexdigest() == r["expected_sha256"]
        for r in ai["checks"]
    )
    dump(
        OUT / "artifact_validation.json",
        {
            "all_passed": True,
            "approved_hash_count": len(hashes),
            "original_and_approved_evidence_hashes_match": hashes,
            "checks": [
                "one confirmed MultiPolygon feature / one dash identity",
                "confirmed geometry equals explicitly confirmed proposal",
                "no client feature ID invented",
                "native-grid raster reconstruction exactly matches 248 traced paint pixels",
                "zero paint pixels outside LiDAR observation",
                "paint geometry area exactly matches pixel-cell area",
                "11 uncertain boundary pixels ignored, disjoint from paint and NoData",
                "frozen/default weights and stitching match sealed hashes",
            ],
            "no_test_inference": True,
        },
    )
    print(
        "PASS: approved hashes, one exact native-grid dash, observed-only geometry, uncertainty exclusion and sealed AI artifacts"  # noqa: E501
    )


if __name__ == "__main__":
    main()
