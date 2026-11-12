"""Record the explicit single-mark confirmation and bind immutable review inputs."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import rasterio
from affine import Affine
from rasterio.features import shapes
from shapely import coverage_union_all
from shapely.geometry import shape

from phase5_11_common import ROOT, load
from roadmarking.rasterization.pipeline import dump

OUT = ROOT / "outputs/reports/stage8b"


def main():
    # This script is invoked only after the user's explicit confirmation in this chat.
    record = load(OUT / "human_review_acceptance.json")
    approval = OUT / "approval_manifest.json"
    if approval.exists():
        raise ValueError("Approval already issued; never overwrite its hash-bound record")
    proposed = pyogrio.read_dataframe(
        OUT / "single_dash_proposed.gpkg", layer="proposed_dash_observed_paint"
    )
    assert len(proposed) == 1
    proposed["geometry_status"] = "HUMAN_CONFIRMED_SINGLE_MARK_CORRECTION"
    final = OUT / "single_dash_confirmed.gpkg"
    pyogrio.write_dataframe(
        proposed, final, layer="confirmed_dash_observed_paint", driver="GPKG", promote_to_multi=True
    )
    winner = load(OUT / "missing_mark_location.json")["winner"]
    tile = winner["tile"]
    affine = Affine(*tile["affine"])
    with np.load(OUT / "single_dash_support.npz") as data:
        paint = data["proposed_paint"]
        guard = data["pending_review_extent"]
    with rasterio.open(tile["layers"]["normalized"]) as ds:
        normalized = ds.read(1)
    fringe = guard & ~paint & (normalized > 0.20)
    if fringe.any():
        geometry = coverage_union_all(
            [
                shape(g)
                for g, v in shapes(fringe.astype("uint8"), mask=fringe, transform=affine)
                if v
            ]
        )
        pyogrio.write_dataframe(
            gpd.GeoDataFrame(
                [
                    {
                        "local_mark_id": proposed.iloc[0]["local_mark_id"],
                        "policy": "UNCERTAIN_PAINT_BOUNDARY_NOT_VERIFIED_BACKGROUND",
                        "pixels": int(fringe.sum()),
                        "geometry": geometry,
                    }
                ],
                geometry="geometry",
                crs=proposed.crs,
            ),
            final,
            layer="localized_uncertain_boundary_ignore",
            driver="GPKG",
            promote_to_multi=True,
            append=True,
        )
    for manifest in [
        ROOT / "outputs/reports/stage7_new_test/20261107T051050944913Z/new_region_candidates.json",
        Path(load(ROOT / "outputs/reports/stage8_new_solid_review/latest.json")["report_dir"])
        / "new_region_candidates.json",
    ]:
        spec = load(manifest)
        for path in [
            manifest,
            Path(spec["candidate_store"]),
            Path(spec["review_edits_store"]),
            *[Path(r["path"]) for r in spec.get("references", [])],
            *Path(spec["output"]).glob("*.qgs"),
            *Path(spec["output"]).glob("review_previews/*.png"),
        ]:
            record["input_sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    record.update(
        {
            "new_geometry_approval": "USER_CONFIRMED_SINGLE_MARK_CORRECTION",
            "confirmation_text": "Confirm this single-mark correction",
            "confirmation_context": "Explicit answer to the focused T7_05 before/after confirmation question in this chat.",  # noqa: E501
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "confirmed_patch": str(final),
            "uncertain_boundary_ignored_pixels": int(fringe.sum()),
            "evaluation_use_authorized": False,
            "final_hash_bound_approval_manifest": str(approval),
        }
    )
    dump(OUT / "human_review_acceptance.json", record)
    bound = [
        OUT / "human_review_acceptance.json",
        OUT / "single_dash_before_after.png",
        OUT / "single_dash_proposed.gpkg",
        final,
        OUT / "single_dash_support.npz",
        OUT / "single_dash_geometry_nodata_audit.json",
    ]
    files = dict(record["input_sha256"])
    files.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in bound})
    dump(
        approval,
        {
            "scope": "Reviewed geometry/classifications of eight T7 and two S8 regions, plus the one explicitly confirmed new dash patch. Does not certify source coverage, split independence, model performance or authorize test inference.",  # noqa: E501
            "reviewed_regions": record["reviewed_regions"],
            "single_mark_confirmation": record["confirmation_text"],
            "recorded_at_utc": record["recorded_at_utc"],
            "sha256": files,
            "effective_geometry": {
                "base_packages_immutable": True,
                "T7_05_additive_patch": str(final),
                "one_new_local_dash_identity": str(proposed.iloc[0]["local_mark_id"]),
                "client_feature_id_assigned": False,
                "uncertain_boundary_ignore_pixels": int(fringe.sum()),
            },
            "training_and_test_consumption_authorized": False,
        },
    )
    print(
        f"Hash-bound geometry approval issued; one confirmed dash, {int(fringe.sum())} uncertain fringe pixels ignored"  # noqa: E501
    )


if __name__ == "__main__":
    main()
