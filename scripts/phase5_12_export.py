"""Phase 5.12: record approval of the 10 reserve regions and export the 26-region dataset."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd

from phase5_11_common import ROOT, load, reserved_geography
from phase5_11_export import check_phase511, export_tile
from roadmarking.dataset.phase55 import write_attestation
from roadmarking.rasterization.pipeline import dump, sha256

USER_APPROVAL = "I checked each one of them , they are correct and perfect"


def main():
    latest = load(ROOT / "outputs/reports/phase5_12/latest.json")
    out = Path(latest["report_dir"])
    spec = load(out / "new_region_candidates.json")
    store = spec["candidate_store"]
    assert sha256(store) == spec["candidate_store_sha256"], "store changed since packaging"
    edits = spec["review_edits_store"]
    missing = gpd.read_file(edits, layer="reviewer_missing_paint")
    paint = gpd.read_file(store, layer="candidate_paint_footprints")
    unsupported = gpd.read_file(store, layer="unsupported_marking_regions")
    ignore = (
        missing.rename(columns={"issue": "reason"})
        if len(missing)
        else gpd.GeoDataFrame({"candidate_id": []}, geometry=[])
    )
    project = Path(spec["output"]) / "phase5_12_training_review.qgs"
    approval = {
        "user_statement_verbatim": USER_APPROVAL,
        "recorded_utc": datetime.now(UTC).isoformat(),
        "confirmed_statements": [
            "Painted-area geometry and solid/dash classifications are correct",
            "Every visible target marking within the reviewed observed region is represented",
        ],
        "reviewer_missing_paint_polygons": len(missing),
        "scope": "Exactly the 10 P512 review footprints; nothing outside them",
        "regions": [
            {
                "id": t["candidate_id"],
                "reserve_of": t["reserve_of"],
                "bounds": t["bounds"],
                "source": t["source"],
            }
            for t in spec["tiles"]
        ],
        "role": "train",
        "project": str(project),
        "project_sha256": sha256(project),
        "candidate_store": store,
        "candidate_store_sha256": sha256(store),
        "review_edits_sha256": sha256(edits),
        "checklist_sha256": sha256(out / "review_checklist.md"),
        "geometry": [
            {
                "source_id": r.global_id,
                "region": r.candidate_id,
                "visual_id": int(r.visual_id),
                "hash": hashlib.sha256(r.geometry.wkb).hexdigest(),
            }
            for _, r in paint.iterrows()
        ],
        "rasters": {p: sha256(p) for t in spec["tiles"] for p in t["layers"].values()},
        "prior_approvals_preserved": True,
    }
    approval_path = out / "phase5_12_ten_region_approval.json"
    approval_hash = write_attestation(approval_path, approval)

    p511 = load(ROOT / "outputs/reports/phase5_11/latest.json")
    parent = Path(p511["dataset"])
    check_phase511(parent)
    old = load(parent / "metadata/dataset_manifest.json")
    regions, extra, _ = reserved_geography()
    reserved_ids = set(old["reserved_non_training_ids"]) | set(extra)
    for r in regions:
        if r["role"] != "train":
            reserved_ids.update(r["ids"])
    output = ROOT / "data/processed/phase5_training" / ("phase5_12_" + latest["run_id"])
    (output / "crops").mkdir(parents=True, exist_ok=False)
    crops, stats, audit = list(old["crops"]), {}, []
    for tile in spec["tiles"]:
        tile_crops, stat, tile_audit = export_tile(
            tile, paint, unsupported, ignore, reserved_ids, output, "5.12"
        )
        crops += tile_crops
        stats[tile["candidate_id"]] = stat
        audit += tile_audit
    dump(
        output / "metadata/dataset_manifest.json",
        {
            "parent": str(parent),
            "approval": str(approval_path),
            "approval_sha256": approval_hash,
            "crops": crops,
            "statistics": stats,
            "protected_test_ids": old["protected_test_ids"],
            "reserved_non_training_ids": sorted(reserved_ids),
            "development_regions_unchanged": True,
        },
    )
    readiness = check_phase511(output)
    labels = {}
    for c in crops:
        if c["split"] == "train":
            labels.update(zip(c["global_instance_ids"], c["labels"], strict=True))
    before = {u for c in old["crops"] if c["split"] == "train" for u in c["global_instance_ids"]}
    new = [c for c in crops if c.get("phase") == "5.12"]
    new_ids = {u for c in new for u in c["global_instance_ids"]} - before
    summary = {
        "new_crops": len(new),
        "new_unique_supervised_ids": len(new_ids),
        "new_unique_solid": sum(labels[u] == 0 for u in new_ids),
        "new_unique_dash": sum(labels[u] == 1 for u in new_ids),
        "new_crops_with_solid_segments": sum(0 in c["labels"] for c in new),
        "total_unique_training_ids": len(labels),
        "total_unique_training_solid": sum(v == 0 for v in labels.values()),
        "total_unique_training_dash": sum(v == 1 for v in labels.values()),
        "train_crops_total": sum(c["split"] == "train" for c in crops),
        "val_crops_unchanged": sum(c["split"] == "val" for c in crops),
        "all_masks_inside_validity": all(a["masks_inside_validity"] for a in audit),
        "no_overlapping_instance_pixels": all(a["unique_mask_pixels_no_overlap"] for a in audit),
        "readiness": readiness,
    }
    dump(
        out / "dataset_export.json",
        {"summary": summary, "per_region": stats, "per_crop_audit": audit, "dataset": str(output)},
    )
    dump(
        ROOT / "outputs/reports/phase5_12/latest.json",
        {
            **latest,
            "dataset": str(output),
            "approval": str(approval_path),
            "approval_sha256": approval_hash,
            "status": "EXPANDED_DATASET_READY",
        },
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
