"""Stage 6: record approval; export S6_02..S6_10 to training and S6_01 as fresh development.

Training export reuses phase5_11_export.export_tile (identical rules; user-reported ignore
zones exclude only non-target pixels). S6_01 is exported exactly like the PHASE58
development area (reviewed_masks + plan_native_crops) and is NEVER training: its IDs are
added to the reserved non-training set.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation
from shapely.geometry import box

from phase5_11_common import ROOT, load, reserved_geography
from phase5_11_export import check_phase511, export_tile
from roadmarking.dataset.labels import LabelFeature
from roadmarking.dataset.native_development import plan_native_crops
from roadmarking.dataset.phase55 import write_attestation
from roadmarking.dataset.verified_masks import reviewed_masks
from roadmarking.rasterization.pipeline import dump, sha256

USER_APPROVAL = "Approve, S6_01 as dev"
DEV = "S6_01"


def export_dev(tile, paint, unsupported, out_dir):
    """PHASE58-style development export (never training)."""
    name = tile["candidate_id"]
    affine = Affine(*tile["affine"])
    with rasterio.open(tile["layers"]["observed"]) as ds:
        observed = ds.read(1).astype(bool)
    rows = paint[paint.candidate_id == name]
    features = [
        LabelFeature(i + 1, r.global_id, int(r.visual_id), "polygon", r.geometry)
        for i, (_, r) in enumerate(rows.iterrows())
    ]
    u = unsupported[unsupported.candidate_id == name]
    excluded = np.zeros(observed.shape, bool)
    if len(u):
        excluded = binary_dilation(
            rasterize([(g, 1) for g in u.geometry], observed.shape, transform=affine).astype(bool),
            iterations=2,
        )
    labels, _ = reviewed_masks(
        affine,
        observed.shape,
        observed,
        np.ones_like(observed),
        features,
        completeness=box(*tile["bounds"]),
    )
    valid = np.isin(labels.state, [2, 3]) & observed & ~excluded
    targets, classes, ids = [], [], []
    for entry in labels.instances:
        m = (labels.instance == entry["local_id"]) & valid
        if entry["local_id"] and m.sum() >= 4:
            targets.append(m)
            classes.append(entry["visual_id"] - 1)
            ids.append(entry["global_id"])
        elif entry["local_id"]:
            valid[m] = False
    targets = np.stack(targets) if targets else np.empty((0, 1024, 1024), bool)
    plans, support, planning = plan_native_crops(targets, valid)
    rgb = np.asarray(Image.open(tile["images"]["grayscale"]).convert("RGB"))
    out_dir.mkdir(parents=True, exist_ok=False)
    crops = []
    for row, col, size in plans:
        v = valid[row : row + size, col : col + size].copy()
        m = targets[:, row : row + size, col : col + size].copy()
        keep = m.sum((1, 2)) >= 4
        if (~keep).any():
            v &= ~m[~keep].any(0)
        path = out_dir / f"{name}_s{size}_r{row}_c{col}.npz"
        np.savez_compressed(
            path,
            rgb=rgb[row : row + size, col : col + size],
            masks=m[keep] & v,
            labels=np.asarray(classes, int)[keep],
            validity=v,
        )
        crops.append(
            {
                "crop_id": path.stem,
                "file": str(path),
                "sha256": sha256(path),
                "size": size,
                "tile_id": name,
                "window": [row, col, size, size],
                "affine": list(affine * Affine.translation(col, row))[:6],
                "global_instance_ids": np.asarray(ids)[keep].tolist(),
            }
        )
    eligible = {u for c in crops for u in c["global_instance_ids"]}
    sel = np.array([u in eligible for u in ids], bool)
    if len(sel) and (~sel).any():
        valid[targets[~sel].any(0)] = False
    valid &= support
    gt = out_dir / f"{name}_ground_truth.npz"
    np.savez_compressed(
        gt,
        masks=targets[sel] & valid,
        labels=np.asarray(classes, int)[sel],
        validity=valid,
        observed=observed,
    )
    manifest = {
        "role": "FRESH_DEVELOPMENT_NEVER_TRAIN_NEVER_SELECTION_YET",
        "region": name,
        "crops": crops,
        "ground_truth": str(gt),
        "ground_truth_sha256": sha256(gt),
        "eligible_ids": np.asarray(ids)[sel].tolist(),
        "crop_planning": planning,
        "solid_ids": int((np.asarray(classes)[sel] == 0).sum()) if len(sel) else 0,
        "dash_ids": int((np.asarray(classes)[sel] == 1).sum()) if len(sel) else 0,
        "scored_valid_pixels": int(valid.sum()),
    }
    dump(out_dir / "manifest.json", manifest)
    return manifest


def main():
    latest = load(ROOT / "outputs/reports/stage6_review/latest.json")
    out = Path(latest["report_dir"])
    spec = load(out / "new_region_candidates.json")
    store = spec["candidate_store"]
    assert sha256(store) == spec["candidate_store_sha256"], "store changed since packaging"
    feedback = load(out / "review_feedback.json")
    edits = spec["review_edits_store"]
    assert sha256(edits) == feedback["review_edits_sha256_after"], "edits changed since feedback"
    paint = gpd.read_file(store, layer="candidate_paint_footprints")
    unsupported = gpd.read_file(store, layer="unsupported_marking_regions")
    ignore = gpd.read_file(edits, layer="reviewer_missing_paint").rename(
        columns={"issue": "reason"}
    )
    project = Path(spec["output"]) / "stage6_review_training_review.qgs"
    roles = {
        t["candidate_id"]: ("fresh_development" if t["candidate_id"] == DEV else "train")
        for t in spec["tiles"]
    }
    approval = {
        "user_statement_verbatim": USER_APPROVAL,
        "recorded_utc": datetime.now(UTC).isoformat(),
        "confirmed_statements": [
            "Painted-area geometry and solid/dash classifications are correct",
            "Every visible target marking in the observed area is represented, except the "
            "user-reported S6_04 spots, which are excluded as ignore zones",
        ],
        "roles": roles,
        "scope": "Exactly the 10 S6 review footprints",
        "project": str(project),
        "project_sha256": sha256(project),
        "candidate_store_sha256": sha256(store),
        "review_edits_sha256": sha256(edits),
        "review_feedback_sha256": sha256(out / "review_feedback.json"),
        "geometry": [
            {
                "source_id": r.global_id,
                "region": r.candidate_id,
                "visual_id": int(r.visual_id),
                "hash": hashlib.sha256(r.geometry.wkb).hexdigest(),
            }
            for _, r in paint.iterrows()
        ],
        "ignore_zones": [
            {
                "region": r.candidate_id,
                "reason": r.reason,
                "hash": hashlib.sha256(r.geometry.wkb).hexdigest(),
            }
            for _, r in ignore.iterrows()
        ],
        "rasters": {p: sha256(p) for t in spec["tiles"] for p in t["layers"].values()},
        "prior_approvals_preserved": True,
    }
    approval_path = out / "stage6_ten_region_approval.json"
    approval_hash = write_attestation(approval_path, approval)
    # ---- fresh development region first (its IDs become reserved for training)
    dev_tile = next(t for t in spec["tiles"] if t["candidate_id"] == DEV)
    dev_dir = ROOT / "data/processed/stage6_fresh_dev" / latest["run_id"]
    dev_manifest = export_dev(dev_tile, paint, unsupported, dev_dir)
    dev_ids = set(paint[paint.candidate_id == DEV].global_id)
    # ---- training regions
    p12 = load(ROOT / "outputs/reports/phase5_12/latest.json")
    parent = Path(p12["dataset"])
    check_phase511(parent)
    old = load(parent / "metadata/dataset_manifest.json")
    regions, extra, _ = reserved_geography()
    reserved = set(old["reserved_non_training_ids"]) | set(extra) | dev_ids
    for r in regions:
        if r["role"] != "train":
            reserved.update(r["ids"])
    output = ROOT / "data/processed/phase5_training" / ("stage6_" + latest["run_id"])
    (output / "crops").mkdir(parents=True, exist_ok=False)
    crops, stats, audit = list(old["crops"]), {}, []
    for tile in spec["tiles"]:
        if roles[tile["candidate_id"]] != "train":
            continue
        tc, st, ta = export_tile(tile, paint, unsupported, ignore, reserved, output, "5.12")
        for c in tc:
            c["stage"] = "6"
        crops += tc
        stats[tile["candidate_id"]] = st
        audit += ta
    dump(
        output / "metadata/dataset_manifest.json",
        {
            "parent": str(parent),
            "approval": str(approval_path),
            "approval_sha256": approval_hash,
            "crops": crops,
            "statistics": stats,
            "protected_test_ids": old["protected_test_ids"],
            "reserved_non_training_ids": sorted(reserved),
            "development_regions_unchanged": True,
            "fresh_development_region": {"region": DEV, "dir": str(dev_dir)},
            "protected_final_test_consumed": True,
        },
    )
    readiness = check_phase511(output)
    labels = {}
    for c in crops:
        if c["split"] == "train":
            labels.update(zip(c["global_instance_ids"], c["labels"], strict=True))
    before = {u for c in old["crops"] if c["split"] == "train" for u in c["global_instance_ids"]}
    new = [c for c in crops if c.get("stage") == "6"]
    new_ids = {u for c in new for u in c["global_instance_ids"]} - before
    summary = {
        "new_training_regions": sorted(stats),
        "new_crops": len(new),
        "new_unique_supervised_ids": len(new_ids),
        "new_unique_solid": sum(labels[u] == 0 for u in new_ids),
        "new_unique_dash": sum(labels[u] == 1 for u in new_ids),
        "total_unique_training_ids": len(labels),
        "total_unique_training_solid": sum(v == 0 for v in labels.values()),
        "total_unique_training_dash": sum(v == 1 for v in labels.values()),
        "train_crops_total": sum(c["split"] == "train" for c in crops),
        "ignore_zone_pixels_excluded": {
            k: v["excluded_pixels"].get("user_ignore_zone_non_target", 0) for k, v in stats.items()
        },
        "all_masks_inside_validity": all(a["masks_inside_validity"] for a in audit),
        "no_overlapping_instance_pixels": all(a["unique_mask_pixels_no_overlap"] for a in audit),
        "dev_ids_absent_from_training": not (dev_ids & set(labels)),
        "fresh_dev": {k: dev_manifest[k] for k in ("solid_ids", "dash_ids", "scored_valid_pixels")}
        | {"crops": len(dev_manifest["crops"])},
        "readiness": readiness,
    }
    assert summary["dev_ids_absent_from_training"]
    dump(
        out / "dataset_export.json",
        {"summary": summary, "per_region": stats, "per_crop_audit": audit, "dataset": str(output)},
    )
    dump(
        ROOT / "outputs/reports/stage6_review/latest.json",
        {
            **latest,
            "dataset": str(output),
            "fresh_dev": str(dev_dir),
            "approval": str(approval_path),
            "approval_sha256": approval_hash,
            "status": "EXPANDED_DATASET_READY",
        },
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
