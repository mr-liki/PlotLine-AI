"""Phase 5.11 Stage 5-6: record the user's approval and export the 16 approved regions.

Approval scope is exactly the 16 yellow review footprints. NoData, unsupported symbols
(2 px margin), ambiguous boundaries and user-reported ignore zones (non-target pixels)
are excluded from supervision. Existing train/val crops are carried over unchanged.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation
from shapely.geometry import box

from phase5_7_expand_approved import check_expanded
from phase5_11_common import ROOT, load, reserved_geography
from roadmarking.dataset.labels import LabelFeature
from roadmarking.dataset.phase55 import write_attestation
from roadmarking.dataset.verified_masks import reviewed_masks
from roadmarking.rasterization.pipeline import dump, sha256

matplotlib.use("Agg")
import matplotlib.pyplot as plt

USER_APPROVAL = "Approve all 16"


def check_phase511(dataset):
    dataset = Path(dataset)
    manifest = load(dataset / "metadata/dataset_manifest.json")
    parent = load(Path(manifest["parent"]) / "metadata/dataset_manifest.json")
    if "reserved_non_training_ids" in parent:
        check_phase511(manifest["parent"])
    else:
        check_expanded(manifest["parent"])
    assert sha256(manifest["approval"]) == manifest["approval_sha256"]
    split_ids = {}
    for c in manifest["crops"]:
        split_ids.setdefault(c["split"], set()).update(c["global_instance_ids"])
    reserved = set(manifest["reserved_non_training_ids"])
    assert not (split_ids["train"] & split_ids.get("val", set()))
    assert not (split_ids["train"] & reserved)
    assert not (split_ids["train"] & set(manifest["protected_test_ids"]))
    for crop in manifest["crops"]:
        if crop.get("phase") not in ("5.11", "5.12"):
            continue
        assert sha256(crop["file"]) == crop["sha256"]
        with np.load(crop["file"]) as d:
            assert not d["masks"][:, ~d["validity"]].any()
            assert np.isin(d["labels"], [0, 1]).all()
            assert d["rgb"].shape == (512, 512, 3) and d["validity"].shape == (512, 512)
    # Dataset-content check only: says nothing about whether the protected test was opened.
    return {"training_ready": True, "protected_test_crops_in_dataset": False}


def export_tile(tile, paint, unsupported, ignore, reserved_ids, output, phase):
    """Export one approved region to native 512 crops (shared by Phase 5.11 and 5.12)."""
    crops, audit = [], []
    name = tile["candidate_id"]
    with rasterio.open(tile["layers"]["observed"]) as ds:
        observed, affine = ds.read(1).astype(bool), ds.transform
    assert np.allclose(tuple(affine)[:6], tile["affine"])
    rows = paint[paint.candidate_id == name]
    assert not (set(rows.global_id) & reserved_ids), f"{name}: reserved ID leakage"
    features = [
        LabelFeature(i + 1, r.global_id, int(r.visual_id), "polygon", r.geometry)
        for i, (_, r) in enumerate(rows.iterrows())
    ]
    labels, _ = reviewed_masks(
        affine,
        observed.shape,
        observed,
        np.ones_like(observed),
        features,
        completeness=box(*tile["bounds"]),
    )
    valid = np.isin(labels.state, [2, 3]) & observed
    excluded = {
        "nodata": int((~observed).sum()),
        "state_not_supervised": int((observed & ~valid).sum()),
    }
    u = unsupported[unsupported.candidate_id == name]
    if len(u):
        mask = binary_dilation(
            rasterize([(g, 1) for g in u.geometry], observed.shape, transform=affine).astype(bool),
            iterations=2,
        )
        excluded["unsupported_symbols_with_margin"] = int((valid & mask).sum())
        valid &= ~mask
    target = labels.instance > 0
    z = ignore[ignore.candidate_id == name]
    if len(z):
        zone = rasterize([(g, 1) for g in z.geometry], observed.shape, transform=affine).astype(
            bool
        )
        excluded["user_ignore_zone_non_target"] = int((valid & zone & ~target).sum())
        valid &= ~(zone & ~target)
    masks, classes, ids = [], [], []
    for entry in labels.instances:
        m = (labels.instance == entry["local_id"]) & valid
        if entry["local_id"] and m.sum() >= 4:
            masks.append(m)
            classes.append(entry["visual_id"] - 1)
            ids.append(entry["global_id"])
        elif entry["local_id"]:
            valid[m] = False
    masks = np.stack(masks) if masks else np.empty((0, *observed.shape), bool)
    rgb = np.asarray(Image.open(tile["images"]["grayscale"]).convert("RGB"))
    stats_entry = {
        "solid": classes.count(0),
        "dash": classes.count(1),
        "valid_pixels": int(valid.sum()),
        "excluded_pixels": excluded,
        "observable_ids": ids,
    }
    for row in (0, 256, 512):
        for col in (0, 256, 512):
            v = valid[row : row + 512, col : col + 512].copy()
            m = masks[:, row : row + 512, col : col + 512].copy()
            keep = m.sum((1, 2)) >= 4
            if (~keep).any():
                v &= ~m[~keep].any(0)
            if v.mean() < 0.1:
                continue
            file = output / "crops" / f"{name}_r{row}_c{col}.npz"
            kept = m[keep] & v
            np.savez_compressed(
                file,
                rgb=rgb[row : row + 512, col : col + 512],
                masks=kept,
                labels=np.asarray(classes, int)[keep],
                validity=v,
            )
            crops.append(
                {
                    "crop_id": file.stem,
                    "file": str(file),
                    "sha256": sha256(file),
                    "tile_id": name,
                    "phase": phase,
                    "size": 512,
                    "window": [row, col, 512, 512],
                    "affine": list(affine * Affine.translation(col, row))[:6],
                    "priority_tag": tile["priority_tag"],
                    "split": "train",
                    "global_instance_ids": np.asarray(ids)[keep].tolist(),
                    "labels": np.asarray(classes, int)[keep].tolist(),
                }
            )
            audit.append(
                {
                    "crop_id": file.stem,
                    "valid_fraction": float(v.mean()),
                    "instances": int(keep.sum()),
                    "solid": int((np.asarray(classes)[keep] == 0).sum()),
                    "dash": int((np.asarray(classes)[keep] == 1).sum()),
                    "foreground_pixels": int(kept.any(0).sum()) if len(kept) else 0,
                    "masks_inside_validity": bool(not kept[:, ~v].any()),
                    "unique_mask_pixels_no_overlap": (
                        bool((kept.sum(0) <= 1).all()) if len(kept) else True
                    ),
                }
            )
    return crops, stats_entry, audit


def main():
    latest = load(ROOT / "outputs/reports/phase5_11/latest.json")
    out = Path(latest["report_dir"])
    spec = load(out / "new_region_candidates.json")
    feedback_path = out / "review_feedback.json"
    feedback = load(feedback_path)
    store = spec["candidate_store"]
    assert sha256(store) == feedback["candidate_store_sha256_after"], "store changed since review"
    project = Path(spec["output"]) / "phase5_11_training_review.qgs"
    paint = gpd.read_file(store, layer="candidate_paint_footprints")
    unsupported = gpd.read_file(store, layer="unsupported_marking_regions")
    ignore = gpd.read_file(store, layer="review_ignore_zones")
    approval = {
        "user_statement_verbatim": USER_APPROVAL,
        "recorded_utc": datetime.now(UTC).isoformat(),
        "confirmed_statements": [
            "Painted marking geometry is correctly positioned",
            "Solid/dash classes are correct",
            "Every visible target marking inside each footprint is represented, except the "
            "three user-reported spots which are excluded as ignore zones",
            "Observed non-target pixels inside each footprint may be treated as background "
            "(outside ignore zones, unsupported symbols and NoData)",
        ],
        "scope": "Exactly the 16 review footprints below; nothing outside them",
        "regions": [
            {"id": t["candidate_id"], "bounds": t["bounds"], "source": t["source"]}
            for t in spec["tiles"]
        ],
        "role": "train",
        "project": str(project),
        "project_sha256": sha256(project),
        "candidate_store": store,
        "candidate_store_sha256": sha256(store),
        "review_feedback_sha256": sha256(feedback_path),
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
    approval_path = out / "phase5_11_sixteen_region_approval.json"
    approval_hash = write_attestation(approval_path, approval)

    p57 = load(ROOT / "outputs/reports/phase5_7_expanded/latest.json")
    parent = Path(p57["dataset"])
    check_expanded(parent)
    old = load(parent / "metadata/dataset_manifest.json")
    regions, extra_ids, _ = reserved_geography()
    reserved_ids = set(extra_ids)
    for r in regions:
        if r["role"] != "train":
            reserved_ids.update(r["ids"])
    reserved_ids |= {
        u for c in old["crops"] if c["split"] == "val" for u in c["global_instance_ids"]
    }
    output = ROOT / "data/processed/phase5_training" / ("phase5_11_" + latest["run_id"])
    (output / "crops").mkdir(parents=True, exist_ok=False)
    crops, stats, audit = list(old["crops"]), {}, []
    for tile in spec["tiles"]:
        tile_crops, stat, tile_audit = export_tile(
            tile, paint, unsupported, ignore, reserved_ids, output, "5.11"
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
    new = [c for c in crops if c.get("phase") == "5.11"]
    old_train = {u for c in old["crops"] if c["split"] == "train" for u in c["global_instance_ids"]}
    labels_by_id = {}
    for c in crops:
        if c["split"] == "train":
            labels_by_id.update(zip(c["global_instance_ids"], c["labels"], strict=True))
    new_ids = {u for c in new for u in c["global_instance_ids"]} - old_train
    summary = {
        "new_crops": len(new),
        "new_unique_supervised_ids": len(new_ids),
        "new_unique_solid": sum(labels_by_id[u] == 0 for u in new_ids),
        "new_unique_dash": sum(labels_by_id[u] == 1 for u in new_ids),
        "total_unique_training_ids": len(labels_by_id),
        "total_unique_training_solid": sum(v == 0 for v in labels_by_id.values()),
        "total_unique_training_dash": sum(v == 1 for v in labels_by_id.values()),
        "train_crops_total": sum(c["split"] == "train" for c in crops),
        "val_crops_unchanged": sum(c["split"] == "val" for c in crops),
        "new_background_only_crops": sum(not c["labels"] for c in new),
        "all_masks_inside_validity": all(a["masks_inside_validity"] for a in audit),
        "no_overlapping_instance_pixels": all(a["unique_mask_pixels_no_overlap"] for a in audit),
        "readiness": readiness,
    }
    dump(
        out / "dataset_export.json",
        {"summary": summary, "per_region": stats, "per_crop_audit": audit, "dataset": str(output)},
    )
    # Visual audit of a deterministic sample of new image-mask pairs.
    sample = new[:: max(1, len(new) // 12)][:12]
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for ax, crop in zip(axes.flat, sample, strict=False):
        with np.load(crop["file"]) as d:
            img = d["rgb"].astype(float) / 255
            v, m, lab = d["validity"], d["masks"], d["labels"]
        img[~v] = img[~v] * 0.35 + np.array([0.35, 0, 0.35])
        for mask, cls in zip(m, lab, strict=True):
            img[mask] = (1, 0.2, 0.1) if cls == 0 else (0.1, 0.5, 1)
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(f"{crop['crop_id']}: {len(lab)} inst", fontsize=8)
        ax.axis("off")
    for ax in axes.flat[len(sample) :]:
        ax.axis("off")
    fig.suptitle("Phase 5.11 exported pairs: red solid, blue dash, purple = not supervised")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "galleries" / "exported_pairs_audit.png", dpi=80)
    plt.close(fig)
    dump(
        ROOT / "outputs/reports/phase5_11/latest.json",
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
