"""ONE-TIME final test of the frozen candidate on NEW_FINAL_TEST_RESERVED.

A write-once opening record is created BEFORE any protected imagery or label is read; if it
already exists the script refuses to run. Ground truth is exported exactly as for the
PHASE58 development area (reviewed_masks, NoData / unsupported-symbol / boundary exclusion,
plan_native_crops). The frozen candidate and the current default are both scored; nothing is
tuned afterwards.
"""

import hashlib
import json
import sys
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

import phase5_10_recovery as rec
from final_candidate import LOST, pipeline
from phase5_11_common import ROOT, load
from roadmarking.dataset.labels import LabelFeature
from roadmarking.dataset.native_development import plan_native_crops
from roadmarking.dataset.phase55 import write_attestation
from roadmarking.dataset.verified_masks import reviewed_masks
from roadmarking.evaluation.harness import block_records, paired_block_bootstrap, summarize
from roadmarking.inference.prediction_recovery import packed_iou
from roadmarking.rasterization.pipeline import dump, sha256
from stage1_scoring import DEFAULT, FROZEN, run_inference

OUT = ROOT / "outputs/reports/final_test"
DATA = ROOT / "data/processed/final_test"
NAME = "NEW_FINAL_TEST_RESERVED"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    opening = OUT / "test_opened.json"
    frozen_path = ROOT / "outputs/reports/final_candidate/frozen_candidate.json"
    if (OUT / "final_test_results.json").exists():
        sys.exit("Final test already completed once; refusing to run again.")
    resume = opening.exists()
    if resume and load(opening)["frozen_candidate_sha256"] != sha256(frozen_path):
        sys.exit("Opening record belongs to a different candidate; refusing.")
    frozen = load(frozen_path)
    assert sha256(frozen["weights"]) == frozen["weights_sha256"], "candidate weights changed"
    assert (
        sha256(ROOT / "src/roadmarking/inference/stitching.py") == frozen["stitching_code_sha256"]
    ), "stitching code changed since freezing"
    approval_path = (
        ROOT / "outputs/reports/phase5_7_expanded/20261106T111240383727Z/"
        "new_three_region_approval.json"
    )
    approval = load(approval_path)
    assert approval["roles"][NAME] == "protected_test"
    spec = load(
        ROOT / "outputs/reports/phase5_7_new_review/20261106T105319835838Z/"
        "new_region_candidates.json"
    )
    tile = next(t for t in spec["tiles"] if t["candidate_id"] == NAME)
    for p in tile["layers"].values():
        if p in approval["rasters"]:
            assert sha256(p) == approval["rasters"][p], f"protected raster changed: {p}"
    if not resume:
        write_attestation(
            opening,
            {
                "opened_utc": datetime.now(UTC).isoformat(),
                "region": NAME,
                "frozen_candidate_sha256": sha256(frozen_path),
                "approval_sha256": sha256(approval_path),
                "authorization": "User message: 'Run final test'",
                "policy": "single evaluation; results reported as-is; no tuning afterwards",
            },
        )

    # ---- ground-truth export (identical rules to the PHASE58 development export)
    frame = gpd.read_file(spec["candidate_store"], layer="candidate_paint_footprints")
    frame = frame[frame.candidate_id == NAME]
    approved = {g["hash"] for g in approval["geometry"] if g["region"] == NAME}
    frame = frame[[hashlib.sha256(g.wkb).hexdigest() in approved for g in frame.geometry]]
    affine = Affine(*tile["affine"])
    with rasterio.open(tile["layers"]["observed"]) as ds:
        observed = ds.read(1).astype(bool)
    features = [
        LabelFeature(i + 1, r.global_id, int(r.visual_id), "polygon", r.geometry)
        for i, (_, r) in enumerate(frame.iterrows())
        if int(r.visual_id) in (1, 2)
    ]
    unsupported = [r.geometry for _, r in frame.iterrows() if int(r.visual_id) not in (1, 2)]
    excluded = np.zeros(observed.shape, bool)
    if unsupported:
        excluded = binary_dilation(
            rasterize([(g, 1) for g in unsupported], observed.shape, transform=affine).astype(bool),
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
        mask = (labels.instance == entry["local_id"]) & valid
        if entry["local_id"] and mask.sum() >= 4:
            targets.append(mask)
            classes.append(entry["visual_id"] - 1)
            ids.append(entry["global_id"])
        elif entry["local_id"]:
            valid[mask] = False
    targets = np.stack(targets) if targets else np.empty((0, 1024, 1024), bool)
    plans, support, planning = plan_native_crops(targets, valid)
    assert plans and all(size == 512 for _, _, size in plans), planning
    rgb = np.asarray(Image.open(tile["images"]["grayscale"]).convert("RGB"))
    DATA.mkdir(parents=True, exist_ok=True)
    crops = []
    for row, col, size in plans:
        v = valid[row : row + size, col : col + size].copy()
        m = targets[:, row : row + size, col : col + size].copy()
        keep = m.sum((1, 2)) >= 4
        if (~keep).any():
            v &= ~m[~keep].any(0)
        path = DATA / f"{NAME}_s{size}_r{row}_c{col}.npz"
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
                "tile_id": NAME,
                "window": [row, col, size, size],
                "affine": list(affine * Affine.translation(col, row))[:6],
                "global_instance_ids": np.asarray(ids)[keep].tolist(),
            }
        )
    eligible = {u for c in crops for u in c["global_instance_ids"]}
    sel = np.array([u in eligible for u in ids], bool)
    if (~sel).any():
        valid[targets[~sel].any(0)] = False
    valid &= support
    gt_targets = targets[sel] & valid
    gt_classes = np.asarray(classes, int)[sel]
    gt_ids = np.asarray(ids)[sel].tolist()
    gt_path = DATA / f"{NAME}_ground_truth.npz"
    np.savez_compressed(
        gt_path, masks=gt_targets, labels=gt_classes, validity=valid, observed=observed
    )
    # ---- inference: frozen candidate and current default
    cand_dir = Path(frozen["weights"]).parents[1]
    run_inference(cand_dir, DATA, crops, OUT / "inference/candidate")
    run_inference(DEFAULT / "checkpoint_step0050", DATA, crops, OUT / "inference/default")
    results = {}
    settings = {
        "default_pipeline": (
            "default",
            {
                "confidence": FROZEN["confidence"],
                "mask_quality": FROZEN["mask_quality"],
                "nms_iou": 0.3,
            },
            False,
        ),
        "candidate_frozen_pipeline": (
            "candidate",
            {"confidence": frozen["thresholds_V1"], "mask_quality": 0.0, "nms_iou": 0.3},
            True,
        ),
    }
    for name, (model, thr, use_stitch) in settings.items():
        comp = rec.tile_component(NAME, tile, crops, OUT / "inference" / model, gt_path, gt_ids)
        comp["overlaps"] = rec.overlap_matrix(comp)
        masks, scores, cls, keep, audit = pipeline(comp, thr, use_stitch)
        iou = packed_iou(masks, comp["targets"], comp["valid"]).reshape(
            len(scores), len(comp["ids"])
        )
        recs, dets = block_records(comp, masks, scores, cls, keep, iou)
        results[name] = {
            "records": recs,
            "detections": dets,
            "audit": audit,
            "summary": summarize(recs.values(), dets.values()),
        }
    boot = paired_block_bootstrap(
        results["default_pipeline"]["records"],
        results["default_pipeline"]["detections"],
        results["candidate_frozen_pipeline"]["records"],
        results["candidate_frozen_pipeline"]["detections"],
        n=2000,
        seed=0,
    )
    out = {
        "status": "FINAL_TEST_COMPLETE_REPORTED_AS_IS",
        "region": NAME,
        "opened_record_sha256": sha256(opening),
        "ground_truth": {
            "solid_ids": int((gt_classes == 0).sum()),
            "dash_ids": int((gt_classes == 1).sum()),
            "scored_valid_pixels": int(valid.sum()),
            "observed_pixels": int(observed.sum()),
            "crops": len(crops),
            "crop_planning": planning,
        },
        "results": {k: {"summary": v["summary"], "audit": v["audit"]} for k, v in results.items()},
        "candidate_vs_default": boot,
        "lost_edge_id_present": LOST in gt_ids,
        "frozen_candidate": str(frozen_path),
    }
    dump(OUT / "final_test_results.json", out)
    print(
        json.dumps(
            {
                "ground_truth": out["ground_truth"],
                "summary": {k: v["summary"] for k, v in out["results"].items()},
                "bootstrap_blocks": boot["blocks"],
            },
            indent=1,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
