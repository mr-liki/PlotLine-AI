"""Record new three-region approval and export train/dev only; protect fresh final test."""

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

from roadmarking.dataset.labels import LabelFeature
from roadmarking.dataset.phase55 import check_dataset, write_attestation
from roadmarking.dataset.verified_masks import reviewed_masks
from roadmarking.rasterization.pipeline import dump, sha256


def check_expanded(root):
    root = Path(root)
    manifest = json.loads((root / "metadata/dataset_manifest.json").read_text())
    check_dataset(manifest["parent"])
    assert sha256(manifest["approval"]) == manifest["approval_sha256"]
    protected = set(manifest["protected_test_ids"])
    train = {
        uid for c in manifest["crops"] if c["split"] == "train" for uid in c["global_instance_ids"]
    }
    val = {
        uid for c in manifest["crops"] if c["split"] == "val" for uid in c["global_instance_ids"]
    }
    assert not (train & val) and not ((train | val) & protected)
    for crop in manifest["crops"]:
        assert crop["tile_id"] != "NEW_FINAL_TEST_RESERVED"
        assert sha256(crop["file"]) == crop["sha256"]
        with np.load(crop["file"]) as data:
            assert not data["masks"][:, ~data["validity"]].any()
            assert np.isin(data["labels"], [0, 1]).all()
    # Dataset-content check only: says nothing about whether the protected test was opened.
    return {"training_ready": True, "protected_test_crops_in_dataset": False}


def main():
    root = Path(__file__).resolve().parents[1]
    prior = json.loads((root / "outputs/reports/phase5_7/latest.json").read_text())
    review = json.loads((root / "outputs/reports/phase5_7_new_review/latest.json").read_text())
    spec = json.loads((Path(review["report_dir"]) / "new_region_candidates.json").read_text())
    run = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    report = root / "outputs/reports/phase5_7_expanded" / run
    output = root / "data/processed/phase5_training" / ("phase5_7_expanded_" + run)
    report.mkdir(parents=True)
    output.mkdir(parents=True)
    parent = Path(prior["dataset"])
    check_dataset(parent)
    old = json.loads((parent / "metadata/dataset_manifest.json").read_text())
    frame = gpd.read_file(spec["candidate_store"], layer="candidate_paint_footprints")
    roles = {
        "NEW_TRAIN_REVIEW": "train",
        "NEW_DEV_REVIEW": "val",
        "NEW_FINAL_TEST_RESERVED": "protected_test",
    }
    project = Path(spec["output"]) / "independent_test_review.qgs"
    approval = {
        "statement": (
            "User checked the new QGIS review package and confirms " "it is correct in all means"
        ),
        "scope": (
            "Geometry and complete visible targets within three new reviewed "
            "footprints; NoData excluded"
        ),
        "roles": roles,
        "project": str(project),
        "project_sha256": sha256(project),
        "candidate_store_sha256": sha256(spec["candidate_store"]),
        "geometry": [
            {
                "source_id": r.global_id,
                "region": r.candidate_id,
                "hash": hashlib.sha256(r.geometry.wkb).hexdigest(),
                "visual_id": int(r.visual_id),
            }
            for _, r in frame.iterrows()
        ],
        "rasters": {p: sha256(p) for t in spec["tiles"] for p in t["layers"].values()},
        "prior_approvals_preserved": True,
    }
    approval_path = report / "new_three_region_approval.json"
    approval_hash = write_attestation(approval_path, approval)
    crops = [{**c, "file": str(parent / c["file"])} for c in old["crops"]]
    protected_ids = set(frame[frame.candidate_id == "NEW_FINAL_TEST_RESERVED"].global_id)
    stats = {}
    for tile in spec["tiles"]:
        name = tile["candidate_id"]
        if roles[name] == "protected_test":
            continue
        with rasterio.open(tile["layers"]["observed"]) as ds:
            observed, affine = ds.read(1).astype(bool), ds.transform
        rows = frame[frame.candidate_id == name]
        assert not (set(rows.global_id) & protected_ids)
        features = [
            LabelFeature(i + 1, r.global_id, int(r.visual_id), "polygon", r.geometry)
            for i, (_, r) in enumerate(rows.iterrows())
            if r.visual_id in (1, 2)
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
        unsupported = [r.geometry for _, r in rows.iterrows() if r.visual_id not in (1, 2)]
        if unsupported:
            valid &= ~binary_dilation(
                rasterize([(g, 1) for g in unsupported], observed.shape, transform=affine).astype(
                    bool
                ),
                iterations=2,
            )
        masks, classes, ids = [], [], []
        for entry in labels.instances:
            mask = (labels.instance == entry["local_id"]) & valid
            if entry["local_id"] and mask.sum() >= 4:
                masks.append(mask)
                classes.append(entry["visual_id"] - 1)
                ids.append(entry["global_id"])
            elif entry["local_id"]:
                valid[mask] = False
        masks = np.stack(masks) if masks else np.empty((0, 1024, 1024), bool)
        rgb = np.asarray(Image.open(tile["images"]["grayscale"]).convert("RGB"))
        stats[name] = {
            "role": roles[name],
            "observable_ids": ids,
            "solid": classes.count(0),
            "dash": classes.count(1),
            "valid_pixels": int(valid.sum()),
        }
        for row in (0, 256, 512):
            for col in (0, 256, 512):
                v = valid[row : row + 512, col : col + 512].copy()
                m = masks[:, row : row + 512, col : col + 512].copy()
                keep = m.sum((1, 2)) >= 4
                if (~keep).any():
                    v &= ~m[~keep].any(0)
                if v.mean() < (0.1 if roles[name] == "train" else 0.01):
                    continue
                file = output / f"{name}_r{row}_c{col}.npz"
                np.savez_compressed(
                    file,
                    rgb=rgb[row : row + 512, col : col + 512],
                    masks=m[keep] & v,
                    labels=np.asarray(classes)[keep],
                    validity=v,
                )
                transform = affine * Affine.translation(col, row)
                crops.append(
                    {
                        "crop_id": file.stem,
                        "file": str(file),
                        "sha256": sha256(file),
                        "tile_id": name,
                        "size": 512,
                        "window": [row, col, 512, 512],
                        "affine": list(transform)[:6],
                        "split": roles[name],
                        "global_instance_ids": np.asarray(ids)[keep].tolist(),
                        "labels": np.asarray(classes)[keep].tolist(),
                    }
                )
    assert not (protected_ids & {uid for c in crops for uid in c["global_instance_ids"]})
    dump(
        output / "metadata/dataset_manifest.json",
        {
            "parent": str(parent),
            "approval": str(approval_path),
            "approval_sha256": approval_hash,
            "crops": crops,
            "statistics": stats,
            "protected_test_ids": sorted(protected_ids),
        },
    )
    dump(report / "readiness.json", check_expanded(output))
    synthetic = root / "road_markings_synthetic_500"
    dump(
        report / "synthetic_assessment.json",
        {
            "source": str(synthetic),
            "hashes": {
                p.name: sha256(p)
                for p in [
                    synthetic / "README.md",
                    synthetic / "generate_dataset.py",
                    synthetic / "annotations.json",
                ]
            },
            "used_in_current_pilot": False,
            "reason": (
                "Grouped dash instances and unverified native spatial scale; "
                "requires a separate adapter and real-validation ablation"
            ),
            "real_world_evaluation": False,
        },
    )
    context = {
        "run_id": run,
        "report_dir": str(report),
        "dataset": str(output),
        "initialization": prior["selected_checkpoint"],
        "new_statistics": stats,
    }
    dump(report.parent / "latest.json", context)
    print(json.dumps(context, indent=2))


if __name__ == "__main__":
    main()
