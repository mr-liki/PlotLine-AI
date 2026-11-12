"""Phase 5.10: write the predeclared, write-once protocol and input hash manifest.

Run BEFORE any candidate filtering configuration is scored. No GT is read here except
for file hashing; no predictions are scored.
"""

import itertools
import json
from datetime import UTC, datetime
from pathlib import Path

from roadmarking.dataset.phase55 import write_attestation
from roadmarking.inference.prediction_recovery import RECONCILIATION_RULES
from roadmarking.rasterization.pipeline import dump, sha256

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models/mask2former_swin_s/phase5_7_expanded_20261106T111240383727Z"
SELECTED = CHECKPOINT / "checkpoint_step0050"
EXPECTED_WEIGHT = "98ab4462a1d23a4b7a3ee7b08b1480eb3f959e4b80968792107c7bb496a77786"
FROZEN = {"confidence": {"0": 0.85, "1": 0.55}, "mask_quality": 0.8, "nms_iou": 0.3}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sources():
    p57 = load(ROOT / "outputs/reports/phase5_7_expanded/latest.json")
    p59 = load(ROOT / "outputs/reports/phase5_9/latest.json")
    p55 = load(ROOT / "outputs/reports/phase5_5/latest.json")
    manifest = load(Path(p57["dataset"]) / "metadata/dataset_manifest.json")
    p55_manifest = load(Path(p55["dataset"]) / "metadata/dataset_manifest.json")
    p54 = load(Path(p55_manifest["parent_dataset"]) / "metadata/dataset_manifest.json")
    new_dev_spec = (
        ROOT
        / "outputs/reports/phase5_7_new_review/20261106T105319835838Z/new_region_candidates.json"
    )
    new_pos_spec = (
        ROOT
        / "outputs/reports/phase5_8_actual_review/20261106T115446528985Z/new_region_candidates.json"
    )
    return {
        "p57": p57,
        "p59": p59,
        "manifest_path": Path(p57["dataset"]) / "metadata/dataset_manifest.json",
        "manifest": manifest,
        "p54_path": Path(p55_manifest["parent_dataset"]) / "metadata/dataset_manifest.json",
        "p54": p54,
        "new_dev_spec": new_dev_spec,
        "new_pos_spec": new_pos_spec,
    }


def main():
    s = sources()
    weight = SELECTED / "model/model.safetensors"
    assert sha256(weight) == EXPECTED_WEIGHT, "Selected model weights changed"
    bundle = load(SELECTED / "bundle.json")
    assert bundle["metrics"]["postprocessing"] == FROZEN
    report59 = Path(s["p59"]["report_dir"])
    dataset59 = Path(s["p59"]["dataset"])
    val = [c for c in s["manifest"]["crops"] if c["split"] == "val"]
    legacy = [c["crop_id"] for c in val if not c["tile_id"].startswith("NEW_DEV_REVIEW")]
    sparse = [c["crop_id"] for c in val if c["tile_id"].startswith("NEW_DEV_REVIEW")]
    new59 = [c["crop_id"] for c in load(dataset59 / "manifest.json")["crops"]]
    assert len(legacy) == 5 and len(sparse) == 2 and len(new59) == 9
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = ROOT / "outputs/reports/phase5_10" / run_id
    out.mkdir(parents=True, exist_ok=False)

    grid = [
        {"confidence": {"0": a, "1": b}, "mask_quality": q, "nms_iou": n}
        for a, b, q, n in itertools.product(
            (0.55, 0.70, 0.85), (0.35, 0.55), (0.65, 0.80), (0.30, 0.50)
        )
    ]
    assert FROZEN in grid and len(grid) == 24

    # ----- Input hash manifest (model, approvals, saved predictions, GT, rasters) -----
    p54_tiles = {t["tile_id"]: t for t in s["p54"]["tiles"]}
    files = [
        weight,
        *sorted((SELECTED / "model").glob("*.json")),
        *sorted((SELECTED / "processor").glob("*.json")),
        SELECTED / "bundle.json",
        s["manifest_path"],
        s["p54_path"],
        s["new_dev_spec"],
        s["new_pos_spec"],
        report59 / "approved_positive_development_attestation.json",
        report59 / "frozen_model.json",
        report59 / "report.json",
        report59 / "native_crop_and_visibility_audit.json",
        dataset59 / "manifest.json",
        dataset59 / "PHASE58_POSITIVE_DEV_REVIEW_ground_truth.npz",
        dataset59 / "PHASE58_POSITIVE_DEV_REVIEW_approved_region_masks.npz",
        Path(s["manifest"]["approval"]),
        CHECKPOINT / "validation_step0050/metrics.json",
    ]
    for crop in val:
        files += [
            Path(crop["file"]),
            CHECKPOINT / "validation_step0050" / (crop["crop_id"] + ".npz"),
        ]
        if crop["tile_id"] in p54_tiles:
            files.append(
                Path(p54_tiles[crop["tile_id"]]["source_sample"]["phase3_layers"]["observed"])
            )
    for crop in load(dataset59 / "manifest.json")["crops"]:
        files += [Path(crop["file"]), report59 / "inference" / (crop["crop_id"] + ".npz")]
    for spec, region in (
        (s["new_dev_spec"], "NEW_DEV_REVIEW"),
        (s["new_pos_spec"], "PHASE58_POSITIVE_DEV_REVIEW"),
    ):
        for tile in load(spec)["tiles"]:
            # Only the named development tile is touched; the protected tile is never opened.
            if tile["candidate_id"] != region:
                continue
            files += [Path(tile["layers"]["observed"]), Path(tile["images"]["grayscale"])]
    hashes = {str(p): sha256(p) for p in dict.fromkeys(files)}
    approval = load(report59 / "approved_positive_development_attestation.json")
    assert all(sha256(p) == h for p, h in approval["rasters"].items())
    manifest_hash = write_attestation(out / "input_hash_manifest.json", {"sha256": hashes})

    protocol = {
        "phase": "5.10",
        "title": "Development-only prediction recovery and fragment reconciliation",
        "run_id": run_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "PREDECLARED_BEFORE_ANY_CANDIDATE_SCORING",
        "input_hash_manifest_sha256": manifest_hash,
        "model": {
            "checkpoint": str(SELECTED),
            "weight_sha256": EXPECTED_WEIGHT,
            "architecture": "facebook/mask2former-swin-small-coco-instance (Swin-S [2,2,18,2])",
            "classes": {"0": "longitudinal_solid", "1": "longitudinal_dash", "2": "no_object"},
            "weights_modified": False,
            "new_neural_inference": False,
            "training": False,
        },
        "data_roles": {
            "LEGACY_POSITIVE_DEV": {"crops": legacy, "use": "positive geographic development"},
            "NEW_POSITIVE_DEV": {
                "crops": new59,
                "region": "PHASE58_POSITIVE_DEV_REVIEW",
                "use": "positive geographic development (never training, never final test)",
            },
            "SPARSE_NEGATIVE_CONTROL": {
                "crops": sparse,
                "region": "NEW_DEV_REVIEW",
                "use": "negative control only; no positive targets in retained crops; "
                "not evidence of positive generalization",
            },
            "excluded": {
                "NEW_FINAL_TEST_RESERVED": "sealed; imagery, labels and predictions not read",
                "PHASE5_6_TEST_REGIONS": "historical diagnostic benchmarks; not used for tuning",
                "NEW_TRAIN_REVIEW_and_training_crops": "training data; not used",
                "synthetic_collections": "not used in Phase 5.10",
            },
            "components_processed_separately": True,
            "note": "Far-separated components are never placed on a shared raster canvas.",
        },
        "definitions": {
            "proposals": "Saved raw proposals (class != no-object, >=4 valid pixels, "
            "confidence = class softmax x mask quality >= .25 floor) from saved NPZs only.",
            "mask": "saved raw proposal AND saved crop validity, embedded on the component "
            "canvas by native affine/window (reproduces the original evaluation protocol)",
            "evidence": "saved raw proposal AND raw source observed TIFF within the crop window",
            "support": "raw source observed TIFF within the crop window",
            "intensity": "established grayscale rendering (shared normalization, gamma .7); "
            "Phase 5.9 grayscale PNG for the new region, saved crop RGB channel 0 for legacy",
            "ground_truth": "approved fragments merged per stable source ID within a "
            "component; clipped to scored valid support; used ONLY for evaluation and for "
            "this predeclared development operating-point selection",
            "scored_validity": "LEGACY: union of saved crop validity. NEW: Phase 5.9 exported "
            "evaluation validity (already verified inside actual crop validity). SPARSE: union "
            "of saved crop validity.",
            "filtering": "per-crop threshold (score>=conf[class], quality>=mask_quality) + "
            "same-class NMS (IoU>nms_iou suppresses), then global same-class NMS on the "
            "component canvas; IoU on crop-validity-clipped masks; ordering (-score, index)",
            "matching": "complete-instance greedy score-ordered same-class matching at IoU>=.50 "
            "(calibration.operating_counts) on scored validity",
            "ap": "101-recall-point per-class AP, AP50, AP75 and AP@[.5:.95] from cached IoU; "
            "confidence-truncated by the .25 proposal floor and by the operating point; not "
            "unrestricted pooled COCO AP",
            "pixel_metrics": "class-agnostic and per-class union masks within scored validity",
            "foreground_iou": "|GT union & pred union| / |(GT union | pred union) & valid|",
            "observed_skeleton_coverage": "skeleton of observed GT union within valid, fraction "
            "within 1 native pixel of predicted union (observed-support diagnostic only)",
            "solid_fragmentation": "per solid GT ID: same-class predicted masks with >=4 "
            "intersecting valid pixels",
            "dash_separation": "dash predictions covering >=4 px and >=10% of two or more "
            "approved dash masks (Phase 5.9 definition)",
            "crop_appearance_metrics": "per-crop filtering only, scored against crop targets "
            "(overlapping appearances; not unique geographic identities)",
            "duplicates": "unmatched predictions with IoU>=.5 to an already matched target",
        },
        "baseline_reproduction": {
            "configuration": FROZEN,
            "required_NEW_POSITIVE_DEV": {
                "0": {"tp": 0, "fp": 5, "fn": 8},
                "1": {"tp": 0, "fp": 18, "fn": 11},
                "global_nms_suppressed": 15,
                "post_nms_predictions": 23,
                "foreground_iou": 0.2059161701682868,
            },
            "required_crop_appearance_macro_f1": {
                "LEGACY_POSITIVE_DEV": 0.5927272727272728,
                "NEW_POSITIVE_DEV": 0.0971647,
            },
            "on_failure": "STOP; investigate clipping, ordering, threshold comparison, mapping "
            "and NMS. Do not proceed with changed definitions.",
            "legacy_geographic": "computed under identical definitions; no prior expected value "
            "(Phase 5.7 replay helper is context only)",
        },
        "search": {
            "grid": grid,
            "grid_size": len(grid),
            "frozen_included": True,
            "all_confidences_above_saved_floor": True,
            "expansion_after_results": "PROHIBITED",
        },
        "constraints": {
            "scope": "Geographic complete-instance metrics per positive component and class, "
            "compared with the frozen baseline under identical definitions",
            "C1_tp_floor": "TP >= baseline TP for every positive component and class",
            "C2_precision_floor_when_baseline_tp_positive": "precision >= baseline precision - .05",
            "C3_precision_floor_when_baseline_tp_zero_and_gain": "TP>0 needs precision>=.10",
            "C4_fp_budget": "per positive component, total and per class: FP <= "
            "max(baseline FP + 2, ceil(1.25 x baseline FP))",
            "C5_negative_control": "SPARSE_NEGATIVE_CONTROL geographic FP <= baseline FP",
        },
        "selection": {
            "order": [
                "maximize worst positive-component macro F1",
                "maximize mean positive-component macro F1",
                "minimize total FP across all three components",
                "prefer the frozen configuration",
                "prefer higher solid conf, higher dash conf, higher quality, lower NMS IoU",
            ],
            "improvement_required": "selected (worst, mean) macro F1 must strictly exceed the "
            "frozen baseline's tuple; otherwise the frozen configuration is retained",
            "caveat": "Selection uses the same two development geographies it is reported on; "
            "selected metrics are optimistic and not an unbiased generalization estimate.",
        },
        "reconciliation": {
            "rules": RECONCILIATION_RULES,
            "tuned": False,
            "inputs_forbidden": ["annotation geometry", "annotation IDs", "GT validity"],
            "applied_to": ["selected filtering output", "frozen filtering output (diagnostic)"],
            "adoption_criterion": "Recommend enabling only if, in BOTH positive components, "
            "every class has TP non-decreasing and FP non-increasing, the negative control FP "
            "is non-increasing, and at least one strict improvement exists (TP gain, FP drop, "
            "or fewer solid fragments). Otherwise report as not adopted.",
            "dashes": "never unioned",
        },
        "comparisons": [
            "saved raw proposals before filtering",
            "existing frozen filtering",
            "selected constrained filtering",
            "selected filtering + conservative reconciliation",
            "frozen filtering + conservative reconciliation (diagnostic)",
        ],
        "prohibited": [
            "weight modification",
            "retraining",
            "protected-test threshold optimization or any protected-test read",
            "annotation geometry for repair",
            "bridging missing LiDAR observations",
            "joining separate dashes",
            "extrapolating solids beyond observed evidence",
            "automatic annotation approval",
            "repeat annotation review",
            "production or 3D digitization",
            "grid expansion after observing results",
        ],
    }
    protocol_hash = write_attestation(out / "protocol.json", protocol)
    dump(
        ROOT / "outputs/reports/phase5_10/latest.json",
        {
            "run_id": run_id,
            "report_dir": str(out),
            "protocol_sha256": protocol_hash,
            "status": "PROTOCOL_PREDECLARED",
        },
    )
    print(
        json.dumps({"report_dir": str(out), "protocol_sha256": protocol_hash, "files": len(hashes)})
    )


if __name__ == "__main__":
    main()
