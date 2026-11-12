"""Phase 5.13 Stage 4: threshold-margin sensitivity (diagnostic only) and metric variability.

Re-scores SAVED predictions of the three models at the frozen setting and at one
predeclared diagnostic setting (solid confidence .80; everything else frozen). This is
NOT a selection step: the frozen setting remains the primary comparison.
"""

import json
from pathlib import Path

import numpy as np

import phase5_10_recovery as rec
from phase5_11_common import ROOT, load
from phase5_11_evaluate import FROZEN, components
from roadmarking.inference.prediction_recovery import filter_geographic
from roadmarking.rasterization.pipeline import dump

DIAGNOSTIC = {"confidence": {"0": 0.80, "1": 0.55}, "mask_quality": 0.8, "nms_iou": 0.3}
LOST = "4b28eb52fa431a19bf87cd0a9314c37a7967d31fa3ed5f4e49f7454bfdec084b"


def main():
    out = ROOT / "outputs/reports/phase5_13"
    l11 = load(ROOT / "outputs/reports/phase5_11/latest.json")
    l12 = load(ROOT / "outputs/reports/phase5_12/latest.json")
    p59 = load(ROOT / "outputs/reports/phase5_9/latest.json")
    sources = {
        "default_step50": (
            ROOT
            / "models/mask2former_swin_s/phase5_7_expanded_20261106T111240383727Z"
            / "validation_step0050",
            Path(p59["report_dir"]) / "inference",
        ),
        "p511_step400": (
            Path(l11["training_run"]) / "validation_step0400",
            Path(l11["report_dir"]) / "evaluation/step0400_phase58",
        ),
        "p512_step400": (
            Path(l12["training_run"]) / "validation_step0400",
            Path(l12["report_dir"]) / "evaluation/step0400_phase58",
        ),
    }
    results = {}
    for name, (leg, new) in sources.items():
        comps = components(leg, new)
        results[name] = {}
        for label, cfg in (("frozen_solid_.85", FROZEN), ("diagnostic_solid_.80", DIAGNOSTIC)):
            entry = {}
            for comp in comps:
                sel = filter_geographic(
                    comp["scores"],
                    comp["classes"],
                    comp["qualities"],
                    comp["overlaps"],
                    comp["crop_ids"],
                    cfg,
                )
                m = rec.score(
                    comp,
                    comp["masks"][sel],
                    comp["scores"][sel],
                    comp["classes"][sel],
                    comp["iou"][sel],
                    comp["inter"][sel],
                )
                e = {
                    "per_class": {
                        c: {k: m["per_class"][c][k] for k in ("tp", "fp", "fn")}
                        for c in ("solid", "dash")
                    },
                    "macro_f1": m["macro_f1"],
                    "foreground_iou": m["foreground_iou"],
                    "pixel": {
                        k: m["pixel"]["all_supported_paint"][k] for k in ("precision", "recall")
                    },
                }
                if LOST in comp["ids"]:
                    j = comp["ids"].index(LOST)
                    solid = comp["masks"][sel][comp["classes"][sel] == 0]
                    union = solid.any(0) if len(solid) else np.zeros(comp["shape"], bool)
                    e["lost_edge_paint_coverage"] = float(
                        (union & comp["targets"][j]).sum() / comp["targets"][j].sum()
                    )
                entry[comp["name"]] = e
            results[name][label] = entry
    # Measurement variability: spread of the primary metrics across late checkpoints.
    variability = {}
    for phase, latest in (("phase5_11", l11), ("phase5_12", l12)):
        ev = load(Path(latest["report_dir"]) / "checkpoint_evaluation.json")["evaluations"]
        for area in ("LEGACY_POSITIVE_DEV", "NEW_POSITIVE_DEV"):
            for metric in ("foreground_iou", "macro_f1"):
                vals = [ev[s][area]["filtered"][metric] for s in ("300", "400", "500", "600")]
                variability[f"{phase}|{area}|{metric}"] = {
                    "values_steps_300_600": vals,
                    "std": float(np.std(vals)),
                    "range": float(max(vals) - min(vals)),
                }
    dump(
        out / "stage4_threshold_sensitivity.json",
        {
            "status": "DIAGNOSTIC_ONLY_NOT_SELECTION",
            "diagnostic_setting": DIAGNOSTIC,
            "results": results,
            "checkpoint_variability": variability,
        },
    )
    print(json.dumps(results, indent=1))
    print(json.dumps(variability, indent=1))


if __name__ == "__main__":
    main()
