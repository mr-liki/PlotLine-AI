"""Phase 5.13 Stage 4 probe: do candidates lose solid confidence on TRAINING edge lines too?

Runs the three models on approved training crops only (never development/test imagery for
this probe) and, for every solid edge instance, records the best-overlapping query's solid
probability, mask quality and confidence (= class prob x quality), plus raw IoU.
"""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from phase5_11_common import EDGE, ROOT, client_features, load
from roadmarking.inference.calibration import iou_matrix
from roadmarking.rasterization.pipeline import dump
from roadmarking.training.pilot import load_crop

MODELS = {
    "default_step50": ROOT
    / "models/mask2former_swin_s/phase5_7_expanded_20261106T111240383727Z"
    / "checkpoint_step0050",
}


def main():
    l11 = load(ROOT / "outputs/reports/phase5_11/latest.json")
    l12 = load(ROOT / "outputs/reports/phase5_12/latest.json")
    MODELS["p511_step400"] = Path(l11["training_run"]) / "checkpoint_step0400"
    MODELS["p512_step400"] = Path(l12["training_run"]) / "checkpoint_step0400"
    designations = {
        f["global_id"]: f["designation"] for f in client_features(check_network=False)[0]
    }
    manifest = load(Path(l12["dataset"]) / "metadata/dataset_manifest.json")
    dataset = Path(l12["dataset"])
    groups = {"original": [], "new_5_11_5_12": []}
    for c in manifest["crops"]:
        if c["split"] != "train":
            continue
        if not any(
            designations.get(u) in EDGE and lab == 0
            for u, lab in zip(c["global_instance_ids"], c["labels"], strict=True)
        ):
            continue
        groups["original" if not c.get("phase") else "new_5_11_5_12"].append(c)
    sample = {k: v[:: max(1, len(v) // 40)][:40] for k, v in groups.items()}
    results = {}
    for name, folder in MODELS.items():
        model = (
            Mask2FormerForUniversalSegmentation.from_pretrained(
                folder / "model", local_files_only=True
            )
            .to("cuda")
            .eval()
        )
        processor = Mask2FormerImageProcessor.from_pretrained(
            folder / "processor", local_files_only=True
        )
        rows = {k: [] for k in sample}
        for key, crops in sample.items():
            for crop in crops:
                pixels, targets, labels, validity, _ = load_crop(dataset, crop, processor, "cuda")
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    pred = model(pixel_values=pixels, output_auxiliary_logits=False)
                logits = torch.nn.functional.interpolate(
                    pred.masks_queries_logits.float(),
                    size=validity[0].shape,
                    mode="bilinear",
                    align_corners=False,
                )[0]
                cls = pred.class_queries_logits[0].float().softmax(-1).cpu().numpy()
                prob = logits.sigmoid().cpu().numpy()
                masks = logits.cpu().numpy() > 0
                v = validity[0].cpu().numpy()
                truth = targets[0].cpu().numpy().astype(bool)
                tl = labels[0].cpu().numpy()
                ious = iou_matrix(masks, truth, v)
                for j, uid in enumerate(crop["global_instance_ids"]):
                    if tl[j] != 0 or designations.get(uid) not in EDGE or truth[j].sum() < 50:
                        continue
                    q = int(np.argmax(ious[:, j]))
                    m = masks[q] & v
                    quality = float(prob[q][m].mean()) if m.any() else 0.0
                    rows[key].append(
                        {
                            "solid_prob": float(cls[q][0]),
                            "quality": quality,
                            "confidence": float(cls[q][0]) * quality,
                            "raw_iou": float(ious[q, j]),
                            "fg_prob_on_paint": float(prob[q][truth[j] & v].mean()),
                        }
                    )
        results[name] = {
            key: {
                "instances": len(r),
                **{
                    f"median_{k}": float(np.median([x[k] for x in r]))
                    for k in ("solid_prob", "quality", "confidence", "raw_iou", "fg_prob_on_paint")
                },
                "fraction_conf_ge_0.85": float(np.mean([x["confidence"] >= 0.85 for x in r])),
            }
            for key, r in rows.items()
        }
        del model
        torch.cuda.empty_cache()
    dump(
        ROOT / "outputs/reports/phase5_13/stage4_training_edge_calibration_probe.json",
        {
            "note": "Training crops only; probes calibration/forgetting, not generalization",
            "sample_crops": {k: [c["crop_id"] for c in v] for k, v in sample.items()},
            "results": results,
        },
    )
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
