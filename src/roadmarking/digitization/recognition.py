"""Accepted recognition on a tile grid with the frozen production contract.

Per tile: proposals with the established raw rule (argmax class, score = class probability x
mean sigmoid on observed support >= 0.25, area >= 4), masks clipped to observed support,
class-wise NMS 0.3 with zero confidence (per-crop NMS), then production ``stitch`` across
crops with the registered V1 thresholds applied AFTER merging (frozen_candidate contract).
Raw NoData behaviour is recorded separately and never delivered.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from roadmarking.inference.calibration import filter_instances, iou_matrix
from roadmarking.inference.stitching import stitch

RAW_SCORE, MIN_AREA, NMS_IOU = 0.25, 4, 0.3


@torch.no_grad()
def tile_proposals(model, processor, gray_tile, observed_tile, device="cuda"):
    rgb = np.repeat(gray_tile[..., None], 3, axis=2)
    pixels = processor(images=rgb, return_tensors="pt")["pixel_values"].to(device)
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(pixel_values=pixels, output_auxiliary_logits=False)
    native = torch.nn.functional.interpolate(
        out.masks_queries_logits.float(),
        size=observed_tile.shape,
        mode="bilinear",
        align_corners=False,
    )[0]
    probs = out.class_queries_logits[0].float().softmax(-1)
    top, labels = probs.max(-1)
    no_object = probs.shape[-1] - 1
    v = torch.as_tensor(observed_tile, device=native.device)
    raw = native > 0
    sel = raw & v
    area = sel.flatten(1).sum(1)
    quality = (native.sigmoid() * sel).flatten(1).sum(1) / area.clamp(min=1)
    score = top * quality
    keep = (labels != no_object) & (area >= MIN_AREA) & (score >= RAW_SCORE)
    idx = torch.nonzero(keep).flatten().tolist()
    return dict(
        raw_masks=raw[idx].cpu().numpy(),
        scores=score[idx].cpu().numpy(),
        labels=labels[idx].cpu().numpy(),
        qualities=quality[idx].cpu().numpy(),
        query_indices=idx,
    )


def recognize(model, processor, gray, observed, grid, thresholds, device="cuda"):
    masks, scores, classes, crop_ids, provenance = [], [], [], [], []
    diag = dict(raw_proposals=0, raw_mask_px_on_nodata=0, per_crop_nms_kept=0, latency_ms=[])
    for tid, (r, c) in grid.windows.items():
        sl = np.s_[r : r + grid.tile, c : c + grid.tile]
        obs = observed[sl]
        if obs.sum() < 50:
            continue
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pr = tile_proposals(model, processor, gray[sl], obs, device)
        torch.cuda.synchronize()
        diag["latency_ms"].append(1000 * (time.perf_counter() - t0))
        n = len(pr["labels"])
        diag["raw_proposals"] += n
        if not n:
            continue
        diag["raw_mask_px_on_nodata"] += int((pr["raw_masks"].any(0) & ~obs).sum())
        clipped = pr["raw_masks"] & obs[None]
        sel = filter_instances(
            pr["scores"],
            pr["labels"],
            iou_matrix(clipped, clipped, obs),
            pr["qualities"],
            dict(confidence={"0": 0.0, "1": 0.0, "2": 1.01}, mask_quality=0.0, nms_iou=NMS_IOU),
        )
        diag["per_crop_nms_kept"] += len(sel)
        for i in sel:
            m = np.zeros(observed.shape, bool)
            m[sl] = clipped[i]
            masks.append(m)
            scores.append(float(pr["scores"][i]))
            classes.append(int(pr["labels"][i]))
            crop_ids.append(tid)
            provenance.append(
                dict(
                    tile=tid,
                    query=int(pr["query_indices"][i]),
                    score=float(pr["scores"][i]),
                    cls=int(pr["labels"][i]),
                )
            )
    instances, audit = stitch(masks, scores, classes, crop_ids, grid.windows, observed, thresholds)
    for k, inst in enumerate(instances):
        inst["instance_id"] = k
        inst["provenance"] = [provenance[m] for m in inst["members"]]
        if (inst["mask"] & ~observed).any():
            raise AssertionError("Accepted paint outside observed support")
    diag.update(audit)
    diag["accepted_instances"] = len(instances)
    diag["accepted_px_outside_support"] = 0
    return instances, diag


__all__ = ["recognize", "tile_proposals"]
