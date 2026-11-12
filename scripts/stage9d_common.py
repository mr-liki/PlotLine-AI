"""Shared Stage 9D helpers: raw-query inspection, fixed-operating-point metrics, loaders.

Nothing here modifies sources, approvals or checkpoints. N9_01 is evaluation-only.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import label as connected

from roadmarking.training.synthetic_text_v1 import load_synthetic_text_sample
from stage9c_audit import ROOT, dump, sha256

BASE = ROOT / "outputs/reports/stage9d"
SYNTH = ROOT / "road_text_lidar_synthetic_500/road_text_lidar_synthetic_500"
TEXT = 2
NO_OBJECT = 3
SCORE_THRESHOLD = 0.25
MIN_PIXELS = 4


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def new_run(name):
    run = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = BASE / name / run
    out.mkdir(parents=True)
    dump(BASE / name / "latest.json", {"run": str(out)})
    return out


def frozen_checks():
    checks = read(ROOT / "outputs/reports/stage8b/ai_preservation.json")["checks"]
    return checks


def verify_frozen(checks):
    bad = [r["path"] for r in checks if sha256(Path(r["path"])) != r["expected_sha256"]]
    if bad:
        raise RuntimeError(f"Preserved artifact hash changed: {bad}")


def synthetic_rgb(sample):
    image = sample["intensity"].numpy()
    return np.repeat((image * 255).round().astype("uint8")[..., None], 3, axis=-1)


def synthetic(split, stem, processor, device="cuda"):
    sample = load_synthetic_text_sample(SYNTH / split, stem)
    rgb = synthetic_rgb(sample)
    pixels = processor(images=rgb, return_tensors="pt").pixel_values.to(device)
    return (
        pixels,
        sample["mask_labels"].to(device),
        sample["class_labels"].to(device),
        sample["valid_mask"].to(device),
        rgb,
        sample,
    )


def synthetic_stems(split, positive_only=True):
    stems = []
    for path in sorted((SYNTH / split / "metadata").glob("*.json")):
        meta = read(path)
        if not positive_only or meta["text_present"]:
            stems.append(path.stem)
    return stems


@torch.no_grad()
def raw_outputs(model, pixels, shape):
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(pixel_values=pixels, output_auxiliary_logits=False)
    native = torch.nn.functional.interpolate(
        out.masks_queries_logits.float(), size=shape, mode="bilinear", align_corners=False
    )[0]
    probs = out.class_queries_logits[0].float().softmax(-1)
    logits = out.class_queries_logits[0].float()
    return native, probs, logits


def postprocess(native, probs, valid, num_labels):
    """The unchanged Stage 9C fixed operating point (argmax incl. no-object)."""
    valid_t = torch.as_tensor(valid, device=native.device)
    masks = (native > 0) & valid_t
    area = masks.flatten(1).sum(1)
    quality = (native.sigmoid() * masks).flatten(1).sum(1) / area.clamp(min=1)
    top, labels = probs.max(-1)
    scores = top * quality
    keep = (scores >= SCORE_THRESHOLD) & (area >= MIN_PIXELS)
    masks_np = masks.cpu().numpy()
    keep_np, labels_np = keep.cpu().numpy(), labels.cpu().numpy()
    return {i: masks_np[keep_np & (labels_np == i)].any(axis=0) for i in range(num_labels)}


def counts(pred, target, scored):
    return np.array(
        [
            (pred & target & scored).sum(),
            (pred & ~target & scored).sum(),
            (~pred & target & scored).sum(),
        ],
        dtype=np.int64,
    )


def metrics(values):
    tp, fp, fn = map(int, values)
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        iou=tp / (tp + fp + fn) if tp + fp + fn else None,
    )


def query_table(native, probs, logits, target, valid):
    """Per-query raw diagnostics against a binary text target on valid pixels."""
    valid_t = torch.as_tensor(valid, device=native.device)
    target_t = torch.as_tensor(target, device=native.device) & valid_t
    masks = (native > 0) & valid_t
    area = masks.flatten(1).sum(1).float()
    inter = (masks & target_t).flatten(1).sum(1).float()
    union = (masks | target_t).flatten(1).sum(1).float()
    quality = (native.sigmoid() * masks).flatten(1).sum(1) / area.clamp(min=1)
    soft = native.sigmoid() * valid_t
    soft_dice = (
        2 * (soft * target_t).flatten(1).sum(1) / (soft.flatten(1).sum(1) + target_t.sum() + 1e-6)
    )
    return dict(
        area=area.cpu().numpy(),
        iou=(inter / union.clamp(min=1)).cpu().numpy(),
        recall=(inter / max(int(target_t.sum()), 1)).cpu().numpy(),
        precision=(inter / area.clamp(min=1)).cpu().numpy(),
        quality=quality.cpu().numpy(),
        soft_dice=soft_dice.cpu().numpy(),
        probs=probs.cpu().numpy(),
        logits=logits.cpu().numpy(),
    )


def shape_stats(mask):
    """Elongation of connected components: line-like masks have high aspect ratio."""
    lab, n = connected(mask)
    ratios = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(ys) < 8:
            continue
        cov = np.cov(np.stack([ys, xs]).astype(float))
        ev = np.sort(np.linalg.eigvalsh(cov))
        ratios.append(float(np.sqrt(ev[1] / max(ev[0], 1e-6))))
    return dict(components=int(n), median_elongation=float(np.median(ratios)) if ratios else None)


__all__ = [
    "BASE",
    "NO_OBJECT",
    "ROOT",
    "SYNTH",
    "TEXT",
    "counts",
    "dump",
    "frozen_checks",
    "metrics",
    "new_run",
    "postprocess",
    "query_table",
    "raw_outputs",
    "read",
    "sha256",
    "shape_stats",
    "synthetic",
    "synthetic_stems",
    "verify_frozen",
]
