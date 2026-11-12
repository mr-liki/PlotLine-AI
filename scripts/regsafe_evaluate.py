"""Step 8: frozen vs candidate by image condition, unsupported-paint confusion, galleries.

Condition buckets are descriptive splits of the same reused development crops; they are not
independent tests. The 300-image collection is rendered for visual review only (no metrics).
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from regsafe_common import (
    BASE,
    QUAL_300,
    SYN_PACKAGE,
    dump,
    frozen_paths,
    load_model,
    load_processor,
    read,
    verify_frozen,
)
from regsafe_eval import CLASSES, LEVELS
from regsafe_supervision import width_proxy
from roadmarking.inference.calibration import filter_instances, iou_matrix
from roadmarking.training.pilot import load_crop
from stage9d_common import metrics
from stage9e_lib import V1, development_crops, evaluate, proposals, real_context

UNSUPPORTED = {
    "arrow": {5, 6, 7, 8, 9, 10},
    "text": {16},
    "crossing": {3, 15},
    "other_paint": {2, 4, 11, 12, 13, 14, 17, 18, 19},
}
COLORS = {0: (255, 60, 60), 1: (60, 220, 255)}


def v1_select(pr, valid):
    if not len(pr["labels"]):
        return []
    return filter_instances(
        pr["scores"],
        pr["labels"],
        iou_matrix(pr["masks"], pr["masks"], valid),
        pr["qualities"],
        dict(V1, confidence={**V1["confidence"], "2": 1.01}),
    )


def attributes(root, crop, processor):
    _, masks, _labels, validity, rgb = load_crop(root, crop, processor, "cpu")
    valid = validity[0].numpy().astype(bool)
    truth = masks[0].bool().numpy()
    g = rgb[..., 0]
    fg = truth.any(0) if len(truth) else np.zeros_like(valid)
    widths = [w for w in (width_proxy(m & valid) for m in truth) if w is not None]
    bg = g[valid & ~fg]
    return dict(
        width=float(np.median(widths)) if widths else None,
        contrast=float(np.median(g[fg & valid]) - np.median(bg)) if fg.any() and len(bg) else None,
        observed_fraction=float((g > 0).mean()),
    )


def per_crop(model, processor, dev):
    rows = []
    for root, crop in dev:
        r = evaluate(model, processor, [(root, crop)])
        rows.append(r)
    return rows


def aggregate(rows):
    out = {}
    for lvl in LEVELS:
        out[lvl] = {}
        for c in CLASSES:
            pix = np.sum([[r[lvl]["pixel"][c][k] for k in ("tp", "fp", "fn")] for r in rows], 0)
            ins = np.sum(
                [[r[lvl]["instance_iou50"][c][k] for k in ("tp", "fp", "fn")] for r in rows], 0
            )
            out[lvl][c] = dict(pixel=metrics(pix), instance=metrics(ins))
    return out


def buckets(rows, attrs, domain):
    result = {}
    for key in ("width", "contrast", "observed_fraction"):
        if domain == "synthetic" and key == "observed_fraction":
            continue
        vals = [a[key] for a in attrs if a[key] is not None]
        if not vals:
            continue
        cut = float(np.median(vals))
        lo = [r for r, a in zip(rows, attrs, strict=True) if a[key] is not None and a[key] <= cut]
        hi = [r for r, a in zip(rows, attrs, strict=True) if a[key] is not None and a[key] > cut]
        names = {
            "width": ("thin", "thick"),
            "contrast": ("low_contrast", "high_contrast"),
            "observed_fraction": ("sparse_support", "dense_support"),
        }[key]
        result[names[0]] = dict(cut=cut, crops=len(lo), metrics=aggregate(lo) if lo else None)
        result[names[1]] = dict(cut=cut, crops=len(hi), metrics=aggregate(hi) if hi else None)
    return result


@torch.no_grad()
def unsupported_confusion(model, processor, syn_dev):
    out = defaultdict(int)
    for root, crop in syn_dev:
        pixel, _, _, validity, _ = load_crop(root, crop, processor, "cuda")
        valid = validity[0].cpu().numpy().astype(bool)
        sem = np.array(Image.open(SYN_PACKAGE / "semantic_masks" / f"{crop['crop_id']}.png"))
        pr = proposals(model, pixel, valid)
        sel = v1_select(pr, valid)
        for c in (0, 1):
            union = np.zeros_like(valid)
            for i in sel:
                if pr["labels"][i] == c:
                    union |= pr["masks"][i]
            for fam, ids in UNSUPPORTED.items():
                region = np.isin(sem, [i + 1 for i in ids])
                out[f"{fam}_px_under_v1_{CLASSES[c]}"] += int((union & region).sum())
    return dict(out)


def overlay(gray, masks, labels):
    img = np.repeat(gray[..., None], 3, 2).astype(float) * 0.6
    for m, lab in zip(masks, labels, strict=True):
        img[m] = 0.35 * img[m] + 0.65 * np.array(COLORS[int(lab)])
    return Image.fromarray(img.clip(0, 255).astype(np.uint8))


@torch.no_grad()
def predict(model, pixel, valid):
    pr = proposals(model, pixel, valid)
    sel = v1_select(pr, valid)
    return [pr["masks"][i] & valid for i in sel], [pr["labels"][i] for i in sel]


def panel(models, processor, root, crop, title, gt=True):
    pixel, masks, labels, validity, rgb = load_crop(root, crop, processor, "cuda")
    valid = validity[0].cpu().numpy().astype(bool)
    gray = rgb[..., 0]
    tiles = [("input", Image.fromarray(gray).convert("RGB"))]
    if gt:
        tiles.append(
            ("ground truth", overlay(gray, masks[0].bool().cpu().numpy(), labels[0].cpu().numpy()))
        )
    for name, model in models.items():
        tiles.append((f"{name} V1", overlay(gray, *predict(model, pixel, valid))))
    w = 256
    canvas = Image.new("RGB", (w * len(tiles), w + 18), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (name, im) in enumerate(tiles):
        canvas.paste(im.resize((w, w), Image.NEAREST), (i * w, 18))
        draw.text((i * w + 4, 3), name, fill="black")
    draw.text((4, w + 2 - 0), "", fill="black")
    return canvas, title


def qualitative(models, processor, path):
    gray = np.array(Image.open(path).convert("L"))
    rgb = np.repeat(gray[..., None], 3, 2)
    pixel = processor(images=rgb, return_tensors="pt")["pixel_values"].cuda()
    valid = np.ones(gray.shape, bool)
    tiles = [("input", Image.fromarray(gray).convert("RGB"))]
    for name, model in models.items():
        tiles.append((f"{name} V1", overlay(gray, *predict(model, pixel, valid))))
    w = 256
    canvas = Image.new("RGB", (w * len(tiles), w + 18), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (name, im) in enumerate(tiles):
        canvas.paste(im.resize((w, w), Image.NEAREST), (i * w, 18))
        draw.text((i * w + 4, 3), name, fill="black")
    return canvas


def main():
    train_dir = Path(read(BASE / "training/latest.json")["run"])
    decision = read(train_dir / "decision.json")
    step = decision["selected_checkpoint"] or decision["last_step"]
    info = read(train_dir / "training_info.json")
    cand_path = Path(info["candidate_dir"]) / f"step{step:04d}" / "model"
    out = train_dir / "evaluation"
    out.mkdir(exist_ok=True)
    checks, frozen, _ = frozen_paths()
    processor = load_processor()
    sup_dir = Path(read(BASE / "supervision/latest.json")["run"])
    syn = read(sup_dir / "synthetic_records.json")
    syn_dev = [(Path(r["file"]).parent, r) for r in syn if r["role"] == "development"]
    real_dev = development_crops(real_context()["ctx"])
    models = {"frozen": load_model(frozen), "candidate": load_model(cand_path)}
    real_attrs = [attributes(r, c, processor) for r, c in real_dev]
    syn_attrs = [attributes(r, c, processor) for r, c in syn_dev]
    result = dict(
        candidate=str(cand_path),
        candidate_step=step,
        accepted=bool(decision["selected_checkpoint"]),
    )
    for name, model in models.items():
        rr = per_crop(model, processor, real_dev)
        sr = per_crop(model, processor, syn_dev)
        mixed = [
            r
            for r, (_, c) in zip(sr, syn_dev, strict=True)
            if c["theme"] in ("mixed_intersection", "text", "arrow_left", "zebra")
        ]
        faded = [r for r, (_, c) in zip(sr, syn_dev, strict=True) if c["theme"] == "worn_lane"]
        result[name] = dict(
            real=dict(all=aggregate(rr), conditions=buckets(rr, real_attrs, "real")),
            synthetic=dict(
                all=aggregate(sr),
                conditions=buckets(sr, syn_attrs, "synthetic"),
                difficult_mixed_paint=aggregate(mixed) if mixed else None,
                faded_worn_lane=aggregate(faded) if faded else None,
                by_theme={
                    t: aggregate(
                        [r for r, (_, c) in zip(sr, syn_dev, strict=True) if c["theme"] == t]
                    )
                    for t in sorted({c["theme"] for _, c in syn_dev})
                },
            ),
            unsupported_paint_confusion=unsupported_confusion(model, processor, syn_dev),
        )
    dump(out / "condition_metrics.json", result)
    gal = out / "gallery"
    gal.mkdir(exist_ok=True)
    entries = []
    for i, (root, crop) in enumerate(real_dev):
        im, _ = panel(models, processor, root, crop, crop["crop_id"])
        name = f"real_{i:02d}.png"
        im.save(gal / name)
        entries.append(("Real development (reused)", crop["crop_id"], name))
    themes = defaultdict(list)
    for root, crop in syn_dev:
        themes[crop["theme"]].append((root, crop))
    for t, items in sorted(themes.items()):
        for root, crop in items[:6]:
            im, _ = panel(models, processor, root, crop, crop["crop_id"])
            name = f"syn_{crop['crop_id']}.png"
            im.save(gal / name)
            entries.append((f"Synthetic development: {t}", crop["crop_id"], name))
    qual = sorted((QUAL_300 / "images").glob("*.png"))[::12]
    for p in qual:
        im = qualitative(models, processor, p)
        name = f"q300_{p.stem}.png"
        im.save(gal / name)
        entries.append(("300-image qualitative (visual only, no metrics)", p.stem, name))
    sections = defaultdict(list)
    for sec, cid, name in entries:
        sections[sec].append((cid, name))
    html = [
        "<!doctype html><meta charset='utf-8'><title>Frozen vs candidate gallery</title>",
        "<style>body{font-family:sans-serif;margin:16px}img{max-width:100%}"
        "figure{margin:8px 0}</style>",
        f"<h1>Frozen vs candidate (step {step}, "
        f"{'accepted' if result['accepted'] else 'REJECTED'})</h1>",
        "<p>Red = solid, cyan = dash. V1 operating point before stitching, clipped to valid "
        "support on development crops. The 300-image panels are unclipped full-image V1 for "
        "visual review.</p>",
    ]
    for sec, items in sections.items():
        html.append(f"<h2>{sec}</h2>")
        for cid, name in items:
            html.append(
                f"<figure><img src='gallery/{name}'><figcaption>{cid}</figcaption></figure>"
            )
    (out / "gallery.html").write_text("\n".join(html), encoding="utf-8")
    verify_frozen(checks)
    print(out, len(entries))


if __name__ == "__main__":
    sys.exit(main())
