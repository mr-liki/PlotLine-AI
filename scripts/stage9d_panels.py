"""Stage 9D visual comparison panels: frozen baseline vs Stage 9D candidate.

Colours: green = true positive, red = missed target, blue = false positive.
Solid/dash panels: solid TP green, dash TP yellow, text predictions magenta.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from roadmarking.training.pilot import load_crop
from stage9c_attested_pilot import source_context, text_arrays
from stage9d_common import BASE, TEXT, frozen_checks, read, synthetic, verify_frozen
from stage9d_eval import development_crops, predict_full


def label(img, text):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 + 7 * len(text), 14], fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 255, 255))
    return np.array(im)


def text_overlay(rgb, pred, target, valid):
    ov = rgb.copy()
    ov[target & ~pred & valid] = [255, 0, 0]
    ov[pred & ~target & valid] = [0, 128, 255]
    ov[pred & target & valid] = [0, 255, 0]
    return ov


def line_overlay(rgb, post, truth, valid):
    ov = (rgb * 0.6).astype(np.uint8)
    ov[truth[0] & ~post[0] & valid] = [255, 0, 0]
    ov[truth[1] & ~post[1] & valid] = [255, 128, 0]
    ov[post[0] & ~truth[0] & valid] = [0, 128, 255]
    ov[post[1] & ~truth[1] & valid] = [0, 255, 255]
    ov[post[0] & truth[0]] = [0, 255, 0]
    ov[post[1] & truth[1]] = [255, 255, 0]
    ov[post[TEXT] & valid] = [255, 0, 255]
    return ov


def main():
    checks = frozen_checks()
    verify_frozen(checks)
    out = Path(read(BASE / "finetune/latest.json")["run"])
    decision = read(out / "gate_evaluation.json")
    frozen_path = Path(checks[0]["path"]).parent
    processor = Mask2FormerImageProcessor.from_pretrained(
        frozen_path.parent / "processor", local_files_only=True
    )
    models = {
        "frozen": Mask2FormerForUniversalSegmentation.from_pretrained(
            frozen_path, local_files_only=True
        )
        .cuda()
        .eval(),
        "stage9d": Mask2FormerForUniversalSegmentation.from_pretrained(
            Path(decision["candidate_corrected_copy"]) / "model", local_files_only=True
        )
        .cuda()
        .eval(),
    }
    ctx = source_context()
    rgb, positive, non_text, observed = text_arrays(ctx[0], ctx[9], ctx[10])
    pix = processor(images=rgb, return_tensors="pt").pixel_values.cuda()
    row = [label(rgb[:280], "N9_01 input")]
    for name, model in models.items():
        res = predict_full(model, pix, observed)
        ov = text_overlay(rgb, res["post"][TEXT], positive, observed)
        ov[non_text & res["post"][TEXT]] = [255, 0, 255]
        row.append(label(ov[:280], f"{name} text"))
    Image.fromarray(np.concatenate(row, 1)).save(out / "panel_n9_01_text.png")
    rows = []
    for root, crop in development_crops(ctx)[:6]:
        pixel, masks, labels, valid, crgb = load_crop(root, crop, processor, "cuda")
        v = valid[0].cpu().numpy().astype(bool)
        m = masks[0].bool().cpu().numpy()
        lab = labels[0].cpu().numpy()
        truth = {i: m[lab == i].any(0) if (lab == i).any() else np.zeros_like(v) for i in (0, 1)}
        cells = [label(crgb, crop["crop_id"][:28])]
        for name, model in models.items():
            res = predict_full(model, pixel, v)
            cells.append(label(line_overlay(crgb, res["post"], truth, v), name))
        rows.append(np.concatenate(cells, 1)[::2, ::2])
    Image.fromarray(np.concatenate(rows, 0)).save(out / "panel_solid_dash_dev.png")
    rows = []
    for stem in ("text_0401", "text_0412", "text_0425", "text_0440"):
        split = "val"
        try:
            pixels, mk, lb, valid, srgb, _ = synthetic(split, stem, processor)
        except FileNotFoundError:
            continue
        v = valid.cpu().numpy().astype(bool)
        t = mk[0].bool().cpu().numpy() if len(lb) else np.zeros_like(v)
        cells = [label(srgb, f"synthetic val {stem}")]
        for name, model in models.items():
            res = predict_full(model, pixels, v)
            cells.append(label(text_overlay(srgb, res["post"][TEXT], t, v), name))
        rows.append(np.concatenate(cells, 1)[::2, ::2])
    Image.fromarray(np.concatenate(rows, 0)).save(out / "panel_synthetic_val.png")
    verify_frozen(checks)
    print(out)


if __name__ == "__main__":
    main()
