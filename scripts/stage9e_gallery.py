"""Stage 9E recognition galleries: frozen vs Stage 9E real-only arms (V1 accepted, clipped).

Colours: solid TP green, dash TP yellow, missed solid red, missed dash orange,
solid FP blue, dash FP cyan; grey = ignored/NoData.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from roadmarking.inference.calibration import filter_instances, iou_matrix
from roadmarking.training.pilot import load_crop
from stage9d_common import ROOT, frozen_checks, read, verify_frozen
from stage9e_lib import BASE, V1, development_crops, proposals, real_context


def label(img, text):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 + 6 * len(text), 13], fill=(0, 0, 0))
    d.text((3, 1), text, fill=(255, 255, 255))
    return np.array(im)


def panel(model, processor, root, crop):
    pixel, masks, labels, validity, rgb = load_crop(root, crop, processor, "cuda")
    valid = validity[0].cpu().numpy().astype(bool)
    truth = masks[0].bool().cpu().numpy()
    tl = labels[0].cpu().numpy()
    pr = proposals(model, pixel, valid)
    sel = []
    if len(pr["labels"]):
        sel = filter_instances(
            pr["scores"],
            pr["labels"],
            iou_matrix(pr["masks"], pr["masks"], valid),
            pr["qualities"],
            V1,
        )
    pm, plab = pr["masks"][sel], pr["labels"][sel]
    pred = {
        c: (pm[plab == c].any(0) if (plab == c).any() else np.zeros_like(valid)) & valid
        for c in (0, 1)
    }
    t = {c: truth[tl == c].any(0) if (tl == c).any() else np.zeros_like(valid) for c in (0, 1)}
    ov = (rgb * 0.55).astype(np.uint8)
    ov[~valid] = (ov[~valid] * 0.4 + 40).astype(np.uint8)
    ov[t[0] & ~pred[0] & valid] = [255, 0, 0]
    ov[t[1] & ~pred[1] & valid] = [255, 140, 0]
    ov[pred[0] & ~t[0]] = [0, 110, 255]
    ov[pred[1] & ~t[1]] = [0, 255, 255]
    ov[pred[0] & t[0]] = [0, 255, 0]
    ov[pred[1] & t[1]] = [255, 255, 0]
    return ov


def main():
    checks = frozen_checks()
    verify_frozen(checks)
    frozen = Path(checks[0]["path"]).parent
    processor = Mask2FormerImageProcessor.from_pretrained(
        frozen.parent / "processor", local_files_only=True
    )
    recipes = Path(read(BASE / "recipes/latest.json")["run"])
    mroot = ROOT / "models/mask2former_swin_s" / f"stage9e_{recipes.name}"
    arms = [("frozen", frozen)] + [
        (p.name.replace("_s9061", "").replace("_s9062", " seed2"), p / "model")
        for p in sorted(mroot.iterdir())
        if (p / "model").exists() and not p.name.startswith(("C1", "C2"))
    ]
    models = [
        (
            n,
            Mask2FormerForUniversalSegmentation.from_pretrained(p, local_files_only=True)
            .cuda()
            .eval(),
        )
        for n, p in arms
    ]
    rows = []
    for root, crop in development_crops(real_context()["ctx"])[:8]:
        cells = []
        for name, model in models:
            cells.append(
                label(panel(model, processor, root, crop), f"{name} | {crop['crop_id'][:22]}")
            )
        rows.append(np.concatenate(cells, 1)[::2, ::2])
    out = Path(read(BASE / "recipes/latest.json")["run"])
    Image.fromarray(np.concatenate(rows, 0)).save(out / "gallery_solid_dash_v1.png")
    verify_frozen(checks)
    print(out / "gallery_solid_dash_v1.png", [n for n, _ in arms])


if __name__ == "__main__":
    main()
