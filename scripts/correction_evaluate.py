"""Three-way comparison (A frozen, B preserved step 100, C corrected) by condition, galleries."""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from correction_common import BASE, model_a_dir, model_b_dir, read
from regsafe_common import QUAL_300, dump, frozen_paths, load_model, load_processor, verify_frozen
from regsafe_evaluate import (
    aggregate,
    attributes,
    buckets,
    panel,
    per_crop,
    qualitative,
    unsupported_confusion,
)
from roadmarking.training.pilot import load_crop
from stage9e_lib import development_crops, real_context


@torch.no_grad()
def repeatability(path, processor, crop):
    outs = []
    for _ in range(2):
        m = load_model(path)
        x = load_crop(crop[0], crop[1], processor, "cuda")[0]
        o = m(pixel_values=x)
        outs.append((o.class_queries_logits.cpu(), o.masks_queries_logits.cpu()))
        del m
        torch.cuda.empty_cache()
    return all(torch.equal(a, b) for a, b in zip(outs[0], outs[1], strict=True))


def main():
    run = Path(read(BASE / "training/latest.json")["run"])
    decision = read(run / "decision.json")
    info = read(run / "training_info.json")
    step = decision["focus_checkpoint"]
    c_path = Path(info["candidate_dir"]) / f"step{step:04d}" / "model"
    out = run / "evaluation"
    out.mkdir(exist_ok=True)
    checks, _, _ = frozen_paths()
    processor = load_processor()
    sup = Path(read(BASE.parent / "regression_safe_sprint/supervision/latest.json")["run"])
    syn = read(sup / "synthetic_records.json")
    syn_dev = [(Path(r["file"]).parent, r) for r in syn if r["role"] == "development"]
    real_dev = development_crops(real_context()["ctx"])
    paths = {
        "A_frozen": model_a_dir() / "model",
        "B_step100": model_b_dir() / "model",
        "C_corrected": c_path,
    }
    models = {k: load_model(p) for k, p in paths.items()}
    ra = [attributes(r, c, processor) for r, c in real_dev]
    sa = [attributes(r, c, processor) for r, c in syn_dev]
    res = dict(
        candidate=str(c_path), step=step, accepted=decision["selected_checkpoint"] is not None
    )
    for name, m in models.items():
        rr, sr = per_crop(m, processor, real_dev), per_crop(m, processor, syn_dev)
        res[name] = dict(
            real=dict(all=aggregate(rr), conditions=buckets(rr, ra, "real")),
            synthetic=dict(
                all=aggregate(sr),
                conditions=buckets(sr, sa, "synthetic"),
                by_theme={
                    t: aggregate(
                        [r for r, (_, c) in zip(sr, syn_dev, strict=True) if c["theme"] == t]
                    )
                    for t in sorted({c["theme"] for _, c in syn_dev})
                },
            ),
            unsupported_paint_confusion=unsupported_confusion(m, processor, syn_dev),
        )
    res["repeatable_clean_reload_C"] = repeatability(c_path, processor, real_dev[0])
    dump(out / "three_way_metrics.json", res)
    gal = out / "gallery"
    gal.mkdir(exist_ok=True)
    entries = []
    for i, (root, crop) in enumerate(real_dev):
        im, _ = panel(models, processor, root, crop, crop["crop_id"])
        im.save(gal / f"real_{i:02d}.png")
        entries.append(("Real development (reused 14 crops)", crop["crop_id"], f"real_{i:02d}.png"))
    themes = defaultdict(list)
    for root, crop in syn_dev:
        themes[crop["theme"]].append((root, crop))
    for t, items in sorted(themes.items()):
        for root, crop in items[:4]:
            im, _ = panel(models, processor, root, crop, crop["crop_id"])
            im.save(gal / f"syn_{crop['crop_id']}.png")
            entries.append(
                (f"Synthetic development: {t}", crop["crop_id"], f"syn_{crop['crop_id']}.png")
            )
    for p in sorted((QUAL_300 / "images").glob("*.png"))[::15]:
        qualitative(models, processor, p).save(gal / f"q300_{p.stem}.png")
        entries.append(("300-image qualitative (visual only)", p.stem, f"q300_{p.stem}.png"))
    sections = defaultdict(list)
    for sec, cid, name in entries:
        sections[sec].append((cid, name))
    html = [
        "<!doctype html><meta charset='utf-8'><title>A / B / C comparison</title>",
        "<style>body{font-family:sans-serif;margin:16px}img{max-width:100%}</style>",
        f"<h1>Frozen (A) vs step-100 (B) vs corrected step {step} (C) - "
        f"{decision['status']}</h1><p>Red = solid, cyan = dash, V1 before stitching.</p>",
    ]
    for sec, items in sections.items():
        html.append(f"<h2>{sec}</h2>")
        html += [
            f"<figure><img src='gallery/{n}'><figcaption>{c}</figcaption></figure>"
            for c, n in items
        ]
    (out / "gallery.html").write_text("\n".join(html), encoding="utf-8")
    verify_frozen(checks)
    print(out, len(entries), res["repeatable_clean_reload_C"], np.nan)


if __name__ == "__main__":
    sys.exit(main())
