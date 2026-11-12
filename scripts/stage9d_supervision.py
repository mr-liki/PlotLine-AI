"""Stage 9D-B: verify synthetic text supervision without modifying any source."""

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label as connected
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from roadmarking.training.pilot import load_crop
from roadmarking.training.road_text_head import (
    CLASS_NAMES,
    NO_OBJECT_INDEX,
    make_three_class_head,
    model_class_from_visual,
)
from roadmarking.training.synthetic_text_v1 import load_synthetic_text_sample
from roadmarking.training.valid_criterion import ValidMask2FormerCriterion
from stage9c_attested_pilot import source_context
from stage9d_common import (
    SYNTH,
    dump,
    frozen_checks,
    new_run,
    sha256,
    verify_frozen,
)


def main():
    out = new_run("supervision")
    checks = frozen_checks()
    verify_frozen(checks)
    ctx = source_context()
    audit, train = ctx[1], ctx[6]
    dataset = ctx[4]
    source = Path(audit["source"])
    hashes_ok = all(sha256(source / p) == h for p, h in audit["source_file_sha256"].items())
    result = dict(source=str(source), source_hashes_verified=hashes_ok)
    # Class mapping
    result["class_mapping"] = dict(
        names=list(CLASS_NAMES),
        synthetic_visual_9_to_model=model_class_from_visual(9),
        no_object_index=NO_OBJECT_INDEX,
    )
    assert model_class_from_visual(9) == 2 and NO_OBJECT_INDEX == 3
    frozen = Mask2FormerForUniversalSegmentation.from_pretrained(
        Path(checks[0]["path"]).parent, local_files_only=True
    )
    old = frozen.class_predictor
    new = make_three_class_head(old, seed=903, source_labels=CLASS_NAMES[:2])
    result["head_migration"] = dict(
        old_rows=old.out_features,
        new_rows=new.out_features,
        solid_dash_rows_identical=bool(torch.equal(new.weight[:2], old.weight[:2]))
        and bool(torch.equal(new.bias[:2], old.bias[:2])),
        no_object_moved_2_to_3=bool(torch.equal(new.weight[3], old.weight[2])),
        text_row_bias=float(new.bias[2]),
    )
    # Per-sample checks over every split
    per_split = {}
    problems = []
    for split in ("train", "val", "test"):
        stats = Counter()
        align = []
        for meta_path in sorted((SYNTH / split / "metadata").glob("*.json")):
            stem = meta_path.stem
            s = load_synthetic_text_sample(SYNTH / split, stem)
            masks, labels, valid = s["mask_labels"], s["class_labels"], s["valid_mask"]
            img = s["intensity"].numpy()
            stats["samples"] += 1
            if not len(labels):
                stats["negatives"] += 1
                if not bool(valid.all()):
                    problems.append(f"{split}/{stem}: negative with ignore pixels")
                continue
            stats["positives"] += 1
            if labels.tolist() != [2]:
                problems.append(f"{split}/{stem}: labels {labels.tolist()}")
            vis = masks[0].bool().numpy()
            v = valid.numpy()
            with Image.open(SYNTH / split / "clean_masks" / f"{stem}.png") as raster:
                clean = np.array(raster) > 0
            stats["visible_pixels"] += int(vis.sum())
            stats["ignored_dropout_pixels"] += int((~v).sum())
            stats["valid_pixels"] += int(v.sum())
            if (vis & ~v).any():
                problems.append(f"{split}/{stem}: visible pixel marked invalid")
            if ((clean & ~vis) != ~v).any():
                problems.append(f"{split}/{stem}: ignore != intended minus visible")
            holes = binary_fill_holes(clean) & ~clean
            stats["letter_hole_pixels_valid_background"] += int((holes & v & ~vis).sum())
            stats["letter_hole_pixels_marked_positive"] += int((holes & vis).sum())
            stats["visible_components"] += int(connected(vis)[1])
            stats["phrase_instances"] += 1
            bg = v & ~vis & ~clean
            align.append(float(img[vis].mean() - img[bg].mean()))
            if img[vis].mean() <= img[bg].mean():
                problems.append(f"{split}/{stem}: visible paint not brighter than background")
        stats = dict(stats)
        stats["mean_fg_minus_bg_intensity_0_1"] = float(np.mean(align)) if align else None
        stats["min_fg_minus_bg_intensity_0_1"] = float(np.min(align)) if align else None
        stats["ignore_fraction_of_intended"] = stats.get("ignored_dropout_pixels", 0) / max(
            stats.get("ignored_dropout_pixels", 0) + stats.get("visible_pixels", 0), 1
        )
        per_split[split] = stats
    result["per_split"] = per_split
    result["problems"] = problems
    result["phrase_instance_policy"] = (
        "one instance per image = whole visible phrase (multi-word phrases kept as one "
        "instance); no character instances manufactured"
    )
    # Criterion behaviour on synthetic and real samples
    criterion = ValidMask2FormerCriterion(num_classes=3, bilinear=True)
    g = torch.Generator().manual_seed(0)
    crit = []
    for stem in [p.stem for p in sorted((SYNTH / "train" / "metadata").glob("*.json"))][:40]:
        s = load_synthetic_text_sample(SYNTH / "train", stem)
        if not len(s["class_labels"]):
            continue
        masks, labels, valid = s["mask_labels"], s["class_labels"], s["valid_mask"]
        idx, w = criterion.sample(masks, valid)
        fg = masks.flatten(1)[:, idx].sum().item()
        # Random predictions -> match exists, losses finite
        pred_masks = torch.randn(100, 128, 128, generator=g)
        pred_cls = torch.randn(100, 4, generator=g)
        losses, (qi, _ti) = criterion.one_image(pred_masks, pred_cls, masks, labels, valid, True)
        # ignore invariance: flip target inside invalid region
        flipped = masks.clone()
        flipped[0][~valid] = 1 - flipped[0][~valid]
        losses2, _ = criterion.one_image(pred_masks, pred_cls, flipped, labels, valid, True)
        crit.append(
            dict(
                stem=stem,
                sampled_points=len(idx),
                sampled_text_positive_points=int(fg),
                weights_sum=float(w.sum()),
                matched=len(qi),
                losses_finite=all(bool(torch.isfinite(v)) for v in losses.values()),
                ignore_invariant=all(bool(torch.equal(losses[k], losses2[k])) for k in losses),
                valid_all=bool(valid.all()),
            )
        )
    result["criterion_checks"] = dict(
        samples=len(crit),
        all_have_text_positive_points=all(r["sampled_text_positive_points"] > 0 for r in crit),
        min_text_positive_points=min(r["sampled_text_positive_points"] for r in crit),
        all_matched=all(r["matched"] == 1 for r in crit),
        all_finite=all(r["losses_finite"] for r in crit),
        all_ignore_invariant=all(r["ignore_invariant"] for r in crit),
        weights_sum_to_one=all(abs(r["weights_sum"] - 1) < 1e-4 for r in crit),
        synthetic_fully_valid_fraction=float(np.mean([r["valid_all"] for r in crit])),
        note=(
            "Synthetic images contain ignored dropout, so unmatched-query no-object CE is "
            "disabled by design; unmatched masks still receive null-mask BCE on valid pixels."
        ),
    )
    processor = Mask2FormerImageProcessor.from_pretrained(
        Path(checks[0]["path"]).parent.parent / "processor", local_files_only=True
    )
    real_valid = []
    for c in train[:60]:
        _, _m, lab, v, _ = load_crop(dataset, c, processor, "cpu")
        real_valid.append(dict(valid_all=bool(v[0].all()), labels=lab[0].tolist()))
    result["real_train_crops"] = dict(
        inspected=len(real_valid),
        fully_valid_fraction=float(np.mean([r["valid_all"] for r in real_valid])),
        label_values=sorted({x for r in real_valid for x in r["labels"]}),
    )
    verify_frozen(checks)
    result["sources_unmodified"] = all(
        sha256(source / p) == h for p, h in audit["source_file_sha256"].items()
    )
    dump(out / "synthetic_supervision_verification.json", result)
    print(out, len(problems), "problems")


if __name__ == "__main__":
    main()
