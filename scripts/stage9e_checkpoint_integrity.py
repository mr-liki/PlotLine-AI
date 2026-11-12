"""Stage 9E-J: checkpoint integrity — train, save, clean-process reload, exact comparison.

Cases: (a) three-class migrated head (fixed loader) after 3 real + synthetic steps;
(b) two-class real-only decoder fine-tune after 3 steps. Inference is fp32 without
autocast on fixed inputs (one real development crop, one synthetic validation image),
before saving in the training process and after strict reload in a fresh process.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from roadmarking.training.pilot import load_crop
from stage9d_common import ROOT, dump, frozen_checks, read, sha256, synthetic, verify_frozen
from stage9d_train_lib import load_three_class, make_optimizer, set_trainable
from stage9e_lib import development_crops, new_run, real_context

LRS = dict(query_decoder=5e-6, mask_embedder=5e-6, class_predictor=5e-5)


def inputs(processor):
    rc = real_context()
    root, crop = development_crops(rc["ctx"])[0]
    real = load_crop(root, crop, processor, "cuda")[0]
    syn = synthetic("val", "text_0401", processor)[0]
    return [real, syn]


@torch.no_grad()
def outputs(model, xs):
    model.eval()
    res = []
    for x in xs:
        o = model(pixel_values=x, output_auxiliary_logits=False)
        res.append(
            (
                o.class_queries_logits.float().cpu().numpy(),
                o.masks_queries_logits.float().cpu().numpy(),
            )
        )
    return res


def train_and_save(kind, out):
    torch.manual_seed(9071)
    checks = frozen_checks()
    frozen = Path(checks[0]["path"]).parent
    processor = Mask2FormerImageProcessor.from_pretrained(
        frozen.parent / "processor", local_files_only=True
    )
    if kind == "three_class":
        model = load_three_class(frozen, 9071).cuda()
    else:
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            frozen, local_files_only=True
        ).cuda()
    set_trainable(model, ["query_decoder", "mask_embedder", "class_predictor"])
    optimizer, _ = make_optimizer(model, LRS)
    from roadmarking.training.deterministic_sampling import with_exact_bilinear
    from roadmarking.training.valid_criterion import ValidMask2FormerCriterion

    criterion = with_exact_bilinear(ValidMask2FormerCriterion)(
        num_classes=model.config.num_labels, bilinear=True
    ).cuda()
    rc = real_context()
    batches = []
    for c in rc["full"][:2]:
        p, m, lab, v, _ = load_crop(rc["dataset"], c, processor, "cuda")
        batches.append((p, m[0], lab[0], v[0]))
    if kind == "three_class":
        p, m, lab, v, _, _ = synthetic("train", "text_0001", processor)
        batches.append((p, m, lab, v))
    model.train()
    for p, m, lab, v in batches:
        optimizer.zero_grad(set_to_none=True)
        o = model(pixel_values=p, output_auxiliary_logits=False)
        losses, _ = criterion(
            o.masks_queries_logits.float(), o.class_queries_logits.float(), [m], [lab], [v], [True]
        )
        criterion.total(losses).backward()
        optimizer.step()
    before = outputs(model, inputs(processor))
    path = out / f"{kind}_checkpoint"
    model.save_pretrained(path / "model")
    processor.save_pretrained(path / "processor")
    np.savez(
        out / f"{kind}_before.npz",
        **{f"c{i}": c for i, (c, _) in enumerate(before)},
        **{f"m{i}": m for i, (_, m) in enumerate(before)},
    )
    dump(out / f"{kind}_config_before.json", model.config.to_dict())
    dump(out / f"{kind}_buffers_before.json", {n: list(b.shape) for n, b in model.named_buffers()})


def reload_and_compare(kind, out):
    processor = Mask2FormerImageProcessor.from_pretrained(
        out / f"{kind}_checkpoint/processor", local_files_only=True
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        out / f"{kind}_checkpoint/model", local_files_only=True
    ).cuda()
    after = outputs(model, inputs(processor))
    before = np.load(out / f"{kind}_before.npz")
    diffs = []
    for i, (c, m) in enumerate(after):
        diffs.append(
            dict(
                class_max_abs=float(np.abs(before[f"c{i}"] - c).max()),
                mask_max_abs=float(np.abs(before[f"m{i}"] - m).max()),
            )
        )
    cfg_before = read(out / f"{kind}_config_before.json")
    cfg_after = json.loads(json.dumps(model.config.to_dict()))
    ignore = {"_name_or_path", "transformers_version", "torch_dtype", "dtype"}
    cfg_diff = sorted(
        k
        for k in set(cfg_before) | set(cfg_after)
        if k not in ignore and cfg_before.get(k) != cfg_after.get(k)
    )
    buffers = {n: list(b.shape) for n, b in model.named_buffers()}
    dump(
        out / f"{kind}_reload.json",
        dict(
            strict_from_pretrained=True,
            output_diffs=diffs,
            outputs_identical=all(
                d["class_max_abs"] == 0 and d["mask_max_abs"] == 0 for d in diffs
            ),
            config_differences=cfg_diff,
            buffers_identical=buffers == read(out / f"{kind}_buffers_before.json"),
            num_labels=model.config.num_labels,
            id2label=model.config.id2label,
            class_rows=model.class_predictor.out_features,
            criterion_empty_weight=list(model.criterion.empty_weight.shape),
            weights_sha256=sha256(out / f"{kind}_checkpoint/model/model.safetensors"),
        ),
    )


def main():
    if len(sys.argv) == 4:
        {"train": train_and_save, "reload": reload_and_compare}[sys.argv[1]](
            sys.argv[2], Path(sys.argv[3])
        )
        return
    out = new_run("checkpoint_integrity")
    checks = frozen_checks()
    verify_frozen(checks)
    for kind in ("three_class", "two_class_real_only"):
        subprocess.run([sys.executable, __file__, "train", kind, str(out)], check=True)
        subprocess.run([sys.executable, __file__, "reload", kind, str(out)], check=True)
    stage9d = Path(read(ROOT / "outputs/reports/stage9d/finetune/latest.json")["run"])
    gate = read(stage9d / "gate_evaluation.json")
    rejected = Path(gate["candidate"]).parent
    kept = sorted(str(p.relative_to(rejected)) for p in rejected.rglob("model.safetensors"))
    summary = {k: read(out / f"{k}_reload.json") for k in ("three_class", "two_class_real_only")}
    summary["frozen_two_class_hashes_verified"] = True
    summary["stage9d_rejected_checkpoints_retained"] = dict(
        root=str(rejected), weights=kept, promoted=False
    )
    verify_frozen(checks)
    dump(out / "checkpoint_integrity.json", summary)
    print(
        out,
        {
            k: (v["outputs_identical"], v["config_differences"], v["criterion_empty_weight"])
            for k, v in summary.items()
            if isinstance(v, dict) and "outputs_identical" in v
        },
    )


if __name__ == "__main__":
    main()
