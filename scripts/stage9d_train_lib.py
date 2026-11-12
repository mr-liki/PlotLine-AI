"""Stage 9D training helpers: explicit component freezing, optimizer audit, losses."""

from pathlib import Path

import numpy as np
import torch
from transformers import Mask2FormerForUniversalSegmentation

from roadmarking.training.road_text_head import CLASS_NAMES, make_three_class_head
from roadmarking.training.valid_criterion import ValidMask2FormerCriterion
from stage9d_common import TEXT, counts, metrics, postprocess, query_table, raw_outputs

GROUPS = {
    "backbone": "model.pixel_level_module.encoder.",
    "pixel_decoder": "model.pixel_level_module.decoder.",
    "mask_embedder": "model.transformer_module.decoder.mask_predictor.",
    "query_decoder": "model.transformer_module.",
    "class_predictor": "class_predictor.",
}


def group_of(name):
    for group, prefix in GROUPS.items():
        if name.startswith(prefix):
            return group
    return "other"


def load_three_class(frozen_path, seed):
    model = Mask2FormerForUniversalSegmentation.from_pretrained(frozen_path, local_files_only=True)
    old = model.class_predictor
    model.class_predictor = make_three_class_head(old, seed=seed, source_labels=CLASS_NAMES[:2])
    model.config.num_labels = 3
    model.config.id2label = dict(enumerate(CLASS_NAMES))
    model.config.label2id = {v: k for k, v in model.config.id2label.items()}
    # The internal HF criterion owns an empty_weight buffer sized num_labels + 1. Rebuild it,
    # otherwise the saved checkpoint cannot be reloaded strictly (Stage 9D G7 defect).
    model.criterion = type(model.criterion)(model.config, model.weight_dict)
    return model


def set_trainable(model, groups):
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        if group_of(name) in groups:
            param.requires_grad_(True)
    return audit_trainable(model)


def audit_trainable(model):
    out = {}
    for name, param in model.named_parameters():
        g = group_of(name)
        row = out.setdefault(g, dict(tensors=0, params=0, trainable_tensors=0, trainable_params=0))
        row["tensors"] += 1
        row["params"] += param.numel()
        if param.requires_grad:
            row["trainable_tensors"] += 1
            row["trainable_params"] += param.numel()
    return out


def make_optimizer(model, lrs, weight_decay=0.01):
    groups = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            groups.setdefault(group_of(name), []).append(param)
    optimizer = torch.optim.AdamW(
        [dict(params=p, lr=lrs[g], name=g) for g, p in groups.items()], weight_decay=weight_decay
    )
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    in_opt = {id(p) for g in optimizer.param_groups for p in g["params"]}
    membership = dict(
        optimizer_equals_requires_grad=trainable == in_opt,
        groups={
            g["name"]: dict(lr=g["lr"], tensors=len(g["params"])) for g in optimizer.param_groups
        },
    )
    return optimizer, membership


def gradient_audit(model):
    out = {}
    for name, param in model.named_parameters():
        g = group_of(name)
        row = out.setdefault(
            g, dict(with_grad=0, zero_grad=0, nonfinite=0, sq=0.0, frozen_with_grad=0)
        )
        if param.grad is None:
            continue
        if not param.requires_grad:
            row["frozen_with_grad"] += 1
        if not torch.isfinite(param.grad).all():
            row["nonfinite"] += 1
        n = float(param.grad.float().norm())
        row["sq"] += n * n
        row["with_grad" if n > 0 else "zero_grad"] += 1
    for row in out.values():
        row["norm"] = row.pop("sq") ** 0.5
    return out


def step_loss(model, criterion, pixels, masks, labels, valid, aux=True):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(pixel_values=pixels, output_auxiliary_logits=aux)
    with torch.autocast("cuda", enabled=False):
        aux_logits = (
            [{k: v.float() for k, v in a.items()} for a in (pred.auxiliary_logits or [])]
            if aux
            else None
        )
        losses, matches = criterion(
            pred.masks_queries_logits.float(),
            pred.class_queries_logits.float(),
            [masks],
            [labels],
            [valid],
            [True],
            aux_logits,
        )
        total = criterion.total(losses)
    return total, losses, matches


def new_criterion():
    return ValidMask2FormerCriterion(num_classes=3, bilinear=True)


@torch.no_grad()
def eval_text_sample(model, pixels, target, valid):
    """Raw and fixed-operating-point text metrics on one binary-target sample."""
    native, probs, logits = raw_outputs(model, pixels, valid.shape)
    table = query_table(native, probs, logits, target, valid)
    post = postprocess(native, probs, valid, 3)
    best = int(np.argmax(table["iou"]))
    return dict(
        counts=counts(post[TEXT], target, valid),
        cross_class_on_text=int(((post[0] | post[1]) & target & valid).sum()),
        best_query_iou=float(table["iou"][best]),
        best_query_text_prob=float(table["probs"][best, TEXT]),
        best_query_quality=float(table["quality"][best]),
        max_text_prob=float(table["probs"][:, TEXT].max()),
        argmax_text_queries=int((table["probs"].argmax(-1) == TEXT).sum()),
        post=post,
        native_best=(native[best] > 0).cpu().numpy() & valid,
    )


def summarize(rows):
    total = np.sum([r["counts"] for r in rows], axis=0)
    return dict(
        pixel=metrics(total),
        per_sample_iou_mean=float(np.mean([metrics(r["counts"])["iou"] or 0 for r in rows])),
        best_query_iou_mean=float(np.mean([r["best_query_iou"] for r in rows])),
        best_query_text_prob_mean=float(np.mean([r["best_query_text_prob"] for r in rows])),
        max_text_prob_mean=float(np.mean([r["max_text_prob"] for r in rows])),
        samples_with_text_tp=int(sum(r["counts"][0] > 0 for r in rows)),
        cross_class_on_text_pixels=int(sum(r["cross_class_on_text"] for r in rows)),
        samples=len(rows),
    )


__all__ = [
    "GROUPS",
    "Path",
    "audit_trainable",
    "eval_text_sample",
    "gradient_audit",
    "load_three_class",
    "make_optimizer",
    "new_criterion",
    "set_trainable",
    "step_loss",
    "summarize",
]
