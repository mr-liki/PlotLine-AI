"""Explicit opt-in head preparation for a future reviewed three-class experiment.

No checkpoint is opened or mutated here. Existing two-class models retain their
original no-object row. This utility only returns a new head/configuration.
"""

from copy import deepcopy

import torch
from torch import nn

CLASS_NAMES = ("longitudinal_solid", "longitudinal_dash", "road_text")
VISUAL_TO_MODEL = {1: 0, 2: 1, 9: 2}
NO_OBJECT_INDEX = 3


def model_class_from_visual(visual_id):
    """Never let the legacy visual-id-minus-one rule turn arrows into text."""
    if visual_id not in VISUAL_TO_MODEL:
        raise ValueError("Unsupported visual class must remain excluded from this experiment")
    return VISUAL_TO_MODEL[visual_id]


def three_class_config(config):
    result = deepcopy(config)
    result["id2label"] = {str(i): name for i, name in enumerate(CLASS_NAMES)}
    result["label2id"] = {name: i for i, name in enumerate(CLASS_NAMES)}
    result["num_labels"] = len(CLASS_NAMES)
    return result


def make_three_class_head(old_head, *, seed, source_labels, init_std=0.02):
    """Copy only verified solid/dash rows; move no-object 2->3; seed text row 2.

    For COCO initialization, do not call this road-head adapter. Instead initialize
    an entirely new three-class head while transferring compatible non-head weights.
    """
    if tuple(source_labels) != CLASS_NAMES[:2] or old_head.out_features != 3:
        raise ValueError("Expected an explicitly identified two-foreground-class road head")
    if old_head.bias is None or init_std <= 0:
        raise ValueError(
            "A biased classification head and positive initialization std are required"
        )
    # Preserve the caller's RNG as well as its source weights.
    with torch.random.fork_rng(devices=[]):
        new_head = nn.Linear(old_head.in_features, 4, device="cpu", dtype=old_head.weight.dtype)
    new_head = new_head.to(old_head.weight.device)
    generator = torch.Generator(device=old_head.weight.device).manual_seed(seed)
    with torch.no_grad():
        new_head.weight.normal_(0, init_std, generator=generator)
        new_head.bias.zero_()
        new_head.weight[:2].copy_(old_head.weight[:2])
        new_head.bias[:2].copy_(old_head.bias[:2])
        new_head.weight[3].copy_(old_head.weight[2])
        new_head.bias[3].copy_(old_head.bias[2])
    return new_head
