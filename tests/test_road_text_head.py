import pytest
import torch
from torch import nn

from roadmarking.training.road_text_head import (
    CLASS_NAMES,
    VISUAL_TO_MODEL,
    make_three_class_head,
    model_class_from_visual,
    three_class_config,
)
from roadmarking.training.valid_criterion import ValidMask2FormerCriterion


def test_head_keeps_legacy_semantics_and_does_not_reinterpret_no_object_as_text():
    old = nn.Linear(8, 3)
    weights, bias = old.weight.detach().clone(), old.bias.detach().clone()
    rng = torch.random.get_rng_state()
    new = make_three_class_head(old, seed=895, source_labels=CLASS_NAMES[:2])
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert torch.equal(old.weight, weights) and torch.equal(old.bias, bias)
    assert torch.equal(new.weight[:2], weights[:2])
    assert torch.equal(new.weight[3], weights[2])
    assert torch.equal(new.bias[3], bias[2])
    assert not torch.equal(new.weight[2], weights[2])
    same = make_three_class_head(old, seed=895, source_labels=CLASS_NAMES[:2])
    other = make_three_class_head(old, seed=896, source_labels=CLASS_NAMES[:2])
    assert torch.equal(new.weight, same.weight)
    assert not torch.equal(new.weight[2], other.weight[2])
    with pytest.raises(ValueError, match="explicitly identified"):
        make_three_class_head(old, seed=1, source_labels=("car", "person"))


def test_config_mapping_is_versioned_and_source_is_immutable():
    old = {"id2label": {"0": "longitudinal_solid", "1": "longitudinal_dash"}}
    new = three_class_config(old)
    assert len(old["id2label"]) == 2
    assert new["id2label"]["2"] == "road_text"
    assert new["num_labels"] == 3 and VISUAL_TO_MODEL == {1: 0, 2: 1, 9: 2}
    assert model_class_from_visual(9) == 2
    with pytest.raises(ValueError, match="Unsupported visual class"):
        model_class_from_visual(3)  # visual arrow must never become model road_text
    from transformers import Mask2FormerConfig

    configured = Mask2FormerConfig.from_dict(new)
    assert configured.num_labels == 3 and configured.id2label[2] == "road_text"


def test_three_class_matching_and_losses_ignore_nodata_for_text_instances():
    torch.manual_seed(1)
    valid = torch.ones(16, 16, dtype=torch.bool)
    valid[:, 8:10] = False
    targets = torch.zeros(3, 16, 16)
    targets[0, 2:4, 1:7] = 1
    targets[1, 6:8, 1:7] = 1
    targets[2, 10:14, 2:6] = 1
    masks = torch.randn(5, 16, 16, requires_grad=True)
    classes = torch.randn(5, 4, requires_grad=True)
    labels = torch.tensor([0, 1, 2])
    criterion = ValidMask2FormerCriterion(num_classes=3, max_points=256)
    losses, matches = criterion.one_image(masks, classes, targets, labels, valid, True)
    changed = masks.detach().clone()
    changed[:, ~valid] = 123
    altered, other_matches = criterion.one_image(changed, classes, targets, labels, valid, True)
    assert all(torch.equal(a, b) for a, b in zip(matches, other_matches, strict=True))
    assert all(torch.allclose(losses[k], altered[k]) for k in losses)
    sum(losses.values()).backward()
    assert torch.equal(masks.grad[:, ~valid], torch.zeros_like(masks.grad[:, ~valid]))
    assert torch.isfinite(classes.grad).all()
    with pytest.raises(ValueError, match="foreground class"):
        criterion.one_image(masks, classes, targets, torch.tensor([0, 1, 3]), valid, True)


def test_fully_valid_unmatched_query_uses_new_no_object_row_not_road_text():
    valid = torch.ones(8, 8, dtype=torch.bool)
    target = torch.zeros(1, 8, 8)
    target[0, 2:6, 2:6] = 1
    masks = torch.full((2, 8, 8), -8.0)
    masks[0, 2:6, 2:6] = 8
    classes = torch.zeros(2, 4)
    criterion = ValidMask2FormerCriterion(num_classes=3, max_points=64)
    baseline, matches = criterion.one_image(masks, classes, target, torch.tensor([2]), valid, True)
    assert matches[0].tolist() == [0]
    correct = classes.clone()
    correct[1, 3] = 5
    false_text = classes.clone()
    false_text[1, 2] = 5
    lower, _ = criterion.one_image(masks, correct, target, torch.tensor([2]), valid, True)
    higher, _ = criterion.one_image(masks, false_text, target, torch.tensor([2]), valid, True)
    assert lower["loss_class"] < baseline["loss_class"] < higher["loss_class"]
