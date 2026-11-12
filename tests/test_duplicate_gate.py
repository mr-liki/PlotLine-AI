"""Duplicate-aware no-object gate."""

import torch

from roadmarking.training.duplicate_gate import DuplicateGatedCriterion
from roadmarking.training.gated_criterion import GatedNoObjectCriterion

H = W = 64
P = 16


def scene():
    valid = torch.ones(H, W, dtype=torch.bool)
    valid[:, 48:] = False  # NoData
    targets = torch.zeros(1, H, W)
    targets[0, 8:12, 8:44] = 1  # solid line reaching the NoData edge
    labels = torch.tensor([0])
    masks = torch.full((4, P, P), -6.0)
    masks[0, 2, 2:11] = 6.0  # q0 matches the line
    masks[1, 2, 4:16] = 5.0  # q1 duplicate on the line, continues into NoData
    masks[2, 12, 9:16] = 6.0  # q2 claims only unobserved cells + 3 valid background cells
    classes = torch.tensor([[3.0, 0, 0], [0, 3.0, 0], [0, 3.0, 0], [0, 0, 3.0]], requires_grad=True)
    return masks, classes, targets, labels, valid


def test_duplicate_on_reviewed_paint_gets_no_object_even_when_touching_nodata():
    masks, classes, targets, labels, valid = scene()
    base, _ = GatedNoObjectCriterion().one_image(masks, classes, targets, labels, valid, True)
    crit = DuplicateGatedCriterion()
    res, (qi, _) = crit.one_image(masks, classes, targets, labels, valid, True)
    assert qi.tolist() == [0]
    assert crit.last_duplicates == [1]
    res["loss_class"].backward()
    assert classes.grad[1].abs().sum() > 0  # duplicate supervised toward no-object
    assert classes.grad[2].abs().sum() == 0  # unobserved-area query still unsupervised
    for k in ("loss_bce", "loss_dice", "loss_unmatched_valid_mask"):
        assert torch.allclose(res[k], base[k])


def test_no_targets_means_no_duplicates():
    masks, classes, targets, labels, valid = scene()
    crit = DuplicateGatedCriterion()
    crit.one_image(masks, classes, targets[:0], labels[:0], valid, True)
    assert crit.last_duplicates in ([], [0])
