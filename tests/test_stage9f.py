"""C5 keeps ignored labels out of supervision and distills only soft mask targets."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from roadmarking.training.gated_criterion import GatedNoObjectCriterion

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from stage9e_lib import teacher_loss


def test_c5_ignored_targets_only_come_from_the_teacher():
    valid = torch.ones(16, 16, dtype=torch.bool)
    valid[:, 12:] = False
    targets = torch.zeros(1, 16, 16)
    targets[:, 2:4, 2:8] = 1
    masks = torch.where(targets.bool(), 4.0, -4.0)
    classes = torch.tensor([[0.0, 3.0, 0.0]])
    labels = torch.tensor([1])
    criterion = GatedNoObjectCriterion()
    changed = targets.clone()
    changed[:, ~valid] = 1 - changed[:, ~valid]
    before, _ = criterion.one_image(masks, classes, targets, labels, valid, True)
    after, _ = criterion.one_image(masks, classes, changed, labels, valid, True)
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)

    def prediction(logits):
        return SimpleNamespace(
            masks_queries_logits=logits[None], class_queries_logits=classes.detach()[None]
        )

    logits = torch.zeros((len(masks), *valid.shape), requires_grad=True)
    teacher_logits = torch.full_like(logits, -1.0, requires_grad=True)
    student, teacher = prediction(logits), prediction(teacher_logits.detach())
    weights = {"mask": 5.0, "class": 2.0}
    valid_only = teacher_loss(student, teacher, criterion, targets, valid, weights)[0]
    loss = teacher_loss(student, teacher, criterion, targets, valid, weights, include_invalid=True)[
        0
    ]
    mutated = teacher_loss(
        student, teacher, criterion, changed, valid, weights, include_invalid=True
    )[0]
    torch.testing.assert_close(loss, mutated, rtol=0, atol=0)
    gradient = torch.autograd.grad(loss - valid_only, logits)[0]
    expected = 5.0 * (0.5 - torch.sigmoid(torch.tensor(-1.0))) / (len(masks) * (~valid).sum())
    torch.testing.assert_close(gradient[:, ~valid], expected.expand_as(gradient[:, ~valid]))
    assert not gradient[:, valid].any()
