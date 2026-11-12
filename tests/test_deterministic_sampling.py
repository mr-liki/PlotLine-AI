"""Exact bilinear gather equals grid_sample sampling and has deterministic gradients."""

import torch

from roadmarking.training.deterministic_sampling import with_exact_bilinear
from roadmarking.training.gated_criterion import GatedNoObjectCriterion
from roadmarking.training.valid_criterion import ValidMask2FormerCriterion


def test_matches_grid_sample_values_and_gradients():
    g = torch.Generator().manual_seed(0)
    pred = torch.randn(5, 16, 16, generator=g, requires_grad=True)
    rows = torch.randint(0, 64, (300,), generator=g)
    cols = torch.randint(0, 64, (300,), generator=g)
    rows[:4] = torch.tensor([0, 63, 0, 63])
    cols[:4] = torch.tensor([0, 0, 63, 63])  # border padding
    reference = ValidMask2FormerCriterion(bilinear=True)
    exact = with_exact_bilinear(ValidMask2FormerCriterion)(bilinear=True)
    a = reference.logits_at(pred, rows, cols, 64, 64)
    (ga,) = torch.autograd.grad(a.sum(), pred)
    b = exact.logits_at(pred, rows, cols, 64, 64)
    (gb,) = torch.autograd.grad(b.sum(), pred)
    assert torch.allclose(a, b, atol=1e-5)
    assert torch.allclose(ga, gb, atol=1e-5)


def test_losses_match_reference_and_nearest_mode_untouched():
    g = torch.Generator().manual_seed(1)
    masks = torch.randn(6, 16, 16, generator=g)
    classes = torch.randn(6, 3, generator=g)
    targets = torch.zeros(1, 64, 64)
    targets[0, 10:14, 5:40] = 1
    valid = torch.ones(64, 64, dtype=torch.bool)
    valid[:, 50:] = False
    labels = torch.tensor([1])
    for base in (ValidMask2FormerCriterion, GatedNoObjectCriterion):
        ref, _ = base(bilinear=True).one_image(masks, classes, targets, labels, valid, True)
        new, _ = with_exact_bilinear(base)(bilinear=True).one_image(
            masks, classes, targets, labels, valid, True
        )
        for k in ref:
            assert torch.allclose(ref[k], new[k], atol=1e-5)
    near = with_exact_bilinear(ValidMask2FormerCriterion)(bilinear=False)
    ref = ValidMask2FormerCriterion(bilinear=False)
    a, _ = near.one_image(masks, classes, targets, labels, valid, True)
    b, _ = ref.one_image(masks, classes, targets, labels, valid, True)
    assert all(torch.equal(a[k], b[k]) for k in a)
