"""Mathematical tests for evidence-gated no-object supervision (Stage 9E-B)."""

import pytest
import torch

from roadmarking.training.gated_criterion import GatedNoObjectCriterion
from roadmarking.training.valid_criterion import ValidMask2FormerCriterion

H = W = 64  # native
P = 16  # prediction grid (stride 4)


def scene():
    """One dash target in valid space; columns >= 48 are NoData/ignored."""
    valid = torch.ones(H, W, dtype=torch.bool)
    valid[:, 48:] = False
    targets = torch.zeros(1, H, W)
    targets[0, 8:12, 8:32] = 1
    labels = torch.tensor([1])
    masks = torch.full((4, P, P), -6.0)
    masks[0, 2:3, 2:8] = 6.0  # q0 fits the target -> matched
    masks[1, 10:12, 2:8] = 6.0  # q1 false positive on verified-valid background
    masks[2, 10:12, 12:16] = 6.0  # q2 claims ignored (NoData / unsupported) cells
    # q3 claims nothing
    classes = torch.tensor(
        [[0.0, 3.0, 0.0], [0.0, 3.0, 0.0], [0.0, 3.0, 0.0], [0.0, 3.0, 0.0]], requires_grad=True
    )
    return masks, classes, targets, labels, valid


def run(criterion, masks, classes, targets, labels, valid):
    return criterion.one_image(masks, classes, targets, labels, valid, True)


def test_default_criterion_unchanged_and_mask_terms_identical():
    masks, classes, targets, labels, valid = scene()
    base, base_match = run(ValidMask2FormerCriterion(), masks, classes, targets, labels, valid)
    gated, gated_match = run(GatedNoObjectCriterion(), masks, classes, targets, labels, valid)
    assert [m.tolist() for m in base_match] == [m.tolist() for m in gated_match] == [[0], [0]]
    for key in ("loss_bce", "loss_dice", "loss_unmatched_valid_mask"):
        assert torch.equal(base[key], gated[key])
    assert not torch.equal(base["loss_class"], gated["loss_class"])


def test_gate_selects_only_verified_valid_or_empty_unmatched_queries():
    masks, classes, targets, labels, valid = scene()
    criterion = GatedNoObjectCriterion()
    losses, _ = run(criterion, masks, classes, targets, labels, valid)
    losses["loss_class"].backward()
    grad = classes.grad
    assert grad[1].abs().sum() > 0  # valid-background false positive -> no-object
    assert grad[3].abs().sum() > 0  # empty query -> no-object
    assert torch.equal(grad[2], torch.zeros(3))  # ignored-region query untouched
    assert grad[1, 2] < 0  # descent increases the no-object logit
    assert criterion.last_gate[-1] == dict(
        gated=2, unmatched=3, excluded_uncertain=1, gated_nonempty=1
    )


def test_matched_query_keeps_positive_supervision():
    masks, classes, targets, labels, valid = scene()
    losses, _ = run(GatedNoObjectCriterion(), masks, classes, targets, labels, valid)
    losses["loss_class"].backward()
    assert classes.grad[0, 1] < 0  # descent raises the dash logit for the matched query


def test_no_gradient_from_gate_into_masks():
    masks, classes, targets, labels, valid = scene()
    masks = masks.clone().requires_grad_(True)
    gated, _ = run(GatedNoObjectCriterion(), masks, classes.detach(), targets, labels, valid)
    # With class logits detached, the class loss has no autograd path to the masks.
    assert not gated["loss_class"].requires_grad
    gated["loss_dice"].backward()  # mask terms still train the masks
    assert masks.grad.abs().sum() > 0


def test_more_nodata_elsewhere_does_not_add_penalties():
    masks, classes, targets, labels, valid = scene()
    first, _ = run(GatedNoObjectCriterion(), masks, classes, targets, labels, valid)
    extra = valid.clone()
    extra[56:, :40] = False  # NoData far from every claimed region
    second, _ = run(GatedNoObjectCriterion(), masks, classes, targets, labels, extra)
    assert torch.allclose(first["loss_class"], second["loss_class"])


def test_targets_inside_ignored_pixels_do_not_change_loss():
    masks, classes, targets, labels, valid = scene()
    first, _ = run(GatedNoObjectCriterion(), masks, classes, targets, labels, valid)
    changed = targets.clone()
    changed[0][~valid] = 1 - changed[0][~valid]
    second, _ = run(GatedNoObjectCriterion(), masks, classes, changed, labels, valid)
    for key in first:
        assert torch.equal(first[key], second[key])


def test_fully_valid_images_use_ordinary_no_object_supervision():
    masks, classes, targets, labels, valid = scene()
    full = torch.ones_like(valid)
    base, _ = run(ValidMask2FormerCriterion(), masks, classes, targets, labels, full)
    gated, _ = run(GatedNoObjectCriterion(), masks, classes, targets, labels, full)
    for key in base:
        assert torch.equal(base[key], gated[key])


def test_partial_query_overlap_with_ignored_region_is_excluded():
    masks, classes, targets, labels, valid = scene()
    masks[1, 10:12, 8:14] = 6.0  # q1 now extends into ignored columns >= 48 native
    criterion = GatedNoObjectCriterion(min_valid_fraction=0.95)
    run(criterion, masks, classes, targets, labels, valid)
    assert criterion.last_gate[-1]["gated_nonempty"] == 0


def test_requires_completeness_and_valid_threshold():
    masks, classes, targets, labels, valid = scene()
    with pytest.raises(ValueError):
        GatedNoObjectCriterion().one_image(masks, classes, targets, labels, valid, False)
    with pytest.raises(ValueError):
        GatedNoObjectCriterion(min_valid_fraction=0.2)
