"""Valid-supervision criterion tests; controlled tensors are not training data."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from roadmarking.dataset.complete_regions import check_complete_dataset
from roadmarking.training.pilot import instance_metrics
from roadmarking.training.valid_criterion import ValidMask2FormerCriterion


def example():
    targets = torch.zeros(2, 16, 16)
    targets[0, 2:6, 2:5] = 1
    targets[1, 8:12, 8:11] = 1
    valid = torch.ones(16, 16, dtype=torch.bool)
    valid[0:2] = False  # padding/missing outside paint
    valid[3, 3] = False  # missing inside paint
    valid[12:15, 2:6] = False  # unsupported/unreviewed marking
    masks = torch.randn(1, 4, 16, 16, generator=torch.Generator().manual_seed(895))
    classes = torch.randn(1, 4, 3, generator=torch.Generator().manual_seed(123))
    return masks, classes, targets, valid


def test_mask_match_class_and_aux_losses_are_invariant_to_ignored_values():
    masks, classes, targets, valid = example()
    criterion = ValidMask2FormerCriterion(max_points=1000)
    aux = [{"masks_queries_logits": masks * 0.7, "class_queries_logits": classes * 0.8}]
    first, matches = criterion(
        masks, classes, [targets], [torch.tensor([0, 1])], [valid], [True], aux
    )
    changed = masks.clone()
    changed[:, :, ~valid] = 999
    changed_targets = targets.clone()
    changed_targets[:, ~valid] = 1
    second, second_matches = criterion(
        changed,
        classes,
        [changed_targets],
        [torch.tensor([0, 1])],
        [valid],
        [True],
        [{"masks_queries_logits": changed * 0.7, "class_queries_logits": classes * 0.8}],
    )
    for key in first:
        torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
    assert len(first) == 8
    assert torch.equal(matches[0][0], second_matches[0][0])
    assert torch.equal(matches[0][1], second_matches[0][1])


def test_finite_gradients_and_zero_ignored_pixel_gradients_including_aux():
    masks, classes, targets, valid = example()
    masks.requires_grad_()
    classes.requires_grad_()
    auxiliary = masks.detach().clone().requires_grad_()
    criterion = ValidMask2FormerCriterion()
    losses, matches = criterion(
        masks,
        classes,
        [targets],
        [torch.tensor([0, 1])],
        [valid],
        [True],
        [{"masks_queries_logits": auxiliary, "class_queries_logits": classes}],
    )
    total = criterion.total(losses)
    assert torch.isfinite(total)
    total.backward()
    for logits in (masks, auxiliary):
        assert torch.isfinite(logits.grad).all()
        assert (logits.grad[:, :, ~valid] == 0).all()
        assert logits.grad[:, :, valid].abs().sum() > 0
    unmatched = sorted(set(range(4)) - set(matches[0][0].tolist()))
    assert (classes.grad[:, unmatched] == 0).all()


@pytest.mark.parametrize("partial", [False, True])
def test_empty_regions_and_no_object_classification(partial):
    masks = torch.zeros(1, 3, 8, 8, requires_grad=True)
    classes = torch.zeros(1, 3, 3, requires_grad=True)
    valid = torch.ones(8, 8, dtype=torch.bool)
    if partial:
        valid[0] = False
    criterion = ValidMask2FormerCriterion()
    losses, _ = criterion(
        masks, classes, [torch.empty(0, 8, 8)], [torch.empty(0, dtype=torch.long)], [valid], [True]
    )
    criterion.total(losses).backward()
    assert torch.isfinite(classes.grad).all()
    assert bool(classes.grad.abs().sum() > 0) != partial
    assert (masks.grad[:, :, valid] > 0).all()
    assert (masks.grad[:, :, ~valid] == 0).all()


def test_incomplete_region_cannot_be_bypassed_with_a_validity_mask():
    masks, classes, targets, valid = example()
    with pytest.raises(ValueError, match="completeness"):
        ValidMask2FormerCriterion()(
            masks, classes, [targets], [torch.tensor([0, 1])], [valid], [False]
        )


def test_no_valid_pixels_rejected_and_sampling_preserves_foreground():
    masks, classes, targets, valid = example()
    criterion = ValidMask2FormerCriterion(max_points=32)
    points, weights = criterion.sample(targets, valid)
    assert valid.flatten()[points].all()
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0))
    assert targets.flatten(1)[:, points].sum() > 0
    with pytest.raises(ValueError, match="No valid"):
        criterion(masks, classes, [targets], [torch.tensor([0, 1])], [valid & False], [True])


def test_deterministic_low_resolution_partial_object_targets():
    masks, classes, targets, valid = example()
    criterion = ValidMask2FormerCriterion()
    arguments = (masks[:, :, ::4, ::4], classes, [targets], [torch.tensor([0, 1])], [valid], [True])
    first, matches = criterion(*arguments)
    second, other = criterion(*arguments)
    for key in first:
        torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
    assert torch.equal(matches[0][0], other[0][0])


def test_instance_metrics_perfect_empty_and_duplicate_predictions():
    target = np.zeros((1, 8, 8), bool)
    target[:, 2:5, 2:5] = True
    valid = np.ones((8, 8), bool)
    perfect = instance_metrics(target, [0.9], [0], target, [0], valid)
    assert perfect["longitudinal_solid"]["AP"] == 1
    assert perfect["longitudinal_dash"]["status"] == "NOT_EVALUATED"
    empty = instance_metrics(np.empty((0, 8, 8), bool), [], [], target, [0], valid)
    assert empty["longitudinal_solid"]["missed_instances"] == 1
    duplicate = instance_metrics(
        np.repeat(target, 2, axis=0), [0.9, 0.8], [0, 0], target, [0], valid
    )
    assert duplicate["longitudinal_solid"]["false_positives"] == 1
    assert duplicate["longitudinal_solid"]["precision_at_iou50"] == 0.5


def test_real_complete_export_preserves_approval_and_excludes_unsupported_nodata():
    root = Path(__file__).resolve().parents[1]
    pointer = root / "outputs/reports/phase5_3/dataset_latest.json"
    if not pointer.exists():
        pytest.skip("Real Phase 5.3 export not installed")
    dataset = Path(json.loads(pointer.read_text())["dataset"])
    result = check_complete_dataset(dataset)
    assert result["training_ready"]
    assert result["preserved_outlines"] == 95
    assert result["splits"]["cross_split_shared_instances"] == 0
