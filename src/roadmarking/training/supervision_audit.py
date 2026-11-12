"""Executable diagnostics of the installed stock criterion, without model weights."""

import inspect
from pathlib import Path
from unittest.mock import patch

from roadmarking.rasterization.pipeline import sha256


def audit_stock_criterion():
    import torch
    import transformers
    import transformers.models.mask2former.modeling_mask2former as model
    from transformers import Mask2FormerConfig

    config = Mask2FormerConfig(num_labels=2, train_num_points=64)
    criterion = model.Mask2FormerLoss(config, {})
    target = torch.zeros(1, 8, 8)
    target[:, 2:6, 2:4] = 1
    logits = torch.zeros(1, 64, requires_grad=True)
    changed = logits.detach().clone()
    changed[:, 0] = 12  # assumed missing/unknown; stock BCE has no validity mask
    baseline = model.sigmoid_cross_entropy_loss(logits, target.flatten(1), 1)
    altered = model.sigmoid_cross_entropy_loss(changed, target.flatten(1), 1)
    dice_before = model.dice_loss(logits, target.flatten(1), 1)
    dice_after = model.dice_loss(changed, target.flatten(1), 1)
    (baseline + dice_before).backward()
    # The second query could represent an unannotated marking: it is still assigned
    # the no-object class, independently of any spatial mask supplied to the processor.
    classes = torch.tensor([[[8.0, -8.0, -8.0], [8.0, -8.0, -8.0]]], requires_grad=True)
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    label_loss = criterion.loss_labels(classes, [torch.tensor([0])], indices)["loss_cross_entropy"]
    label_loss.backward()
    matcher = model.Mask2FormerHungarianMatcher(
        cost_class=0, cost_mask=1, cost_dice=0, num_points=8
    )
    masks = torch.zeros(1, 2, 8, 8)
    masks[0, 0, 0, 0], masks[0, 1, 0, 0] = -12, 12
    changed_masks = masks.clone()
    changed_masks[0, 0, 0, 0], changed_masks[0, 1, 0, 0] = 12, -12

    def ignored_point(*size, **kwargs):
        return torch.full(size, 0.0625, device=kwargs.get("device"))

    with patch.object(torch, "rand", side_effect=ignored_point):
        match_before = matcher(masks, classes.detach(), [target], [torch.tensor([0])])
        match_after = matcher(changed_masks, classes.detach(), [target], [torch.tensor([0])])
    result = {
        "transformers_version": transformers.__version__,
        "implementation": inspect.getfile(model),
        "implementation_sha256": sha256(Path(inspect.getfile(model))),
        "matcher_signature": str(inspect.signature(model.Mask2FormerHungarianMatcher.forward)),
        "mask_loss_signature": str(inspect.signature(model.Mask2FormerLoss.loss_masks)),
        "criterion_signature": str(inspect.signature(model.Mask2FormerLoss.forward)),
        "bce_before": float(baseline.detach()),
        "bce_after_ignored_logit_change": float(altered),
        "dice_before": float(dice_before.detach()),
        "dice_after_ignored_logit_change": float(dice_after),
        "stock_mask_loss_ignores_unknown_pixels": False,
        "finite_mask_gradients": bool(torch.isfinite(logits.grad).all()),
        "unmatched_query_class_loss": float(label_loss.detach()),
        "unmatched_query_receives_no_object_gradient": bool(classes.grad[0, 1].abs().sum() > 0),
        "matched_query_before_ignored_change": match_before[0][0].tolist(),
        "matched_query_after_ignored_change": match_after[0][0].tolist(),
        "stock_matching_changes_with_ignored_pixel": not torch.equal(
            match_before[0][0], match_after[0][0]
        ),
        "matcher_uses_unrestricted_random_points": "torch.rand"
        in inspect.getsource(model.Mask2FormerHungarianMatcher.forward),
        "auxiliary_losses_recompute_same_matcher_and_losses": "self.forward"
        in inspect.getsource(model.Mask2FormerLoss.forward),
        "custom_criterion_implemented": False,
        "reason": (
            "Pixel masking cannot establish unknown instance completeness or no-object labels"
        ),
        "model_weights_loaded": False,
        "model_forward_executed": False,
    }
    if baseline == altered or dice_before == dice_after:
        raise ValueError("Expected installed stock criterion limitation was not reproduced")
    return result
