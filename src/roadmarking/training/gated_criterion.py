"""Evidence-gated no-object supervision for partially observed crops (Stage 9E).

``ValidMask2FormerCriterion`` disables unmatched-query no-object CE on every image that
contains any ignored pixel. Real LiDAR crops always contain NoData, so unmatched queries
never learn "no object" and fine-tuning can inflate confident false-positive proposals.

This subclass changes only the classification term on partially valid images. An
unmatched query receives no-object CE only when the region it claims (``logit > 0``,
the inference mask) lies at least ``min_valid_fraction`` inside verified-valid pixels,
i.e. inside reviewed, complete, observed supervision where every visible target is
annotated and already matched. A query claiming nothing is also eligible: it asserts no
object anywhere. Any query claiming more than the tolerance inside NoData, unreviewed
or unsupported-paint (ignored) pixels stays unsupervised, so missing observations and
unsupported paint are never converted into negatives. The presence of NoData elsewhere
in the crop has no effect on a query's eligibility.

Matching, mask BCE/Dice, null-mask loss and fully valid behaviour are inherited
unchanged. The gate is computed without gradient.
"""

import torch
import torch.nn.functional as functional

from roadmarking.training.valid_criterion import ValidMask2FormerCriterion


class GatedNoObjectCriterion(ValidMask2FormerCriterion):
    def __init__(self, *args, min_valid_fraction=0.95, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0.5 <= min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must lie in [0.5, 1]")
        self.min_valid_fraction = min_valid_fraction
        self.last_gate = []

    @torch.no_grad()
    def claimed_valid_fraction(self, masks, valid):
        """Fraction of each query's claimed prediction cells that are verified valid."""
        cells = functional.adaptive_avg_pool2d(valid.float()[None, None], masks.shape[-2:])[0, 0]
        claimed = (masks.float() > 0).float()
        area = claimed.flatten(1).sum(1)
        inside = (claimed * cells).flatten(1).sum(1)
        return torch.where(area > 0, inside / area.clamp(min=1), torch.ones_like(area)), area

    def one_image(self, masks, classes, targets, labels, valid, complete):
        result, (qi, ti) = super().one_image(masks, classes, targets, labels, valid, complete)
        if bool(valid.all()):
            self.last_gate.append(None)
            return result, (qi, ti)
        fraction, area = self.claimed_valid_fraction(masks, valid)
        unmatched = torch.ones(len(masks), device=masks.device, dtype=torch.bool)
        unmatched[qi] = False
        gate = unmatched & (fraction >= self.min_valid_fraction)
        selected = gate.clone()
        selected[qi] = True
        class_targets = torch.full((len(classes),), self.num_classes, device=classes.device)
        class_targets[qi] = labels[ti]
        weights = torch.ones(self.num_classes + 1, device=classes.device)
        weights[-1] = self.no_object_weight
        if selected.any():
            result["loss_class"] = functional.cross_entropy(
                classes[selected].float(), class_targets[selected], weight=weights
            )
        self.last_gate.append(
            dict(
                gated=int(gate.sum()),
                unmatched=int(unmatched.sum()),
                excluded_uncertain=int((unmatched & ~gate).sum()),
                gated_nonempty=int((gate & (area > 0)).sum()),
            )
        )
        return result, (qi, ti)
