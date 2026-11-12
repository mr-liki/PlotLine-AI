"""Duplicate-aware no-object supervision on top of the evidence gate.

``GatedNoObjectCriterion`` skips every unmatched query that claims > 5 % NoData/ignored
pixels, so a duplicate query lying on an already matched, reviewed marking (e.g. a weak dash
query over a solid line that continues through point holes) is never told "no object".
This subclass additionally sends an unmatched query to no-object when at least
``min_paint_fraction`` of its claimed VERIFIED-VALID cells are reviewed paint: the object it
describes is already represented by the matched query, whatever it claims in NoData. Queries
over unreviewed or unobserved areas without reviewed paint stay unsupervised as before.
Matching and all mask losses are unchanged.
"""

import torch
import torch.nn.functional as functional

from roadmarking.training.gated_criterion import GatedNoObjectCriterion


class DuplicateGatedCriterion(GatedNoObjectCriterion):
    def __init__(self, *args, min_paint_fraction=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_paint_fraction = min_paint_fraction
        self.last_duplicates = []

    @torch.no_grad()
    def duplicate_queries(self, masks, targets, valid, qi):
        unmatched = torch.ones(len(masks), dtype=torch.bool, device=masks.device)
        unmatched[qi] = False
        if not len(targets):
            return torch.zeros_like(unmatched)
        hw = masks.shape[-2:]
        valid_cells = functional.adaptive_avg_pool2d(valid.float()[None, None], hw)[0, 0] >= 0.5
        paint = (targets.bool().any(0) & valid.bool()).float()
        paint_cells = functional.adaptive_max_pool2d(paint[None, None], hw)[0, 0] > 0
        claimed = (masks > 0) & valid_cells[None]
        area = claimed.flatten(1).sum(1)
        on_paint = (claimed & paint_cells[None]).flatten(1).sum(1)
        frac = on_paint / area.clamp(min=1)
        return unmatched & (area > 0) & (frac >= self.min_paint_fraction)

    def one_image(self, masks, classes, targets, labels, valid, complete):
        result, (qi, ti) = super().one_image(masks, classes, targets, labels, valid, complete)
        if bool(valid.all()):
            return result, (qi, ti)
        frac, _ = self.claimed_valid_fraction(masks, valid)
        unmatched = torch.ones(len(masks), device=masks.device, dtype=torch.bool)
        unmatched[qi] = False
        gate = unmatched & (frac >= self.min_valid_fraction)
        dup = self.duplicate_queries(masks, targets, valid, qi) & ~gate
        self.last_duplicates.append(int(dup.sum()))
        if not dup.any():
            return result, (qi, ti)
        selected = gate | dup
        selected[qi] = True
        class_targets = torch.full((len(classes),), self.num_classes, device=classes.device)
        class_targets[qi] = labels[ti]
        weights = torch.ones(self.num_classes + 1, device=classes.device)
        weights[-1] = self.no_object_weight
        result["loss_class"] = functional.cross_entropy(
            classes[selected].float(), class_targets[selected], weight=weights
        )
        return result, (qi, ti)


__all__ = ["DuplicateGatedCriterion"]
