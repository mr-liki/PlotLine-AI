"""Mask2Former criterion for complete reviewed regions with excluded observations.

Only valid native pixels participate in matching and mask losses. Deterministic,
stratified point sampling preserves population weights and narrow paint examples.
Unmatched-query no-object CE is disabled for partially valid images: a query over
excluded observations/unsupported paint has no safe global class label. Matched
queries retain classification supervision. Fully valid images use ordinary
no-object supervision. Unmatched masks receive null-mask BCE only in the valid
domain, justified by region completeness and one-to-one matching. This conservative
classification policy is an engineering-pilot limitation.
"""

import numpy as np
import torch
import torch.nn.functional as functional
from scipy.optimize import linear_sum_assignment
from skimage.morphology import skeletonize
from torch import nn


class ValidMask2FormerCriterion(nn.Module):
    def __init__(
        self,
        num_classes=2,
        max_points=4096,
        no_object_weight=0.1,
        bilinear=False,
        skeleton_recall_weight=0.0,
    ):
        """``bilinear`` samples predicted logits at native pixel centres exactly as inference
        upsamples them (F.interpolate, align_corners=False); the original nearest mapping
        assigns contradictory targets to one output cell on sub-cell-wide lines.
        ``skeleton_recall_weight`` adds 1 - soft recall of each matched target's valid
        skeleton (Kirchhoff et al., ECCV 2024). Defaults reproduce the original criterion.
        """
        super().__init__()
        self.num_classes = num_classes
        self.max_points = max_points
        self.no_object_weight = no_object_weight
        self.bilinear = bilinear
        self.skeleton_recall_weight = skeleton_recall_weight
        self.weights = {
            "loss_class": 2.0,
            "loss_bce": 5.0,
            "loss_dice": 5.0,
            "loss_unmatched_valid_mask": 1.0,
        }
        if skeleton_recall_weight:
            self.weights["loss_skeleton_recall"] = float(skeleton_recall_weight)

    def sample(self, targets, valid):
        """Return native flat indices and weights summing to one on valid pixels."""
        valid = valid.bool().flatten()
        if not valid.any():
            raise ValueError("No valid supervision pixels")
        foreground = targets.bool().any(0).flatten() if len(targets) else torch.zeros_like(valid)
        strata = [torch.where(valid & foreground)[0], torch.where(valid & ~foreground)[0]]
        active = sum(bool(len(s)) for s in strata)
        if self.max_points < active:
            raise ValueError("Point budget must retain every nonempty sampling stratum")
        indices, weights = [], []
        for pool in strata:
            if not len(pool):
                continue
            n = min(len(pool), self.max_points // active)
            positions = torch.linspace(0, len(pool) - 1, n, device=pool.device).long()
            indices.append(pool[positions])
            weights.append(
                torch.full((n,), len(pool) / (n * valid.sum().item()), device=pool.device)
            )
        return torch.cat(indices), torch.cat(weights)

    def logits_at(self, predictions, rows, cols, height, width):
        """Predicted logits at native pixels: nearest (original) or bilinear (inference)."""
        if not self.bilinear:
            py = (rows * predictions.shape[-2] // height).clamp_max(predictions.shape[-2] - 1)
            px = (cols * predictions.shape[-1] // width).clamp_max(predictions.shape[-1] - 1)
            return predictions[:, py, px].float()
        gx = (cols.float() + 0.5) / width * 2 - 1
        gy = (rows.float() + 0.5) / height * 2 - 1
        grid = torch.stack((gx, gy), -1).view(1, 1, -1, 2).to(predictions.dtype)
        out = functional.grid_sample(
            predictions[None].float(),
            grid.float(),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return out[0, :, 0, :]

    def sampled_logits(self, predictions, targets, valid):
        indices, weights = self.sample(targets, valid)
        height, width = valid.shape
        rows, cols = indices // width, indices % width
        if self.bilinear:
            return (
                self.logits_at(predictions, rows, cols, height, width),
                targets.flatten(1)[:, indices].float(),
                weights,
            )
        # Nearest native-to-output mapping avoids mixing excluded native neighbors
        # through bilinear interpolation. The Swin-S output grid remains unchanged.
        py = (rows * predictions.shape[-2] // height).clamp_max(predictions.shape[-2] - 1)
        px = (cols * predictions.shape[-1] // width).clamp_max(predictions.shape[-1] - 1)
        return predictions[:, py, px].float(), targets.flatten(1)[:, indices].float(), weights

    @staticmethod
    def costs(logits, targets, weights):
        probabilities = logits.sigmoid()
        bce = (functional.softplus(logits) * weights).sum(-1)[:, None] - (
            logits * weights
        ) @ targets.T
        numerator = 2 * (probabilities * weights) @ targets.T
        denominator = (probabilities * weights).sum(-1)[:, None] + (targets * weights).sum(-1)[None]
        dice = 1 - (numerator + 1e-6) / (denominator + 1e-6)
        return bce, dice

    def one_image(self, masks, classes, targets, labels, valid, complete):
        if not complete:
            raise ValueError("Region completeness evidence is required, even with validity masking")
        if masks.shape[0] < len(labels):
            raise ValueError("More visible instances than model queries")
        if len(labels) != len(targets) or (
            labels.numel() and (labels.min() < 0 or labels.max() >= self.num_classes)
        ):
            raise ValueError("Invalid foreground class identities")
        if not torch.isfinite(masks).all() or not torch.isfinite(classes).all():
            raise ValueError("Nonfinite model predictions")
        sampled, truth, weights = self.sampled_logits(masks, targets, valid)
        if len(labels) and not (truth.sum(-1) > 0).all():
            raise ValueError("Every target must retain sampled valid foreground pixels")
        bce, dice = self.costs(sampled, truth, weights)
        cost = 5 * bce + 5 * dice - 2 * classes.float().softmax(-1)[:, labels]
        if not torch.isfinite(cost).all():
            raise ValueError("Nonfinite Hungarian matching cost")
        qi, ti = linear_sum_assignment(cost.detach().cpu().numpy())
        qi = torch.as_tensor(qi, device=masks.device, dtype=torch.long)
        ti = torch.as_tensor(ti, device=masks.device, dtype=torch.long)
        # FP16 dense-mask sums can overflow even when every prediction is finite.
        # Empty-target images use this differentiable zero, so accumulate in FP32.
        zero = masks.float().sum() * 0 + classes.float().sum() * 0
        if len(qi):
            mask_bce = bce[qi, ti].mean()
            mask_dice = dice[qi, ti].mean()
        else:
            mask_bce = mask_dice = zero
        unmatched = torch.ones(len(masks), device=masks.device, dtype=torch.bool)
        unmatched[qi] = False
        # Completeness plus one-to-one instance matching establishes that there
        # is no additional supported object assigned to an unmatched query in
        # the valid region. Its mask can therefore be null *only on valid pixels*.
        # No global no-object class is assigned to a potential excluded object.
        null_mask_loss = (
            (functional.softplus(sampled[unmatched]) * weights).sum(-1).mean()
            if unmatched.any()
            else zero
        )
        if bool(valid.all()):
            class_targets = torch.full((len(classes),), self.num_classes, device=classes.device)
            class_targets[qi] = labels[ti]
            class_weights = torch.ones(self.num_classes + 1, device=classes.device)
            class_weights[-1] = self.no_object_weight
            class_loss = functional.cross_entropy(
                classes.float(), class_targets, weight=class_weights
            )
        elif len(qi):
            # In partial images, no global negative class is assigned to an
            # unmatched query that may correspond to excluded paint or NoData.
            class_loss = functional.cross_entropy(classes[qi].float(), labels[ti])
        else:
            class_loss = zero
        result = {
            "loss_class": class_loss,
            "loss_bce": mask_bce,
            "loss_dice": mask_dice,
            "loss_unmatched_valid_mask": null_mask_loss,
        }
        if self.skeleton_recall_weight:
            result["loss_skeleton_recall"] = self.skeleton_recall(
                masks, targets, valid, qi, ti, zero
            )
        return result, (qi, ti)

    def skeleton_recall(self, masks, targets, valid, qi, ti, zero):
        """Mean over matched pairs of 1 - mean predicted probability on the valid skeleton."""
        if not len(qi):
            return zero
        height, width = valid.shape
        valid_np = valid.bool().cpu().numpy()
        losses = []
        for q, t in zip(qi.tolist(), ti.tolist(), strict=True):
            skel = skeletonize(targets[t].bool().cpu().numpy()) & valid_np
            ys, xs = np.nonzero(skel)
            if not len(ys):
                continue
            rows = torch.as_tensor(ys, device=masks.device)
            cols = torch.as_tensor(xs, device=masks.device)
            logits = self.logits_at(masks[q : q + 1], rows, cols, height, width)[0]
            losses.append(1 - logits.sigmoid().mean())
        return torch.stack(losses).mean() if losses else zero

    def forward(self, masks, classes, targets, labels, validity, complete, auxiliary=None):
        losses, matches = {}, []
        if not (len(masks) == len(targets) == len(labels) == len(validity) == len(complete)):
            raise ValueError("Batch/target/validity lengths differ")
        for i in range(len(masks)):
            current, matched = self.one_image(
                masks[i], classes[i], targets[i], labels[i], validity[i], complete[i]
            )
            matches.append(matched)
            for name, value in current.items():
                losses[name] = losses.get(name, 0) + value / len(masks)
        for index, prediction in enumerate(auxiliary or []):
            current, _ = self.forward(
                prediction["masks_queries_logits"],
                prediction["class_queries_logits"],
                targets,
                labels,
                validity,
                complete,
            )
            losses.update({f"{name}_aux{index}": value for name, value in current.items()})
        return losses, matches

    def total(self, losses):
        return sum(
            value * next(weight for name, weight in self.weights.items() if key.startswith(name))
            for key, value in losses.items()
        )
