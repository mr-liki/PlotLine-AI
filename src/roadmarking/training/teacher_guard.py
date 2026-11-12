"""Guarded knowledge preservation against a frozen teacher (regression-safe sprint).

Two terms, both computed against a detached frozen teacher:

``guarded_teacher_loss``
    Teacher consistency only where ground truth is silent. Queries matched to a target are
    supervised by ground truth alone (no mask or class distillation). Unmatched queries that
    the evidence gate sends to no-object keep their ground-truth class target, so class
    distillation skips them. Mask distillation is evaluated only on verified-valid points
    (the criterion's own stratified sample). On a verified-background point where the student
    is already below the teacher, the point is released: ground truth says no paint there, so
    the student may correct a teacher false positive but is never pulled up toward it.

``nodata_cap_loss``
    One-sided bound on unobserved (NoData) pixels near observed support. It never raises a
    logit and never forces agreement: it only penalises a student mask logit that exceeds a
    cap. Within ``near`` (<= 2 px of support, below the 4 px output-cell resolution) the cap
    is the teacher logit, so the student may not claim more NoData than the frozen model. In
    ``far`` (3-8 px) the cap is ``min(teacher, 0)``, so the student may not claim NoData at
    all at the inference threshold. Caps never go below ``floor`` (logit -2): a query that is
    already well below the threshold is unconstrained. The term is a safety constraint on the
    model's output, not a statement that NoData contains no paint, and NoData is never
    converted into background targets.
"""

import torch
import torch.nn.functional as functional

CAP_FLOOR = -2.0


def stratified_points(pools, max_points):
    """Deterministic evenly spaced points per pool with weights preserving pool populations."""
    pools = [p for p in pools if len(p)]
    if not pools:
        return None, None, None
    total = sum(len(p) for p in pools)
    idx, weights, which = [], [], []
    for k, pool in enumerate(pools):
        n = max(1, min(len(pool), round(max_points * len(pool) / total)))
        positions = torch.linspace(0, len(pool) - 1, n, device=pool.device).long()
        idx.append(pool[positions])
        weights.append(torch.full((n,), len(pool) / (n * total), device=pool.device))
        which.append(torch.full((n,), k, device=pool.device, dtype=torch.long))
    return torch.cat(idx), torch.cat(weights), torch.cat(which)


def guarded_teacher_loss(
    student_masks,
    student_classes,
    teacher_masks,
    teacher_classes,
    criterion,
    targets,
    valid,
    matched_queries,
    gated_queries,
    weights,
):
    """Return (total, diagnostics). ``*_masks``: [Q, h, w] logits; ``*_classes``: [Q, C+1]."""
    q = student_masks.shape[0]
    device = student_masks.device
    unmatched = torch.ones(q, dtype=torch.bool, device=device)
    unmatched[matched_queries] = False
    gated = torch.zeros(q, dtype=torch.bool, device=device)
    if gated_queries is not None:
        gated |= gated_queries.to(device)
    kd_class = unmatched & ~gated
    idx, w = criterion.sample(targets, valid)
    height, width = valid.shape
    rows, cols = idx // width, idx % width
    s = criterion.logits_at(student_masks.float(), rows, cols, height, width)
    with torch.no_grad():
        t = criterion.logits_at(teacher_masks.float(), rows, cols, height, width).sigmoid()
        fg = (
            targets.bool().any(0).flatten()[idx]
            if len(targets)
            else torch.zeros_like(idx, dtype=torch.bool)
        )
        relief = (~fg)[None] & (s.detach().sigmoid() < t)
        keep = (~relief).float() * unmatched[:, None].float()
    bce = functional.binary_cross_entropy_with_logits(s, t, reduction="none")
    zero = s.sum() * 0
    mask_term = (bce * keep * w).sum() / unmatched.sum().clamp(min=1) if unmatched.any() else zero
    if kd_class.any():
        log_s = student_classes[kd_class].float().log_softmax(-1)
        p_t = teacher_classes[kd_class].float().softmax(-1).detach()
        class_term = functional.kl_div(log_s, p_t, reduction="batchmean")
    else:
        class_term = student_classes.float().sum() * 0
    total = weights["mask"] * mask_term + weights["class"] * class_term
    diag = dict(
        teacher_mask=float(mask_term.detach()),
        teacher_class=float(class_term.detach()),
        kd_mask_queries=int(unmatched.sum()),
        kd_class_queries=int(kd_class.sum()),
        relief_fraction=float(relief[unmatched].float().mean()) if unmatched.any() else 0.0,
    )
    return total, diag


def nodata_cap_loss(student_masks, teacher_masks, near, far, criterion, max_points=4096):
    """One-sided NoData cap. ``near``/``far``: native bool masks of unobserved pixels."""
    height, width = near.shape
    pools = [torch.where(near.flatten())[0], torch.where(far.flatten())[0]]
    idx, w, which = stratified_points(pools, max_points)
    zero = student_masks.float().sum() * 0
    if idx is None:
        return zero, dict(nodata_points=0, nodata_violation=0.0, nodata_claimed_fraction=0.0)
    # ``which`` indexes the non-empty pools; map back to near/far identity.
    kinds = [k for k, p in enumerate(pools) if len(p)]
    is_far = torch.as_tensor(kinds, device=idx.device)[which] == 1
    rows, cols = idx // width, idx % width
    s = criterion.logits_at(student_masks.float(), rows, cols, height, width)
    with torch.no_grad():
        t = criterion.logits_at(teacher_masks.float(), rows, cols, height, width)
        cap = torch.where(is_far[None], t.clamp(max=0.0), t).clamp(min=CAP_FLOOR)
    violation = functional.relu(s - cap)
    loss = (violation.sum(0) * w).sum()
    diag = dict(
        nodata_points=len(idx),
        nodata_violation=float(loss.detach()),
        nodata_claimed_fraction=float((s.detach() > 0).any(0).float().mean()),
    )
    return loss, diag


def support_cell_indices(rows, cols, height, width, ph, pw):
    """Output cells (y0, y1, x0, x1) used by exact bilinear sampling at native pixel centres."""
    y = ((rows.double() + 0.5) * ph / height - 0.5).clamp(0, ph - 1)
    x = ((cols.double() + 0.5) * pw / width - 0.5).clamp(0, pw - 1)
    y0, x0 = y.floor().long(), x.floor().long()
    return y0, (y0 + 1).clamp(max=ph - 1), x0, (x0 + 1).clamp(max=pw - 1)


def paint_cells(targets, valid, ph, pw):
    """[T, ph, pw] bool: output cells that contain observed (valid) paint of each target."""
    if not len(targets):
        return torch.zeros((0, ph, pw), dtype=torch.bool, device=valid.device)
    observed = (targets.bool() & valid.bool()[None]).float()[:, None]
    return functional.adaptive_max_pool2d(observed, (ph, pw))[:, 0] > 0


def protected_nodata_cap_loss(
    student_masks,
    teacher_masks,
    nodata,
    targets,
    valid,
    matched_queries,
    matched_targets,
    criterion,
    max_points=4096,
    apply_to=None,
):
    """Correction-sprint cap: unobserved pixels only, with observed paint protected.

    Points are sampled ONLY from ``nodata`` (true unobserved pixels near support); observed
    paint, verified background and observed-but-ignored pixels are never constrained here.
    Cap = clamp(min(teacher, 0), min=-2): no query may claim an unobserved pixel at the
    inference threshold. Exemption: a (matched query, point) pair is unconstrained when the
    point's bilinear support cells contain that query's own observed target paint, because
    lowering those cells would lower the logits rendering ground-truth paint. Unmatched queries
    and matched queries away from their own paint cells are capped.
    """
    height, width = nodata.shape
    ph, pw = student_masks.shape[-2:]
    pool = torch.where(nodata.flatten())[0]
    zero = student_masks.float().sum() * 0
    if not len(pool):
        return zero, dict(nodata_points=0, nodata_violation=0.0, protected_pairs=0)
    idx, w, _ = stratified_points([pool], max_points)
    rows, cols = idx // width, idx % width
    s = criterion.logits_at(student_masks.float(), rows, cols, height, width)
    with torch.no_grad():
        t = criterion.logits_at(teacher_masks.float(), rows, cols, height, width)
        cap = t.clamp(max=0.0).clamp(min=CAP_FLOOR)
        free = torch.zeros_like(s, dtype=torch.bool)
        if len(matched_queries):
            cells = paint_cells(targets[matched_targets], valid, ph, pw)
            y0, y1, x0, x1 = support_cell_indices(rows, cols, height, width, ph, pw)
            touch = cells[:, y0, x0] | cells[:, y0, x1] | cells[:, y1, x0] | cells[:, y1, x1]
            free[matched_queries] = touch
    violation = functional.relu(s - cap) * (~free).float()
    if apply_to is not None:  # restrict the cap to a subset of queries
        violation = violation * apply_to.to(violation.device).float()[:, None]
    loss = (violation.sum(0) * w).sum()
    diag = dict(
        nodata_points=len(idx),
        nodata_violation=float(loss.detach()),
        protected_pairs=int(free.sum()),
        nodata_claimed_fraction=float((s.detach() > 0).any(0).float().mean()),
    )
    return loss, diag


__all__ = [
    "CAP_FLOOR",
    "guarded_teacher_loss",
    "nodata_cap_loss",
    "paint_cells",
    "protected_nodata_cap_loss",
    "stratified_points",
    "support_cell_indices",
]
