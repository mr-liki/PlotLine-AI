"""Guarded teacher consistency and one-sided NoData cap (regression-safe sprint)."""

import torch

from roadmarking.training.deterministic_sampling import with_exact_bilinear
from roadmarking.training.gated_criterion import GatedNoObjectCriterion
from roadmarking.training.teacher_guard import (
    CAP_FLOOR,
    guarded_teacher_loss,
    nodata_cap_loss,
    stratified_points,
)

H = W = 64
P = 16
W8 = {"mask": 5.0, "class": 2.0}


def crit():
    return with_exact_bilinear(GatedNoObjectCriterion)(num_classes=2, bilinear=True)


def scene():
    valid = torch.ones(H, W, dtype=torch.bool)
    valid[:, 48:] = False
    targets = torch.zeros(1, H, W)
    targets[0, 8:12, 8:32] = 1
    teacher = torch.full((3, P, P), -6.0)
    teacher[0, 2:3, 2:8] = 6.0
    teacher[1, 10:12, 2:8] = 6.0  # teacher false positive on verified background
    tc = torch.tensor([[0.0, 3.0, 0.0], [0.0, 3.0, 0.0], [3.0, 0.0, 0.0]])
    return valid, targets, teacher, tc


def test_identical_student_has_zero_teacher_loss():
    valid, targets, teacher, tc = scene()
    s = teacher.clone().requires_grad_(True)
    sc = tc.clone().requires_grad_(True)
    total, diag = guarded_teacher_loss(
        s, sc, teacher, tc, crit(), targets, valid, torch.tensor([0]), None, W8
    )
    total.backward()
    # BCE against soft target has nonzero minimum; its gradient at s == t is zero.
    assert s.grad.abs().max() < 1e-6 and sc.grad.abs().max() < 1e-6
    assert diag["kd_mask_queries"] == 2 and diag["kd_class_queries"] == 2


def test_matched_query_receives_no_distillation():
    valid, targets, teacher, tc = scene()
    s = (teacher + torch.randn_like(teacher)).requires_grad_(True)
    sc = (tc + 1).requires_grad_(True)
    total, _ = guarded_teacher_loss(
        s, sc, teacher, tc, crit(), targets, valid, torch.tensor([0]), None, W8
    )
    total.backward()
    assert s.grad[0].abs().max() == 0 and sc.grad[0].abs().max() == 0
    assert s.grad[2].abs().max() > 0


def test_gated_query_keeps_ground_truth_class_only():
    valid, targets, teacher, tc = scene()
    s = teacher.clone().requires_grad_(True)
    sc = (tc + torch.tensor([0.0, 0.0, 1.0])).requires_grad_(True)
    gated = torch.tensor([False, True, False])
    total, diag = guarded_teacher_loss(
        s, sc, teacher, tc, crit(), targets, valid, torch.tensor([0]), gated, W8
    )
    total.backward()
    assert sc.grad[1].abs().max() == 0 and sc.grad[2].abs().max() > 0
    assert diag["kd_class_queries"] == 1


def test_teacher_false_positive_on_background_is_not_imitated():
    valid, targets, teacher, tc = scene()
    s = teacher.clone()
    s[1] = -6.0  # student removed the teacher's background false positive
    s.requires_grad_(True)
    total, diag = guarded_teacher_loss(
        s, tc.clone(), teacher, tc, crit(), targets, valid, torch.tensor([0]), None, W8
    )
    total.backward()
    assert s.grad[1].abs().max() == 0 and diag["relief_fraction"] > 0


def test_teacher_receives_no_gradient():
    valid, targets, teacher, tc = scene()
    teacher.requires_grad_(True)
    s = torch.zeros(3, P, P, requires_grad=True)
    total, _ = guarded_teacher_loss(
        s, tc.detach().clone(), teacher, tc, crit(), targets, valid, torch.tensor([0]), None, W8
    )
    total.backward()
    assert teacher.grad is None


def bands():
    near = torch.zeros(H, W, dtype=torch.bool)
    far = torch.zeros(H, W, dtype=torch.bool)
    near[:, 48:50] = True
    far[:, 51:56] = True
    return near, far


def test_nodata_cap_is_one_sided_and_never_rewards_claims():
    near, far = bands()
    teacher = torch.full((2, P, P), -6.0)
    teacher[0, :, 12:14] = 3.0  # teacher leaks into near and far NoData
    s = teacher.clone().requires_grad_(True)
    loss, _ = nodata_cap_loss(s, teacher, near, far, crit())
    loss.backward()
    # Equal to the teacher: allowed in near band, penalised in far band (cap 0).
    assert loss > 0 and (s.grad >= 0).all()
    lowered = (teacher - 10).requires_grad_(True)
    loss2, _ = nodata_cap_loss(lowered, teacher, near, far, crit())
    assert loss2 == 0  # a student below every cap is never pushed up


def test_nodata_cap_floor_leaves_negative_queries_free():
    near, far = bands()
    teacher = torch.full((1, P, P), -20.0)
    ok = torch.full((1, P, P), CAP_FLOOR - 1.0)
    bad = torch.full((1, P, P), CAP_FLOOR + 1.0)
    assert nodata_cap_loss(ok, teacher, near, far, crit())[0] == 0
    assert nodata_cap_loss(bad, teacher, near, far, crit())[0] > 0


def test_nodata_cap_empty_bands():
    empty = torch.zeros(H, W, dtype=torch.bool)
    s = torch.zeros(1, P, P, requires_grad=True)
    loss, diag = nodata_cap_loss(s, s.detach(), empty, empty, crit())
    assert loss == 0 and diag["nodata_points"] == 0


def test_stratified_points_weights_sum_to_one():
    pools = [torch.arange(10), torch.arange(100, 1100)]
    _idx, w, which = stratified_points(pools, 64)
    assert abs(float(w.sum()) - 1) < 1e-6 and set(which.tolist()) == {0, 1}


def protected_scene():
    """64x64 native, 16x16 cells. Observed paint ends at column 47; NoData from column 48."""
    from roadmarking.training.teacher_guard import protected_nodata_cap_loss

    valid = torch.zeros(H, W, dtype=torch.bool)
    valid[:, :48] = True
    nodata = torch.zeros(H, W, dtype=torch.bool)
    nodata[:, 48:56] = True
    targets = torch.zeros(1, H, W)
    targets[0, 20:22, 30:48] = 1  # observed line touching the NoData edge
    student = torch.full((2, P, P), -6.0)
    student[0, 5, 7:14] = 4.0  # q0: matched line, continues 2 cells into NoData
    student[1, 12, 10:14] = 4.0  # q1: unmatched false positive leaking into NoData
    return protected_nodata_cap_loss, valid, nodata, targets, student


def test_protected_cap_samples_only_unobserved_pixels_and_spares_own_paint_cells():
    fn, valid, nodata, targets, student = protected_scene()
    s = student.clone().requires_grad_(True)
    loss, diag = fn(
        s, student, nodata, targets, valid, torch.tensor([0]), torch.tensor([0]), crit()
    )
    loss.backward()
    assert loss > 0 and diag["protected_pairs"] > 0
    # The matched query's cells holding observed paint (cell column 11, rows 5) get no push.
    assert s.grad[0, 5, 11] == 0 and s.grad[0, 5, 7:11].abs().sum() == 0
    # Its continuation two cells into NoData (column 13) and the false positive are capped.
    assert s.grad[0, 5, 13] > 0 and s.grad[1, 12, 12:14].sum() > 0
    assert (s.grad >= 0).all()  # one-sided: never raises a logit


def test_protected_cap_ignores_observed_ignored_and_background_pixels():
    fn, valid, nodata, targets, student = protected_scene()
    empty = torch.zeros_like(nodata)
    s = student.clone().requires_grad_(True)
    loss, diag = fn(s, student, empty, targets, valid, torch.tensor([0]), torch.tensor([0]), crit())
    assert loss == 0 and diag["nodata_points"] == 0


def test_protected_cap_unmatched_paint_cells_are_not_protected():
    fn, valid, nodata, targets, student = protected_scene()
    s = student.clone().requires_grad_(True)
    loss, _ = fn(
        s,
        student,
        nodata,
        targets,
        valid,
        torch.tensor([], dtype=torch.long),
        torch.tensor([], dtype=torch.long),
        crit(),
    )
    loss.backward()
    assert s.grad[0, 5, 11] > 0  # without a match, no ground-truth claim protects the query
