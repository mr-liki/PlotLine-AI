import torch
import torch.nn.functional as functional

from roadmarking.training.valid_criterion import ValidMask2FormerCriterion


def _line_targets(size=64):
    targets = torch.zeros((1, size, size))
    targets[0, 30:33, 4:60] = 1  # 3-px line, sub-cell wide on a 1/4 grid
    return targets


def test_bilinear_sampling_matches_inference_upsampling():
    torch.manual_seed(0)
    pred = torch.randn(2, 16, 16)
    crit = ValidMask2FormerCriterion(bilinear=True)
    rows = torch.tensor([0, 5, 17, 63])
    cols = torch.tensor([3, 40, 22, 63])
    sampled = crit.logits_at(pred, rows, cols, 64, 64)
    up = functional.interpolate(pred[None], size=(64, 64), mode="bilinear", align_corners=False)
    assert torch.allclose(sampled, up[0][:, rows, cols], atol=1e-5)


def test_default_criterion_keeps_original_nearest_mapping():
    pred = torch.arange(16 * 16, dtype=torch.float32).view(1, 16, 16)
    crit = ValidMask2FormerCriterion()
    rows, cols = torch.tensor([0, 7, 63]), torch.tensor([3, 9, 60])
    out = crit.logits_at(pred, rows, cols, 64, 64)
    assert torch.equal(out[0], pred[0, rows // 4, cols // 4])
    assert "loss_skeleton_recall" not in crit.weights


def test_skeleton_recall_rewards_line_coverage_and_trains_end_to_end():
    targets = _line_targets()
    labels = torch.tensor([0])
    valid = torch.ones((64, 64), dtype=torch.bool)
    crit = ValidMask2FormerCriterion(bilinear=True, skeleton_recall_weight=0.5)
    good = functional.avg_pool2d(targets[None] * 40 - 10, 4)[0]  # line-shaped logits
    bad = torch.full((1, 16, 16), -10.0)
    masks = torch.cat([good, bad]).requires_grad_(True)
    classes = torch.zeros((2, 3))
    losses, (qi, _) = crit.one_image(masks, classes, targets, labels, valid, True)
    assert "loss_skeleton_recall" in losses and qi.tolist() == [0]
    assert float(losses["loss_skeleton_recall"]) < 0.5
    total = crit.total(losses)
    total.backward()
    assert torch.isfinite(masks.grad).all()
    bad_only, _ = crit.one_image(
        bad.clone().requires_grad_(True), classes[:1], targets, labels, valid, True
    )
    assert float(bad_only["loss_skeleton_recall"]) > 0.9
