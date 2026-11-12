"""Stage 7B: mathematical checks of the thin-line (bilinear) supervision fix.

Synthetic line shapes here are software tests only, never training data.
"""

import pytest
import torch
import torch.nn.functional as functional

from roadmarking.training.valid_criterion import ValidMask2FormerCriterion

H = W = 64


def _grid(height=H, width=W):
    rows, cols = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    return rows.flatten(), cols.flatten()


def _line(width, offset, height=H, size=W):
    target = torch.zeros((1, height, size))
    row = 28 + offset
    target[0, row : row + width, 4 : size - 4] = 1
    return target


def _fit(bilinear, target, steps=150):
    """Best 1/4-resolution logits for a native target under each sampler (valid everywhere)."""
    crit = ValidMask2FormerCriterion(bilinear=bilinear)
    rows, cols = _grid()
    truth = target.flatten()
    logits = torch.zeros((1, H // 4, W // 4), requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.3)
    for _ in range(steps):
        loss = functional.binary_cross_entropy_with_logits(
            crit.logits_at(logits, rows, cols, H, W)[0], truth
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    prob = crit.logits_at(logits.detach(), rows, cols, H, W)[0].sigmoid()
    return float(loss.detach()), prob


@pytest.mark.parametrize("height,width", [(64, 64), (64, 128), (48, 80)])
def test_bilinear_equals_inference_upsampling_on_non_square_grids(height, width):
    torch.manual_seed(1)
    pred = torch.randn(3, height // 4, width // 4)
    crit = ValidMask2FormerCriterion(bilinear=True)
    rows, cols = _grid(height, width)
    sampled = crit.logits_at(pred, rows, cols, height, width)
    up = functional.interpolate(pred[None], (height, width), mode="bilinear", align_corners=False)
    assert torch.allclose(sampled, up[0].flatten(1), atol=1e-5)
    # The align_corners=True convention differs, so the convention is really pinned down.
    other = functional.interpolate(pred[None], (height, width), mode="bilinear", align_corners=True)
    assert not torch.allclose(sampled, other[0].flatten(1), atol=1e-3)


def test_pixel_centre_coordinates_follow_half_pixel_convention():
    # A coarse ramp v(x)=x in output-cell units: native pixel c sits at x=(c+0.5)/4-0.5.
    ramp = torch.arange(16, dtype=torch.float32).repeat(16, 1)[None]
    crit = ValidMask2FormerCriterion(bilinear=True)
    cols = torch.arange(2, 62)  # interior: border clamping applies only at the edges
    out = crit.logits_at(ramp, torch.full_like(cols, 30), cols, H, W)[0]
    assert torch.allclose(out, (cols.float() + 0.5) / 4 - 0.5, atol=1e-5)
    edge = crit.logits_at(ramp, torch.tensor([30, 30]), torch.tensor([0, 63]), H, W)[0]
    assert edge.tolist() == [0.0, 15.0]  # padding_mode="border" (inference replicates edges)


def test_nearest_mapping_assigns_one_logit_per_cell_so_thin_lines_conflict():
    crit = ValidMask2FormerCriterion()
    pred = torch.randn(1, 16, 16)
    rows, cols = _grid()
    out = crit.logits_at(pred, rows, cols, H, W)[0].view(H, W)
    # All 16 native pixels of a cell share one value: a 1-3 px line inside a 4 px cell
    # therefore receives contradictory foreground/background targets on the same logit.
    assert torch.equal(out[28:32, 8:12], out[28, 8].expand(4, 4))


@pytest.mark.parametrize("width", [1, 2, 3, 5])
@pytest.mark.parametrize("offset", [0, 1, 2, 3])
def test_bilinear_fits_thin_lines_better_than_nearest(width, offset):
    """Best achievable fit of 1/4-resolution logits to a native line under each sampler.

    Width >= 2 px (>= 0.10 m): bilinear is strictly better at every sub-cell offset, and
    for 3 px (0.15 m edge lines) the nearest sampler cannot recover the line at all when it
    straddles a cell boundary. 1 px (0.05 m) is below what a 1/4 grid can represent:
    neither sampler recovers it at cell-edge offsets, and bilinear is only marginally
    worse there (documented limit, not a defect of the fix).
    """
    target = _line(width, offset)
    loss_nearest, prob_nearest = _fit(False, target)
    loss_bilinear, prob_bilinear = _fit(True, target)
    on = target.flatten() > 0
    recall_bilinear = float((prob_bilinear[on] > 0.5).float().mean())
    if width == 1:
        assert loss_bilinear <= loss_nearest + 0.005
        assert float((prob_nearest[on] > 0.5).float().mean()) == 0.0
        return
    assert loss_bilinear < loss_nearest
    assert float(prob_bilinear[on].mean()) > float(prob_nearest[on].mean())
    assert float((prob_bilinear[~on] > 0.5).float().mean()) < 0.01  # no smearing into FP
    if width >= 3:
        assert recall_bilinear >= 0.95


def test_nearest_cannot_recover_a_3px_line_straddling_cell_boundary():
    target = _line(3, 2)  # rows 30..32 straddle the 28-31 / 32-35 output cells
    on = target.flatten() > 0
    _, nearest = _fit(False, target)
    _, bilinear = _fit(True, target)
    assert float((nearest[on] > 0.5).float().mean()) == 0.0
    assert float((bilinear[on] > 0.5).float().mean()) >= 0.95


def _image(target, valid, bilinear=True, seed=0, scale=1.0):
    torch.manual_seed(seed)
    masks = (torch.randn(3, 16, 16) * scale).requires_grad_(True)
    classes = torch.randn(3, 3)
    crit = ValidMask2FormerCriterion(bilinear=bilinear)
    losses, _ = crit.one_image(masks, classes, target, torch.tensor([0]), valid, True)
    return crit, masks, losses


@pytest.mark.parametrize("bilinear", [False, True])
def test_nodata_pixels_never_sampled_and_never_change_the_loss(bilinear):
    target = _line(3, 1)
    valid = torch.ones((H, W), dtype=torch.bool)
    valid[:, 40:] = False
    crit = ValidMask2FormerCriterion(bilinear=bilinear)
    indices, weights = crit.sample(target, valid)
    assert valid.flatten()[indices].all() and torch.isclose(weights.sum(), torch.tensor(1.0))
    _, _, base = _image(target, valid, bilinear)
    altered = target.clone()
    altered[0, :, 40:] = 1 - altered[0, :, 40:]  # arbitrary content under NoData
    _, _, other = _image(altered, valid, bilinear)
    for key in base:
        assert torch.allclose(base[key], other[key]), key


def test_gradient_flows_only_to_cells_supporting_valid_pixels():
    target = _line(2, 1)
    valid = torch.zeros((H, W), dtype=torch.bool)
    valid[:, :24] = True  # coarse columns 0..5 support it; bilinear reaches column 6
    crit, masks, losses = _image(target, valid, bilinear=True)
    crit.total(losses).backward()
    grad = masks.grad.abs().sum((0, 1))
    assert torch.isfinite(masks.grad).all()
    assert (grad[:6] > 0).all()
    assert (grad[7:] == 0).all()


@pytest.mark.parametrize("bilinear", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_losses_and_gradients_finite_for_extreme_logits(bilinear, dtype):
    target = _line(1, 0)
    valid = torch.ones((H, W), dtype=torch.bool)
    valid[:8] = False
    masks = torch.full((3, 16, 16), 6.0e4 if dtype == torch.float16 else 1.0e4, dtype=dtype)
    masks[1] = -masks[1]
    masks.requires_grad_(True)
    crit = ValidMask2FormerCriterion(bilinear=bilinear, skeleton_recall_weight=0.5)
    losses, _ = crit.one_image(
        masks, torch.zeros(3, 3, dtype=dtype), target, torch.tensor([0]), valid, True
    )
    total = crit.total(losses)
    total.backward()
    assert torch.isfinite(total) and torch.isfinite(masks.grad.float()).all()


@pytest.mark.parametrize("bilinear", [False, True])
def test_auxiliary_losses_use_identical_supervision(bilinear):
    target = _line(3, 2)
    valid = torch.ones((H, W), dtype=torch.bool)
    valid[50:] = False
    torch.manual_seed(3)
    masks = torch.randn(1, 4, 16, 16)
    classes = torch.randn(1, 4, 3)
    crit = ValidMask2FormerCriterion(bilinear=bilinear, skeleton_recall_weight=0.25)
    aux = [{"masks_queries_logits": masks.clone(), "class_queries_logits": classes.clone()}] * 2
    losses, _ = crit(masks, classes, [target], [torch.tensor([0])], [valid], [True], aux)
    main = {k: v for k, v in losses.items() if "_aux" not in k}
    assert "loss_skeleton_recall" in main
    for index in range(2):
        for key, value in main.items():
            assert torch.allclose(losses[f"{key}_aux{index}"], value), (key, index)
    # Every auxiliary term is weighted exactly like its main term.
    assert torch.isclose(crit.total(losses), 3 * crit.total(main))
