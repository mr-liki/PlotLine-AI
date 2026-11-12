"""Calibration, spatial mask identity and exact resumable-state tests."""

import random

import numpy as np
import pytest
import torch

from roadmarking.inference.calibration import filter_instances, iou_matrix, native_world_iou
from roadmarking.training.checkpoint_manager import load_training_state, save_training_state
from roadmarking.training.valid_criterion import ValidMask2FormerCriterion


def test_mask_nms_suppresses_duplicates_without_joining_separate_dashes():
    masks = np.zeros((3, 12, 12), bool)
    masks[0, 2:4, 2:5] = True
    masks[1] = masks[0]
    masks[2, 7:9, 2:5] = True
    overlap = iou_matrix(masks, masks, np.ones((12, 12), bool))
    config = {"confidence": {"0": 0.25, "1": 0.25}, "mask_quality": 0.5, "nms_iou": 0.5}
    retained = filter_instances(
        np.array([0.9, 0.8, 0.7]), np.array([1, 1, 1]), overlap, np.array([0.9, 0.9, 0.9]), config
    )
    assert retained == [0, 2]
    assert not (masks[0] & masks[2]).any()
    assert not masks[:, 4:7].any()


def test_world_iou_accounts_for_crop_translation():
    a = np.zeros((8, 8), bool)
    b = np.zeros((8, 8), bool)
    a[2:4, 5:7] = True
    b[2:4, 1:3] = True
    first = {"affine": [0.05, 0, 0, 0, -0.05, 1]}
    second = {"affine": [0.05, 0, 0.2, 0, -0.05, 1]}
    assert native_world_iou(a, first, b, second) == 1


def test_large_fp16_empty_target_has_finite_zero_and_null_mask_gradients():
    masks = torch.ones(1, 4, 128, 128, dtype=torch.float16, requires_grad=True)
    classes = torch.ones(1, 4, 3, dtype=torch.float16, requires_grad=True)
    valid = torch.ones(512, 512, dtype=torch.bool)
    valid[0] = False
    criterion = ValidMask2FormerCriterion()
    losses, _ = criterion(
        masks,
        classes,
        [torch.empty(0, 512, 512)],
        [torch.empty(0, dtype=torch.long)],
        [valid],
        [True],
    )
    total = criterion.total(losses)
    assert torch.isfinite(total)
    total.backward()
    assert torch.isfinite(masks.grad).all()


def test_training_checkpoint_restores_optimizer_scheduler_scaler_and_rng(tmp_path):
    random.seed(33)
    np.random.seed(34)
    torch.manual_seed(35)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    path = tmp_path / "state.pt"
    save_training_state(path, optimizer, scheduler, scaler, 1, {"max_steps": 10}, "DATASET_A")
    expected = (random.random(), np.random.random(), torch.rand(3))
    new_optimizer = torch.optim.AdamW(model.parameters(), lr=0.3)
    new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(new_optimizer, T_max=10)
    state = load_training_state(path, new_optimizer, new_scheduler, scaler, "DATASET_A")
    assert state["step"] == 1
    assert len(new_optimizer.state) == len(optimizer.state)
    assert new_scheduler.last_epoch == scheduler.last_epoch
    assert new_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(3), expected[2], atol=0, rtol=0)
    with pytest.raises(ValueError, match="dataset"):
        load_training_state(path, new_optimizer, new_scheduler, scaler, "DATASET_B")
    with path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="integrity"):
        load_training_state(path, new_optimizer, new_scheduler, scaler, "DATASET_A")
