import numpy as np
import pytest

from roadmarking.dataset.native_development import plan_native_crops


def test_full_context_uses_native512_without_unnecessary_smaller_crops():
    valid = np.ones((1024, 1024), bool)
    targets = np.zeros((1, 1024, 1024), bool)
    targets[0, 200:210, 100:900] = True
    plans, covered, audit = plan_native_crops(targets, valid)
    assert len(plans) == 9 and {p[2] for p in plans} == {512}
    assert covered.all() and set(audit) == {"512", "384", "256"}


def test_sparse_context_can_use_smaller_native_crop_without_lowering_fraction():
    valid = np.zeros((1024, 1024), bool)
    valid[400:480, 400:480] = True
    targets = np.zeros((1, 1024, 1024), bool)
    targets[0, 420:425, 410:465] = True
    plans, covered, _ = plan_native_crops(targets, valid)
    # 80x80 is below ten percent even in a 256 crop: no manufactured eligibility.
    assert not plans and not covered.any()
    valid[390:490, 390:490] = True
    plans, covered, _ = plan_native_crops(targets, valid)
    assert plans and all(size == 256 for _, _, size in plans)
    assert not covered[~valid].any()
    with pytest.raises(ValueError):
        plan_native_crops(targets[:, :512, :512], valid)
