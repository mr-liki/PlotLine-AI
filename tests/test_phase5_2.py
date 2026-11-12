"""Controlled supervision and approval tests; fixtures are never training data."""

import json

import numpy as np
import pytest
from affine import Affine
from shapely.geometry import box

from roadmarking.dataset.approved_export import (
    reconcile_approval,
    safe_crop_counts,
    spatial_components,
)
from roadmarking.dataset.labels import BACKGROUND, INSTANCE, NO_DATA, UNKNOWN, LabelFeature
from roadmarking.dataset.verified_masks import processor_targets, reviewed_masks
from roadmarking.training.supervision_audit import audit_stock_criterion

A = Affine(0.05, 0, 0, 0, -0.05, 3.2)


@pytest.mark.parametrize(
    "case",
    [
        "one_object",
        "background",
        "missing_inside",
        "missing_outside",
        "partial",
        "unreviewed",
        "padding",
        "multiple",
        "empty_unknown",
    ],
)
def test_supervision_cases_never_silently_train_on_unknown(case):
    observed = np.ones((64, 64), bool)
    valid = observed.copy()
    objects = (
        []
        if case in {"background", "empty_unknown"}
        else [LabelFeature(1, "SYNTHETIC_ONLY_ONE", 1, "polygon", box(0.4, 1.2, 0.8, 2.8))]
    )
    completeness = None if case in {"unreviewed", "empty_unknown"} else box(0, 0, 3.2, 3.2)
    if case == "missing_inside":
        observed[10, 10] = False
    elif case == "missing_outside":
        observed[0, 0] = False
    elif case == "padding":
        observed[0] = False
        valid[0] = False
    elif case == "partial":
        objects = [LabelFeature(1, "SYNTHETIC_ONLY_PARTIAL", 1, "polygon", box(-1, 1, 1, 2))]
    elif case == "multiple":
        objects.append(LabelFeature(2, "SYNTHETIC_ONLY_TWO", 2, "polygon", box(1.4, 1.2, 1.8, 2.8)))
    labels, categories = reviewed_masks(A, observed.shape, observed, valid, objects, completeness)
    rgb = np.zeros((64, 64, 3), np.uint8)
    unsafe = case in {"missing_inside", "missing_outside", "padding", "unreviewed", "empty_unknown"}
    if unsafe:
        with pytest.raises(ValueError, match="cannot ignore"):
            processor_targets(rgb, labels)
    else:
        encoded = processor_targets(rgb, labels)
        assert encoded["pixel_values"].shape == (1, 3, 64, 64)
        assert encoded["class_labels"][0].tolist() == (
            [0, 1] if case == "multiple" else [] if case == "background" else [0]
        )
    if not completeness:
        assert not (labels.state == BACKGROUND).any()
    assert (labels.state[~observed] == NO_DATA).all()
    assert (labels.instance[~observed] == 0).all()
    if case == "padding":
        assert (categories[0] == 6).all()
    if case == "empty_unknown":
        assert (labels.state == UNKNOWN).all()


def test_crop_audit_checks_every_native_window():
    state = np.full((10, 10), INSTANCE, np.uint8)
    state[0, 0] = NO_DATA
    audit = safe_crop_counts(state, (4, 10, 11))
    assert audit["4"]["eligible_windows"] == 48
    assert audit["10"]["eligible_windows"] == 0
    assert audit["10"]["best_supervised_fraction"] == 0.99
    assert audit["11"]["best_supervised_fraction"] is None


def test_stale_approval_cannot_be_transferred(tmp_path):
    project = tmp_path / "annotation_projects/review.qgs"
    store = tmp_path / "annotations_draft/road_marking_annotations.gpkg"
    project.parent.mkdir()
    store.parent.mkdir()
    project.write_text("changed after approval")
    store.write_text("fixture")
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(
            {
                "record_type": "USER_ATTESTATION_OF_REVIEWED_ANNOTATION_APPEARANCE",
                "project": str(project),
                "artifact_sha256": {str(project): "stale", str(store): "stale"},
            }
        )
    )
    with pytest.raises(ValueError, match="Approved artifact changed"):
        reconcile_approval(approval)


def test_shared_identity_and_guard_define_components():
    samples = [
        {"tile_id": "a", "native_xy_bbox": [0, 0, 10, 10], "instances": [{"global_id": "x"}]},
        {"tile_id": "b", "native_xy_bbox": [5, 0, 15, 10], "instances": [{"global_id": "y"}]},
        {"tile_id": "c", "native_xy_bbox": [100, 0, 110, 10], "instances": [{"global_id": "y"}]},
        {"tile_id": "d", "native_xy_bbox": [200, 0, 210, 10], "instances": []},
    ]
    assert spatial_components(samples) == [["a", "b", "c"], ["d"]]


def test_stock_loss_has_ignored_pixel_and_unlabeled_query_failure_modes():
    audit = audit_stock_criterion()
    assert audit["bce_before"] != audit["bce_after_ignored_logit_change"]
    assert audit["dice_before"] != audit["dice_after_ignored_logit_change"]
    assert audit["unmatched_query_receives_no_object_gradient"]
    assert audit["stock_matching_changes_with_ignored_pixel"]
    assert audit["finite_mask_gradients"]
    assert audit["matcher_uses_unrestricted_random_points"]
    assert audit["auxiliary_losses_recompute_same_matcher_and_losses"]
    assert not audit["model_weights_loaded"]
