import numpy as np

from roadmarking.inference.calibration import iou_matrix
from roadmarking.inference.prediction_recovery import packed_iou, reconcile_supported


def _proposal(mask, crop, cls=0, score=0.9):
    return {
        "mask": mask.copy(),
        "evidence": mask.copy(),
        "support": np.ones_like(mask),
        "class": cls,
        "score": score,
        "crops": {crop},
        "members": [crop],
    }


def test_packed_native_iou_matches_reference_with_ignored_pixels():
    rng = np.random.default_rng(7)
    a = rng.random((4, 31, 37)) > 0.8
    b = rng.random((3, 31, 37)) > 0.8
    valid = rng.random((31, 37)) > 0.25
    np.testing.assert_allclose(packed_iou(a, b, valid), iou_matrix(a, b, valid))


def test_observed_collinear_solid_union_conserves_pixels_and_separate_dashes():
    a, b = np.zeros((60, 60), bool), np.zeros((60, 60), bool)
    a[29:32, 5:37] = True
    b[29:32, 25:55] = True
    observed = np.ones_like(a)
    image = np.full(a.shape, 40, np.uint8)
    image[a | b] = 220
    first, second = _proposal(a, "a"), _proposal(b, "b", score=0.8)
    first["support"][:, 37:] = False
    second["support"][:, :25] = False
    result, events = reconcile_supported([first, second], observed, image)
    assert len(result) == 1 and events
    assert np.array_equal(result[0]["mask"], a | b)
    result, _ = reconcile_supported([_proposal(a, "a", 1), _proposal(b, "b", 1)], observed, image)
    assert len(result) == 2


def test_no_nodata_bridge_parallel_merge_or_unanchored_island():
    a, b = np.zeros((60, 60), bool), np.zeros((60, 60), bool)
    a[28:33, 5:38] = True
    b[31:36, 25:55] = True
    observed = np.ones_like(a)
    image = np.full(a.shape, 40, np.uint8)
    image[a | b] = 220
    result, _ = reconcile_supported(
        [_proposal(a, "a"), _proposal(b, "b", score=0.8)], observed, image
    )
    assert len(result) == 2
    b[:] = False
    b[28:33, 40:55] = True
    observed[:, 38:40] = False
    result, _ = reconcile_supported(
        [_proposal(a, "a"), _proposal(b, "b", score=0.8)], observed, image
    )
    assert len(result) == 2
    b[:] = False
    b[28:33, 25:55] = True
    b[2:5, 2:8] = True
    observed[:] = True
    result, _ = reconcile_supported(
        [_proposal(a, "a"), _proposal(b, "b", score=0.8)], observed, image
    )
    assert len(result) == 2


def test_cached_ap_matches_reference_instance_metrics():
    from roadmarking.inference.prediction_recovery import cached_class_ap
    from roadmarking.training.pilot import instance_metrics

    rng = np.random.default_rng(3)
    targets = np.zeros((4, 40, 40), bool)
    for k in range(4):
        targets[k, 8 * k : 8 * k + 5, 3:35] = True
    preds = np.concatenate(
        [targets ^ (rng.random(targets.shape) > 0.9), rng.random((3, 40, 40)) > 0.95]
    )
    scores = rng.random(len(preds))
    classes = np.array([0, 1, 0, 1, 1, 0, 1])
    target_classes = np.array([0, 1, 0, 1])
    valid = rng.random((40, 40)) > 0.1
    reference = instance_metrics(preds, scores, classes, targets, target_classes, valid)
    cached = cached_class_ap(packed_iou(preds, targets, valid), scores, classes, target_classes)
    for name in ("longitudinal_solid", "longitudinal_dash"):
        for key in ("AP", "AP50", "AP75"):
            assert np.isclose(reference[name][key], cached[name][key])


def test_geographic_filter_is_two_stage_and_thresholds_inclusive():
    from roadmarking.inference.prediction_recovery import filter_geographic

    config = {"confidence": {"0": 0.5, "1": 0.5}, "mask_quality": 0.8, "nms_iou": 0.3}
    scores = np.array([0.9, 0.5, 0.95, 0.6])
    classes = np.array([0, 0, 0, 1])
    qualities = np.array([0.9, 0.8, 0.79, 0.9])
    overlaps = np.eye(4)
    overlaps[0, 1] = overlaps[1, 0] = 0.5
    selected = filter_geographic(scores, classes, qualities, overlaps, ["a", "b", "a", "b"], config)
    # Inclusive thresholds keep index 1 locally; quality .79 removes index 2; global NMS
    # then removes index 1 as a cross-crop duplicate of index 0.
    assert selected == [0, 3]
