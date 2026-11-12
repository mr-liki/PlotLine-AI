"""Regression cases for the Stage 9G experimental stitching correction."""

import numpy as np

from roadmarking.inference.stitching import stitch
from roadmarking.inference.stitching_supported_v1 import stitch_supported_v1

THRESHOLDS = {"0": 0.79, "1": 0.88}


def test_support_hole_splits_one_identity_without_changing_class():
    observed = np.ones((32, 32), bool)
    observed[:, 12:18] = False
    raw = np.zeros_like(observed)
    raw[8, 5:25] = True
    windows = {"a": (0, 0)}
    for cls in (0, 1):
        clipped = raw & observed
        previous, _ = stitch([clipped], [0.95], [cls], ["a"], windows, observed, THRESHOLDS)
        assert len(previous) == 1
        outputs, audit = stitch_supported_v1(
            [raw],
            [0.95],
            [cls],
            ["a"],
            windows,
            {"a": observed},
            THRESHOLDS,
        )
        assert len(outputs) == 2
        assert all(e["class"] == cls and e["members"] == [0] for e in outputs)
        assert np.array_equal(np.logical_or.reduce([e["mask"] for e in outputs]), clipped)
        assert audit["removed_pixels"] == 6
        assert audit["split_queries"] == 1


def test_other_crop_support_cannot_authorize_pixels_and_overlap_still_merges():
    a = np.zeros((32, 32), bool)
    b = a.copy()
    a[10, 4:16] = True
    b[10, 8:20] = True
    a_support = np.ones_like(a)
    a_support[10, 12:16] = False
    windows = {"a": (0, 0), "b": (0, 0)}
    support = {"a": a_support, "b": np.ones_like(b)}
    out, audit = stitch_supported_v1(
        [a, b], [0.95, 0.91], [0, 0], ["a", "b"], windows, support, THRESHOLDS
    )
    assert audit["removed_pixels"] == 4
    assert len(out) == 1 and out[0]["members"] == [0, 1]
    assert np.array_equal(out[0]["mask"], (a & a_support) | b)


def test_adjacent_same_class_fragments_are_not_joined_by_proximity_alone():
    support = np.ones((32, 32), bool)
    a, b = np.zeros_like(support), np.zeros_like(support)
    a[10, 2:9] = True
    b[10, 11:20] = True
    windows = {"a": (0, 0)}
    old, audit = stitch([a, b], [0.95, 0.95], [0, 0], ["a", "a"], windows, support, THRESHOLDS)
    assert len(old) == 1 and audit["solid_gap_links"] == 1
    out, audit = stitch_supported_v1(
        [a, b], [0.95, 0.95], [0, 0], ["a", "a"], windows, {"a": support}, THRESHOLDS
    )
    assert len(out) == 2 and audit["solid_gap_links"] == 0
