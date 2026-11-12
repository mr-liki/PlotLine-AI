"""Check reconciliation invariants independently of neural model quality."""

import numpy as np

from roadmarking.inference.continuity import reconcile


def test_solid_union_preserves_observed_support_and_does_not_extrapolate():
    a, b = np.zeros((20, 20), bool), np.zeros((20, 20), bool)
    a[5:8, 1:12] = True
    b[5:8, 9:19] = True
    valid = np.ones_like(a)
    valid[6, 10] = False
    proposals = [
        {"mask": m, "support": valid.copy(), "label": 0, "score": s}
        for m, s in ((a, 0.9), (b, 0.8))
    ]
    # Full supports require meaningful overlap agreement; use actual intersecting crop supports.
    proposals[0]["support"][:, 12:] = False
    proposals[1]["support"][:, :9] = False
    selected, events = reconcile(proposals, valid, union_solids=True)
    assert len(selected) == 1 and "union_observed_solid_fragments" in events
    assert np.array_equal(selected[0]["mask"], (a | b) & valid)
    assert np.array_equal(a[5:8, 1:12], np.ones((3, 11), bool))


def test_separate_dashes_and_nodata_gap_are_never_connected():
    a, b = np.zeros((20, 20), bool), np.zeros((20, 20), bool)
    a[5:8, :8], b[5:8, 12:] = True, True
    valid = np.ones_like(a)
    valid[:, 8:12] = False
    for cls in (0, 1):
        selected, _ = reconcile(
            [
                {"mask": m, "support": valid, "label": cls, "score": s}
                for m, s in ((a, 0.9), (b, 0.8))
            ],
            valid,
            union_solids=True,
        )
        assert len(selected) == 2
        assert not np.stack([p["mask"] for p in selected])[:, ~valid].any()
