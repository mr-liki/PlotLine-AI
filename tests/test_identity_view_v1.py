"""Stage 9H identity and proposed native-grid no-object gate specification tests."""

import numpy as np

from roadmarking.inference.identity_view_v1 import fragment_view, no_object_eligibility


def test_fragment_view_keeps_paint_and_preserves_class_specific_meaning():
    mask = np.zeros((6, 6), bool)
    mask[2, 1:5] = True
    observed = np.ones_like(mask)
    observed[:, 3] = False
    for cls in (0, 1, 2, 3):
        fragments, links = fragment_view(
            [{"mask": mask, "members": [0], "score": 0.9, "class": cls, "phrase_evidence": True}],
            [mask],
            observed,
            observed,
        )
        if cls == 3:
            assert fragments == links == []
            continue
        assert len(fragments) == 2
        assert np.array_equal(fragments[0]["mask"] | fragments[1]["mask"], mask & observed)
        assert all(f["physical_id"] is None and f["members"] == [0] for f in fragments)
        assert all(f["class"] == cls for f in fragments)
        assert bool(links) == (cls == 0)
        if cls == 2:
            assert (
                fragments[0]["logical_group"] == fragments[1]["logical_group"] == "text_proposal_0"
            )
        else:
            assert all(f["logical_group"] is None for f in fragments)


def example():
    domain = np.ones((5, 5), bool)
    claims = np.zeros((4, 5, 5), bool)
    claims[:3, 1, 1:4] = True
    return claims, domain, np.array([True, False, False, False])


def test_empty_and_matched_queries_are_excluded_but_reviewed_duplicates_are_eligible():
    claims, domain, matched = example()
    assert no_object_eligibility(claims, domain, domain, domain, matched).tolist() == [
        False,
        True,
        True,
        False,
    ]


def test_one_native_pixel_outside_each_evidence_mask_excludes_the_query():
    claims, domain, matched = example()
    excluded = domain.copy()
    excluded[1, 3] = False
    for observed, reviewed, supported in (
        (excluded, domain, domain),
        (domain, excluded, domain),
        (domain, domain, excluded),
    ):
        assert not no_object_eligibility(claims, observed, reviewed, supported, matched).any()


def test_auxiliary_claims_and_matches_are_checked_independently():
    claims, domain, matched = example()
    observed = domain.copy()
    observed[0, 0] = False
    final = no_object_eligibility(claims, observed, domain, domain, matched)
    claims[1, 0, 0] = True
    auxiliary = no_object_eligibility(claims, observed, domain, domain, ~matched)
    assert final.tolist() == [False, True, True, False]
    assert auxiliary.tolist() == [True, False, False, False]


def test_ignored_target_values_cannot_enter_native_eligibility():
    claims, domain, matched = example()
    reviewed = domain.copy()
    reviewed[1, 3] = False
    labels = np.zeros_like(domain)
    before = no_object_eligibility(claims, domain, reviewed, domain, matched)
    labels[~reviewed] = True
    np.testing.assert_array_equal(
        before, no_object_eligibility(claims, domain, reviewed, domain, matched)
    )
