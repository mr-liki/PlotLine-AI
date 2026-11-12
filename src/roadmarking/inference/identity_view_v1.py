"""Experimental prediction identity views. The frozen inference path remains separate."""

import numpy as np
from scipy.ndimage import label


def fragment_view(accepted, source_masks, observed, valid):
    """Retain accepted paint with query provenance, without asserting physical identity."""
    domain = observed & valid
    fragments, relations = [], []
    for index, entry in enumerate(accepted):
        cls = entry["class"]
        if cls not in (0, 1, 2):
            continue
        clipped = entry["mask"] & domain
        components, count = label(clipped, structure=np.ones((3, 3), bool))
        fragment_ids = []
        for component in range(1, count + 1):
            mask = components == component
            members = [j for j in entry["members"] if (source_masks[j] & mask).any()]
            if not members:
                raise ValueError("Accepted fragment has no originating query")
            fragment_ids.append(len(fragments))
            fragments.append(
                dict(
                    mask=mask,
                    members=members,
                    score=entry["score"],
                    **{"class": cls},
                    physical_id=None,
                    accepted_parent=index,
                    logical_group=(
                        f"text_proposal_{index}"
                        if cls == 2 and entry.get("phrase_evidence")
                        else None
                    ),
                )
            )
        if cls == 0 and len(fragment_ids) > 1:
            relations.append(
                dict(
                    fragments=fragment_ids,
                    parent=index,
                    status="uncertain_continuation",
                )
            )
    return fragments, relations


def no_object_eligibility(claimed_native, observed, reviewed_complete, supported_domain, matched):
    """Executable specification for a future gate. This has no training integration."""
    evidence = observed & reviewed_complete & supported_domain
    return (
        claimed_native.any(axis=(1, 2)) & ~(claimed_native & ~evidence).any(axis=(1, 2)) & ~matched
    )
