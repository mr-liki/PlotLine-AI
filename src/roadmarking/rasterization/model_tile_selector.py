"""Auditable tile decisions; black or faint paint is never a rejection rule."""

import numpy as np


def select_model_tile(observed, valid, search, evidence, unknown, *, uncertain=None):
    masks = [np.asarray(x, bool) for x in (observed, valid, search, evidence, unknown)]
    if len({x.shape for x in masks}) != 1:
        raise ValueError("Selection masks must align")
    observed, valid, search, evidence, unknown = masks
    if (observed & ~valid).any():
        raise ValueError("Observed data outside valid source footprint")
    if uncertain is not None:
        if np.shape(uncertain) != observed.shape:
            raise ValueError("Uncertainty mask must align")
        unknown = unknown | (observed & np.asarray(uncertain, bool))
    if not observed.any():
        decision, reason = "SKIP", "No observed source pixels; not an intensity decision"
    elif unknown.any():
        decision, reason = "KEEP_FOR_REVIEW", "Unclassified observations retained conservatively"
    else:
        decision, reason = "KEEP", "Observed search/candidate evidence; semantics unverified"
    return {
        "decision": decision,
        "reason": reason,
        "review_status": "UNREVIEWED",
        "observed_ratio": float(observed.mean()),
        "valid_ratio": float(valid.mean()),
        "search_ratio": float(search.mean()),
        "evidence_ratio": float(evidence.mean()),
    }
