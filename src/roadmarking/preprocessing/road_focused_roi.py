"""Permissive spatial evidence, never an intensity filter or a road classifier."""

import numpy as np
from scipy.ndimage import distance_transform_edt


def road_focused_roi(
    observed,
    search,
    strict,
    uncertain,
    *,
    resolution=0.05,
    margin=4.0,
    structures=None,
    approved=None,
):
    """Keep unknown observations until an attributed boundary review exists.

    Margin is in native coordinate units; no physical unit or CRS is inferred.
    Dilation changes selection evidence only, never intensity pixels.
    """
    arrays = [np.asarray(x, bool) for x in (observed, search, strict, uncertain)]
    if len({x.shape for x in arrays}) != 1 or arrays[0].ndim != 2:
        raise ValueError("Aligned two-dimensional masks required")
    if not np.isfinite(resolution) or resolution <= 0 or not np.isfinite(margin) or margin < 0:
        raise ValueError("Invalid native resolution or margin")
    observed, search, strict, uncertain = arrays
    expanded = strict.copy()
    if strict.any() and margin:
        expanded |= distance_transform_edt(~strict) * resolution <= margin + 1e-12
    evidence = search | expanded | uncertain
    for extra in (structures, approved):
        if extra is not None:
            if np.shape(extra) != observed.shape:
                raise ValueError("Auxiliary mask is not aligned")
            evidence |= np.asarray(extra, bool)
    unknown = observed & ~evidence
    return {"roi": evidence | unknown, "evidence": evidence, "unknown": unknown}
