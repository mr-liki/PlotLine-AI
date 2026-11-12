import numpy as np

from roadmarking.evaluation.link_audit import audit_connections, output_gaps


def _bar(r0, r1, c0, c1, shape=(120, 200)):
    m = np.zeros(shape, bool)
    m[r0:r1, c0:c1] = True
    return m


def _scene():
    observed = np.ones((120, 200), bool)
    observed[:, 80:110] = False  # LiDAR NoData strip
    valid = observed.copy()
    return observed, valid


def test_line_across_nodata_is_not_accepted():
    observed, valid = _scene()
    target = (_bar(50, 53, 10, 190) & observed)[None]  # one ID, unobserved in the strip
    out = (_bar(50, 53, 10, 190) & valid)[None]
    recs = audit_connections(out, np.array([0]), [0], target, [0], valid, observed)
    assert len(recs) == 1 and recs[0]["identity"] == "single_id"
    assert recs[0]["category"] == "nodata_crossing_unsupported" and not recs[0]["accepted"]
    gap = recs[0]["gaps"][0]
    assert gap["unobserved_fraction"] == 1.0 and gap["painted_fraction"] == 0.0
    assert 28 <= gap["length_px"] <= 32


def test_observed_unpainted_gap_and_supported_line():
    observed = np.ones((120, 200), bool)
    valid = observed.copy()
    target = np.stack([_bar(50, 53, 10, 60), _bar(50, 53, 90, 140)])
    out = (_bar(50, 53, 10, 60) | _bar(50, 53, 90, 140))[None]
    recs = audit_connections(out, np.array([1]), [0], target, [1, 1], valid, observed)
    assert recs[0]["category"] == "separate_dashes"
    assert recs[0]["gaps"][0]["kind"] == "observed_unpainted"
    whole = _bar(50, 53, 10, 140)[None]
    assert audit_connections(whole, np.array([0]), [0], whole, [0], valid, observed) == []


def test_bridge_attribution_single_mask_vs_stitch_merge():
    observed, valid = _scene()
    left, right = _bar(50, 53, 10, 80), _bar(50, 53, 110, 190)
    paint = np.zeros_like(valid)
    merged = output_gaps(left | right, valid, observed, paint, [left, right])
    spanning = output_gaps(left | right, valid, observed, paint, [left | right, left])
    assert merged[0]["bridged_by"] == "stitch_merge"
    assert spanning[0]["bridged_by"] == "single_model_mask"


def test_excluded_but_observed_gap_is_not_nodata():
    observed = np.ones((120, 200), bool)
    valid = observed.copy()
    valid[:, 80:110] = False  # e.g. an unsupported symbol margin, still observed
    out = (_bar(50, 53, 10, 190) & valid)[None]
    recs = audit_connections(out, np.array([0]), [0], out, [0], valid, observed)
    gap = recs[0]["gaps"][0]
    assert gap["kind"] == "observed_unpainted" and gap["excluded_fraction"] == 1.0
    assert recs[0]["category"] == "observed_gap_unpainted"
