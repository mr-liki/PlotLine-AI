import numpy as np

from roadmarking.inference.stitching import stitch

THR = {"0": 0.5, "1": 0.5}


def _canvas():
    return np.zeros((512, 768), bool)


def test_overlapping_solid_fragments_merge_without_new_pixels():
    a, b = _canvas(), _canvas()
    a[250:253, 10:500] = True  # crop A spans cols 0-511
    b[250:253, 300:760] = True  # crop B spans cols 256-767
    windows = {"A": (0, 0), "B": (0, 256)}
    valid = np.ones_like(a)
    out, audit = stitch([a, b], [0.9, 0.6], [0, 0], ["A", "B"], windows, valid, THR)
    assert len(out) == 1 and audit["solid_overlap_merges"] == 1
    assert np.array_equal(out[0]["mask"], a | b) and out[0]["score"] == 0.9


def test_parallel_lines_do_not_merge_and_nodata_gap_is_not_bridged():
    a, b = _canvas(), _canvas()
    a[100:103, 10:200] = True
    b[100:103, 210:400] = True  # 7-px collinear gap
    windows = {"A": (0, 0), "B": (0, 0)}
    valid = np.ones_like(a)
    out, audit = stitch([a, b], [0.9, 0.9], [0, 0], ["A", "A"], windows, valid, THR)
    assert len(out) == 1 and audit["solid_gap_links"] == 1  # observed gap -> linked
    valid[:, 200:210] = False
    out, audit = stitch([a, b], [0.9, 0.9], [0, 0], ["A", "A"], windows, valid, THR)
    assert len(out) == 2 and audit["solid_gap_links"] == 0  # NoData gap -> never bridged
    c = _canvas()
    c[120:123, 10:200] = True  # parallel, 20 px apart
    out, _ = stitch([a, c], [0.9, 0.9], [0, 0], ["A", "A"], windows, np.ones_like(a), THR)
    assert len(out) == 2


def test_dash_duplicates_keep_one_and_thresholds_apply_after_merge():
    a, b = _canvas(), _canvas()
    a[300:303, 300:340] = True
    b[300:303, 301:340] = True
    windows = {"A": (0, 0), "B": (0, 256)}
    out, audit = stitch([a, b], [0.7, 0.8], [1, 1], ["A", "B"], windows, np.ones_like(a), THR)
    assert len(out) == 1 and audit["dash_duplicates_removed"] == 1
    s1, s2 = _canvas(), _canvas()
    s1[250:253, 10:500] = True
    s2[250:253, 300:760] = True
    out, _ = stitch([s1, s2], [0.4, 0.55], [0, 0], ["A", "B"], windows, np.ones_like(a), THR)
    assert len(out) == 1  # low-score fragment survives inside a merged instance
