"""Label safety of the real-background marking generator."""

import numpy as np

from roadmarking.training.realistic_generator import SEM, GenConfig, Tile, generate


def tile():
    rng = np.random.default_rng(0)
    gray = rng.integers(40, 110, (1024, 1024)).astype(np.uint8)
    holes = rng.random((1024, 1024)) < 0.3
    gray[holes] = 0  # sparse unobserved returns
    gray[:, 900:] = 0  # NoData corridor edge
    valid = gray > 0
    line = np.zeros((1024, 1024), bool)
    line[500:502, 50:850] = True
    gray[line & valid] = 240
    ign = np.zeros_like(valid)
    ign[100:140, 100:300] = True
    valid &= ~ign
    return Tile("T", gray, valid, [(line & valid, 0)], gray[line & valid])


def run(seed, **kw):
    t = tile()
    cfg = GenConfig(markings=(3, 3), p_template=0.0, **kw)
    return t, generate(t, np.random.default_rng(seed), t.paint_pool, (), cfg)


def test_targets_are_observed_valid_and_disjoint():
    for seed in range(12):
        _, out = run(seed)
        g, m, v = out["gray"], out["masks"], out["validity"]
        assert m.shape[1:] == (512, 512)
        assert not (m & ~v[None]).any()
        added = m[
            len(m)
            - sum(a["pieces"] if a["kind"] == "dash" else 1 for a in out["record"]["added"]) :
        ]
        assert not (added & (g == 0)[None]).any()  # never labelled on unobserved pixels
        assert (m.sum(0) <= 1).all()  # instances never overlap


def test_unobserved_pixels_stay_unpainted_and_invalid():
    for seed in range(6):
        t, out = run(seed)
        rec = out["record"]
        oy, ox = rec["window"]
        src = t.gray[oy : oy + 512, ox : ox + 512]
        src = np.rot90(src, rec["rot90"])
        src = np.flip(src, -1) if rec["flip"] else src
        unobs = src == 0
        assert (out["gray"][unobs] == 0).all()
        assert not out["validity"][unobs].any()
        assert (out["semantic"][unobs] == SEM["nodata"]).all()


def test_existing_reviewed_line_and_ignore_preserved():
    for seed in range(6):
        _, out = run(seed)
        assert 0 in out["labels"].tolist()  # reviewed solid kept
        sem = out["semantic"]
        oy, ox = out["record"]["window"]
        if oy < 140 and ox < 300:  # window contains the reviewed ignore region
            assert (sem == SEM["ignore"]).sum() > 0
        assert not out["validity"][sem == SEM["ignore"]].any()


def test_paint_never_darkens_and_markings_are_added():
    added = 0
    for seed in range(10):
        t, out = run(seed)
        rec = out["record"]
        oy, ox = rec["window"]
        src = np.rot90(t.gray[oy : oy + 512, ox : ox + 512], rec["rot90"])
        src = np.flip(src, -1) if rec["flip"] else src
        assert (out["gray"].astype(int) >= src.astype(int)).all()
        added += len(rec["added"])
    assert added >= 10


def test_deterministic():
    _, a = run(3)
    _, b = run(3)
    assert np.array_equal(a["gray"], b["gray"]) and np.array_equal(a["masks"], b["masks"])
