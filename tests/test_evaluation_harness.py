import numpy as np
from shapely.geometry import box

from roadmarking.evaluation.harness import (
    ap_from_flags,
    assign_folds,
    block_records,
    buffer_counts,
    greedy_flags,
    paired_block_bootstrap,
    region_groups,
    summarize,
)
from roadmarking.inference.calibration import operating_counts
from roadmarking.inference.prediction_recovery import cached_class_ap, packed_iou


def _scene(seed=1):
    rng = np.random.default_rng(seed)
    targets = np.zeros((4, 64, 64), bool)
    targets[0, 10:13, 2:62] = True  # long solid
    for k, c in enumerate((5, 25, 45)):
        targets[k + 1, 40:43, c : c + 12] = True  # dashes
    preds = np.concatenate(
        [targets ^ (rng.random(targets.shape) > 0.97), rng.random((2, 64, 64)) > 0.98]
    )
    scores = rng.random(len(preds))
    classes = np.array([0, 1, 1, 1, 0, 1])
    target_classes = np.array([0, 1, 1, 1])
    valid = rng.random((64, 64)) > 0.05
    return targets, target_classes, preds, scores, classes, valid


def test_greedy_flags_and_ap_match_reference_implementations():
    targets, tl, preds, scores, classes, valid = _scene()
    iou = packed_iou(preds, targets, valid)
    flags, _ = greedy_flags(iou, scores, classes, tl)
    ops = operating_counts(
        {"iou": iou, "scores": scores, "labels": classes, "target_labels": tl},
        list(range(len(scores))),
    )
    for c in (0, 1):
        assert flags[classes == c].sum() == ops[str(c)]["tp"]
    ref = cached_class_ap(iou, scores, classes, tl)
    for c, name in ((0, "longitudinal_solid"), (1, "longitudinal_dash")):
        sel = classes == c
        assert np.isclose(
            ap_from_flags(scores[sel], flags[sel], int((tl == c).sum())), ref[name]["AP50"]
        )


def test_region_groups_block_leakage_by_distance_and_shared_ids():
    regions = [
        {"box": box(0, 0, 10, 10), "ids": ["a"]},
        {"box": box(2000, 0, 2010, 10), "ids": ["a"]},  # far but shares an ID
        {"box": box(20, 0, 30, 10), "ids": ["b"]},  # near the first
        {"box": box(9000, 0, 9010, 10), "ids": ["c"]},
    ]
    groups = region_groups(regions, guard=500)
    assert sorted(map(sorted, groups)) == [[0, 1, 2], [3]]
    folds = assign_folds(regions, groups, k=2)
    assert sorted(len(f) for f in folds) == [1, 3]
    proximity_only = region_groups(regions, guard=500, link_shared_ids=False)
    assert sorted(map(sorted, proximity_only)) == [[0, 2], [1], [3]]


def test_buffer_counts_full_hit_and_miss():
    gt = np.zeros((40, 40), bool)
    gt[19:22, 2:38] = True
    valid = np.ones_like(gt)
    hit = buffer_counts(gt.copy(), gt, valid, 3)
    assert hit["completeness_hit"].sum() == hit["completeness_all"].sum() > 0
    miss = buffer_counts(np.zeros_like(gt), gt, valid, 3)
    assert miss["completeness_hit"].sum() == 0


def test_summary_reproduces_counts_and_identical_models_have_zero_ci():
    targets, tl, preds, scores, classes, valid = _scene(3)
    comp = {"targets": targets, "target_labels": tl, "valid": valid}
    iou = packed_iou(preds, targets, valid)
    keep = list(range(len(scores)))
    rec, det = block_records(comp, preds, scores, classes, keep, iou, block=16)
    s = summarize(rec.values(), det.values())
    ops = operating_counts(
        {"iou": iou, "scores": scores, "labels": classes, "target_labels": tl}, keep
    )
    assert s["solid_fp"] == ops["0"]["fp"] and s["dash_fp"] == ops["1"]["fp"]
    assert np.isclose(s["dash_f1"], ops["1"]["f1"])
    boot = paired_block_bootstrap(rec, det, rec, det, n=50, seed=0)
    assert all(m["ci_difference"] == [0.0, 0.0] for m in boot["metrics"].values())
