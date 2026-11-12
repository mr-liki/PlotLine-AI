import numpy as np

from roadmarking.evaluation.full_metrics import aggregate, component_metrics, match
from roadmarking.evaluation.link_audit import classify_outputs, physical_lines, relation


def _bar(r0, r1, c0, c1, shape=(200, 200)):
    m = np.zeros(shape, bool)
    m[r0:r1, c0:c1] = True
    return m


def test_relation_continuation_parallel_other():
    a = _bar(100, 103, 10, 90)
    assert relation(a, _bar(100, 103, 100, 180)) == "continuation"  # 10 px gap, collinear
    assert relation(a, _bar(130, 133, 10, 90)) == "parallel"  # side by side
    assert relation(a, _bar(10, 90, 150, 153)) == "other"  # perpendicular


def test_client_split_line_is_not_a_false_link_but_parallel_join_is():
    t = np.stack([_bar(100, 103, 10, 90), _bar(100, 103, 100, 180), _bar(130, 133, 10, 180)])
    tc = np.array([0, 0, 0])
    valid = np.ones((200, 200), bool)
    lines = physical_lines(t, tc)
    assert lines[0] == lines[1] != lines[2]
    joined_split = _bar(100, 103, 10, 180)
    joined_parallel = _bar(100, 103, 20, 80) | _bar(130, 133, 20, 80) | _bar(100, 133, 50, 53)
    masks = np.stack([joined_split, joined_parallel])
    recs, counts = classify_outputs(masks, np.array([0, 0]), [0, 1], t, tc, valid, lines)
    kinds = {r["output"]: r["kind"] for r in recs}
    assert kinds == {0: "same_physical_line", 1: "parallel_lines_joined"}
    assert counts["stage3_rule_false_links"] == 2 and counts["true_false_links"] == 1


def test_dash_join_and_nodata_bridge():
    t = np.stack([_bar(100, 103, 10, 40), _bar(100, 103, 80, 110)])
    tc = np.array([1, 1])
    valid = np.ones((200, 200), bool)
    valid[:, 45:75] = False
    out = _bar(100, 103, 10, 110) & valid
    recs, counts = classify_outputs(out[None], np.array([1]), [0], t, tc, valid)
    assert recs[0]["kind"] == "dashes_joined" and recs[0]["nodata_bridge"]
    assert counts["stage3_rule_false_links"] == 0  # the Stage-3 rule only counted solids


def test_match_separates_duplicates_from_false_positives():
    iou = np.array([[0.9, 0.0], [0.7, 0.0], [0.0, 0.1]])
    taken, flags, dup = match(iou, [0.9, 0.8, 0.7], [0, 0, 0], [0, 1, 2], [0, 0], 0.5)
    assert taken == {0} and flags == {0: True, 1: False, 2: False}
    assert dup == {0: False, 1: True, 2: False}


def test_component_metrics_counts_each_physical_id_once():
    shape = (200, 200)
    t = np.stack([_bar(100, 103, 10, 190), _bar(50, 53, 20, 40)])
    image = np.zeros(shape, np.uint8)
    image[t.any(0)] = 200
    comp = {
        "name": "A",
        "valid": np.ones(shape, bool),
        "targets": t,
        "target_labels": [0, 1],
        "ids": ["s", "d"],
        "image": image,
    }
    masks = np.stack([t[0], t[0], _bar(150, 153, 10, 40)])
    scores, classes = np.array([0.9, 0.8, 0.7]), np.array([0, 0, 1])
    from roadmarking.inference.prediction_recovery import packed_iou

    iou = packed_iou(masks, t, comp["valid"]).reshape(3, 2)
    m = component_metrics(comp, masks, scores, classes, [0, 1, 2], [0, 1, 2], iou)
    twin = {**m, "component": "B"}  # the same physical IDs seen by a second component
    agg = aggregate([m, twin])
    assert agg["all_gt"] == 2  # deduplicated by physical ID
    assert agg["solid_instance_recall50"] == 1.0 and agg["dash_misses50"] == 1
    assert agg["solid_duplicates50"] == 2 and agg["dash_false_positives50"] == 2
    assert agg["solid_fragmented"] == 1
    assert m["ap"]["ap50"]["solid"] == 1.0 and m["ap"]["ap50"]["dash"] == 0.0
