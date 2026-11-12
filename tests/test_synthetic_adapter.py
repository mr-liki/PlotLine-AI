import numpy as np
import pytest

from roadmarking.dataset.synthetic_adapter import adapt


def test_grouped_dashes_split_and_class_mapping_is_reversed():
    image = np.full((40, 40), 90, np.uint8)
    ids = np.zeros((40, 40), np.uint16)
    ids[3:6, 2:12] = 1
    ids[3:6, 22:32] = 1
    ids[15:35, 15:18] = 2
    semantic = np.where(ids == 1, 1, np.where(ids == 2, 2, 0)).astype(np.uint8)
    masks, classes, valid, meta = adapt(
        image,
        ids,
        semantic,
        [
            {"instance_id": 1, "class_id": 0, "pixel_count": 60},
            {"instance_id": 2, "class_id": 1, "pixel_count": 60},
        ],
    )
    assert classes.tolist() == [1, 1, 0]
    assert len({m["original_group_id"] for m in meta[:2]}) == 1
    assert not masks[:, ~valid].any()


def test_unsupported_dropout_and_tiny_fragments_are_ignored_not_background():
    image = np.full((40, 40), 90, np.uint8)
    ids = np.zeros((40, 40), np.uint16)
    ids[4:7, 2:12] = 1
    ids[20:28, 20:28] = 2
    ids[0, 0] = 1
    semantic = np.where(ids == 1, 1, np.where(ids == 2, 6, 0)).astype(np.uint8)
    image[35, 35] = 0
    masks, _, valid, _ = adapt(
        image,
        ids,
        semantic,
        [
            {"instance_id": 1, "class_id": 0, "pixel_count": 31},
            {"instance_id": 2, "class_id": 5, "pixel_count": 64},
        ],
    )
    assert not valid[0, 0] and not valid[35, 35] and not valid[20:28, 20:28].any()
    assert not masks[:, ~valid].any()
    with pytest.raises(ValueError):
        adapt(image, ids, np.zeros_like(semantic), [])


def test_close_dropout_fragments_are_not_invented_as_separate_physical_dashes():
    image = np.full((40, 40), 90, np.uint8)
    ids = np.zeros((40, 40), np.uint16)
    ids[5:8, 2:12] = 1
    ids[5:8, 13:23] = 1
    masks, _, valid, _ = adapt(
        image,
        ids,
        (ids > 0).astype(np.uint8),
        [{"instance_id": 1, "class_id": 0, "pixel_count": 60}],
    )
    assert len(masks) == 0 and not valid[ids > 0].any()
