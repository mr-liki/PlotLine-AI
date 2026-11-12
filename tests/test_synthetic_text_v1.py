import json

import numpy as np
import pytest
import torch
from PIL import Image

from roadmarking.training.synthetic_text_v1 import load_synthetic_text_sample
from roadmarking.training.valid_criterion import ValidMask2FormerCriterion


@pytest.mark.parametrize("positive", [True, False])
def test_synthetic_phrase_targets_preserve_exact_geometry_and_validity(tmp_path, positive):
    split = tmp_path / "train"
    for folder in ("images", "masks", "clean_masks", "metadata"):
        (split / folder).mkdir(parents=True)
    image = np.zeros((512, 512), np.uint8)
    visible = np.zeros_like(image)
    if positive:
        visible[10:16, 10:16] = 255
        visible[12:14, 12:14] = 0
        visible[12:14, 16:18] = 255
        visible[21, 11] = 255
    intended = visible.copy()
    if positive:
        intended[20, 11] = 255
    image[visible != 0] = 128
    for folder, raster in zip(
        ("images", "masks", "clean_masks"), (image, visible, intended), strict=True
    ):
        Image.fromarray(raster).save(split / folder / "sample.png")
    (split / "metadata/sample.json").write_text(
        json.dumps(
            {
                "file": "sample.png",
                "split": "train",
                "synthetic": True,
                "text_present": positive,
                "transcription": "AB" if positive else "",
            }
        )
    )
    sample = load_synthetic_text_sample(split, "sample")
    second = load_synthetic_text_sample(split, "sample")
    assert sample["mask_labels"].shape == (int(positive), 512, 512)
    assert sample["class_labels"].tolist() == ([2] if positive else [])
    assert sample["intensity"].shape == (512, 512)
    for key in ("mask_labels", "valid_mask", "intensity"):
        assert torch.equal(sample[key], second[key])
    assert sample["valid_mask"][0, 0] and sample["valid_mask"][12, 12]
    if positive:
        assert np.array_equal(sample["mask_labels"][0].numpy(), visible != 0)
        assert not sample["valid_mask"][20, 11]
        assert sample["intensity"][10, 10].item() == pytest.approx(128 / 255)
        assert sample["metadata"]["phrase_instance_id"] == "train/sample"
    masks = torch.zeros((2, 32, 32), requires_grad=True)
    logits = torch.zeros((2, 4), requires_grad=True)
    criterion = ValidMask2FormerCriterion(num_classes=3, max_points=64)
    losses, _ = criterion.one_image(
        masks, logits, sample["mask_labels"], sample["class_labels"], sample["valid_mask"], True
    )
    loss = sum(losses.values())
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
