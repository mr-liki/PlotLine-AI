import numpy as np

from roadmarking.training.augment import augment, dihedral, enforce_min_pixels, intensity


def _crop():
    rgb = np.full((64, 64, 3), 60, np.uint8)
    masks = np.zeros((2, 64, 64), bool)
    masks[0, 30:33, 5:60] = True
    masks[1, 10:12, 10:12] = True  # 4-px dash
    rgb[masks.any(0)] = 220
    valid = np.ones((64, 64), bool)
    valid[:, :3] = False
    return rgb, masks, np.array([0, 1]), valid


def test_dihedral_moves_masks_and_validity_together():
    rgb, masks, _, valid = _crop()
    for k in range(4):
        for flip in (False, True):
            g, m, v = dihedral(rgb[..., 0], masks, valid, k, flip)
            assert (g[m.any(0)] == 220).all() and m.sum() == masks.sum() and v.sum() == valid.sum()


def test_intensity_changes_only_valid_pixels():
    rgb, _, _, valid = _crop()
    g = intensity(rgb[..., 0], valid, np.random.default_rng(0))
    assert (g[~valid] == rgb[..., 0][~valid]).all()


def test_nodata_never_becomes_background_and_small_targets_are_dropped():
    rgb, masks, labels, valid = _crop()
    valid2 = valid.copy()
    valid2[10, 10] = False  # dash falls below 4 px
    m, lab, v = enforce_min_pixels(masks, labels, valid2)
    assert len(m) == 1 and lab.tolist() == [0]
    assert not v[10:12, 10:12].any()  # dropped target's pixels are ignored, not background
    for seed in range(5):
        out_rgb, m, lab, v = augment(rgb, masks, labels, valid, np.random.default_rng(seed))
        assert not m[:, ~v].any() and out_rgb.shape == (64, 64, 3)
        assert (out_rgb[..., 0] == out_rgb[..., 1]).all()


def test_load_augmented_checks_hash_and_keeps_native_size(tmp_path):
    import pytest
    import torch

    from roadmarking.rasterization.pipeline import sha256
    from roadmarking.training.expanded_continuous import load_augmented

    rgb, masks, labels, valid = _crop()
    path = tmp_path / "c.npz"
    np.savez_compressed(path, rgb=rgb, masks=masks, labels=labels, validity=valid)

    class Processor:
        def __call__(self, images, return_tensors):
            return {"pixel_values": torch.zeros((1, 3, *images.shape[:2]))}

    record = {"file": "c.npz", "sha256": sha256(path), "size": 64}
    out = load_augmented(
        tmp_path, record, Processor(), np.random.default_rng(0), augment, device="cpu"
    )
    assert out[0].shape[-2:] == (64, 64) and not (out[1][0].bool() & ~out[3][0][None]).any()
    with pytest.raises(ValueError):
        load_augmented(
            tmp_path,
            {**record, "sha256": "0"},
            Processor(),
            np.random.default_rng(0),
            augment,
            device="cpu",
        )
