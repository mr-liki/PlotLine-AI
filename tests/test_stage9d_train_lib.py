"""Stage 9D: component grouping, explicit freezing, optimizer membership, postprocess."""

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from stage9d_common import counts, metrics, postprocess
from stage9d_train_lib import (
    audit_trainable,
    gradient_audit,
    group_of,
    make_optimizer,
    set_trainable,
)


class FakeMask2Former(nn.Module):
    """Mirrors the Hugging Face Mask2Former parameter prefixes."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.pixel_level_module = nn.Module()
        self.model.pixel_level_module.encoder = nn.Linear(2, 2)
        self.model.pixel_level_module.decoder = nn.Linear(2, 2)
        self.model.transformer_module = nn.Module()
        self.model.transformer_module.queries = nn.Linear(2, 2)
        self.model.transformer_module.decoder = nn.Module()
        self.model.transformer_module.decoder.mask_predictor = nn.Linear(2, 2)
        self.class_predictor = nn.Linear(2, 4)

    def forward(self, x):
        x = self.model.pixel_level_module.decoder(self.model.pixel_level_module.encoder(x))
        x = self.model.transformer_module.decoder.mask_predictor(
            self.model.transformer_module.queries(x)
        )
        return self.class_predictor(x).sum()


def test_group_names_follow_hf_prefixes():
    assert group_of("model.pixel_level_module.encoder.embeddings.x") == "backbone"
    assert group_of("model.pixel_level_module.decoder.mask_projection.weight") == "pixel_decoder"
    assert (
        group_of("model.transformer_module.decoder.mask_predictor.mask_embedder.0.0.weight")
        == "mask_embedder"
    )
    assert group_of("model.transformer_module.decoder.layers.0.self_attn.k_proj.weight") == (
        "query_decoder"
    )
    assert group_of("class_predictor.weight") == "class_predictor"


def test_explicit_freezing_and_optimizer_membership():
    model = FakeMask2Former()
    audit = set_trainable(model, ["query_decoder", "mask_embedder", "class_predictor"])
    assert audit["backbone"]["trainable_params"] == 0
    assert audit["pixel_decoder"]["trainable_params"] == 0
    assert audit["class_predictor"]["trainable_params"] == 12
    lrs = dict(query_decoder=1e-4, mask_embedder=1e-4, class_predictor=1e-3)
    _optimizer, membership = make_optimizer(model, lrs)
    assert membership["optimizer_equals_requires_grad"]
    assert set(membership["groups"]) == set(lrs)
    model(torch.ones(1, 2)).backward()
    grads = gradient_audit(model)
    assert grads["backbone"]["with_grad"] == 0 and grads["pixel_decoder"]["with_grad"] == 0
    assert grads["class_predictor"]["norm"] > 0 and grads["mask_embedder"]["norm"] > 0
    assert all(row["nonfinite"] == 0 for row in grads.values())
    assert audit_trainable(model)["query_decoder"]["trainable_tensors"] == 2


def test_postprocess_requires_text_argmax_and_score():
    native = torch.full((3, 8, 8), -5.0)
    native[0, :4, :4] = 5.0  # text-shaped confident mask
    native[1, 4:, 4:] = 5.0  # mask whose argmax is no-object
    probs = torch.tensor([[0.0, 0.0, 0.9, 0.1], [0.0, 0.0, 0.3, 0.7], [0.0, 0.0, 0.9, 0.1]])
    valid = np.ones((8, 8), bool)
    valid[0, 0] = False
    post = postprocess(native, probs, valid, 3)
    assert post[2][:4, :4].sum() == 15 and not post[2][4:, 4:].any()
    assert not post[2][0, 0]  # never predicted on invalid pixels
    target = np.zeros((8, 8), bool)
    target[:4, :2] = True
    m = metrics(counts(post[2], target, valid))
    assert m["tp"] == 7 and m["fp"] == 8 and m["fn"] == 0
