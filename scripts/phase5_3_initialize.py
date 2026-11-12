"""Initialize the fixed official pretrained Swin-S and a two-class pilot head."""

import json
from pathlib import Path

import torch
import yaml
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from transformers.models.mask2former.modeling_mask2former import Mask2FormerLoss

from roadmarking.rasterization.pipeline import dump, sha256

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    spec = yaml.safe_load((root / "configs/phase5_mask2former.yaml").read_text())["model"]
    output = root / "models/mask2former_swin_s/phase5_3_pretrained_initialization_v2"
    if output.exists():
        raise FileExistsError("Preserve prior initialized weights")
    torch.manual_seed(895)
    torch.set_num_threads(8)
    kwargs = dict(revision=spec["revision"], use_safetensors=True, output_loading_info=True)
    cached = True
    try:
        model, info = Mask2FormerForUniversalSegmentation.from_pretrained(
            spec["checkpoint"], local_files_only=True, cache_dir=str(root / ".hf_cache"), **kwargs
        )
    except OSError:
        cached = False
        model, info = Mask2FormerForUniversalSegmentation.from_pretrained(
            spec["checkpoint"], cache_dir=str(root / ".hf_cache"), **kwargs
        )
    allowed_missing = {
        "model.pixel_level_module.encoder.swin.layernorm.weight",
        "model.pixel_level_module.encoder.swin.layernorm.bias",
    }
    unexpected = set(info.get("unexpected_keys", []))
    if (
        set(info.get("missing_keys", [])) - allowed_missing
        or any(not name.endswith(".relative_position_index") for name in unexpected)
        or info.get("mismatched_keys")
        or info.get("error_msgs")
    ):
        raise ValueError(f"Unexpected base checkpoint loading differences: {info}")
    weight_file = hf_hub_download(
        spec["checkpoint"],
        "model.safetensors",
        revision=spec["revision"],
        cache_dir=str(root / ".hf_cache"),
        local_files_only=True,
    )
    with safe_open(weight_file, framework="pt", device="cpu") as weights:
        for name in unexpected:
            source_name = name.replace("encoder.swin.encoder.", "encoder.encoder.")
            buffer_name = name.replace(
                ".attention.self.relative_position_index",
                ".attention.relative_position_bias.relative_position_index",
            )
            if not torch.equal(
                weights.get_tensor(source_name).flatten(), model.get_buffer(buffer_name).flatten()
            ):
                raise ValueError("Regenerated relative-position indices differ from checkpoint")
    # Transformers' SwinBackbone explicitly ignores its SwinModel final norm on
    # load; its actual feature maps use pretrained hidden_states_norms instead.
    # Verify that changing this unused final norm cannot change those features.
    backbone = model.model.pixel_level_module.encoder.eval()
    norm = backbone.swin.layernorm
    image = torch.zeros(1, 3, 64, 64)
    with torch.no_grad():
        before = backbone(image).feature_maps
        original_weight, original_bias = norm.weight.clone(), norm.bias.clone()
        norm.weight.fill_(3)
        norm.bias.fill_(7)
        after = backbone(image).feature_maps
        for first, second in zip(before, after, strict=True):
            torch.testing.assert_close(first, second, atol=0, rtol=0)
        norm.weight.copy_(original_weight)
        norm.bias.copy_(original_bias)
    processor = Mask2FormerImageProcessor.from_pretrained(
        spec["checkpoint"],
        revision=spec["revision"],
        cache_dir=str(root / ".hf_cache"),
        do_resize=False,
    )
    if list(model.config.backbone_config.depths) != [2, 2, 18, 2]:
        raise ValueError("Checkpoint backbone is not the selected Swin-S")
    original_head = model.class_predictor
    head = torch.nn.Linear(original_head.in_features, 3)
    torch.nn.init.normal_(head.weight, std=model.config.init_std)
    torch.nn.init.zeros_(head.bias)
    with torch.no_grad():
        head.weight[-1].copy_(original_head.weight[-1])
        head.bias[-1].copy_(original_head.bias[-1])
    model.class_predictor = head
    model.config.num_labels = 2
    model.config.id2label = {0: "longitudinal_solid", 1: "longitudinal_dash"}
    model.config.label2id = {v: k for k, v in model.config.id2label.items()}
    model.config.output_auxiliary_logits = True
    # The saved stock criterion buffer must reflect the new class count even
    # though training uses our externally supplied valid-supervision criterion.
    model.criterion = Mask2FormerLoss(model.config, model.weight_dict)
    model.save_pretrained(output, safe_serialization=True)
    processor.save_pretrained(output / "processor")
    record = {
        "checkpoint": spec["checkpoint"],
        "revision": spec["revision"],
        "pretrained_cache_hit": cached,
        "architecture": "Mask2Former Swin-S",
        "backbone_depths": list(model.config.backbone_config.depths),
        "source_loading_info": {k: sorted(v) if isinstance(v, set) else v for k, v in info.items()},
        "compatibility": (
            "Verified regenerated nonpersistent relative indices; final Swin norm is unused "
            "by backbone feature maps (counterfactual forward test passed)"
        ),
        "initialized_layers": ["class_predictor.weight", "class_predictor.bias"],
        "no_object_row": "Preserved from original COCO class head",
        "num_labels": 2,
        "id2label": model.config.id2label,
        "optimizer_steps": 0,
        "is_trained_checkpoint": False,
        "output": str(output),
        "weight_sha256": sha256(output / "model.safetensors"),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "license": (
            "Existing official checkpoint license other; "
            "no commercial deployment/redistribution performed"
        ),
    }
    dump(output / "initialization_provenance.json", record)
    dump(root / "outputs/reports/phase5_3/model_initialization.json", record)
    print(json.dumps(record, indent=2), flush=True)
