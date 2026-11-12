"""Verify checkpoint migration in an isolated, temporary CPU model."""

import json
import tempfile
from pathlib import Path

import torch
from transformers import Mask2FormerForUniversalSegmentation

from roadmarking.training.road_text_head import CLASS_NAMES, make_three_class_head
from stage9c_audit import BASE, ROOT, dump, sha256

run = Path(json.loads((BASE / "latest.json").read_text())["run"])
preservation = ROOT / "outputs/reports/stage8b/ai_preservation.json"
checks = json.loads(preservation.read_text())["checks"]
assert all(sha256(Path(item["path"])) == item["expected_sha256"] for item in checks)
source = Path(checks[0]["path"]).parent
model, info = Mask2FormerForUniversalSegmentation.from_pretrained(
    source, local_files_only=True, output_loading_info=True
)
assert not any(info[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys"))
old = model.class_predictor
labels = tuple(model.config.id2label[i] for i in range(model.config.num_labels))
assert labels == CLASS_NAMES[:2]
model.class_predictor = make_three_class_head(old, seed=895, source_labels=labels)
model.config.num_labels = 3
model.config.id2label = dict(enumerate(CLASS_NAMES))
model.config.label2id = {value: key for key, value in model.config.id2label.items()}
# Rebuild the installed loss module, including its four-element class weight buffer.
model.criterion = type(model.criterion)(model.config, model.weight_dict)
with tempfile.TemporaryDirectory(dir=run) as temp:
    model.save_pretrained(temp)
    restored, info = Mask2FormerForUniversalSegmentation.from_pretrained(
        temp, local_files_only=True, output_loading_info=True
    )
    assert not any(info[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys"))
    assert all(
        torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items()
    )
    assert tuple(restored.class_predictor.weight.shape) == (4, old.in_features)
    assert torch.equal(restored.class_predictor.weight[3], old.weight[2])
    assert torch.equal(restored.class_predictor.weight[:2], old.weight[:2])
assert all(sha256(Path(item["path"])) == item["expected_sha256"] for item in checks)
dump(
    run / "three_class_checkpoint_migration_audit.json",
    {
        "passed": True,
        "source": str(source),
        "source_sha256": checks[0]["expected_sha256"],
        "legacy_shape": list(old.weight.shape),
        "new_shape": list(restored.class_predictor.weight.shape),
        "no_object_index": 3,
        "reloaded_state_equal": True,
        "temporary_checkpoint_only": True,
    },
)
