"""One fixed 200-step real-geography expansion pilot; no synthetic or test sampling."""

import gc
import json
from pathlib import Path

import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from phase5_7_expand_approved import check_expanded
from roadmarking.rasterization.pipeline import dump
from roadmarking.training.continuous import evaluate
from roadmarking.training.expanded_continuous import run_training

root = Path(__file__).resolve().parents[1]
context_path = root / "outputs/reports/phase5_7_expanded/latest.json"
context = json.loads(context_path.read_text())
report, dataset, selected = (Path(context[k]) for k in ("report_dir", "dataset", "initialization"))
manifest = json.loads((dataset / "metadata/dataset_manifest.json").read_text())
check_expanded(dataset)
old_test = json.loads((root / "outputs/reports/phase5_6_independent/latest.json").read_text())
old_approval = json.loads((Path(old_test["report_dir"]) / "three_region_approval.json").read_text())
protected_ids = set(manifest["protected_test_ids"]) | {
    r["source_id"] for r in old_approval["geometry"]
}
assert not (protected_ids & {uid for c in manifest["crops"] for uid in c["global_instance_ids"]})
train_ids = {
    uid for c in manifest["crops"] if c["split"] == "train" for uid in c["global_instance_ids"]
}
val_ids = {
    uid for c in manifest["crops"] if c["split"] == "val" for uid in c["global_instance_ids"]
}
assert not (train_ids & val_ids)
settings = json.loads(
    (
        root / "outputs/reports/phase5_5/20261106T054327725230Z/selected_postprocessing.json"
    ).read_text()
)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    selected / "model", local_files_only=True
).to("cuda")
processor = Mask2FormerImageProcessor.from_pretrained(selected / "processor", local_files_only=True)
validation = [c for c in manifest["crops"] if c["split"] == "val"]
baseline = evaluate(model, processor, dataset, validation, settings, report / "baseline_preflight")
expected = {k: sum(baseline["raw"][str(i)][k] for i in (0, 1)) for k in ("fp", "fn")}
del model, processor
gc.collect()
torch.cuda.empty_cache()
output = root / "models/mask2former_swin_s" / ("phase5_7_expanded_" + context["run_id"])
context.update(
    training_run=str(output),
    train_unique_ids=len(train_ids),
    validation_unique_ids=len(val_ids),
    validation_crops=len(validation),
    protected_test_used=False,
)
dump(context_path, context)
result = run_training(
    dataset,
    selected / "model",
    selected / "processor",
    output,
    settings,
    max_steps=200,
    stop_after=200,
    learning_rate=1e-6,
    sampling_cycle=["solid", "new_region", "dash", "new_region", "verified_negative"],
    dataset_check=check_expanded,
    evaluation_interval=50,
    initial_loss_scale=16,
    expected_baseline_counts=expected,
)
dump(report / "training_result.json", result)
print(json.dumps(result, indent=2))
