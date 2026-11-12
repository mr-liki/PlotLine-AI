"""Equal-budget real-only, ten-percent and twenty-five-percent synthetic ablations."""

import json
from pathlib import Path

from phase5_7_expand_approved import check_expanded
from roadmarking.rasterization.pipeline import dump, sha256
from roadmarking.training.synthetic_continuous import run_training


def readiness(root):
    manifest = json.loads((Path(root) / "metadata/dataset_manifest.json").read_text())
    check_expanded(manifest["parent_real"])
    for crop in manifest["crops"]:
        if sha256(crop["file"]) != crop["sha256"]:
            raise ValueError("Prepared crop changed")
        if crop.get("domain") == "synthetic" and crop["split"] != "train":
            raise ValueError("Synthetic validation/test admitted to training manifest")
    return {"training_ready": True, "synthetic_GSD": None, "protected_tests_used": False}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    context = json.loads((root / "outputs/reports/phase5_8/latest.json").read_text())
    selected = Path(context["initialization"])
    baseline = json.loads((selected / "bundle.json").read_text())["metrics"]
    expected = {k: sum(baseline["raw"][str(i)][k] for i in (0, 1)) for k in ("fp", "fn")}
    settings = baseline["postprocessing"]
    real_cycle = ["solid", "new_region", "dash", "new_region", "verified_negative"]
    experiments = []
    for name, period in (("A_real_only", None), ("B_synthetic10", 10), ("C_synthetic25", 4)):
        cycle = []
        real_index = 0
        for step in range(100):
            if period and (step + 1) % period == 0:
                cycle.append("synthetic")
            else:
                cycle.append(real_cycle[real_index % len(real_cycle)])
                real_index += 1
        output = root / "models/mask2former_swin_s" / ("phase5_8_" + context["run_id"] + "_" + name)
        result = run_training(
            context["dataset"],
            selected / "model",
            selected / "processor",
            output,
            settings,
            max_steps=100,
            stop_after=100,
            learning_rate=1e-6,
            sampling_cycle=cycle,
            dataset_check=readiness,
            evaluation_interval=50,
            initial_loss_scale=16,
            expected_baseline_counts=expected,
        )
        experiments.append(
            {
                "name": name,
                "output": str(output),
                "result": result,
                "synthetic_steps": cycle.count("synthetic"),
                "total_steps": 100,
            }
        )
        dump(Path(context["report_dir"]) / "experiments.json", experiments)
        print(json.dumps({"experiment": name, "result": result}), flush=True)
