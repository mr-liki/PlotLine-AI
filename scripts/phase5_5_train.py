"""Bounded complete-state fine-tuning on all thirteen approved regions."""

import argparse
import json
from pathlib import Path

from roadmarking.dataset.phase55 import check_dataset
from roadmarking.training.continuous import run_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop-after", type=int, default=300)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-loss-scale", type=float)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    context = json.loads((root / "outputs/reports/phase5_5/latest.json").read_text())
    prior = json.loads((root / "outputs/reports/phase5_4/latest.json").read_text())
    selected = Path(prior["selected_checkpoint"])
    settings = json.loads((selected / "postprocessing.json").read_text())
    output = root / "models/mask2former_swin_s" / ("phase5_5_" + context["run_id"])
    result = run_training(
        context["dataset"],
        selected / "model",
        selected / "processor",
        output,
        settings,
        stop_after=args.stop_after,
        max_steps=600,
        resume=args.resume,
        learning_rate=5e-6,
        sampling_cycle=[
            "solid",
            "dash",
            "solid",
            "dash",
            "solid",
            "verified_negative",
            "dash",
            "solid",
            "dash",
            "solid",
            "dash",
            "verified_negative",
        ],
        dataset_check=check_dataset,
        evaluation_interval=100,
        resume_loss_scale=args.resume_loss_scale,
    )
    print(json.dumps(result, indent=2), flush=True)
