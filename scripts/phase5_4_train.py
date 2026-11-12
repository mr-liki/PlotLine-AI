"""Run one bounded continuous training segment, or resume its complete training state."""

import argparse
import json
from pathlib import Path

from roadmarking.training.continuous import run_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop-after", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--experiment", choices=["balanced", "low_lr"], default="balanced")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    context = json.loads((root / "outputs/reports/phase5_4/latest.json").read_text())
    report = Path(context["report_dir"])
    dataset = json.loads((report / "supervised_dataset_expansion.json").read_text())["dataset"]
    baseline = Path(context["baseline_run"])
    settings = json.loads((report / "postprocessing.json").read_text())["selected"]["config"]
    output = root / "models/mask2former_swin_s" / ("phase5_4_" + context["run_id"])
    if args.experiment == "low_lr":
        output = output.with_name(output.name + "_low_lr")
    if args.resume:
        output = args.resume.resolve().parent.parent
    else:
        attempt = 1
        base = output
        while output.exists():
            attempt += 1
            output = base.with_name(base.name + f"_attempt{attempt}")
    result = run_training(
        dataset,
        baseline / "pilot_checkpoint",
        baseline / "processor",
        output,
        settings,
        args.stop_after,
        200,
        args.resume,
        5e-6 if args.experiment == "low_lr" else 1e-5,
        (
            [
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
            ]
            if args.experiment == "low_lr"
            else None
        ),
    )
    print(json.dumps(result, indent=2), flush=True)
