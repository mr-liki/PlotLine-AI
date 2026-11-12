"""One bounded native-resolution CUDA training attempt; OOM attempts stay separate."""

import argparse
import traceback
from pathlib import Path

from roadmarking.rasterization.pipeline import dump
from roadmarking.training.pilot import run_pilot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()
    try:
        result = run_pilot(
            args.dataset,
            args.initialization,
            args.output,
            args.size,
            args.steps,
            args.freeze_backbone,
        )
        print(result["status"], flush=True)
    except Exception as exc:
        import torch

        dump(
            args.output / "failure.json",
            {
                "type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "cuda_peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
                ),
                "cuda_peak_reserved_bytes": (
                    torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
                ),
            },
        )
        traceback.print_exc()
        raise SystemExit(1) from exc
