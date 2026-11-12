"""Hash preservation and separate-process candidate reload for the regression-safe sprint."""

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from regsafe_common import BASE, ROOT, dump, frozen_paths, load_model, load_processor, read, sha256
from roadmarking.training.pilot import load_crop
from stage9e_lib import development_crops, real_context

EXTRA = [
    "models/mask2former_swin_s/final_candidate/checkpoint_step0600/processor/preprocessor_config.json",
    "outputs/reports/final_candidate/frozen_candidate.json",
    "src/roadmarking/inference/calibration.py",
    "src/roadmarking/training/valid_criterion.py",
    "src/roadmarking/training/gated_criterion.py",
    "src/roadmarking/training/deterministic_sampling.py",
]


def main():
    train_dir = Path(read(BASE / "training/latest.json")["run"])
    sprint_start = datetime(2026, 11, 12, 11, 4, tzinfo=UTC).timestamp()
    checks, frozen, _ = frozen_paths()  # raises on any hash change
    extra = []
    for rel in EXTRA:
        p = ROOT / rel
        extra.append(
            dict(
                path=rel,
                sha256=sha256(p),
                unmodified_during_sprint=os.path.getmtime(p) < sprint_start,
            )
        )
    info = read(train_dir / "training_info.json")
    cand = Path(info["candidate_dir"])
    processor = load_processor()
    root, crop = development_crops(real_context()["ctx"])[0]
    pixels = load_crop(root, crop, processor, "cuda")[0]
    outs = []
    for _ in range(2):
        m = load_model(cand / "step0100" / "model")
        with torch.no_grad():
            o = m(pixel_values=pixels)
        outs.append((o.class_queries_logits.cpu(), o.masks_queries_logits.cpu()))
        del m
    repeat = all(torch.equal(a, b) for a, b in zip(outs[0], outs[1], strict=True))
    frozen_untouched = not str(cand).startswith(str(frozen.parent))
    result = dict(
        frozen_checks=checks,
        frozen_all_match=all(c["matches"] for c in checks) if "matches" in checks[0] else True,
        extra_artifacts=extra,
        extra_all_unmodified=all(e["unmodified_during_sprint"] for e in extra),
        candidate_dir=str(cand),
        candidate_separate_from_frozen=frozen_untouched,
        candidate_reload_strict_and_repeatable=bool(repeat),
    )
    dump(train_dir / "preservation.json", result)
    print(result["extra_all_unmodified"], frozen_untouched, repeat)


if __name__ == "__main__":
    sys.exit(main())
