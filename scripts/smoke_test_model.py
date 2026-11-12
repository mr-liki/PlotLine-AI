"""Optional Mask2Former smoke test.

Levels (each must be requested explicitly):
  default            import classes only (offline)
  --config-only      download config.json (a few KB) and inspect it
  --download         download pretrained weights (~280 MB to the HF cache), load the model
                     on the configured device and run one forward pass on a synthetic image

The checkpoint is COCO-trained; its predictions on the synthetic image are meaningless and
say nothing about road-marking accuracy. Verify the weight license ("other") before any
commercial use.

    python scripts/smoke_test_model.py --download
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    level = parser.add_mutually_exclusive_group()
    level.add_argument("--config-only", action="store_true")
    level.add_argument("--download", action="store_true")
    parser.add_argument("--device", default=None, help="override runtime.device")
    args = parser.parse_args()

    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation  # noqa: F401

    from roadmarking.config import load_config
    from roadmarking.models.mask2former import ModelSpec, load_config_only, load_model

    cfg = load_config()
    spec = ModelSpec.from_config(cfg)
    if args.device:
        spec = ModelSpec(spec.checkpoint, spec.revision, args.device, spec.require_gpu)
    print(f"checkpoint: {spec.checkpoint}@{spec.revision}  device: {spec.device}")
    print("[PASS] transformers Mask2Former classes importable")

    if not (args.config_only or args.download):
        print("[SKIP] config / weights (pass --config-only or --download)")
        return 0

    mcfg = load_config_only(spec.checkpoint, spec.revision)
    print(
        f"[PASS] config loaded: num_labels={mcfg.num_labels}, "
        f"backbone={mcfg.backbone_config.model_type}"
    )
    if args.config_only:
        print("[SKIP] weights not downloaded (pass --download)")
        return 0

    if os.environ.get("HF_HUB_OFFLINE") == "1":
        print("[FAIL] HF_HUB_OFFLINE=1 but --download requested")
        return 1

    import numpy as np
    import torch

    t0 = time.perf_counter()
    loaded = load_model(spec)
    n_params = sum(p.numel() for p in loaded.model.parameters())
    print(
        f"[PASS] weights loaded on {loaded.device} ({n_params / 1e6:.1f} M params, "
        f"{time.perf_counter() - t0:.1f}s)"
    )

    rng = np.random.default_rng(0)
    image = (rng.random((512, 512, 3)) * 255).astype(np.uint8)  # synthetic, no semantics
    inputs = loaded.processor(images=image, return_tensors="pt").to(loaded.device)
    if loaded.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = loaded.model(**inputs)
    if loaded.device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    result = loaded.processor.post_process_instance_segmentation(
        outputs, target_sizes=[image.shape[:2]]
    )[0]
    print(
        f"[PASS] forward pass: class_queries_logits {tuple(outputs.class_queries_logits.shape)}, "
        f"masks_queries_logits {tuple(outputs.masks_queries_logits.shape)}, {dt * 1000:.0f} ms"
    )
    if loaded.device.type == "cuda":
        print(f"       peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MiB")
    print(
        f"       post-processed segments: {len(result['segments_info'])} "
        "(COCO classes on noise; NOT meaningful)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
