"""Fresh-process verification of the selected complete Phase 5.4 checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from roadmarking.rasterization.pipeline import dump, sha256
from roadmarking.training.pilot import load_crop

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    folder = args.checkpoint.resolve()
    bundle = json.loads((folder / "bundle.json").read_text())
    if sha256(folder / "model/model.safetensors") != bundle["weight_sha256"]:
        raise ValueError("Selected model checksum mismatch")
    config = json.loads((folder.parent / "training_config.json").read_text())
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = (
        Mask2FormerForUniversalSegmentation.from_pretrained(folder / "model", local_files_only=True)
        .to("cuda")
        .eval()
    )
    processor = Mask2FormerImageProcessor.from_pretrained(
        folder / "processor", local_files_only=True
    )
    crop = bundle["metrics"]["crops"][0]
    pixels, _, _, _, _ = load_crop(config["dataset"], crop, processor, "cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        output = model(pixel_values=pixels, output_auxiliary_logits=False)
    reference = np.load(
        folder.parent / f"validation_step{bundle['step']:04d}" / (crop["crop_id"] + ".npz")
    )
    differences = {}
    for name, values in (
        ("class_logits", output.class_queries_logits),
        ("mask_logits", output.masks_queries_logits),
    ):
        array = values.float().cpu().numpy()
        np.testing.assert_allclose(array, reference[name], atol=0.002, rtol=0.002)
        differences[name] = float(np.max(np.abs(array - reference[name])))
    proof = {
        "fresh_process_model_reload": True,
        "max_abs_logit_differences": differences,
        "optimizer_scheduler_scaler_rng_state_present": (folder / "training_state.pt").exists(),
        "weight_sha256": bundle["weight_sha256"],
    }
    dump(folder.parent / "selected_reload_validation.json", proof)
    print(json.dumps(proof, indent=2))
