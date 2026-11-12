"""Fresh-process checkpoint reload and reproduction of saved held-out logits."""

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
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    result = json.loads((args.run / "result.json").read_text())
    config = result["training_config"]
    evaluation = result["held_out_evaluation"] or result["training_preview"]
    checkpoint = Path(result["checkpoint"])
    if sha256(checkpoint / "model.safetensors") != result["weight_sha256"]:
        raise ValueError("Trained checkpoint changed")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint, local_files_only=True)
    processor = Mask2FormerImageProcessor.from_pretrained(
        result["processor"], local_files_only=True
    )
    model.to("cuda").eval()
    pixels, _, _, _, _ = load_crop(config["dataset"], evaluation["crop"], processor, "cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        outputs = model(pixel_values=pixels, output_auxiliary_logits=False)
    reference = np.load(Path(evaluation["output"]) / "reload_reference.npz")
    differences = {}
    for key, actual in (
        ("class_logits", outputs.class_queries_logits),
        ("mask_logits", outputs.masks_queries_logits),
    ):
        array = actual.float().cpu().numpy()
        np.testing.assert_allclose(array, reference[key], rtol=0.002, atol=0.002)
        differences[key] = float(np.max(np.abs(array - reference[key])))
    dump(
        args.run / "reload_validation.json",
        {
            "checkpoint_reloaded_in_fresh_process": True,
            "logits_reproduced": True,
            "max_abs_logit_differences": differences,
            "tolerance_rtol_atol": [0.002, 0.002],
        },
    )
    print(json.dumps(differences, indent=2))
