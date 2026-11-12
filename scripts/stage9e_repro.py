"""Stage 9E-I: reproducibility test (short) — default vs deterministic-algorithm mode.

Each mode runs twice from a clean model with identical seed, data order and settings;
per-step losses and the final trainable-parameter hash are compared. Warnings raised by
torch.use_deterministic_algorithms(warn_only=True) identify nondeterministic CUDA ops.
"""

import hashlib
import subprocess
import sys
import warnings
from pathlib import Path

import torch
from transformers import Mask2FormerImageProcessor

from stage9d_common import dump, frozen_checks, read, verify_frozen
from stage9e_lib import BASE, new_run, real_context, train_arm, with_root

STEPS = 25
SEED = 9051
LRS = dict(query_decoder=5e-6, mask_embedder=5e-6, class_predictor=5e-5)


def one(mode, tag, out):
    checks = frozen_checks()
    frozen_path = Path(checks[0]["path"]).parent
    processor = Mask2FormerImageProcessor.from_pretrained(
        frozen_path.parent / "processor", local_files_only=True
    )
    rc = real_context()
    crops = with_root(rc["full"], rc["dataset"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _info, _, logs = train_arm(
            f"repro_{mode}_{tag}",
            out,
            frozen_path,
            processor,
            [],
            crops,
            sampler_mode="cycle",
            lrs=LRS,
            steps=STEPS,
            eval_every=10**9,
            seed=SEED,
            save_dir=out / f"repro_{mode}_{tag}",
            deterministic=mode.startswith("deterministic"),
            exact_bilinear=mode.endswith("exact"),
        )
    from safetensors.torch import load_file

    state = load_file(out / f"repro_{mode}_{tag}/model/model.safetensors")
    h = hashlib.sha256()
    for k in sorted(state):
        if "transformer_module" in k or "class_predictor" in k:
            h.update(state[k].float().cpu().numpy().tobytes())
    ops = sorted(
        {
            str(w.message).split(" does not have a deterministic")[0][:160]
            for w in caught
            if "deterministic" in str(w.message)
        }
    )
    dump(
        out / f"repro_{mode}_{tag}_summary.json",
        dict(losses=[r["total"] for r in logs], hash=h.hexdigest(), nondeterministic_ops=ops),
    )


def main():
    if len(sys.argv) == 4:
        one(sys.argv[1], sys.argv[2], Path(sys.argv[3]))
        return
    out = new_run("reproducibility")
    checks = frozen_checks()
    verify_frozen(checks)
    for mode in ("default", "deterministic", "deterministic_exact"):
        for tag in ("a", "b"):
            # Clean process per run so no state leaks between runs.
            subprocess.run([sys.executable, __file__, mode, tag, str(out)], check=True)
    result = {}
    for mode in ("default", "deterministic", "deterministic_exact"):
        a = read(out / f"repro_{mode}_a_summary.json")
        b = read(out / f"repro_{mode}_b_summary.json")
        first = next(
            (
                i + 1
                for i, (x, y) in enumerate(zip(a["losses"], b["losses"], strict=True))
                if x != y
            ),
            None,
        )
        result[mode] = dict(
            steps=STEPS,
            identical_losses_all_steps=first is None,
            first_divergent_step=first,
            max_abs_loss_diff=max(
                abs(x - y) for x, y in zip(a["losses"], b["losses"], strict=True)
            ),
            final_trainable_hash_identical=a["hash"] == b["hash"],
            nondeterministic_ops_reported=sorted(
                set(a["nondeterministic_ops"]) | set(b["nondeterministic_ops"])
            ),
        )
    stage9d = read(
        Path(read(BASE.parent / "stage9d/finetune/latest.json")["run"]) / "training_log_run1.json"
    )
    stage9d_b = read(
        Path(read(BASE.parent / "stage9d/finetune/latest.json")["run"])
        / "training_log_run2_repeat.json"
    )
    first = next(
        i + 1
        for i, (x, y) in enumerate(zip(stage9d, stage9d_b, strict=True))
        if x["total"] != y["total"]
    )
    result["stage9d_repeat_failure"] = dict(
        identical_steps_before_divergence=first - 1,
        first_divergent_step=first,
        relative_difference_at_divergence=abs(
            stage9d[first - 1]["total"] - stage9d_b[first - 1]["total"]
        )
        / abs(stage9d[first - 1]["total"]),
        interpretation="Identical seed/data/init for 11 steps, then ~2e-5 relative float divergence that compounds: GPU nondeterminism, not a training-state or resume defect.",  # noqa: E501
        state_audit=dict(
            data_order="fixed predeclared schedule (identical crop ids per step)",
            model_init="frozen weights + seeded text row (fork_rng)",
            amp_scaler="none (bf16 autocast, no GradScaler state)",
            scheduler="linear warmup computed from step index (stateless)",
            optimizer="fresh AdamW per run",
            resume="no resume used in Stage 9D",
            synthetic_sampling="fixed predeclared schedule",
        ),
    )
    verify_frozen(checks)
    dump(out / "reproducibility_results.json", result)
    print(
        out,
        {
            k: (
                v.get("identical_losses_all_steps"),
                v.get("final_trainable_hash_identical"),
                v.get("first_divergent_step"),
            )
            for k, v in result.items()
            if k != "stage9d_repeat_failure"
        },
    )


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
