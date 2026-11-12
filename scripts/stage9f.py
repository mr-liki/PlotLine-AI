"""Stage 9F: test C5 against the sealed Stage C tolerances."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from roadmarking.training.pilot import load_crop
from stage9d_common import ROOT, dump, frozen_checks, read, verify_frozen
from stage9e_checkpoint_integrity import outputs
from stage9e_lib import (
    development_crops,
    evaluate,
    gate_check,
    new_run,
    real_context,
    train_arm,
    with_root,
)
from stage9e_recipes import LOW, SEEDS, STOP_DROP, TOL, stop_rule

BASE = ROOT / "outputs/reports/stage9f"


def setup():
    checks = frozen_checks()
    verify_frozen(checks)
    frozen = Path(checks[0]["path"]).parent
    processor = Mask2FormerImageProcessor.from_pretrained(
        frozen.parent / "processor", local_files_only=True
    )
    rc = real_context()
    return (
        checks,
        frozen,
        processor,
        development_crops(rc["ctx"]),
        with_root(rc["full"], rc["dataset"]),
    )


def train(name, out, model_dir, frozen, processor, development, crops, seed, steps, **kwargs):
    return train_arm(
        name,
        out,
        frozen,
        processor,
        development,
        crops,
        sampler_mode="cycle",
        lrs=LOW,
        steps=steps,
        eval_every=100,
        seed=seed,
        gated=True,
        teacher_weights={"mask": 5.0, "class": 2.0},
        teacher_include_invalid=True,
        deterministic=True,
        strict_determinism=True,
        exact_bilinear=True,
        save_dir=model_dir,
        **kwargs,
    )


def probe(model, processor, development):
    xs = [load_crop(root, crop, processor, "cuda")[0] for root, crop in development[:1]]
    ((classes, masks),) = outputs(model, xs)
    return hashlib.sha256(classes.tobytes() + masks.tobytes()).hexdigest()


def verify_worker(mode, out, target):
    checks, frozen, processor, development, crops = setup()
    if mode == "reload":
        model, loading = Mask2FormerForUniversalSegmentation.from_pretrained(
            target / "model", local_files_only=True, output_loading_info=True
        )
        assert not any(loading.values()), loading
        processor = Mask2FormerImageProcessor.from_pretrained(
            target / "processor", local_files_only=True
        )
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        fingerprint = probe(model.cuda(), processor, development)
        expected = read(out / f"{target.name}_before.json")
        dump(
            out / f"{target.name}_reload.json",
            dict(
                outputs_bit_exact=fingerprint == expected["outputs"],
                config_matches=json.loads(
                    json.dumps(model.config.to_dict() | {"_name_or_path": None})
                )
                == expected["config"],
                buffers_match={k: list(v.shape) for k, v in model.named_buffers()}
                == expected["buffers"],
            ),
        )
    else:
        _, _, logs = train(mode, out, target, frozen, processor, [], crops, SEEDS[0], 25)
        from safetensors.torch import load_file

        state = load_file(target / "model/model.safetensors")
        digest = hashlib.sha256()
        for name, tensor in sorted(state.items()):
            digest.update(name.encode())
            digest.update(tensor.numpy().tobytes())
        dump(out / f"{mode}_summary.json", dict(logs=logs, parameter_hash=digest.hexdigest()))
    verify_frozen(checks)


def main(out):
    checks, frozen, processor, development, crops = setup()
    stage_c = ROOT / "outputs/reports/stage9e/recipes/20261112T071004863987Z/stage_c_protocol.json"
    assert read(stage_c)["tolerances"] == TOL
    assert len(crops) == 486 and len(development) == 14
    with (out / "protocol.json").open("x", encoding="utf-8") as stream:

        json.dump(
            dict(
                arm="C5",
                parent_protocol=str(stage_c),
                frozen_checks=checks,
                development=[crop["crop_id"] for _, crop in development],
                approved_training_crops=len(crops),
                tolerances=TOL,
                seeds=SEEDS,
                steps=400,
                eval_every=100,
                early_stop_pixel_f1_drop=STOP_DROP,
                sampler="cycle",
                lrs=LOW,
                gated=True,
                exact_bilinear=True,
                strict_determinism=True,
                teacher_mask_weight=5.0,
                teacher_class_weight=2.0,
                invalid_mask_loss=(
                    "Add mean BCE with frozen teacher sigmoid targets on up to "
                    "criterion.max_points evenly spaced invalid pixels in flattened order. "
                    "Keep valid loss unchanged."
                ),
                pass_rule=(
                    "Both seeds must pass all Stage C gates at step 400. "
                    "Stop after first failure."
                ),
                evaluation=["raw", "V1 pre-stitching under Stage C evaluation"],
                synthetic_used=False,
                n9_01_used=False,
                protected_tests_used=False,
            ),
            stream,
            indent=2,
        )
    baseline_model = Mask2FormerForUniversalSegmentation.from_pretrained(
        frozen, local_files_only=True
    ).cuda()
    baseline = evaluate(baseline_model, processor, development)
    del baseline_model
    dump(out / "baseline_frozen.json", baseline)
    results = {}
    models = ROOT / "models/mask2former_swin_s" / f"stage9f_{out.name}"
    status = "REAL_ONLY_TRAINING_STABILIZED"
    for seed in SEEDS:
        name = f"C5_s{seed}"

        def capture(model, evals, stopped, name=name):
            if stopped is None and gate_check(evals[400], baseline, TOL)[1]:
                dump(
                    out / f"{name}_before.json",
                    dict(
                        outputs=probe(model, processor, development),
                        config=model.config.to_dict() | {"_name_or_path": None},
                        buffers={k: list(v.shape) for k, v in model.named_buffers()},
                    ),
                )

        info, evals, _ = train(
            name,
            out,
            models / name,
            frozen,
            processor,
            development,
            crops,
            seed,
            400,
            baseline=baseline,
            stop_rule=stop_rule,
            before_save=capture,
        )
        last = max(evals)
        gates, passed = gate_check(evals[last], baseline, TOL)
        passed = passed and info["stopped_early_at"] is None
        results[name] = dict(passed=passed, step=last, gates=gates)
        dump(out / "results.json", results)
        if not passed:
            status = (
                "LOSS_DESIGN_BLOCKED"
                if any(
                    not ok
                    for key, (ok, _) in gates.items()
                    if "nodata" in key or "footprint" in key
                )
                else "NO_IMPROVEMENT"
            )
            break
    if status == "REAL_ONLY_TRAINING_STABILIZED":
        for name in results:
            worker("reload", out, models / name)
            assert all(read(out / f"{name}_reload.json").values())
        for mode in ("repro_a", "repro_b"):
            worker(mode, out, models / mode)
        exact = read(out / "repro_a_summary.json") == read(out / "repro_b_summary.json")
        dump(out / "reproducibility.json", dict(steps=25, bit_exact=exact))
        assert exact
    verify_frozen(checks)
    dump(out / "decision.json", dict(status=status, seeds=results, models=str(models)))
    print(out, status, flush=True)


def worker(mode, out, target):
    subprocess.run([sys.executable, __file__, mode, str(out), str(target)], check=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    if len(sys.argv) == 4:
        verify_worker(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        # Use the existing versioned run helper with the Stage 9F report root.
        import stage9e_lib

        stage9e_lib.BASE = BASE
        main(new_run("training"))
