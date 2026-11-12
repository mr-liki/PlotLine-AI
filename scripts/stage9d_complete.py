"""Stage 9D completion after the run-1 reload defect.

1. G7: reload run-1 step-1200 weights. The only permitted mismatch is the unused internal
   HF criterion.empty_weight buffer; evaluation counts must match run 1 exactly. A
   corrected copy is then saved and reloaded strictly through from_pretrained.
2. G6: identical-seed repeat of the predeclared schedule (fixed loader; same recipe).
3. Diagnostic control (not a candidate): the same schedule with synthetic steps removed,
   to attribute solid/dash change to text addition versus decoder fine-tuning itself.
"""

from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from stage9c_attested_pilot import source_context, text_arrays
from stage9d_common import BASE, dump, frozen_checks, read, sha256, synthetic_stems, verify_frozen
from stage9d_eval import development_crops, full_eval, n9_groups
from stage9d_finetune import SEED, STEPS, gates, train_once
from stage9d_train_lib import load_three_class


def key_counts(r):
    return [
        r["n9_01"]["text"]["tp"],
        r["n9_01"]["text"]["fp"],
        r["solid_dash"]["longitudinal_solid"]["tp"],
        r["solid_dash"]["longitudinal_solid"]["fp"],
        r["solid_dash"]["longitudinal_dash"]["tp"],
        r["solid_dash"]["longitudinal_dash"]["fp"],
        r["synthetic_val"]["pixel"]["tp"],
        r["synthetic_val"]["pixel"]["fp"],
    ]


def main():
    out = Path(read(BASE / "finetune/latest.json")["run"])
    checks = frozen_checks()
    verify_frozen(checks)
    ctx = source_context()
    frozen_path = Path(checks[0]["path"]).parent
    processor = Mask2FormerImageProcessor.from_pretrained(
        frozen_path.parent / "processor", local_files_only=True
    )
    protocol = read(out / "training_protocol.json")
    schedule = [tuple(x) for x in protocol["schedule"]]
    development = development_crops(ctx)
    text = text_arrays(ctx[0], ctx[9], ctx[10])
    groups = n9_groups(ctx[10], ctx[9], text[1].shape)
    eval_args = (
        development,
        text,
        groups,
        protocol["synthetic_train_fit_eval"],
        protocol["synthetic_val_eval"],
    )
    assert eval_args[3] == synthetic_stems("train")[:24]
    evaluations = read(out / "checkpoint_evaluations.json")
    final = evaluations[str(STEPS)]
    baseline = evaluations["baseline"]
    save_root = ROOT_MODELS / f"stage9d_{out.name}"
    ckpt = save_root / f"step{STEPS:04}"
    # --- G7 reload
    state = load_file(ckpt / "model/model.safetensors")
    model = load_three_class(frozen_path, SEED)
    expected = model.state_dict()
    mismatched = sorted(k for k in state if k in expected and state[k].shape != expected[k].shape)
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    allowed = mismatched == ["criterion.empty_weight"] and not missing and not unexpected
    state.pop("criterion.empty_weight")
    result_keys = model.load_state_dict(state, strict=False)
    model.cuda()
    reloaded_eval, _ = full_eval(model, processor, *eval_args)
    weights_reload_exact = key_counts(reloaded_eval) == key_counts(final)
    fixed_dir = save_root / f"step{STEPS:04}_reloadfix"
    model.save_pretrained(fixed_dir / "model")
    processor.save_pretrained(fixed_dir / "processor")
    del model
    torch.cuda.empty_cache()
    strict = Mask2FormerForUniversalSegmentation.from_pretrained(
        fixed_dir / "model", local_files_only=True
    ).cuda()
    strict_eval, _ = full_eval(strict, processor, *eval_args)
    strict_exact = key_counts(strict_eval) == key_counts(final)
    del strict
    torch.cuda.empty_cache()
    reload = dict(
        original_save_path_strict_reload=False,
        defect="three-class migration did not rebuild HF criterion.empty_weight (size 3 vs 4)",
        defect_fixed_in="scripts/stage9d_train_lib.py load_three_class",
        state_dict_mismatches=mismatched,
        missing=missing,
        unexpected=unexpected,
        only_unused_buffer_mismatched=allowed,
        missing_after_load=list(result_keys.missing_keys),
        weights_reload_counts_identical=weights_reload_exact,
        corrected_copy=str(fixed_dir),
        corrected_copy_sha256=sha256(fixed_dir / "model/model.safetensors"),
        corrected_copy_strict_from_pretrained_identical=strict_exact,
        g7_pass_after_fix=allowed and weights_reload_exact and strict_exact,
        g7_pass_as_originally_saved=False,
    )
    dump(out / "save_reload_audit.json", reload)
    print("reload", reload["g7_pass_after_fix"], flush=True)
    # --- G6 repeat
    evals2, _ = train_once(
        "run2_repeat", frozen_path, processor, ctx, schedule, eval_args, out, None, eval_at=(STEPS,)
    )
    f2 = evals2[STEPS]
    dump(out / "repeat_run_final_evaluation.json", f2)
    repeat_detail = dict(
        n9_iou=[final["n9_01"]["text"]["iou"], f2["n9_01"]["text"]["iou"]],
        solid_f1=[
            final["solid_dash"]["longitudinal_solid"]["f1"],
            f2["solid_dash"]["longitudinal_solid"]["f1"],
        ],
        dash_f1=[
            final["solid_dash"]["longitudinal_dash"]["f1"],
            f2["solid_dash"]["longitudinal_dash"]["f1"],
        ],
        synthetic_val_iou=[
            final["synthetic_val"]["pixel"]["iou"],
            f2["synthetic_val"]["pixel"]["iou"],
        ],
    )
    repeat_ok = all(abs((a or 0) - (b or 0)) <= 0.02 for a, b in repeat_detail.values())
    # --- Diagnostic control: real-only, identical order/LR/steps for the real portion
    control_schedule = [s for s in schedule if s[0] == "real"]
    control_evals, control_info = train_once(
        "control_real_only",
        frozen_path,
        processor,
        ctx,
        control_schedule,
        eval_args,
        out,
        None,
        eval_at=(len(control_schedule),),
    )
    control = dict(
        role="diagnostic ablation only; never a candidate",
        steps=len(control_schedule),
        evaluation=control_evals[len(control_schedule)],
        exposure=control_info["exposure"],
    )
    dump(out / "control_real_only_evaluation.json", control)
    gate, detail = gates(final, baseline, repeat_ok, reload["g7_pass_as_originally_saved"])
    verify_frozen(checks)
    decision = dict(
        candidate=str(ckpt),
        candidate_corrected_copy=str(fixed_dir),
        gates=gate,
        gate_detail=detail,
        repeat=repeat_detail,
        repeat_within_tolerance=repeat_ok,
        save_reload=reload,
        g7_note="Fails as originally saved; passes after the loader fix (corrected copy).",
        preserved_artifact_hashes_ok=True,
        all_gates_passed=all(gate.values()),
        promoted=False,
        selected_model="frozen two-class candidate (unchanged)",
        note="Provisional development gates only; N9_01 is development-used.",
    )
    dump(out / "gate_evaluation.json", decision)
    print(out, gate, flush=True)


ROOT_MODELS = Path(__file__).resolve().parents[1] / "models/mask2former_swin_s"

if __name__ == "__main__":
    main()
