"""Stage 9E-A/C: replicate the Stage 9D real-only regression, then test bounded,
one-factor-at-a-time interventions for a regression-safe real-only decoder recipe.

Protocol, tolerances and stopping rules are written before any arm trains. Arms:
  R0  Stage 9D control replica (65 crops, uniform, LR 5e-5) - diagnostic, 800 steps
  C1  R0 with conservative LR (decoder/mask 5e-6, classifier 5e-5)        - 400 steps
  C2  C1 + full approved real pool, balanced legacy/expanded class cycle   - 400 steps
  C3  C2 + evidence-gated no-object supervision                           - 400 steps
  C4  C3 + soft teacher consistency (only if C1-C3 all fail)               - 400 steps
The first (simplest) passing arm is repeated with a second seed; both must pass.
All arms: backbone and pixel decoder frozen, two-class head, deterministic algorithms with
exact bilinear sampling, bf16 autocast, grad clip 1.0, 50-step warmup.
"""

import sys
from pathlib import Path

from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from stage9d_common import ROOT, dump, frozen_checks, verify_frozen
from stage9e_lib import (
    development_crops,
    evaluate,
    gate_check,
    new_run,
    real_context,
    train_arm,
    with_root,
)

TOL = dict(
    v1_pixel_f1=0.02, raw_pixel_f1=0.03, instance_f1=0.05, complete_recall=0.05, nodata_ratio=1.10
)
STOP_DROP = 0.10
SEEDS = (9061, 9062)
HIGH = dict(query_decoder=5e-5, mask_embedder=5e-5, class_predictor=5e-4)
LOW = dict(query_decoder=5e-6, mask_embedder=5e-6, class_predictor=5e-5)
TEACHER = dict(mask=5.0, class_=2.0)


def stop_rule(res, base):
    return any(
        (base[lvl]["pixel"][c]["f1"] or 0) - (res[lvl]["pixel"][c]["f1"] or 0) > STOP_DROP
        for lvl in ("raw", "v1")
        for c in ("solid", "dash")
    )


def main():
    out = new_run("recipes")
    checks = frozen_checks()
    verify_frozen(checks)
    frozen_path = Path(checks[0]["path"]).parent
    processor = Mask2FormerImageProcessor.from_pretrained(
        frozen_path.parent / "processor", local_files_only=True
    )
    rc = real_context()
    full = with_root(rc["full"], rc["dataset"])
    subset = with_root(rc["stage9d"], rc["dataset"])
    development = development_crops(rc["ctx"])
    arms = {
        "R0_stage9d_replica": dict(
            crops="stage9d_65",
            sampler_mode="uniform",
            lrs=HIGH,
            steps=800,
            gated=False,
            teacher=None,
        ),
        "C1_low_lr": dict(
            crops="stage9d_65",
            sampler_mode="uniform",
            lrs=LOW,
            steps=400,
            gated=False,
            teacher=None,
        ),
        "C2_low_lr_full_balanced": dict(
            crops="full_486", sampler_mode="cycle", lrs=LOW, steps=400, gated=False, teacher=None
        ),
        "C3_plus_gated_no_object": dict(
            crops="full_486", sampler_mode="cycle", lrs=LOW, steps=400, gated=True, teacher=None
        ),
        "C4_plus_teacher": dict(
            crops="full_486",
            sampler_mode="cycle",
            lrs=LOW,
            steps=400,
            gated=True,
            teacher=TEACHER,
            conditional="only if C1-C3 all fail",
        ),
    }
    protocol = dict(
        predeclared_before_training=True,
        warm_start=dict(path=str(frozen_path), sha256=checks[0]["expected_sha256"]),
        development_crops=[c["crop_id"] for _, c in development],
        evaluation_levels=[
            "raw (0.25 proposal point, valid-clipped)",
            "V1 (frozen thresholds + NMS 0.3, pre-stitching)",
        ],
        tolerances=TOL,
        early_stop=f"stop an arm when any raw/V1 solid or dash pixel F1 drops by > {STOP_DROP}",
        pass_rule="all tolerances at the arm's final step, for two seeds",
        arms=arms,
        seeds=SEEDS,
        pool_sizes=dict(full_486=len(full), stage9d_65=len(subset)),
        protected_population_members_in_pool=0,
        protected_population_size=rc["population_members"],
        pool_policy=(
            "Full approved Stage 6 train split (486). Stage 9C/9D additionally excluded 421 crops "
            "sharing shared-context client IDs with T7/S8 tiles; none contains a T7/S8 population "
            "member, none intersects a protected or N9_01 footprint (asserted), and 409 are the "
            "frozen model's own training crops."
        ),
        determinism="torch.use_deterministic_algorithms + exact bilinear sampling (verified bit-exact over 25 steps)",  # noqa: E501
        n9_01_used=False,
        synthetic_used=False,
    )
    dump(out / "stage_c_protocol.json", protocol)
    baseline_model = Mask2FormerForUniversalSegmentation.from_pretrained(
        frozen_path, local_files_only=True
    ).cuda()
    baseline = evaluate(baseline_model, processor, development)
    del baseline_model
    dump(out / "baseline_frozen.json", baseline)
    results = {}
    models_root = ROOT / "models/mask2former_swin_s" / f"stage9e_{out.name}"

    def run(name, seed):
        a = arms[name]
        crops = subset if a["crops"] == "stage9d_65" else full
        tw = (
            dict(mask=a["teacher"]["mask"], **{"class": a["teacher"]["class_"]})
            if a["teacher"]
            else None
        )
        info, evals, _ = train_arm(
            f"{name}_s{seed}",
            out,
            frozen_path,
            processor,
            development,
            crops,
            sampler_mode=a["sampler_mode"],
            lrs=a["lrs"],
            steps=a["steps"],
            eval_every=100,
            seed=seed,
            gated=a["gated"],
            teacher_weights=tw,
            stop_rule=None if name.startswith("R0") else stop_rule,
            baseline=baseline,
            save_dir=models_root / f"{name}_s{seed}",
            deterministic=True,
        )
        last = max(evals)
        checks_, ok = gate_check(evals[last], baseline, TOL)
        ok = ok and info["stopped_early_at"] is None
        results[f"{name}_s{seed}"] = dict(
            final_step=last, passed=ok, checks=checks_, stopped_early_at=info["stopped_early_at"]
        )
        dump(out / "stage_c_results.json", results)
        return ok

    run("R0_stage9d_replica", SEEDS[0])
    passing = None
    for name in ("C1_low_lr", "C2_low_lr_full_balanced", "C3_plus_gated_no_object"):
        if run(name, SEEDS[0]) and passing is None:
            passing = name
    if passing is None and run("C4_plus_teacher", SEEDS[0]):
        passing = "C4_plus_teacher"
    selected = None
    if passing is not None and run(passing, SEEDS[1]):
        selected = passing
    verify_frozen(checks)
    dump(
        out / "stage_c_decision.json",
        dict(
            first_passing_arm=passing,
            seed_replicated_recipe=selected,
            stabilized=selected is not None,
            models=str(models_root),
        ),
    )
    print(out, passing, selected, flush=True)


if __name__ == "__main__":
    sys.exit(main())
