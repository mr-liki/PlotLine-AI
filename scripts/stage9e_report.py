# ruff: noqa: E501  (Markdown report template lines)
"""Stage 9E versioned report and decision."""

import shutil
from pathlib import Path

from stage9d_common import dump, frozen_checks, read, sha256, verify_frozen
from stage9e_lib import BASE, V1, new_run, real_context

STATUS = "REAL_TEXT_DATA_REQUIRED"


def latest(name):
    return Path(read(BASE / name / "latest.json")["run"])


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def main():
    checks = frozen_checks()
    verify_frozen(checks)
    rc = real_context()
    out = new_run("report")
    rec, att, rep, ci = (
        latest("recipes"),
        latest("attribution"),
        latest("reproducibility"),
        latest("checkpoint_integrity"),
    )
    hn, src, pkg = (
        latest("hard_negatives"),
        latest("text_sources"),
        latest("negative_review_package"),
    )
    protocol = read(rec / "stage_c_protocol.json")
    results = read(rec / "stage_c_results.json")
    decision_c = read(rec / "stage_c_decision.json")
    base = read(rec / "baseline_frozen.json")
    attr = read(att / "dash_regression_attribution.json")
    repro = read(rep / "reproducibility_results.json")
    integ = read(ci / "checkpoint_integrity.json")
    hard = read(hn / "hard_negative_audit.json")
    sources = read(src / "real_text_source_inventory.json")
    spec = read(pkg / "review_spec.json")
    qgis = read(pkg / "qgis_project.json")
    qa = BASE / "qa"
    arms = {}
    for key, res in results.items():
        ev = read(rec / f"{key}_evals.json")[str(res["final_step"])]
        arms[key] = (res, ev)

    def row(name, ev, res=None):
        v, r = ev["v1"], ev["raw"]
        status = "-" if res is None else ("PASS" if res["passed"] else "FAIL")
        stop = (
            ""
            if res is None or res["stopped_early_at"] is None
            else f" (stopped {res['stopped_early_at']})"
        )
        return (
            f"| {name}{stop} | {f3(r['pixel']['solid']['f1'])} / {f3(r['pixel']['dash']['f1'])} | "
            f"{f3(v['pixel']['solid']['f1'])} / {f3(v['pixel']['dash']['f1'])} | "
            f"{f3(v['instance_iou50']['solid']['f1'])} / {f3(v['instance_iou50']['dash']['f1'])} | "
            f"{f3(v['complete_marking_recall']['solid'])} / {f3(v['complete_marking_recall']['dash'])} | "
            f"{ev['nodata']['raw_mask_pixels_in_nodata']} / {ev['nodata']['v1_unclipped_outside_footprint']} / {ev['nodata']['v1_accepted_after_clipping_in_nodata']} | {status} |"
        )

    table = "\n".join(
        [row("Frozen baseline", base)] + [row(k, ev, res) for k, (res, ev) in arms.items()]
    )
    failed = {
        k: [c for c, (ok, _) in res["checks"].items() if not ok] for k, (res, _) in arms.items()
    }
    sw = attr["component_swaps"]
    swaps = "\n".join(
        f"| {k} | {f3(v['raw']['solid']['f1'])} / {f3(v['raw']['dash']['f1'])} | {f3(v['v1']['solid']['f1'])} / {f3(v['v1']['dash']['f1'])} | {v['raw']['proposals']} / {v['v1']['proposals']} |"
        for k, v in sw.items()
    )
    fa = attr["dash_false_positive_anatomy"]
    tp_ = attr["training_probe_24_of_65"]
    ho = attr["approved_crops_not_in_R0_pool_24"]
    ex = attr["exposure"]
    opt = attr["optimization"]
    layers_f = attr["per_decoder_layer_raw"]["frozen"]
    layers_r = attr["per_decoder_layer_raw"]["R0"]
    layer_line = ", ".join(
        f"L{i}: {f3(layers_f[f'layer{i}_dash']['f1'])}→{f3(layers_r[f'layer{i}_dash']['f1'])}"
        for i in range(10)
    )
    cov = hard["training_crop_coverage"]
    cov_rows = "\n".join(
        f"| {k} | {v.get('crops', 0)} | {v.get('observed_pixels', 0)} | {v.get('observed_pixels_ignored', 0)} | {v.get('observed_pixels_valid_background', 0)} |"
        for k, v in cov.items()
        if v
    )
    fp9d = hard["false_positives"]["stage9d_rejected_three_class"]
    fpfz = hard["false_positives"]["frozen_two_class"]

    def fprow(block, key, cat):
        r = block[key][cat]
        return f"{r.get('predicted_text', 0)} ({100 * r.get('rate_text', 0):.1f}%)"

    fp_rows = "\n".join(
        f"| {cat} | {fp9d['n9_01_raw'][cat]['pixels']} | {fprow(fp9d, 'n9_01_raw', cat)} | {fprow(fp9d, 'n9_01_v1', cat)} | {fpfz['n9_01_v1'][cat].get('predicted_solid', 0) + fpfz['n9_01_v1'][cat].get('predicted_dash', 0)} |"
        for cat in (
            "reviewed_text",
            "lane_paint",
            "symbols_observed",
            "pavement",
            "bright_non_paint",
            "nodata",
        )
    )
    dev_fp = "; ".join(
        f"{cat} {fprow(fp9d, 'dev_v1', cat)}"
        for cat in ("lane_paint", "symbols_observed", "pavement_valid", "bright_non_paint")
    )
    rd = repro
    c4 = arms.get("C4_plus_teacher_s9061")

    def tail(name):
        p = qa / name
        return (
            p.read_text(encoding="utf-8", errors="replace")
            .replace("﻿", "")
            .strip()
            .splitlines()[-1]
            if p.exists()
            else "not run"
        )

    qa_text = "\n".join(
        [
            f"- pytest (activated environment): `{tail('pytest.log')}`",
            f"- Ruff `src tests scripts`: `{tail('ruff.log')}`",
            f"- Black `--check src tests scripts`: `{tail('black.log')}`",
            f"- Healthcheck: `{tail('healthcheck_summary.log')}`",
        ]
    )
    md = f"""# Stage 9E — stabilize three-class training; real road-text supervision

**Final decision: `{STATUS}`**

- **Real-only decoder fine-tuning is not yet stabilized.** The dash-regression mechanism is now identified and largely corrected. The best recipe (C4: low LR, full approved balanced pool, evidence-gated no-object, soft teacher consistency) meets every predeclared recognition tolerance. It fails only the predeclared NoData-leakage gate, so it was not seed-replicated or adopted.
- **No approved real road-text training data exists.** No acquisition is newer than the last search, and no label source contains text. The three-class prerequisites are therefore unmet, and no three-class training was run.
- The frozen two-class model remains selected, and its hashes were verified throughout. No digitization was done. T7, S8 and the future mixed 500-image test were not touched. N9_01 was not trained on.

## 1. Real-only dash-regression root cause (Stage A)

Every arm was evaluated on the identical 14 development crops at two levels:

- **raw**: the established proposal rule (score = class prob × mean valid sigmoid ≥ 0.25), as used in Stage 9C/9D;
- **V1**: the frozen candidate's established filter (solid ≥ {f3(V1['confidence']['0'])}, dash ≥ {f3(V1['confidence']['1'])}, NMS 0.3), before stitching.

Stage 9D measured only the raw level. The Stage 9D control was replicated deterministically (R0) and reproduced the regression.

**Evidence**

1. **Data starvation.** The 65 crops were {100 * ex['fraction_of_approved']:.1f}% of the {ex['approved_full_pool_crops']} approved Stage 6 training crops. Each crop was seen {ex['exposures_per_crop_min_max'][0]}–{ex['exposures_per_crop_min_max'][1]} times, covering {ex['distinct_tiles']} of {ex['approved_tiles']} tiles and {ex['distinct_physical_groups']} of {ex['approved_physical_groups']} physical marking groups. Stage 9C's identity screen removed 421 crops that share long shared-context client IDs with T7/S8 tiles. None contains a T7/S8 population member and none intersects a protected or N9_01 footprint; both are asserted in code. 409 are the frozen model's own training crops. The screen was stricter than the documented T7/S8 policy.
2. **Overfitting rather than forgetting.** On 24 of the 65 crops, R0's class loss fell from {f3(tp_['frozen_bf16']['mean_losses']['loss_class'])} to {f3(tp_['R0_bf16']['mean_losses']['loss_class'])} and dice from {f3(tp_['frozen_bf16']['mean_losses']['loss_dice'])} to {f3(tp_['R0_bf16']['mean_losses']['loss_dice'])}. On 24 approved crops outside the pool, class loss rose from {f3(ho['frozen_bf16']['mean_losses']['loss_class'])} to {f3(ho['R0_bf16']['mean_losses']['loss_class'])}.
3. **Confident false positives where no-object is never supervised.**
   - Confident unmatched queries per crop rose from {f3(tp_['frozen_bf16']['confident_unmatched_queries_per_crop'])} to {f3(tp_['R0_bf16']['confident_unmatched_queries_per_crop'])}.
   - Dash false-positive proposals rose from {fa['frozen']['fp']} to {fa['R0']['fp']}; those on verified background from {fa['frozen']['fp_location'].get('verified_background', 0)} to {fa['R0']['fp_location'].get('verified_background', 0)}.
   - Dash false positives above the V1 threshold rose from {fa['frozen']['fp_scores_ge_v1_dash_threshold']} to {fa['R0']['fp_scores_ge_v1_dash_threshold']}, with median score {f3(fa['frozen']['fp_score_median'])}→{f3(fa['R0']['fp_score_median'])}.
4. **Optimization pressure.** The decoder LR was {opt['lr_ratio_decoder_vs_original']:.0f}× the frozen model's original 2e-6. The class head moved {100 * opt['relative_parameter_drift']['class_predictor']:.0f}% of its own norm, and {100 * opt['clipped_fraction']:.0f}% of steps hit the gradient clip. There were {opt['nonfinite_steps']} nonfinite steps.
5. **Mixed precision is not a factor.** The bf16 vs fp32 probe class loss was {f3(tp_['frozen_bf16']['mean_losses']['loss_class'])} vs {f3(tp_['frozen_fp32']['mean_losses']['loss_class'])}.
6. **NoData-aware sampling** was verified in Stage 9D; points come from valid pixels only.

**Component swaps** (frozen model with R0 components; F1 shown as solid / dash):

| Model | Raw F1 | V1 F1 | Proposals raw / V1 |
|---|---|---|---|
{swaps}

The largest single-component damage is the mask embedder (raw dash). V1 dash collapses only when the query decoder and mask embedder change together. The class head alone causes moderate damage. The regression is co-adaptation of the mask-generation path, not one layer.

Per-decoder-layer raw dash F1 (frozen→R0): {layer_line}. Late layers lose most.

**Duration.** Raw dash F1 drop was already {f3(base['raw']['pixel']['dash']['f1'] - read(rec / 'R0_stage9d_replica_s9061_evals.json')['200']['raw']['pixel']['dash']['f1'])} by step 200. Damage appears early and fluctuates; it is not monotonic in step count (see `R0_stage9d_replica_s9061_evals.json`).

## 2. No-object supervision audit and safe loss (Stage B)

**Audit.** 0% of real crops are fully valid, so unmatched queries were never trained toward no-object (finding 3 above).

**Alternative.** `roadmarking.training.gated_criterion.GatedNoObjectCriterion` is a subclass; the original criterion file and its recorded hashes are untouched.

- An unmatched query receives no-object CE only when ≥ 95% of its claimed mask (logit > 0) lies on verified-valid pixels, or when it claims nothing.
- Queries touching NoData, unreviewed or unsupported-paint pixels stay unsupervised.
- Matching, valid-only mask losses, ignore masks and fully valid behaviour are inherited unchanged.
- Only the class term changes.

**Tests** (`tests/test_gated_criterion.py`, 9 passing):
- unchanged matching and mask terms;
- the gate selects only verified-valid or empty queries;
- ignored-region queries get zero gradient;
- the matched positive is preserved;
- there is no gradient from the gate into masks;
- extra NoData elsewhere adds no penalty;
- targets in ignored pixels have no effect;
- fully valid images are identical to the original;
- partial overlap with an ignored region is excluded.

**Effect.** C2→C3 (the gate is the only change): raw dash F1 went from a stopped-early collapse to {f3(arms['C3_plus_gated_no_object_s9061'][1]['raw']['pixel']['dash']['f1'])}, and V1 dash F1 to {f3(arms['C3_plus_gated_no_object_s9061'][1]['v1']['pixel']['dash']['f1'])}. V1 dash false-positive instances were {base['v1']['instance_iou50']['dash']['fp']} for the frozen model and {arms['C3_plus_gated_no_object_s9061'][1]['v1']['instance_iou50']['dash']['fp']} for C3.

## 3. Regression-safe recipe feasibility (Stage C)

The protocol was sealed before training (`stage_c_protocol.json`).

- **Tolerances** (drop vs frozen): V1 pixel F1 ≤ {protocol['tolerances']['v1_pixel_f1']}, raw pixel F1 ≤ {protocol['tolerances']['raw_pixel_f1']}, V1 instance F1 ≤ {protocol['tolerances']['instance_f1']}, complete-marking recall ≤ {protocol['tolerances']['complete_recall']}, NoData ≤ {protocol['tolerances']['nodata_ratio']}× baseline, and zero accepted NoData.
- **Early stop** when any F1 drops by more than 0.10.
- **Seeds:** two required.
- **Training setup:** deterministic, with the backbone and pixel decoder frozen and two-class heads.

| Arm (final step) | Raw F1 S/D | V1 F1 S/D | V1 instance F1 S/D | V1 complete recall S/D | NoData raw / V1-unclipped / accepted | Gates |
|---|---|---|---|---|---|---|
{table}

Failed checks: {'; '.join(f'{k}: ' + ', '.join(v) for k, v in failed.items() if v)}.

**Result: no recipe stabilized** (first passing arm: {decision_c['first_passing_arm']}; seed-replicated recipe: {decision_c['seed_replicated_recipe']}). C4 meets every recognition tolerance and improves on the frozen baseline. Its V1 solid/dash F1 is {f3(c4[1]['v1']['pixel']['solid']['f1'])}/{f3(c4[1]['v1']['pixel']['dash']['f1'])} vs {f3(base['v1']['pixel']['solid']['f1'])}/{f3(base['v1']['pixel']['dash']['f1'])}, with dash complete recall unchanged. It exceeds the NoData-leakage tolerance: raw masks reach into unobserved pixels about 2.2× more than the frozen model. Accepted predictions after valid-support clipping are 0 for every arm, but clipping is not used to hide the raw behaviour. Per the protocol, C4 was not seed-replicated and no further arm was added.

The development set is small: 14 dash instances, so one instance equals 0.071 complete recall. Several tolerances are finer than one instance.

## 4. NoData safety (Stage H)

The three levels are reported separately for every arm in the table above:

1. raw query masks in NoData;
2. V1-accepted masks outside the observed footprint, before clipping;
3. accepted predictions after deterministic valid-support clipping.

For the frozen baseline these are {base['nodata']['raw_mask_pixels_in_nodata']} / {base['nodata']['v1_unclipped_outside_footprint']} / {base['nodata']['v1_accepted_after_clipping_in_nodata']} pixels.

The established contract already clips: `continuity.reconcile` masks with valid support, and stitching never bridges NoData. That code is hash-preserved and unchanged.

NoData on development crops is measured with a proxy, invalid and zero intensity, because the crop files store only validity. N9_01 uses its true observed raster.

**Proposed next intervention** (needs approval as a new sealed protocol): extend the soft teacher mask consistency to NoData pixels. The frozen model's own soft masks would anchor the student's raw extent there. These are soft targets only and never negatives.

## 5. Reproducibility (Stage I)

- **Stage 9D repeat failure.** The two runs were identical for {rd['stage9d_repeat_failure']['identical_steps_before_divergence']} steps, then diverged at step {rd['stage9d_repeat_failure']['first_divergent_step']} by {rd['stage9d_repeat_failure']['relative_difference_at_divergence']:.1e} relative. Data order, initialization, scheduler, optimizer and AMP state were identical, there was no GradScaler, and no resume was used.
- **Cause.** GPU nondeterminism, not a state defect.
- **Default mode** (25 steps, two clean processes): diverged at step {rd['default']['first_divergent_step']}.
- **Deterministic algorithms only:** diverged at step {rd['deterministic']['first_divergent_step']}. The op PyTorch reported was `{', '.join(rd['deterministic']['nondeterministic_ops_reported'])}`, the backward of the criterion's bilinear point sampling.
- **Deterministic plus exact bilinear gather** (`roadmarking.training.deterministic_sampling`, numerically equal to `grid_sample`, 2 tests): identical losses at all 25 steps = {rd['deterministic_exact']['identical_losses_all_steps']}, identical final trainable-parameter hash = {rd['deterministic_exact']['final_trainable_hash_identical']}.

This is bit-exact on this GPU and software stack over 25 steps. No claim is made beyond that. All Stage C arms used this mode.

## 6. Checkpoint integrity (Stage J)

Each case was trained for 3 steps, saved, reloaded strictly in a clean process, and compared with fp32 inference on a real development crop and a synthetic validation image:

| Case | Outputs identical | Config differences | Class rows | criterion.empty_weight |
|---|---|---|---|---|
| Three-class migration | {integ['three_class']['outputs_identical']} | {integ['three_class']['config_differences'] or 'none'} | {integ['three_class']['class_rows']} | {integ['three_class']['criterion_empty_weight']} |
| Two-class real-only | {integ['two_class_real_only']['outputs_identical']} | {integ['two_class_real_only']['config_differences'] or 'none'} | {integ['two_class_real_only']['class_rows']} | {integ['two_class_real_only']['criterion_empty_weight']} |

The frozen two-class hashes were verified. Rejected Stage 9D checkpoints are retained ({len(integ['stage9d_rejected_checkpoints_retained']['weights'])} weight files) and not promoted. No cleanup was done without a storage policy.

## 7. Real road-text training sources (Stage D)

- The share holds {sources['share_lidar_files']} LiDAR files (newest {sources['share_newest_utc'][:10]}); there are {sources['local_lidar_files']} local files.
- **{len(sources['sources_newer_than_last_search'])} sources** are newer than the Stage 9C.2 search ({sources['stage9c2_search_utc'][:10]}).
- Text features: {sources['client_reference_text_features']} in the client reference layers, and {sources['approved_geometry_text_features']} in {sources['approved_geometry_files_checked']} approved geometry files.
- No rescan was performed ({sources['rescan_reason']}).
- The future mixed-500 test footprint inventory has not been supplied (`{sources['future_mixed_500_test_inventory_supplied']}`), so separation from it cannot be verified yet.

**Real text training candidates: none.** There is no text QGIS package.

Options that need your decision:
{chr(10).join('- ' + x for x in sources['new_methods_possible_with_user_approval'])}

## 8. Non-text hard negatives (Stage E)

Client-located non-line paint inside the {len(rc['full'])} approved training crops (client geometry locates only; it does not approve):

| Class | Crops | Observed px | Ignored | Treated as valid background |
|---|---:|---:|---:|---:|
{cov_rows}

Unsupported paint is ignored (dilated 2 px) everywhere; the about 90 edge pixels on valid background are geometry-offset slivers. So nothing currently tells the model that arrows or hatching are not text.

**False positives on N9_01** (rejected Stage 9D three-class model; the frozen model shown for solid/dash):

| Category | Pixels | 9D text, raw | 9D text, V1 (text ≥ 0.5) | Frozen solid+dash, V1 |
|---|---:|---:|---:|---:|
{fp_rows}

On development crops, 9D text at V1: {dev_fp}.

**Review package:** {len(spec['crops'])} training crops and {spec['symbols']} symbols ({spec['symbols_by_class']}). Files:
- `{spec['gpkg']}`, with the editable layer `non_text_review_EDITABLE`;
- QGIS project `{qgis['project']}`: {qgis['valid_layers']}/{qgis['expected_layers']} valid layers, {qgis['bookmarks']} bookmarks, reopened = {qgis['project_reopened']}, no approval recorded.

**Policy.** Symbols stay ignored for solid/dash. Only human-CONFIRMED symbols may become text-class negatives.

## 9. Three-class experiment (Stages F/G)

**Not run.** The prerequisites failed:
- the regression-safe recipe did not pass (NoData gate);
- no approved real text positives exist;
- the hard negatives are pending human review.

Class mapping (0 solid, 1 dash, 2 road_text, no-object 3) and migration save/reload pass (section 6).

## 10. QA

{qa_text}

## Artifacts

Recipes {rec} · attribution {att} · reproducibility {rep} · checkpoints {ci} · hard negatives {hn} · sources {src} · review package {pkg}

Nothing was committed.
"""
    (out / "report.md").write_text(md, encoding="utf-8")
    for p in (rec / "gallery_solid_dash_v1.png",):
        if p.exists():
            shutil.copy2(p, out / p.name)
    dump(
        out / "decision.json",
        dict(
            status=STATUS,
            real_only_recipe_stabilized=False,
            best_recipe="C4_plus_teacher (recognition tolerances met; NoData gate failed; single seed)",
            three_class_run=False,
            real_text_training_data=False,
            hard_negative_review_pending=True,
            selected_model="frozen two-class candidate (unchanged)",
            promoted=None,
        ),
    )
    dump(
        out / "package_manifest.json",
        {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()},
    )
    verify_frozen(checks)
    dump(BASE / "latest.json", {"run": str(out), "status": STATUS})
    print(out)


if __name__ == "__main__":
    main()
