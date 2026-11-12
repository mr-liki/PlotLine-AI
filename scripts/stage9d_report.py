# ruff: noqa: E501  (Markdown report template lines)
"""Stage 9D versioned report: compiles diagnosis, supervision, learnability, fine-tuning,
regression, transfer, domain gap, decision and QA into one Markdown + JSON package."""

import shutil
from pathlib import Path

from stage9c_attested_pilot import source_context
from stage9d_common import BASE, dump, frozen_checks, new_run, read, sha256, verify_frozen

STATUS = "THREE_CLASS_PILOT_COMPLETED"


def latest(name):
    return Path(read(BASE / name / "latest.json")["run"])


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def main():
    checks = frozen_checks()
    verify_frozen(checks)
    ctx = source_context()  # re-verifies synthetic and N9_01 approval hashes
    out = new_run("report")
    diag_dir, sup_dir = latest("diagnosis"), latest("supervision")
    learn_dir, ft_dir, gap_dir = latest("learnability"), latest("finetune"), latest("domain_gap")
    diag = read(diag_dir / "zero_recall_diagnosis.json")
    sup = read(sup_dir / "synthetic_supervision_verification.json")
    learn = read(learn_dir / "learnability_A_decoder_mask_head.json")
    protocol = read(ft_dir / "training_protocol.json")
    evals = read(ft_dir / "checkpoint_evaluations.json")
    gate = read(ft_dir / "gate_evaluation.json")
    reload_audit = read(ft_dir / "save_reload_audit.json")
    repeat = read(ft_dir / "repeat_run_final_evaluation.json")
    control = read(ft_dir / "control_real_only_evaluation.json")
    info = read(ft_dir / "training_info.json")
    gap = read(gap_dir / "domain_gap_assessment.json")
    qa_dir = BASE / "qa"
    base, final = evals["baseline"], evals["1200"]
    g7_after_fix = (
        reload_audit["weights_reload_counts_identical"]
        and reload_audit["corrected_copy_strict_from_pretrained_identical"]
        and reload_audit["state_dict_mismatches"] == ["criterion.empty_weight"]
    )
    gates = dict(gate["gates"])
    decision = dict(
        status=STATUS,
        candidate=gate["candidate"],
        candidate_corrected_copy=gate["candidate_corrected_copy"],
        gates=gates,
        g7_detail=dict(
            as_originally_saved=False,
            after_loader_fix=g7_after_fix,
            note=(
                "Strict from_pretrained failed on criterion.empty_weight (3 vs 4). Fixed in "
                "load_three_class. Weight reload and the corrected copy reproduce evaluation "
                "counts exactly. The audit's key-name comparison also listed 421 frozen Swin "
                "backbone tensors only because Transformers stores them under legacy on-disk "
                "names; the backbone was frozen and counts match exactly."
            ),
        ),
        all_gates_passed=False,
        failed_gates=[k for k, v in gates.items() if not v],
        promoted=False,
        selected_model="frozen two-class candidate (final_candidate/checkpoint_step0600), unchanged",
        frozen_hashes_verified=True,
    )
    dump(out / "model_checkpoint_decision.json", decision)
    data_req = dict(
        status="REAL_TEXT_TRAINING_DATA_STILL_REQUIRED_FOR_SAFE_THREE_CLASS_MODEL",
        reason=(
            "Synthetic-only text training produced nontrivial N9_01 text recall but also text "
            "on lane paint, arrows and road texture; N9_01 is the only real text evidence and "
            "is development-used. No geographically separate real text training data exists."
        ),
        requirement=dict(
            minimum="several hundred observed text-stroke instances (target >= 30 phrases) "
            "across >= 3 acquisitions not overlapping N9_01, T7, S8 or future mixed test",
            negatives="real arrows, hatching, zebra, stop/give-way bars and worn paint "
            "annotated as non-text in the same crops",
            labels="phrase-level valid-pixel masks; uncertain strokes ignored; no invented "
            "characters",
            sources=[
                "additional client-provided LiDAR road sections",
                "new valid LiDAR acquisitions",
                "approved project sources not yet inspected",
                "explicitly commissioned real text-positive annotations",
            ],
        ),
        prohibited=["N9_01 as training data", "protected T7/S8/mixed test consumption"],
    )
    dump(out / "real_data_requirements.json", data_req)
    for src in (
        ft_dir / "panel_n9_01_text.png",
        ft_dir / "panel_solid_dash_dev.png",
        ft_dir / "panel_synthetic_val.png",
        ft_dir / "n9_01_text_progression.png",
        learn_dir / "tiny_subset_A_decoder_mask_head.png",
    ):
        if src.exists():
            shutil.copy2(src, out / src.name)
    s9c = diag["stage9c_step40"]
    n9c = s9c["n9_01"]
    sv = s9c["synthetic_val_summary"]
    grads = s9c["classifier_only_gradient_check"]
    rows_ckpt = []
    for k in ("baseline", "300", "600", "900", "1200"):
        e = evals[k]
        sd, n9 = e["solid_dash"], e["n9_01"]
        rows_ckpt.append(
            f"| {k} | {f3(sd['longitudinal_solid']['f1'])} | {f3(sd['longitudinal_dash']['f1'])} "
            f"| {pct(n9['text']['recall'])} | {pct(n9['text']['precision'])} "
            f"| {f3(n9['text']['iou'])} | {f3(e['synthetic_train_fit']['pixel']['iou'])} "
            f"| {f3(e['synthetic_val']['pixel']['iou'])} |"
        )

    def sd_row(name, e):
        sd = e["solid_dash"]
        s, d = sd["longitudinal_solid"], sd["longitudinal_dash"]
        return (
            f"| {name} | {f3(s['precision'])} / {f3(s['recall'])} / {f3(s['f1'])} / "
            f"{f3(s['iou'])} | {f3(d['precision'])} / {f3(d['recall'])} / {f3(d['f1'])} / "
            f"{f3(d['iou'])} | {f3(sd['instance_recall']['solid'])} / "
            f"{f3(sd['instance_recall']['dash'])} | {f3(sd['long_line_coverage_mean'])} | "
            f"{f3(sd['dash_separation_rate'])} | {sd['nodata_raw_pixels_of_kept_queries']} | "
            f"{sd['text_predicted_pixels_on_line_dev_crops']} |"
        )

    n9f = final["n9_01"]
    groups = "\n".join(
        f"| {g} | {', '.join(v['statuses'])} | {v['text_pixels']} | {pct(v['recall'])} |"
        for g, v in sorted(n9f["text_groups"].items())
    )
    synp = gap["synthetic_val_profile"]
    n9p = gap["n9_01_profile"]
    g_minor = [v["minor_extent_px"] for v in n9p["groups"].values() if "minor_extent_px" in v]
    g_major = [v["major_extent_px"] for v in n9p["groups"].values() if "major_extent_px" in v]
    g_width = [
        v["stroke_width_closed_px"]
        for v in n9p["groups"].values()
        if v.get("stroke_width_closed_px")
    ]
    probe_n9 = "\n".join(
        f"| N9_01 x{r['factor']} | {pct(r['text']['recall'])} | {pct(r['text']['precision'])} | "
        f"{f3(r['text']['iou'])} | {r['unscored_observed_text_pixels']} |"
        for r in gap["probe_n9_upscaled"]
    )
    probe_syn = "\n".join(
        f"| synthetic val x{r['scale']} / {r['density']} | {pct(r['text']['recall'])} | "
        f"{pct(r['text']['precision'])} | {f3(r['text']['iou'])} | n/a |"
        for r in gap["probe_synthetic"]
    )
    rep = gate["repeat"]
    qa_pytest = (qa_dir / "pytest.log").read_text(encoding="utf-8", errors="replace").strip()
    qa_pytest = qa_pytest.splitlines()[-1]
    md = (
        f"""# Stage 9D — zero text recall diagnosis and actual text mask training

**Final status: `{STATUS}`**

The tiny-subset test showed the network learns text-shaped masks once its mask-generating
components are trainable. One predeclared, bounded three-class run then produced real
N9_01 text detection (recall {pct(n9f['text']['recall'])}, IoU {f3(n9f['text']['iou'])}).
The candidate **fails** the predeclared gates G4 (solid/dash regression), G5 (NoData and text
false positives) and G6 (reproducibility). G7 failed as saved and passes after a loader fix.
The candidate is **rejected**. The frozen two-class model remains selected and unchanged.
No digitization was performed. Protected T7/S8/mixed tests were not touched. N9_01 was
evaluation-only. The synthetic test split was not used.

## 1. Zero-recall root cause (Stage A)

Model inspected: Stage 9C classifier-only step 40. The fixed 9C operating point is
argmax over classes incl. no-object == text, prob x mask quality >= 0.25, area >= 4.

| Question | Evidence | Answer |
|---|---|---|
| Text logits positive? | N9_01: {n9c['queries_positive_text_logit']} of 100 queries have positive text logit; max text prob {f3(n9c['max_text_prob'])}; synthetic val max text prob mean {f3(sv['max_text_prob']['mean'])} | Barely, never dominant |
| Raw queries overlap real text? | {n9c['queries_with_any_overlap']} queries touch N9_01 text; best single query IoU {f3(n9c['best_iou_query']['iou'])}, recall {pct(n9c['best_iou_query']['recall'])}; union of all 100 raw masks covers {pct(n9c['union_all_raw_queries_text_recall'])} | Weakly; no text-shaped query exists |
| Masks rejected by class threshold? | Best-overlap query argmax = class {n9c['best_iou_query']['argmax_class']} (solid); 0 queries argmax text; synthetic val: {sv['best_query_rejections']['class_argmax_not_text']}/47 best queries rejected by class | Yes, but the masks were poor anyway |
| Masks fail mask-quality filter? | Best query quality {f3(n9c['best_iou_query']['quality'])}; score-threshold rejections {sv['best_query_rejections']['score_below_threshold']}/47 | No |
| Only solid/dash-shaped masks? | Best query median component elongation {f3(n9c['best_query_shape']['median_elongation'])} vs text target {f3(n9c['text_target_shape']['median_elongation'])}; synthetic val best-query IoU mean {f3(sv['best_query_iou']['mean'])} | Yes, frozen masks are line/fragment-shaped |
| Real/synthetic normalization differs? | Same processor (rescale 1/255, ImageNet mean/std) and grayscale-to-RGB replication for all inputs | No normalization mismatch; appearance differs (sect. 9) |
| Phrase masks and matching correct? | Every synthetic positive matched one query; but matched query IoU with the text target was {', '.join(f3(g['matched_query_iou']) for g in grads)} | Matching works, but it can only pick from non-text masks |
| Text samples give valid gradients? | Text-row grad norm ~{f3(grads[0]['grad_norm_text_row'])}; mask branch requires_grad = {grads[0]['mask_branch_requires_grad']} | Classifier only; zero mask gradients by design |

**Root cause.** It was not the classifier alone. The frozen mask branch produced no text-shaped query masks.
Classifier-only training therefore taught the text label on arbitrary line/fragment queries
(matched IoU 0.00–0.20). Even a perfect classifier could recover at most
{pct(n9c['union_all_raw_queries_text_recall'])} of N9_01 text from those masks. The text
probability also stayed below 0.07 after 40 steps (10 synthetic). Raw and postprocessed results are in
`zero_recall_diagnosis.json`. No threshold was changed.

## 2. Synthetic supervision verification (Stage B)

Problems found: **{len(sup['problems'])}**. Source hashes verified: {sup['source_hashes_verified']}.
Sources unmodified after run: {sup['sources_unmodified']}.

- Text uses model class {sup['class_mapping']['synthetic_visual_9_to_model']}; no-object index {sup['class_mapping']['no_object_index']}.
- Solid/dash rows are copied unchanged: {sup['head_migration']['solid_dash_rows_identical']}. No-object moved 2->3: {sup['head_migration']['no_object_moved_2_to_3']}.
- Masks align with intensity: train visible paint is brighter than background by a mean of {f3(sup['per_split']['train']['mean_fg_minus_bg_intensity_0_1'])} (min {f3(sup['per_split']['train']['min_fg_minus_bg_intensity_0_1'])}) on a 0–1 scale.
- Phrase identity: one instance per image, holding the whole visible phrase. No character masks were created.
- Letter holes: {sup['per_split']['train']['letter_hole_pixels_valid_background']} train hole pixels are valid background; 0 are marked positive.
- Generator dropout (intended minus visible) is ignored, not negative. That is {pct(sup['per_split']['train']['ignore_fraction_of_intended'])} of intended text pixels in train.
- Criterion: every sampled positive retains text points (min {sup['criterion_checks']['min_text_positive_points']}). All samples matched: {sup['criterion_checks']['all_matched']}. Losses finite: {sup['criterion_checks']['all_finite']}. Loss is invariant to targets in ignored pixels: {sup['criterion_checks']['all_ignore_invariant']}.
- Structural note: {pct(sup['criterion_checks']['synthetic_fully_valid_fraction'])} of synthetic and {pct(sup['real_train_crops']['fully_valid_fraction'])} of inspected real crops are fully valid. Unmatched-query no-object CE is therefore never applied on these images, and false positives are suppressed only by the null-mask loss.

## 3. Tiny-subset learnability test (Stage C) — PASSED

This is a software diagnostic on 8 fixed synthetic train images, evaluated on the same 8 images.
It is **not** evidence of generalization.
Variant A trains the query decoder, mask embedder and classifier; the backbone and pixel decoder are frozen.
It ran 240 steps.

| Step | Text pixel IoU | Precision | Recall | Best-query IoU | Max text prob |
|---|---:|---:|---:|---:|---:|
"""
        + "\n".join(
            f"| {k} | {f3(v['pixel']['iou'])} | {pct(v['pixel']['precision'])} | "
            f"{pct(v['pixel']['recall'])} | {f3(v['best_query_iou_mean'])} | {f3(v['max_text_prob_mean'])} |"
            for k, v in learn["evaluations"].items()
        )
        + f"""

Gate checks: {', '.join(f'{k}={v}' for k, v in learn['gate_checks'].items())}.
Variant B (adds the pixel decoder) was not needed.

## 4. Trainable-component and gradient audit

| Group | Tensors | Params | Trainable params (Stage D) |
|---|---:|---:|---:|
"""
        + "\n".join(
            f"| {g} | {r['tensors']} | {r['params']:,} | {r['trainable_params']:,} |"
            for g, r in info["trainable"].items()
        )
        + f"""

The optimizer holds exactly the requires_grad set: {info['optimizer']['optimizer_equals_requires_grad']}.
Per-group learning rates are {protocol['learning_rates']}. At steps 1–3 and every 300 steps, gradients were finite and nonzero
for all trainable groups and absent for frozen groups (`gradient_audit_run1.json`).
Freezing was set explicitly by parameter-name prefix, not by a config flag.

## 5. Predeclared protocol (Stage D)

`training_protocol.json` was written before the first optimizer step.

- Warm start from the frozen candidate (SHA-256 `{protocol['warm_start']['sha256'][:16]}…`) with the verified three-class migration.
- {protocol['steps']} steps, mix {protocol['schedule_mix']}: 2:1 real rehearsal, all 65 approved real train crops, all 400 synthetic train images (380 positive, 20 negative), about one synthetic epoch.
- Evaluations at steps {protocol['intermediate_evaluations']}. The primary candidate is the final step; there was no checkpoint selection on N9_01.
- The budget follows from Stage C (IoU 0.60 by step 60 at LR 1e-4), with half that LR used here.
- The domain adaptation was none. N9_01 was not trained on. The synthetic test split and protected tests were not used.

Run time was {info['elapsed_seconds']:.0f} s including the 4 evaluations. Peak GPU allocation was {info['gpu_peak_allocated_bytes'] / 2**20:.0f} MiB.

## 6. Bounded training results

| Checkpoint | Solid F1 | Dash F1 | N9_01 text recall | N9_01 text precision* | N9_01 text IoU | Synthetic train-fit IoU | Synthetic val IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows_ckpt)}

\\*N9_01 precision is scored only against reviewed text and observed lane paint. Other observed
pixels are unscored (see section 8).

## 7. Solid/dash regression (Stage E)

Same 14 development crops and unchanged definitions. Each cell is precision / recall / F1 / IoU.

| Model | Solid P/R/F1/IoU | Dash P/R/F1/IoU | Instance recall S/D | Long-line coverage | Dash separation | Kept-query NoData px | Text px on line crops |
|---|---|---|---|---|---|---:|---:|
{sd_row('Frozen baseline', base)}
{sd_row('Stage 9D step 1200 (candidate)', final)}
{sd_row('Repeat run (same seed)', dict(solid_dash=repeat['solid_dash']))}
{sd_row('Control: real-only, same recipe (not a candidate)', control['evaluation'])}

Cross-class pixels at step 1200 were {final['solid_dash']['cross_class_pixels']}. The baseline had {base['solid_dash']['cross_class_pixels']}.

**Interpretation.** Dash precision collapsed from {f3(base['solid_dash']['longitudinal_dash']['precision'])}
to {f3(final['solid_dash']['longitudinal_dash']['precision'])}: dash false positives appeared on kerbs and
vegetation. Solid recall rose but solid precision fell. The real-only control also loses most dash F1
({f3(control['evaluation']['solid_dash']['longitudinal_dash']['f1'])}). The main damage therefore comes from
**fine-tuning the transformer decoder on the 65 real crops under this criterion**: no-object CE
is never applied on partially valid crops. It is not specific to adding text. Adding text worsens dash
further and introduces text predictions on line crops.
Class-row preservation did not preserve performance.

## 8. Synthetic-to-real text transfer (Stage F)

1. **Synthetic training fit** (24 seen train images): IoU {f3(final['synthetic_train_fit']['pixel']['iou'])}.
2. **Synthetic held-out val** (47 positive + 3 negative): IoU {f3(final['synthetic_val']['pixel']['iou'])}, precision {pct(final['synthetic_val']['pixel']['precision'])}, recall {pct(final['synthetic_val']['pixel']['recall'])}. Negative-image text FP pixels: {final['synthetic_val']['negative_image_text_fp_pixels']}. Phrases are shared across splits, so this is style-level, not phrase-level, held-out evidence.
3. **Real N9_01 development** (step 1200, fixed operating point):

| Metric | Value |
|---|---|
| Text-pixel precision / recall / IoU | {pct(n9f['text']['precision'])} / {pct(n9f['text']['recall'])} / {f3(n9f['text']['iou'])} |
| TP / FP (on lane paint) / FN pixels | {n9f['text']['tp']} / {n9f['text']['fp']} / {n9f['text']['fn']} |
| Reviewed lane-paint pixels predicted as text | {n9f['text_on_lane_paint_pixels']} of {n9f['scored_lane_negative_pixels']} ({pct(n9f['text_on_lane_paint_pixels'] / n9f['scored_lane_negative_pixels'])}) |
| Text predicted on unscored observed pixels | {n9f['predicted_text_unscored_observed_pixels']} px; {n9f['false_positive_text_regions_away_from_text']['count']} regions / {n9f['false_positive_text_regions_away_from_text']['pixels']} px lie more than 6 px from reviewed text (arrow, road texture) |
| Stroke components hit / missed | {pct(n9f['stroke_component_recall'])} hit; {n9f['missed_stroke_components']} of {n9f['stroke_components']} missed |
| Kept text queries | {n9f['text_queries_kept']}; class conf {', '.join(f3(x) for x in n9f['text_query_class_confidence'])}; mask quality {', '.join(f3(x) for x in n9f['text_query_mask_quality'])} |
| Text on NoData | {n9f['predicted_text_on_nodata']} px |
| Recognized paint within valid observations | {n9f['recognized_paint_within_valid_observations']} px |

Recall per approved text group (pixels inside each group's bounding box; uncertain strokes remain ignored, and no character values were inferred):

| Group | Review status | Text px | Recall |
|---|---|---:|---:|
{groups}

N9_01 is development-used. Its source-reference provenance is user-attested, not verified, so
this is **not** independent generalization evidence.

## 9. Real/synthetic domain gap (Stage G)

| Property | Synthetic val (median) | N9_01 |
|---|---|---|
| Paint / background intensity | {synp['fg']['median']:.0f} / {synp['bg']['median']:.0f} | {n9p['fg']:.0f} / {n9p['bg']:.0f} |
| Background std | {synp['bg_std']['median']:.0f} | {n9p['bg_std']:.0f} |
| Local contrast (paint − 4 px ring) | {synp['local_contrast']['median']:.0f} | {n9p['local_contrast']:.0f} |
| Stroke width (px) | {synp['stroke_width_intended_px']['median']:.1f} (intended) | {min(g_width):.1f}–{max(g_width):.1f} (closed observed) |
| Phrase short / long extent (px) | {synp['phrase_minor_extent_px']['median']:.0f} / {synp['phrase_major_extent_px']['median']:.0f} | {min(g_minor):.0f}–{max(g_minor):.0f} / {min(g_major):.0f}–{max(g_major):.0f} (per group) |
| Observed fraction inside text | {pct(synp['observation_density_in_text']['median'])} (diagonal scan hatching) | {pct(n9p['observation_density_in_text_footprint'])} (point grid) |
| Observed background | {pct(synp['observation_density_background']['median'])} | {pct(n9p['observation_density_tile'])} of tile |
| Components per 1000 paint px | {synp['components_per_1000_fg']['median']:.0f} | {n9p['components_per_1000_fg']:.0f} |
| Orientation | mixed (median {synp['orientation_deg']['median']:.0f}°) | near-axis-aligned |
| Text area fraction of crop | {pct(synp['text_area_fraction']['median'])} | small groups |

The synthetic samples carry no geospatial coordinates or GSD, so scale is compared in pixels only.

Predeclared probe on the candidate. It is analysis only and was not used for selection or thresholds:

| Input | Recall | Precision | IoU | Unscored observed text px |
|---|---:|---:|---:|---:|
{probe_n9}
{probe_syn}

At x4 (a 2048 px input) the model kept no queries at all, which is out of range for this model.

**Assessment.** Synthetic phrases are about 2–3x larger than N9_01 groups, with 2–3x wider
strokes, about 3.5x lower background intensity and a different dropout texture. Despite this, the candidate
detects N9_01 text at native scale. Upscaling N9_01 2x helps modestly, so scale is a contributor
but not the blocker. Synthetic downscaling mainly costs precision. The dominant real failure is
**specificity**: lane paint, an arrow and textured road surface become text. Synthetic data has no
real lane paint, arrows, hatching or worn markings as text negatives, and the criterion
never supervises no-object on partially valid images. The evidence points to a lack of real hard
negatives and decoder over-plasticity, not to an intensity-normalization fault.
No extra synthetic data was generated.

## 10. Model selection (Stage I)

| Gate | Definition | Result |
|---|---|---|
"""
        + "\n".join(
            f"| {k} | {protocol['acceptance_gates'][next(g for g in protocol['acceptance_gates'] if g.startswith(k + '_'))]} | {'PASS' if v else ('FAIL as saved; PASS after loader fix' if k == 'G7' and g7_after_fix else 'FAIL')} |"
            for k, v in gates.items()
        )
        + f"""

- G4 drops (absolute; positive = worse than baseline): {', '.join(f'{k} {v:+.3f}' for k, v in gate['gate_detail']['drops'].items())}.
- G6 repeat (run 1 vs run 2): N9_01 IoU {f3(rep['n9_iou'][0])} vs {f3(rep['n9_iou'][1])}; solid F1 {f3(rep['solid_f1'][0])} vs {f3(rep['solid_f1'][1])}; dash F1 {f3(rep['dash_f1'][0])} vs {f3(rep['dash_f1'][1])}. GPU nondeterminism exceeds the 0.02 tolerance.
- G7: {decision['g7_detail']['note']} After the fix, reload is exact: {g7_after_fix}.

**Decision: reject.** The candidate is not promoted, and the frozen two-class model remains selected.
The candidate weights are kept for audit only (`{Path(gate['candidate_corrected_copy']).name}`).
Hashes of the preserved checkpoints and of `stitching.py` were re-verified after every script.

## 11. Real-data requirements (Stage H)

{data_req['reason']}

- Minimum: {data_req['requirement']['minimum']}.
- Negatives: {data_req['requirement']['negatives']}.
- Labels: {data_req['requirement']['labels']}.
- Sources: {'; '.join(data_req['requirement']['sources'])}.

Reliable real-text recognition is **not yet achievable** without such data. Synthetic-only iteration should
stop here. The source search was not repeated.

**Evidence-backed next experiment, only once real negatives and real text exist:**

- Keep solid/dash behaviour anchored, either by output distillation from the frozen model on real crops or by lower-plasticity adaptation (for example, freezing the original queries and adding text-only queries).
- Add real non-text hard negatives, including arrows, to the text loss.

## 12. QA

- pytest (activated environment): `{qa_pytest}`. Running python directly without activation gave 1 environment-only failure (GDAL/pyarrow DLL path) and a pdal skip.
- Ruff `src tests scripts`: {(qa_dir / 'ruff.log').read_text(encoding='utf-8').strip()}. Repo-wide findings exist only in the supplied `road_markings_synthetic_*` generator scripts, which were left unmodified.
- Black `--check src tests scripts`: {(qa_dir / 'black.log').read_text(encoding='utf-8').strip().splitlines()[-1]}
- Healthcheck (activated environment): 9/9 PASS.

## Artifacts

- Diagnosis: `{diag_dir}`
- Supervision: `{sup_dir}`
- Learnability: `{learn_dir}`
- Fine-tune, gates, panels: `{ft_dir}`
- Domain gap: `{gap_dir}`

Nothing was committed.
"""
    )
    (out / "report.md").write_text(md, encoding="utf-8")
    manifest = {str(p.relative_to(out)): sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
    dump(out / "package_manifest.json", manifest)
    verify_frozen(checks)
    assert ctx is not None
    dump(BASE / "latest.json", {"run": str(out), "status": STATUS})
    print(out)


if __name__ == "__main__":
    main()
