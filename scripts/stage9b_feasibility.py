"""Record three-class preparation and explicit blockers; never open model weights."""

from pathlib import Path

from phase5_11_common import ROOT, load
from roadmarking.rasterization.pipeline import dump, sha256
from roadmarking.training.road_text_head import three_class_config


def main():
    out = Path(load(ROOT / "outputs/reports/stage9b/latest.json")["report_dir"])
    source = (
        ROOT / "models/mask2former_swin_s/final_candidate/checkpoint_step0600/model/config.json"
    )
    proposed = three_class_config(load(source))
    dump(out / "proposed_model_config.json", proposed)
    dump(
        out / "recognition_class_specification.json",
        {
            "version": "stage9b-v1",
            "status": "DRAFT_DATA_REVIEW_REQUIRED",
            "foreground": [
                {"model_id": 0, "name": "longitudinal_solid", "visual_id": 1},
                {"model_id": 1, "name": "longitudinal_dash", "visual_id": 2},
                {"model_id": 2, "name": "road_text", "visual_id": 9},
            ],
            "no_object_logit": 3,
            "legacy_no_object_logit_unchanged": 2,
            "config_source": str(source),
            "config_source_sha256": sha256(source),
            "unsupported_visual_ids": "All except 1,2,9. In particular visual 3 (arrow) must not become model 2 via subtraction.",  # noqa: E501
            "active_models_modified": False,
        },
    )
    dump(
        out / "text_annotation_specification.json",
        {
            "available_original_requirements_evidence": [
                "docs/architecture.md other_marks schema",
                "configs/classes.yaml inscriptions",
                "configs/phase4_taxonomy.yaml character fields",
                "Original client Other Marks field schema LETRA/ALGARISMO/ID_GRUPO",
            ],
            "authoritative_catalog_document_found": False,
            "client_text_features_found": 0,
            "instance_unit": "Individual letter/digit when observed paint supports reliable separation; preserve holes and disconnected observed fragments in one identity.",  # noqa: E501
            "required_fields": [
                "instance_id",
                "text_group_id",
                "character_value or null",
                "character_value_status",
                "client_global_id or null",
                "source_file",
                "native_affine",
                "observed_pixels",
                "review_status",
            ],
            "character_values": "Only human-confirmed literal letter/digit values. Unknown remains null; road_text classification is not OCR.",  # noqa: E501
            "inseparable_or_partial_text": "Group-fragment review candidates cannot be exported as individual-character GT until partition/identity is confirmed. Boundary-clipped glyphs remain explicit partials; do not invent outside paint.",  # noqa: E501
            "unconfirmed_region_policy": "Observed unknown text extent ignored, NoData excluded. N9_01 and other Stage-9 regions remain unapproved for supervision until new text/unknown issues and complete visible-target coverage are reviewed.",  # noqa: E501
            "scope": "Review annotations only; no production vectors, 3D, transcription engine or final GIS deliverable.",  # noqa: E501
        },
    )
    dump(
        out / "three_class_feasibility.json",
        {
            "status": "BLOCKED_BY_UNAPPROVED_TEXT_GEOMETRY_AND_NO_CONFIDENT_TEXT_POSITIVE_TRAINING_EXAMPLES",  # noqa: E501
            "architecture": "Mask2Former Swin-S unchanged",
            "training_run": False,
            "tested_preparation": [
                "visual-to-model explicit mapping 1/2/9 -> 0/1/2; unsupported arrows rejected",
                "new classification head preserves solid/dash rows and moves old no-object 2 -> 3; new text row seeded",  # noqa: E501
                "source head and RNG unchanged by utility",
                "three-class Hungarian matching and valid-only losses; NoData-invariant costs/matches/gradients",  # noqa: E501
            ],
            "installed_transformers_structure": "Mask2FormerForUniversalSegmentation.class_predictor = Linear(config.hidden_dim, config.num_labels + 1), verified from installed source. No full model constructed or forward pass used.",  # noqa: E501
            "future_warm_start_recipe": "Only after approval, construct a NEW model using proposed_model_config.json; load compatible non-head weights from an immutable two-class source; install make_three_class_head result from a separately loaded copy. Reconstruct model internal criterion with num_labels=3 and use external ValidMask2FormerCriterion(num_classes=3) for reviewed valid-only losses. Save config, initialization seed/source SHA and optimizer/RNG state in a NEW unpromoted checkpoint directory. Never reinterpret old no-object row as text.",  # noqa: E501
            "future_coco_recipe": "If initializing from pretrained COCO, transfer compatible non-head weights and initialize the whole four-row classification head reproducibly; COCO foreground rows are not solid/dash/text. Do not call the road-head adapter with COCO labels.",  # noqa: E501
            "remaining_integration_gates": [
                "Two-class completeness approvals do not automatically certify text coverage; confirm new text-target coverage without repeating accepted line geometry review.",  # noqa: E501
                "Check character count against num_queries and ensure sampling retains every approved character's valid pixels; fail closed rather than dropping tiny glyphs.",  # noqa: E501
                "Copied solid/dash classifier rows do not preserve their probabilities after softmax gains a text competitor; validate existing-class precision/recall and calibration stability without silent threshold retuning.",  # noqa: E501
                "Actual approved, supported text-positive training instances outside development/protected word groups",  # noqa: E501
                "Dataset version with explicit three-class conversion and completeness/uncertainty masks; legacy visual_id-1 conversion forbidden",  # noqa: E501
                "Versioned three-class evaluation/training driver: legacy pilot/full_metrics paths are two-class and must not silently drop text",  # noqa: E501
                "Separate text-positive independent test population; T7/S8 remain unchanged line/dash cohorts",  # noqa: E501
                "Human-approved class/region/group exposure budget and numerical reliability targets",  # noqa: E501
            ],
            "future_experiment_proposal": {
                "reference": "Frozen two-class candidate unchanged",
                "seeds": [895, 896, 897],
                "maximum_steps": 400,
                "checkpoint_selection": "Predeclared development-only checks at 200/400; conservative 200 default; all three classes and dependence-aware uncertainty, no protected-test selection",  # noqa: E501
                "class_exposure": "Balance actual observed solid/dash/text pixels and character/physical/word groups; cap repeated words, adjacent crops and repeated acquisition observations",  # noqa: E501
                "validity": "Valid-only matching/loss; unknown glyph boundaries, unsupported arrows/symbols and NoData excluded, never negative paint",  # noqa: E501
                "existing_class_regression": "Compare solid/dash pixel and instance precision/recall against frozen reference on group-safe development; no text gain may hide reduced solid/dash accuracy",  # noqa: E501
                "text_metrics": "Road_text paint precision/recall and per-character-instance IoU/recall separately; literal character/OCR accuracy only for human-confirmed values and an independently implemented/tested recognizer",  # noqa: E501
                "calibration": "Preserve fixed 0/1 reference thresholds and stitching. Predeclare the new text operating-point rule before future execution; no calibration or threshold tuning in Stage 9B",  # noqa: E501
                "promotion": "Never automatic",
            },
        },
    )
    print(
        "Saved inactive three-class configuration, character schema, tested adaptation recipe and explicit training blockers"  # noqa: E501
    )


if __name__ == "__main__":
    main()
