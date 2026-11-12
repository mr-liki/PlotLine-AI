"""Group-safe split proposals, provenance qualifications and no-run experiment plan."""

import hashlib
from collections import defaultdict

from phase5_11_common import EDGE, MERGE, ROOT, load
from roadmarking.dataset.coverage import solid_shape
from roadmarking.rasterization.pipeline import dump
from stage9_common import context, identity_group, out_dir


def main():
    out = out_dir()
    ctx = context()
    audit = load(out / "physical_line_audit.json")
    inventory_path = out / "solid_acquisition_inventory.json"
    inventory = load(inventory_path)
    for row in inventory["per_id_acquisition"]:
        feature = ctx["by_id"][row["global_id"]]
        shape, _, curvature = solid_shape(feature["centerline"])
        row.update(
            {
                "shape": shape,
                "curvature_degrees_per_50_native": float(curvature),
                "edge_line": feature["designation"] in EDGE,
                "merge_line": feature["designation"] in MERGE,
            }
        )
    dump(inventory_path, inventory)
    training = [r for r in ctx["used"] if r["role"].startswith("train")]
    trained_groups = {identity_group(ctx, uid) for r in training for uid in r["ids"]}
    old_dev = load(ROOT / "outputs/reports/stage8/8_4_dev_paint.json")["per_id"]
    safe, excluded = [], []
    for row in old_dev:
        target = {**row, "physical_group": identity_group(ctx, row["global_id"])}
        (excluded if target["physical_group"] in trained_groups else safe).append(target)
    train_stats = load(ROOT / "outputs/reports/stage8/8_4_paint_statistics.json")["stage6"][
        "per_id"
    ]
    for group in audit["exact_three_groups"]:
        group["observed_training_pixel_evidence"] = [
            r
            for r in train_stats
            if r["split"] == "train"
            and r["global_id"] in group["member_ids"]
            and r["observed_paint_pixels"] > 0
        ]
        group["observed_development_pixel_evidence"] = [
            r
            for r in old_dev
            if r["global_id"] in group["member_ids"] and r["observed_paint_pixels"] > 0
        ]
    manifest_index = defaultdict(list)
    for path in (ROOT / "data/processed/phase5_training").glob("*/metadata/dataset_manifest.json"):
        manifest_index[hashlib.sha256(path.read_bytes()).hexdigest()].append(path)
    model_provenance = []
    for name, folder in (
        (
            "original_default",
            ROOT
            / "models/mask2former_swin_s/phase5_7_expanded_20261106T111240383727Z/checkpoint_step0050",  # noqa: E501
        ),
        (
            "frozen_candidate",
            ROOT / "models/mask2former_swin_s/final_candidate/checkpoint_step0600",
        ),
    ):
        bundle = load(folder / "bundle.json")
        paths = manifest_index[bundle["dataset_sha256"]]
        ids = set()
        for path in paths:
            ids.update(
                uid
                for c in load(path)["crops"]
                if c["split"] == "train"
                for uid in c["global_instance_ids"]
            )
        model_provenance.append(
            {
                "model": name,
                "checkpoint": str(folder),
                "sealed_dataset_sha256": bundle["dataset_sha256"],
                "matching_manifest_paths": [str(p) for p in paths],
                "overlap_groups_in_own_training_manifest": [
                    {
                        "reported_group": g["reported_group"],
                        "training_target_ids": sorted(ids & set(g["member_ids"])),
                    }
                    for g in audit["exact_three_groups"]
                ],
                "limitation": "Manifest IDs establish target exposure; historical scores are not rescored. Parent/warm-start exposure may add dependencies and is not erased by later reassignment.",  # noqa: E501
            }
        )
    audit["model_training_provenance"] = model_provenance
    dump(out / "physical_line_audit.json", audit)
    spec_path = out / "new_region_candidates.json"
    spec = load(spec_path) if spec_path.exists() else {"tiles": []}
    dev_tiles = [t for t in spec["tiles"] if t["proposed_role"] == "development_candidate"]
    train_tiles = [t for t in spec["tiles"] if t["proposed_role"] == "training_candidate"]
    proposal = {
        "status": "PROPOSAL_ONLY_NO_ROLE_OR_APPROVAL_CHANGES",
        "historical_dev_group_safe_target_subset": safe,
        "historical_dev_dependent_targets_to_qualify_or_exclude_in_future_protocol": excluded,
        "excluded_target_policy": "Exclude dependent paint plus declared uncertainty margin from any new unseen-group scoring; never verified negative background. Preserve all historical scores and geometry.",  # noqa: E501
        "new_development_regions_pending_first_review": [t["candidate_id"] for t in dev_tiles],
        "new_training_regions_pending_first_review": [t["candidate_id"] for t in train_tiles],
        "new_roles_reserved_before_model_evaluation": True,
        "physical_and_geographic_guards": "No shared physical/marking groups across proposed train/development. At least 500 native units between opposite roles and from all protected/historical footprints; no footprint overlap with any existing approved region.",  # noqa: E501
        "sampling_unit": "Road section and physical marking group; non-overlapping or overlapping crops are exposures, not independent validation samples.",  # noqa: E501
        "alternative_reassignment_not_implemented": {
            "proposal": "If NEW_POSITIVE_DEV is to be re-established as unseen-physical-line development, retire all three connected groups from future training datasets and all acquisitions of those groups.",  # noqa: E501
            "affected_approved_training_regions": sorted(
                {
                    r["region"]
                    for g in audit["exact_three_groups"]
                    for m in g["members"]
                    for r in m["regions"]
                    if r["role"].startswith("train")
                }
            ),
            "implication": "Changing dataset roles cannot remove memorization from existing weights. Requires explicit approval and a future clean-origin training protocol. Frozen/default models and original approvals remain immutable.",  # noqa: E501
            "applied": False,
        },
        "readiness": "New review data is promising but unapproved; three new development solid groups do not establish broad solid/merge/faded or unseen-acquisition reliability. Existing clean historical development remains repeatedly used selection evidence, not a fresh unbiased test.",  # noqa: E501
    }
    dump(out / "split_proposal.json", proposal)
    dump(
        out / "recognition_readiness.json",
        {
            "dimensions": [
                {
                    "claim": "real painted-pixel recognition",
                    "evidence": "Approved observed-paint masks; local T7 correction preserved; class/source pixel exposure tables and new candidate support",  # noqa: E501
                    "unsupported": "Recognition reliability itself without adequate independent accepted validation and uncertainty bounds; no new performance computed",  # noqa: E501
                },
                {
                    "claim": "complete solid marking recognition",
                    "evidence": "S8 protects 5 client IDs in 4 physical groups; new dev proposes 3 additional groups",  # noqa: E501
                    "unsupported": "Complete physical-line endpoints/length through NoData, broad road-edge/merge diversity; S8 is small and remains untouched",  # noqa: E501
                },
                {
                    "claim": "individual dash recognition",
                    "evidence": "T7 protects 58 dash identities; all new development windows have observed dash paint",  # noqa: E501
                    "unsupported": "Universal unfamiliar/faded dash recall; T7 has not been consumed",  # noqa: E501
                },
                {
                    "claim": "solid-line continuity",
                    "evidence": "Client geometry continuation graph and prior stitching audit",
                    "unsupported": "True continuity across unobserved gaps; no NoData bridge may count as accepted recognition",  # noqa: E501
                },
                {
                    "claim": "thin and faded paint",
                    "evidence": "Client width and candidate normalized paint intensity/contrast measured",  # noqa: E501
                    "unsupported": "Width alone is not visibility, and intensity proxies are not approved faded-paint strata; first review required",  # noqa: E501
                },
                {
                    "claim": "false positives on observed non-road features",
                    "evidence": "Existing sparse control and candidate hard-negative review pixels",
                    "unsupported": "Complete validated non-road negatives and roadside structure taxonomy; unannotated paint/ambiguous symbols cannot become background",  # noqa: E501
                },
                {
                    "claim": "cross-acquisition generalization",
                    "evidence": "Resolved identity-specific support in both July acquisitions and September",  # noqa: E501
                    "unsupported": "Acquisition does not imply geography; all September supported groups repeat July groups. New Sept training proposal would remove it as an unseen condition; current protected cohorts are from trained July acquisition",  # noqa: E501
                },
            ],
            "evidence_populations": {
                "development": "Historical/selection-used; NEW_POSITIVE_DEV dependencies explicitly qualified",  # noqa: E501
                "historically_consumed": "FUTURE_TEST_1/2/3 and NEW_FINAL_TEST_RESERVED (consumed once); no fresh-test claim",  # noqa: E501
                "protected_unseen_physical_targets": "T7 dash IDs and S8 restricted unseen-solid groups, exact Stage-8B exclusions unchanged",  # noqa: E501
                "protected_geography": "Ten T7/S8 footprints preserved; not equivalent to unseen acquisition or whole-image unseen identity",  # noqa: E501
                "unseen_acquisition_conditions": "September not a new physical geography; no approved independent acquisition-held-out evaluation protocol yet",  # noqa: E501
            },
            "dependable_recognition_established": False,
        },
    )
    dump(
        out / "next_experiment_proposal.json",
        {
            "status": "PROPOSAL_ONLY_DO_NOT_TRAIN",
            "entry_conditions": [
                "First geometry and complete-visible-marking approval for only the new regions",
                "Seal group/role membership, whole-acquisition repetitions and observed-support/ignore rules",  # noqa: E501
                "Resolve any ambiguous continuation critical to the chosen split",
                "Predeclare sufficient independent solid/dash group coverage and numerical reliability targets before execution",  # noqa: E501
            ],
            "architecture": "Mask2Former Swin-S unchanged",
            "reference": "Existing frozen candidate with immutable weights, thresholds and stitching; original default preserved",  # noqa: E501
            "controlled_change": "Real-data expansion using newly approved group-safe regions, emphasizing observed solid exposure and unfamiliar dash positives; compare against frozen reference on group-safe development only",  # noqa: E501
            "seeds": [895, 896, 897],
            "maximum_steps_per_seed": 400,
            "selection_rule": "Evaluate at 200 and 400 only, choose step 200 unless the seed-mean equal-region/equal-class predeclared development F1 improves at 400 with no loss in either class recall or safety gates; tie -> 200. No step-600 default and no protected-test selection.",  # noqa: E501
            "exposure_accounting": "Log observed real solid/dash paint pixels per physical group and acquisition; cap dominant long-line group exposures; raw crop count never defines effective sample size. No synthetic mixing or self-training.",  # noqa: E501
            "safety": "Fixed frozen confidence thresholds, NMS and production stitching SHA; retain unsupported-NoData connection rejection in reporting; no score recalibration. Measure score-distribution drift on development only without tuning thresholds in this bounded experiment.",  # noqa: E501
            "reliability_protocol": "Report per-class pixel precision/recall, individual-instance recall, complete-observed-support solid coverage, continuity/NoData errors, thin/faded strata after review, and observed non-road false positives. Use region/physical-group dependence-aware uncertainty; do not bootstrap overlapping crops as independent.",  # noqa: E501
            "calibration_limit": "If fixed-threshold recall or precision fails, stop with NO_IMPROVEMENT rather than tuning on test or silently recalibrating; any calibration study requires a separately predeclared dev-only protocol.",  # noqa: E501
            "prohibited": [
                "repeat Stage-7 2x2 without new evidence",
                "test inference in Stage 9",
                "model promotion",
                "digitization",
            ],
            "future_execution_authorized": False,
        },
    )
    print(
        f"Split proposal: {len(safe)} historical group-safe targets; {len(excluded)} dependent targets; {len(dev_tiles)} new dev / {len(train_tiles)} new training regions pending first review"  # noqa: E501
    )


if __name__ == "__main__":
    main()
