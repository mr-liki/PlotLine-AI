"""Validate and seal road-text review preparation; never enable training."""

import re
import shutil
from itertools import combinations
from pathlib import Path

import numpy as np
import pyogrio
import rasterio
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import box

from phase5_11_common import ROOT, load
from roadmarking.rasterization.pipeline import dump, sha256


def main():
    base = ROOT / "outputs/reports/stage9b"
    latest = load(base / "latest.json")
    out = Path(latest["report_dir"])
    data = Path(latest["data_dir"])
    baseline = load(out / "preservation_baseline.json")["sha256"]
    assert all(sha256(path) == value for path, value in baseline.items())
    approval = load(ROOT / "outputs/reports/stage8b/approval_manifest.json")
    assert all(sha256(path) == value for path, value in approval["sha256"].items())
    observed_hashes = {
        r["region"]: r["observed_raster_sha256"]
        for r in load(ROOT / "outputs/reports/stage8b/source_integrity_and_coverage.json")[
            "regions"
        ]
    }
    for manifest in (
        ROOT / "outputs/reports/stage7_new_test/20261107T051050944913Z/new_region_candidates.json",
        Path(load(ROOT / "outputs/reports/stage8_new_solid_review/latest.json")["report_dir"])
        / "new_region_candidates.json",
    ):
        for tile in load(manifest)["tiles"]:
            assert sha256(tile["layers"]["observed"]) == observed_hashes[tile["candidate_id"]]
    ai = load(ROOT / "outputs/reports/stage8b/ai_preservation.json")
    assert all(sha256(r["path"]) == r["expected_sha256"] for r in ai["checks"])
    stage9 = Path(load(ROOT / "outputs/reports/stage9/latest.json")["report_dir"])
    original = load(stage9 / "new_region_candidates.json")
    old_assets = load(stage9 / "review_assets_manifest.json")["files"]
    assert all(sha256(r["path"]) == r["sha256"] for r in old_assets)
    spec = load(out / "review_spec.json")
    assert sha256(spec["base_stage9_candidates"]) == sha256(original["candidate_store"])
    audit = load(out / "four_region_coverage_audit.json")
    assert len(audit["regions"]) == 4
    assert all(r["unobserved_paint_pixels_exported"] == 0 for r in audit["regions"])
    store = spec["text_candidates"]
    chars = pyogrio.read_dataframe(store, layer="character_candidates")
    fragments = pyogrio.read_dataframe(store, layer="group_and_unknown_fragments")
    assert chars["character_value"].isna().all() and chars["client_global_id"].isna().all()
    assert set(chars["model_id"]) == {2} and set(chars["visual_id"]) == {9}
    assert fragments["model_id"].isna().all()
    assert not chars["training_eligible"].any()
    for t in original["tiles"]:
        rid = t["candidate_id"]
        with rasterio.open(t["layers"]["observed"]) as ds:
            observed = ds.read(1).astype(bool)
        with np.load(out / f"{rid}_review_safety.npz") as masks:
            assert np.array_equal(masks["observed_support"], observed)
            assert not (masks["unconfirmed_review_extent"] & ~observed).any()
        for frame in (chars, fragments):
            for _, row in frame[frame.candidate_id == rid].iterrows():
                mask = rasterize(
                    [(row.geometry, 1)],
                    out_shape=observed.shape,
                    transform=Affine(*t["affine"]),
                    dtype="uint8",
                ).astype(bool)
                assert not (mask & ~observed).any()
                assert int(mask.sum()) == row["observed_pixels"]
                assert abs(row.geometry.area - mask.sum() * 0.05**2) < 1e-8
    search = load(out / "text_training_search.json")
    assert all(
        r["full_source_integrity_passed"]
        and r["actual_points"] == r["expected_points"]
        and not r["dropped_points"]
        for r in search["source_integrity"]
    )
    dev = [
        box(*t["bounds"])
        for t in original["tiles"]
        if t["proposed_role"] == "development_candidate"
    ]
    for t in search["source_backed_search_windows"]:
        assert t["distance_to_development_protected_historical_native"] >= 500
        assert all(box(*t["bounds"]).distance(g) >= 500 for g in dev)
        t["text_visibility"] = "NO_CONFIDENT_TEXT_POSITIVE_IDENTIFIED_ON_VISUAL_AUDIT"
        t["text_positive_training_candidate"] = False
    for a, b in combinations(search["source_backed_search_windows"], 2):
        assert box(*a["bounds"]).distance(box(*b["bounds"])) >= 500
    search["visual_inspection_outcome"] = {
        "cached_training_windows_inspected": 46,
        "new_source_backed_windows_inspected": len(search["source_backed_search_windows"]),
        "confident_text_positive_training_regions": 0,
        "candidate_training_text_annotations_approved": 0,
        "scope_limit": "Bounded visual audit of 35 phase5_11/12/stage6 caches, 11 additional legacy training caches, and eight new source-backed views. Does not prove all available LiDAR lacks road text. Generic arrows, lane lines, posts or texture are not invented as text examples.",  # noqa: E501
    }
    dump(out / "text_training_search.json", search)
    qgis = load(out / "qgis_validation.json")
    assert qgis["all_layers_valid"] and qgis["existing_base_layers_read_only"]
    validation = {
        "all_passed": True,
        "original_approval_hashes_preserved": len(approval["sha256"]),
        "original_stage9_review_assets_preserved": len(old_assets),
        "models_and_stitching_match_sealed_references": True,
        "base_solid_dash_geometry_byte_identical": True,
        "candidate_geometry_rasterizes_to_exact_observed_pixel_counts": True,
        "character_values_or_client_ids_invented": False,
        "unsupported_fragments_mapped_to_text": False,
        "search_training_vs_development_guard_ge_500_native": True,
        "supervised_export_or_training_authorized": False,
    }
    dump(out / "artifact_validation.json", validation)
    for filename in (
        "search.log",
        "annotations.log",
        "qgis.log",
        "feasibility.log",
        "head_tests.log",
        "pytest.log",
        "healthcheck.log",
        "healthcheck.json",
        "ruff.log",
        "black.log",
    ):
        if (base / filename).exists():
            shutil.copyfile(base / filename, out / filename)
    tests = re.search(
        r"(\d+) passed, (\d+) skipped", (out / "pytest.log").read_text(encoding="utf-8")
    )
    assert tests
    assert "All checks passed!" in (out / "ruff.log").read_text(encoding="utf-8")
    assert "left unchanged" in (out / "black.log").read_text(encoding="utf-8")
    assert "10/10 checks OK" in (out / "healthcheck.log").read_text(encoding="utf-8")
    status = "TEXT_ANNOTATION_REVIEW_REQUIRED_TRAINING_BLOCKED"
    lines = [
        "# Stage 9B — required road-text recognition and review preparation",
        "",
        f"Status: **{status}**",
        "",
        "Road text is a required foreground class, not a permanent ignore category. Three-class training is blocked: draft text/character geometry is unapproved and no confident text-positive training region was identified in the bounded inspected search. N9_01 stays development geography and will not be borrowed for training.",  # noqa: E501
        "",
        "## Recognition taxonomy and preserved checkpoints",
        "",
        "| Model foreground ID | Name | Stable visual ID |",
        "|---|---|---:|",
        "| 0 | longitudinal_solid | 1 |",
        "| 1 | longitudinal_dash | 2 |",
        "| 2 | road_text | 9 |",
        "",
        "The proposed three-class no-object logit is 3. Existing two-class checkpoints keep no-object at 2 and are not relabeled or overwritten. `configs/recognition_stage9b.yaml` is an inactive, training-disabled versioned specification; `proposed_model_config.json` is a configuration proposal, not a trained checkpoint. Unsupported arrows/symbols remain excluded. Visual arrow 3 is explicitly rejected by the new converter; visual-minus-one conversion must not silently turn it into model road_text 2.",  # noqa: E501
        "",
        "## Annotation granularity and source requirements",
        "",
        "Available original evidence is the client Other Marks schema (LETRA, ALGARISMO, ID_GRUPO), the preserved reference GeoPackages and project architecture/taxonomy documents. No authoritative catalog document was found; exact catalog semantics cannot be independently certified. Both Other Marks snapshots contain zero populated character features. N9_01 has four arrow polygons but no supplied text geometry.",  # noqa: E501
        "Use individual character instances when observed paint supports reliable separation, linked by a shared physical inscription/group ID. Preserve holes, disconnected observed fragments, native affine coordinates and source provenance. Letter/digit values remain **null until human-confirmed**. No source client ID is invented. Partial, faint or inseparable text is a group-fragment review candidate, not automatic character GT. One road_text semantic class does not itself read letters or perform OCR.",  # noqa: E501
        "",
        "## N9_01 focused correction and four-window audit",
        "",
        "N9_01 contains five proposed character-split masks for the most separable word area, plus partial/faint group fragments. A sixth faint area is explicitly unclassified possible text. Geometry traces observed native pixels only; no NoData gaps or character holes are filled. These are pixel-level proposals requiring boundary, separation, identity and complete-visible-target confirmation; the character count is not a claim of complete annotation.",  # noqa: E501
        "Focused QA: [N9_01 text before/after](N9_01_text_before_after.png). Existing solid/dash candidate geometry is copied byte-for-byte into a read-only base layer. The new text layer does not expand existing arrows or reclassify symbols.",  # noqa: E501
        "",
        "| Region | Proposed role retained | Source-image audit | New review handling |",
        "|---|---|---|---|",
        "| N9_01 | development | Visible road text, including faint/partial glyphs | Cyan character-split drafts; orange fragments; unknown extents ignored pending review |",  # noqa: E501
        "| N9_02 | development | No confident text identified; elongated bright feature unoutlined | Unclassified supported-feature inspection; no text assignment |",  # noqa: E501
        "| N9_03 | development | No confident text identified; small bright L-like feature unoutlined | Unclassified supported-feature inspection; no letter/class guessed |",  # noqa: E501
        "| N9_04 | training | No confident road text identified in September source view | Completeness review still required; no invented text-positive sample |",  # noqa: E501
        "",
        "Four-region source images and NoData/observed masks remain aligned at 0.05 native-unit pixels with CRS unset. Full per-candidate pixel counts, native geometry and source-file provenance are in road_text_candidates.gpkg and four_region_coverage_audit.json. Unconfirmed observed extents are blocked from verified background; NoData remains excluded independently. No region is declared fully approved and no supervised training targets are exported.",  # noqa: E501
        "",
        "## Text-positive training search and independence",
        "",
        "Inspected **46 cached training windows** and **8 additional source-backed views** across geographically separated road sections. The eight new search windows were rendered only after full-source integrity passed (expected/actual counts equal, zero dropped points, frozen normalization reused). They are >=500 native units from development/protected/historical footprints, pairwise separated, and their line groups do not overlap development groups. The four Stage-9 views include the September acquisition; the new search views cover the two July acquisitions. Acquisition labels do not establish independence by themselves.",  # noqa: E501
        "**No confident text-positive training region was found in the inspected set.** The eight views are search negatives/uncertain observations, not fabricated text training candidates. This is not proof that every uninspected raw source lacks text. Authoritative geometry cannot locate text here because all provided character fields are empty. Additional known text-bearing training coverage, or further independently verified source imagery, is needed before claiming sufficient training data.",  # noqa: E501
        "N9_01–N9_03 retain proposed development roles and N9_04 retains proposed training. T7/S8, their approved classes, masks, confirmed T7_05 correction and physical identity exclusions remain untouched. No protected label/image was used for model selection or text annotation discovery. Repeated observations of one physical word must share an identity/group and stay in one split; character crops from one inscription are not independent examples. No independent text-positive final-test population is currently established, so unseen road-text recognition cannot be claimed.",  # noqa: E501
        "",
        "## One efficient QGIS review gate",
        "",
        f"Project: **{qgis['project']}**. It reopens with **{qgis['layer_count']} valid layers** and **{qgis['bookmarks']} bookmarks**, opening directly on the N9_01 text correction. Existing solid/dash and unsupported-symbol layers are read-only. New character geometry/value and missing-target layers are editable review copies. Observed/NoData display masks are byte copies; original source files and earlier packages remain unchanged.",  # noqa: E501
        "Review the new character boundaries/splits and any readable character values; partition partial/faint groups only where observations permit. Leave unknown values blank. Inspect the two unclassified features and confirm complete visible-target coverage in the four still-unapproved Stage-9 regions. No repeat approval is requested for T7/S8 or other previously approved geometry. The eight negative search views are documented, not presented as positive regions needing training approval.",  # noqa: E501
        "",
        "## Three-class training feasibility — preparation only",
        "",
        "The opt-in head utility returns a new classifier: it preserves solid/dash rows, copies old no-object row 2 to new no-object row 3, and initializes text row 2 with a recorded seed. It leaves the source head and RNG unchanged. Installed Transformers source confirms the head is `class_predictor = Linear(hidden_dim, num_labels + 1)`. A future model must be constructed from the new config with a matching three-class internal criterion, compatible non-head weights and the custom `ValidMask2FormerCriterion(num_classes=3)`. A COCO initializer needs a fresh head, not invented COCO-to-road label correspondences.",  # noqa: E501
        "Software tests validate deterministic initialization, legacy-row preservation, class conversion, rejection of arrow-to-text confusion, three-class matching and valid-only/NoData-invariant mask costs and gradients. These use small software tensors, not real-data training, a model checkpoint adaptation run, or inference.",  # noqa: E501
        "Actual execution remains blocked. Legacy training/evaluation drivers include two-class paths and must receive an explicitly versioned three-class integration rather than silently omitting text. Once data is approved, use character masks, observed-pixel/class/word/group-balanced exposure, dependence-safe validation, reproducible new checkpoint/config storage and separate text segmentation/instance metrics. Compare solid/dash precision/recall against the unchanged frozen reference to detect regression. Reading literal characters requires separately validated recognition/value supervision; all proposed values are currently unknown. No automatic promotion, current score recalibration or protected-test selection is authorized.",  # noqa: E501
        "",
        "## Verification and stop",
        "",
        f"**{tests.group(1)} passed, {tests.group(2)} skipped**; Ruff and Black pass; **healthcheck 10/10**. Four new head/loss tests passed. Artifact validation checks approval/model/source preservation, exact mask-to-native-geometry pixel counts, zero unobserved annotation pixels, unchanged solid/dash geometry and training/development guards. Source/head initialization plans and annotation schemas are saved separately.",  # noqa: E501
        "**Stop at human review.** The required road-text class is specified, but character geometry/values and complete coverage are not approved and real text-positive training examples are not secured. No three-class training, test inference, model promotion, digitization, production GIS export or commit occurred.",  # noqa: E501
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    dump(
        out / "final_status.json",
        {
            "status": status,
            "training_enabled": False,
            "text_positive_training_regions_confirmed": 0,
            "text_geometry_human_approved": False,
            "human_review_project": qgis["project"],
            "no_protected_test_consumption": True,
        },
    )
    dump(
        out / "review_assets_manifest.json",
        {
            "files": [
                {"path": str(p), "sha256": sha256(p), "size": p.stat().st_size}
                for p in sorted(data.rglob("*"))
                if p.is_file()
            ]
        },
    )
    sources = [
        *ROOT.glob("scripts/stage9b*.py"),
        ROOT / "src/roadmarking/training/road_text_head.py",
        ROOT / "tests/test_road_text_head.py",
        ROOT / "configs/recognition_stage9b.yaml",
    ]
    dump(
        out / "source_manifest.json",
        {"files": [{"path": str(p), "sha256": sha256(p)} for p in sources]},
    )
    dump(
        out / "package_manifest.json",
        {
            "version": latest["run_id"],
            "status": status,
            "files": [
                {
                    "path": p.relative_to(out).as_posix(),
                    "sha256": sha256(p),
                    "size": p.stat().st_size,
                }
                for p in sorted(out.rglob("*"))
                if p.is_file() and p.name != "package_manifest.json"
            ],
        },
    )
    dump(
        base / "latest.json",
        {
            **latest,
            "status": status,
            "report": str(out / "report.md"),
            "review_project": qgis["project"],
        },
    )
    (base / "report.md").write_text(
        f"# Stage 9B\n\n**{status}**\n\n[Versioned report]({latest['run_id']}/report.md)\n\nRoad text is required; first text-geometry review and real text-positive training data are still needed.\n",  # noqa: E501
        encoding="utf-8",
    )
    print(f"Stage 9B sealed: {status}")


if __name__ == "__main__":
    main()
