"""Stage 7A: protected-test provenance audit and geographic split roles (records only)."""

import json
import os
from datetime import UTC, datetime

from phase5_11_common import ROOT, load
from roadmarking.rasterization.pipeline import dump, sha256

OUT = ROOT / "outputs/reports/stage7"


def mtime(p):
    return datetime.fromtimestamp(os.path.getmtime(p), UTC).isoformat()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ft = ROOT / "outputs/reports/final_test"
    fd = ROOT / "data/processed/final_test"
    review = ROOT / "data/processed/phase5_7_new_review/20261106T105319835838Z"
    opening = load(ft / "test_opened.json")
    frozen = ROOT / "outputs/reports/final_candidate/frozen_candidate.json"
    results = load(ft / "final_test_results.json")
    events = [
        {
            "utc": mtime(review / "images/NEW_FINAL_TEST_RESERVED.png"),
            "event": "Phase 5.7: protected-region imagery rendered from LAZ for HUMAN annotation "
            "review only (QGIS); user approved its geometry (new_three_region_approval.json)",
            "imagery_loaded_by_model": False,
            "labels_read_for_metrics": False,
            "inference": False,
            "metrics": False,
        },
        {
            "utc": "2026-11-06 .. 2026-11-07 (Phases 5.8 to Stage 5)",
            "event": "Only metadata (bounds, source-ID list) read by leakage guards "
            "(phase5_11_common.reserved_geography, stage0 freeze); training regions kept "
            ">=500 units away and ID-disjoint",
            "imagery_loaded_by_model": False,
            "labels_read_for_metrics": False,
            "inference": False,
            "metrics": False,
        },
        {
            "utc": mtime(frozen),
            "event": "Final candidate frozen (write-once), before any test access",
        },
        {
            "utc": opening["opened_utc"],
            "event": "User-authorized one-time final test opened ('Run final test'); write-once "
            "opening record created before reading test data",
            "frozen_candidate_sha256_at_opening": opening["frozen_candidate_sha256"],
        },
        {
            "utc": mtime(fd / "NEW_FINAL_TEST_RESERVED_ground_truth.npz"),
            "event": "Approved labels exported to ground truth; protected imagery cropped",
            "imagery_loaded_by_model": True,
            "labels_read_for_metrics": True,
        },
        {
            "utc": mtime(ft / "inference/candidate"),
            "event": "Inference run on protected imagery: frozen candidate and default",
            "inference": True,
        },
        {
            "utc": mtime(ft / "final_test_results.json"),
            "event": "Test metrics computed and reported as-is",
            "metrics": True,
        },
    ]
    influence = {
        "frozen_candidate_unchanged_since_opening": sha256(frozen)
        == opening["frozen_candidate_sha256"],
        "test_results_used_for_training_or_selection": False,
        "evidence": "No script after the opening reads final_test_results.json or the test "
        "crops except final_test_report.py (re-render only). Stage 6 regions were "
        "chosen (diversity) and packaged before the test; their approval and export "
        "used no test output.",
    }
    roles = {
        "development (used repeatedly for evaluation and selection)": [
            "LEGACY_POSITIVE_DEV (5 legacy INTERCHANGE crops)",
            "PHASE58_POSITIVE_DEV_REVIEW",
            "NEW_DEV_REVIEW (sparse negative control)",
        ],
        "selection-used validation (NOT untouched)": [
            "26 P511/P512 regions: leave-one-region-out held-out folds in Stages 1-5 and used "
            "to fit thresholds; they are also training data for full-data models"
        ],
        "historically evaluated independent benchmarks (diagnostic only)": [
            "FUTURE_TEST_1, FUTURE_TEST_2, FUTURE_TEST_3 (Phase 5.6)"
        ],
        "historical final test (CONSUMED once, never to be called untouched)": [
            "NEW_FINAL_TEST_RESERVED"
        ],
        "fresh development (untouched until the Stage 7D baseline is recorded)": ["S6_01"],
        "unreviewed held-out candidate (not used)": ["PHASE58_NEW_DEV_1"],
        "untouched final-test regions available now": [],
    }
    discrepancy = (
        "Reports written before 2026-11-07 04:33 UTC correctly said the protected test was "
        "unused. The user then authorized and ran it once, so it is consumed. The Stage 6 export "
        "printed 'protected_final_test_used: false' because check_phase511() returned that "
        "constant as a statement about dataset CONTENT (no protected crops in training), not "
        "about test consumption. The key is renamed to 'protected_test_crops_in_dataset' to "
        "remove the ambiguity, and the Stage 6 manifest records 'protected_final_test_consumed: "
        "true'."
    )
    audit = {
        "protected_region": "NEW_FINAL_TEST_RESERVED",
        "status": "CONSUMED_ONCE_HISTORICAL",
        "answers": {
            "imagery_loaded": "yes, once, at the authorized final test (and "
            "rendered for human review in Phase 5.7)",
            "labels_read": "yes, once, at the authorized final test",
            "inference_on_protected_imagery": "yes, once (candidate and default)",
            "test_metrics_calculated": "yes, once",
            "affected_training_or_selection": "no",
        },
        "events": events,
        "influence": influence,
        "discrepancy_explanation": discrepancy,
        "split_roles": roles,
        "final_test_summary": {
            k: v["summary"]["foreground_iou"] for k, v in results["results"].items()
        },
    }
    dump(OUT / "7A_protected_test_provenance.json", audit)
    md = [
        "# Stage 7A: protected-test provenance audit",
        "",
        "**Status of NEW_FINAL_TEST_RESERVED: CONSUMED ONCE (historical).** It must not be "
        "presented as untouched again.",
        "",
        "| Question | Answer |",
        "|---|---|",
    ]
    md += [f"| {k.replace('_', ' ')} | {v} |" for k, v in audit["answers"].items()]
    md += ["", "## Timeline (from file records)", "", "| UTC | Event |", "|---|---|"]
    md += [f"| {e['utc']} | {e['event']} |" for e in events]
    md += [
        "",
        f"Frozen candidate unchanged since opening: "
        f"**{influence['frozen_candidate_unchanged_since_opening']}**. "
        f"{influence['evidence']}",
        "",
        "## The apparent discrepancy",
        "",
        discrepancy,
        "",
        "## Geographic split roles",
        "",
    ]
    for k, v in roles.items():
        md.append(f"- **{k}**: {', '.join(v) if v else 'none: new regions must be prepared (7J)'}")
    (OUT / "7A_protected_test_provenance.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(
        json.dumps(audit["answers"], indent=1),
        influence["frozen_candidate_unchanged_since_opening"],
    )


if __name__ == "__main__":
    main()
