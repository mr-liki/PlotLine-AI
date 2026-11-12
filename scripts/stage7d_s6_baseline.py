"""Stage 7D: write-once first-use baseline on the fresh development region S6_01.

The opening record is written BEFORE any S6_01 imagery is passed to a model. Both fixed
reference pipelines (default V0; frozen candidate V1 + stitching) are scored once, with no
threshold changes. After this script S6_01 is development data, used for selection.
"""

import json
import sys
from datetime import UTC, datetime

from phase5_11_common import load
from roadmarking.dataset.phase55 import write_attestation
from roadmarking.evaluation.full_metrics import aggregate
from roadmarking.evaluation.harness import paired_block_bootstrap
from roadmarking.rasterization.pipeline import dump, sha256
from stage1_scoring import run_inference
from stage7_common import (
    FROZEN_CANDIDATE,
    OUT,
    S6_DEV,
    SHIFTS,
    evaluate_component,
    headline,
    s6_component,
    settings,
    shifted,
)

BASE = OUT / "7D_S6_01_baseline"


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    opening, results_path = BASE / "opened.json", BASE / "baseline_results.json"
    if results_path.exists():
        sys.exit("S6_01 baseline already recorded once; refusing to overwrite.")
    frozen = load(FROZEN_CANDIDATE)
    assert sha256(frozen["weights"]) == frozen["weights_sha256"]
    manifest = load(S6_DEV)
    for crop in manifest["crops"]:
        assert sha256(crop["file"]) == crop["sha256"], crop["crop_id"]
    assert sha256(manifest["ground_truth"]) == manifest["ground_truth_sha256"]
    cfg = settings()
    if not opening.exists():
        write_attestation(
            opening,
            {
                "opened_utc": datetime.now(UTC).isoformat(),
                "region": "S6_01",
                "authorization": "User message: 'Approve, S6_01 as dev'; Stage 7D instruction",
                "role_after_this_run": "DEVELOPMENT (selection-used from Stage 7H on)",
                "fixed_settings": cfg,
                "frozen_candidate_sha256": sha256(FROZEN_CANDIDATE),
                "s6_manifest_sha256": sha256(S6_DEV),
                "policy": "single baseline; no threshold retuning on S6_01; results as-is",
            },
        )
    out = {"status": "S6_01_FIRST_USE_BASELINE", "opened_record_sha256": sha256(opening)}
    evals = {}
    for name, c in cfg.items():
        folder = BASE / "inference" / name
        run_inference(c["checkpoint"], S6_DEV.parent, manifest["crops"], folder)
        comp = s6_component(folder)
        evals[name] = evaluate_component(comp, c["thresholds"], c["stitch"])
        sens = {}
        for delta in SHIFTS:
            if delta:
                r = evaluate_component(comp, shifted(c["thresholds"], delta), c["stitch"])
                sens[f"{delta:+.2f}"] = headline(r)
        out[name] = {
            "settings": c,
            "headline": headline(evals[name]),
            "metrics": aggregate([evals[name]["full"]]),
            "ap": evals[name]["full"]["ap"],
            "links": evals[name]["full"]["stitch_audit"],
            "link_records": evals[name]["full"]["link_records"],
            "per_target": evals[name]["full"]["per_target"],
            "harness_summary": evals[name]["harness_summary"],
            "confidence_sensitivity": sens,
        }
    a, b = evals["default_V0"], evals["frozen_candidate_V1_stitch"]
    out["candidate_minus_default_block_bootstrap"] = paired_block_bootstrap(
        a["records"], a["detections"], b["records"], b["detections"], n=2000, seed=0
    )
    out["caution"] = (
        "Single region, 3 solid / 2 dash IDs: instance metrics move in steps of 20-33 points; "
        "block-bootstrap intervals are indicative only."
    )
    dump(results_path, out)
    print(json.dumps({k: out[k]["headline"] for k in cfg}, indent=1, default=str))


if __name__ == "__main__":
    main()
