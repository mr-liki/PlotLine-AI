"""Collect machine-readable final Phase 5.5 verification and delivery evidence."""

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from roadmarking.rasterization.pipeline import dump, sha256

root = Path(__file__).resolve().parents[1]
context = json.loads((root / "outputs/reports/phase5_5/latest.json").read_text())
report = Path(context["report_dir"])
run = Path(context["selected_checkpoint"]).parent
suite = ET.parse(root / "outputs/reports/phase5_5_final.xml").getroot().find("testsuite")
health = json.loads((report / "healthcheck.json").read_text())
reload = json.loads((run / "selected_reload_validation.json").read_text())
proof = json.loads((run / "resume_proof_step300.json").read_text())
checks = {
    "pytest": dict(suite.attrib),
    "ruff": (root / "outputs/reports/phase5_5_ruff.log").read_text(),
    "black": (root / "outputs/reports/phase5_5_black.log").read_text(encoding="utf-8"),
    "healthcheck_local_passes": sum(x["status"] == "pass" for x in health),
    "healthcheck_skips": [x["name"] for x in health if x["status"] == "skip"],
    "fresh_model_reload": reload,
    "real_resume": proof,
    "failure_evidence": "failed_segment_training_log.json; phase5_5_train600.log",
}
assert int(suite.attrib["failures"]) == 0 and int(suite.attrib["errors"]) == 0
assert all(x["status"] != "fail" for x in health)
assert "All checks passed!" in checks["ruff"]
assert "would be left unchanged" in checks["black"]
assert reload["fresh_process_model_reload"]
assert proof["restored_step"] == 300 and proof["scaler_scale"] == 16
dump(report / "verification.json", checks)
for filename in (
    "phase5_5_train300.log",
    "phase5_5_train600.log",
    "phase5_5_train600_recovery.log",
    "phase5_5_tests_final.log",
    "phase5_5_final.xml",
    "phase5_5_ruff.log",
    "phase5_5_black.log",
):
    shutil.copy2(root / "outputs/reports" / filename, report / filename)
shutil.copy2(run / "selected_reload_validation.json", report / "selected_reload_validation.json")
request = Path("C:/Users/nt01173/.codex/attachments/24239edc-cb91-4064-a591-689d91c0c064")
request /= "Pasted text.txt"
dump(report / "request_provenance.json", {"file": str(request), "sha256": sha256(request)})
evaluation = json.loads((report / "recognition_evaluation.json").read_text())
chosen = next(
    x for x in evaluation["candidate_comparisons"] if x["step"] == evaluation["selected_step"]
)
seconds = chosen["frozen"]["model_only_seconds_total"]
with (report / "report.md").open("a", encoding="utf-8") as stream:
    stream.write("\n## Final verification and inference timing\n\n")
    stream.write(
        "298 tests passed, one optional network test skipped. Ruff passed; "
        "Black passed. Nine local health checks passed; optional "
        "network model-config check skipped. Fresh-process reload passed; "
        f"logit differences: `{reload['max_abs_logit_differences']}`.\n\n"
    )
    stream.write(
        f"Selected model-only GPU inference: {seconds/5:.4f} s/crop "
        f"({5/seconds:.2f} native 512 crops/s), five sequential samples. "
        "Excludes I/O, mask filtering and geographic stitching.\n\n"
    )
    stream.write("| Model / class | AP50 | AP75 |\n|---|---:|---:|\n")
    for model_key, name in (
        ("baseline_frozen", "Phase 5.4"),
        ("selected_retuned", "Selected Phase 5.5"),
    ):
        for cls, ap in evaluation[model_key]["mean_per_crop_confidence_truncated_ap"].items():
            stream.write(f"| {name} / {cls} | {ap['AP50']:.4f} | {ap['AP75']:.4f} |\n")
    stream.write(
        "\nUse `selected_inference.json` to load the selected checkpoint and its "
        "calibrated postprocessing together. All quantitative results are "
        "development-only. Independent test: not evaluated.\n"
    )
print(json.dumps(checks, indent=2))
