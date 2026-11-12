"""Independent acceptance test (acceptance_v1): register the rules, then evaluate ONCE.

Models: A = frozen default; C = realistic_v1_fix step0500 (best candidate). Gates are those of
protocol revision 3, fixed before any model sees this set: every recognition and
false-positive gate, FP-query raw NoData <= 1.10x (hard), accepted NoData after clipping == 0,
reload repeatable; deep (3-8 px) raw NoData is report-only (user decision 2026-11-12).
Production stitching is not applicable (independent, non-overlapping windows).
Minimum evidence: >= 40 crops with >= 40 solid and >= 40 dash instances, else INCONCLUSIVE.
"""

import hashlib
import json
import sys
from pathlib import Path

from correction_train import boundary_recall
from gen_train import CLASSES, acceptance, nodata_decomposed
from regsafe_common import (
    ROOT,
    dump,
    frozen_paths,
    load_model,
    load_processor,
    sha256,
    verify_frozen,
)
from regsafe_eval import run
from regsafe_train import reload_identical

MIN = dict(crops=40, solid=40, dash=40)
REPORT_ONLY = ("deep_nodata",)


def candidate_path():
    fix = Path(
        json.loads((ROOT / "outputs/reports/realistic_v1_fix/latest.json").read_text())["run"]
    )
    dec = json.loads((fix / "decision_rev3.json").read_text())
    return Path(dec["candidate"])


def main(stage):
    data = Path(json.loads((ROOT / "data/processed/acceptance_v1/latest.json").read_text())["run"])
    recs = json.loads((data / "records.json").read_text())
    proto = data / "acceptance_protocol.json"
    cpath = candidate_path()
    if stage == "register":
        assert not proto.exists() and not (data / "acceptance_results.json").exists()
        dump(
            proto,
            dict(
                dataset=str(data),
                records_sha256=sha256(data / "records.json"),
                crops=len(recs),
                solid=sum(r["labels"].count(0) for r in recs),
                dash=sum(r["labels"].count(1) for r in recs),
                models=dict(
                    A=str(frozen_paths()[1]),
                    C=str(cpath),
                    C_sha256=sha256(cpath / "model.safetensors"),
                ),
                gates="protocol rev3 acceptance gates (gen_train.acceptance) "
                "minus report-only deep NoData",
                report_only=list(REPORT_ONLY),
                minimum_evidence=MIN,
                decision="ACCEPTED if every gate passes with minimum evidence; "
                "NOT_ACCEPTED otherwise; "
                "INCONCLUSIVE if minimum evidence is not met. No model is promoted automatically.",
            ),
        )
        print("registered", hashlib.sha256(proto.read_bytes()).hexdigest())
        return
    assert proto.exists()
    p = json.loads(proto.read_text())
    assert p["records_sha256"] == sha256(data / "records.json")
    test = [(Path(r["file"]).parent, r) for r in recs]
    checks, frozen, _ = frozen_paths()
    processor = load_processor()
    a, c = load_model(frozen), load_model(cpath)
    stub = test[:1]
    ra, rc = run(a, processor, test, stub), run(c, processor, test, stub)
    nda, ndc = nodata_decomposed(a, processor, test), nodata_decomposed(c, processor, test)
    table = acceptance(
        rc, ra, ndc, nda, dict(stitch=None, reload=reload_identical(c, cpath, processor, test[0]))
    )
    report_only = {k: v for k, v in table.items() if k.startswith(REPORT_ONLY)}
    gates = {k: v for k, v in table.items() if k not in report_only}
    failed = [k for k, v in gates.items() if not v[0]]
    enough = p["crops"] >= MIN["crops"] and p["solid"] >= MIN["solid"] and p["dash"] >= MIN["dash"]
    status = "INCONCLUSIVE" if not enough else ("ACCEPTED" if not failed else "NOT_ACCEPTED")
    by_group = {}
    for g in sorted({r["group"] for _, r in test}):
        sub = [x for x in test if x[1]["group"] == g]
        by_group[g] = dict(
            crops=len(sub),
            A=run(a, processor, sub, stub)["real"],
            C=run(c, processor, sub, stub)["real"],
        )
    br = dict(A=boundary_recall(a, processor, test), C=boundary_recall(c, processor, test))
    verify_frozen(checks)
    dump(
        data / "acceptance_results.json",
        dict(A=ra, C=rc, nodata=dict(A=nda, C=ndc), by_group=by_group, boundary_recall=br),
    )
    dump(
        data / "acceptance_decision.json",
        dict(
            status=status,
            gates=gates,
            report_only=report_only,
            failed=failed,
            promoted=False,
            frozen_default=True,
        ),
    )
    print("DECISION", status, failed)
    for lvl in ("raw", "v1"):
        for cl in CLASSES:
            px = {k: r["real"][lvl]["pixel"][cl] for k, r in (("A", ra), ("C", rc))}
            print(
                lvl,
                cl,
                {
                    k: [round(v[m] or 0, 3) for m in ("precision", "recall", "f1")]
                    for k, v in px.items()
                },
            )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
