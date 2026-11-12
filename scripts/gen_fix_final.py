"""Protocol revision 3 for the fix run (user-approved 2026-11-12): deep raw NoData is REPORT-ONLY.

Unchanged: every recognition gate, the false-positive-query NoData gate, accepted-after-clipping
NoData == 0, stitched unsupported pixels == 0, reload, and the selection rule otherwise.
Usage: ``register`` (writes the revision before any test use), then ``evaluate``.
"""

import hashlib
import json
import sys
from pathlib import Path

from correction_train import boundary_recall, stitch_safety
from gen_fix import select_ok
from gen_train import CLASSES, acceptance, nodata_decomposed
from regsafe_common import (
    MODELS,
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
from stage9e_lib import development_crops, real_context

REPORT_ONLY = ("deep_nodata",)


def splits(fix_dir):
    prev = Path(
        json.loads((ROOT / "outputs/reports/realistic_v1_training/latest.json").read_text())["run"]
    )
    rev2 = json.loads((prev / "protocol_rev2.json").read_text())
    data = Path(json.loads((ROOT / "data/processed/realistic_v1/latest.json").read_text())["run"])
    man = json.loads((data / "manifest.json").read_text())
    rc = real_context()
    ds = Path(rc["dataset"])
    by = {c["crop_id"]: c for c in rc["full"]}
    dev = {c["crop_id"]: (Path(r), c) for r, c in development_crops(rc["ctx"])}
    s6 = [by[i] for i in man["real_splits"]["real_test"]["crops"] if i in by]
    val = [(ds, c) for c in s6 if c["tile_id"] in rev2["val_tiles"]]
    test = [(ds, c) for c in s6 if c["tile_id"] not in rev2["val_tiles"]] + [
        dev[i] for i in man["real_splits"]["real_test"]["crops"] if i not in by
    ]
    assert len(test) == rev2["test_crops"]
    return val, test, dev


def main(stage):
    fix = Path(
        json.loads((ROOT / "outputs/reports/realistic_v1_fix/latest.json").read_text())["run"]
    )
    rev3 = fix / "protocol_rev3.json"
    if stage == "register":
        assert not rev3.exists() and not (fix / "test_results.json").exists()
        dump(
            rev3,
            dict(
                change="deep (3-8 px) raw NoData of the fix candidate is report-only (user approved)",  # noqa: E501
                reason="Delivered output is clipped to observed support (accepted NoData must be 0 and "  # noqa: E501
                "stitched unsupported pixels must be 0, both still hard gates); the deep leakage comes "  # noqa: E501
                "from true lines continuing under unobserved gaps.",
                unchanged="all recognition gates, FP-query NoData <= 1.10x, accepted/stitched NoData == 0, "  # noqa: E501
                "reload, selection rule except the deep NoData term",
                rev2_sha256=hashlib.sha256((fix / "protocol.json").read_bytes()).hexdigest(),
                test_use="second use of the 67-crop real test (first: realistic_v1 step1500)",
            ),
        )
        print("registered", hashlib.sha256(rev3.read_bytes()).hexdigest())
        return
    assert rev3.exists()
    val, test, dev = splits(fix)
    checks, frozen, _ = frozen_paths()
    processor = load_processor()
    teacher = load_model(frozen)
    stub = val[:1]
    base = run(teacher, processor, val, stub)
    nd_base = nodata_decomposed(teacher, processor, val)
    stamp = fix.name
    sel, best = {}, None
    for ck in sorted((MODELS / f"realistic_v1_fix_{stamp}").glob("step*")):
        m = load_model(ck / "model")
        res = run(m, processor, val, stub)
        nd = nodata_decomposed(m, processor, val)
        nd_cut = {**nd, "deep_nodata": 0}  # report-only
        ok, score, gaps = select_ok(res, base, nd_cut, nd_base)
        gaps["deep_nodata_report_only"] = [nd_base["deep_nodata"], nd["deep_nodata"]]
        sel[ck.name] = dict(
            eligible=ok, score=score, gaps=gaps, sha256=sha256(ck / "model/model.safetensors")
        )
        if ok and (best is None or score > sel[best]["score"]):
            best = ck.name
        print(ck.name, sel[ck.name], flush=True)
        del m
    dump(fix / "selection_rev3.json", sel)
    if best is None:
        dump(fix / "decision_rev3.json", dict(status="NO_SAFE_IMPROVEMENT", frozen_default=True))
        print("NO ELIGIBLE CHECKPOINT")
        return
    cpath = MODELS / f"realistic_v1_fix_{stamp}" / best / "model"
    cand = load_model(cpath)
    all_f, all_c = run(teacher, processor, test, stub), run(cand, processor, test, stub)
    nd_f, nd_c = nodata_decomposed(teacher, processor, test), nodata_decomposed(
        cand, processor, test
    )
    dev_test = [x for x in test if x[1]["crop_id"] in dev]
    try:
        stitch = dict(
            frozen=stitch_safety(teacher, processor, dev_test, fix / "stitch_f"),
            candidate=stitch_safety(cand, processor, dev_test, fix / "stitch_c"),
        )
    except Exception as e:
        stitch = dict(error=repr(e))
    extra = dict(
        stitch=stitch.get("candidate"), reload=reload_identical(cand, cpath, processor, test[0])
    )
    table = acceptance(all_c, all_f, nd_c, nd_f, extra)
    report_only = {k: v for k, v in table.items() if k.startswith(REPORT_ONLY)}
    gates = {k: v for k, v in table.items() if k not in report_only}
    failed = [k for k, v in gates.items() if not v[0]]
    status = (
        "IMPROVED_PROVISIONAL"
        if not failed
        else (
            "RECOGNITION_REGRESSION"
            if any(("drop" in f) or ("false_positive" in f) for f in failed)
            else "NO_SAFE_IMPROVEMENT"
        )
    )
    br = dict(
        frozen=boundary_recall(teacher, processor, test),
        candidate=boundary_recall(cand, processor, test),
    )
    verify_frozen(checks)
    dump(
        fix / "test_results.json",
        dict(
            frozen=all_f,
            candidate=all_c,
            nodata=dict(frozen=nd_f, candidate=nd_c),
            boundary_recall=br,
            stitching=stitch,
            test_crops=len(test),
        ),
    )
    dump(
        fix / "decision_rev3.json",
        dict(
            status=status,
            selected=best,
            candidate=str(cpath),
            gates=gates,
            report_only=report_only,
            failed=failed,
            frozen_default=status != "IMPROVED_PROVISIONAL",
            promoted=False,
            test_use="second",
        ),
    )
    print("DECISION", status, best, failed, flush=True)
    for c in CLASSES:
        print(
            c,
            {
                lvl: {
                    m: (
                        round(all_f["real"][lvl]["pixel"][c][m], 3),
                        round(all_c["real"][lvl]["pixel"][c][m], 3),
                    )
                    for m in ("precision", "recall", "f1")
                }
                for lvl in ("raw", "v1")
            },
        )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
