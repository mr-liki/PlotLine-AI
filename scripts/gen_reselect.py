"""Protocol revision 2 (registered before ANY test evaluation of a new model).

Why: revision 1 selected on legacy P512 tiles, which were in the frozen model's training data,
so the frozen reference was memorisation-inflated there and no new model could be fairly
compared. Revision 2 selects on 3 of the 9 S6 expanded tiles (unseen by frozen AND by the new
model); the other 6 S6 tiles + the 14 development crops remain the untouched real test.
Selection rule, acceptance gates, thresholds and checkpoints are unchanged.
"""

import hashlib
import json
import sys
from pathlib import Path

from correction_train import boundary_recall, stitch_safety
from gen_train import CLASSES, GATES, SELECT_TOL, acceptance, nodata_decomposed, selection_ok
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

VAL_TILE_EVERY = 3  # tiles sorted by id; every third tile (offset 0) -> validation


def main(stage):
    tr = Path(
        json.loads((ROOT / "outputs/reports/realistic_v1_training/latest.json").read_text())["run"]
    )
    data = Path(json.loads((ROOT / "data/processed/realistic_v1/latest.json").read_text())["run"])
    man = json.loads((data / "manifest.json").read_text())
    rc = real_context()
    ds = Path(rc["dataset"])
    by = {c["crop_id"]: c for c in rc["full"]}
    dev = {c["crop_id"]: (Path(r), c) for r, c in development_crops(rc["ctx"])}
    s6 = [by[i] for i in man["real_splits"]["real_test"]["crops"] if i in by]
    tiles = sorted({c["tile_id"] for c in s6})
    val_tiles = tiles[::VAL_TILE_EVERY]
    val = [(ds, c) for c in s6 if c["tile_id"] in val_tiles]
    test = [(ds, c) for c in s6 if c["tile_id"] not in val_tiles] + [
        dev[i] for i in man["real_splits"]["real_test"]["crops"] if i not in by
    ]
    for cls in (0, 1):
        assert any(cls in c["labels"] for _, c in val)
    proto = tr / "protocol_rev2.json"
    if stage == "register":
        assert not (tr / "test_results.json").exists() and not proto.exists()
        dump(
            proto,
            dict(
                reason=__doc__.strip(),
                val_tiles=val_tiles,
                val_crops=len(val),
                test_crops=len(test),
                test_tiles=[t for t in tiles if t not in val_tiles],
                test_dev_crops=sum(1 for _, c in test if c["crop_id"] in dev),
                selection="max mean V1 F1 on rev2 val; eligible only if per-class V1 "
                f"precision/recall drop vs frozen <= {SELECT_TOL}",
                gates=GATES,
                checkpoints="all saved run checkpoints",
                rev1_protocol_sha256=hashlib.sha256(
                    (tr / "protocol.json").read_bytes()
                ).hexdigest(),
            ),
        )
        print(
            "registered",
            hashlib.sha256(proto.read_bytes()).hexdigest(),
            val_tiles,
            len(val),
            len(test),
        )
        return
    assert proto.exists()
    checks, frozen, _ = frozen_paths()
    processor = load_processor()
    teacher = load_model(frozen)
    stub = val[:1]
    base = run(teacher, processor, val, stub)
    ck_root = MODELS / f"realistic_v1_{tr.name}"
    sel = {}
    best = None
    for ck in sorted(ck_root.glob("step*")):
        m = load_model(ck / "model")
        res = run(m, processor, val, stub)
        ok, score, gaps = selection_ok(res, base)
        sel[ck.name] = dict(
            eligible=ok,
            score=score,
            gaps=gaps,
            sha256=sha256(ck / "model/model.safetensors"),
            f1={c: res["real"]["v1"]["pixel"][c]["f1"] for c in CLASSES},
        )
        if ok and (best is None or score > sel[best]["score"]):
            best = ck.name
        print(ck.name, sel[ck.name], flush=True)
        del m
    sel["frozen"] = dict(
        score=sum(base["real"]["v1"]["pixel"][c]["f1"] or 0 for c in CLASSES) / 2,
        f1={c: base["real"]["v1"]["pixel"][c]["f1"] for c in CLASSES},
    )
    dump(tr / "selection_rev2.json", sel)
    if best is None:
        dump(
            tr / "decision_rev2.json",
            dict(
                status="NO_SAFE_IMPROVEMENT",
                reason="no eligible checkpoint on rev2 val",
                frozen_default=True,
            ),
        )
        print("NO ELIGIBLE CHECKPOINT")
        return
    cpath = ck_root / best / "model"
    cand = load_model(cpath)
    all_f, all_c = run(teacher, processor, test, stub), run(cand, processor, test, stub)
    nd_f, nd_c = nodata_decomposed(teacher, processor, test), nodata_decomposed(
        cand, processor, test
    )
    dev_test = [x for x in test if x[1]["crop_id"] in dev]
    try:
        stitch = dict(
            frozen=stitch_safety(teacher, processor, dev_test, tr / "stitch_f"),
            candidate=stitch_safety(cand, processor, dev_test, tr / "stitch_c"),
        )
    except Exception as e:
        stitch = dict(error=repr(e))
    extra = dict(
        stitch=stitch.get("candidate"), reload=reload_identical(cand, cpath, processor, test[0])
    )
    table = acceptance(all_c, all_f, nd_c, nd_f, extra)
    failed = [k for k, v in table.items() if not v[0]]
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
        tr / "test_results.json",
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
        tr / "decision_rev2.json",
        dict(
            status=status,
            selected=best,
            candidate=str(cpath),
            gates=table,
            failed=failed,
            frozen_default=status != "IMPROVED_PROVISIONAL",
            promoted=False,
        ),
    )
    print("DECISION", status, best, failed)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
