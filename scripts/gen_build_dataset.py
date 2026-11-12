"""Build realistic_v1: real-background generated train/val/stress-test sets plus real splits.

Geography (all asserted):
* real TEST  = expanded S6_02..S6_10 crops (77) + the 14 development crops: never used as
  backgrounds, never in training, frozen model never trained on them;
* real VAL   = legacy P512_* tiles (model selection only);
* real TRAIN = remaining legacy tiles; generated TRAIN/VAL backgrounds come only from these
  train/val tiles respectively;
* STRESS test (generated) = backgrounds from real-test tiles with held-out (east) CC0 templates.
T7/S8 protected populations and N9_01 are excluded by real_context assertions.
"""

import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from regsafe_common import ROOT, dump, sha256
from roadmarking.training.realistic_generator import GenConfig, assemble_tile, generate
from stage9c_attested_pilot import crop_box
from stage9e_lib import development_crops, real_context

OUT = ROOT / "data/processed/realistic_v1"
OREGON = ROOT / "data/processed/open_sources/oregon_philomath_templates"
COUNTS = dict(train=4000, val=400, stress_test=600)
SEEDS = dict(train=31001, val=31002, stress_test=31003)
CLASS_SEM = {2: "crosswalk_text", 3: "bike", 4: "arrow", 5: "arrow", 6: "arrow"}


def oregon_templates():
    scene = np.load(OREGON / "scene_rasters.npz")
    cls, inten = scene["cls"], scene["paint_intensity"]
    meta = json.loads((OREGON / "templates.json").read_text())
    width = cls.shape[1]
    out = {"west": [], "east": []}
    for t in meta["templates"]:
        if t["kind"] not in {
            "crosswalk_text",
            "bike",
            "arrow_left",
            "arrow_right",
            "arrow_straight",
        }:
            continue
        r0, r1, c0, c1 = t["bbox"]
        sub = cls[r0:r1, c0:c1]
        cid = int(np.bincount(sub[sub > 0]).argmax())
        mask = sub == cid
        if mask.sum() < 30 or max(mask.shape) > 300:
            continue
        ys, xs = np.nonzero(mask)
        p = np.stack([xs, ys], 1).astype(float)
        p -= p.mean(0)
        _ev, evec = np.linalg.eigh(np.cov(p.T))
        ang = float(np.degrees(np.arctan2(evec[1, 1], evec[0, 1])))
        side = "west" if (c0 + c1) / 2 < 0.6 * width else "east"
        out[side].append(
            dict(
                mask=mask,
                intensity=np.where(mask, inten[r0:r1, c0:c1], 0),
                angle_deg=ang,
                semantic=CLASS_SEM[cid],
                source=f"oregon_cc0:{t['kind']}:{r0},{c0}",
            )
        )
    return out


def write_sample(folder, name, s):
    path = folder / f"{name}.npz"
    np.savez_compressed(
        path,
        rgb=np.repeat(s["gray"][..., None], 3, axis=2),
        masks=s["masks"],
        labels=s["labels"],
        validity=s["validity"],
        semantic=s["semantic"],
    )
    return path


def main():
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = OUT / stamp
    rc = real_context()
    root = Path(rc["dataset"])
    full = rc["full"]
    dev = []
    for droot, c in development_crops(rc["ctx"]):
        with np.load(Path(droot) / c["file"]) as d:
            dev.append({**c, "labels": d["labels"].tolist(), "_root": str(droot)})
    legacy = [c for c in full if c["geography"] == "legacy"]
    expanded = [c for c in full if c["geography"] == "expanded"]
    by_tile = defaultdict(list)
    for c in legacy + expanded:
        by_tile[c["tile_id"]].append(c)
    test_tiles = sorted({c["tile_id"] for c in expanded})
    val_tiles = sorted({c["tile_id"] for c in legacy if c["tile_id"].startswith("P512_")})
    train_tiles = sorted({c["tile_id"] for c in legacy} - set(val_tiles))
    for a, b in ((train_tiles, val_tiles), (train_tiles, test_tiles), (val_tiles, test_tiles)):
        assert not set(a) & set(b)
    real = dict(
        real_train=[c for c in legacy if c["tile_id"] in train_tiles],
        real_val=[c for c in legacy if c["tile_id"] in val_tiles],
        real_test=expanded + dev,
    )
    # Geographic guard: test crops >= 500 native units from every train/val crop.
    from shapely.ops import unary_union

    trainval = unary_union([crop_box(c) for c in real["real_train"] + real["real_val"]])
    test_dist = min(crop_box(c).distance(trainval) for c in real["real_test"])
    assert test_dist >= 500, test_dist
    for k, v in real.items():
        assert all(sum(cls in c["labels"] for c in v) for cls in (0, 1)), k

    def load(c):
        with np.load(root / c["file"]) as d:
            return {k: d[k] for k in ("rgb", "masks", "labels", "validity")}

    templates = oregon_templates()
    tiles = {}
    for tid in train_tiles + val_tiles + test_tiles:
        tiles[tid] = assemble_tile(tid, by_tile[tid], load)
    global_pool = np.concatenate([tiles[t].paint_pool for t in train_tiles])
    plan = dict(
        train=(train_tiles, templates["west"]),
        val=(val_tiles, templates["west"]),
        stress_test=(test_tiles, templates["east"]),
    )
    manifest = dict(
        created=stamp,
        generator="roadmarking.training.realistic_generator",
        config=GenConfig().__dict__,
        splits={},
        real_splits={},
        geography=dict(
            train_tiles=train_tiles,
            val_tiles=val_tiles,
            test_tiles=test_tiles,
            test_min_distance_native_units=test_dist,
            dev_crops=[c["crop_id"] for c in dev],
        ),
        templates={k: len(v) for k, v in templates.items()},
        sources=dict(oregon="CC0 1.0 doi:10.7910/DVN/0STTJR"),
    )
    for split, (tids, tmpl) in plan.items():
        folder = out / split
        folder.mkdir(parents=True)
        rng = np.random.default_rng(SEEDS[split])
        pool = (
            global_pool
            if split != "stress_test"
            else np.concatenate([tiles[t].paint_pool for t in tids if len(tiles[t].paint_pool)])
        )
        records = []
        for i in range(COUNTS[split]):
            tid = tids[i % len(tids)]
            s = generate(tiles[tid], rng, pool, tmpl, GenConfig())
            path = write_sample(folder, f"{split}_{i:05d}", s)
            records.append(
                dict(
                    crop_id=path.stem,
                    file=str(path),
                    sha256=sha256(path),
                    size=512,
                    labels=s["labels"].tolist(),
                    geography="generated_real_background",
                    tile_id=tid,
                    record=s["record"],
                )
            )
            if i % 500 == 0:
                print(split, i, flush=True)
        dump(folder / "records.json", records)
        manifest["splits"][split] = dict(
            count=len(records),
            tiles=len(tids),
            solid_instances=sum(r["labels"].count(0) for r in records),
            dash_instances=sum(r["labels"].count(1) for r in records),
            added_markings=sum(len(r["record"]["added"]) for r in records),
            templates_placed=sum("template" in r["record"] for r in records),
        )
    for k, v in real.items():
        manifest["real_splits"][k] = dict(
            crops=[c["crop_id"] for c in v],
            count=len(v),
            solid_instances=sum(c["labels"].count(0) for c in v),
            dash_instances=sum(c["labels"].count(1) for c in v),
            dataset=str(root),
            roots=sorted({c.get("_root", str(root)) for c in v}),
        )
    dump(out / "manifest.json", manifest)
    dump(OUT / "latest.json", {"run": str(out)})
    print(out)
    print(json.dumps({k: v for k, v in manifest.items() if k in ("splits", "templates")}, indent=1))
    print(
        {
            k: (v["count"], v["solid_instances"], v["dash_instances"])
            for k, v in manifest["real_splits"].items()
        }
    )


if __name__ == "__main__":
    sys.exit(main())
