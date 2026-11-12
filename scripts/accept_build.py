"""Independent real acceptance set from UNUSED client annotations + raw LAS (no human review).

Images use the production path (RasterAccumulator max-intensity, per-acquisition 1-99.5 %
calibration, gamma 0.7 grayscale). Labels come from the client's own per-feature vectors:
each dash is its own feature; solid = guide/continuous lines; class from DESIGNACAO.
Automatic checks replace human review (all thresholds fixed here, before any model sees it):

* instance accepted only if >= 50 % of its observed pixels are bright paint (gray >= local
  background + 40); otherwise it becomes ignore (uncertain alignment/wear);
* window rejected if > 25 % of its instances fail, or if unlabeled bright elongated
  structures cover > 0.5 % of observed pixels (possible missing annotation); smaller ones and
  all "Other Marks" (arrows, text, crossings) plus a 1 px ring around labels are ignored;
* windows lie >= 500 m from every training/validation/test/protected/N9_01 footprint.
"""

import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
from affine import Affine
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation, label, uniform_filter
from shapely.geometry import box
from shapely.ops import unary_union

from regsafe_common import ROOT, dump, sha256
from roadmarking.io.pointcloud import ChunkReadReport, iter_laz_chunks
from roadmarking.preprocessing.intensity import calibrate_histogram
from roadmarking.rasterization.rasterizer import RasterAccumulator, world_to_pixel
from roadmarking.visualization.intensity_render import grayscale_intensity
from stage9c_attested_pilot import crop_box
from stage9d_common import read
from stage9e_lib import development_crops, real_context

OUT = ROOT / "data/processed/acceptance_v1"
# Local verified copies: identical point counts and header bounds to the INPUTS\LAS originals.
SHARE = ROOT / "data/raw/pointclouds/acceptance_v1_source"
FILES = [
    "RIGHT/Job_20250708_0834_Track01_53.laz",
    "RIGHT/Job_20250708_0834_Track01_25.laz",
    "LEFT/Job_20250708_0834_Track02_2.laz",
    "LEFT/Job_20250708_0834_Track02_28.laz",
    "INTERCHANGE/Job_20250708_1218_Track01.laz",
    "INTERCHANGE/Job_20250708_1218_Track02.laz",
    "RIGHT/Job_20250708_0834_Track01_52.laz",
    "RIGHT/Job_20250708_0834_Track01_24.laz",
    "RIGHT/Job_20250708_0834_Track01_51.laz",
    "RIGHT/Job_20250708_0834_Track01_23.laz",
    "LEFT/Job_20250708_0834_Track02_30.laz",
    "LEFT/Job_20250708_0834_Track02_3.laz",
    "LEFT/Job_20250708_0834_Track02_29.laz",
    "LEFT/Job_20250708_0834_Track02_4.laz",
    "INTERCHANGE/Job_20250708_0834_Track17.laz",
    "INTERCHANGE/Job_20250708_1218_Track06.laz",
]
SIZE, CELL = 512, 0.05
SPACING_M, MIN_SEP_M, PER_FILE = 120.0, 80.0, 20
SOLID = {"Guia direita", "Guia esquerda", "Linha contínua", "Linha contínua na sequência de LBTg"}
DASH = {"Linha descontínua", "Linha de aceleração", "Linha de abrandamento"}
BRIGHT_DELTA, MIN_SUPPORT, MAX_FAILED, MAX_UNLABELED = 40, 0.5, 0.25, 0.005


def exclusion():
    rc = real_context()
    dev = [c for _, c in development_crops(rc["ctx"])]
    used = [crop_box(c) for c in rc["full"] + dev]
    reports = ROOT / "outputs/reports"
    prot = []
    for st in ("stage7_new_test", "stage8_new_solid_review"):
        f = reports / st
        p = f / "latest.json"
        d = (
            Path(read(p)["report_dir"])
            if p.exists()
            else next(x for x in f.iterdir() if (x / "new_region_candidates.json").is_file())
        )
        prot += [box(*t["bounds"]) for t in read(d / "new_region_candidates.json")["tiles"]]
    return unary_union(used + prot + [box(*rc["ctx"][9]["bounds"])]).buffer(500)


def load_lines():
    ann = ROOT / "data/raw/annotations"
    lines = pd.concat(
        [
            pyogrio.read_dataframe(ann / "Highway/Longitudinal Marks .gpkg"),
            pyogrio.read_dataframe(ann / "Interchange/Interchange-Longitudinal Marks.gpkg"),
        ]
    )
    lines = lines[lines.geometry.notna()].reset_index(drop=True)
    lines["width"] = lines["LARGURA"].astype(float)
    lines["label"] = lines["DESIGNACAO"].map(
        lambda d: 0 if d in SOLID else (1 if d in DASH else -1)
    )
    other = pd.concat(
        [
            pyogrio.read_dataframe(ann / "Highway/Other Marks.gpkg"),
            pyogrio.read_dataframe(ann / "Interchange/interchange_Other Marks.gpkg"),
        ]
    )
    other = other[other.geometry.notna()]
    return lines, other


def windows_for(header_box, free, taken):
    inner = header_box.buffer(-15)
    cands = []
    for geom in free:
        if geom.is_empty:
            continue
        for d in np.arange(0, geom.length, SPACING_M):
            p = geom.interpolate(d)
            if inner.contains(p):
                cands.append(p)
    out = []
    for p in cands:
        if all(p.distance(q) >= MIN_SEP_M for q in out + taken):
            out.append(p)
        if len(out) >= PER_FILE:
            break
    return out


def window_affine(p):
    x0 = np.floor((p.x - SIZE * CELL / 2) / CELL) * CELL
    y0 = np.ceil((p.y + SIZE * CELL / 2) / CELL) * CELL
    return Affine(CELL, 0, x0, 0, -CELL, y0)


def burn(geoms, transform):
    shapes = [(g, 1) for g in geoms if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros((SIZE, SIZE), bool)
    return rasterize(
        shapes,
        out_shape=(SIZE, SIZE),
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    ).astype(bool)


def label_window(gray, observed, transform, lines, other):
    region = box(
        *([transform.c, transform.f - SIZE * CELL, transform.c + SIZE * CELL, transform.f])
    )
    obs_f = observed.astype(np.float32)
    bg = uniform_filter(gray.astype(np.float32) * obs_f, 31) / np.maximum(
        uniform_filter(obs_f, 31), 1e-3
    )
    bright = observed & (gray >= bg + BRIGHT_DELTA)
    sel = lines[lines.intersects(region)]
    masks, labels, failed, ignore = [], [], 0, np.zeros_like(observed)
    any_ann = np.zeros_like(observed)
    stats = Counter()
    for _, r in sel.iterrows():
        geom = r.geometry.buffer(r.width / 2, cap_style="flat").intersection(region)
        m = burn([geom], transform)
        any_ann |= burn([r.geometry.buffer(0.5).intersection(region)], transform)
        if r.label < 0:
            ignore |= burn([r.geometry.buffer(r.width / 2 + 0.1).intersection(region)], transform)
            stats["unsupported_line_ignored"] += 1
            continue
        mo = m & observed
        if mo.sum() < 6:
            ignore |= m
            stats["too_little_observed"] += 1
            continue
        support = (mo & bright).sum() / mo.sum()
        if support < MIN_SUPPORT:
            ignore |= m
            failed += 1
            stats["failed_support"] += 1
            continue
        masks.append(m)
        labels.append(int(r.label))
        stats[f"accepted_class{int(r.label)}"] += 1
    og = other[other.intersects(region)]
    if len(og):
        ignore |= burn([g.buffer(0.15).intersection(region) for g in og.geometry], transform)
        any_ann |= burn([g.buffer(0.5).intersection(region) for g in og.geometry], transform)
        stats["other_marks_ignored"] += len(og)
    union = np.zeros_like(observed)
    for m in masks:
        union |= m
    ignore |= binary_dilation(union) & ~union  # 1 px annotation-edge uncertainty
    stray = bright & ~binary_dilation(any_ann, iterations=10)
    comps, n = label(stray, structure=np.ones((3, 3), bool))
    stray_px = 0
    for i in range(1, n + 1):
        c = comps == i
        if c.sum() < 30:
            continue
        ys, xs = np.nonzero(c)
        p = np.stack([ys, xs], 1).astype(float)
        p -= p.mean(0)
        ev = np.linalg.eigvalsh(np.cov(p.T))
        if ev[1] > 9 * max(ev[0], 1e-6):  # elongated, paint-like
            ignore |= binary_dilation(c, iterations=2)
            stray_px += int(c.sum())
    total = len(masks) + failed
    reject = None
    if total == 0:
        reject = "no_supported_instances"
    elif failed / total > MAX_FAILED:
        reject = "too_many_failed_instances"
    elif stray_px > MAX_UNLABELED * max(observed.sum(), 1):
        reject = "possible_missing_annotation"
    valid = observed & ~ignore
    masks = [m & valid for m in masks]
    keep = [i for i, m in enumerate(masks) if m.sum() >= 6]
    return dict(
        masks=[masks[i] for i in keep],
        labels=[labels[i] for i in keep],
        valid=valid,
        stats=dict(stats, stray_px=stray_px, failed=failed),
        reject=reject,
    )


def main():
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = OUT / stamp
    (out / "crops").mkdir(parents=True)
    excl = exclusion()
    lines, other = load_lines()
    free = [g for g in lines.geometry.difference(excl.buffer(20)) if not g.is_empty]
    taken, records, rejected = [], [], Counter()
    for rel in FILES:
        src = SHARE / Path(rel).name
        import laspy

        with laspy.open(src) as f:
            h = f.header
            hb = box(h.mins[0], h.mins[1], h.maxs[0], h.maxs[1])
        pts = windows_for(hb, free, taken)
        taken += pts
        wins = [(p, window_affine(p)) for p in pts]
        accs = [RasterAccumulator((SIZE, SIZE)) for _ in wins]
        hist = np.zeros(65536, np.uint64)
        report = ChunkReadReport(src, 0, 0)
        t0 = time.perf_counter()
        for chunk in iter_laz_chunks(src, report):
            x, y = np.asarray(chunk.x), np.asarray(chunk.y)
            raw = np.asarray(chunk.intensity)
            hist += np.bincount(raw, minlength=65536).astype(np.uint64)
            for (_p, tr), acc in zip(wins, accs, strict=True):
                k = (x >= tr.c) & (x < tr.c + SIZE * CELL) & (y <= tr.f) & (y > tr.f - SIZE * CELL)
                if not k.any():
                    continue
                cols, rows = world_to_pixel(tr, x[k], y[k])
                cols, rows = np.asarray(cols).astype(np.int64), np.asarray(rows).astype(np.int64)
                ok = (cols >= 0) & (rows >= 0) & (cols < SIZE) & (rows < SIZE)
                acc.add(cols[ok], rows[ok], raw[k][ok])
        cal = calibrate_histogram(hist, low=1, high=99.5)
        print(
            rel,
            "points",
            report.points_read,
            "windows",
            len(wins),
            round(time.perf_counter() - t0),
            "s",
            flush=True,
        )
        for i, ((_p, tr), acc) in enumerate(zip(wins, accs, strict=True)):
            layers = acc.layers(cal)
            observed = layers["observed"].astype(bool)
            gray = grayscale_intensity(np.clip(np.nan_to_num(layers["normalized"]), 0, 1), observed)
            lab = label_window(gray, observed, tr, lines, other)
            cid = f"ACC_{Path(rel).stem}_{i:02d}"
            if lab["reject"] or observed.mean() < 0.1:
                rejected[lab["reject"] or "sparse"] += 1
                continue
            path = out / "crops" / f"{cid}.npz"
            np.savez_compressed(
                path,
                rgb=np.repeat(gray[..., None], 3, axis=2),
                masks=np.asarray(lab["masks"], bool).reshape(len(lab["masks"]), SIZE, SIZE),
                labels=np.asarray(lab["labels"], np.int64),
                validity=lab["valid"],
            )
            records.append(
                dict(
                    crop_id=cid,
                    file=str(path),
                    sha256=sha256(path),
                    size=SIZE,
                    labels=lab["labels"],
                    affine=list(tr)[:6],
                    source=str(src),
                    native_xy_bbox=[tr.c, tr.f - SIZE * CELL, tr.c + SIZE * CELL, tr.f],
                    group=Path(rel).parent.name,
                    qa=lab["stats"],
                    calibration=cal.as_dict(),
                )
            )
        dump(out / "records.json", records)
    assert all(
        box(*r["native_xy_bbox"]).distance(excl.boundary) >= 0
        and not box(*r["native_xy_bbox"]).intersects(excl)
        for r in records
    )
    summary = dict(
        created=stamp,
        files=FILES,
        crops=len(records),
        rejected=dict(rejected),
        solid_instances=sum(r["labels"].count(0) for r in records),
        dash_instances=sum(r["labels"].count(1) for r in records),
        by_group=Counter(r["group"] for r in records),
        rules=dict(
            bright_delta=BRIGHT_DELTA,
            min_support=MIN_SUPPORT,
            max_failed=MAX_FAILED,
            max_unlabeled=MAX_UNLABELED,
            exclusion_buffer_m=500,
        ),
    )
    dump(out / "manifest.json", summary)
    dump(OUT / "latest.json", {"run": str(out)})
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())
