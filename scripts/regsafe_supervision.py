"""Step 2-3: clean synthetic supervision, real-image statistics and the real-first exposure plan.

Synthetic (supplied reconstructed geometry, 500 images):
* A target pixel must agree on all three rasters: instance id, semantic class and paint.
* The 395 JSON/raster count disagreements are audited per object (direction, overdraw
  evidence, class). Every raster pixel is kept only where all layers agree.
* Unresolved paint (paint without an instance, small/overdraw-contact dash shards,
  zero-raster objects) and unsupported classes (text, arrows, crossings, double lines,
  stop lines, ...) are IGNORED, never background.
* Synthetic NoData is never inferred from dark pixels.
* Dash sequences are split into connected visible pieces; a piece touching another
  instance's overdraw is ignored because its identity cannot be certified.

Real: the approved 486 crops with their original validity (NoData + ignore masks).
"""

import csv
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt, label
from skimage.morphology import skeletonize

from regsafe_common import BASE, SYN_PACKAGE, SYN_RAW, dump, frozen_paths, read, sha256
from stage9c_attested_pilot import crop_box
from stage9e_lib import development_crops, real_context, with_root

HOLDOUT = {"worn_lane", "mixed_intersection", "zebra", "arrow_left", "text"}
LINE = {0: 1, 1: 0}  # package class -> model label (lane_dashed -> dash 1, lane_solid -> solid 0)
MIN_DASH_PX, MIN_SOLID_PX = 16, 4


def png(path):
    with Image.open(path) as im:
        a = np.array(im)
    assert a.shape == (512, 512), path
    return a


def width_proxy(mask):
    """Median full stroke width: 2 x distance-to-edge sampled on the skeleton."""
    if mask.sum() < 4:
        return None
    sk = skeletonize(mask)
    if not sk.any():
        return None
    return float(2 * np.median(distance_transform_edt(mask)[sk]) - 1)


def convert_synthetic(out):
    taxonomy = read(SYN_PACKAGE / "CLASS_MAP.json")["classes"]
    assert taxonomy["0"] == "lane_dashed" and taxonomy["1"] == "lane_solid"
    raw = {p.name: p for p in SYN_RAW.rglob("*.png")}
    with (SYN_PACKAGE / "manifest.csv").open(newline="", encoding="utf-8-sig") as s:
        manifest = list(csv.DictReader(s))
    assert len(manifest) == len(raw) == 500
    folder = out / "synthetic_clean"
    folder.mkdir()
    audit = Counter()
    disagreements = []
    records = []
    for row in manifest:
        name = row["image"]
        cid = Path(name).stem
        image = SYN_PACKAGE / "images" / name
        assert sha256(image) == sha256(raw[name]), name
        img = png(image)
        sem, inst, paint = (
            png(SYN_PACKAGE / k / name) for k in ("semantic_masks", "instance_masks", "paint_masks")
        )
        meta = read(SYN_PACKAGE / "annotations_json" / f"{cid}.json")
        objects = meta["objects"]
        assert len(objects) == int(row["objects"])
        valid = (sem == 0) & (inst == 0) & (paint == 0)  # verified generator background
        audit["paint_without_instance_px_ignored"] += int(((paint != 0) & (inst == 0)).sum())
        targets, labels, stats = [], [], []
        for obj in objects:
            cls = obj["class_id"]
            mask = inst == obj["id"]
            n = int(mask.sum())
            consistent = mask & (sem == cls + 1) & (paint != 0)
            audit["layer_disagreement_px_ignored"] += int((mask & ~consistent).sum())
            others = (inst != 0) & (inst != obj["id"])
            if n != obj["pixel_count"]:
                x0, y0, x1, y1 = obj["bbox_xyxy"]
                bb = np.zeros_like(mask)
                bb[y0:y1, x0:x1] = True
                near = binary_dilation(mask, iterations=3) & bb if n else bb
                overdraw = bool((others & near).any())
                deficit = obj["pixel_count"] - n
                kind = (
                    "zero_raster_unresolved"
                    if n == 0
                    else ("overdraw_by_later_object" if overdraw else "unexplained_deficit")
                )
                audit[f"disagreement_{kind}"] += 1
                audit[
                    "disagreement_raster_fewer" if deficit > 0 else "disagreement_raster_more"
                ] += 1
                disagreements.append(
                    dict(
                        image=name,
                        object=obj["id"],
                        class_name=obj["class_name"],
                        raster_px=n,
                        json_px=obj["pixel_count"],
                        deficit=deficit,
                        kind=kind,
                        line_class=cls in LINE,
                    )
                )
                if kind == "unexplained_deficit" and cls in LINE:
                    # Untrustworthy line object: ignore all of it.
                    audit["line_objects_ignored_unexplained"] += 1
                    continue
            if cls not in LINE:
                audit[f"unsupported_ignored_{obj['class_name']}"] += 1
                continue
            if cls == 0:
                comps, k = label(consistent, structure=np.ones((3, 3), bool))
                pieces = [comps == i for i in range(1, k + 1)]
            else:
                pieces = [consistent]
            contact = binary_dilation(others, iterations=1)
            for piece in pieces:
                if piece.sum() < (MIN_DASH_PX if cls == 0 else MIN_SOLID_PX):
                    audit["small_line_pieces_ignored"] += 1
                    continue
                if cls == 0 and (piece & contact).any():
                    audit["dash_pieces_touching_overdraw_ignored"] += 1
                    continue
                valid |= piece
                targets.append(piece)
                labels.append(LINE[cls])
                stats.append(
                    dict(
                        label=LINE[cls],
                        px=int(piece.sum()),
                        width=width_proxy(piece),
                        contrast=float(np.median(img[piece]) - np.median(img[valid & ~piece])),
                    )
                )
        masks = np.asarray(targets, bool).reshape(len(targets), 512, 512)
        assert not (masks & ~valid).any() and valid.any()
        path = folder / f"{cid}.npz"
        np.savez_compressed(
            path,
            rgb=np.repeat(img[..., None], 3, axis=2),
            masks=masks,
            labels=np.asarray(labels, np.int64),
            validity=valid,
        )
        fg = masks.any(0) if len(masks) else np.zeros_like(valid)
        bgv = img[valid & ~fg]
        records.append(
            dict(
                crop_id=cid,
                file=str(path),
                sha256=sha256(path),
                size=512,
                labels=labels,
                geography="synthetic",
                theme=row["source_theme"],
                role="development" if row["source_theme"] in HOLDOUT else "train",
                background_median=float(np.median(bgv)) if len(bgv) else None,
                background_std=float(bgv.std()) if len(bgv) else None,
                paint_median=float(np.median(img[fg])) if fg.any() else None,
                width_median=(
                    float(np.median([s["width"] for s in stats if s["width"]]))
                    if any(s["width"] for s in stats)
                    else None
                ),
                line_objects=stats,
                ignored_px=int((~valid).sum()),
            )
        )
    return records, dict(audit), disagreements


def real_statistics(crops, dataset):
    rows = []
    for c in crops:
        with np.load(Path(dataset) / c["file"]) as d:
            g = d["rgb"][..., 0]
            valid = d["validity"]
            masks = d["masks"].astype(bool)
            labels = d["labels"]
        fg = masks.any(0) if len(masks) else np.zeros_like(valid)
        obs = g > 0
        bg = g[valid & ~fg & obs]
        if not len(bg):
            continue
        r = dict(
            crop_id=c["crop_id"],
            background_median=float(np.median(bg)),
            background_std=float(bg.std()),
            observed_fraction=float(obs.mean()),
            valid_fraction=float(valid.mean()),
        )
        # Isolated unobserved pixels inside the observed hull: interior sparsity.
        hull = binary_dilation(obs, iterations=2)
        r["interior_hole_fraction"] = float((hull & ~obs).sum() / max(hull.sum(), 1))
        if (fg & valid).any():
            r["paint_median"] = float(np.median(g[fg & valid]))
            r["contrast"] = r["paint_median"] - r["background_median"]
            widths = [width_proxy(m & valid) for m in masks]
            widths = [w for w in widths if w is not None]
            r["width_median"] = float(np.median(widths)) if widths else None
            r["labels"] = [int(x) for x in labels]
        rows.append(r)
    return rows


def pct(values, q=(5, 25, 50, 75, 95)):
    a = np.asarray([v for v in values if v is not None], float)
    return {f"p{p}": float(np.percentile(a, p)) for p in q} | {"n": len(a)} if len(a) else None


def main():
    out = BASE / "supervision" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True)
    frozen_paths()
    syn, audit, disagreements = convert_synthetic(out)
    rc = real_context()
    real_train = with_root(rc["full"], rc["dataset"])
    dev = development_crops(rc["ctx"])
    dev_ids = {c["crop_id"] for _, c in dev}
    dev_boxes = [crop_box(c) for _, c in dev]
    dev_tiles = {c.get("tile_id") for _, c in dev}
    overlap = [
        c["crop_id"]
        for c in real_train
        if c["crop_id"] in dev_ids
        or c.get("tile_id") in dev_tiles
        or any(crop_box(c).intersects(b) for b in dev_boxes)
    ]
    real_stats = real_statistics(real_train, rc["dataset"])
    labeled = [r for r in real_stats if "contrast" in r]
    syn_train = [r for r in syn if r["role"] == "train"]
    syn_dev = [r for r in syn if r["role"] == "development"]
    groups = defaultdict(set)
    for c in real_train:
        for g in c["global_instance_ids"]:
            groups[g].add(c["crop_id"])
    summary = dict(
        synthetic_audit=audit,
        disagreement_total=len(disagreements),
        disagreement_by_kind=Counter(d["kind"] for d in disagreements),
        disagreement_line_objects=sum(d["line_class"] for d in disagreements),
        synthetic_split={
            role: dict(
                images=len(rs),
                solid_targets=sum(r["labels"].count(0) for r in rs),
                dash_targets=sum(r["labels"].count(1) for r in rs),
                solid_images=sum(0 in r["labels"] for r in rs),
                dash_images=sum(1 in r["labels"] for r in rs),
                negative_images=sum(not r["labels"] for r in rs),
            )
            for role, rs in (("train", syn_train), ("development", syn_dev))
        },
        real_train=dict(
            crops=len(real_train),
            solid_crop_instances=sum(c["labels"].count(0) for c in real_train),
            dash_crop_instances=sum(c["labels"].count(1) for c in real_train),
            negative_crops=sum(not c["labels"] for c in real_train),
            physical_marking_groups=len(groups),
            crops_per_group=pct([len(v) for v in groups.values()]),
        ),
        development_geography_overlap=overlap,
        protected=dict(
            population_members_checked=rc["population_members"],
            protected_boxes_checked=rc["protected_boxes"],
            n9_01="excluded by real_context footprint assertion",
        ),
        image_statistics=dict(
            real=dict(
                background_median=pct([r["background_median"] for r in real_stats]),
                background_std=pct([r["background_std"] for r in real_stats]),
                contrast=pct([r["contrast"] for r in labeled]),
                width=pct([r.get("width_median") for r in labeled]),
                observed_fraction=pct([r["observed_fraction"] for r in real_stats]),
                interior_hole_fraction=pct([r["interior_hole_fraction"] for r in real_stats]),
            ),
            synthetic=dict(
                background_median=pct([r["background_median"] for r in syn]),
                background_std=pct([r["background_std"] for r in syn]),
                contrast=pct([o["contrast"] for r in syn for o in r["line_objects"]]),
                width=pct([o["width"] for r in syn for o in r["line_objects"]]),
            ),
        ),
    )
    if overlap:
        raise RuntimeError(f"Validation geography overlaps training: {overlap[:5]}")
    dump(out / "supervision_audit.json", summary)
    dump(out / "disagreements.json", disagreements)
    dump(out / "synthetic_records.json", syn)
    dump(out / "real_image_statistics.json", real_stats)
    dump(BASE / "supervision" / "latest.json", {"run": str(out)})
    print(out)


if __name__ == "__main__":
    sys.exit(main())
