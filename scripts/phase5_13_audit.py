"""Phase 5.13 Stages 2-3: training-label neighbourhood audit and actual sampling exposure."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
from affine import Affine
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation, distance_transform_edt

from phase5_11_common import EDGE, ROOT, client_features, load
from roadmarking.rasterization.pipeline import dump

DISTANCES = list(range(-2, 7))


def group(crop):
    return crop.get("phase") or ("legacy_val" if crop["split"] == "val" else "original")


def ring_stats(crop, designations, footprints):
    with np.load(crop["file"]) as d:
        gray = d["rgb"][..., 0].astype(float)
        masks, labels, valid = d["masks"].astype(bool), d["labels"], d["validity"].astype(bool)
    fg = masks.any(0) if len(masks) else np.zeros(valid.shape, bool)
    rows = []
    for m, cls, uid in zip(masks, labels, crop["global_instance_ids"], strict=True):
        inside, outside = distance_transform_edt(m), distance_transform_edt(~m)
        signed = np.where(m, -np.ceil(inside) + 1, np.ceil(outside))
        paint_median = float(np.median(gray[m & valid]))
        entry = {
            "cls": int(cls),
            "edge": designations.get(uid) in EDGE,
            "fg_px": int(m.sum()),
            "bands": {},
        }
        for dist in DISTANCES:
            band = signed == dist
            n = int(band.sum())
            if not n:
                continue
            v = band & valid
            bg = v & ~fg
            entry["bands"][dist] = {
                "pixels": n,
                "valid": int(v.sum()),
                "fg_any": int((band & fg).sum()),
                "background": int(bg.sum()),
                "ignored": int((band & ~valid).sum()),
                "intensity_valid_sum": float(gray[v].sum()),
                "bright_background": int((bg & (gray >= 0.75 * paint_median)).sum()),
            }
        fp = footprints.get((crop["crop_id"], uid))
        if fp is not None:
            entry["footprint_px"] = int(fp.sum())
            entry["footprint_valid_px"] = int((fp & valid).sum())
            entry["footprint_valid_labelled_background"] = int((fp & valid & ~fg).sum())
            entry["footprint_valid_labelled_other_target"] = int((fp & valid & fg & ~m).sum())
            entry["fg_outside_footprint_1px"] = int((m & ~binary_dilation(fp)).sum())
        rows.append(entry)
    return rows


def footprints_for(crops, stores):
    """Rasterise the approved painted footprints into each new crop's window."""
    out = {}
    for store in stores:
        paint = gpd.read_file(store, layer="candidate_paint_footprints")
        by_region = {k: v for k, v in paint.groupby("candidate_id")}
        for crop in crops:
            if crop["tile_id"] not in by_region:
                continue
            transform = Affine(*crop["affine"])
            for _, r in by_region[crop["tile_id"]].iterrows():
                if r.global_id not in crop["global_instance_ids"]:
                    continue
                out[(crop["crop_id"], r.global_id)] = rasterize(
                    [(r.geometry, 1)], (512, 512), transform=transform
                ).astype(bool)
    return out


def summarise(rows):
    agg = defaultdict(Counter)
    for r in rows:
        for dist, b in r["bands"].items():
            for k, v in b.items():
                agg[dist][k] += v
    table = {}
    for dist in sorted(agg, key=int):
        b = agg[dist]
        table[str(dist)] = {
            "valid_fraction": b["valid"] / max(1, b["pixels"]),
            "fg_fraction_of_valid": b["fg_any"] / max(1, b["valid"]),
            "background_fraction_of_valid": b["background"] / max(1, b["valid"]),
            "ignored_fraction": b["ignored"] / max(1, b["pixels"]),
            "mean_intensity_valid": b["intensity_valid_sum"] / max(1, b["valid"]),
            "bright_background_fraction_of_valid": b["bright_background"] / max(1, b["valid"]),
        }
    return table


def cross_tile(crops):
    """Same global pixel, different crops: target vs supervised-background contradictions."""
    tiles = defaultdict(list)
    for c in crops:
        tiles[c["tile_id"]].append(c)
    conflicts = compared = 0
    for members in tiles.values():
        canvas_fg, canvas_bg = {}, {}
        for c in members:
            with np.load(c["file"]) as d:
                masks, valid = d["masks"].astype(bool), d["validity"].astype(bool)
            fg = masks.any(0) if len(masks) else np.zeros(valid.shape, bool)
            r0, c0 = c["window"][0], c["window"][1]
            for kind, arr in (("fg", fg & valid), ("bg", valid & ~fg)):
                ys, xs = np.nonzero(arr)
                keys = (ys + r0) * 2048 + xs + c0
                (canvas_fg if kind == "fg" else canvas_bg).setdefault(c["crop_id"], set()).update(
                    keys.tolist()
                )
        ids = list(canvas_fg)
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                overlap = (canvas_fg[a] | canvas_bg[a]) & (canvas_fg[b] | canvas_bg[b])
                compared += len(overlap)
                conflicts += len((canvas_fg[a] & canvas_bg[b]) | (canvas_bg[a] & canvas_fg[b]))
    return {
        "overlapping_supervised_pixels_compared": compared,
        "target_vs_background_conflicts": conflicts,
    }


def exposure(run, manifest_crops, designations):
    log = load(Path(run) / "training_log.json")
    by_id = {c["crop_id"]: c for c in manifest_crops}
    cache = {}
    totals = Counter()
    for entry in log:
        crop = by_id[entry["crop_id"]]
        if crop["crop_id"] not in cache:
            with np.load(crop["file"]) as d:
                masks, labels = d["masks"].astype(bool), d["labels"]
            px = masks.sum((1, 2)) if len(masks) else np.zeros(0)
            cache[crop["crop_id"]] = {
                "solid_px": int(px[labels == 0].sum()),
                "dash_px": int(px[labels == 1].sum()),
                "edge_px": int(
                    sum(
                        p
                        for p, uid in zip(px, crop["global_instance_ids"], strict=True)
                        if designations.get(uid) in EDGE
                    )
                ),
                "long_solid_inst": int(((labels == 0) & (px >= 500)).sum()),
                "short_inst": int((px < 200).sum()),
                "solid": bool((labels == 0).any()),
                "dash": bool((labels == 1).any()),
                "negative": not len(labels),
                "group": group(crop),
                "valid_px": 0,
            }
        c = cache[crop["crop_id"]]
        totals["steps"] += 1
        totals["solid_positive_steps"] += c["solid"]
        totals["dash_positive_steps"] += c["dash"]
        totals["hard_negative_steps"] += c["negative"]
        totals[f"steps_{c['group']}"] += 1
        for k in ("solid_px", "dash_px", "edge_px", "long_solid_inst", "short_inst"):
            totals[k] += c[k]
        totals[f"stratum_{entry['sampling_stratum']}"] += 1
    t = dict(totals)
    t["solid_to_dash_pixel_ratio"] = t["solid_px"] / max(1, t["dash_px"])
    t["solid_px_per_step"] = t["solid_px"] / t["steps"]
    t["unique_crops_seen"] = len(cache)
    return t


def main():
    out = ROOT / "outputs/reports/phase5_13"
    designations = {
        f["global_id"]: f["designation"] for f in client_features(check_network=False)[0]
    }
    l12 = load(ROOT / "outputs/reports/phase5_12/latest.json")
    l11 = load(ROOT / "outputs/reports/phase5_11/latest.json")
    m12 = load(Path(l12["dataset"]) / "metadata/dataset_manifest.json")
    train = [c for c in m12["crops"] if c["split"] == "train"]
    stores = [
        load(Path(l11["report_dir"]) / "new_region_candidates.json")["candidate_store"],
        load(Path(l12["report_dir"]) / "new_region_candidates.json")["candidate_store"],
    ]
    new = [c for c in train if c.get("phase") in ("5.11", "5.12")]
    fps = footprints_for(new, stores)
    rows = defaultdict(list)
    for crop in train:
        for r in ring_stats(crop, designations, fps):
            key = (group(crop), "solid" if r["cls"] == 0 else "dash", r["edge"])
            rows[key].append(r)
    neighbourhood = {
        f"{g}|{c}|edge={e}": {
            "instances": len(v),
            "fg_px": sum(x["fg_px"] for x in v),
            "bands_signed_px_from_target": summarise(v),
        }
        for (g, c, e), v in sorted(rows.items())
    }
    contradictions = {}
    for (g, c, e), v in rows.items():
        withfp = [x for x in v if "footprint_px" in x]
        if not withfp:
            continue
        contradictions[f"{g}|{c}|edge={e}"] = {
            k: int(sum(x[k] for x in withfp))
            for k in (
                "footprint_px",
                "footprint_valid_px",
                "footprint_valid_labelled_background",
                "footprint_valid_labelled_other_target",
                "fg_outside_footprint_1px",
            )
        }
        contradictions[f"{g}|{c}|edge={e}"]["supervised_fraction_of_valid_footprint"] = sum(
            x["fg_px"] for x in withfp
        ) / max(1, sum(x["footprint_valid_px"] for x in withfp))
    class_ok = all(
        (lab == 0)
        == (
            designations.get(uid) is not None
            and designations[uid]
            in EDGE | {"Linha contínua", "Linha contínua na sequência de LBTg"}
        )
        or lab == 1
        for c in new
        for uid, lab in zip(c["global_instance_ids"], c["labels"], strict=True)
    )
    p512_solid = [c for c in train if c.get("phase") == "5.12" and 0 in c["labels"]]
    audit = {
        "neighbourhood_by_group_class_edge": neighbourhood,
        "footprint_vs_target_contradictions_new_crops": contradictions,
        "cross_tile_consistency_new_crops": cross_tile(new),
        "class_ids_consistent_with_client_designation": bool(class_ok),
        "phase5_12_crops_with_solid": len(p512_solid),
        "definitions": {
            "signed_px": "distance from the instance target mask; <=0 inside, 1 = first pixel "
            "outside",
            "bright_background": "valid background pixel >= 0.75 x the instance's median "
            "paint intensity (kerb/guardrail/bright surface proxy)",
        },
    }
    dump(out / "stage2_label_audit.json", audit)
    runs = {
        "phase5_7_expanded (produced default; 200 steps)": (
            ROOT / "models/mask2former_swin_s/phase5_7_expanded_20261106T111240383727Z",
            load(
                ROOT / "data/processed/phase5_training/phase5_7_expanded_20261106T111240383727Z/"
                "metadata/dataset_manifest.json"
            )["crops"],
        ),
        "phase5_11 (600 steps)": (
            Path(l11["training_run"]),
            load(Path(l11["dataset"]) / "metadata/dataset_manifest.json")["crops"],
        ),
        "phase5_12 (600 steps)": (Path(l12["training_run"]), m12["crops"]),
    }
    exp = {}
    for name, (run, crops) in runs.items():
        for c in crops:
            if not Path(c["file"]).is_absolute():
                c["file"] = str(
                    Path(
                        load(
                            ROOT / "data/processed/phase5_training/"
                            "phase5_7_expanded_20261106T111240383727Z/metadata/"
                            "dataset_manifest.json"
                        )["parent"]
                    )
                    / c["file"]
                )
        exp[name] = exposure(run, crops, designations)
    dump(out / "stage3_training_exposure.json", exp)
    print(
        json.dumps(
            {
                "contradictions": contradictions,
                "cross_tile": audit["cross_tile_consistency_new_crops"],
                "class_ok": class_ok,
                "exposure": exp,
            },
            indent=1,
        )
    )
    for key in neighbourhood:
        if "solid" in key:
            b = neighbourhood[key]["bands_signed_px_from_target"]
            print(
                key,
                neighbourhood[key]["instances"],
                {
                    d: (
                        round(v["mean_intensity_valid"], 1),
                        round(v["background_fraction_of_valid"], 2),
                        round(v["ignored_fraction"], 2),
                        round(v["bright_background_fraction_of_valid"], 2),
                    )
                    for d, v in b.items()
                    if d in ("0", "1", "2", "3", "4")
                },
            )


if __name__ == "__main__":
    main()
