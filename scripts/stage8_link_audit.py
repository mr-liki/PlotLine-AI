"""Stage 8 Task 1: stitching-safety audit with exact geometry and visual evidence.

Re-creates, without changing any rule, (a) the Stage-3 P_stitch outputs on the held-out
LORO pools (default / R0 / R1, the 7F sources) in each tile's own pixel frame, and (b) the
production pipelines on the development areas (default V0; frozen candidate V1 +
stitching). Every output joining >= 2 IDs, or containing a gap without observed paint, is
classified by physical identity and by what lies between its parts (observed paint,
observed unpainted road, LiDAR NoData). A NoData crossing without paint support is never
accepted as successful recognition. Results are reconciled against the 7F totals.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pyogrio
import rasterio
from affine import Affine
from PIL import Image
from rasterio.features import shapes
from shapely.geometry import LineString, shape
from shapely.ops import unary_union

from phase5_11_common import ROOT, load
from roadmarking.evaluation.link_audit import audit_connections, physical_lines
from roadmarking.inference.prediction_recovery import filter_geographic
from roadmarking.inference.stitching import stitch
from roadmarking.rasterization.pipeline import dump
from stage1_scoring import DEFAULT, crop_nms, fit_thresholds, load_pool, variant_score
from stage7_common import settings
from stage7h_evaluate import build

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = ROOT / "outputs/reports/stage8/8_1_link_audit"
SOURCES = ("default", "R0_baseline", "R1_bilinear")
CATEGORIES = (
    "continuous_marking",
    "separate_dashes",
    "different_or_parallel",
    "nodata_crossing_unsupported",
    "observed_gap_unpainted",
)


def tile_specs():
    tiles = {}
    for phase in ("phase5_11", "phase5_12"):
        spec = load(
            Path(load(ROOT / f"outputs/reports/{phase}/latest.json")["report_dir"])
            / "new_region_candidates.json"
        )
        tiles.update({t["candidate_id"]: t for t in spec["tiles"]})
    return tiles


def stage3_region(crops, thr, tile):
    ("Stage-3 P_stitch on one pool region, in the tile's own pixel frame " "(with members).")
    with rasterio.open(tile["layers"]["observed"]) as ds:
        observed = ds.read(1).astype(bool)
    shape_ = observed.shape
    valid = np.zeros(shape_, bool)
    truths, labels = {}, {}
    masks, scores, classes, crop_ids, pos = [], [], [], [], {}
    for crop in crops:
        c = crop["crop"]
        r0, c0 = c["window"][0], c["window"][1]
        pos[c["crop_id"]] = (r0, c0)
        sl = (slice(r0, r0 + 512), slice(c0, c0 + 512))
        valid[sl] |= crop["valid"]
        for m, lab, uid in zip(crop["targets"], crop["tl"], c["global_instance_ids"], strict=True):
            truths.setdefault(uid, np.zeros(shape_, bool))[sl] |= m
            labels[uid] = int(lab)
        s = variant_score(crop["rows"], "V1_refit_thresholds")
        for k in crop_nms(crop["rows"], s):
            g = np.zeros(shape_, bool)
            g[sl] = crop["rows"][k]["mask"]
            masks.append(g)
            scores.append(float(s[k]))
            classes.append(crop["rows"][k]["class"])
            crop_ids.append(c["crop_id"])
    ids = sorted(truths)
    targets = np.stack([truths[u] for u in ids]) if ids else np.zeros((0, *shape_), bool)
    tl = np.asarray([labels[u] for u in ids], int)
    inst, _ = stitch(masks, scores, classes, crop_ids, pos, valid, {"0": -1.0, "1": -1.0})
    keep = [k for k, i in enumerate(inst) if i["score"] >= thr[str(i["class"])]]
    sm = np.stack([i["mask"] for i in inst]) if inst else np.zeros((0, *shape_), bool)
    sc = np.asarray([i["class"] for i in inst], int)
    members = [[masks[m] for m in i["members"]] for i in inst]
    return {
        "masks": sm,
        "classes": sc,
        "keep": keep,
        "members": members,
        "targets": targets,
        "tl": tl,
        "ids": ids,
        "valid": valid,
        "observed": observed,
        "affine": Affine(*tile["affine"]),
        "image": np.asarray(Image.open(tile["images"]["grayscale"]).convert("L")),
    }


def production(comp, thr, use_stitch):
    ("final_candidate.pipeline, identical rules, additionally returning " "member masks.")
    if not use_stitch:
        keep = filter_geographic(
            comp["scores"],
            comp["classes"],
            comp["qualities"],
            comp["overlaps"],
            comp["crop_ids"],
            thr,
        )
        members = [[comp["masks"][i]] for i in range(len(comp["scores"]))]
        return comp["masks"], comp["classes"], keep, members
    local = []
    for crop in comp["crops"]:
        idx = [i for i, c in enumerate(comp["crop_ids"]) if c == crop["crop_id"]]
        part = filter_geographic(
            comp["scores"][idx],
            comp["classes"][idx],
            comp["qualities"][idx],
            comp["overlaps"][np.ix_(idx, idx)],
            [crop["crop_id"]] * len(idx),
            {"confidence": {"0": 0.0, "1": 0.0}, "mask_quality": 0.0, "nms_iou": 0.3},
        )
        local += [idx[k] for k in part]
    windows = {c["crop_id"]: (c["row"], c["col"]) for c in comp["crops"]}
    lm = [comp["masks"][i] for i in local]
    inst, _ = stitch(
        lm,
        [float(comp["scores"][i]) for i in local],
        [int(comp["classes"][i]) for i in local],
        [comp["crop_ids"][i] for i in local],
        windows,
        comp["valid"],
        thr["confidence"],
    )
    masks = np.stack([i["mask"] for i in inst]) if inst else np.zeros((0, *comp["shape"]), bool)
    classes = np.array([i["class"] for i in inst], int)
    members = [[lm[m] for m in i["members"]] for i in inst]
    return masks, classes, list(range(len(inst))), members


def comp_affine(comp):
    """Component pixel frame -> native coordinates, from any member crop."""
    c = comp["crops"][0]
    return Affine(*c["affine"][:6]) * Affine.translation(-c["col"], -c["row"])


def to_native(affine, rc):
    x, y = affine * (rc[1] + 0.5, rc[0] + 0.5)
    return [float(x), float(y)]


def polygon(mask, affine):
    geoms = [shape(g) for g, v in shapes(mask.astype(np.uint8), mask=mask, transform=affine) if v]
    return unary_union(geoms) if geoms else None


def panel(path, region, rec, title):
    m = region["masks"][rec["output"]]
    hits = [region["targets"][j] for j in rec["touched_ids"]]
    sel = m.copy()
    for h in hits:
        sel |= h
    rows, cols = np.nonzero(sel)
    pad = 25
    r0, r1 = max(0, rows.min() - pad), min(m.shape[0], rows.max() + pad)
    c0, c1 = max(0, cols.min() - pad), min(m.shape[1], cols.max() + pad)
    img = np.repeat((region["image"][r0:r1, c0:c1] / 255.0)[..., None], 3, -1)
    unobs = ~region["observed"][r0:r1, c0:c1]
    img[unobs] = img[unobs] * 0.35 + np.array([0.05, 0.1, 0.35])
    om = m[r0:r1, c0:c1]
    img[om] = img[om] * 0.45 + np.array([0.55, 0.0, 0.55])
    colours = [(0.1, 0.9, 0.9), (1.0, 0.9, 0.1), (0.3, 1.0, 0.3), (1.0, 0.5, 0.2)]
    for k, h in enumerate(hits):
        hb = h[r0:r1, c0:c1]
        edge = hb & ~np.pad(hb, 1)[2:, 1:-1] | hb & ~np.pad(hb, 1)[:-2, 1:-1]
        img[edge] = colours[k % len(colours)]
    fig, ax = plt.subplots(figsize=(min(14, 2 + (c1 - c0) / 60), min(8, 2 + (r1 - r0) / 60)))
    ax.imshow(np.clip(img, 0, 1), interpolation="nearest")
    style = {"nodata_unsupported": "r", "observed_unpainted": "orange", "paint_supported": "lime"}
    for g in rec["gaps"]:
        ax.plot(
            [g["from_px"][1] - c0, g["to_px"][1] - c0],
            [g["from_px"][0] - r0, g["to_px"][0] - r0],
            color=style[g["kind"]],
            lw=2,
        )
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def summarise(records, outputs):
    by_class = Counter(r["output_class"] for r in records)
    cross = defaultdict(Counter)
    for r in records:
        cross[r["identity"]][r["category"]] += 1
    gaps = [g for r in records for g in r["gaps"]]
    nodata = [r for r in records if r["category"] == "nodata_crossing_unsupported"]
    return {
        "denominators": outputs,
        "questionable_outputs": len(records),
        "category_counts": {c: sum(r["category"] == c for r in records) for c in CATEGORIES},
        "identity_by_category": {k: dict(v) for k, v in cross.items()},
        "questionable_by_output_class": {"solid": by_class[0], "dash": by_class[1]},
        "not_accepted": sum(not r["accepted"] for r in records),
        "not_accepted_rate_of_outputs": sum(not r["accepted"] for r in records)
        / max(1, outputs["outputs"]),
        "nodata_crossings": {
            "outputs": len(nodata),
            "connections": sum(g["kind"] == "nodata_unsupported" for g in gaps),
            "by_bridge": dict(
                Counter(g["bridged_by"] for g in gaps if g["kind"] == "nodata_unsupported")
            ),
            "by_identity": dict(Counter(r["identity"] for r in nodata)),
            "single_id_outputs_bridging_nodata": sum(r["identity"] == "single_id" for r in nodata),
        },
        "stage7f_reconstruction": {
            "multi_id_outputs": sum(r["stage7f_counted"] for r in records),
            "stage7f_nodata_flags": sum(r["stage7f_nodata_bridge"] for r in records),
            "stage3_rule_false_links": sum(
                r["stage7f_counted"] and r["output_class"] == 0 and r["solid_hits"] >= 2
                for r in records
            ),
        },
    }


def audit_region(region, name, source, run, evidence, rows):
    lines = physical_lines(region["targets"], region["tl"])
    recs = audit_connections(
        region["masks"],
        region["classes"],
        region["keep"],
        region["targets"],
        region["tl"],
        region["valid"],
        region["observed"],
        region["members"],
        lines,
    )
    aff = region["affine"]
    for rec in recs:
        rec["solid_hits"] = int(sum(region["tl"][j] == 0 for j in rec["touched_ids"]))
        rec["touched_global_ids"] = [region["ids"][j] for j in rec["touched_ids"]]
        rec.update(source=source, run=run, region=name)
        for g in rec["gaps"]:
            g["from_native"] = to_native(aff, g["from_px"])
            g["to_native"] = to_native(aff, g["to_px"])
        key = f"{source}_{run}_{name}_o{rec['output']}"
        rec["evidence_png"] = str(evidence / f"{key}.png")
        if not rec["accepted"] or rec["identity"] != "single_id":
            panel(
                rec["evidence_png"],
                region,
                rec,
                f"{key} | {rec['category']} | identity={rec['identity']} | "
                f"members={rec['members']} (red=NoData gap, orange=unpainted, lime=paint)",
            )
        geom = polygon(region["masks"][rec["output"]] & region["valid"], aff)
        rows["outputs"].append(
            {
                "key": key,
                "source": source,
                "region": name,
                "category": rec["category"],
                "identity": rec["identity"],
                "accepted": int(rec["accepted"]),
                "touched_ids": ",".join(rec["touched_global_ids"]),
                "geometry": geom,
            }
        )
        for g in rec["gaps"]:
            rows["connections"].append(
                {
                    "key": key,
                    "source": source,
                    "region": name,
                    "kind": g["kind"],
                    "bridged_by": g["bridged_by"],
                    "length_native": g["length_px"] * 0.05,
                    "unobserved_fraction": g["unobserved_fraction"],
                    "painted_fraction": g["painted_fraction"],
                    "geometry": LineString([g["from_native"], g["to_native"]]),
                }
            )
    keep = region["keep"]
    counts = {
        "outputs": len(keep),
        "output_solids": int(sum(region["classes"][k] == 0 for k in keep)),
        "output_dashes": int(sum(region["classes"][k] == 1 for k in keep)),
    }
    return recs, counts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    folds = load(ROOT / "outputs/reports/stage0_harness/folds.json")["folds"]
    l12 = load(ROOT / "outputs/reports/phase5_12/latest.json")
    manifest = load(Path(l12["dataset"]) / "metadata/dataset_manifest.json")
    pool_regions = {r: f["fold"] for f in folds for r in f["regions"] if r.startswith("P51")}
    tiles = tile_specs()
    old = load(ROOT / "outputs/reports/stage7/7F_stage3_link_breakdown.json")
    rows = {"outputs": [], "connections": []}
    result = {}
    for source in SOURCES:
        evidence = OUT / "evidence" / source
        evidence.mkdir(parents=True, exist_ok=True)
        if source == "default":
            sets = [
                (
                    "default",
                    [
                        c
                        for c in manifest["crops"]
                        if c["split"] == "train" and c["tile_id"] in pool_regions
                    ],
                    ROOT / "outputs/reports/stage1_scoring/default_pool_inference",
                )
            ]
        else:
            sets = [
                (
                    f"s{seed}",
                    [
                        c
                        for c in manifest["crops"]
                        if c["split"] == "train"
                        and c["tile_id"] in set(f["regions"])
                        and c["tile_id"].startswith("P51")
                    ],
                    ROOT / f"outputs/reports/stage2_loro/inference/{source}_s{seed}_f{f['fold']}",
                )
                for seed in (895, 896)
                for f in folds
            ]
        by_tag = defaultdict(lambda: defaultdict(list))
        for tag, pool, folder in sets:
            for crop in load_pool(pool, folder).values():
                by_tag[tag][crop["crop"]["tile_id"]].append(crop)
        records, counts = [], Counter()
        for tag, regions in by_tag.items():
            for f in range(5):
                fit = [c for r, k in pool_regions.items() if k != f for c in regions.get(r, [])]
                thr = fit_thresholds(fit, "V1_refit_thresholds")
                for r in [r for r, k in pool_regions.items() if k == f and r in regions]:
                    region = stage3_region(regions[r], thr, tiles[r])
                    recs, cnt = audit_region(region, r, source, tag, evidence, rows)
                    records += recs
                    counts.update(cnt)
            print(f"{source} {tag}: {len(records)} questionable", flush=True)
        summ = summarise(records, dict(counts))
        b = old[source]["breakdown"]
        summ["stage7f_reconciliation"] = {
            "stage7f_output_instances": b["output_instances"],
            "recomputed_output_instances": counts["outputs"],
            "stage7f_output_solids": b["output_solids"],
            "recomputed_output_solids": counts["output_solids"],
            "stage7f_multi_id_outputs": sum(
                b.get(k, 0)
                for k in (
                    "same_physical_line",
                    "dashes_joined",
                    "solid_dash_joined",
                    "parallel_lines_joined",
                    "different_lines_joined",
                )
            ),
            "recomputed_multi_id_outputs": summ["stage7f_reconstruction"]["multi_id_outputs"],
            "stage7f_nodata_bridges": b.get("nodata_bridges", 0),
            "recomputed_stage7f_nodata_flags": summ["stage7f_reconstruction"][
                "stage7f_nodata_flags"
            ],
            "stage7f_stage3_rule": b.get("stage3_rule_false_links", 0),
            "recomputed_stage3_rule": summ["stage7f_reconstruction"]["stage3_rule_false_links"],
        }
        result[source] = {"summary": summ, "records": records}
    # Production pipelines on the development areas (Stage-7 reference inference).
    cfg = settings()
    p59 = load(ROOT / "outputs/reports/phase5_9/latest.json")
    base = ROOT / "outputs/reports/stage7/7D_S6_01_baseline/inference"
    folders = {
        "default_V0": (
            DEFAULT / "validation_step0050",
            Path(p59["report_dir"]) / "inference",
            base / "default_V0",
        ),
        "frozen_candidate_V1_stitch": (
            ROOT / "outputs/reports/final_candidate/inference/legacy",
            ROOT / "outputs/reports/final_candidate/inference/new",
            base / "frozen_candidate_V1_stitch",
        ),
    }
    for name, (leg, new, s6) in folders.items():
        evidence = OUT / "evidence" / name
        evidence.mkdir(parents=True, exist_ok=True)
        comps, _ = build(leg, new, s6)
        c = cfg[name]
        records, counts = [], Counter()
        for cname, comp in comps.items():
            masks, classes, keep, members = production(comp, c["thresholds"], c["stitch"])
            region = {
                "masks": masks,
                "classes": classes,
                "keep": keep,
                "members": members,
                "targets": comp["targets"],
                "tl": np.asarray(comp["target_labels"], int),
                "ids": comp["ids"],
                "valid": comp["valid"],
                "observed": comp["observed"],
                "affine": comp_affine(comp),
                "image": comp["image"],
            }
            recs, cnt = audit_region(region, cname, name, "dev", evidence, rows)
            records += recs
            counts.update(cnt)
        result[name] = {"summary": summarise(records, dict(counts)), "records": records}
        print(f"{name}: {len(records)} questionable", flush=True)
    store = OUT / "link_evidence.gpkg"
    if store.exists():
        store.unlink()
    for layer, frame in rows.items():
        frame = [r for r in frame if r["geometry"] is not None]
        if frame:
            pyogrio.write_dataframe(
                gpd.GeoDataFrame(frame, geometry="geometry", crs=None),
                store,
                layer=layer,
                driver="GPKG",
                promote_to_multi=True,
                append=store.exists(),
            )
    dump(OUT / "link_audit.json", result)
    print(
        json.dumps(
            {
                k: {
                    kk: v["summary"][kk]
                    for kk in (
                        "category_counts",
                        "nodata_crossings",
                        "not_accepted_rate_of_outputs",
                    )
                }
                | (
                    {"reconciliation": v["summary"]["stage7f_reconciliation"]}
                    if "stage7f_reconciliation" in v["summary"]
                    else {}
                )
                for k, v in result.items()
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
