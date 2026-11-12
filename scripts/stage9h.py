"""Stage 9H sealed inference comparison on the Stage 9F development replay."""

import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from affine import Affine
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from roadmarking.evaluation.link_audit import classify_outputs, output_gaps
from roadmarking.inference.calibration import filter_instances, iou_matrix
from roadmarking.inference.identity_view_v1 import fragment_view
from roadmarking.inference.prediction_recovery import packed_iou
from roadmarking.inference.stitching import stitch
from roadmarking.inference.stitching_supported_v1 import stitch_supported_v1
from roadmarking.training.pilot import load_crop
from stage9c_attested_pilot import source_context
from stage9d_common import ROOT, dump, frozen_checks, read, sha256, verify_frozen
from stage9e_lib import V1, development_crops, proposals
from stage9g import EIGHT, STAGE9F, observed_crop, observed_sources, union

BASE = ROOT / "outputs/reports/stage9h"
G = ROOT / "outputs/reports/stage9g/20261112T082610231988Z"


def pack(masks, shape):
    return np.packbits(np.asarray(masks, bool).reshape(len(masks), int(np.prod(shape))), axis=1)


def prepare(entries, sources, model, processor, out):
    groups = {}
    for root, crop in entries:
        key = "legacy" if crop["tile_id"].startswith("INTERCHANGE") else crop["tile_id"]
        groups.setdefault(key, []).append((root, crop))
    for group, crops in groups.items():
        left = min(c["affine"][2] for _, c in crops)
        top = max(c["affine"][5] for _, c in crops)
        grid = Affine(0.05, 0, left, 0, -0.05, top)
        windows = {}
        for _, c in crops:
            col, row = ~grid * (c["affine"][2], c["affine"][5])
            pos = (round(row), round(col))
            assert (grid * Affine.translation(*pos[::-1])).almost_equals(
                Affine(*c["affine"]), precision=1e-7
            )
            windows[c["crop_id"]] = pos
        shape = tuple(max(p[axis] for p in windows.values()) + 512 for axis in (0, 1))
        obs, valid, seen = (np.zeros(shape, bool) for _ in range(3))
        masks, scores, classes, ids, query_keys = [], [], [], [], []
        support, targets, target_classes, crop_records = {}, {}, {}, {}
        for root, c in crops:
            cid = c["crop_id"]
            r, col = windows[cid]
            sl = np.s_[r : r + 512, col : col + 512]
            source = sources[c["tile_id"]]
            observed = observed_crop(source, c)
            x, gt, tl, v, _ = load_crop(root, c, processor, "cuda")
            local = v[0].cpu().numpy().astype(bool)
            assert not (local & ~observed).any()
            assert not (seen[sl] & (obs[sl] != observed)).any()
            obs[sl] |= observed
            valid[sl] |= local
            seen[sl] = True
            support[cid] = np.zeros(shape, bool)
            support[cid][sl] = local
            for uid, mask, cls in zip(
                c["global_instance_ids"], gt[0].cpu().numpy(), tl[0].tolist(), strict=True
            ):
                if uid not in targets:
                    targets[uid] = np.zeros(shape, bool)
                    target_classes[uid] = cls
                assert target_classes[uid] == cls
                targets[uid][sl] |= mask.astype(bool) & local
            p = proposals(model, x, local)
            pm, pl = p["masks"], p["labels"]
            selected = (
                filter_instances(
                    p["scores"],
                    pl,
                    iou_matrix(pm, pm, local),
                    p["qualities"],
                    dict(V1, confidence={"0": 0.0, "1": 0.0}),
                )
                if len(pm)
                else []
            )
            np.savez_compressed(
                out / "queries" / f"{cid}.npz",
                raw=pack(pm, local.shape),
                valid=local,
                observed=observed,
                query_indices=p["query_indices"],
                scores=p["scores"],
                classes=pl,
                selected_rows=selected,
            )
            crop_records[cid] = dict(
                crop_file=str(root / c["file"]),
                crop_sha256=c["sha256"],
                observed_source=source,
                native_affine=c["affine"],
                window=list(windows[cid]),
            )
            for i in selected:
                m = np.zeros(shape, bool)
                m[sl] = pm[i] & local
                masks.append(m)
                scores.append(float(p["scores"][i]))
                classes.append(int(pl[i]))
                ids.append(cid)
                query_keys.append([cid, p["query_indices"][i]])
        keys = list(targets)
        yield group, dict(
            masks=masks,
            scores=scores,
            classes=classes,
            ids=ids,
            query_keys=query_keys,
            windows=windows,
            support=support,
            observed=obs,
            valid=valid,
            targets=np.asarray(list(targets.values()), bool).reshape(len(keys), *shape),
            target_classes=np.asarray([target_classes[k] for k in keys], int),
            target_ids=keys,
            affine=list(grid)[:6],
            crops=crop_records,
        )


def measured(function):
    tracemalloc.start()
    start = time.perf_counter()
    try:
        result = function()
        return result, dict(
            seconds=time.perf_counter() - start,
            peak_traced_bytes=tracemalloc.get_traced_memory()[1],
        )
    finally:
        tracemalloc.stop()


def score(outputs, d, baseline=None):
    obs, valid = d["observed"], d["valid"]
    truth, tl = d["targets"], d["target_classes"]
    masks = np.asarray([e["mask"] for e in outputs], bool).reshape(len(outputs), *valid.shape)
    classes = np.asarray([e["class"] for e in outputs], int)
    iou = packed_iou(masks, truth, valid).reshape(len(masks), len(truth))
    physical_proxies = classify_outputs(masks, classes, range(len(masks)), truth, tl, valid)
    gaps = []
    for i, mask in enumerate(masks):
        members = [d["masks"][j] for j in outputs[i]["members"]]
        gaps.extend(
            dict(output=i, **g)
            for g in output_gaps(
                mask, valid, obs, union(truth, valid.shape), members, min_component=4
            )
            if g["unobserved_fraction"] > 0
        )
    result = {}
    for cls in (0, 1):
        sel, tgt = masks[classes == cls], truth[tl == cls]
        pred, gt = union(sel, valid.shape) & valid, union(tgt, valid.shape) & valid
        eligible = iou[np.ix_(classes == cls, tl == cls)] >= 0.5
        ri, ci = linear_sum_assignment(eligible, maximize=True)
        matched = eligible[ri, ci]
        tp = int(matched.sum())
        dup = int(eligible.any(1).sum()) - tp
        multiple_dashes = 0
        if cls == 1 and len(tgt):
            multiple_dashes = sum(
                int(((tgt & m).sum((1, 2)) >= max(1, 0.1 * m.sum())).sum() > 1) for m in sel
            )
        prev = baseline[cls] if baseline is not None else pred
        result[str(cls)] = dict(
            pixel_tp=int((pred & gt).sum()),
            pixel_fp=int((pred & ~gt).sum()),
            pixel_fn=int((gt & ~pred).sum()),
            instances=len(sel),
            targets=len(tgt),
            instance_tp=tp,
            instance_fp=len(sel) - tp,
            instance_fn=len(tgt) - tp,
            duplicates_iou50=dup,
            multi_dash_outputs=int(multiple_dashes),
            fragments=sum(label(m, EIGHT)[1] for m in sel),
            lost_observed_pixels=int((prev & ~pred).sum()),
            lost_observed_target_pixels=int((prev & ~pred & gt).sum()),
        )
    return dict(
        by_class=result,
        unsupported_pixels=int((union(masks, valid.shape) & ~obs).sum()),
        unsupported_gap_segments=gaps,
        physical_group_proxy=physical_proxies,
    )


def save_outputs(out, group, name, outputs, d):
    records = []
    for entry in outputs:
        record = {k: v for k, v in entry.items() if k != "mask"}
        record["queries"] = [d["query_keys"][i] for i in entry["members"]]
        records.append(record)
    np.savez_compressed(
        out / f"{group}_{name}_masks.npz",
        masks=pack([e["mask"] for e in outputs], d["valid"].shape),
        shape=d["valid"].shape,
        observed=d["observed"],
        valid=d["valid"],
    )
    return records


def compare_group(group, d, out):
    args = (d["masks"], d["scores"], d["classes"], d["ids"], d["windows"])
    orig, a_perf = measured(lambda: stitch(*args, d["valid"], V1["confidence"])[0])
    split, b_perf = measured(lambda: stitch_supported_v1(*args, d["support"], V1["confidence"])[0])
    (fragments, relations), c_perf = measured(
        lambda: fragment_view(orig, d["masks"], d["observed"], d["valid"])
    )
    baseline = [
        union(np.asarray([e["mask"] for e in orig if e["class"] == cls]), d["valid"].shape)
        for cls in (0, 1)
    ]
    rows = {}
    provenance = {}
    for name, outputs, perf in (
        ("A", orig, a_perf),
        ("B", split, b_perf),
        ("C", fragments, c_perf),
    ):
        rows[name] = dict(score(outputs, d, baseline), performance=perf)
        if name == "C":
            rows[name]["performance"]["incremental_to_A"] = True
            rows[name]["performance"]["end_to_end_stitch_seconds"] = (
                a_perf["seconds"] + c_perf["seconds"]
            )
        provenance[name] = save_outputs(out, group, name, outputs, d)
    for cls in ("0", "1"):
        for key in ("pixel_tp", "pixel_fp", "pixel_fn"):
            assert rows["A"]["by_class"][cls][key] == rows["C"]["by_class"][cls][key]
    dump(
        out / f"{group}_provenance.json",
        dict(
            native_affine=d["affine"],
            crops=d["crops"],
            target_ids=d["target_ids"],
            uncertain_relations=relations,
            outputs=provenance,
        ),
    )
    return rows


def main():
    checks = frozen_checks()
    verify_frozen(checks)
    for previous in (G, STAGE9F):
        for f in read(previous / "package_manifest.json")["files"]:
            assert sha256(ROOT / f["path"]) == f["sha256"]
    ctx = source_context()
    dev = development_crops(ctx)
    assert [c["crop_id"] for _, c in dev] == read(STAGE9F / "protocol.json")["development"]
    out = BASE / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    (out / "queries").mkdir(parents=True)
    dump(
        out / "protocol.json",
        dict(
            predeclared=True,
            frozen=checks,
            development=[c["crop_id"] for _, c in dev],
            A="Frozen NMS, existing stitching, V1 thresholds",
            B="Same eligible query pool, Stage 9G supported splitting and V1",
            C=(
                "A's accepted paint, exported as class-aware observed fragments "
                "and separate relations"
            ),
            metrics="Native valid pixel confusion and IoU, one-to-one same-class IoU >= .5, "
            "duplicate and multi-ID proxies, gap connectors, fragments, lost coverage, "
            "CPU wall time and peak traced allocations",
            veto="Zero unsupported accepted pixels. No verified physical ID without evidence.",
            scope=(
                "Repeated diagnostic only. No training, tuning, "
                "protected test, promotion or text."
            ),
            threshold=V1,
        ),
    )
    frozen = Path(checks[0]["path"]).parent
    processor = Mask2FormerImageProcessor.from_pretrained(
        frozen.parent / "processor", local_files_only=True
    )
    model = (
        Mask2FormerForUniversalSegmentation.from_pretrained(frozen, local_files_only=True)
        .cuda()
        .eval()
    )
    torch.set_num_threads(4)
    sources = observed_sources(ctx[3])
    results = {
        group: compare_group(group, d, out)
        for group, d in prepare(dev, sources, model, processor, out)
    }
    verify_frozen(checks)
    dump(out / "metrics.json", results)
    print(out, flush=True)


if __name__ == "__main__":
    main()
