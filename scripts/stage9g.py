"""Read-only Stage 9G comparison on the exact Stage 9F development crops."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rasterio
import torch
from affine import Affine
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, label
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from roadmarking.evaluation.link_audit import output_gaps
from roadmarking.inference.calibration import filter_instances, iou_matrix
from roadmarking.inference.stitching import stitch
from roadmarking.inference.stitching_supported_v1 import stitch_supported_v1
from roadmarking.training.pilot import load_crop
from stage9c_attested_pilot import source_context
from stage9d_common import ROOT, dump, frozen_checks, read, sha256, verify_frozen
from stage9e_lib import V1, development_crops, proposals

BASE = ROOT / "outputs/reports"
STAGE9F = BASE / "stage9f/training/20261112T074209709141Z"
EIGHT = np.ones((3, 3), bool)
BINS = (0, 1.5, 3, 10, 20, np.inf)
PROB_BINS = np.linspace(0, 1, 11)


def observed_sources(state):
    p55 = read(BASE / "phase5_5/latest.json")
    parent = read(Path(p55["dataset"]) / "metadata/dataset_manifest.json")["parent_dataset"]
    p54 = read(Path(parent) / "metadata/dataset_manifest.json")
    sources = {t["tile_id"]: t["source_sample"]["phase3_layers"]["observed"] for t in p54["tiles"]}
    for path in (
        BASE / "phase5_7_new_review/20261106T105319835838Z/new_region_candidates.json",
        Path(state["report_dir"]) / "new_region_candidates.json",
    ):
        sources.update({t["candidate_id"]: t["layers"]["observed"] for t in read(path)["tiles"]})
    return sources


def observed_crop(path, crop):
    row, col, height, width = crop["window"]
    with rasterio.open(path) as ds:
        window = rasterio.windows.Window(col, row, width, height)
        expected = ds.window_transform(window)
        observed = ds.read(1, window=window).astype(bool)
    assert expected.almost_equals(Affine(*crop["affine"]), precision=1e-7), crop["crop_id"]
    assert observed.shape == (height, width)
    return observed


def union(masks, shape):
    return masks.any(0) if len(masks) else np.zeros(shape, bool)


def bin_counts(distances, mask, bins):
    return np.histogram(distances[mask], bins=bins)[0].tolist()


def one_level(
    masks, labels, scores, quality, valid, observed, truth, tl, obs_dist, paint_dist, proxy
):
    present = union(masks, valid.shape)
    outside = present & ~observed
    prediction = present & valid
    truth_union = union(truth, valid.shape) & valid
    categories = np.zeros((3, *valid.shape), bool)
    counts = []
    split = 0
    for mask, cls, score, q in zip(masks, labels, scores, quality, strict=True):
        clipped = mask & valid
        split += label(clipped, EIGHT)[1] > 1
        matches = truth[tl == cls] & valid
        ious = (matches & clipped).sum((1, 2)) / np.maximum((matches | clipped).sum((1, 2)), 1)
        best = float(ious.max(initial=0))
        if (mask & ~observed).any():
            cc, _ = label(mask, EIGHT)
            anchored = np.isin(cc, np.unique(cc[clipped])) & mask if clipped.any() else cc < 0
            categories[2] |= mask & ~anchored
            categories[int(best < 0.5)] |= anchored
        counts.append(
            dict(
                cls=int(cls),
                score=float(score),
                class_confidence=float(score / q),
                quality=float(q),
                true_nodata=int((mask & ~observed).sum()),
                valid=int(clipped.sum()),
                same_class_iou=best,
            )
        )
    # Count each geographic pixel once. Prefer an extension of an IoU-matched mask.
    categories[1] &= ~categories[0]
    categories[2] &= ~(categories[0] | categories[1])
    assert np.array_equal(categories.any(0) & ~observed, outside)
    pixels = {}
    for cls in (0, 1):
        p = union(masks[labels == cls], valid.shape) & valid
        t = union(truth[tl == cls], valid.shape) & valid
        pixels[str(cls)] = [int((p & t).sum()), int((p & ~t).sum()), int((t & ~p).sum())]
    return dict(
        eligible_outputs=len(masks),
        true_nodata=int(outside.sum()),
        old_proxy=int((present & proxy).sum()),
        invalid=int((present & ~valid).sum()),
        observed_unknown=int((present & observed & ~valid).sum()),
        clipped_nodata=int((prediction & ~observed).sum()),
        valid_pred=int(prediction.sum()),
        paint_covered=int((truth_union & prediction).sum()),
        clipped_disconnected=int(split),
        nodata_by_observation_distance=bin_counts(obs_dist, outside, BINS),
        nodata_by_paint_distance=bin_counts(paint_dist, outside, BINS),
        nodata_by_origin=[int((cat & outside).sum()) for cat in categories],
        pixel_confusion=pixels,
        proposals=counts,
    )


def overlay(rgb, observed, valid, masks, labels, truth):
    canvas = (rgb * 0.55).astype(np.uint8)
    canvas[~observed] = (30, 39, 56)
    canvas[observed & ~valid] = (65, 62, 73)
    canvas[truth] = (225, 225, 225)
    for cls, color in ((0, (43, 220, 160)), (1, (79, 180, 255))):
        canvas[union(masks[labels == cls], valid.shape) & valid] = color
    present = union(masks, valid.shape)
    canvas[present & observed & ~valid] = (190, 120, 245)
    canvas[present & ~observed] = (255, 127, 64)
    return Image.fromarray(canvas)


def compare(model, processor, inputs):
    results, images, proposals_by_crop = [], [], []
    for root, crop, observed in inputs:
        x, masks, labels, validity, rgb = load_crop(root, crop, processor, "cuda")
        valid = validity[0].cpu().numpy().astype(bool)
        assert not (valid & ~observed).any()
        truth, tl = masks[0].cpu().numpy().astype(bool), labels[0].cpu().numpy()
        truth_union = union(truth, valid.shape) & valid
        obs_dist = distance_transform_edt(~observed)
        paint_dist = (
            distance_transform_edt(~truth_union)
            if truth_union.any()
            else np.full(valid.shape, np.inf)
        )
        proxy = ~valid & (rgb[..., 0] == 0)
        pred = proposals(model, x, valid)
        pm, pl = pred["masks"], pred["labels"]
        selected = np.array(
            (
                filter_instances(
                    pred["scores"],
                    pl,
                    iou_matrix(pm, pm, valid),
                    pred["qualities"],
                    V1,
                )
                if len(pm)
                else []
            ),
            dtype=int,
        )
        prob = pred["sigmoid_max"] if len(pm) else np.zeros(valid.shape)
        record = dict(
            crop=crop["crop_id"],
            source_observed=int(observed.sum()),
            valid=int(valid.sum()),
            area=valid.size,
            paint=int(truth_union.sum()),
            old_proxy=int(proxy.sum()),
            proxy_overcounts=int((proxy & observed).sum()),
            proxy_misses=int((~observed & ~proxy).sum()),
            probability_histograms={
                key: bin_counts(prob, domain, PROB_BINS)
                for key, domain in {
                    "reviewed_valid": valid,
                    "observed": observed,
                    "unobserved": ~observed,
                    "observed_unknown": observed & ~valid,
                }.items()
            },
        )
        panels = []
        for name, idx in (("raw", np.arange(len(pm))), ("v1", selected)):
            record[name] = one_level(
                pm[idx],
                pl[idx],
                pred["scores"][idx],
                pred["qualities"][idx],
                valid,
                observed,
                truth,
                tl,
                obs_dist,
                paint_dist,
                proxy,
            )
            panels.append(overlay(rgb, observed, valid, pm[idx], pl[idx], truth_union))
        panels.append(
            overlay(rgb, observed, valid, pm[selected] & valid, pl[selected], truth_union)
        )
        results.append(record)
        images.append(panels)
        proposals_by_crop.append((crop, observed, valid, truth_union, pred))
    return results, images, proposals_by_crop


def stitch_audit(rows):
    groups = {}
    for crop, observed, valid, truth, pred in rows:
        # Legacy tiles share a native lattice; fresh development has its own origin.
        key = "legacy" if crop["tile_id"].startswith("INTERCHANGE") else crop["tile_id"]
        groups.setdefault(key, []).append((crop, observed, valid, truth, pred))
    result = {}
    for key, members in groups.items():
        left = min(c["affine"][2] for c, *_ in members)
        top = max(c["affine"][5] for c, *_ in members)
        grid = Affine(0.05, 0, left, 0, -0.05, top)
        windows = {}
        for c, *_ in members:
            col, row = ~grid * (c["affine"][2], c["affine"][5])
            row, col = round(row), round(col)
            assert (grid * Affine.translation(col, row)).almost_equals(
                Affine(*c["affine"]), precision=1e-7
            ), c["crop_id"]
            windows[c["crop_id"]] = row, col
        shape = tuple(max(p[axis] for p in windows.values()) + 512 for axis in (0, 1))
        obs_grid, valid_grid, paint, seen = (np.zeros(shape, bool) for _ in range(4))
        masks, scores, classes, ids = [], [], [], []
        crop_support = {}
        overlap_pixels = 0
        for c, observed, valid, truth, pred in members:
            cid = c["crop_id"]
            r, col = windows[cid]
            sl = np.s_[r : r + 512, col : col + 512]
            overlap_pixels += int(seen[sl].sum())
            assert not (seen[sl] & (obs_grid[sl] != observed)).any()
            obs_grid[sl] |= observed
            valid_grid[sl] |= valid
            paint[sl] |= truth
            seen[sl] = True
            crop_support[cid] = np.zeros(shape, bool)
            crop_support[cid][sl] = valid
            pm, pl = pred["masks"], pred["labels"]
            selected = (
                filter_instances(
                    pred["scores"],
                    pl,
                    iou_matrix(pm, pm, valid),
                    pred["qualities"],
                    dict(V1, confidence={"0": 0.0, "1": 0.0}),
                )
                if len(pm)
                else []
            )
            for i in selected:
                m = np.zeros(shape, bool)
                m[sl] = pm[i] & valid
                masks.append(m)
                scores.append(float(pred["scores"][i]))
                classes.append(int(pl[i]))
                ids.append(cid)
        outputs, audit = stitch(masks, scores, classes, ids, windows, valid_grid, V1["confidence"])
        corrected, correction = stitch_supported_v1(
            masks, scores, classes, ids, windows, crop_support, V1["confidence"]
        )
        unsupported = []
        pixel_leak = 0
        mixed = 0
        disconnected = 0
        for i, entry in enumerate(outputs):
            m = entry["mask"]
            pixel_leak += int((m & ~obs_grid).sum())
            mixed += any(classes[j] != entry["class"] for j in entry["members"])
            disconnected += label(m, EIGHT)[1] > 1
            gaps = output_gaps(
                m,
                valid_grid,
                obs_grid,
                paint,
                [masks[j] for j in entry["members"]],
                min_component=4,
            )
            for gap in gaps:
                if gap["unobserved_fraction"] > 0:
                    unsupported.append(dict(output=i, cls=entry["class"], **gap))
        result[key] = dict(
            overlap_pixels=overlap_pixels,
            outputs=len(outputs),
            accepted_nodata=pixel_leak,
            mixed_class=mixed,
            disconnected=disconnected,
            gaps_crossing_unobserved=unsupported,
            experimental_supported_v1=dict(
                outputs=len(corrected),
                accepted_nodata=sum(int((m["mask"] & ~obs_grid).sum()) for m in corrected),
                disconnected=sum(label(m["mask"], EIGHT)[1] > 1 for m in corrected),
                mixed_class=sum(
                    any(classes[j] != m["class"] for j in m["members"]) for m in corrected
                ),
                old_union_pixels=int(union(np.asarray([m["mask"] for m in outputs]), shape).sum()),
                corrected_union_pixels=int(
                    union(np.asarray([m["mask"] for m in corrected]), shape).sum()
                ),
                **correction,
            ),
            **audit,
        )
    return result


def main():
    torch.set_num_threads(4)
    checks = frozen_checks()
    verify_frozen(checks)
    manifest = read(STAGE9F / "package_manifest.json")
    for file in manifest["files"]:
        assert sha256(ROOT / file["path"]) == file["sha256"], file["path"]
    decision = read(STAGE9F / "decision.json")
    assert decision["status"] == "LOSS_DESIGN_BLOCKED"
    ctx = source_context()
    development = development_crops(ctx)
    assert [c["crop_id"] for _, c in development] == read(STAGE9F / "protocol.json")["development"]
    sources = observed_sources(ctx[3])
    inputs = [(root, c, observed_crop(sources[c["tile_id"]], c)) for root, c in development]
    out = BASE / "stage9g" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    (out / "overlays").mkdir(parents=True)
    models = {
        "frozen": Path(checks[0]["path"]).parent,
        "C5": Path(decision["models"]) / "C5_s9061/model",
    }
    result, images = {}, {}
    for name, path in models.items():
        processor = Mask2FormerImageProcessor.from_pretrained(
            path.parent / "processor", local_files_only=True
        )
        model = (
            Mask2FormerForUniversalSegmentation.from_pretrained(path, local_files_only=True)
            .cuda()
            .eval()
        )
        rows, images[name], stitch_inputs = compare(model, processor, inputs)
        result[name] = dict(crops=rows, stitched=stitch_audit(stitch_inputs))
        del model
        torch.cuda.empty_cache()
    for i, (_, crop, _) in enumerate(inputs):
        figure = Image.new("RGB", (1536, 1088), "#19202b")
        draw = ImageDraw.Draw(figure)
        for y, name in enumerate(models):
            for x, (title, panel) in enumerate(
                zip(
                    ("Raw proposals", "V1 before clipping", "V1 after clipping"),
                    images[name][i],
                    strict=True,
                )
            ):
                draw.text((512 * x + 8, 544 * y + 8), f"{name} | {title}", fill="white")
                figure.paste(panel, (512 * x, 544 * y + 32))
        figure.save(out / "overlays" / f"{crop['crop_id']}.png")
    verify_frozen(checks)
    dump(
        out / "diagnostics.json",
        dict(
            distance_bin_edges_px=[0, 1.5, 3, 10, 20, "infinity"],
            probability_bin_edges=PROB_BINS.tolist(),
            origin_labels=["IoU>=0.5 anchored extension", "other anchored extension", "detached"],
            crop_sources={c["crop_id"]: sources[c["tile_id"]] for _, c in development},
            models=result,
        ),
    )
    print(out, flush=True)


if __name__ == "__main__":
    main()
