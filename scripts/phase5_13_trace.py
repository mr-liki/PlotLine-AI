"""Phase 5.13 Stage 1: trace the lost legacy road-edge solid through three models.

Uses the full saved query outputs (class_logits, mask_logits) of each model on the SAME
legacy development crops, so every query is visible, not only proposals above the .25 floor.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

import phase5_10_recovery as rec
from phase5_11_common import ROOT, client_features, load
from phase5_11_evaluate import FROZEN
from roadmarking.inference.calibration import iou_matrix
from roadmarking.inference.prediction_recovery import filter_geographic
from roadmarking.rasterization.pipeline import dump

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT = ROOT / "models/mask2former_swin_s/phase5_7_expanded_20261106T111240383727Z"


def model_folders():
    r11 = Path(load(ROOT / "outputs/reports/phase5_11/latest.json")["training_run"])
    r12 = Path(load(ROOT / "outputs/reports/phase5_12/latest.json")["training_run"])
    return {
        "default_step50": DEFAULT / "validation_step0050",
        "p511_step400": r11 / "validation_step0400",
        "p512_step400": r12 / "validation_step0400",
    }


def legacy(folder):
    p57 = load(ROOT / "outputs/reports/phase5_7_expanded/latest.json")
    p55 = load(ROOT / "outputs/reports/phase5_5/latest.json")
    manifest = load(Path(p57["dataset"]) / "metadata/dataset_manifest.json")
    p54 = load(
        Path(load(Path(p55["dataset"]) / "metadata/dataset_manifest.json")["parent_dataset"])
        / "metadata/dataset_manifest.json"
    )
    protocol = load(ROOT / "outputs/reports/phase5_10/20261106T132745062304Z/protocol.json")
    comp = rec.legacy_component(
        manifest, p54, protocol["data_roles"]["LEGACY_POSITIVE_DEV"]["crops"], folder
    )
    comp["overlaps"] = rec.overlap_matrix(comp)
    return comp


def width_profile(prob, target, valid, bins=range(-3, 9)):
    inside = distance_transform_edt(target)
    outside = distance_transform_edt(~target)
    signed = np.where(target, -inside + 1, outside)  # 1 = first pixel outside the paint
    out = {}
    for b in bins:
        sel = valid & (np.round(signed) == b)
        out[str(b)] = float(prob[sel].mean()) if sel.any() else None
    return out


def main():
    out = ROOT / "outputs/reports/phase5_13"
    out.mkdir(parents=True, exist_ok=True)
    features = {f["global_id"]: f for f in client_features(check_network=False)[0]}
    folders = model_folders()
    comps = {name: legacy(folder) for name, folder in folders.items()}
    base = comps["default_step50"]
    solids = [j for j in range(len(base["ids"])) if base["target_labels"][j] == 0]
    sel_by_model = {}
    for name, comp in comps.items():
        sel_by_model[name] = filter_geographic(
            comp["scores"],
            comp["classes"],
            comp["qualities"],
            comp["overlaps"],
            comp["crop_ids"],
            FROZEN,
        )

    # The lost road-edge marking: the solid target with the largest drop in covered pixels
    # from the default's final masks to both candidates' final masks.
    def covered(comp, j):
        sel = sel_by_model[comp["name_key"]]
        masks = comp["masks"][sel][comp["classes"][sel] == 0]
        union = masks.any(0) if len(masks) else np.zeros(comp["shape"], bool)
        return int((union & comp["targets"][j]).sum())

    for k, c in comps.items():
        c["name_key"] = k
    drop = {
        j: covered(base, j) - max(covered(comps[m], j) for m in ("p511_step400", "p512_step400"))
        for j in solids
    }
    lost = max(drop, key=drop.get)
    uid = base["ids"][lost]
    f = features[uid]
    target = base["targets"][lost]
    skel = skeletonize(target) & base["valid"]
    identity = {
        "global_id": uid,
        "source_identity": f["source_identity"],
        "designation": f["designation"],
        "client_layer": f["layer"],
        "width": f["width"],
        "observed_target_pixels": int(target.sum()),
        "clipped_centerline_length_native": float(
            f["centerline"]
            .intersection(
                __import__("shapely").box(
                    base["origin_native_xy"][0],
                    base["origin_native_xy"][1] - base["shape"][0] * 0.05,
                    base["origin_native_xy"][0] + base["shape"][1] * 0.05,
                    base["origin_native_xy"][1],
                )
            )
            .length
        ),
        "final_mask_pixels_covered_by_model": {m: covered(comps[m], lost) for m in comps},
        "other_solid_targets_drop": {base["ids"][j]: drop[j] for j in solids if j != lost},
    }
    # ---------- per-model, per-crop full-query analysis
    results, profiles = {}, {}
    for name, comp in comps.items():
        folder = folders[name]
        per_crop = []
        gp = np.zeros(comp["shape"], float)  # max foreground prob over solid-class-argmax queries
        gp_any = np.zeros(comp["shape"], float)  # max over all non-no-object queries
        for crop in comp["crops"]:
            r, c = crop["row"], crop["col"]
            with np.load(folder / (crop["crop_id"] + ".npz")) as d:
                cl = torch.from_numpy(d["class_logits"])[0].softmax(-1).numpy()
                ml = torch.nn.functional.interpolate(
                    torch.from_numpy(d["mask_logits"]),
                    size=(512, 512),
                    mode="bilinear",
                    align_corners=False,
                )[0].numpy()
                saved_scores, saved_labels = d["scores"], d["labels"]
                selected_local = d["selected"].astype(int).tolist()
            v = crop["saved_validity"]
            t = target[r : r + 512, c : c + 512] & v
            if t.sum() < 4:
                continue
            prob = 1 / (1 + np.exp(-ml))
            argmax = cl.argmax(-1)
            for q in range(len(cl)):
                if argmax[q] == 0:
                    gp[r : r + 512, c : c + 512] = np.maximum(gp[r : r + 512, c : c + 512], prob[q])
                if argmax[q] != 2:
                    gp_any[r : r + 512, c : c + 512] = np.maximum(
                        gp_any[r : r + 512, c : c + 512], prob[q]
                    )
            masks = ml > 0
            ious = iou_matrix(masks, t[None], v)[:, 0]
            q = int(np.argmax(ious))
            m = masks[q] & v
            quality = float(prob[q][m].mean()) if m.sum() else 0.0
            label = int(argmax[q])
            conf = float(cl[q].max() * quality)
            # Was this query among saved proposals (>= .25 floor) and kept by frozen filter?
            saved_idx = None
            if label != 2 and m.sum() >= 4 and conf >= 0.25:
                matches = [
                    i
                    for i, (s, lab) in enumerate(zip(saved_scores, saved_labels, strict=True))
                    if lab == label and abs(s - conf) < 1e-4
                ]
                saved_idx = matches[0] if matches else None
            reason = (
                "argmax_no_object"
                if label == 2
                else (
                    "below_proposal_floor_.25"
                    if conf < 0.25
                    else (
                        "below_class_threshold"
                        if conf < FROZEN["confidence"][str(label)]
                        else (
                            "below_mask_quality"
                            if quality < FROZEN["mask_quality"]
                            else (
                                "kept_locally"
                                if saved_idx in selected_local
                                else "suppressed_by_local_nms"
                            )
                        )
                    )
                )
            )
            if reason == "kept_locally":
                gi = [
                    i
                    for i, (cid, li) in enumerate(
                        zip(comp["crop_ids"], comp["local_index"], strict=True)
                    )
                    if cid == crop["crop_id"] and li == saved_idx
                ]
                if gi and gi[0] not in sel_by_model[name]:
                    reason = "suppressed_by_global_nms"
            tsk = skeletonize(t) & v
            per_crop.append(
                {
                    "crop": crop["crop_id"],
                    "target_pixels_in_crop": int(t.sum()),
                    "best_query": q,
                    "class_probs_solid_dash_noobj": [float(x) for x in cl[q]],
                    "argmax_class": ["solid", "dash", "no_object"][label],
                    "mask_quality": quality,
                    "confidence_class_x_quality": conf,
                    "raw_mask_iou_vs_paint_in_crop": float(ious[q]),
                    "raw_mask_pixels": int(m.sum()),
                    "centreline_coverage_best_query": float(m[tsk].mean()) if tsk.any() else None,
                    "mean_fg_prob_on_paint_best_query": float(prob[q][t].mean()),
                    "max_solid_query_prob_on_paint_mean": None,
                    "fate": reason,
                    "solid_prob_of_best_dash_or_solid_query": float(cl[q][0]),
                }
            )
        # Final (post-filter) mask IoU against the complete identity.
        sel = sel_by_model[name]
        fmask = comp["masks"][sel][comp["classes"][sel] == 0]
        final_union = fmask.any(0) if len(fmask) else np.zeros(comp["shape"], bool)
        best_final = max(
            (
                float((m & target).sum() / max(1, ((m | target) & comp["valid"]).sum()))
                for m in comp["masks"][sel]
            ),
            default=0.0,
        )
        results[name] = {
            "per_crop": per_crop,
            "final_best_single_mask_iou_complete_identity": best_final,
            "final_solid_union_coverage_of_paint": float(
                (final_union & target).sum() / max(1, target.sum())
            ),
            "final_solid_union_centreline_coverage": float(final_union[skel].mean()),
            "max_solid_query_fg_prob_along_centreline": {
                "mean": float(gp[skel].mean()),
                "p10": float(np.percentile(gp[skel], 10)),
                "fraction_above_0.5": float((gp[skel] > 0.5).mean()),
            },
            "max_any_class_query_fg_prob_along_centreline": {
                "mean": float(gp_any[skel].mean()),
                "fraction_above_0.5": float((gp_any[skel] > 0.5).mean()),
            },
            "width_profile_max_solid_query_prob_by_signed_px": width_profile(
                gp, target, comp["valid"]
            ),
            "cross_tile_fates": [p["fate"] for p in per_crop],
            "cross_tile_argmax_classes": [p["argmax_class"] for p in per_crop],
        }
        profiles[name] = (gp, gp_any, final_union)
    for name in results:
        for p in results[name]["per_crop"]:
            del p["max_solid_query_prob_on_paint_mean"]
    dump(out / "stage1_lost_edge_trace.json", {"identity": identity, "models": results})

    # ---------- side-by-side diagnostic
    ys, xs = np.nonzero(target)
    y0, y1 = max(0, ys.min() - 40), min(target.shape[0], ys.max() + 40)
    x0, x1 = max(0, xs.min() - 40), min(target.shape[1], xs.max() + 40)
    gray = base["image"][y0:y1, x0:x1].astype(float) / 255
    fig, axes = plt.subplots(3, 3, figsize=(18, 10))
    for row, name in enumerate(comps):
        gp, gp_any, final = profiles[name]
        t = target[y0:y1, x0:x1]
        img = np.repeat(gray[..., None], 3, -1)
        img[t] = (0.1, 0.9, 0.2)
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"{name}: approved edge paint (green)", fontsize=9)
        axes[row, 1].imshow(gp[y0:y1, x0:x1], vmin=0, vmax=1, cmap="magma")
        axes[row, 1].contour(t, levels=[0.5], colors="cyan", linewidths=0.4)
        axes[row, 1].set_title(
            f"max solid-query foreground prob (centreline >.5: "
            f"{results[name]['max_solid_query_fg_prob_along_centreline']['fraction_above_0.5']:.2f})",
            fontsize=9,
        )
        img2 = np.repeat(gray[..., None], 3, -1)
        img2[final[y0:y1, x0:x1]] = (1, 0.3, 0.1)
        img2[t & ~final[y0:y1, x0:x1]] = (1, 0, 1)
        axes[row, 2].imshow(img2)
        axes[row, 2].set_title(
            f"final solid masks (red) / missed paint (magenta); coverage "
            f"{results[name]['final_solid_union_coverage_of_paint']:.2f}",
            fontsize=9,
        )
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle(f"Lost legacy road-edge solid {uid[:12]} ({identity['designation']})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "stage1_lost_edge_side_by_side.png", dpi=90)
    plt.close(fig)
    print(
        json.dumps(
            {
                "identity": identity,
                "summary": {
                    m: {k: v for k, v in r.items() if k != "per_crop"} for m, r in results.items()
                },
                "per_crop": {
                    m: [
                        (
                            p["crop"][-12:],
                            p["argmax_class"],
                            round(p["class_probs_solid_dash_noobj"][0], 3),
                            round(p["mask_quality"], 3),
                            round(p["raw_mask_iou_vs_paint_in_crop"], 3),
                            p["fate"],
                        )
                        for p in r["per_crop"]
                    ]
                    for m, r in results.items()
                },
            },
            indent=1,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
