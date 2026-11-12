"""Final-test report and visual from STORED predictions (no re-evaluation, no tuning)."""

import json

import matplotlib
import numpy as np

import phase5_10_recovery as rec
from final_candidate import pipeline
from phase5_11_common import ROOT, load
from stage1_scoring import FROZEN

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = ROOT / "outputs/reports/final_test"
DATA = ROOT / "data/processed/final_test"
NAME = "NEW_FINAL_TEST_RESERVED"


def main():
    r = load(OUT / "final_test_results.json")
    frozen = load(ROOT / "outputs/reports/final_candidate/frozen_candidate.json")
    spec = load(
        ROOT / "outputs/reports/phase5_7_new_review/20261106T105319835838Z/"
        "new_region_candidates.json"
    )
    tile = next(t for t in spec["tiles"] if t["candidate_id"] == NAME)
    crops = [
        {
            "crop_id": p.stem,
            "file": str(p),
            "window": [
                int(p.stem.split("_r")[1].split("_")[0]),
                int(p.stem.split("_c")[-1]),
                512,
                512,
            ],
        }
        for p in sorted(DATA.glob(f"{NAME}_s512_*.npz"))
    ]
    gt = DATA / f"{NAME}_ground_truth.npz"
    with np.load(gt) as d:
        n = len(d["labels"])
    panels = []
    for model, thr, stitch_on, title in (
        (
            "default",
            {
                "confidence": FROZEN["confidence"],
                "mask_quality": FROZEN["mask_quality"],
                "nms_iou": 0.3,
            },
            False,
            "Current default pipeline",
        ),
        (
            "candidate",
            {"confidence": frozen["thresholds_V1"], "mask_quality": 0.0, "nms_iou": 0.3},
            True,
            "Frozen candidate (V1 + stitching)",
        ),
    ):
        comp = rec.tile_component(
            NAME, tile, crops, OUT / "inference" / model, gt, [f"id{i}" for i in range(n)]
        )
        comp["overlaps"] = rec.overlap_matrix(comp)
        masks, _, cls, keep, _ = pipeline(comp, thr, stitch_on)
        panels.append((title, masks, cls, keep, comp))
    comp = panels[0][4]
    gray = comp["image"].astype(float) / 255
    fig, axes = plt.subplots(1, 3, figsize=(21, 7.5))
    base = np.repeat(gray[..., None], 3, -1)
    gt_img = base.copy()
    tl = comp["target_labels"]
    for c, col in ((0, (1, 0.25, 0.1)), (1, (0.1, 0.6, 1))):
        if (tl == c).any():
            gt_img[comp["targets"][tl == c].any(0)] = col
    axes[0].imshow(gt_img)
    axes[0].set_title("Approved ground truth (red solid, blue dash)")
    for ax, (title, masks, _cls, keep, _) in zip(axes[1:], panels, strict=True):
        img = base.copy()
        truth = comp["targets"].any(0)
        pred = masks[keep].any(0) if keep else np.zeros_like(truth)
        img[pred & truth] = (0.1, 0.9, 0.2)
        img[pred & ~truth & comp["valid"]] = (1, 0.85, 0)
        img[truth & ~pred] = (1, 0, 1)
        ax.imshow(img)
        ax.set_title(f"{title}\ngreen = correct paint, yellow = false, magenta = missed")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "final_test_comparison.png", dpi=80)
    plt.close(fig)
    m = r["candidate_vs_default"]["metrics"]
    g = r["ground_truth"]
    rows = [
        ("Foreground IoU", "foreground_iou"),
        ("Pixel precision", "pixel_precision"),
        ("Pixel recall", "pixel_recall"),
        ("Solid buffer completeness", "solid_buffer_completeness"),
        ("Solid buffer correctness", "solid_buffer_correctness"),
        ("Solid complete-line F1 (IoU≥.5)", "solid_f1"),
        ("Solid AP50", "solid_ap50"),
        ("Dash F1 (IoU≥.5)", "dash_f1"),
    ]
    cand = r["results"]["candidate_frozen_pipeline"]["summary"]
    dflt = r["results"]["default_pipeline"]["summary"]
    md = [
        "# One-time final test: frozen candidate on NEW_FINAL_TEST_RESERVED",
        "",
        f"Authorized by the user and opened once (`test_opened.json`, SHA-256 "
        f"`{r['opened_record_sha256']}`). The frozen candidate "
        f"(weights `{frozen['weights_sha256'][:12]}…`, thresholds {frozen['thresholds_V1']}, "
        "stitching) was scored exactly as frozen. Results are reported as-is, with no "
        "tuning before or after.",
        "",
        f"Ground truth: the user-approved Phase 5.7 outlines, exported with the "
        f"development rules. That gives {g['solid_ids']} solid and {g['dash_ids']} dash "
        f"identities, {g['crops']} native 512 crops and {g['scored_valid_pixels']:,} scored "
        "valid pixels. This region was never used in training, selection, thresholds or "
        "calibration.",
        "",
        "| Metric | Current default | Frozen candidate | Difference [95% block-bootstrap CI] |",
        "|---|---:|---:|---|",
    ]
    for label, k in rows:
        if k in m:
            md.append(
                f"| {label} | {m[k]['a']:.3f} | {m[k]['b']:.3f} | {m[k]['difference']:+.3f}"
                f" [{m[k]['ci_difference'][0]:+.3f}, {m[k]['ci_difference'][1]:+.3f}]"
                f"{' *' if m[k]['difference_excludes_zero'] else ''} |"
            )
    md += [
        f"| Solid / dash FP | {dflt['solid_fp']:.0f} / {dflt['dash_fp']:.0f} | "
        f"{cand['solid_fp']:.0f} / {cand['dash_fp']:.0f} | — |",
        "",
        f"`*` = CI excludes 0. Bootstrap over {r['candidate_vs_default']['blocks']} "
        "non-overlapping 12.8 m blocks.",
        "",
        "## Interpretation",
        "",
        "- On a road it had never seen, the current default recognises almost nothing: "
        "foreground IoU 0.03, pixel recall 0.04, and no solid line covered. The frozen "
        "candidate covers **90% of solid-line length**, with 89% correctness and pixel "
        "recall 0.90. Every pixel-level and buffer metric improves significantly.",
        "- Complete-instance matching remains hard. 2 of 8 solid lines match at IoU≥0.5 "
        "(F1 0.29, CI touching 0), and **neither of the 2 dashes is matched by either "
        "model**. With 10 instances, instance-level results here are very uncertain.",
        "- The result agrees with the leave-one-region-out evidence: the large gains come "
        "from pixel and line coverage, while dash instances in unfamiliar geography are the "
        "remaining weakness.",
        "",
        "## Caveats",
        "",
        "- This is one region from one acquisition, with 8 solid and 2 dash identities. It "
        "is an unbiased check on this region, not a population-wide accuracy claim.",
        "- The protected test is now consumed. Any further model change needs a new, "
        "never-used test region. Stage 6's S6_01 is the proposed next fresh area.",
        "- Not production-ready: dash instance recognition and full 3D digitization remain "
        "out of scope.",
        "",
        "Visual: `final_test_comparison.png`.",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"report": str(OUT / "report.md")}))


if __name__ == "__main__":
    main()
