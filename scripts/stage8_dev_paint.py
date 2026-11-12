("Paint support statistics on established development areas, using " "saved artifacts.")

import numpy as np
import shapely
from scipy.ndimage import binary_dilation

from phase5_11_common import ROOT, client_features, load
from roadmarking.rasterization.pipeline import dump
from stage1_scoring import DEFAULT
from stage7h_evaluate import build
from stage8_link_audit import comp_affine


def main():
    features = {f["global_id"]: f for f in client_features(check_network=False)[0]}
    p59 = load(ROOT / "outputs/reports/phase5_9/latest.json")
    comps, _ = build(
        DEFAULT / "validation_step0050",
        ROOT / p59["report_dir"] / "inference",
        ROOT / "outputs/reports/stage7/7D_S6_01_baseline/inference/default_V0",
    )
    records = []
    for name, comp in comps.items():
        affine = comp_affine(comp)
        all_paint = comp["targets"].any(0)
        for uid, cls, target in zip(
            comp["ids"], comp["target_labels"], comp["targets"], strict=True
        ):
            mask = target & comp["observed"] & comp["valid"]
            r, c = np.nonzero(mask)
            x, y = affine * (c + 0.5, r + 0.5)
            feature = features.get(uid)
            supported = None
            if feature is not None and feature["centerline"] is not None and len(r):
                along = shapely.line_locate_point(feature["centerline"], shapely.points(x, y))
                supported = int(np.count_nonzero(np.bincount(np.floor(along).astype(int)) >= 3))
            ring = (
                binary_dilation(mask, iterations=10)
                & ~binary_dilation(mask, iterations=2)
                & comp["valid"]
                & ~all_paint
            )
            contrast = (
                float(np.median(comp["image"][mask]) - np.median(comp["image"][ring]))
                if mask.any() and ring.any()
                else None
            )
            records.append(
                {
                    "region": name,
                    "split": "development",
                    "class": "solid" if cls == 0 else "dash",
                    "global_id": uid,
                    "observed_paint_pixels": int(mask.sum()),
                    "supported_length_native": supported,
                    "contrast_grayscale": contrast,
                    "acquisition": None,
                    "acquisition_status": "Resolve using original crop source_sample metadata; not assumed from filename.",  # noqa: E501
                }
            )
        print(f"{name}: paint measured", flush=True)
    summary = []
    for name in comps:
        for cls in ("solid", "dash"):
            group = [r for r in records if r["region"] == name and r["class"] == cls]
            contrasts = [
                r["contrast_grayscale"] for r in group if r["contrast_grayscale"] is not None
            ]
            summary.append(
                {
                    "region": name,
                    "class": cls,
                    "ids": len(group),
                    "observed_paint_pixels": sum(r["observed_paint_pixels"] for r in group),
                    "supported_length_native": sum(
                        r["supported_length_native"] or 0 for r in group
                    ),
                    "contrast_mean_over_ids": float(np.mean(contrasts)) if contrasts else None,
                }
            )
    dump(
        ROOT / "outputs/reports/stage8/8_4_dev_paint.json",
        {
            "per_id": records,
            "summary": summary,
            "method": (
                "Established development only; observed valid target masks; >=3px per "
                "1-unit centreline bin; rendered grayscale contrast. Saved inference "
                "artifacts loaded for existing component assembly; no model executed."
            ),
        },
    )


if __name__ == "__main__":
    main()
