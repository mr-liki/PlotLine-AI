"""Produce Stage 9C diagnostics and seal the pretraining decision."""

import json
from pathlib import Path

import numpy as np
import pyogrio
import rasterio
from PIL import Image
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

from stage9c_audit import BASE, ROOT, dump, quantiles, sha256

run = Path(json.loads((BASE / "latest.json").read_text())["run"])
audit = json.loads((run / "synthetic_text_dataset_audit.json").read_text())
source = Path(audit["source"])
assert all(sha256(source / name) == value for name, value in audit["source_file_sha256"].items())
assert audit["valid_samples"] == 500 and not audit["errors"]
stage9b = ROOT / "outputs/reports/stage9b"
final = stage9b / "stroke_gap_review"
approval = json.loads((final / "n9_01_text_finalization_manifest.json").read_text("utf-8-sig"))
assert approval["final_qgis_confirmation"]
assert all(sha256(Path(f["path"])) == f["sha256"] for f in approval["files"])
state = json.loads((stage9b / "latest.json").read_text())
report9b = Path(state["report_dir"])
spec = json.loads((report9b / "review_spec.json").read_text())
tile = next(tile for tile in spec["tiles"] if tile["candidate_id"] == "N9_01")
characters = pyogrio.read_dataframe(
    spec["text_candidates"], layer="character_candidates", where="candidate_id='N9_01'"
)
with rasterio.open(tile["layers"]["normalized"]) as ds:
    intensity = ds.read(1)
    paint = rasterize(
        [(geometry, 1) for geometry in characters.geometry],
        out_shape=ds.shape,
        transform=ds.transform,
    ).astype(bool)
with rasterio.open(tile["layers"]["observed"]) as ds:
    observed = ds.read(1).astype(bool)
assert not (paint & ~observed).any()
with np.load(report9b / "N9_01_review_safety.npz") as archive:
    ignore = archive["unconfirmed_review_extent"]
context = observed & ~ignore
roi = np.zeros_like(context)
roi[:248, :234] = True
context &= roi
samples = [r for r in audit["sample_audit"] if r["split"] == "train" and r["text"]]
synthetic = {
    "median_per_image_paint_q05_q50_q95": (
        np.median([r["foreground_intensity"] for r in samples], axis=0) / 255
    ).tolist(),
    "median_per_image_background_q05_q50_q95": (
        np.median([r["background_intensity"] for r in samples], axis=0) / 255
    ).tolist(),
    "visible_stroke_diameter_proxy_px": np.median(
        [r["stroke_diameter_proxy_px"] for r in samples], axis=0
    ).tolist(),
    "intended_paint_removed_fraction": sum(r["ignore_pixels"] for r in samples)
    / sum(r["intended_pixels"] for r in samples),
    "positive_tiles_touching_border": sum(r["visible_touches_border"] for r in samples),
}
real = {
    "paint_q05_q50_q95": quantiles(intensity[paint]),
    "observed_context_q05_q50_q95": quantiles(intensity[context]),
    "visible_stroke_diameter_proxy_px": quantiles(2 * distance_transform_edt(paint)[paint]),
    "observed_fraction_in_crop": float(observed[roi].mean()),
    "diagnostic_character_masks": len(characters),
    "context_is_not_verified_background": True,
    "role": "development_only",
}
dump(
    run / "domain_gap_analysis.json",
    {
        "synthetic_train_positive": synthetic,
        "N9_01_diagnostic_subset": real,
        "synthetic_gsd": None,
        "physical_width_comparison": False,
        "synthetic_alterations_fitted_on_N9_01": False,
        "limitations": [
            "Reference is one real development tile and its five delineated character drafts.",
            "Intensity values reflect different generation and normalization mechanisms.",
            "Black pixels alone do not identify scanner density or dropout.",
            "No augmentation fitted. Future candidates need separate training-source evidence.",
        ],
    },
)
# Compare intended silhouettes after centering and normalizing their bounding box.
# Shared typefaces can yield matches even when the actual generated samples differ.
shapes = []
positives = [record for record in audit["sample_audit"] if record["text"]]
for record in positives:
    with Image.open(source / record["split"] / "clean_masks" / record["name"]) as image:
        canonical = image.crop(image.getbbox()).resize((48, 48), Image.Resampling.NEAREST)
        shapes.append((np.array(canonical).ravel() != 0).astype("float32"))
matrix = np.stack(shapes)
intersection = matrix @ matrix.T
area = matrix.sum(axis=1)
iou = intersection / (area[:, None] + area[None, :] - intersection)
cross_split = [
    {"a": positives[i]["name"], "b": positives[j]["name"], "iou": round(float(iou[i, j]), 4)}
    for i, j in zip(*np.where(np.triu(iou >= 0.85, k=1)), strict=True)
    if positives[i]["split"] != positives[j]["split"]
]
dump(
    run / "split_similarity_audit.json",
    {
        "method": "Bounding-box-normalized intended silhouette IoU at 48 by 48, threshold 0.85",
        "cross_split_pairs": sorted(cross_split, key=lambda pair: pair["iou"], reverse=True),
        "limitation": "Sensitive to orientation and deformation; no generator provenance implied.",
    },
)
comparison = final / "N9_01_final_comparison.png"
with Image.open(comparison) as im:
    im.save(run / "real_development_annotation_gallery.png")
dump(
    run / "class_mapping.json",
    {
        "model": {"0": "longitudinal_solid", "1": "longitudinal_dash", "2": "road_text"},
        "synthetic_yolo_class_0_maps_to": 2,
        "no_object": 3,
        "legacy_no_object": 2,
    },
)
dump(
    run / "synthetic_text_adapter_report.json",
    {
        "version": 1,
        "state": "validated_for_offline_conversion_only",
        "granularity": "one phrase target, with original transcription metadata",
        "visible_binary_mask": "foreground",
        "intended_minus_visible": "ignore",
        "outside_intended": "background",
        "intensity_zero_implies_nodata": False,
        "resize": False,
        "normalization": "uint8 / 255 once",
        "character_targets": None,
    },
)
search = json.loads((report9b / "text_training_search.json").read_text())
dump(
    run / "new_real_text_training_candidates.json",
    {
        "candidates": [],
        "prior_source_backed_search": search["visual_inspection_outcome"],
        "new_source_backed_search_windows": 0,
        "N9_01": "development_only",
        "N9_04": "proposed training, no confirmed text",
    },
)
reason = (
    "The corpus README states that synthesis was influenced by a user-provided screenshot. "
    "No supplied metadata identifies that screenshot or its use. The Stage 9C brief excludes "
    "N9_01 from synthetic generation and training. Compliance cannot be verified from the archive."
)
dump(
    run / "training_protocol.json",
    {
        "status": "unexecuted",
        "gate": reason,
        "approved_development_role": "N9_01 only for text",
        "initialization": "frozen Swin-S migration checked on CPU",
        "synthetic_exposure_proposals": [0.10, 0.25],
        "registered_experiment": False,
        "data_hashes": "synthetic_text_dataset_audit.json",
        "adaptation_seed": 895,
        "protected_tests": ["T7", "S8"],
    },
)
for name in ("real_text_development_metrics", "solid_dash_regression_metrics"):
    dump(run / f"{name}.json", {"status": "not_run", "scores": None})
status = "BLOCKED"
decision = {
    "status": status,
    "reason": reason,
    "training_started": False,
    "protected_test_inference": False,
    "promotion": False,
    "digitization_started": False,
    "source_hashes_reverified": len(audit["source_file_sha256"]),
    "N9_01_approval_hashes_reverified": len(approval["files"]),
    "gpu": "RTX 3050 6144 MiB inventory; no training VRAM measurement",
}
dump(run / "model_selection_decision.json", decision)
dump(run / "report.json", {**decision, "splits": audit["splits"], "domain_gap": synthetic})
migration = (run / "three_class_checkpoint_migration_audit.json").exists()
dups = sum(len(groups) for groups in audit["decoded_exact_duplicate_nonempty"].values())
shared = len(audit["phrases_shared_across_splits"])
(run / "report.md").write_text(
    f"""# Stage 9C

**{status}**

{reason}

All 500 paired native 512 x 512 grayscale PNG samples passed the structural audit.
The source includes 475 positives and 25 negatives. JSON boxes use inclusive bounds
of intended paint. The YOLO boxes use the same center and inclusive size.
The visible binary masks are the supervised paint. Intended paint absent from the visible
mask is excluded from loss. Valid dark background is preserved.

The phrase is the only supplied instance unit. Transcriptions are retained as metadata.
Per-character masks, generator seeds, font groups and physical GSD are unavailable.
The duplicate audit found {dups} exact nonempty decoded raster groups.
{shared} phrase strings occur in multiple splits. Shared generator style and unavailable
template IDs prevent a claim of layout independence. The normalized-silhouette check
found {len(cross_split)} cross-split pairs at IoU 0.85 or higher.
No synthetic split was used for training.

The offline adapter converted all 500 pairs in a separate verification pass and preserves
holes and disconnected strokes exactly. Checkpoint reload
audit present: {migration}. N9_01 remains development-only and its 23 sealed file hashes
were verified. See the domain-gap JSON for pixel-level diagnostic statistics and limitations.
The Stage 6 approved export records 500 line identities, 72 solid and 428 dash, in
486 crops. No Stage 9 candidate geography was added to that export.
The earlier source-backed search found zero confirmed separate real text training regions.
This stage did not locate additional examples. No new real QGIS package was generated.

T7/S8 remained protected. No training, evaluation scores, promotion or digitization occurred.
QA logs are included for the checks that ran.
""",
    encoding="utf-8",
)
dump(BASE / "synthetic_text_dataset_audit.json", audit)
dump(BASE / "latest.json", {"run": str(run), "status": status})
manifest = run / "package_manifest.json"
dump(
    manifest,
    {
        "files": {
            p.name: sha256(p)
            for p in sorted(run.iterdir())
            if p.is_file() and p.stat().st_size and p != manifest
        }
    },
)
assert all(
    sha256(run / name) == value for name, value in json.loads(manifest.read_text())["files"].items()
)
print(run)
