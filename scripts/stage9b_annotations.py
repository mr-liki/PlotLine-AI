"""Observed-only road-text candidates and uncertainty review; never supervised GT."""

import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import rasterio
from affine import Affine
from PIL import Image, ImageDraw
from rasterio.features import rasterize, shapes
from scipy.ndimage import binary_dilation
from shapely import coverage_union_all
from shapely.geometry import shape

from phase5_11_common import ROOT, load
from roadmarking.rasterization.pipeline import dump, sha256


def geometry(mask, affine):
    return coverage_union_all(
        [shape(g) for g, v in shapes(mask.astype("uint8"), mask=mask, transform=affine) if v]
    )


def main():
    base = ROOT / "outputs/reports/stage9b"
    latest = load(base / "latest.json")
    out = Path(latest["report_dir"])
    data = ROOT / "data/processed/stage9b_review" / latest["run_id"]
    data.mkdir(parents=True, exist_ok=True)
    stage9 = Path(load(ROOT / "outputs/reports/stage9/latest.json")["report_dir"])
    spec = load(stage9 / "new_region_candidates.json")
    base_store = data / "stage9_candidates_readonly.gpkg"
    shutil.copyfile(spec["candidate_store"], base_store)
    originals = pyogrio.read_dataframe(base_store, layer="candidate_paint_footprints")
    unsupported = pyogrio.read_dataframe(base_store, layer="unsupported_marking_regions")
    characters, fragments, uncertainty, audits = [], [], [], []
    words = [
        ("G01", (0, 0, 62, 40), "PARTIAL_TEXT_AT_TILE_BOUNDARY", 0.25),
        ("G02", (140, 0, 224, 40), "PARTIAL_TEXT_AT_TILE_BOUNDARY", 0.25),
        ("G03", (55, 40, 144, 109), "CHARACTER_SPLITS_REQUIRE_HUMAN_CONFIRMATION", 0.25),
        ("G04", (0, 118, 63, 180), "FAINT_TEXT_FRAGMENT_GROUP_REQUIRES_REVIEW", 0.23),
        ("G05", (130, 118, 194, 180), "FAINT_TEXT_FRAGMENT_GROUP_REQUIRES_REVIEW", 0.23),
        ("G06", (48, 180, 110, 237), "POSSIBLE_FAINT_TEXT_UNCLASSIFIED", 0.25),
    ]
    for t in spec["tiles"]:
        region = t["candidate_id"]
        affine = Affine(*t["affine"])
        with rasterio.open(t["layers"]["normalized"]) as ds:
            intensity = ds.read(1)
        with rasterio.open(t["layers"]["observed"]) as ds:
            observed = ds.read(1).astype(bool)
        all_known = rasterize(
            [(g, 1) for g in [*originals.geometry, *unsupported.geometry]],
            out_shape=observed.shape,
            transform=affine,
            dtype="uint8",
        ).astype(bool)
        known_margin = binary_dilation(all_known, iterations=1)
        block = np.zeros_like(observed)
        text = np.zeros_like(observed)
        if region == "N9_01":
            for group, roi, status, cutoff in words:
                x0, y0, x1, y1 = roi
                extent = np.zeros_like(observed)
                extent[y0:y1, x0:x1] = True
                block |= extent & observed
                paint = extent & observed & (intensity > cutoff) & ~known_margin
                assert not (paint & ~observed).any()
                group_id = f"{region}_TEXT_{group}"
                if group == "G03":
                    bands = [(40, 55), (55, 68), (68, 81), (81, 95), (95, 109)]
                    for i, (a, b) in enumerate(bands, 1):
                        char = paint.copy()
                        char[:a] = False
                        char[b:] = False
                        if not char.any():
                            continue
                        characters.append(
                            {
                                "candidate_id": region,
                                "instance_id": f"{group_id}_C{i:02d}",
                                "text_group_id": group_id,
                                "visual_id": 9,
                                "model_id": 2,
                                "class_name": "road_text",
                                "character_value": None,
                                "character_value_status": "HUMAN_CONFIRMATION_REQUIRED",
                                "client_global_id": None,
                                "source_identity": None,
                                "source_file": t["source"],
                                "observed_pixels": int(char.sum()),
                                "review_status": "DRAFT_CHARACTER_SPLIT_REVIEW_REQUIRED",
                                "training_eligible": 0,
                                "geometry": geometry(char, affine),
                            }
                        )
                        text |= char
                elif paint.any():
                    fragments.append(
                        {
                            "candidate_id": region,
                            "instance_id": f"{group_id}_FRAGMENT",
                            "text_group_id": group_id,
                            "suggested_class": (
                                "road_text" if group != "G06" else "UNCLASSIFIED_POSSIBLE_TEXT"
                            ),
                            "model_id": None,
                            "character_value": None,
                            "client_global_id": None,
                            "source_file": t["source"],
                            "observed_pixels": int(paint.sum()),
                            "review_status": status,
                            "export_as_character_ground_truth": 0,
                            "geometry": geometry(paint, affine),
                        }
                    )
                    if group != "G06":
                        text |= paint
                uncertainty.append(
                    {
                        "candidate_id": region,
                        "issue_id": group_id,
                        "issue": status,
                        "policy": "UNCONFIRMED_TEXT_EXTENT_IGNORE_NOT_BACKGROUND",
                        "geometry": geometry(extent & observed, affine),
                    }
                )
        # Explicitly uncertain features; never automatically interpreted as characters.
        ambiguous_rois = {"N9_02": (820, 530, 870, 640), "N9_03": (90, 775, 165, 845)}
        if region in ambiguous_rois:
            x0, y0, x1, y1 = ambiguous_rois[region]
            extent = np.zeros_like(observed)
            extent[y0:y1, x0:x1] = True
            bright = extent & observed & (intensity > 0.4) & ~known_margin
            if bright.any():
                fragments.append(
                    {
                        "candidate_id": region,
                        "instance_id": f"{region}_UNCLASSIFIED_BRIGHT_01",
                        "text_group_id": None,
                        "suggested_class": "UNCLASSIFIED_BRIGHT_FEATURE_NOT_ASSIGNED_TO_TEXT",
                        "model_id": None,
                        "character_value": None,
                        "client_global_id": None,
                        "source_file": t["source"],
                        "observed_pixels": int(bright.sum()),
                        "review_status": "INSPECT_PAINT_VS_ROADSIDE_OBJECT_OR_OTHER_SYMBOL",
                        "export_as_character_ground_truth": 0,
                        "geometry": geometry(bright, affine),
                    }
                )
                uncertainty.append(
                    {
                        "candidate_id": region,
                        "issue_id": f"{region}_BRIGHT_01",
                        "issue": "UNCLASSIFIED_SUPPORTED_FEATURE",
                        "policy": "UNKNOWN_IGNORE_NOT_BACKGROUND",
                        "geometry": geometry(extent & observed & ~all_known, affine),
                    }
                )
                block |= extent & observed & ~all_known
        np.savez_compressed(
            out / f"{region}_review_safety.npz",
            unconfirmed_review_extent=block,
            text_candidate_observed_paint=text,
            observed_support=observed,
        )
        source_image = Path(t["images"]["grayscale"])
        image = data / "images" / source_image.name
        image.parent.mkdir(exist_ok=True)
        shutil.copyfile(source_image, image)
        for suffix in (".pgw", ".pngw", ".wld"):
            p = source_image.with_suffix(suffix)
            if p.exists():
                shutil.copyfile(p, image.with_suffix(suffix))
        t["images"] = {"grayscale": str(image)}
        audits.append(
            {
                "region": region,
                "road_text_visibility": (
                    "CONFIRMED_VISIBLE_TEXT_CANDIDATES_REQUIRE_GEOMETRY_REVIEW"
                    if region == "N9_01"
                    else "NO_CONFIDENT_ROAD_TEXT_IDENTIFIED_IN_VISIBLE_SOURCE_IMAGE"
                ),
                "additional_review_issue": (
                    "UNCLASSIFIED_SUPPORTED_BRIGHT_FEATURE" if region in ambiguous_rois else None
                ),
                "text_candidate_observed_pixels": int(text.sum()),
                "review_extent_observed_pixels": int(block.sum()),
                "unobserved_paint_pixels_exported": int((text & ~observed).sum()),
                "source_image_sha256": sha256(source_image),
                "source_geometry_unchanged": True,
                "complete_visible_target_review_status": "NOT_APPROVED",
                "role": t["proposed_role"],
            }
        )
    store = data / "road_text_candidates.gpkg"
    for layer, rows in (
        ("character_candidates", characters),
        ("group_and_unknown_fragments", fragments),
        ("review_uncertainty_extents", uncertainty),
    ):
        if rows:
            pyogrio.write_dataframe(
                gpd.GeoDataFrame(rows, geometry="geometry", crs=None),
                store,
                layer=layer,
                driver="GPKG",
                promote_to_multi=True,
                append=store.exists(),
            )
    editable = [{**r, "review_decision": "", "reviewer_note": ""} for r in characters]
    pyogrio.write_dataframe(
        gpd.GeoDataFrame(editable, geometry="geometry", crs=None),
        store,
        layer="reviewer_characters",
        driver="GPKG",
        promote_to_multi=True,
        append=True,
    )
    pyogrio.write_dataframe(
        gpd.GeoDataFrame(
            {
                "candidate_id": [],
                "suggested_class": [],
                "character_value": [],
                "text_group_id": [],
                "reviewer_note": [],
            },
            geometry=[],
            crs=None,
        ),
        store,
        layer="reviewer_missing_paint",
        driver="GPKG",
        geometry_type="MultiPolygon",
        append=True,
    )
    # Focused QA for the text-bearing area, with labels only as local instance IDs.
    t = next(t for t in spec["tiles"] if t["candidate_id"] == "N9_01")
    original = (
        Image.open(t["images"]["grayscale"])
        .convert("RGB")
        .crop((0, 0, 240, 280))
        .resize((720, 840), Image.Resampling.NEAREST)
    )
    after = original.copy()
    draw = ImageDraw.Draw(after)
    inverse = ~Affine(*t["affine"])

    def outline(geom, color):
        parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for part in parts:
            for ring in [part.exterior, *part.interiors]:
                draw.line(
                    [(x * 3, y * 3) for x, y in [inverse * p for p in ring.coords]],
                    fill=color,
                    width=1,
                )

    for row in characters:
        outline(row["geometry"], "cyan")
    for row in fragments:
        if row["candidate_id"] == "N9_01":
            outline(row["geometry"], "orange")
    panel = Image.new("RGB", (1460, 880), "white")
    panel.paste(original, (0, 30))
    panel.paste(after, (740, 30))
    d = ImageDraw.Draw(panel)
    d.text((10, 8), "BEFORE: source text has no client outlines", fill="black")
    d.text((750, 8), "DRAFT: cyan character splits; orange fragments/uncertain text", fill="black")
    d.text(
        (10, 865),
        "Native observed pixels only | values blank | separation, boundaries and completeness require human confirmation",  # noqa: E501
        fill="black",
    )
    panel.save(out / "N9_01_text_before_after.png")
    dump(
        out / "four_region_coverage_audit.json",
        {
            "regions": audits,
            "character_split_candidates": len(characters),
            "fragment_and_unknown_candidates": len(fragments),
            "character_values_invented": False,
            "supervision_enabled": False,
            "limitations": "Photometric traces are draft pixel-support candidates, not automatic OCR, full glyph recovery or complete-visible-target approval. Faint and tile-clipped text requires manual partition/boundary confirmation; uncertainty extents prevent unconfirmed paint becoming verified background.",  # noqa: E501
        },
    )
    dump(
        out / "review_spec.json",
        {
            "stage": "9B",
            "output": str(data),
            "base_stage9_candidates": str(base_store),
            "text_candidates": str(store),
            "tiles": spec["tiles"],
            "training_eligible": False,
            "first_review_scope": "New text geometry/labels, ambiguous supported features and complete visible-target coverage in the four still-unapproved Stage-9 regions; do not repeat T7/S8 reviews or alter original solid/dash geometry.",  # noqa: E501
        },
    )
    dump(base / "latest.json", {**latest, "data_dir": str(data)})
    print(
        f"Created {len(characters)} character-split drafts and {len(fragments)} fragment/unknown candidates; no character values or approvals invented"  # noqa: E501
    )


if __name__ == "__main__":
    main()
