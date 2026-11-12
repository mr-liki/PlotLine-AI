"""Propose one observed-paint dash patch; original review geometry stays immutable."""

import hashlib
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import rasterio
from affine import Affine
from PIL import Image, ImageDraw
from rasterio.features import shapes
from scipy.ndimage import binary_dilation, label
from shapely import coverage_union_all
from shapely.geometry import shape

from phase5_11_common import ROOT, client_features, load
from roadmarking.rasterization.pipeline import dump

OUT = ROOT / "outputs/reports/stage8b"
LOCAL_ID = "stage8b_T7_05_missing_dash_001"


def main():
    if (OUT / "approval_manifest.json").exists():
        raise ValueError("Approval issued: preserve the bound proposal and QA evidence")
    location = load(OUT / "missing_mark_location.json")["winner"]
    tile = location["tile"]
    assert tile["candidate_id"] == "T7_05"
    affine = Affine(*tile["affine"])
    with rasterio.open(tile["layers"]["normalized"]) as ds:
        normalized, crs = ds.read(1), ds.crs
    with rasterio.open(tile["layers"]["observed"]) as ds:
        observed = ds.read(1).astype(bool)
    # Localized, source-image paint delineation. No model score or threshold is used.
    r0, r1, c0, c1 = 780, 840, 540, 650
    local = normalized[r0:r1, c0:c1]
    supported = observed[r0:r1, c0:c1]
    bright = (local > 0.30) & supported
    groups, n = label(binary_dilation(bright, iterations=2))
    candidates = [(int((bright & (groups == k)).sum()), k) for k in range(1, n + 1)]
    count, chosen = max(candidates)
    assert count > 100
    selected = bright & (groups == chosen)
    mask = np.zeros(observed.shape, bool)
    mask[r0:r1, c0:c1] = selected
    assert not (mask & ~observed).any()
    geometry = coverage_union_all(
        [shape(g) for g, v in shapes(mask.astype("uint8"), mask=mask, transform=affine) if v]
    )
    spec = load(location["manifest"])
    original = pyogrio.read_dataframe(spec["candidate_store"], layer="candidate_paint_footprints")
    area_overlap = float(original.geometry.intersection(geometry).area.sum())
    assert area_overlap < 1e-8, "Reported missing marking already has an outline"
    features = client_features(check_network=False)[0]
    client_overlaps = [f["global_id"] for f in features if f["geometry"].intersects(geometry)]
    assert (
        not client_overlaps
    ), "Inspect existing source feature association before proposing a new ID"
    store = OUT / "single_dash_proposed.gpkg"
    row = {
        "local_mark_id": LOCAL_ID,
        "client_global_id": None,
        "candidate_id": "T7_05",
        "class_name": "dash",
        "visual_id": 2,
        "model_class": 1,
        "geometry_status": "PROPOSED_PENDING_SINGLE_MARK_CONFIRMATION",
        "observed_paint_pixels": int(mask.sum()),
        "source": tile["source"],
        "geometry": geometry,
    }
    pyogrio.write_dataframe(
        gpd.GeoDataFrame([row], geometry="geometry", crs=crs),
        store,
        layer="proposed_dash_observed_paint",
        driver="GPKG",
        promote_to_multi=True,
    )
    # This administrative mask prevents pending/uncertain local paint becoming background.
    # It never changes source observations or the accepted annotation store.
    pending = binary_dilation(mask, iterations=2) & observed
    pending_geometry = coverage_union_all(
        [shape(g) for g, v in shapes(pending.astype("uint8"), mask=pending, transform=affine) if v]
    )
    pyogrio.write_dataframe(
        gpd.GeoDataFrame(
            [
                {
                    "local_mark_id": LOCAL_ID,
                    "policy": "PENDING_CORRECTION_EXCLUDE_FROM_SCORING_AND_VERIFIED_BACKGROUND",
                    "geometry": pending_geometry,
                }
            ],
            geometry="geometry",
            crs=crs,
        ),
        store,
        layer="pending_mark_review_extent",
        driver="GPKG",
        promote_to_multi=True,
        append=True,
    )
    np.savez_compressed(
        OUT / "single_dash_support.npz", proposed_paint=mask, pending_review_extent=pending
    )
    rows, cols = np.nonzero(mask)
    native = np.column_stack(affine * (cols + 0.5, rows + 0.5))
    axis = np.linalg.svd(native - native.mean(0), full_matrices=False)[2][0]
    projected = native @ axis
    sensitivity = {
        str(thr): int(((local > thr) & supported & binary_dilation(selected, iterations=1)).sum())
        for thr in (0.25, 0.30, 0.35, 0.40)
    }
    envelope = geometry.minimum_rotated_rectangle
    envelope_mask = np.zeros(observed.shape, dtype="uint8")
    from rasterio.features import rasterize

    envelope_mask = rasterize(
        [(envelope, 1)], out_shape=observed.shape, transform=affine, dtype="uint8"
    ).astype(bool)
    audit = {
        "region": "T7_05",
        "local_mark_id": LOCAL_ID,
        "client_global_id": None,
        "class_convention": {"name": "dash", "visual_id": 2, "model_class": 1},
        "source": tile["source"],
        "native_bounds": list(geometry.bounds),
        "native_centroid": [geometry.centroid.x, geometry.centroid.y],
        "native_observed_endpoint_centres": [
            native[np.argmin(projected)].tolist(),
            native[np.argmax(projected)].tolist(),
        ],
        "observed_paint_pixels": int(mask.sum()),
        "area_native2": geometry.area,
        "paint_pixels_without_observations": int((mask & ~observed).sum()),
        "oriented_envelope_pixels": int(envelope_mask.sum()),
        "oriented_envelope_nodata_pixels": int((envelope_mask & ~observed).sum()),
        "paint_geometry_filled_across_nodata": False,
        "delineation": "Within a visually isolated 110x60 pixel source-raster ROI, retain normalized intensity >0.30 AND observed. Two-pixel dilation groups fragments only; exported paint is the original selected pixels, not the dilated mask. One MultiPolygon feature preserves one dash identity. This local annotation proposal changes no AI threshold or score.",  # noqa: E501
        "normalized_intensity_sensitivity_pixel_counts": sensitivity,
        "CRS": (
            str(crs)
            if crs
            else "CRS_UNCONFIRMED; original native coordinates preserved; no EPSG or metre assumption"  # noqa: E501
        ),
        "existing_outline_overlap_area": area_overlap,
        "intersecting_client_feature_ids": client_overlaps,
        "approval": "PENDING_SINGLE_MARK_CONFIRMATION",
        "effective_ground_truth_modified": False,
        "background_policy": "Pending review extent is excluded; no visible unannotated local paint is asserted to be verified background.",  # noqa: E501
    }
    dump(OUT / "single_dash_geometry_nodata_audit.json", audit)
    # Focused before/after, same source image, extent and existing outlines.
    roi = (500, 760, 680, 860)
    image = Image.open(tile["images"]["grayscale"]).convert("RGB")
    panels = []
    inverse = ~affine
    for after in (False, True):
        im = image.crop(roi).resize((900, 500), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(im)

        def draw_geom(geom, color, draw=draw):
            parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
            for part in parts:
                for ring in [part.exterior, *part.interiors]:
                    coords = [inverse * xy for xy in ring.coords]
                    coords = [((x - roi[0]) * 5, (y - roi[1]) * 5) for x, y in coords]
                    draw.line(coords, fill=color, width=2)

        for _, feature in original.iterrows():
            if feature.geometry.intersects(pending_geometry.buffer(5)):
                draw_geom(feature.geometry, "#0080ff" if feature["class_name"] == "dash" else "red")
        if after:
            draw_geom(geometry, "#0080ff")
        panels.append(im)
    qa = Image.new("RGB", (1820, 560), "white")
    draw = ImageDraw.Draw(qa)
    for k, panel in enumerate(panels):
        qa.paste(panel, (k * 920, 35))
        draw.text(
            (k * 920 + 10, 10),
            (
                "AFTER - proposed one-dash outline; confirmation required"
                if k
                else "BEFORE - accepted outlines, one missing dash"
            ),
            fill="black",
        )
    draw.text(
        (10, 540),
        "T7_05 | native 0.05 grid | blue=dash, red=solid | observed paint only; no source geometry edited",  # noqa: E501
        fill="black",
    )
    qa.save(OUT / "single_dash_before_after.png")
    acceptance = load(OUT / "human_review_acceptance.json")
    for path in [
        Path(spec["candidate_store"]),
        Path(spec["review_edits_store"]),
        Path(location["preview"]),
        *[Path(r["path"]) for r in spec.get("references", [])],
    ]:
        acceptance["input_sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    acceptance["localized_exception_region"] = "T7_05"
    acceptance["proposed_local_mark_id"] = LOCAL_ID
    dump(OUT / "human_review_acceptance.json", acceptance)
    print(audit)


if __name__ == "__main__":
    main()
