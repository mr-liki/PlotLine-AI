"""Stage 9E-E: QGIS review data for verified non-text hard negatives.

For approved training crops that contain client-located non-line paint (arrows,
hatching, give-way, safety markings, ...), write read-only intensity/validity GeoTIFFs,
crop footprints, read-only client symbol candidates and a separate editable review layer.
Nothing is approved here; client geometry is a locator only. N9_01, development crops and
protected T7/S8 data are excluded by construction (training split only, asserted).
"""

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.features import rasterize
from shapely import wkt
from shapely.strtree import STRtree

from phase5_11_common import client_features
from stage9c_attested_pilot import crop_box
from stage9d_common import ROOT, dump
from stage9e_hard_negatives import NAMES
from stage9e_lib import new_run, real_context


def main():
    out = new_run("negative_review_package")
    data = ROOT / "data/processed/stage9e_negative_review" / out.name
    (data / "rasters").mkdir(parents=True)
    rc = real_context()
    feats, _, _ = client_features(check_network=False)
    geoms = [
        wkt.loads(f["geometry"]) if isinstance(f["geometry"], str) else f["geometry"] for f in feats
    ]
    tree = STRtree(geoms)
    symbol_rows, crop_rows = {}, []
    for crop in rc["full"]:
        fp = crop_box(crop)
        hits = [i for i in tree.query(fp) if int(feats[i]["visual_id"]) in NAMES]
        if not hits:
            continue
        with np.load(rc["dataset"] / crop["file"]) as z:
            rgb, valid = z["rgb"][..., 0].copy(), z["validity"].copy()
        transform = Affine(*crop["affine"][:6])
        observed = 0
        for i in hits:
            m = rasterize(
                [(geoms[i], 1)], out_shape=valid.shape, transform=transform, all_touched=True
            ).astype(bool)
            obs = int((m & (rgb > 0)).sum())
            observed += obs
            row = symbol_rows.setdefault(
                feats[i]["global_id"],
                dict(
                    global_id=feats[i]["global_id"],
                    visual_id=int(feats[i]["visual_id"]),
                    class_name=NAMES[int(feats[i]["visual_id"])],
                    crops=[],
                    observed_pixels=0,
                    geometry=geoms[i],
                ),
            )
            row["crops"].append(crop["crop_id"])
            row["observed_pixels"] += obs
        for name, arr in (("intensity", rgb), ("validity", valid.astype(np.uint8))):
            with rasterio.open(
                data / "rasters" / f"{crop['crop_id']}_{name}.tif",
                "w",
                driver="GTiff",
                height=arr.shape[0],
                width=arr.shape[1],
                count=1,
                dtype="uint8",
                transform=transform,
            ) as ds:
                ds.write(arr, 1)
        crop_rows.append(
            dict(
                crop_id=crop["crop_id"],
                tile_id=crop["tile_id"],
                geography=crop["geography"],
                symbols=len(hits),
                observed_symbol_pixels=observed,
                geometry=fp,
            )
        )
    sym = gpd.GeoDataFrame(
        [{**r, "crops": ";".join(r["crops"])} for r in symbol_rows.values()],
        geometry="geometry",
        crs=None,
    )
    gpkg = data / "stage9e_non_text_review.gpkg"
    sym.to_file(gpkg, layer="client_symbol_candidates_READ_ONLY", driver="GPKG")
    review = sym[
        ["global_id", "visual_id", "class_name", "crops", "observed_pixels", "geometry"]
    ].copy()
    review["review_status"] = "PENDING_HUMAN_CONFIRMATION"
    review["confirmed_visible_paint"] = None
    review["confirmed_non_text"] = None
    review["footprint_correct"] = None
    review["reviewer_note"] = None
    review.to_file(gpkg, layer="non_text_review_EDITABLE", driver="GPKG")
    gpd.GeoDataFrame(crop_rows, geometry="geometry", crs=None).to_file(
        gpkg, layer="training_crop_footprints", driver="GPKG"
    )
    by_class = {}
    for r in symbol_rows.values():
        by_class[r["class_name"]] = by_class.get(r["class_name"], 0) + 1
    spec = dict(
        data=str(data),
        gpkg=str(gpkg),
        crops=[r["crop_id"] for r in crop_rows],
        symbols=len(symbol_rows),
        symbols_by_class=by_class,
        observed_symbol_pixels=int(sum(r["observed_pixels"] for r in symbol_rows.values())),
        role="TRAINING_SPLIT_NON_TEXT_CONTEXT_REVIEW_ONLY",
        approval_recorded=False,
        excluded=["N9_01", "development crops", "T7", "S8", "future mixed 500 test"],
        instructions=(
            "For each symbol in non_text_review_EDITABLE: confirm the paint is visible in intensity, "  # noqa: E501
            "the footprint covers it, and it is NOT road text. Set review_status to CONFIRMED or "
            "REJECTED. Only CONFIRMED symbols may become text-class negatives; solid/dash "
            "supervision is unchanged."
        ),
    )
    dump(out / "review_spec.json", spec)
    dump(data / "review_spec.json", spec)
    print(out, len(crop_rows), len(symbol_rows), by_class)


if __name__ == "__main__":
    main()
