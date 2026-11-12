"""Choose new development geography from actual validated points, not header boxes."""

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import shapely
from affine import Affine
from scipy.spatial import cKDTree
from shapely.geometry import box

from roadmarking.dataset.geometry import painted_area_from_centerline
from roadmarking.dataset.labels import LabelFeature
from roadmarking.dataset.taxonomy import Taxonomy, to_float
from roadmarking.dataset.verified_masks import reviewed_masks
from roadmarking.io.pointcloud import ChunkReadReport, iter_laz_chunks
from roadmarking.io.source_inventory import fingerprint, safe_header
from roadmarking.preprocessing.intensity import IntensityCalibration
from roadmarking.rasterization.pipeline import dump, sha256, write_tif
from roadmarking.rasterization.rasterizer import RasterAccumulator, world_to_pixel
from roadmarking.visualization.intensity_render import save_intensity_png, write_png_worldfile

root = Path(__file__).resolve().parents[1]
context = json.loads((root / "outputs/reports/phase5_8/latest.json").read_text())
main_report = Path(context["report_dir"])
run = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
report = root / "outputs/reports/phase5_8_actual_review" / run
output = root / "data/processed/phase5_8_actual_review" / run
report.mkdir(parents=True)
output.mkdir(parents=True)
audit = json.loads(
    (root / "outputs/reports/phase5/20261105T172517_source_audit/report.json").read_text()
)
train = json.loads((Path(context["real_parent"]) / "metadata/dataset_manifest.json").read_text())
old = json.loads((Path(train["parent"]) / "metadata/dataset_manifest.json").read_text())
used = set().union(*(set(r["source_global_ids"]) for r in old["regions"]))
bounds = [box(*r["native_xy_bbox"]) for r in old["regions"]]
for folder in ("phase5_6a/20261106T093757339115Z", "phase5_7_new_review/20261106T105319835838Z"):
    spec = json.loads(
        (root / "outputs/reports" / folder / "new_region_candidates.json").read_text()
    )
    used.update(uid for t in spec["tiles"] for uid in t["source_global_ids"])
    bounds.extend(box(*t["bounds"]) for t in spec["tiles"])
taxonomy = Taxonomy.load(root / "configs/phase4_taxonomy.yaml")
features, references = [], []
for index, source in enumerate(audit["annotation_layers"]):
    local = Path(source["local_snapshot"])
    assert sha256(local) == source["sha256"]
    copy = output / "references" / f"client_{index}.gpkg"
    copy.parent.mkdir(exist_ok=True)
    shutil.copyfile(local, copy)
    references.append({"path": str(copy), "layer": source["layer"]})
    frame = gpd.read_file(local, layer=source["layer"], fid_as_index=True)
    duplicate = frame.geometry.to_wkb().duplicated(keep=False)
    for fid, row in frame.iterrows():
        g = row.geometry
        if g is None or g.is_empty or not g.is_valid or duplicate.loc[fid]:
            continue
        line = g.geom_type in {"LineString", "MultiLineString"}
        decision = (
            taxonomy.classify_line(row.to_dict())
            if line
            else taxonomy.classify_polygon(row.to_dict())
        )
        width = to_float(row.get("LARGURA")) if line else None
        if line and (not width or width <= 0 or decision.visual_id not in (1, 2)):
            continue
        identity = f"{source['sha256']}|{source['layer']}|fid={fid}"
        paint = painted_area_from_centerline(g, width) if line else shapely.force_2d(g)
        features.append(
            {
                "global_id": hashlib.sha256(identity.encode()).hexdigest(),
                "source_identity": identity,
                "visual_id": decision.visual_id or -1,
                "width": width,
                "geometry": paint,
            }
        )
tree = shapely.STRtree([f["geometry"] for f in features])
candidates = []
for f in features:
    if f["visual_id"] != 2 or f["global_id"] in used:
        continue
    center = f["geometry"].centroid
    region = box(center.x - 25.6, center.y - 25.6, center.x + 25.6, center.y + 25.6)
    if min(region.distance(b) for b in bounds) < 500:
        continue
    hits = tree.query(region, predicate="intersects")
    if len(hits) < 8 or {features[i]["global_id"] for i in hits} & used:
        continue
    candidates.append((center.x, center.y, region, hits))
raw = json.loads((Path(audit["report_directory"]) / "raw_integrity.json").read_text())
source_spec = json.loads(
    (
        root
        / "outputs/reports/phase5_7_new_review/20261106T105319835838Z/new_region_candidates.json"
    ).read_text()
)
sources = [
    t["source"]
    for t in source_spec["tiles"]
    if t["candidate_id"] in ("NEW_DEV_REVIEW", "NEW_TRAIN_REVIEW")
]
calibration_path = Path(
    next(
        p
        for p in json.loads(
            (
                root / "outputs/reports/phase5_6a/20261106T093757339115Z/frozen_model.json"
            ).read_text()
        )["artifact_sha256"]
        if p.endswith("shared_normalization.json")
    )
)
cal = json.loads(calibration_path.read_text())
calibration = IntensityCalibration(
    **{
        k: cal[k]
        for k in ("low", "high", "percentile_low", "percentile_high", "statistics", "method")
    }
)
chosen = None
scans = []
for source in sources:
    path = Path(source)
    record = next(r for r in raw if r["path"] == source)
    before = fingerprint(path)
    assert before == record["fingerprint"]
    header = safe_header(path)
    parts = []
    progress = ChunkReadReport(path, 0, 0)
    for points in iter_laz_chunks(path, progress):
        parts.append(
            (np.asarray(points.x), np.asarray(points.y), np.asarray(points.intensity).copy())
        )
    assert progress.points_read == header.point_count and not progress.dropped_points
    assert fingerprint(path) == before
    x = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    intensity = np.concatenate([p[2] for p in parts])
    del parts
    scans.append(
        {"source": source, "points": progress.points_read, "dropped": 0, "fingerprint": before}
    )
    points_tree = cKDTree(np.column_stack((x[::100], y[::100])))
    if not candidates:
        break
    distances, _ = points_tree.query(np.asarray([(c[0], c[1]) for c in candidates]))
    for index in np.argsort(distances)[:20]:
        if distances[index] > 2:
            continue
        _, _, region, hits = candidates[index]
        x0, y0, x1, y1 = region.bounds
        keep = (x >= x0) & (x < x1) & (y > y0) & (y <= y1)
        affine = Affine(0.05, 0, x0, 0, -0.05, y1)
        accumulator = RasterAccumulator((1024, 1024))
        cols, rows = world_to_pixel(affine, x[keep], y[keep])
        inside = (cols >= 0) & (cols < 1024) & (rows >= 0) & (rows < 1024)
        accumulator.add(cols[inside], rows[inside], intensity[keep][inside])
        layers = accumulator.layers(calibration)
        observed = layers["observed"].astype(bool)
        fs = [
            LabelFeature(
                i + 1,
                features[j]["global_id"],
                features[j]["visual_id"],
                "polygon",
                features[j]["geometry"].intersection(region),
            )
            for i, j in enumerate(hits)
            if features[j]["visual_id"] in (1, 2)
        ]
        labels, _ = reviewed_masks(
            affine, observed.shape, observed, np.ones_like(observed), fs, completeness=region
        )
        counts = {
            cls: sum(r["visual_id"] == cls and r["pixels"] >= 4 for r in labels.instances)
            for cls in (1, 2)
        }
        pixels = int(((labels.instance > 0) & observed).sum())
        if observed.mean() >= 0.1 and pixels >= 64 and counts[1] >= 1 and counts[2] >= 3:
            chosen = (source, region, hits, affine, layers, counts, pixels)
            break
    if chosen:
        break
dump(report / "actual_point_scans.json", scans)
if not chosen:
    dump(
        main_report / "new_development_search.json",
        {
            "status": "NO_MEANINGFUL_CANDIDATE_IN_TWO_VALIDATED_ACQUISITIONS",
            "candidate_count": len(candidates),
            "report": str(report),
        },
    )
else:
    source, region, hits, affine, layers, counts, pixels = chosen
    name = "PHASE58_POSITIVE_DEV_REVIEW"
    layers["nodata"] = (~layers["observed"].astype(bool)).astype(np.uint8)
    layers["padding"] = np.zeros((1024, 1024), np.uint8)
    paths = {}
    for key, array in layers.items():
        p = output / "rasters" / name / (key + ".tif")
        write_tif(p, array, affine, None)
        paths[key] = str(p)
    png = output / "images" / (name + ".png")
    save_intensity_png(png, layers["normalized"], layers["observed"], gamma=0.7)
    write_png_worldfile(png, affine)
    rows = [
        {
            **features[j],
            "geometry": features[j]["geometry"].intersection(region),
            "candidate_id": name,
            "review_status": "NEW_OBSERVED_DEV_REVIEW_PENDING",
            "training_eligible": 0,
        }
        for j in hits
    ]
    store = output / "candidate_outlines.gpkg"
    pyogrio.write_dataframe(
        gpd.GeoDataFrame(rows, geometry="geometry", crs=None),
        store,
        layer="candidate_paint_footprints",
        driver="GPKG",
        promote_to_multi=True,
    )
    spec = {
        "output": str(output),
        "candidate_store": str(store),
        "references": references,
        "tiles": [
            {
                "candidate_id": name,
                "bounds": list(region.bounds),
                "affine": list(affine)[:6],
                "layers": paths,
                "images": {"grayscale": str(png)},
                "source": source,
                "observed_fraction": float(layers["observed"].mean()),
                "observed_solid_ids": counts[1],
                "observed_dash_ids": counts[2],
                "observed_paint_pixels": pixels,
                "source_global_ids": [features[j]["global_id"] for j in hits],
            }
        ],
    }
    dump(report / "new_region_candidates.json", spec)
    dump(
        root / "outputs/reports/phase5_8_actual_review/latest.json",
        {"report_dir": str(report), "output": str(output)},
    )
    dump(
        main_report / "new_development_search.json",
        {"status": "REAL_DATA_REVIEW_REQUIRED", "report": str(report), "support": spec["tiles"][0]},
    )
    print(json.dumps(spec["tiles"][0], indent=2))
