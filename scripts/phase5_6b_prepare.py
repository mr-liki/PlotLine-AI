"""Create auditable editable test candidates without approving or moving geometry."""

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Polygon

from roadmarking.rasterization.pipeline import dump, sha256

root = Path(__file__).resolve().parents[1]
prior = root / "outputs/reports/phase5_6a/20261106T093757339115Z"
spec = json.loads((prior / "new_region_candidates.json").read_text())
run = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
output = root / "data/processed/phase5_6b_correction" / run
report = root / "outputs/reports/phase5_6b" / run
output.mkdir(parents=True)
report.mkdir(parents=True)
screenshot = Path("C:/Users/nt01173/AppData/Local/Temp/")
screenshot /= "codex-clipboard-d0d0d47b-8f8f-4a71-ac06-c004de6708a4.png"
query = cv2.imread(str(screenshot), 0)
orb = cv2.ORB_create(5000)
keys, desc = orb.detectAndCompute(query, None)
matches = []
for tile in spec["tiles"]:
    path = Path(spec["output"]) / "review_previews_verified" / (tile["candidate_id"] + ".png")
    image = cv2.imread(str(path), 0)
    target_keys, target_desc = orb.detectAndCompute(image, None)
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc, target_desc, k=2)
    good = [a for a, b in pairs if a.distance < 0.7 * b.distance]
    src = np.float32([keys[m.queryIdx].pt for m in good])
    dst = np.float32([target_keys[m.trainIdx].pt for m in good])
    matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3)
    matches.append((int(inliers.sum()), tile, matrix, image.shape))
matches.sort(key=lambda item: item[0], reverse=True)
count, affected, matrix, shape = matches[0]
assert count >= 20 and count > matches[1][0] * 2
corners = np.float32(
    [[0, 0], [query.shape[1], 0], [query.shape[1], query.shape[0]], [0, query.shape[0]]]
)
pixels = cv2.perspectiveTransform(corners[None], matrix)[0]
# QGIS preview renders the full square bounds at 1024 pixels.
x0, y0, x1, y1 = affected["bounds"]
world = [(x0 + x / shape[1] * (x1 - x0), y1 - y / shape[0] * (y1 - y0)) for x, y in pixels]
issue = Polygon(world)
frame = gpd.read_file(
    spec["candidate_store"], layer="candidate_paint_footprints", fid_as_index=True
)
frame["original_fid"] = frame.index.astype(int)
frame["correction_id"] = [
    hashlib.sha256((r.candidate_id + "|" + r.global_id).encode()).hexdigest()
    for _, r in frame.iterrows()
]
frame["original_geometry_sha256"] = [hashlib.sha256(g.wkb).hexdigest() for g in frame.geometry]
frame["edit_action"] = "KEEP"
frame["edit_reason"] = ""
frame["review_status"] = "PENDING_EXPLICIT_GEOMETRY_AND_COMPLETENESS_APPROVAL"
frame["issue_candidate"] = (
    (frame.candidate_id == affected["candidate_id"]) & frame.geometry.intersects(issue)
).astype(int)
store = output / "editable_test_candidates.gpkg"
frame.to_file(store, layer="corrections", driver="GPKG", index=False)
with sqlite3.connect(store) as db:
    db.execute(
        "CREATE TABLE edit_history (event_id INTEGER PRIMARY KEY, utc TEXT, operation TEXT, "
        "correction_id TEXT, source_global_id TEXT, old_geometry_hex TEXT, "
        "new_geometry_hex TEXT, old_class INTEGER, new_class INTEGER, reason TEXT)"
    )
    for operation, old, new in (
        ("UPDATE", "old", "new"),
        ("DELETE", "old", "old"),
        ("INSERT", "new", "new"),
    ):
        db.execute(
            f"CREATE TRIGGER log_{operation.lower()} AFTER {operation} ON corrections "
            f"BEGIN INSERT INTO edit_history(utc,operation,correction_id,source_global_id,"
            f"old_geometry_hex,new_geometry_hex,old_class,new_class,reason) VALUES "
            f"(strftime('%Y-%m-%dT%H:%M:%fZ','now'),'{operation}',{new}.correction_id,"
            f"{new}.global_id,hex({old}.geom),hex({new}.geom),{old}.visual_id,"
            f"{new}.visual_id,{new}.edit_reason); END"
        )
observations = []
for tile in spec["tiles"]:
    with rasterio.open(tile["layers"]["observed"]) as ds:
        observed, affine = ds.read(1).astype(bool), ds.transform
    for _, row in frame[
        (frame.candidate_id == tile["candidate_id"]) & (frame.issue_candidate == 1)
    ].iterrows():
        mask = rasterize([(row.geometry, 1)], observed.shape, transform=affine).astype(bool)
        observations.append(
            {
                "correction_id": row.correction_id,
                "global_id": row.global_id,
                "source_identity": row.source_identity,
                "original_geometry_sha256": row.original_geometry_sha256,
                "observed_pixels": int((mask & observed).sum()),
                "unobserved_pixels": int((mask & ~observed).sum()),
                "status": "SCREENSHOT_AREA_REQUIRES_HUMAN_FEATURE_SELECTION",
            }
        )
dump(report / "issue_candidates.json", observations)
dump(
    report / "change_log.json",
    {
        "geometry_changes": [],
        "class_changes": [],
        "reason": "No automatic corrections authorized from brightness or screenshot alone",
        "baseline_rows": len(frame),
        "baseline_geometry_hashes": dict(
            zip(frame.correction_id, frame.original_geometry_sha256, strict=True)
        ),
    },
)
dump(
    report / "approval_record.json",
    {
        "affected_region": affected["candidate_id"],
        "other_regions": [t["candidate_id"] for t in spec["tiles"] if t != affected],
        "other_region_observation": "User reports visually satisfactory; geometry retained",
        "complete_visible_target_approval": {
            t["candidate_id"]: "NOT_EXPLICITLY_CONFIRMED" for t in spec["tiles"]
        },
        "ground_truth_export_allowed": False,
        "evaluation_allowed": False,
    },
)
dump(
    report / "screenshot_registration.json",
    {
        "screenshot_sha256": sha256(screenshot),
        "region": affected["candidate_id"],
        "ransac_inliers": count,
        "comparison_inliers": {t["candidate_id"]: n for n, t, _, _ in matches},
        "world_polygon": [[float(x), float(y)] for x, y in world],
        "homography": matrix.tolist(),
        "scope": "Locates issue vicinity; does not establish which outlines are incorrect",
    },
)
frozen = json.loads((prior / "frozen_model.json").read_text())
for filename, expected in frozen["artifact_sha256"].items():
    assert sha256(filename) == expected
assert frame.geometry.is_valid.all()
dump(
    report / "preservation_manifest.json",
    {
        str(p): sha256(p)
        for p in [
            Path(spec["candidate_store"]),
            Path(spec["output"]) / "independent_test_review.qgs",
            prior / "new_region_candidates.json",
            *[Path(t["layers"]["observed"]) for t in spec["tiles"]],
        ]
    },
)
context = {
    "output": str(output),
    "report_dir": str(report),
    "candidate_store": str(store),
    "affected_region": affected["candidate_id"],
    "spec": str(prior / "new_region_candidates.json"),
    "issue_bounds": list(issue.bounds),
    "status": "HUMAN_CORRECTION_REQUIRED",
}
dump(report.parent / "latest.json", context)
(report / "report.md").write_text(
    "# Phase 5.6B — human correction required\n\n"
    f"Screenshot registration identifies {affected['candidate_id']} ({count} RANSAC inliers). "
    f"{len(observations)} outlines intersect its vicinity; issue_candidates.json records exact "
    "IDs, geometry hashes and observed/unobserved pixel counts. These are investigation "
    "candidates, not automatic deletion proposals.\n\n"
    "Original layers, project, rasters and frozen AI artifacts are preserved. All candidate "
    "geometries are initially identical. Other two regions remain unedited. Their reported "
    "visual satisfaction is recorded without inventing completeness approval.\n\n"
    "Edit only the correction layer. Use vertex editing for position/width, visual_id 1 for "
    "solid and 2 for dash; other symbols remain unsupported exclusions. For removal use "
    "edit_action=REMOVE (retain geometry), for changes use ADJUST, for additions use ADD "
    "with a fresh correction_id and blank source identity. Record edit_reason. Never delete "
    "an outline just because its pixels are unobserved. Database triggers retain before/after "
    "geometry bytes and class history for edits/inserts/deletes. Final geometry SHA-256 "
    "must be computed and human-confirmed before export.\n\n"
    "After correction, explicit confirmation of corrected geometry and complete visible "
    "target coverage is required per region, excluding NoData, ambiguous areas and unsupported "
    "symbols. No model inference, metrics, retraining or threshold changes performed.\n",
    encoding="utf-8",
)
print(json.dumps(context, indent=2))
