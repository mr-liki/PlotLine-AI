"""Finish review copies after strict XY/Z readback; never approves training labels."""

import json
import os
import subprocess
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import shapely
from shapely.geometry import Point

from roadmarking.dataset.rigid_registration import transform_geometry_xy
from roadmarking.rasterization.pipeline import dump, sha256


def main():
    root = Path(__file__).resolve().parents[1]
    directory = root / "outputs/reports/phase4_fitted_annotations"
    report = json.loads((directory / "report.json").read_text())
    out = Path(report["output_root"])
    old = Path(report["geopackage"])
    gpkg = out / "annotations_fitted_XY_ONLY_UNVERIFIED.gpkg"
    if gpkg.exists():
        raise FileExistsError("Use the completed review package; do not overwrite it")
    matrix = np.asarray(report["transform_3x3"])
    rotation, translation = matrix[:2, :2], matrix[:2, 2]
    checks = []
    for item in report["exported_layers"]:
        original = Path(item["source"])
        local = root / "data/raw/annotations" / original.parent.name / original.name
        if sha256(local) != item["source_sha256"]:
            raise ValueError("Original annotation copy changed")
        source_layer = next(layer for layer, kind in pyogrio.list_layers(local) if kind)
        frame = gpd.read_file(local, layer=source_layer, fid_as_index=True, engine="pyogrio")
        expected = frame.geometry.map(lambda g: transform_geometry_xy(g, rotation, translation))
        result = frame.copy()
        result["fit_source_geometry_valid"] = result.geometry.is_valid.astype(int)
        result.geometry = expected
        result = result.set_crs(None, allow_override=True)
        result["fit_source_file"] = str(original)
        result["fit_source_layer"] = source_layer
        result["fit_source_fid"] = result.index.astype(str)
        result["fit_global_id"] = [
            f"{original.parent.name}/{original.name}|{source_layer}|fid={fid}"
            for fid in result.index
        ]
        result["fit_status"] = "UNVERIFIED_GEOMETRIC_PLACEMENT"
        result["fit_review_status"] = "PROPOSED_UNREVIEWED"
        result["fit_training_eligible"] = 0
        result["fit_z_status"] = "ORIGINAL_Z_PRESERVED_NOT_REGISTERED"
        pyogrio.write_dataframe(result, gpkg, layer=item["layer"], driver="GPKG")
        readback = gpd.read_file(gpkg, layer=item["layer"], engine="pyogrio")
        a = shapely.get_coordinates(frame.geometry.array, include_z=True)
        b = shapely.get_coordinates(readback.geometry.array, include_z=True)
        if len(a) != len(b) or not np.isfinite(b[:, :2]).all():
            raise ValueError("Vertex loss or nonfinite copied XY")
        reference = shapely.get_coordinates(expected.array, include_z=True)
        np.testing.assert_allclose(reference[:, :2], b[:, :2], atol=1e-8, rtol=0)
        np.testing.assert_allclose(a[:, 2], b[:, 2], atol=0, rtol=0, equal_nan=True)
        if len(readback) != len(frame):
            raise ValueError("Feature count changed")
        checks.append(
            {
                "layer": item["layer"],
                "features": len(frame),
                "vertices": len(a),
                "nonfinite_target_xy": 0,
                "z_preserved": True,
                "unknown_z_vertices_preserved": int(np.isnan(a[:, 2]).sum()),
            }
        )
    footprint = gpd.read_file(old, layer="raw_header_footprints", engine="pyogrio")
    pyogrio.write_dataframe(footprint, gpkg, layer="raw_header_footprints", driver="GPKG")
    cache = np.load(out / "metadata/registration_samples.npz")
    points = []
    for group, key in (("FIT_RECORDS", "target"), ("CHECK_RECORDS", "check")):
        for index, xy in enumerate(cache[key]):
            points.append(
                {
                    "sample_group": group,
                    "sample_index": index,
                    "role": "BRIGHT_PROXY_NOT_VERIFIED_PAINT",
                    "geometry": Point(xy),
                }
            )
    pyogrio.write_dataframe(
        gpd.GeoDataFrame(points, crs=None), gpkg, layer="raw_bright_sample_points", driver="GPKG"
    )
    if not all(sha256(Path(p)) == digest for p, digest in report["source_hashes"].items()):
        raise ValueError("Original network annotation changed")
    # Disable the earlier provisional export after checking workspace containment.
    superseded = old.with_suffix(".gpkg.superseded")
    if not old.resolve().is_relative_to(root) or not superseded.resolve().is_relative_to(root):
        raise ValueError("Unexpected output location")
    old.rename(superseded)
    report["geopackage"] = str(gpkg)
    report["geometry_roundtrip"] = checks
    report["superseded_export"] = str(superseded)
    report["export_correction"] = (
        "Original unknown Z preserved independently; 73 affected XY vertices corrected "
        "without changing the fitted matrix"
    )
    report["paint_alignment_gate"] = "FAILED_GROSS_PROXY_RESIDUAL_AND_NO_VERIFIED_LANDMARKS"
    report["qgis_project"] = str(out / "fitted_annotations_UNVERIFIED.qgs")
    dump(directory / "report.json", report)
    dump(out / "metadata/fit_report.json", report)
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    process = subprocess.run(
        [
            "C:/Program Files/QGIS 3.44.7/bin/python-qgis.bat",
            str(root / "scripts/create_fitted_annotation_review.py"),
            str(out / "metadata/fit_report.json"),
        ],
        env=env,
        text=True,
        capture_output=True,
    )
    if process.returncode:
        raise RuntimeError(process.stdout + process.stderr)
    report["qgis_project_reopened"] = True
    dump(directory / "report.json", report)
    dump(out / "metadata/fit_report.json", report)
    (directory / "report.md").write_text(
        "# Requested annotation fit — review only\n\n"
        f"Fitted copy: `{gpkg}`\n\n"
        f"QGIS project: `{report['qgis_project']}`\n\n"
        f"Features: {report['features_exported']:,}; all source attributes retained. "
        "XY-only rigid transform, scale 1; source Z retained without registration. "
        "Unknown Z remains unknown and no longer affects XY.\n\n"
        f"Median nearest bright-sample residual: {report['fitted_native_distances']['p50']:.3f}; "
        f"p95: {report['fitted_native_distances']['p95']:.3f} native units. "
        "The best of 30 tested starts does not align paint at marking scale.\n\n"
        "Status: **UNVERIFIED GEOMETRIC PLACEMENT — NOT ELIGIBLE FOR TRAINING**. "
        "No training masks, weights or model training were generated. "
        "Original annotations and raw files were opened read-only.\n",
        encoding="utf-8",
    )
    dump(
        out / "metadata/completion.json",
        {
            "status": "UNVERIFIED_REVIEW_COPY",
            "geopackage_sha256": sha256(gpkg),
            "geometry_roundtrip": checks,
            "training_eligible": False,
        },
    )
    print(
        json.dumps(
            {
                "geopackage": str(gpkg),
                "qgis_project": report["qgis_project"],
                "features": report["features_exported"],
                "geometry_roundtrip": "PASSED",
                "training_eligible": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
