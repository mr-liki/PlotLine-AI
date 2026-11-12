"""Independent persisted-output QA, QGIS review bundle and validation summary."""

# ruff: noqa: E501 -- complete Markdown sentences in the audit report.

import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import box, shape

from roadmarking.analysis.corridor_pilot import NS, load_window, read_json
from roadmarking.preprocessing.corridor_boundaries import build_boundaries
from roadmarking.preprocessing.road_cross_sections import SectionSettings, extract_sections
from roadmarking.rasterization.pipeline import dump, sha256


def read_layer(path):
    with rasterio.open(path) as ds:
        return ds.read(1)


def bright_components(mask):
    # Only measures a pixel graph; no inferred pixels are written to the raster.
    labels, count = ndimage.label(mask, np.ones((3, 3)))
    sizes = np.bincount(labels.ravel())
    return {"components": count, "largest_pixels": int(sizes[1:].max(initial=0))}


def render_views(directory, tile, raw_max, old_layers, layers, out):
    """Fixed display ranges prevent categorical colors changing with absent classes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    tid = tile["tile_id"]
    transform = Affine(*tile["affine"])
    extent = (transform.c, transform.c + 51.2, transform.f - 51.2, transform.f)
    cal = tile["normalization"]
    raw_normalized = np.where(
        raw_max >= 0,
        np.clip((raw_max - cal["low"]) / max(cal["high"] - cal["low"], 1024), 0, 1),
        -9999,
    )
    difference = np.where(
        layers["observed"] > 0,
        np.where(old_layers["observed"] > 0, 1, 2),
        np.where(raw_max >= 0, 3, 0),
    )
    views = [
        (raw_normalized, "A Original observations (fixed file calibration)", "gray", 0, 1),
        (old_layers["normalized"], "B Phase 3 intensity; cyan=old mask boundary", "gray", 0, 1),
        (raw_normalized, "C Cyan=proposed reference alternatives", "gray", 0, 1),
        (raw_normalized, "D Orange=sampled perpendicular sections", "gray", 0, 1),
        (
            layers["strict_surface_candidate"] + 2 * layers["uncertain_road"],
            "E Green=strict candidate; amber=uncertain search",
            ListedColormap(["black", "green", "orange"]),
            0,
            2,
        ),
        (layers["normalized"], "F Measured search intensity (same calibration)", "gray", 0, 1),
        (
            difference,
            "G Grey=retained; green=recovered; red=still excluded",
            ListedColormap(["black", "gray", "lime", "red"]),
            0,
            3,
        ),
        (
            np.log1p(layers["count"]),
            "Original observations: log(1+count)",
            "viridis",
            0,
            float(np.log1p(layers["count"]).max()),
        ),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(23, 12), sharex=True, sharey=True)
    for ax, (arr, title, cmap, lo, hi) in zip(axes.ravel(), views, strict=True):
        ax.imshow(
            np.ma.masked_less(arr, 0),
            extent=extent,
            origin="upper",
            cmap=cmap,
            vmin=lo,
            vmax=max(hi, 1e-8),
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
    old_mask = read_layer(out / "road_masks" / f"{tid}_old.tif") > 0
    axes[0, 1].contour(
        old_mask, levels=[0.5], extent=extent, origin="upper", colors="cyan", linewidths=0.5
    )
    for feature in read_json(out / "reference_alignments" / f"{tid}.geojson")["features"]:
        xy = np.asarray(feature["geometry"]["coordinates"])
        axes[0, 2].plot(xy[:, 0], xy[:, 1], "c-", linewidth=0.65)
    sections = read_json(out / "cross_sections" / f"{tid}.json")
    for section in sections[:: max(1, len(sections) // 16)]:
        xy = np.asarray(section["center"]) + np.array([-30, 30])[:, None] * np.asarray(
            section["normal"]
        )
        axes[0, 3].plot(xy[:, 0], xy[:, 1], color="orange", linewidth=0.65)
    for ax in axes.ravel():
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
    fig.suptitle(
        tid
        + " | Native XY; PROPOSED_UNREVIEWED. White in intensity views / black in G = no measurements."
    )
    fig.tight_layout()
    fig.savefig(directory / "figures" / f"{tid}_views.png", dpi=110)
    plt.close(fig)


def qgis_bundle(report, root, directory):
    out = Path(report["output_root"])
    manifest = read_json(out / "metadata/tile_manifest.json")
    cache_root = root / "data/processed" / NS / "point_cache"
    evidence_dir = directory / "source_point_evidence"
    evidence_dir.mkdir(exist_ok=True)
    summaries = {}
    for source, stats in report["source_streams"].items():
        rows, vectors = [], []
        # Exact cached identity includes both the source and selected halo windows.
        matching = [
            p.parent
            for p in cache_root.glob("*/completion.json")
            if read_json(p)["identity"] == stats["identity"]
        ]
        if len(matching) != 1:
            raise ValueError("Ambiguous or missing source point cache")
        cache = matching[0]
        package = directory / f"{source}_review.gpkg"
        if package.exists():
            raise ValueError("Review package exists; do not overwrite potential manual edits")
        for tile in [t for t in manifest if t["source"] == source]:
            tid = tile["tile_id"]
            reference = read_json(out / "reference_alignments" / f"{tid}.geojson")
            t = Affine(*tile["affine"])
            bounds = [t.c - 30, t.f - 51.2 - 30, t.c + 51.2 + 30, t.f + 30]
            points = load_window(cache, bounds, stats["identity"]["partition_m"])
            tree = cKDTree(points["xyz"][:, :2])
            ids, names, distances = [], [], []
            for feature in reference["features"]:
                props, geometry = feature["properties"], shape(feature["geometry"])
                rows.append(
                    {
                        "tile_id": tid,
                        "reference_id": props["id"],
                        "method": props["method"],
                        "supported": props["reference_supported"],
                        "review_status": "PROPOSED_UNREVIEWED",
                        "uncertainty": json.dumps(props.get("uncertainty", [])),
                        "geometry": geometry,
                    }
                )
                probes = np.array(
                    [geometry.interpolate(d).coords[0] for d in np.arange(0, geometry.length, 0.5)]
                )
                distance, index = tree.query(probes, k=1)
                good = distance <= 0.4
                ids.extend(index[good].tolist())
                names.extend([props["id"]] * int(good.sum()))
                distances.extend(distance[good].tolist())
            selected = points[np.asarray(ids, int)]
            evidence_path = evidence_dir / f"{tid}.npz"
            np.savez_compressed(
                evidence_path,
                source_index=selected["source_index"],
                xyz=selected["xyz"],
                raw_intensity=selected["intensity"],
                reference_id=np.asarray(names),
                xy_distance=np.asarray(distances),
            )
            vectors.append(
                {
                    "tile_id": tid,
                    "geometry": box(t.c, t.f - 51.2, t.c + 51.2, t.f),
                    "review_status": "PROPOSED_UNREVIEWED",
                    "purpose": "review_extent",
                }
            )
        if rows:
            gpd.GeoDataFrame(rows, crs=None).to_file(
                package, layer="reference_candidates", driver="GPKG"
            )
        gpd.GeoDataFrame(vectors, crs=None).to_file(package, layer="review_windows", driver="GPKG")
        shared = read_json(out / "corridor_boundaries" / f"{source}_shared.geojson")
        for feature in shared["features"]:
            geometry = shape(feature["geometry"])
            if not geometry.is_empty:
                gpd.GeoDataFrame(
                    [{"review_status": "PROPOSED_UNREVIEWED", "geometry": geometry}], crs=None
                ).to_file(package, layer=feature["properties"]["kind"], driver="GPKG")
        # Empty manual edit layers do not pretend to be human-reviewed samples.
        for name, kind in (("paint_review", "LineString"), ("road_boundary_review", "Polygon")):
            gpd.GeoDataFrame(
                {
                    "review_status": [],
                    "reviewer": [],
                    "reviewed_at": [],
                    "kind": [],
                    "geometry": [],
                },
                geometry="geometry",
                crs=None,
            ).to_file(package, layer=name, driver="GPKG", geometry_type=kind)
        summaries[source] = {
            "review_package": str(package),
            "reference_features": len(rows),
            "source_sha256": stats["identity"]["source_sha256"],
            "evidence": "nearest original source records within 0.4 native units of reference probes; proximity is not verified paint",
        }
    return summaries


def section_sensitivity(report, root):
    """Investigate spacing and search extent on one original-point region per source."""
    out = Path(report["output_root"])
    cache_root = root / "data/processed" / NS / "point_cache"
    targets = {
        "Job_20260325_0838_Track01_3": "Job_20260325_0838_Track01_3__r0023808_c0026880",
        "Job_20260325_0838_Track01_8": "Job_20260325_0838_Track01_8__r0036096_c0021504",
    }
    result = {}
    manifest = {t["tile_id"]: t for t in read_json(out / "metadata/tile_manifest.json")}
    for source, tid in targets.items():
        if tid not in manifest:
            continue
        stats = report["source_streams"][source]
        cache = next(
            p.parent
            for p in cache_root.glob("*/completion.json")
            if read_json(p)["identity"] == stats["identity"]
        )
        transform = Affine(*manifest[tid]["affine"])
        core = box(transform.c, transform.f - 51.2, transform.c + 51.2, transform.f)
        features = read_json(out / "reference_alignments" / f"{tid}.geojson")["features"]
        supported = [
            f
            for f in features
            if f["properties"]["reference_supported"] and shape(f["geometry"]).intersects(core)
        ]
        if not supported:
            result[source] = {"status": "NO_SUPPORTED_REFERENCE_REQUIRES_REVIEW"}
            continue
        chosen = max(supported, key=lambda f: shape(f["geometry"]).length)
        reference = dict(chosen["properties"], coords=chosen["geometry"]["coordinates"])
        points = load_window(
            cache,
            [transform.c - 30, transform.f - 81.2, transform.c + 81.2, transform.f + 30],
            stats["identity"]["partition_m"],
        )
        cfg = SectionSettings(**report["identity"]["settings"]["sections"])
        cases = []
        for spacing, width in ((0.5, 30), (1, 30), (2, 30), (1, 20), (1, 40)):
            setting = replace(cfg, spacing_m=spacing, half_width_m=width)
            profiles = extract_sections(points["xyz"], [reference], setting)
            boundary = build_boundaries(profiles, setting)
            widths = [d["width_m"] for d in boundary["diagnostics"] if d["width_m"] is not None]
            noises = [v for p in profiles for v in p["noise"] if v is not None]
            cases.append(
                {
                    "spacing": spacing,
                    "search_half_width": width,
                    "stations": len(profiles),
                    "supported_station_width_percentiles": (
                        np.percentile(widths, [10, 50, 90]).tolist() if widths else []
                    ),
                    "profile_MAD_percentiles": (
                        np.percentile(noises, [10, 50, 90]).tolist() if noises else []
                    ),
                    "alternative_segments": sum(d["alternatives"] for d in boundary["diagnostics"]),
                    "queried_point_window_halo": 30,
                    "note": "40-unit search can reach the bounded cache-window limit; no extrapolated points",
                }
            )
        result[source] = {
            "tile_id": tid,
            "reference_id": reference["id"],
            "review_status": "PROPOSED_UNREVIEWED",
            "cases": cases,
        }
    return result


def audit(root):
    directory = root / "outputs/reports" / NS
    latest = read_json(directory / "latest.json")
    report = read_json(latest["report"])
    out, reports = Path(report["output_root"]), Path(report["report_directory"])
    snapshot = reports / "code"
    code_paths = [
        snapshot / "corridor_pilot.py",
        *[snapshot / p.name for p in sorted((root / "src/roadmarking/preprocessing").glob("*.py"))],
        snapshot / "marking_retention.py",
        snapshot / "rasterizer.py",
    ]
    snapshot_verified = (
        all(p.exists() for p in code_paths)
        and hashlib.sha256("".join(sha256(p) for p in code_paths).encode()).hexdigest()
        == report["identity"]["code_hash"]
    )
    manifest = read_json(out / "metadata/tile_manifest.json")
    original = {
        t["tile_id"]: t
        for t in read_json(root / "data/processed/phase3/metadata/tile_manifest.json")
    }
    rows = []
    for tile in manifest:
        tid = tile["tile_id"]
        layers = {k: read_layer(v) for k, v in tile["layers"].items()}
        old = original[tid]
        old_mask = read_layer(old["layers"]["road_state"]) > 0
        strict = layers["strict_surface_candidate"] > 0
        raw = read_layer(out / "rasters" / tid / "original_maximum.tif")
        old_normalized = read_layer(old["layers"]["normalized"])
        old_observed = read_layer(old["layers"]["observed"]) > 0
        calibration = tile["normalization"]
        normalized_raw = np.clip(
            (raw - calibration["low"]) / max(calibration["high"] - calibration["low"], 1024), 0, 1
        )
        bright_original = (raw >= 0) & (normalized_raw >= 0.5)
        bright_old = old_normalized >= 0.5
        bright_new = layers["normalized"] >= 0.5
        rows.append(
            {
                "tile_id": tid,
                "grid_affine_unchanged": tile["affine"] == old["affine"],
                "strict_gained_area": float((strict & ~old_mask).sum() * 0.05**2),
                "strict_removed_area": float((~strict & old_mask).sum() * 0.05**2),
                "old_nodata_pct_within_old_mask": 100
                * int((old_mask & ~old_observed).sum())
                / max(int(old_mask.sum()), 1),
                "new_nodata_pct_within_search": 100
                * int((layers["marking_search"].astype(bool) & (layers["observed"] == 0)).sum())
                / max(int(layers["marking_search"].sum()), 1),
                "nodata_comparison_note": "Different spatial denominators; broad search NoData is not road-surface missing rate",
                "fixed_calibration_bright_original": int(bright_original.sum()),
                "fixed_calibration_bright_old": int(bright_old.sum()),
                "fixed_calibration_bright_new": int(bright_new.sum()),
                "bright_graph_original": bright_components(bright_original),
                "bright_graph_old": bright_components(bright_old),
                "bright_graph_new": bright_components(bright_new),
                "original_count_reconstructed": int(layers["count"].sum()),
                "old_selected_points": int(read_layer(old["layers"]["count"]).sum()),
                "nodata_observed_consistent": bool(
                    np.array_equal(layers["observed"] > 0, layers["maximum"] >= 0)
                ),
                "float_nodata_consistent": bool(
                    np.all(layers["normalized"][layers["count"] == 0] == -9999)
                ),
            }
        )
        old_layers = {
            "observed": read_layer(old["layers"]["observed"]),
            "normalized": old_normalized,
        }
        render_views(reports, tile, raw, old_layers, layers, out)
    bundles = qgis_bundle(report, root, reports)
    sensitivity = section_sensitivity(report, root)
    xml = ET.parse(directory / "pytest.xml").getroot()
    suites = list(xml.iter("testsuite"))
    tests = {
        k: sum(int(s.attrib.get(k, 0)) for s in suites)
        for k in ("tests", "failures", "errors", "skipped")
    }
    tests["passed"] = tests["tests"] - tests["failures"] - tests["errors"] - tests["skipped"]
    health = read_json(directory / "healthcheck.json")
    validation = {
        "pytest": tests,
        "healthcheck": {s: sum(v["status"] == s for v in health) for s in ("pass", "skip", "fail")},
        "ruff": (directory / "ruff.log").read_text().strip(),
        "black": (directory / "black.log").read_text().strip(),
    }
    dump(directory / "validation.json", validation)
    audit_result = {
        "run_id": report["run_id"],
        "status": "PROVISIONAL",
        "tiles": rows,
        "review_bundles": bundles,
        "section_sensitivity": sensitivity,
        "executed_pilot_code_snapshot_verified": snapshot_verified,
        "validation": validation,
        "note": "Bright-component counts are pixel graph diagnostics, not semantic paint continuity or verified marking recall.",
        "protected_outputs_unchanged": all(
            sha256(p) == digest for p, digest in report["protected_artifact_hashes"].items()
        ),
    }
    dump(reports / "quality_audit.json", audit_result)
    audit_text = [
        "\n## Persisted-output audit\n",
        f"Validation: {json.dumps(validation)}\n",
        f"Additional QA: [quality_audit.json](quality_audit.json). Grid affines unchanged: {all(r['grid_affine_unchanged'] for r in rows)}; explicit NoData consistency: {all(r['float_nodata_consistent'] and r['nodata_observed_consistent'] for r in rows)}. Bright-feature component and gained/removed strict-area metrics are in the audit JSON. No component is asserted to be verified paint.\n",
        "Native-XY QGIS review packages (all candidate layers PROPOSED_UNREVIEWED; manual edit layers initially empty):\n",
    ]
    audit_text += [
        f"- [{source} review package]({Path(v['review_package']).name})"
        for source, v in bundles.items()
    ]
    audit_text += [
        "\nSource-indexed original XYZ/raw-intensity evidence is in source_point_evidence/. Proximity samples support trace review but are not manual marking truth.\n"
    ]
    with (reports / "report.md").open("a", encoding="utf-8") as f:
        f.write("\n".join(audit_text))
    print(
        json.dumps(
            {
                "status": "PROVISIONAL",
                "tiles_audited": len(rows),
                "validation": validation,
                "report": str(reports / "report.md"),
            }
        )
    )


if __name__ == "__main__":
    audit(Path(__file__).resolve().parents[1])
