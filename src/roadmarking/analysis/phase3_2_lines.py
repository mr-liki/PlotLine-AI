"""Controlled Phase 3.2 provenance, real-point evidence and line-continuity pilot."""

# ruff: noqa: E501 -- report prose is kept in complete Markdown lines.

import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import yaml
from affine import Affine
from rasterio.features import rasterize
from scipy.spatial import cKDTree
from shapely.geometry import LineString, box, mapping
from shapely.ops import unary_union

from roadmarking.analysis.corridor_pilot import accumulate, load_window, read_json
from roadmarking.preprocessing.intensity import IntensityCalibration
from roadmarking.preprocessing.stripe_reference import detect_references
from roadmarking.preprocessing.stripe_tracking import (
    LineEvidence,
    TrackingSettings,
    clip_candidates,
    densify,
    line_quality,
    stitch_candidates,
    trace_stripes,
)
from roadmarking.rasterization.pipeline import MemoryMonitor, dump, sha256, write_tif
from roadmarking.rasterization.rasterizer import world_to_pixel

GROUPS = {
    "3_straight_overlap": [
        "Job_20260325_0838_Track01_3__r0014592_c0014592",
        "Job_20260325_0838_Track01_3__r0014592_c0015360",
    ],
    "3_boundary_overlap": [
        "Job_20260325_0838_Track01_3__r0023808_c0026880",
        "Job_20260325_0838_Track01_3__r0024576_c0027648",
    ],
    "3_ambiguous_blob": ["Job_20260325_0838_Track01_3__r0017664_c0017664"],
    "8_overlap": [
        "Job_20260325_0838_Track01_8__r0028416_c0016896",
        "Job_20260325_0838_Track01_8__r0028416_c0017664",
    ],
    "8_curve": ["Job_20260325_0838_Track01_8__r0029952_c0018432"],
}


def geometry_parts(geometry):
    if geometry.geom_type == "LineString":
        return [geometry] if geometry.length > 0 else []
    return [g for part in getattr(geometry, "geoms", []) for g in geometry_parts(part)]


def _frame(rows, geometry_type="LineString"):
    if rows:
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=None)
    return gpd.GeoDataFrame(
        {"id": [], "role": [], "review_status": [], "geometry": []}, geometry="geometry", crs=None
    )


def _line_row(candidate, **extra):
    return {
        "id": candidate["id"],
        "role": candidate["role"],
        "review_status": "PROPOSED_UNREVIEWED",
        "candidate_class": candidate["candidate_class"],
        "width": candidate["width"],
        "flags": json.dumps(candidate["flags"]),
        "parent_id": candidate.get("parent_id"),
        "dash_group": candidate.get("dash_group_id"),
        "generator": "stripe_tracking.trace_stripes / stitch_candidates",
        **candidate["quality"],
        **extra,
        "geometry": LineString(candidate["coords"]),
    }


def _source_evidence(points, raw, evidence, candidate, path):
    cols, rows = world_to_pixel(evidence.transform, points["xyz"][:, 0], points["xyz"][:, 1])
    inside = (
        (cols >= 0)
        & (rows >= 0)
        & (cols < raw["maximum"].shape[1])
        & (rows < raw["maximum"].shape[0])
    )
    indices = np.flatnonzero(inside)
    indices = indices[
        evidence.paint[rows[indices], cols[indices]]
        & (points["intensity"][indices] == raw["maximum"][rows[indices], cols[indices]])
    ]
    if len(indices):
        tree = cKDTree(points["xyz"][indices, :2])
        probes, _ = densify(candidate["coords"], evidence.transform.a / 2)
        distance, near = tree.query(probes, k=min(4, len(indices)))
        chosen = np.unique(
            indices[
                np.asarray(near)[
                    np.asarray(distance) <= max(candidate["width"] / 2, evidence.support_radius)
                ]
            ]
        )
    else:
        chosen = np.empty(0, int)
    np.savez_compressed(
        path,
        xyz=points["xyz"][chosen],
        raw_intensity=points["intensity"][chosen],
        source_index=points["source_index"][chosen],
    )
    return len(chosen)


def _styles(directory):
    style = {
        "candidate_reference_alignments": ("0,180,230,255", "dash", 0.35),
        "candidate_painted_stripes": ("255,170,0,255", "solid", 0.5),
        "unsupported_line_qa": ("230,30,30,255", "dot", 0.35),
        "legacy_context_references": ("150,90,40,255", "dash", 0.25),
        "verified_road_markings": ("0,150,60,255", "solid", 0.7),
    }
    for name, (color, line_style, width) in style.items():
        root = ET.Element("qgis", version="3.34", styleCategories="Symbology")
        renderer = ET.SubElement(root, "renderer-v2", type="singleSymbol", symbollevels="0")
        symbols = ET.SubElement(renderer, "symbols")
        symbol = ET.SubElement(symbols, "symbol", type="line", name="0", alpha="1")
        layer = ET.SubElement(
            symbol, "layer", {"class": "SimpleLine", "enabled": "1", "locked": "0"}
        )
        for key, value in (
            ("line_color", color),
            ("line_style", line_style),
            ("line_width", str(width)),
            ("line_width_unit", "MM"),
        ):
            ET.SubElement(layer, "prop", k=key, v=value)
        ET.ElementTree(root).write(
            directory / f"{name}.qml", encoding="utf-8", xml_declaration=True
        )


def figure(path, raw, old, corrected, qa, evidence, scope, strict, search, title, suspect=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = evidence.transform
    h, w = raw["maximum"].shape
    extent = [t.c, t.c + w * t.a, t.f + h * t.e, t.f]
    fig, axes = plt.subplots(2, 4, figsize=(23, 12), sharex=True, sharey=True)
    views = [
        (raw["normalized"], "Original observed intensity", "gray"),
        (raw["observed"], "Observed mask (NoData=black)", "gray"),
        (np.log1p(raw["count"]), "Original source point coverage", "viridis"),
        (raw["normalized"], "Search=blue / strict=green outlines", "gray"),
        (raw["normalized"], "Old finite PCA reference proposals (brown)", "gray"),
        (raw["normalized"], "Corrected candidate painted stripes (orange)", "gray"),
        (raw["normalized"], "Unsupported / ambiguous proposals (red)", "gray"),
        (evidence.contrast, "Local contrast; no semantic ground truth", "magma"),
    ]
    for ax, (arr, label, cmap) in zip(axes.ravel(), views, strict=True):
        limits = {"vmin": 0, "vmax": 1} if cmap == "gray" else {}
        ax.imshow(
            np.ma.masked_less(arr, 0),
            extent=extent,
            origin="upper",
            cmap=cmap,
            interpolation="nearest",
            **limits,
        )
        ax.set_title(label, fontsize=10)
        ax.tick_params(labelsize=7)

    def outline(ax, geometry, color):
        for ring in getattr(geometry, "geoms", [geometry]):
            if ring.geom_type == "Polygon":
                xy = np.asarray(ring.exterior.coords)
                ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=0.7)

    outline(axes[0, 3], search, "blue")
    outline(axes[0, 3], strict, "green")
    for ax in axes.ravel():
        outline(ax, scope, "cyan")
    for proposal in old:
        xy = np.asarray(proposal["coords"])
        axes[1, 0].plot(xy[:, 0], xy[:, 1], color="#96602e", linewidth=0.8, alpha=0.8)
    for ax, proposals, color in ((axes[1, 1], corrected, "orange"), (axes[1, 2], qa, "red")):
        for proposal in proposals:
            xy = np.asarray(proposal["coords"])
            ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=0.9)
    for ax in axes.ravel():
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    if suspect is not None:
        line = np.asarray(suspect["coords"])
        for ax in axes.ravel():
            ax.plot(line[:, 0], line[:, 1], color="magenta", linewidth=1.6)
        bounds = LineString(line).buffer(3).bounds
        for ax in axes.ravel():
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])
    fig.suptitle(
        title
        + " | Cyan=displayed tile footprints, NOT verified road; native XY; all lines provisional"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=115)
    plt.close(fig)


def run(root, previous, inventory, suspects, selected_groups=None):
    settings = yaml.safe_load((root / "configs/phase3_2.yaml").read_text())
    cfg = TrackingSettings(**settings["tracking"])
    old_root = Path(previous["output_root"])
    old_manifest = {t["tile_id"]: t for t in read_json(old_root / "metadata/tile_manifest.json")}
    resolution = settings["diagnostic_resolution"]
    identity = {
        "previous_run": previous["run_id"],
        "groups": selected_groups or GROUPS,
        "settings": settings,
        "tracking_code": sha256(root / "src/roadmarking/preprocessing/stripe_tracking.py"),
        "runner_code": sha256(Path(__file__)),
    }
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    out = root / "data/processed/phase3_2" / run_id
    reports = root / "outputs/reports/phase3_2"
    if out.exists():
        raise ValueError("Run directory exists; preserve it and use a changed identity")
    for name in ("rasters", "vectors", "source_evidence", "metadata", "styles"):
        (out / name).mkdir(parents=True)
    (reports / "figures").mkdir(exist_ok=True)
    _styles(out / "styles")
    protected_paths = [
        root / "data/processed/phase3/metadata/tile_manifest.json",
        Path(previous["report_directory"]) / "report.json",
        old_root / "metadata/tile_manifest.json",
        *Path(previous["report_directory"]).glob("*review.gpkg"),
    ]
    source_paths = {
        source: root / "data/raw/pointclouds" / f"{source}.laz"
        for source in previous["source_streams"]
    }
    protected = {str(p): sha256(p) for p in [*protected_paths, *source_paths.values()]}
    results, feature_diagnostics, manifest = [], [], []
    start = time.perf_counter()
    with MemoryMonitor() as monitor:
        for group, tids in (selected_groups or GROUPS).items():
            began = time.perf_counter()
            tiles = [old_manifest[tid] for tid in tids]
            source = tiles[0]["source"]
            stats = previous["source_streams"][source]
            if sha256(source_paths[source]) != stats["identity"]["source_sha256"]:
                raise ValueError("Intact source identity changed")
            cache = next(
                p.parent
                for p in (root / "data/processed/phase3_1_topodot_inspired/point_cache").glob(
                    "*/completion.json"
                )
                if read_json(p)["identity"] == stats["identity"]
            )
            tile_boxes = [
                box(t["affine"][2], t["affine"][5] - 51.2, t["affine"][2] + 51.2, t["affine"][5])
                for t in tiles
            ]
            scope = unary_union(tile_boxes)
            x0, y0, x1, y1 = scope.bounds
            halo = settings["halo"]
            # Existing source grid: resolution is an integer multiple of 0.05.
            transform = Affine(resolution, 0, x0 - halo, 0, -resolution, y1 + halo)
            shape_ = (
                round((y1 - y0 + 2 * halo) / resolution),
                round((x1 - x0 + 2 * halo) / resolution),
            )
            bounds = [
                transform.c,
                transform.f + shape_[0] * transform.e,
                transform.c + shape_[1] * transform.a,
                transform.f,
            ]
            # Verify each shared cache partition that can be read for this window.
            lower = np.floor(np.asarray(bounds[:2]) / stats["identity"]["partition_m"]).astype(int)
            upper = np.floor(np.asarray(bounds[2:]) / stats["identity"]["partition_m"]).astype(int)
            for xx in range(lower[0], upper[0] + 1):
                for yy in range(lower[1], upper[1] + 1):
                    name = f"p_{xx}_{yy}.bin"
                    if (
                        name in stats["artifacts"]
                        and sha256(cache / name) != stats["artifacts"][name]
                    ):
                        raise ValueError("Original-point cache corruption")
            points = load_window(cache, bounds, stats["identity"]["partition_m"])
            calibration = IntensityCalibration(**tiles[0]["normalization"])
            raw = accumulate(points, transform, shape_, calibration)
            evidence = LineEvidence(raw["normalized"], raw["observed"] > 0, transform)
            search = unary_union(
                [
                    g
                    for f in read_json(
                        old_root / "corridor_boundaries" / f"{source}_shared.geojson"
                    )["features"]
                    if f["properties"]["kind"] == "marking_search"
                    for g in [__import__("shapely").geometry.shape(f["geometry"])]
                ]
            )
            strict = unary_union(
                [
                    __import__("shapely").geometry.shape(f["geometry"])
                    for f in read_json(
                        old_root / "corridor_boundaries" / f"{source}_shared.geojson"
                    )["features"]
                    if f["properties"]["kind"] == "strict_surface_candidate"
                ]
            )
            t = time.perf_counter()
            legacy, _e_old = detect_references(raw["normalized"], raw["observed"] > 0, transform)
            old_seconds = time.perf_counter() - t
            t = time.perf_counter()
            candidates, qa, detector = trace_stripes(evidence, cfg)
            corrected, stitch = stitch_candidates(candidates, evidence, cfg)
            corrected = clip_candidates(corrected, scope, evidence, cfg)
            new_seconds = time.perf_counter() - t
            package = out / "vectors" / f"{group}_review.gpkg"
            old_rows = []
            for proposal in legacy:
                old_rows.append(
                    {
                        "id": proposal["id"],
                        "role": "LEGACY_PROVISIONAL_ROAD_REFERENCE",
                        "review_status": "PROPOSED_UNREVIEWED",
                        "generator": "stripe_reference.detect_references (same raster replay)",
                        **line_quality(proposal["coords"], evidence, search),
                        "geometry": LineString(proposal["coords"]),
                    }
                )
            _frame(old_rows).to_file(package, layer="legacy_context_references", driver="GPKG")
            candidate_rows, reference_rows = [], []
            for candidate in corrected:
                number = _source_evidence(
                    points,
                    raw,
                    evidence,
                    candidate,
                    out / "source_evidence" / f"{group}_{candidate['id']}.npz",
                )
                candidate["quality"]["number_source_observations"] = number
                candidate["quality"]["source_observations_per_unit"] = number / max(
                    candidate["quality"]["total_length"], 1e-8
                )
                candidate["quality"].update(line_quality(candidate["coords"], evidence, search))
                if not number:
                    candidate["flags"].append("NO_ASSOCIATED_ORIGINAL_BRIGHT_MAXIMUM")
                row = _line_row(candidate, source=source, group=group)
                candidate_rows.append(row)
                if candidate["quality"]["total_length"] >= 8 and not candidate["flags"]:
                    reference_rows.append(
                        dict(row, role="PROVISIONAL_CORRIDOR_REFERENCE_NOT_MARKING")
                    )
            _frame(candidate_rows).to_file(
                package, layer="candidate_painted_stripes", driver="GPKG"
            )
            _frame(reference_rows).to_file(
                package, layer="candidate_reference_alignments", driver="GPKG"
            )
            qa_rows = [_line_row(c, source=source, group=group) for c in qa]
            # Preserve every original selected suspect feature with its actual FID.
            previous_frame = gpd.read_file(
                Path(previous["report_directory"]) / f"{source}_review.gpkg",
                layer="reference_candidates",
                fid_as_index=True,
            )
            selected = previous_frame[previous_frame.tile_id.isin(tids)]
            for fid, row in selected.iterrows():
                q = line_quality(np.asarray(row.geometry.coords), evidence, search)
                diagnostic = {
                    "source": source,
                    "fid": int(fid),
                    "tile_id": row.tile_id,
                    "reference_id": row.reference_id,
                    "group": group,
                    "generator": "legacy _segment / observed-only endpoint joiner / untrimmed halo export",
                    "outside_displayed_scope_length": row.geometry.difference(scope).length,
                    **q,
                    "interpretation": (
                        "PAINT_LIKE_SUPPORT_ONLY_NOT_VERIFIED_PAINT"
                        if q["supported_fraction"] >= cfg.minimum_supported_fraction
                        else "PROVISIONAL_REFERENCE_WITH_UNSUPPORTED_INTERVALS"
                    ),
                    "screenshot_identification": "not asserted; screenshot not supplied",
                }
                feature_diagnostics.append(diagnostic)
                qa_rows.append(
                    {
                        "id": f"legacy_fid_{fid}",
                        "role": "LEGACY_REFERENCE_QA_NOT_FINAL_MARKING",
                        "review_status": "PROPOSED_UNREVIEWED",
                        "source_fid": int(fid),
                        "tile_id": row.tile_id,
                        "reference_id": row.reference_id,
                        "flags": diagnostic["interpretation"],
                        **q,
                        "geometry": row.geometry,
                    }
                )
            _frame(qa_rows).to_file(package, layer="unsupported_line_qa", driver="GPKG")
            _frame([]).to_file(
                package, layer="verified_road_markings", driver="GPKG", geometry_type="LineString"
            )
            gpd.GeoDataFrame(
                [
                    {
                        "role": "DISPLAY_SCOPE_NOT_ROAD",
                        "review_status": "PROPOSED_UNREVIEWED",
                        "geometry": scope,
                    }
                ],
                crs=None,
            ).to_file(package, layer="displayed_tile_footprints", driver="GPKG")
            for layer, geom in (
                ("strict_surface_candidate", strict.intersection(box(*bounds))),
                ("wide_marking_search", search.intersection(box(*bounds))),
            ):
                gpd.GeoDataFrame(
                    [
                        {
                            "role": "NOT_PAINT",
                            "review_status": "PROPOSED_UNREVIEWED",
                            "geometry": geom,
                        }
                    ],
                    crs=None,
                ).to_file(package, layer=layer, driver="GPKG")
            raster_paths = {}
            for name in ("maximum", "normalized", "observed", "count"):
                path = out / "rasters" / group / f"{name}.tif"
                write_tif(path, raw[name], transform, None)
                raster_paths[name] = str(path)
                with rasterio.open(path) as ds:
                    if (
                        ds.crs is not None
                        or ds.transform != transform
                        or not np.array_equal(ds.read(1), raw[name])
                    ):
                        raise ValueError("Raster/source-grid readback failed")
            for name, mask in (
                ("strict_surface_candidate", strict),
                ("wide_marking_search", search),
            ):
                image = rasterize(
                    [(mapping(mask), 1)], out_shape=shape_, transform=transform, dtype="uint8"
                )
                path = out / "rasters" / group / f"{name}.tif"
                write_tif(path, image, transform, None)
                raster_paths[name] = str(path)
            manifest.append(
                {
                    "group": group,
                    "source": source,
                    "source_sha256": stats["identity"]["source_sha256"],
                    "tile_ids": tids,
                    "affine": list(transform)[:6],
                    "resolution": resolution,
                    "raster_layers": raster_paths,
                    "vector_package": str(package),
                    "review_status": "PROPOSED_UNREVIEWED",
                    "verified_features": 0,
                }
            )
            old_scope = []
            for proposal in legacy:
                old_scope += [
                    line_quality(np.asarray(part.coords), evidence)
                    for part in geometry_parts(LineString(proposal["coords"]).intersection(scope))
                ]
            old_unsupported = sum(q["unsupported_length"] for q in old_scope)
            new_unsupported = sum(c["quality"]["unsupported_length"] for c in corrected)
            # Canonical world-coordinate candidates are clipped for each tile,
            # preserving their parent IDs. No endpoint-distance seam snapping.
            memberships = []
            cross = 0
            for candidate in corrected:
                geometry = LineString(candidate["coords"])
                members = [
                    tid
                    for tid, footprint in zip(tids, tile_boxes, strict=True)
                    if geometry.intersection(footprint).length > resolution
                ]
                shared_length = (
                    geometry.intersection(tile_boxes[0].intersection(tile_boxes[-1])).length
                    if len(tile_boxes) > 1
                    else 0.0
                )
                if len(members) > 1 and shared_length > resolution:
                    candidate["quality"]["cross_tile_joins"] = len(members) - 1
                    cross += 1
                memberships.append(
                    {
                        "candidate_id": candidate["id"],
                        "tile_ids": members,
                        "shared_overlap_supported_length": shared_length,
                        "matching": "canonical measured-support graph in shared world coordinates; parent ID preserved",
                    }
                )
            for tid, footprint in zip(tids, tile_boxes, strict=True):
                clipped = clip_candidates(corrected, footprint, evidence, cfg)
                dump(out / "metadata" / f"{tid}_stripe_fragments.json", clipped)
            result = {
                "group": group,
                "source": source,
                "points_in_bounded_cache_window": len(points),
                "legacy_same_input_replay_references": len(legacy),
                "saved_legacy_features_investigated": len(selected),
                "corrected_candidates": len(corrected),
                "new_qa_fragments": len(qa),
                "legacy_unsupported_length_in_scope": old_unsupported,
                "corrected_unsupported_length_in_scope": new_unsupported,
                "old_scope_length": sum(q["total_length"] for q in old_scope),
                "corrected_scope_length": sum(c["quality"]["total_length"] for c in corrected),
                "verified_accuracy": None,
                "canonical_candidates_crossing_tile_overlaps": cross,
                "continuity": memberships,
                "stitching": stitch,
                "detector": detector,
                "old_detection_seconds": old_seconds,
                "corrected_detection_seconds": new_seconds,
                "total_group_seconds": time.perf_counter() - began,
                "scope_note": "same raw point set / same 0.1-unit raster for old and new; saved historical proposals originated on a different 0.2-unit discovery raster",
            }
            results.append(result)
            dump(out / "metadata" / f"{group}_candidates.json", corrected)
            dump(reports / f"{group}.json", result)
            figure(
                reports / "figures" / f"{group}_comparison.png",
                raw,
                legacy,
                corrected,
                qa,
                evidence,
                scope,
                strict,
                search,
                group,
            )
            # Reproducible examples: long unsupported, short, paint-supported,
            # outside tile context and overlap-crossing can refer to one feature.
            local = [d for d in feature_diagnostics if d["group"] == group]
            selectors = {
                "long_unsupported": lambda d: d["unsupported_length"],
                "short_fragment": lambda d: -d["total_length"],
                "plausible_reference": lambda d: d["supported_length"],
                "outside_footprint": lambda d: d["outside_displayed_scope_length"],
            }
            for category, key in selectors.items():
                if not local:
                    continue
                diagnostic = max(local, key=key)
                row = selected.loc[diagnostic["fid"]]
                figure(
                    reports / "figures" / f"{group}_{category}_fid{diagnostic['fid']}.png",
                    raw,
                    legacy,
                    corrected,
                    qa,
                    evidence,
                    scope,
                    strict,
                    search,
                    f"{group} {category}: source FID {diagnostic['fid']}",
                    suspect={"coords": np.asarray(row.geometry.coords)},
                )
            print(
                json.dumps(
                    {
                        "group": group,
                        "old": len(legacy),
                        "new": len(corrected),
                        "qa": len(qa),
                        "old_unsupported_length": old_unsupported,
                        "new_unsupported_length": new_unsupported,
                    }
                ),
                flush=True,
            )
    changed = [path for path, digest in protected.items() if sha256(path) != digest]
    if changed:
        raise ValueError(f"Protected input changed: {changed}")
    dump(out / "metadata/manifest.json", manifest)
    dump(out / "metadata/identity.json", identity)
    dump(reports / "feature_diagnostics.json", feature_diagnostics)
    report = {
        "status": "PROVISIONAL",
        "run_id": run_id,
        "output_root": str(out),
        "groups": results,
        "inventory_layers": len(inventory),
        "historical_reference_features": len(suspects),
        "historical_references_outside_own_tile": sum(
            s["outside_own_tile_length"] > 0.1 for s in suspects
        ),
        "investigated_features": len(feature_diagnostics),
        "total_seconds": time.perf_counter() - start,
        "peak_rss_bytes": monitor.peak,
        "protected_hashes": protected,
        "protected_unchanged": True,
        "verified_road_markings": 0,
        "manual_review_available": False,
        "full_survey_processed": False,
        "raw_observations_modified": False,
        "new_output_bytes": sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
        "artifact_hashes": {str(p): sha256(p) for p in out.rglob("*") if p.is_file()},
        "scope": "8 selected previous pilot tiles in 5 groups; original cached point partitions reused; no full LAZ decoding",
    }
    dump(reports / "report.json", report)
    write_report(root, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "run_id": run_id,
                "seconds": report["total_seconds"],
                "report": str(reports / "report.md"),
            }
        )
    )


def write_report(root, report):
    directory = root / "outputs/reports/phase3_2"
    text = [
        "# Phase 3.2: line geometry and continuity correction\n",
        f"**{report['status']}**. No manual ground truth; zero verified road-marking vectors. Run `{report['run_id']}`.\n",
        "## What the extra lines represent\n",
        f"The two previous packages contain {report['historical_reference_features']} candidate road-reference LineStrings and no final marking lines. {report['historical_references_outside_own_tile']} extend beyond their own tile footprint because extraction used 30-unit halos and qgis_bundle exported the complete proposals. Full inventory: [layer_inventory.md](layer_inventory.md), [layer_inventory.json](layer_inventory.json), [source feature IDs](suspect_features.json). No screenshot was supplied, so exact screenshot color/FID attribution is not asserted.\n",
        "## Root causes\n",
        "1. **Unclipped context:** finite PCA endpoints from observed halo samples were exported outside displayed raster footprints. Outside a tile is not outside observed roadway.\n2. **Wrong support test:** the old endpoint joiner accepted observed pavement across gaps; it did not require bright-stripe support. Dashed elements could consequently appear as long reference lines.\n3. **Straight local approximation:** each old PCA fragment had two vertices, losing within-fragment curvature and potentially passing away from the bright ridge. No infinite Hough/RANSAC line was found.\n4. **Duplicates and IDs:** overlapping halo detections were exported repeatedly with window-local IDs.\n5. **Layer semantics:** provisional references lacked a separate candidate-stripe/final-marking product.\n6. **Irregular polygons:** the previous observation footprint uses 1-unit cells and strict boundaries use station strips. They are not painted-line geometries and were not silently repaired in this phase.\n",
        "## Minimal correction\n",
        "The legacy detector is unchanged for replay. The opt-in stripe_tracking module reuses local contrast, then traces a thinned ridge graph, splits junctions, centers finite endpoints on original bright cells, applies evidence-constrained local smoothing and rejects/flags unsupported geometry. Joins require class/width/tangent compatibility, short physical distance, original bright support and observed coverage. Dash/short-stroke grouping adds an ID without connecting the gap. Raw images and previous outputs are unchanged.\n",
        "## Controlled old/new comparison\n",
        "Old and corrected detectors use identical original cached points and the same 0.1-unit raster. Saved historical proposals are also audited separately; their original discovery grid was 0.2. All distances are provisional native units. Quality support is proximity to local bright observations, not verified paint truth. A road reference may legitimately span an unpainted dash gap: lack of paint support does not by itself invalidate a road alignment. The old/new layers have different purposes and are not combined as final-marking accuracy statistics. Different feature counts and lengths alone are not accuracy evidence.\n",
        "| Group | Old refs | New stripe candidates | QA fragments | Old unsupported length in displayed scope | New unsupported length | Canonical overlap crossings | Old / new detector s |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for group in report["groups"]:
        text.append(
            f"| {group['group']} | {group['legacy_same_input_replay_references']} | {group['corrected_candidates']} | {group['new_qa_fragments']} | {group['legacy_unsupported_length_in_scope']:.2f} | {group['corrected_unsupported_length_in_scope']:.2f} | {group['canonical_candidates_crossing_tile_overlaps']} | {group['old_detection_seconds']:.2f} / {group['corrected_detection_seconds']:.2f} |"
        )
    text += [
        "\n## Evidence for suspect features\n",
        f"{report['investigated_features']} historical feature diagnostics: [feature_diagnostics.json](feature_diagnostics.json). Each includes source, original GeoPackage FID, tile/reference ID, generator, observed/bright support, out-of-footprint length and explicit provisional interpretation. Four reproducible examples per group (long unsupported, short, plausible, outside footprint) have separated/common-coordinate intensity, NoData, density, boundary outlines and line views under [figures](figures/). Magenta identifies the suspect feature. These are data-rendered overlays, not screenshots from the user's QGIS project.\n",
        "## Continuity and uncertainty\n",
        "Canonical candidates are constructed in shared world coordinates, then clipped into per-tile fragments with a common parent ID. Crossings count only candidates with actual shared-overlap support, not nearby endpoints. Matching uses measured ridge graph topology; optional endpoint joins additionally check feature class, width, tangent, bright support and observed coverage. Dashed groups have separate geometries. Junction branches stay separate unless the observed graph actually connects. Small interpolations are labeled; no intentional extrapolation is emitted as a stripe. Rejected/uncertain fragments and all selected old references remain in the QA/context layers.\n",
        "## QGIS review products\n",
        f"Packages and raster paths are in `{report['output_root']}/metadata/manifest.json`. Layer roles: legacy_context_references (brown, diagnostic/off by default), candidate_reference_alignments (cyan dashed), candidate_painted_stripes (orange), unsupported_line_qa (red dotted), verified_road_markings (empty), strict_surface_candidate and wide_marking_search (outline only). Raw maximum/normalized intensity, observed mask and point count are separate GeoTIFFs. QML styles are supplied; no CRS is assigned. See [documentation](../../../docs/phase3_2_line_geometry_correction.md) for layer order and reviewer workflow.\n",
        "## Performance\n",
        f"{report['total_seconds']:.1f} s, {report['peak_rss_bytes']/2**20:.1f} MiB sampled peak RSS; {report['new_output_bytes']/2**20:.1f} MiB data outputs. This is a small cached-point pilot, not comparable to full-survey preprocessing throughput. No whole-survey processing or full LAZ reread.\n",
        "## Validation and final status\n",
        "Automated validation is recorded in validation.json after the final test/check run. No client annotations were used; manual marking accuracy, recall and physical dash labels are unverified. CRS, units, vertical datum, sparse/occluded observations and unmatched annotation issues remain. All candidates require review; no training, weights download, final labeling or Git commit. **PROVISIONAL**, with full processing gated on review.\n",
    ]
    validation = directory / "validation.json"
    if validation.exists():
        text.append(f"Validation: {read_json(validation)}\n")
    (directory / "report.md").write_text("\n".join(text), encoding="utf-8")
