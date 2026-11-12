"""Trace previous QGIS line provenance, then run a bounded evidence-based pilot."""

# ruff: noqa: E501 -- provenance/report prose.

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pyogrio
from shapely.geometry import box
from shapely.ops import unary_union

from roadmarking.analysis.corridor_pilot import read_json
from roadmarking.rasterization.pipeline import dump

ROOT = Path(__file__).resolve().parents[1]


def inventory(root=ROOT):
    report_dir = root / "outputs/reports/phase3_2"
    report_dir.mkdir(parents=True, exist_ok=True)
    previous = read_json(
        read_json(root / "outputs/reports/phase3_1_topodot_inspired/latest.json")["report"]
    )
    manifest = read_json(Path(previous["output_root"]) / "metadata/tile_manifest.json")
    tile_lookup = {t["tile_id"]: t for t in manifest}
    records, suspects = [], []
    generators = {
        "reference_candidates": (
            "A candidate reference alignments",
            "stripe_reference.detect_references → _segment / endpoint joiner → audit_phase3_1_topodot_pilot.qgis_bundle",
        ),
        "review_windows": ("review extents", "corridor_pilot._extent → qgis_bundle"),
        "strict_surface_candidate": (
            "B surface candidate polygons",
            "road_cross_sections.extract_sections → corridor_boundaries.build_boundaries → qgis_bundle",
        ),
        "marking_search": (
            "B unverified search polygons",
            "corridor_pilot.observed_support_envelope + profile search → qgis_bundle",
        ),
        "uncertain_search": (
            "B uncertain polygons",
            "marking_search minus strict candidate → qgis_bundle",
        ),
        "paint_review": ("D empty manual paint edit layer", "qgis_bundle empty manual edit layer"),
        "road_boundary_review": (
            "B empty manual boundary edit layer",
            "qgis_bundle empty manual edit layer",
        ),
    }
    for package in sorted(Path(previous["report_directory"]).glob("*review.gpkg")):
        source = package.stem.replace("_review", "")
        footprints = unary_union(
            [
                box(t["affine"][2], t["affine"][5] - 51.2, t["affine"][2] + 51.2, t["affine"][5])
                for t in manifest
                if t["source"] == source
            ]
        )
        for layer, geometry_type in pyogrio.list_layers(package):
            frame = gpd.read_file(package, layer=layer, fid_as_index=True)
            category, function = generators[layer]
            record = {
                "source": source,
                "package": str(package),
                "layer": layer,
                "category": category,
                "generator": function,
                "geometry_type": geometry_type,
                "features": len(frame),
                "bounds": frame.total_bounds.tolist() if len(frame) else None,
                "crs": str(frame.crs),
                "fields": list(frame.columns),
                "source_windows": (
                    sorted(frame.tile_id.unique())
                    if "tile_id" in frame
                    else "shared selected pilot windows"
                ),
                "ids": (
                    [
                        {
                            "fid": int(fid),
                            "tile_id": row.get("tile_id"),
                            "reference_id": row.get("reference_id"),
                        }
                        for fid, row in frame.iterrows()
                    ]
                    if layer == "reference_candidates"
                    else []
                ),
                "final_output": False,
                "clipped_to_verified_region": False,
                "direct_intensity_basis": layer == "reference_candidates",
                "extrapolation_assessment": (
                    "finite PCA endpoints; joined intervals can cross non-paint observations; 30-unit halo context is not tile-clipped"
                    if layer == "reference_candidates"
                    else "not a painted line product"
                ),
            }
            records.append(record)
            if layer == "reference_candidates":
                for fid, row in frame.iterrows():
                    tile = tile_lookup[row.tile_id]
                    t = tile["affine"]
                    own = box(t[2], t[5] - 51.2, t[2] + 51.2, t[5])
                    outside = row.geometry.difference(own).length
                    suspects.append(
                        {
                            "source": source,
                            "layer": layer,
                            "fid": int(fid),
                            "tile_id": row.tile_id,
                            "reference_id": row.reference_id,
                            "length": row.geometry.length,
                            "outside_own_tile_length": outside,
                            "outside_all_displayed_tiles_length": row.geometry.difference(
                                footprints
                            ).length,
                            "generator": function,
                            "flags": ["HALO_REFERENCE_NOT_TILE_CLIPPED"] if outside > 0.1 else [],
                            "status": "PROVISIONAL_REFERENCE_NOT_MARKING",
                        }
                    )
    dump(report_dir / "layer_inventory.json", records)
    dump(report_dir / "suspect_features.json", suspects)
    text = [
        "# Phase 3.2 vector provenance (before correction)\n",
        "No screenshots were attached to this request. Attribution is based on the actual packages: reference_candidates is their only nonempty line layer. Its QGIS color depends on the user's project. No final road-marking layer exists.\n",
        "| Source | Layer | Category | Geometry | Count | Bounds | Generator |",
        "|---|---|---|---|---:|---|---|",
    ]
    for r in records:
        text.append(
            f"| {r['source']} | {r['layer']} | {r['category']} | {r['geometry_type']} | {r['features']} | {r['bounds']} | {r['generator']} |"
        )
    text += [
        "\nCross-section lines exist only as JSON/profile diagnostics and orange matplotlib overlays in Phase 3.1, not as populated GeoPackage line features. Candidate painted stripes were not exported as a distinct product. Final road markings do not exist.\n",
        "All source FIDs, tile IDs and reference IDs are in layer_inventory.json and suspect_features.json. Outside a displayed tile does not imply outside the road or outside observed data. No previous feature was removed or edited.\n",
        "The local PCA fit is finite (2nd–98th support percentiles); there is no infinite Hough/RANSAC line. The actual implementation risks are straight two-point approximations, bare-observation gap acceptance, duplicate halo proposals and indistinguishable line roles. Polygon stair-steps originate in 1-unit observation footprints and station strips, not painted-stripe tracing.\n",
    ]
    (report_dir / "layer_inventory.md").write_text("\n".join(text), encoding="utf-8")
    print(
        json.dumps(
            {
                "layers": len(records),
                "references": len(suspects),
                "outside_own_tile": sum(s["outside_own_tile_length"] > 0.1 for s in suspects),
            }
        )
    )
    return previous, records, suspects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("inventory", "pilot"), default="pilot")
    parser.add_argument("--group", action="append", help="Optional named pilot group")
    args = parser.parse_args()
    previous, layers, suspects = inventory()
    if args.stage == "pilot":
        from roadmarking.analysis.phase3_2_lines import GROUPS, run

        selected = {key: GROUPS[key] for key in args.group} if args.group else None
        run(ROOT, previous, layers, suspects, selected)


if __name__ == "__main__":
    main()
