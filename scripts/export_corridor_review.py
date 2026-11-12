"""Export attributed QGIS edits to the strict native-XY guided/review JSON format.

This never creates approval or verification attributes. A reviewer must supply
them in QGIS. The source identity is checked against the existing Phase 3 manifest.
"""

import argparse
import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import mapping

from roadmarking.analysis.corridor_pilot import read_json
from roadmarking.preprocessing.corridor_discovery import load_references
from roadmarking.quality.marking_retention import reviewed_samples
from roadmarking.rasterization.pipeline import INTACT, dump, sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gpkg", type=Path)
    parser.add_argument("--source", required=True, choices=tuple(INTACT))
    parser.add_argument("--kind", required=True, choices=("reference", "samples"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if args.output.exists():
        raise ValueError("Refusing to overwrite an existing review export")
    tiles = [
        tile
        for tile in read_json(root / "data/processed/phase3/metadata/tile_manifest.json")
        if Path(tile["source_file"]).stem == args.source
    ]
    digest = sha256(tiles[0]["source_file"])
    if digest != tiles[0]["source_sha256"]:
        raise ValueError("Original source identity changed")
    layers = (
        ("reference_candidates",)
        if args.kind == "reference"
        else ("paint_review", "road_boundary_review")
    )
    features = []
    for layer in layers:
        frame = gpd.read_file(args.gpkg, layer=layer)
        if frame.crs is not None:
            raise ValueError("Native review package must have no assigned CRS")
        status = "APPROVED" if args.kind == "reference" else "MANUALLY_VERIFIED"
        frame = frame[frame["review_status"] == status]
        for _, row in frame.iterrows():
            attribution = [str(row.get(key, "")).strip() for key in ("reviewer", "reviewed_at")]
            if any(value.lower() in ("", "nan", "nat", "none") for value in attribution):
                raise ValueError("Reviewer and review time required; exporter never supplies them")
            props = {key: str(row[key]) for key in ("review_status", "reviewer", "reviewed_at")}
            if args.kind == "samples":
                props["kind"] = "paint" if layer == "paint_review" else "road_boundary"
            features.append(
                {"type": "Feature", "geometry": mapping(row.geometry), "properties": props}
            )
    if not features:
        raise ValueError("No attributed, explicitly approved/verified features found")
    data = {
        "type": "FeatureCollection",
        "coordinate_frame": "NATIVE_XY",
        "source_sha256": digest,
        "features": features,
    }
    dump(args.output, data)
    if args.kind == "reference":
        boxes = [tile["native_xy_bbox"] for tile in tiles]
        extent = [
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        ]
        load_references(args.output, digest, extent, approved=True)
    else:
        reviewed_samples(args.output, digest)
    print(
        json.dumps({"output": str(args.output), "features": len(features), "source_sha256": digest})
    )


if __name__ == "__main__":
    main()
