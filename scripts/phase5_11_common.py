"""Shared Phase 5.11 loaders: hashed client features and role-tagged reserved geography."""

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import shapely
from shapely.geometry import box

from roadmarking.dataset.geometry import painted_area_from_centerline
from roadmarking.dataset.taxonomy import Taxonomy, to_float
from roadmarking.rasterization.pipeline import sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCE_AUDIT = ROOT / "outputs/reports/phase5/20261105T172517_source_audit"
NATIVE = 0.05
MERGE = {"Linha de aceleração", "Linha de abrandamento"}
EDGE = {"Guia direita", "Guia esquerda"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def client_features(check_network=True):
    """All valid client features with stable IDs (same scheme as every earlier phase)."""
    audit = load(SOURCE_AUDIT / "report.json")
    taxonomy = Taxonomy.load(ROOT / "configs/phase4_taxonomy.yaml")
    features, layers = [], []
    for source in audit["annotation_layers"]:
        local = Path(source["local_snapshot"])
        assert sha256(local) == source["sha256"], "client snapshot changed"
        network_match = sha256(source["source"]) == source["sha256"] if check_network else None
        layers.append(
            {
                **{k: source[k] for k in ("source", "local_snapshot", "layer", "sha256")},
                "network_file_matches_snapshot": network_match,
            }
        )
        frame = gpd.read_file(local, layer=source["layer"], fid_as_index=True)
        duplicate = frame.geometry.to_wkb().duplicated(keep=False)
        for fid, row in frame.iterrows():
            g = row.geometry
            if g is None or g.is_empty or not g.is_valid or duplicate.loc[fid]:
                continue
            line = g.geom_type in {"LineString", "MultiLineString"}
            attrs = row.to_dict()
            decision = taxonomy.classify_line(attrs) if line else taxonomy.classify_polygon(attrs)
            width = to_float(attrs.get("LARGURA")) if line else None
            if line and (not width or width <= 0 or decision.visual_id not in (1, 2)):
                continue
            identity = f"{source['sha256']}|{source['layer']}|fid={fid}"
            features.append(
                {
                    "global_id": hashlib.sha256(identity.encode()).hexdigest(),
                    "source_identity": identity,
                    "layer": source["layer"],
                    "designation": None if not line else str(attrs.get("DESIGNACAO")),
                    "code": attrs.get("COD_RST") if not line else None,
                    "group": None if attrs.get("ID_GRUPO") is None else str(attrs.get("ID_GRUPO")),
                    "client_sig": None if attrs.get("ID_SIG") is None else str(attrs.get("ID_SIG")),
                    "visual_id": decision.visual_id or -1,
                    "width": width,
                    "centerline": shapely.force_2d(g) if line else None,
                    "geometry": (
                        painted_area_from_centerline(g, width) if line else shapely.force_2d(g)
                    ),
                }
            )
    return features, layers, audit


def reserved_geography():
    """Footprints and source IDs by role. Only metadata is read; no protected imagery/labels."""
    p57 = load(ROOT / "outputs/reports/phase5_7_expanded/latest.json")
    manifest = load(Path(p57["dataset"]) / "metadata/dataset_manifest.json")
    p55 = load(Path(manifest["parent"]) / "metadata/dataset_manifest.json")
    regions = []
    for r in p55["regions"]:
        role = "train" if r.get("existing_split", "train") == "train" else "legacy_dev"
        regions.append(
            {
                "id": r["tile_id"],
                "role": role,
                "bounds": r["native_xy_bbox"],
                "ids": list(r["source_global_ids"]),
            }
        )
    specs = {
        "outputs/reports/phase5_6a/20261106T093757339115Z/new_region_candidates.json": {
            "FUTURE_TEST_1": "historical_benchmark",
            "FUTURE_TEST_2": "historical_benchmark",
            "FUTURE_TEST_3": "historical_benchmark",
        },
        "outputs/reports/phase5_7_new_review/20261106T105319835838Z/new_region_candidates.json": {
            "NEW_TRAIN_REVIEW": "train",
            "NEW_DEV_REVIEW": "sparse_dev",
            "NEW_FINAL_TEST_RESERVED": "protected_final_test",
        },
        "outputs/reports/phase5_8_actual_review/20261106T115446528985Z/"
        "new_region_candidates.json": {"PHASE58_POSITIVE_DEV_REVIEW": "positive_dev"},
        "outputs/reports/phase5_8_new_review/20261106T114606200459Z/new_region_candidates.json": {
            "PHASE58_NEW_DEV_1": "unreviewed_candidate_held_out"
        },
    }
    for path, roles in specs.items():
        for tile in load(ROOT / path)["tiles"]:
            if tile["candidate_id"] in roles:
                regions.append(
                    {
                        "id": tile["candidate_id"],
                        "role": roles[tile["candidate_id"]],
                        "bounds": tile["bounds"],
                        "ids": list(tile["source_global_ids"]),
                        "source": tile.get("source"),
                    }
                )
    extra = set(manifest["protected_test_ids"])
    for r in load(
        ROOT / "outputs/reports/phase5_4/20261105T173740792650Z/"
        "future_test_region_reservations.json"
    )["regions"]:
        extra.update(r["reference_annotation_ids"])
    for r in regions:
        r["box"] = box(*r["bounds"])
    return regions, extra, manifest
