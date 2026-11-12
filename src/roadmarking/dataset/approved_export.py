"""Export hash-bound user-approved outlines without inventing region completeness."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from shapely.geometry import box, mapping

from roadmarking.dataset.annotation_store import metadata, read_annotations
from roadmarking.dataset.labels import BACKGROUND, INSTANCE, LabelFeature
from roadmarking.dataset.qa import tile_overlay
from roadmarking.dataset.verified_masks import reviewed_masks
from roadmarking.dataset.writer import read_tile_inputs, write_sample
from roadmarking.rasterization.pipeline import dump, sha256, write_tif


def reconcile_approval(approval_path):
    """Validate existing approval; never create a human review event or change a store."""
    approval_path = Path(approval_path)
    approval = json.loads(approval_path.read_text())
    if approval.get("record_type") != "USER_ATTESTATION_OF_REVIEWED_ANNOTATION_APPEARANCE":
        raise ValueError("Expected recorded user approval of outline appearance")
    hashes = approval.get("artifact_sha256", {})
    project = Path(approval["project"])
    store = project.parent.parent / "annotations_draft/road_marking_annotations.gpkg"
    if str(project) not in hashes or str(store) not in hashes:
        raise ValueError("Project and annotation store must both be bound to approval")
    for filename, expected in hashes.items():
        if sha256(Path(filename)) != expected:
            raise ValueError(f"Approved artifact changed: {filename}")
    meta = metadata(store)
    for filename, expected in meta["source_artifact_sha256"].items():
        if sha256(Path(filename)) != expected:
            raise ValueError(f"Source raster artifact changed: {filename}")
    return approval, store, meta


def safe_crop_counts(state, sizes=(512, 384, 256)):
    """Exhaustively count native windows requiring no unsupported ignored pixels."""
    safe = np.isin(state, [BACKGROUND, INSTANCE])
    integral = np.pad(safe.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    results = {}
    for size in sizes:
        if min(safe.shape) < size:
            results[str(size)] = {"eligible_windows": 0, "best_supervised_fraction": None}
            continue
        count = (
            integral[size:, size:]
            - integral[:-size, size:]
            - integral[size:, :-size]
            + integral[:-size, :-size]
        )
        results[str(size)] = {
            "eligible_windows": int((count == size * size).sum()),
            "best_supervised_fraction": float(count.max() / (size * size)),
        }
    return results


def spatial_components(samples, guard=10.0):
    """Keep overlapping footprints and any shared physical instance together."""
    parent = list(range(len(samples)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i, sample in enumerate(samples):
        ids = {r["global_id"] for r in sample["instances"]}
        for j in range(i):
            other = samples[j]
            shared = ids & {r["global_id"] for r in other["instances"]}
            near = box(*sample["native_xy_bbox"]).distance(box(*other["native_xy_bbox"])) <= guard
            if near or shared:
                parent[find(i)] = find(j)
    groups = {}
    for i, sample in enumerate(samples):
        groups.setdefault(find(i), []).append(sample["tile_id"])
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def export_approved(approval_path, output, taxonomy_path):
    """Positive-mask export. Completeness must be explicit before any training is allowed."""
    approval_path = Path(approval_path).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    approval, store, meta = reconcile_approval(approval_path)
    protected = dict(approval["artifact_sha256"])
    protected.update(meta["source_artifact_sha256"])
    output.mkdir(parents=True)
    tiles = meta["tiles"]
    taxonomy = json.loads(Path(taxonomy_path).read_text())
    classes = {c["visual_id"]: c for c in taxonomy["classes"]}
    records, rejected = [], []
    for record in read_annotations(store):
        if record["layer"] == "review_areas":
            continue
        geometry = record["geometry"]
        if geometry is None or geometry.is_empty or not geometry.is_valid:
            rejected.append(
                {"annotation_id": record["annotation_id"], "reason": "INVALID_GEOMETRY"}
            )
            continue
        if record["layer"] != "paint_polygons":
            rejected.append(
                {"annotation_id": record["annotation_id"], "reason": "NO_APPROVED_AREA"}
            )
            continue
        records.append(
            {
                "annotation_id": record["annotation_id"],
                "global_id": record["global_instance_id"],
                "visual_id": record["visual_class_id"],
                "geometry": geometry,
                "source": str(store),
                "layer": record["layer"],
                "fid": record["annotation_id"],
                "source_kind": "APPROVED_LINE_DERIVED_FOOTPRINT",
                "width": record["marking_width"],
            }
        )
    # Only original symbol features intersecting the six displayed windows inherit
    # the outline appearance approval. No approval is extended to entire source layers.
    for filename, digest in approval["artifact_sha256"].items():
        source = Path(filename)
        if not source.name.startswith("original_client_qc_"):
            continue
        import pyogrio

        for layer_name, _ in pyogrio.list_layers(source):
            if "other" not in layer_name.lower():
                continue
            frame = gpd.read_file(source, layer=layer_name, fid_as_index=True)
            for fid, row in frame.iterrows():
                geometry = row.geometry
                if geometry is None or geometry.is_empty:
                    continue
                if not any(geometry.intersects(box(*t["bounds"])) for t in tiles):
                    continue
                identity = f"{digest}|{layer_name}|fid={fid}"
                if not geometry.is_valid or geometry.geom_type not in {"Polygon", "MultiPolygon"}:
                    rejected.append({"annotation_id": identity, "reason": "INVALID_SYMBOL_AREA"})
                    continue
                code = str(row.get("COD_RST", ""))
                candidate = taxonomy["client_code_candidates"].get(code)
                if candidate is None:
                    rejected.append({"annotation_id": identity, "reason": "UNMAPPED_SYMBOL_CODE"})
                    continue
                records.append(
                    {
                        "annotation_id": identity,
                        "global_id": hashlib.sha256(identity.encode()).hexdigest(),
                        "visual_id": candidate,
                        "geometry": geometry,
                        "source": filename,
                        "layer": layer_name,
                        "fid": int(fid),
                        "client_code": code,
                        "source_kind": "APPROVED_ORIGINAL_SYMBOL_OUTLINE",
                        "width": None,
                    }
                )
    ids = [r["global_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate global annotation identity")
    records.sort(key=lambda r: r["global_id"])
    geometries, reconciliation = [], []
    for i, record in enumerate(records, 1):
        geometry = record["geometry"]
        scope = [t["tile_id"] for t in tiles if geometry.intersects(box(*t["bounds"]))]
        details = {k: v for k, v in record.items() if k != "geometry"}
        details.update(
            geometry_sha256=hashlib.sha256(geometry.wkb).hexdigest(),
            reviewed_tile_scope=scope,
            class_mapping_status=classes[record["visual_id"]]["status"],
            instance_uid=i,
        )
        reconciliation.append(details)
        geometries.append({"type": "Feature", "properties": details, "geometry": mapping(geometry)})
    dump(
        output / "metadata/approved_geometry.geojson",
        {"type": "FeatureCollection", "features": geometries},
    )
    dump(
        output / "metadata/approval_reconciliation.json",
        {
            "approval_path": str(approval_path),
            "approval_sha256": sha256(Path(approval_path)),
            "approval": approval,
            "annotations": reconciliation,
            "rejected": rejected,
            "approval_scope": "OUTLINES_WITHIN_SIX_DISPLAYED_WINDOWS_ONLY",
            "invented_review_events": False,
            "verified_complete_regions": 0,
        },
    )
    samples, crops = [], {}
    for tile in tiles:
        # The approved project displayed owned copies of these exact intensity rasters.
        copied = Path(approval["project"]).parent / "rasters" / tile["tile_id"]
        for section, key in (
            ("layers", "normalized"),
            ("images", "grayscale"),
            ("masks", "observed"),
            ("masks", "valid"),
        ):
            source = Path(tile[section][key])
            if sha256(copied / source.name) != sha256(source):
                raise ValueError("Displayed raster differs from exported raster")
            protected[str(source)] = sha256(source)
        inputs = read_tile_inputs(tile)
        with rasterio.open(tile["masks"]["valid"]) as ds:
            valid = ds.read(1).astype(bool)
            if ds.transform != Affine(*tile["affine"]):
                raise ValueError("Raster affine mismatch")
        selected = [
            (i, r)
            for i, r in enumerate(records, 1)
            if r["geometry"].intersects(box(*tile["bounds"]))
        ]
        features = [
            LabelFeature(i, r["global_id"], r["visual_id"], "polygon", r["geometry"])
            for i, r in selected
        ]
        shape = (tile["height"], tile["width"])
        labels, categories = reviewed_masks(
            Affine(*tile["affine"]), shape, inputs["observed"], valid, features
        )
        # There is no complete-region/background approval in the recorded evidence.
        # Even a custom pixel-aware criterion cannot supply missing no-object labels.
        rgb = np.asarray(Image.open(tile["images"]["rgb"]))
        gray = np.asarray(Image.open(tile["images"]["grayscale"]))
        if rgb.dtype != np.uint8 or not np.array_equal(rgb, np.repeat(gray[..., None], 3, axis=2)):
            raise ValueError("Expected exact replicated-grayscale RGB")
        for key in ("rgb", "grayscale"):
            source = Path(tile["images"][key])
            protected[str(source)] = sha256(source)
        rgb_path = output / "images_rgb" / (tile["tile_id"] + ".png")
        rgb_path.parent.mkdir(parents=True, exist_ok=True)
        # Preserve source bytes; do not re-normalize or downsample.
        rgb_path.write_bytes(Path(tile["images"]["rgb"]).read_bytes())
        sample = write_sample(
            output,
            tile,
            inputs,
            labels,
            {
                "role": "APPROVED_POSITIVES_WITH_INCOMPLETE_REGION_SUPERVISION",
                "approval_sha256": sha256(Path(approval_path)),
                "outlines_approved": True,
                "stock_loss_eligible": False,
                "custom_loss_eligible": False,
                "rgb_file": str(rgb_path.relative_to(output)),
                "rgb_sha256": sha256(rgb_path),
                "semantic_role": "VISUAL_IDS_INCLUDE_EXPLICITLY_PROVISIONAL_SYMBOL_CLASS_HINTS",
                "training_target_mapping": taxonomy["confirmed_training_mapping"],
                "annotation_ids": [r["annotation_id"] for _, r in selected],
                "rejection_reasons": [
                    "NO_COMPLETE_REGION_OR_BACKGROUND_EVIDENCE",
                    "UNKNOWN_NODATA_AND_BOUNDARY_PIXELS",
                    "UNLABELED_INSTANCE_NO_OBJECT_SUPERVISION_UNRESOLVED",
                ],
            },
        )
        crop = safe_crop_counts(labels.state)
        crops[tile["tile_id"]] = crop
        sample["native_crop_audit"] = crop
        for name, array in {
            "categories": categories,
            "observed": inputs["observed"],
            "valid": valid,
            "valid_supervision": labels.state == INSTANCE,
            "padding": ~valid,
        }.items():
            path = output / "validity_masks" / tile["tile_id"] / (name + ".tif")
            write_tif(path, array.astype(np.uint8), Affine(*tile["affine"]), None)
        sample["valid_supervision_role"] = "POSITIVE_PIXELS_ONLY_NOT_SAFE_FULL_INSTANCE_LOSS"
        dump(output / "metadata/tiles" / (tile["tile_id"] + ".json"), sample)
        tile_overlay(
            output / "qa_previews" / (tile["tile_id"] + ".png"),
            tile,
            inputs["normalized"],
            [r["geometry"] for _, r in selected],
            labels,
            "USER-APPROVED OUTLINES / background unknown / " + tile["tile_id"],
        )
        samples.append(sample)
    groups = spatial_components(samples)
    split = {
        "method": "FOOTPRINT_GUARD_AND_SHARED_GLOBAL_ID_COMPONENTS",
        "guard_native_units": 10,
        "candidate_components": groups,
        "assignment": {},
        "training_eligible_samples": 0,
        "reason": "No safe fully supervised samples; no independent accuracy allocation claimed",
    }
    dump(output / "splits/spatial_groups.json", split)
    dump(output / "metadata/class_mapping.json", taxonomy)
    changed = [p for p, h in protected.items() if sha256(Path(p)) != h]
    if changed:
        raise ValueError(f"Protected inputs changed: {changed}")
    manifest = {
        "dataset_version": output.name,
        "status": "SUPERVISION_INCOMPLETE",
        "outlines_approved": True,
        "training_ready": False,
        "samples": samples,
        "approved_annotation_count": len(records),
        "approved_region_scope": [t["tile_id"] for t in tiles],
        "unique_outline_counts_by_visual_id": dict(Counter(r["visual_id"] for r in records)),
        "class_mapping": taxonomy,
        "splits": split,
        "rejected_annotations": rejected,
        "protected_inputs": protected,
        "protected_hash_mismatches": changed,
        "pixel_counts_are_overlapping_tile_appearances": True,
        "verified_complete_regions": 0,
    }
    dump(output / "metadata/dataset_manifest.json", manifest)
    return manifest


def validate_approved_export(root):
    """Read actual output arrays and approval hashes; readiness flags cannot bypass safety."""
    root = Path(root)
    manifest = json.loads((root / "metadata/dataset_manifest.json").read_text())
    reconciliation = json.loads((root / "metadata/approval_reconciliation.json").read_text())
    approval_path = Path(reconciliation["approval_path"])
    if sha256(approval_path) != reconciliation["approval_sha256"]:
        raise ValueError("Approval attestation changed")
    reconcile_approval(approval_path)
    for filename, expected in manifest["protected_inputs"].items():
        if sha256(Path(filename)) != expected:
            raise ValueError(f"Protected input changed: {filename}")
    outline_ids = {r["global_id"] for r in reconciliation["annotations"]}
    seen_tiles, state_counts, verified = set(), Counter(), 0
    for sample in manifest["samples"]:
        if sample["tile_id"] in seen_tiles:
            raise ValueError("Duplicate tile identity")
        seen_tiles.add(sample["tile_id"])
        arrays = {}
        for key, filename in sample["files"].items():
            path = root / filename
            if sha256(path) != sample["file_sha256"][key]:
                raise ValueError(f"Output hash mismatch: {path}")
            with rasterio.open(path) as ds:
                if ds.transform != Affine(*sample["affine"]) or ds.shape != (
                    sample["height"],
                    sample["width"],
                ):
                    raise ValueError("Pixel alignment mismatch")
                arrays[key] = ds.read()
        state, semantic, instance = (arrays[k][0] for k in ("state", "semantic", "instance"))
        observed = arrays["image"][1].astype(bool)
        if (instance[~observed] > 0).any() or (semantic[~observed] != 255).any():
            raise ValueError("NoData was supervised as paint/background")
        if not np.array_equal(instance > 0, state == INSTANCE):
            raise ValueError("Instance/state mismatch")
        if not (semantic[~np.isin(state, [BACKGROUND, INSTANCE])] == 255).all():
            raise ValueError("Unknown/boundary pixels acquired supervised class IDs")
        validity = root / "validity_masks" / sample["tile_id"]
        validity_arrays = {}
        for name in ("categories", "observed", "valid", "valid_supervision", "padding"):
            with rasterio.open(validity / (name + ".tif")) as ds:
                if ds.transform != Affine(*sample["affine"]) or ds.shape != state.shape:
                    raise ValueError("Validity-mask alignment mismatch")
                validity_arrays[name] = ds.read(1)
        coverage = validity_arrays["valid"].astype(bool)
        categories = state.copy()
        categories[~coverage] = 6
        expected_validity = {
            "observed": observed,
            "valid_supervision": state == INSTANCE,
            "padding": ~coverage,
            "categories": categories,
        }
        if any(not np.array_equal(validity_arrays[k], v) for k, v in expected_validity.items()):
            raise ValueError("Validity-mask content mismatch")
        for record in sample["instances"]:
            if record["global_id"] not in outline_ids:
                raise ValueError("Instance lacks approved outline provenance")
            if record["local_id"]:
                selected = instance == record["local_id"]
                if int(selected.sum()) != record["pixels"]:
                    raise ValueError("Instance pixel count mismatch")
                if not (semantic[selected] == record["visual_id"]).all():
                    raise ValueError("Instance/class mismatch")
        rgb_path = root / sample["rgb_file"]
        if sha256(rgb_path) != sample["rgb_sha256"]:
            raise ValueError("RGB input changed")
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image)
        if rgb.shape != (*state.shape, 3) or not np.array_equal(
            rgb, np.repeat(rgb[..., :1], 3, axis=2)
        ):
            raise ValueError("RGB sampling/channels changed")
        if sample["native_crop_audit"] != safe_crop_counts(state):
            raise ValueError("Native crop evidence changed")
        state_counts.update(
            {str(k): int(v) for k, v in zip(*np.unique(state, return_counts=True), strict=True)}
        )
        verified += int((state == BACKGROUND).sum())
    groups = spatial_components(manifest["samples"])
    if groups != manifest["splits"]["candidate_components"]:
        raise ValueError("Spatial component evidence changed")
    # This exporter intentionally does not invent complete-region evidence. If such
    # evidence is later supplied, a separate version must consume and validate it.
    if verified or manifest["training_ready"] or manifest["splits"]["assignment"]:
        raise ValueError("Outline approval alone cannot enable background or training")
    return {
        "integrity_passed": True,
        "outlines_approved": True,
        "image_mask_pairs": len(seen_tiles),
        "pixel_states": dict(state_counts),
        "verified_background_pixels": verified,
        "spatial_components": groups,
        "training_ready": False,
        "status": "SUPERVISION_INCOMPLETE",
    }
