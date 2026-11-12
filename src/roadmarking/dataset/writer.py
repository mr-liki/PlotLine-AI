"""Incremental, deterministic writing of image / label GeoTIFFs and per-tile metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio

from roadmarking.dataset.geometry import tile_affine
from roadmarking.dataset.labels import STATE_NAMES, TileLabels, instance_id_to_semantic_id

NORMALIZED_NODATA = -9999.0


def read_tile_inputs(tile: dict) -> dict[str, np.ndarray]:
    out = {}
    for name in ("normalized", "observed", "count", "filled_mask"):
        with rasterio.open(tile["layers"][name]) as src:
            out[name] = src.read(1)
    return out


def _profile(tile: dict, dtype: str, count: int, nodata=None) -> dict:
    return {
        "driver": "GTiff",
        "height": tile["height"],
        "width": tile["width"],
        "count": count,
        "dtype": dtype,
        "crs": None,  # CRS_UNCONFIRMED: never assigned
        "transform": tile_affine(tile),
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }


def write_sample(root: Path, tile: dict, inputs: dict, labels: TileLabels, extra: dict) -> dict:
    """Write one training sample. Images keep raw normalized values and NoData.

    images/<id>.tif: band 1 normalized intensity (float32, NoData -9999, not re-stretched),
    band 2 observed (1/0), band 3 filled_mask (1 = interpolated, never observed).
    instance_masks/<id>_instance.tif (uint32), <id>_semantic.tif (uint8, 255 ignore),
    <id>_state.tif (uint8 label_state). Conversion to 3-channel RGB happens in Phase 5.
    """
    tid = tile["tile_id"]
    images = root / "images"
    masks = root / "instance_masks"
    meta = root / "metadata" / "tiles"
    for d in (images, masks, meta):
        d.mkdir(parents=True, exist_ok=True)
    norm = inputs["normalized"].astype(np.float32)
    stack = np.stack(
        [norm, inputs["observed"].astype(np.float32), inputs["filled_mask"].astype(np.float32)]
    )
    paths = {
        "image": images / f"{tid}.tif",
        "instance": masks / f"{tid}_instance.tif",
        "semantic": masks / f"{tid}_semantic.tif",
        "state": masks / f"{tid}_state.tif",
    }
    with rasterio.open(
        paths["image"], "w", **_profile(tile, "float32", 3, NORMALIZED_NODATA)
    ) as dst:
        dst.write(stack)
        dst.descriptions = ("normalized_intensity", "observed", "filled_mask")
    for key, array, dtype in (
        ("instance", labels.instance, "uint32"),
        ("semantic", labels.semantic, "uint8"),
        ("state", labels.state, "uint8"),
    ):
        with rasterio.open(paths[key], "w", **_profile(tile, dtype, 1)) as dst:
            dst.write(array.astype(dtype), 1)
    state_counts = np.bincount(labels.state.ravel(), minlength=len(STATE_NAMES))
    record = {
        "tile_id": tid,
        "source_file": tile["source_file"],
        "affine": tile["affine"],
        "native_xy_bbox": tile["native_xy_bbox"],
        "resolution": tile["resolution"],
        "height": tile["height"],
        "width": tile["width"],
        "crs_status": tile["crs_status"],
        "phase3_layers": tile["layers"],
        "files": {k: str(v.relative_to(root)) for k, v in paths.items()},
        "file_sha256": {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in paths.items()},
        "instance_id_to_semantic_id": {
            str(k): v for k, v in instance_id_to_semantic_id(labels).items()
        },
        "instances": labels.instances,
        "label_state_pixels": {STATE_NAMES[i]: int(c) for i, c in enumerate(state_counts)},
        **extra,
    }
    (meta / f"{tid}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record
