"""Native-grid crops and unique coverage on sparse, existing raster windows."""

import numpy as np
from affine import Affine


def candidate_origins(regions, size, overlap):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Invalid tile dimensions")
    stride = size - overlap
    origins = set()
    for row, col, height, width in regions:
        if row % stride or col % stride:
            raise ValueError("Source region must start on the shared stride grid")
        origins.update(
            (r, c)
            for r in range(row, row + height, stride)
            for c in range(col, col + width, stride)
        )
    return sorted(origins)


def crop_sources(sources, row, col, size, fill=0, *, union=False):
    """Sources are (array, global_row, global_col). Reject inconsistent overlaps."""
    if not sources or size <= 0:
        raise ValueError("Nonempty sources and positive size required")
    result = np.full((size, size), fill, dtype=sources[0][0].dtype)
    valid = np.zeros((size, size), bool)
    for array, sr, sc in sources:
        r0, c0 = max(row, sr), max(col, sc)
        r1, c1 = min(row + size, sr + array.shape[0]), min(col + size, sc + array.shape[1])
        if r1 <= r0 or c1 <= c0:
            continue
        target = np.s_[r0 - row : r1 - row, c0 - col : c1 - col]
        patch = array[r0 - sr : r1 - sr, c0 - sc : c1 - sc]
        shared = valid[target]
        if not union and not np.array_equal(result[target][shared], patch[shared]):
            raise ValueError("Source overlap disagrees")
        result[target] = np.maximum(result[target], patch) if union else patch
        valid[target] = True
    return result, valid


def tile_transform(base, row, col):
    return base @ Affine.translation(col, row)


def pixel_keys(mask, row, col):
    """Collision-free signed global coordinates; callers segregate source identities."""
    rr, cc = np.nonzero(mask)
    if row < 0 or col < 0 or row + mask.shape[0] >= 2**32 or col + mask.shape[1] >= 2**32:
        raise ValueError("Global pixel coordinates outside uint32 range")
    return ((rr.astype(np.uint64) + row) << np.uint64(32)) | (cc.astype(np.uint64) + col)


def unique_pixel_count(chunks):
    return int(np.unique(np.concatenate(chunks)).size) if chunks else 0


def phase4_virtual_layers(entry, directory):
    """Expose crop layers as VRT references without duplicating source GeoTIFFs."""
    import xml.etree.ElementTree as ET
    from pathlib import Path

    import rasterio

    from roadmarking.rasterization.pipeline import write_tif

    directory = Path(directory)
    size, row, col = entry["height"], entry["row_offset"], entry["col_offset"]
    transform = Affine(*entry["affine"])
    layers = {}
    for name in ("maximum", "mean", "normalized", "count", "observed"):
        types = {
            "maximum": "Int32",
            "mean": "Float32",
            "normalized": "Float32",
            "count": "UInt32",
            "observed": "Byte",
        }
        nodata = -9999 if name in {"mean", "normalized"} else -1 if name == "maximum" else 0
        tree = ET.Element("VRTDataset", rasterXSize=str(size), rasterYSize=str(size))
        ET.SubElement(tree, "GeoTransform").text = ",".join(str(v) for v in transform.to_gdal())
        band = ET.SubElement(tree, "VRTRasterBand", dataType=types[name], band="1")
        ET.SubElement(band, "NoDataValue").text = str(nodata)
        for window in entry["source_windows"]:
            sr, sc = window["row_offset"], window["col_offset"]
            path = window["rasters"][name]
            with rasterio.open(path) as ds:
                height, width = ds.shape
            r0, c0 = max(row, sr), max(col, sc)
            r1, c1 = min(row + size, sr + height), min(col + size, sc + width)
            if r1 <= r0 or c1 <= c0:
                continue
            source = ET.SubElement(band, "SimpleSource")
            ET.SubElement(source, "SourceFilename", relativeToVRT="0").text = path
            ET.SubElement(source, "SourceBand").text = "1"
            ET.SubElement(
                source,
                "SrcRect",
                xOff=str(c0 - sc),
                yOff=str(r0 - sr),
                xSize=str(c1 - c0),
                ySize=str(r1 - r0),
            )
            ET.SubElement(
                source,
                "DstRect",
                xOff=str(c0 - col),
                yOff=str(r0 - row),
                xSize=str(c1 - c0),
                ySize=str(r1 - r0),
            )
        path = directory / f"{name}.vrt"
        ET.ElementTree(tree).write(path, encoding="utf-8", xml_declaration=True)
        layers[name] = str(path)
    filled = directory / "filled_mask.tif"
    write_tif(
        filled, np.zeros((size, size), np.uint8), transform, None, config_version="phase3.3-no-fill"
    )
    layers["filled_mask"] = str(filled)
    return {
        **entry,
        "layers": layers,
        "native_xy_bbox": entry["bounds"],
        "source_file": entry["source_file"],
    }
