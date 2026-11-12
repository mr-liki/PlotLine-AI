"""Deterministic overlapping windows on one common native pixel grid."""

from dataclasses import dataclass

from affine import Affine


@dataclass(frozen=True)
class TileWindow:
    row_off: int
    col_off: int
    height: int
    width: int
    transform: Affine


def iter_tiles(raster, tile_size=1024, overlap=256):
    if tile_size <= 0 or not 0 <= overlap < tile_size:
        raise ValueError("Require tile_size > overlap >= 0")
    height, width = raster.image.shape[:2]
    stride = tile_size - overlap
    for r in range(0, height, stride):
        for c in range(0, width, stride):
            yield TileWindow(
                r,
                c,
                min(tile_size, height - r),
                min(tile_size, width - c),
                raster.transform @ Affine.translation(c, r),
            )


def sparse_tile_origins(blocks, block_size=512, tile_size=1024, overlap=256):
    """Only windows intersecting existing blocks; never enumerate bounding-box tiles."""
    if block_size <= 0 or tile_size <= 0 or not 0 <= overlap < tile_size:
        raise ValueError("Invalid block or tile dimensions")
    stride = tile_size - overlap
    windows = set()
    for br, bc in blocks:
        r0, c0 = br * block_size, bc * block_size
        # A block may intersect a window starting before its own stride cell.
        rmin = max(0, (r0 - tile_size) // stride + 1)
        cmin = max(0, (c0 - tile_size) // stride + 1)
        for rr in range(rmin, (r0 + block_size - 1) // stride + 1):
            for cc in range(cmin, (c0 + block_size - 1) // stride + 1):
                windows.add((rr * stride, cc * stride))
    return sorted(windows)


def tile_id(source, row, col):
    return f"{source}__r{row:07d}_c{col:07d}"
