"""GeoPackage / Shapefile reading and writing (geopandas + pyogrio)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from roadmarking.utils.logging import get_logger

log = get_logger(__name__)


def read_vector(path: str | Path, layer: str | None = None) -> gpd.GeoDataFrame:
    """Read a GPKG/SHP layer, preserving Z coordinates and CRS."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return gpd.read_file(path, layer=layer, engine="pyogrio")


def write_gpkg(
    gdf: gpd.GeoDataFrame, path: str | Path, layer: str, *, overwrite: bool = False
) -> Path:
    """Write a GeoDataFrame to a GeoPackage layer.

    Refuses to replace an existing file unless ``overwrite=True``.
    """
    path = Path(path)
    if gdf.crs is None:
        raise ValueError("Refusing to write GeoPackage without a CRS")
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace layer '{layer}'")
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, layer=layer, driver="GPKG", engine="pyogrio")
    log.info("Wrote %d features to %s:%s", len(gdf), path.name, layer)
    return path
