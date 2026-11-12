"""Stage 9 — validated 3D GeoPackage export."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from roadmarking.gis.attributes import LayerSchema, validate_layer
from roadmarking.io.gis import write_gpkg


class SchemaValidationError(ValueError):
    """Raised when a layer does not satisfy its output schema."""


def export_layer(
    gdf: gpd.GeoDataFrame, schema: LayerSchema, path: Path, *, overwrite: bool = False
) -> Path:
    """Validate ``gdf`` against ``schema`` and write it as a GeoPackage layer."""
    errors = validate_layer(gdf, schema)
    if errors:
        raise SchemaValidationError(f"Layer '{schema.name}' invalid: {errors}")
    columns = [*schema.field_names, gdf.geometry.name]
    return write_gpkg(gdf[columns], path, schema.name, overwrite=overwrite)
