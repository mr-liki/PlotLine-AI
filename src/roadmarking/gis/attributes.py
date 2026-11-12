"""Stage 7 — GIS attribute schemas and validation.

Phase 1 provides schema definitions loaded from ``configs/gis_schema.yaml`` and a
structural validator. Classification rules (DESIGNACAO, COD_RST, POSICAO, measurements,
historical-ID association) are implemented in a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import geopandas as gpd

_PY_TYPES: dict[str, tuple[type, ...]] = {"str": (str,), "float": (float, int), "int": (int,)}


@dataclass(frozen=True)
class FieldSpec:
    name: str
    type: str
    required: bool = False
    unit: str | None = None


@dataclass(frozen=True)
class LayerSchema:
    name: str
    geometry_type: str  # e.g. "LineStringZ", "PolygonZ"
    fields: tuple[FieldSpec, ...]

    @property
    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]

    @property
    def base_geometry(self) -> str:
        return self.geometry_type.removesuffix("Z")


def load_schemas(gis_schema: dict[str, Any]) -> dict[str, LayerSchema]:
    """Build :class:`LayerSchema` objects from the ``gis_schema`` config section."""
    out: dict[str, LayerSchema] = {}
    for name, spec in gis_schema["layers"].items():
        fields = tuple(FieldSpec(**{k: v for k, v in f.items()}) for f in spec["fields"])
        out[name] = LayerSchema(name=name, geometry_type=spec["geometry_type"], fields=fields)
    return out


def validate_layer(gdf: gpd.GeoDataFrame, schema: LayerSchema) -> list[str]:
    """Return a list of schema violations (empty list means valid)."""
    errors: list[str] = []
    missing = [n for n in schema.field_names if n not in gdf.columns]
    if missing:
        errors.append(f"missing fields: {missing}")
    if gdf.crs is None:
        errors.append("CRS is not set")
    for idx, geom in gdf.geometry.items():
        if geom is None or geom.is_empty:
            errors.append(f"row {idx}: empty geometry")
            continue
        if geom.geom_type != schema.base_geometry:
            errors.append(f"row {idx}: geometry {geom.geom_type} != {schema.base_geometry}")
        if not geom.has_z:
            errors.append(f"row {idx}: geometry has no Z coordinates")
    for f in schema.fields:
        if f.name not in gdf.columns:
            continue
        col = gdf[f.name]
        if f.required and col.isna().any():
            errors.append(f"required field {f.name} has null values")
        bad = [v for v in col.dropna() if not isinstance(v, _PY_TYPES[f.type])]
        if bad:
            errors.append(f"field {f.name}: {len(bad)} value(s) not of type {f.type}")
    return errors
