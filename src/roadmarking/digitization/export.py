"""GeoPackage writers: engineering layers (full provenance) and a provisional client-schema layer.

The client deliverable layer mirrors the supplied client file exactly (layer ``Highway``,
MultiLineString Z, EPSG:3763, same 16 fields and dtypes). Fields the two-class model cannot
determine are left NULL and reported; nothing is guessed.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from shapely.geometry import MultiLineString

from .line_tracing import PATTERNS

CLIENT_FIELDS = [
    ("ID_SIG", "float64"),
    ("POSICAO", "object"),
    ("DESIGNACAO", "object"),
    ("LARGURA", "float64"),
    ("COMENTARIO", "object"),
    ("ID_HIST", "object"),
    ("COD_SUBLAN", "object"),
    ("SUBLANCO", "object"),
    ("NO", "object"),
    ("RAMO", "object"),
    ("TRAÇO", "object"),
    ("ESPAÇO", "object"),
    ("OBS", "object"),
    ("ID_GRUPO", "object"),
    ("COD_JAE", "object"),
    ("ELEVATION", "object"),
]
NOMINAL_WIDTH = (0.15, 0.30)
NOMINAL_DASH = (3.0, 4.0, 10.0)
NOMINAL_GAP = (4.0, 10.0)


def snap(value, nominal, rel=0.25):
    if value is None or not np.isfinite(value):
        return None
    best = min(nominal, key=lambda n: abs(n - value))
    return best if abs(best - value) <= rel * best else None


def write_layer(gdf, path, layer, crs):
    gdf = gdf.set_crs(crs, allow_override=True) if isinstance(gdf, gpd.GeoDataFrame) else gdf
    pyogrio.write_dataframe(gdf, path, layer=layer, driver="GPKG", append=Path(path).exists())


def write_table(df: pd.DataFrame, path, layer):
    pyogrio.write_dataframe(df, path, layer=layer, driver="GPKG", append=Path(path).exists())


def client_rows(features, cod_jae_by_designacao):
    rows, unresolved = [], set()
    for i, f in enumerate(features, start=1):
        width = snap(f["width_m"], NOMINAL_WIDTH)
        if f["class_id"] == 0:
            design = None  # guide vs continuous line cannot be told apart by the model
            traco = espaco = None
            unresolved.add("DESIGNACAO(solid subtype)")
        else:
            pattern = f.get("dash_pattern")
            design = {"normal": "Linha descontínua", "aviso": "Linha descontínua de aviso"}.get(
                pattern
            )
            if pattern:
                traco, espaco, width = PATTERNS[pattern]
            elif "dash_pattern" not in f:
                traco = snap(f["length_m"], NOMINAL_DASH)
                espaco = snap(f.get("gap_next_m"), NOMINAL_GAP)
                if (traco, espaco) == (4.0, 10.0):
                    design = "Linha descontínua"
            else:
                traco = espaco = None
            if design is None:
                unresolved.add("DESIGNACAO(dash subtype)")
        rows.append(
            {
                "ID_SIG": float(i),
                "POSICAO": None,
                "DESIGNACAO": design,
                "LARGURA": width if width is not None else round(f["width_m"], 2),
                "COMENTARIO": "AI digitized - PENDING REVIEW",
                "ID_HIST": None,
                "COD_SUBLAN": None,
                "SUBLANCO": None,
                "NO": None,
                "RAMO": None,
                "TRAÇO": None if traco is None else f"{traco:.1f}",
                "ESPAÇO": None if espaco is None else f"{espaco:.1f}",
                "OBS": f"mark_id={f['mark_id']}",
                "ID_GRUPO": None,
                "COD_JAE": cod_jae_by_designacao.get(design) if design else None,
                "ELEVATION": f"{f['mean_z']:.2f}" if np.isfinite(f["mean_z"]) else None,
                "geometry": MultiLineString([f["geometry_3d"]]),
            }
        )
    unresolved |= {"POSICAO", "ID_HIST", "COD_SUBLAN", "SUBLANCO", "NO", "RAMO", "ID_GRUPO"}
    return rows, sorted(unresolved)


__all__ = ["CLIENT_FIELDS", "client_rows", "snap", "write_layer", "write_table"]
