"""Geospatial stack tests on synthetic data (library verification only)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Polygon

from roadmarking import healthcheck as hc
from roadmarking.config import load_config
from roadmarking.gis.attributes import load_schemas, validate_layer
from roadmarking.gis.export import SchemaValidationError, export_layer
from roadmarking.io.gis import read_vector
from roadmarking.rasterization.rasterizer import north_up_transform, pixel_to_world, world_to_pixel

# A projected test CRS only; the project does not assume any EPSG code.
TEST_CRS = "EPSG:3857"


@pytest.fixture(scope="module")
def schemas():
    return load_schemas(load_config()["gis_schema"])


def _line_row(**overrides):
    row = {
        "ID_SIG": "L1",
        "POSICAO": None,
        "DESIGNACAO": "1",
        "COD_JAE": None,
        "LARGURA": 0.12,
        "TRAÇO": None,
        "ESPAÇO": None,
        "ID_GRUPO": "G1",
        "COMENTARIO": None,
        "ID_HIST": "H-001",
        "OBS": None,
    }
    row.update(overrides)
    return row


def test_healthcheck_geospatial() -> None:
    info = hc.check_geospatial()
    assert info["gpkg_driver"]


def test_crs_transformation_roundtrip() -> None:
    t = Transformer.from_crs(TEST_CRS, "EPSG:4326", always_xy=True)
    x, y = 1_113_194.9, 6_446_275.8
    lon, lat = t.transform(x, y)
    x2, y2 = t.transform(lon, lat, direction="INVERSE")
    assert abs(x - x2) < 1e-6 and abs(y - y2) < 1e-6
    assert CRS.from_user_input(TEST_CRS).is_projected


def test_export_longitudinal_layer_preserves_z(tmp_path: Path, schemas) -> None:
    line = LineString([(0, 0, 100.123), (10, 0.5, 100.456), (20, 0, 100.789)])
    gdf = gpd.GeoDataFrame([_line_row()], geometry=[line], crs=TEST_CRS)
    out = export_layer(gdf, schemas["longitudinal_marks"], tmp_path / "marks.gpkg")
    back = read_vector(out, layer="longitudinal_marks")
    g = back.geometry.iloc[0]
    assert g.has_z and np.allclose(np.asarray(g.coords), np.asarray(line.coords))
    assert back.crs == gdf.crs
    assert back["ID_HIST"].iloc[0] == "H-001"
    assert "TRAÇO" in back.columns


def test_export_polygon_layer_with_hole(tmp_path: Path, schemas) -> None:
    shell = [(0, 0, 5.0), (3, 0, 5.1), (3, 3, 5.2), (0, 3, 5.3)]
    hole = [(1, 1, 5.15), (2, 1, 5.15), (2, 2, 5.15), (1, 2, 5.15)]
    poly = Polygon(shell, holes=[hole])
    row = {f: None for f in schemas["other_marks"].field_names}
    row.update(ID_SIG="P1", COD_RST="M15", AREA_PINTADA=float(poly.area))
    gdf = gpd.GeoDataFrame([row], geometry=[poly], crs=TEST_CRS)
    out = export_layer(gdf, schemas["other_marks"], tmp_path / "marks.gpkg")
    g = read_vector(out, layer="other_marks").geometry.iloc[0]
    assert g.has_z and len(g.interiors) == 1
    assert g.equals_exact(poly, 1e-9)


def test_schema_rejects_2d_and_missing_fields(schemas) -> None:
    gdf = gpd.GeoDataFrame([{"ID_SIG": "x"}], geometry=[LineString([(0, 0), (1, 1)])], crs=TEST_CRS)
    errors = validate_layer(gdf, schemas["longitudinal_marks"])
    assert any("missing fields" in e for e in errors)
    assert any("no Z" in e for e in errors)
    with pytest.raises(SchemaValidationError):
        export_layer(gdf, schemas["longitudinal_marks"], Path("unused.gpkg"))


def test_export_refuses_overwrite(tmp_path: Path, schemas) -> None:
    gdf = gpd.GeoDataFrame(
        [_line_row()], geometry=[LineString([(0, 0, 1), (1, 1, 1)])], crs=TEST_CRS
    )
    path = tmp_path / "x.gpkg"
    export_layer(gdf, schemas["longitudinal_marks"], path)
    with pytest.raises(FileExistsError):
        export_layer(gdf, schemas["longitudinal_marks"], path)


def test_pixel_world_transform_roundtrip() -> None:
    t = north_up_transform(x_min=1000.0, y_max=2000.0, gsd=0.05)
    cols, rows = np.array([0, 10, 99]), np.array([0, 5, 42])
    xs, ys = pixel_to_world(t, cols, rows)
    assert np.allclose(xs[0], 1000.025) and np.allclose(ys[0], 1999.975)
    c2, r2 = world_to_pixel(t, xs, ys)
    assert np.array_equal(c2, cols) and np.array_equal(r2, rows)
    with pytest.raises(ValueError):
        north_up_transform(0, 0, 0)
