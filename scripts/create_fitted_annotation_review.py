"""Run with QGIS Python; review a candidate fit without assigning a guessed CRS."""

import json
import sys
from pathlib import Path


def main(report_path):
    from osgeo import ogr
    from qgis.core import (
        Qgis,
        QgsApplication,
        QgsCoordinateReferenceSystem,
        QgsFillSymbol,
        QgsLineSymbol,
        QgsMarkerSymbol,
        QgsProject,
        QgsRectangle,
        QgsReferencedRectangle,
        QgsVectorLayer,
    )

    app = QgsApplication([], False)
    app.initQgis()
    report = json.loads(Path(report_path).read_text())
    output = Path(report["output_root"])
    project = QgsProject.instance()
    project.setCrs(QgsCoordinateReferenceSystem())
    project.setTitle("UNVERIFIED rigid annotation placement — review only")
    project.setFilePathStorage(Qgis.FilePathType.Relative)
    top = project.layerTreeRoot()
    annotations = top.addGroup("Fitted annotations — UNVERIFIED; original Z not registered")
    observations = top.addGroup("Sampled bright LiDAR — incomplete; not verified paint")
    footprints = top.addGroup("Acquisition header footprints — not road boundaries")
    dataset = ogr.Open(report["geopackage"], 0)
    names = [dataset.GetLayerByIndex(i).GetName() for i in range(dataset.GetLayerCount())]
    dataset = None
    extent = None
    for name in names:
        layer = QgsVectorLayer(report["geopackage"] + "|layername=" + name, name, "ogr")
        if not layer.isValid():
            raise RuntimeError("Cannot reopen fitted GeoPackage")
        layer.setCrs(QgsCoordinateReferenceSystem())
        layer.setReadOnly(True)
        project.addMapLayer(layer, False)
        if name.startswith("fit_"):
            group = annotations
            if layer.geometryType() == Qgis.GeometryType.Line:
                layer.renderer().setSymbol(
                    QgsLineSymbol.createSimple({"line_color": "255,100,0", "line_width": "0.25"})
                )
            else:
                layer.renderer().setSymbol(
                    QgsFillSymbol.createSimple(
                        {
                            "color": "255,0,255,0",
                            "outline_color": "255,0,255",
                            "outline_width": "0.2",
                        }
                    )
                )
        elif name == "raw_bright_sample_points":
            group = observations
            layer.renderer().setSymbol(
                QgsMarkerSymbol.createSimple(
                    {"name": "circle", "color": "0,120,255", "size": "0.7", "outline_style": "no"}
                )
            )
        else:
            group = footprints
            layer.renderer().setSymbol(
                QgsFillSymbol.createSimple(
                    {
                        "color": "100,100,100,12",
                        "outline_color": "100,100,100,50",
                        "outline_width": "0.1",
                    }
                )
            )
        group.addLayer(layer)
        if extent is None:
            extent = QgsRectangle(layer.extent())
        else:
            extent.combineExtentWith(layer.extent())
    project.setCrs(QgsCoordinateReferenceSystem())
    project.viewSettings().setDefaultViewExtent(
        QgsReferencedRectangle(extent, QgsCoordinateReferenceSystem())
    )
    path = output / "fitted_annotations_UNVERIFIED.qgs"
    if not project.write(str(path)):
        raise RuntimeError("QGIS project write failed")
    project.clear()
    if not project.read(str(path)):
        raise RuntimeError("QGIS project reopen failed")
    check = {
        "project": str(path),
        "reopened": True,
        "project_crs_valid": project.crs().isValid(),
        "layers": [
            {"name": layer.name(), "valid": layer.isValid()}
            for layer in project.mapLayers().values()
        ],
    }
    if check["project_crs_valid"] or not all(layer["valid"] for layer in check["layers"]):
        raise RuntimeError("Invalid candidate review project")
    (output / "metadata/qgis_review_validation.json").write_text(json.dumps(check, indent=2))
    print(path)
    project.clear()
    app.exitQgis()


if __name__ == "__main__":
    main(sys.argv[1])
