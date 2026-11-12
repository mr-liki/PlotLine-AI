"""Build the Stage 9E non-text review QGIS project (run with QGIS Python)."""

import json
import sys
from pathlib import Path


def main():
    from qgis.core import (
        Qgis,
        QgsApplication,
        QgsBookmark,
        QgsCoordinateReferenceSystem,
        QgsProject,
        QgsRasterLayer,
        QgsReferencedRectangle,
        QgsVectorLayer,
    )

    root = Path(__file__).resolve().parents[1]
    latest = json.loads(
        (root / "outputs/reports/stage9e/negative_review_package/latest.json").read_text()
    )
    spec = json.loads((Path(latest["run"]) / "review_spec.json").read_text())
    data = Path(spec["data"])
    app = QgsApplication([], False)
    app.initQgis()
    project = QgsProject.instance()
    project.setCrs(QgsCoordinateReferenceSystem())
    project.setFilePathStorage(Qgis.FilePathType.Relative)
    project.setTitle(
        "Stage 9E - non-text paint confirmation (training split) - human review required"
    )
    tree = project.layerTreeRoot()
    vectors = tree.addGroup("Review")
    rasters = tree.addGroup("Training crop intensity (read-only)")
    valid_count = 0
    layers = []
    for layer, name, editable in (
        ("non_text_review_EDITABLE", "Non-text review (EDITABLE)", True),
        ("client_symbol_candidates_READ_ONLY", "Client symbol candidates (read-only)", False),
        ("training_crop_footprints", "Approved training crop footprints", False),
    ):
        lyr = QgsVectorLayer(
            f"{data / 'stage9e_non_text_review.gpkg'}|layername={layer}", name, "ogr"
        )
        lyr.setReadOnly(not editable)
        if lyr.isValid():
            valid_count += 1
        project.addMapLayer(lyr, False)
        vectors.addLayer(lyr)
        layers.append(lyr)
    for crop in spec["crops"]:
        ras = QgsRasterLayer(str(data / "rasters" / f"{crop}_intensity.tif"), crop)
        if ras.isValid():
            valid_count += 1
        project.addMapLayer(ras, False)
        rasters.addLayer(ras)
    manager = project.bookmarkManager()
    footprints = layers[2]
    for feature in footprints.getFeatures():
        bm = QgsBookmark()
        bm.setName(f"{feature['crop_id']} ({feature['symbols']} symbols)")
        bm.setExtent(QgsReferencedRectangle(feature.geometry().boundingBox(), project.crs()))
        manager.addBookmark(bm)
    out = data / "stage9e_non_text_review.qgs"
    project.write(str(out))
    project.clear()
    reopened = QgsProject.instance().read(str(out))
    result = dict(
        project=str(out),
        project_reopened=bool(reopened),
        valid_layers=valid_count,
        expected_layers=3 + len(spec["crops"]),
        bookmarks=len(manager.bookmarks()) if reopened else None,
        approval_recorded=False,
    )
    (Path(latest["run"]) / "qgis_project.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    app.exitQgis()
    return 0 if valid_count == result["expected_layers"] else 1


if __name__ == "__main__":
    sys.exit(main())
