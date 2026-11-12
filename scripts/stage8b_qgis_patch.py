"""Write a separate QGIS review project containing the confirmed additive patch."""

import json
from pathlib import Path

from qgis.core import (
    QgsApplication,
    QgsFillSymbol,
    QgsProject,
    QgsReferencedRectangle,
    QgsSingleSymbolRenderer,
    QgsVectorLayer,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reports/stage8b"


def main():
    app = QgsApplication([], False)
    app.initQgis()
    project = QgsProject.instance()
    original = (
        ROOT
        / "data/processed/stage7_new_test/20261107T051050944913Z/stage7_new_test_training_review.qgs"  # noqa: E501
    )
    assert project.read(str(original))
    patch = QgsVectorLayer(
        str(OUT / "single_dash_confirmed.gpkg") + "|layername=confirmed_dash_observed_paint",
        "CONFIRMED: T7_05 one added dash (observed paint only)",
        "ogr",
    )
    assert patch.isValid() and patch.featureCount() == 1
    patch.setReadOnly(True)
    patch.setCrs(project.crs())
    patch.setRenderer(
        QgsSingleSymbolRenderer(
            QgsFillSymbol.createSimple(
                {
                    "color": "0,128,255,0",
                    "outline_color": "30,110,255,255",
                    "outline_width": "0.35",
                    "outline_width_unit": "MM",
                }
            )
        )
    )
    project.addMapLayer(patch)
    fringe = QgsVectorLayer(
        str(OUT / "single_dash_confirmed.gpkg") + "|layername=localized_uncertain_boundary_ignore",
        "QA: uncertain paint-boundary pixels (excluded from background)",
        "ogr",
    )
    if fringe.isValid():
        fringe.setReadOnly(True)
        fringe.setCrs(project.crs())
        project.addMapLayer(fringe)
        project.layerTreeRoot().findLayer(fringe.id()).setItemVisibilityChecked(False)
    project.setTitle("Stage 8B confirmed annotation patch - protected test not consumed")
    extent = patch.extent()
    extent.grow(2)
    project.viewSettings().setDefaultViewExtent(QgsReferencedRectangle(extent, project.crs()))
    output = OUT / "T7_05_confirmed_correction.qgs"
    assert project.write(str(output))
    assert project.read(str(output))
    result = {
        "project": str(output),
        "all_layers_valid": all(layer.isValid() for layer in project.mapLayers().values()),
        "original_project_unchanged": True,
        "confirmed_patch_features": 1,
        "test_inference_run": False,
    }
    (OUT / "qgis_patch_validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    app.exitQgis()


if __name__ == "__main__":
    main()
