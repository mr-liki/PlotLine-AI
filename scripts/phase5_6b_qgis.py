"""Add a focused editable layer to a new copy of the review project."""

import json
from pathlib import Path

from qgis.core import (
    QgsApplication,
    QgsFillSymbol,
    QgsProject,
    QgsRectangle,
    QgsReferencedRectangle,
    QgsVectorLayer,
)

root = Path(__file__).resolve().parents[1]
c = json.loads((root / "outputs/reports/phase5_6b/latest.json").read_text())
spec = json.loads(Path(c["spec"]).read_text())
app = QgsApplication([], False)
app.initQgis()
project = QgsProject.instance()
assert project.read(str(Path(spec["output"]) / "independent_test_review.qgs"))
for node in project.layerTreeRoot().findLayers():
    if "Candidate painted" in node.layer().name():
        node.setItemVisibilityChecked(False)
group = project.layerTreeRoot().insertGroup(0, "CORRECTIONS — pending human verification")
for region in [t["candidate_id"] for t in spec["tiles"]]:
    layer = QgsVectorLayer(c["candidate_store"] + "|layername=corrections", region, "ogr")
    assert layer.isValid()
    layer.setSubsetString(f"candidate_id = '{region}'")
    layer.setReadOnly(region != c["affected_region"])
    layer.setName(region + (" — EDIT HERE" if not layer.readOnly() else " — retained read-only"))
    layer.renderer().setSymbol(
        QgsFillSymbol.createSimple(
            {"color": "0,0,0,0", "outline_color": "255,170,0", "outline_width": ".3"}
        )
    )
    project.addMapLayer(layer, False)
    group.addLayer(layer)
project.setTitle("Phase 5.6B — targeted annotation correction; no test approval yet")
project.viewSettings().setDefaultViewExtent(
    QgsReferencedRectangle(QgsRectangle(*c["issue_bounds"]), project.crs())
)
path = Path(c["output"]) / "targeted_test_correction.qgs"
assert project.write(str(path))
project.clear()
assert project.read(str(path))
assert all(layer.isValid() for layer in project.mapLayers().values())
c["project"] = str(path)
(root / "outputs/reports/phase5_6b/latest.json").write_text(json.dumps(c, indent=2))
(Path(c["report_dir"]) / "qgis_validation.json").write_text(
    json.dumps(
        {
            "project": str(path),
            "reopened": True,
            "all_layers_valid": True,
            "editable_region": c["affected_region"],
            "other_regions_read_only": True,
        },
        indent=2,
    )
)
project.clear()
app.exitQgis()
