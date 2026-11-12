import json
import sys
from pathlib import Path

from qgis.core import (
    QgsApplication,
    QgsFillSymbol,
    QgsProject,
    QgsSingleSymbolRenderer,
    QgsVectorLayer,
)

base = Path.cwd() / "outputs/reports/stage9b"
latest = json.loads((base / "latest.json").read_text())
previous = base / "coverage_followup"
refine = "--refine-text" in sys.argv
out = base / "text_coverage_refinement" if refine else previous
final = "--final-text" in sys.argv
if final:
    refinement = base / "text_coverage_refinement"
    out = base / "final_text_review"
app = QgsApplication([], False)
app.initQgis()
project = QgsProject.instance()
source = str(previous / "stage9b_coverage_followup.qgs") if refine else latest["review_project"]
if final:
    source = str(refinement / "stage9b_coverage_followup.qgs")
if "--lowest-group" in sys.argv:
    source = str(out / "stage9b_coverage_followup.qgs")
    out = base / "lowest_group_review"
if "--bottom-strokes" in sys.argv:
    source = str(out / "stage9b_coverage_followup.qgs")
    out = base / "bottom_stroke_review"
if "--stroke-gaps" in sys.argv:
    source = str(out / "stage9b_coverage_followup.qgs")
    out = base / "stroke_gap_review"
assert project.read(source)
group = project.layerTreeRoot().insertGroup(0, "Coverage follow-up - review only")
layers = (
    ("additional_bright_support", "115,255,105" if refine else "80,230,255", True),
    ("observed_review_extent", "255,220,0", False),
)
if final:
    layers = (
        (
            "stroke_candidates",
            "255,255,90" if "--lowest-group" in sys.argv else "115,255,105",
            True,
        ),
        ("unresolved_extent", "255,220,0", False),
    )
for name, color, visible in layers:
    layer = QgsVectorLayer(f"{out / 'coverage_followup.gpkg'}|layername={name}", name, "ogr")
    assert layer.isValid()
    layer.setCrs(project.crs())
    layer.setReadOnly(True)
    layer.setRenderer(
        QgsSingleSymbolRenderer(
            QgsFillSymbol.createSimple(
                {"color": "0,0,0,0", "outline_color": color, "outline_width": ".16"}
            )
        )
    )
    project.addMapLayer(layer, False)
    group.addLayer(layer).setItemVisibilityChecked(visible)
target = str(out / "stage9b_coverage_followup.qgs")
project.setFileName(target)
assert project.write()
assert project.read(target)
assert all(layer.isValid() for layer in project.mapLayers().values())
print(f"Reopened {len(project.mapLayers())} valid layers")
