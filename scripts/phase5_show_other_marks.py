"""Expose existing client symbol annotations in a new, read-only QGIS review copy.

Run with the installed QGIS python-qgis.bat. No training labels are approved.
"""

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main(root):
    from qgis.core import (
        QgsApplication,
        QgsFeatureRequest,
        QgsFillSymbol,
        QgsMapRendererParallelJob,
        QgsMapSettings,
        QgsProject,
        QgsRectangle,
    )
    from qgis.PyQt.QtCore import QSize

    reports = root / "outputs/reports/phase5"
    conversion = json.loads(
        Path(
            json.loads((reports / "annotation_conversion_latest.json").read_text())["report"]
        ).read_text()
    )
    original = Path(conversion["qgis_project"])
    source_workspace = original.parent.parent
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    workspace = source_workspace.parent / ("symbols_review_" + stamp)
    audit = json.loads(
        Path(json.loads((reports / "source_audit_latest.json").read_text())["report"]).read_text()
    )
    # Confirm that the local reference copies still represent the supplied live source.
    checks = []
    for layer in audit["annotation_layers"]:
        if "other" not in layer["source"].lower():
            continue
        actual = digest(layer["source"])
        if actual != layer["sha256"]:
            raise ValueError("Client source changed; refresh the source audit first")
        checks.append({"source": layer["source"], "sha256": actual})
    shutil.copytree(source_workspace, workspace)
    project_path = workspace / "annotation_projects" / original.name
    app = QgsApplication([], False)
    app.initQgis()
    project = QgsProject.instance()
    if not project.read(str(project_path)):
        raise ValueError("Cannot open copied QGIS project")
    spec = json.loads((source_workspace / "annotation_projects/qgis_spec.json").read_text())
    tiles = spec["tiles"]
    results, other_layers = [], []
    symbol_group = project.layerTreeRoot().insertGroup(0, "CLIENT SYMBOLS - orange outlines")
    for layer in project.mapLayers().values():
        filename = Path(layer.source().split("|")[0])
        if not filename.resolve().is_relative_to(workspace.resolve()):
            raise ValueError(f"Copied project references an external file: {filename}")
        if "Other Marks" not in layer.name():
            continue
        if not layer.isValid():
            raise ValueError("Invalid other-marks reference")
        layer.setReadOnly(True)
        layer.setName("CLIENT SYMBOLS / OTHER PAINT - " + layer.name())
        layer.renderer().setSymbol(
            QgsFillSymbol.createSimple(
                {"color": "255,190,0,55", "outline_color": "255,190,0,255", "outline_width": "0.4"}
            )
        )
        node = project.layerTreeRoot().findLayer(layer.id())
        parent = node.parent()
        visible_node = node.clone()
        symbol_group.addChildNode(visible_node)
        parent.removeChildNode(node)
        visible_node.setItemVisibilityChecked(True)
        other_layers.append(layer)
        codes, types, tile_hits = {}, {}, {}
        for feature in layer.getFeatures():
            code = str(feature["COD_RST"]) if "COD_RST" in layer.fields().names() else "NO_CODE"
            codes[code] = codes.get(code, 0) + 1
            name = feature.geometry().wkbType()
            types[str(name)] = types.get(str(name), 0) + 1
        for tile in tiles:
            rectangle = QgsRectangle(*tile["bounds"])
            hits = []
            for feature in layer.getFeatures(QgsFeatureRequest().setFilterRect(rectangle)):
                if feature.geometry().intersects(rectangle):
                    hits.append({"fid": feature.id(), "code": str(feature["COD_RST"])})
            tile_hits[tile["tile_id"]] = hits
        results.append(
            {
                "layer": layer.name(),
                "features": layer.featureCount(),
                "code_counts": codes,
                "geometry_wkb_types": types,
                "tile_hits": tile_hits,
            }
        )
    if not other_layers:
        raise ValueError("No other-marks layers found")
    project.setTitle("Road marking review - lines AND client symbols (orange)")
    if not project.write(str(project_path)):
        raise ValueError("Could not save project")
    if not project.read(str(project_path)) or not all(
        x.isValid() for x in project.mapLayers().values()
    ):
        raise ValueError("Saved project failed reload validation")
    preview_dir = workspace / "quality_control/client_symbols"
    preview_dir.mkdir(parents=True)
    for tile in tiles:
        settings = QgsMapSettings()
        settings.setLayers(
            [node.layer() for node in project.layerTreeRoot().findLayers() if node.isVisible()]
        )
        settings.setExtent(QgsRectangle(*tile["bounds"]))
        settings.setOutputSize(QSize(1024, 1024))
        job = QgsMapRendererParallelJob(settings)
        job.start()
        job.waitForFinished()
        if not job.renderedImage().save(str(preview_dir / (tile["tile_id"] + ".png"))):
            raise ValueError("Preview save failed")
    report = {
        "project": str(project_path),
        "source_hash_checks": checks,
        "layers": results,
        "training_approval_changed": False,
        "original_workspace_modified": False,
        "note": (
            "Orange outlines show original client geometry, "
            "not AI predictions or newly approved masks."
        ),
    }
    path = reports / (stamp + "_other_marks.json")
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    project.clear()
    app.exitQgis()


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1])
