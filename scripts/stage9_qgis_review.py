"""Headless QGIS review package for NEW role-reserved review candidates (run with python-qgis).

Usage: python-qgis.bat phase5_11_qgis_review.py [phase5_11|phase5_12]

All vector layers are read-only. Nothing here approves anything: every region stays
PENDING_FIRST_APPROVAL and training_eligible=0 until the user confirms it explicitly.
"""

import json
from pathlib import Path


def main(phase="stage9"):
    from qgis.core import (
        Qgis,
        QgsApplication,
        QgsBookmark,
        QgsCategorizedSymbolRenderer,
        QgsCoordinateReferenceSystem,
        QgsFillSymbol,
        QgsLineSymbol,
        QgsMapRendererParallelJob,
        QgsMapSettings,
        QgsProject,
        QgsRasterLayer,
        QgsRectangle,
        QgsReferencedRectangle,
        QgsRendererCategory,
        QgsVectorLayer,
    )
    from qgis.PyQt.QtCore import QSize

    root = Path(__file__).resolve().parents[1]
    latest = json.loads((root / "outputs/reports" / phase / "latest.json").read_text())
    label = phase.replace("phase", "Phase ").replace("_", ".")
    report = Path(latest["report_dir"])
    spec = json.loads((report / "new_region_candidates.json").read_text())
    output = Path(spec["output"])
    app = QgsApplication([], False)
    app.initQgis()
    project = QgsProject.instance()
    crs = QgsCoordinateReferenceSystem()
    project.setCrs(crs)
    project.setFilePathStorage(Qgis.FilePathType.Relative)
    project.setTitle(
        f"{label} NEW role-reserved review candidates - first approval pending, NOT labels"
    )
    tree = project.layerTreeRoot()
    groups = {
        "regions": tree.addGroup("Review region footprints (yellow)"),
        "paint": tree.addGroup("Candidate painted outlines: red solid / blue dash - PENDING"),
        "unsupported": tree.addGroup("Unsupported marking regions (magenta) - excluded"),
        "client": tree.addGroup("Read-only client centerlines / symbols"),
        "images": tree.addGroup("Actual LiDAR intensity - native 0.05 grid"),
        "masks": tree.addGroup("Observation masks (toggle): observed / NoData"),
    }

    def vector(layer_name, title):
        layer = QgsVectorLayer(f"{spec['candidate_store']}|layername={layer_name}", title, "ogr")
        layer.setCrs(crs)
        layer.setReadOnly(True)
        return layer

    paint = vector("candidate_paint_footprints", "Candidate painted footprints")
    categories = []
    for value, colour, label in ((1, "230,30,30", "solid"), (2, "30,110,255", "dash")):
        symbol = QgsFillSymbol.createSimple(
            {"color": "0,0,0,0", "outline_color": colour, "outline_width": "0.35"}
        )
        categories.append(QgsRendererCategory(value, symbol, label))
    paint.setRenderer(QgsCategorizedSymbolRenderer("visual_id", categories))
    project.addMapLayer(paint, False)
    groups["paint"].addLayer(paint)
    unsupported = vector("unsupported_marking_regions", "Unsupported symbols / hatching")
    if unsupported.isValid():
        unsupported.renderer().setSymbol(
            QgsFillSymbol.createSimple(
                {"color": "255,0,255,50", "outline_color": "255,0,255", "outline_width": "0.3"}
            )
        )
        project.addMapLayer(unsupported, False)
        groups["unsupported"].addLayer(unsupported)
    regions = vector("review_region_footprints", "Review region footprints")
    regions.renderer().setSymbol(
        QgsFillSymbol.createSimple(
            {"color": "0,0,0,0", "outline_color": "255,220,0", "outline_width": "0.6"}
        )
    )
    project.addMapLayer(regions, False)
    groups["regions"].addLayer(regions)
    ignore = vector("review_ignore_zones", "User-reported missed paint - IGNORE zones")
    if ignore.isValid():
        ignore.renderer().setSymbol(
            QgsFillSymbol.createSimple(
                {"color": "255,140,0,60", "outline_color": "255,140,0", "outline_width": "0.5"}
            )
        )
        project.addMapLayer(ignore, False)
        groups["regions"].addLayer(ignore)
    editable_names = []
    if spec.get("review_edits_store"):
        group = tree.insertGroup(0, "EDITABLE reviewer layers (only these may be edited)")
        for layer_name, title, style in (
            (
                "reviewer_outline_check",
                "Reviewer: per-outline decision (edit attributes)",
                {"color": "0,0,0,0", "outline_color": "0,255,120", "outline_width": "0.25"},
            ),
            (
                "reviewer_missing_paint",
                "Reviewer: draw MISSING paint / problem areas here",
                {"color": "255,140,0,70", "outline_color": "255,140,0", "outline_width": "0.5"},
            ),
        ):
            layer = QgsVectorLayer(
                f"{spec['review_edits_store']}|layername={layer_name}", title, "ogr"
            )
            layer.setCrs(crs)
            layer.renderer().setSymbol(QgsFillSymbol.createSimple(style))
            project.addMapLayer(layer, False)
            group.addLayer(layer).setItemVisibilityChecked(layer_name == "reviewer_missing_paint")
            editable_names.append(layer.name())
    for item in spec["references"]:
        layer = QgsVectorLayer(
            item["path"] + "|layername=" + item["layer"], "Client " + item["layer"], "ogr"
        )
        layer.setCrs(crs)
        layer.setReadOnly(True)
        polygon = "other" in item["layer"].lower()
        layer.renderer().setSymbol(
            QgsFillSymbol.createSimple(
                {"color": "0,0,0,0", "outline_color": "255,0,255", "outline_width": "0.2"}
            )
            if polygon
            else QgsLineSymbol.createSimple({"line_color": "0,255,255", "line_width": "0.15"})
        )
        project.addMapLayer(layer, False)
        groups["client"].addLayer(layer).setItemVisibilityChecked(False)
    for tile in spec["tiles"]:
        for filename, group, visible in (
            (tile["images"]["grayscale"], "images", True),
            (tile["layers"]["observed"], "masks", False),
            (tile["layers"]["nodata"], "masks", False),
        ):
            layer = QgsRasterLayer(filename, tile["candidate_id"] + " " + Path(filename).stem)
            layer.setCrs(crs)
            project.addMapLayer(layer, False)
            groups[group].addLayer(layer).setItemVisibilityChecked(visible)
        bookmark = QgsBookmark()
        bookmark.setName(
            f"{tile['candidate_id']} - {tile['proposed_role']} ({tile['priority_tag']})"
        )
        bookmark.setGroup(f"{label} new training/development candidates")
        bookmark.setExtent(QgsReferencedRectangle(QgsRectangle(*tile["bounds"]), crs))
        project.bookmarkManager().addBookmark(bookmark)
    project.viewSettings().setDefaultViewExtent(
        QgsReferencedRectangle(QgsRectangle(*spec["tiles"][0]["bounds"]), crs)
    )
    filename = output / f"{phase}_role_review.qgs"
    if not project.write(str(filename)):
        raise ValueError("QGIS project save failed")
    project.clear()
    if not project.read(str(filename)):
        raise ValueError("Saved project reload failed")
    invalid = [layer.name() for layer in project.mapLayers().values() if not layer.isValid()]
    if invalid:
        raise ValueError(f"Invalid layers after reload: {invalid}")
    previews = output / "review_previews"
    previews.mkdir(exist_ok=True)
    for tile in spec["tiles"]:
        settings = QgsMapSettings()
        settings.setLayers(
            [n.layer() for n in project.layerTreeRoot().findLayers() if n.isVisible()]
        )
        settings.setExtent(QgsRectangle(*tile["bounds"]))
        settings.setOutputSize(QSize(1024, 1024))
        job = QgsMapRendererParallelJob(settings)
        job.start()
        job.waitForFinished()
        if not job.renderedImage().save(str(previews / (tile["candidate_id"] + ".png"))):
            raise ValueError("Review preview render failed")
    validation = {
        "project": str(filename),
        "project_reopened": True,
        "all_layers_valid": True,
        "layer_count": len(project.mapLayers()),
        "bookmark_count": len(project.bookmarkManager().bookmarks()),
        "all_client_and_candidate_layers_read_only": all(
            layer.readOnly()
            for layer in project.mapLayers().values()
            if isinstance(layer, QgsVectorLayer) and layer.name() not in editable_names
        ),
        "editable_reviewer_layers": [
            layer.name()
            for layer in project.mapLayers().values()
            if isinstance(layer, QgsVectorLayer) and not layer.readOnly()
        ],
        "previews": str(previews),
        "new_regions_training_eligible": False,
        "existing_approvals_untouched": True,
    }
    (report / "qgis_review_package.json").write_text(json.dumps(validation, indent=2))
    print(json.dumps(validation, indent=2), flush=True)
    project.clear()
    app.exitQgis()


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "stage9")
