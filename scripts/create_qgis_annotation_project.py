"""Run with the installed QGIS Python, headlessly; never guesses a project CRS."""

import json
import sys
from pathlib import Path


def main(spec_path):
    from qgis.core import (
        Qgis,
        QgsApplication,
        QgsBookmark,
        QgsCoordinateReferenceSystem,
        QgsDefaultValue,
        QgsEditorWidgetSetup,
        QgsFillSymbol,
        QgsLineSymbol,
        QgsMapRendererParallelJob,
        QgsMapSettings,
        QgsProject,
        QgsRasterLayer,
        QgsRectangle,
        QgsReferencedRectangle,
        QgsVectorLayer,
    )
    from qgis.PyQt.QtCore import QSize

    app = QgsApplication([], False)
    app.initQgis()
    spec = json.loads(Path(spec_path).read_text())
    output = Path(spec["output_root"])
    project = QgsProject.instance()
    project.setCrs(QgsCoordinateReferenceSystem())
    project.setTitle("Road-marking annotation â€” NATIVE XY / CRS UNCONFIRMED")
    project.setFilePathStorage(Qgis.FilePathType.Relative)
    tree = project.layerTreeRoot()
    annotations = tree.addGroup("EDIT HERE â€” human annotations; verification via review CLI")
    images = tree.addGroup("Clean intensity images â€” native grid")
    precision = tree.addGroup("Full-precision source normalized GeoTIFF references")
    masks = tree.addGroup("Observed and valid masks â€” distinguish dark data / NoData / padding")
    planning = tree.addGroup("Planning footprints â€” NOT ground truth")
    references = tree.addGroup("Original client geometry - NOT verified painted-area masks")
    for item in spec.get("reference_layers", []):
        layer = QgsVectorLayer(item["path"] + "|layername=" + item["layer"], item["name"], "ogr")
        if not layer.isValid():
            raise RuntimeError("Invalid client reference layer")
        layer.setCrs(QgsCoordinateReferenceSystem())
        layer.setReadOnly(True)
        project.addMapLayer(layer, False)
        references.addLayer(layer).setItemVisibilityChecked(False)
    tiles = spec["tiles"]
    for tile in tiles:
        tid = tile["tile_id"]
        for filename, name, group, visible in (
            (tile["images"]["grayscale"], tid, images, True),
            (tile["layers"]["normalized"], f"{tid} full precision", precision, False),
            (tile["masks"]["observed"], f"{tid} observed", masks, False),
            (tile["masks"]["valid"], f"{tid} valid footprint", masks, False),
        ):
            layer = QgsRasterLayer(filename, name)
            if not layer.isValid():
                raise RuntimeError(f"Invalid raster: {filename}")
            layer.setCrs(QgsCoordinateReferenceSystem())
            extent = layer.extent()
            actual = (extent.xMinimum(), extent.yMinimum(), extent.xMaximum(), extent.yMaximum())
            if any(abs(a - b) > 1e-7 for a, b in zip(actual, tile["bounds"], strict=True)):
                raise RuntimeError("QGIS image georeferencing differs from native tile")
            project.addMapLayer(layer, False)
            group.addLayer(layer).setItemVisibilityChecked(visible)
        bookmark = QgsBookmark()
        bookmark.setName(tid)
        bookmark.setGroup(tile["source"])
        bookmark.setExtent(
            QgsReferencedRectangle(QgsRectangle(*tile["bounds"]), QgsCoordinateReferenceSystem())
        )
        project.bookmarkManager().addBookmark(bookmark)
    regions = QgsVectorLayer(
        spec["store"] + "|layername=annotation_regions", "annotation_regions", "ogr"
    )
    regions.renderer().setSymbol(
        QgsFillSymbol.createSimple(
            {"color": "0,0,0,0", "outline_color": "0,200,255", "outline_width": "0.2"}
        )
    )
    project.addMapLayer(regions, False)
    planning.addLayer(regions).setItemVisibilityChecked(False)
    statuses = (
        "DRAFT",
        "PROPOSED_UNREVIEWED",
        "NEEDS_CORRECTION",
        "VERIFIED",
        "REJECTED",
        "UNCERTAIN",
    )
    for layer_name in ("paint_polygons", "paint_centerlines", "review_areas"):
        layer = QgsVectorLayer(spec["store"] + "|layername=" + layer_name, layer_name, "ogr")
        if not layer.isValid():
            raise RuntimeError("Invalid annotation schema")
        layer.setCrs(QgsCoordinateReferenceSystem())
        defaults = {
            "annotation_id": "uuid()",
            "global_instance_id": "uuid()",
            "annotation_source": "'HUMAN_DIGITIZED'",
            "review_status": "'DRAFT'",
            "geometry_type": (
                "'MultiLineString'" if layer_name == "paint_centerlines" else "'MultiPolygon'"
            ),
            "purpose": "'UNCERTAIN'" if layer_name == "review_areas" else "'PAINT'",
            "width_verified": "0",
            "mask_reviewed": "0",
            "source_tile_id": "array_first(overlay_intersects('annotation_regions', \"tile_id\"))",
            "source_raster_id": '"source_tile_id"',
            "source_track": (
                "attribute(get_feature('annotation_regions','tile_id',\"source_tile_id\")"
                ",'source_track')"
            ),
            "source_coordinate_frame": "concat('NATIVE_XY_CRS_UNCONFIRMED:',\"source_track\")",
            "visual_class_name": (
                "CASE \"visual_class_id\" WHEN 1 THEN 'longitudinal_solid' WHEN 2 THEN "
                "'longitudinal_dash' END"
            ),
        }
        for field, expression in defaults.items():
            layer.setDefaultValueDefinition(
                layer.fields().indexFromName(field),
                QgsDefaultValue(
                    expression,
                    field
                    in {
                        "source_raster_id",
                        "source_track",
                        "source_coordinate_frame",
                        "visual_class_name",
                    },
                ),
            )
        maps = {
            "review_status": [{s: s} for s in statuses],
            "visual_class_id": [
                {c["name"]: c["id"]}
                for c in spec["taxonomy"]["visual_classes"]
                if c["enabled_for_annotation"]
            ],
            "source_tile_id": [{t["tile_id"]: t["tile_id"]} for t in tiles],
            "purpose": (
                [{s: s} for s in ("UNCERTAIN", "BACKGROUND", "COMPLETE_REGION")]
                if layer_name == "review_areas"
                else [{"PAINT": "PAINT"}]
            ),
            "width_verified": [{"Not verified": 0}, {"Verified in native units": 1}],
            "mask_reviewed": [{"Not checked": 0}, {"Buffered paint area checked": 1}],
        }
        for field, mapping in maps.items():
            layer.setEditorWidgetSetup(
                layer.fields().indexFromName(field),
                QgsEditorWidgetSetup("ValueMap", {"map": mapping}),
            )
        editor = layer.editFormConfig()
        for field in ("reviewer_id", "review_timestamp"):
            editor.setReadOnly(layer.fields().indexFromName(field), True)
        layer.setEditFormConfig(editor)
        if layer_name == "paint_centerlines":
            layer.renderer().setSymbol(
                QgsLineSymbol.createSimple({"line_color": "255,170,0", "line_width": "0.3"})
            )
        else:
            layer.renderer().setSymbol(
                QgsFillSymbol.createSimple(
                    {
                        "color": "0,0,0,0",
                        "outline_color": (
                            "255,0,255" if layer_name == "review_areas" else "0,255,255"
                        ),
                        "outline_width": "0.3",
                    }
                )
            )
        layer.setAbstract(
            "DRAFT is not ground truth. Source, class or geometry edits invalidate prior review. "
            "Native XY, no confirmed CRS. Use the review CLI after saving and ending editing."
        )
        project.addMapLayer(layer, False)
        annotations.addLayer(layer)
    project.setCrs(QgsCoordinateReferenceSystem())
    project.viewSettings().setDefaultViewExtent(
        QgsReferencedRectangle(QgsRectangle(*tiles[0]["bounds"]), QgsCoordinateReferenceSystem())
    )
    filename = output / "annotation_projects/road_marking_annotation.qgs"
    if not project.write(str(filename)):
        raise RuntimeError("QGIS project write failed")
    project.clear()
    if not project.read(str(filename)):
        raise RuntimeError("QGIS project reopen failed")
    validation = {
        "project_reopened": True,
        "project_crs_confirmed": project.crs().isValid(),
        "layers": [
            {
                "name": layer.name(),
                "valid": layer.isValid(),
                "crs_valid": layer.crs().isValid(),
                "workspace_owned_source": Path(layer.source().split("|")[0])
                .resolve()
                .is_relative_to(output.resolve()),
            }
            for layer in project.mapLayers().values()
        ],
        "bookmark_count": len(project.bookmarkManager().bookmarks()),
        "source_affine_extent_checks": "ALL_RASTERS_MATCH_NATIVE_GRID",
    }
    if validation["project_crs_confirmed"] or not all(
        layer["valid"] for layer in validation["layers"]
    ):
        raise RuntimeError("Invalid project or unverified CRS assigned")
    if not all(layer["workspace_owned_source"] for layer in validation["layers"]):
        raise RuntimeError("QGIS must not cache statistics into protected external inputs")
    (output / "metadata/qgis_validation.json").write_text(json.dumps(validation, indent=2))
    settings = QgsMapSettings()
    settings.setLayers(
        [node.layer() for node in project.layerTreeRoot().findLayers() if node.isVisible()]
    )
    settings.setExtent(QgsRectangle(*tiles[0]["bounds"]))
    settings.setOutputSize(QSize(1024, 1024))
    job = QgsMapRendererParallelJob(settings)
    job.start()
    job.waitForFinished()
    if not job.renderedImage().save(str(output / "quality_control/qgis_workspace_preview.png")):
        raise RuntimeError("QGIS preview render failed")
    print(filename)
    project.clear()
    app.exitQgis()


if __name__ == "__main__":
    main(sys.argv[1])
