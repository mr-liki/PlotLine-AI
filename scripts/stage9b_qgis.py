"""One focused text/omission review project; existing geometry remains read-only."""

import json
import shutil
from pathlib import Path


def main():
    from qgis.core import (
        Qgis,
        QgsApplication,
        QgsBookmark,
        QgsCategorizedSymbolRenderer,
        QgsCoordinateReferenceSystem,
        QgsFillSymbol,
        QgsMapRendererParallelJob,
        QgsMapSettings,
        QgsProject,
        QgsRasterLayer,
        QgsRectangle,
        QgsReferencedRectangle,
        QgsRendererCategory,
        QgsSingleSymbolRenderer,
        QgsVectorLayer,
    )
    from qgis.PyQt.QtCore import QSize

    root = Path(__file__).resolve().parents[1]
    latest = json.loads((root / "outputs/reports/stage9b/latest.json").read_text())
    out = Path(latest["report_dir"])
    spec = json.loads((out / "review_spec.json").read_text())
    data = Path(spec["output"])
    app = QgsApplication([], False)
    app.initQgis()
    project = QgsProject.instance()
    crs = QgsCoordinateReferenceSystem()
    project.setCrs(crs)
    project.setFilePathStorage(Qgis.FilePathType.Relative)
    project.setTitle("Stage 9B - NEW road text and focused omissions - human confirmation required")
    tree = project.layerTreeRoot()
    edits = tree.addGroup("EDIT: new character geometry, values and missing targets only")
    drafts = tree.addGroup("DRAFT: observed text candidates - not approved labels")
    existing = tree.addGroup("READ ONLY: original Stage 9 solid/dash and excluded symbols")
    imagery = tree.addGroup("Source grayscale - unchanged native grid")
    masks = tree.addGroup("Observed / NoData masks (toggle)")

    def add_vector(path, layer, name, group, color, editable=False, visible=True):
        value = QgsVectorLayer(str(path) + "|layername=" + layer, name, "ogr")
        assert value.isValid(), name
        value.setCrs(crs)
        value.setReadOnly(not editable)
        value.setRenderer(
            QgsSingleSymbolRenderer(
                QgsFillSymbol.createSimple(
                    {"color": "0,0,0,0", "outline_color": color, "outline_width": ".22"}
                )
            )
        )
        project.addMapLayer(value, False)
        group.addLayer(value).setItemVisibilityChecked(visible)
        return value

    paint = add_vector(
        spec["base_stage9_candidates"],
        "candidate_paint_footprints",
        "Original solid/dash candidates - geometry unchanged",
        existing,
        "255,0,0",
    )
    categories = []
    for visual, color, name in ((1, "230,30,30", "solid"), (2, "30,110,255", "dash")):
        symbol = QgsFillSymbol.createSimple(
            {"color": "0,0,0,0", "outline_color": color, "outline_width": ".35"}
        )
        categories.append(QgsRendererCategory(visual, symbol, name))
    paint.setRenderer(QgsCategorizedSymbolRenderer("visual_id", categories))
    add_vector(
        spec["base_stage9_candidates"],
        "unsupported_marking_regions",
        "Arrows / unsupported symbols - not road_text",
        existing,
        "255,0,255",
    )
    add_vector(
        spec["text_candidates"],
        "character_candidates",
        "CYAN: proposed character splits (values unknown)",
        drafts,
        "0,255,255",
    )
    add_vector(
        spec["text_candidates"],
        "group_and_unknown_fragments",
        "ORANGE: partial/faint text or unclassified features - partition/class review",
        drafts,
        "255,150,0",
    )
    add_vector(
        spec["text_candidates"],
        "review_uncertainty_extents",
        "YELLOW: unknown extents excluded from background",
        drafts,
        "255,220,0",
        visible=False,
    )
    add_vector(
        spec["text_candidates"],
        "reviewer_characters",
        "EDIT: confirm/correct character geometry and human-readable values",
        edits,
        "0,255,255",
        editable=True,
        visible=False,
    )
    add_vector(
        spec["text_candidates"],
        "reviewer_missing_paint",
        "EDIT: missing paint / uncertain classification (no automatic labels)",
        edits,
        "255,150,0",
        editable=True,
    )
    for t in spec["tiles"]:
        for path, target_group, visible in (
            (t["images"]["grayscale"], imagery, True),
            (t["layers"]["observed"], masks, False),
            (t["layers"]["nodata"], masks, False),
        ):
            source = Path(path)
            if target_group == masks:
                copy = data / "source_masks" / t["candidate_id"] / source.name
                copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, copy)
                path = str(copy)
            value = QgsRasterLayer(str(path), t["candidate_id"] + " " + Path(path).stem)
            value.setCrs(crs)
            assert value.isValid()
            project.addMapLayer(value, False)
            target_group.addLayer(value).setItemVisibilityChecked(visible)
        bookmark = QgsBookmark()
        bookmark.setName(
            t["candidate_id"] + " - " + t["proposed_role"] + " - target-completeness audit"
        )
        bookmark.setGroup("Stage 9B four-window review")
        bookmark.setExtent(QgsReferencedRectangle(QgsRectangle(*t["bounds"]), crs))
        project.bookmarkManager().addBookmark(bookmark)
    t = next(t for t in spec["tiles"] if t["candidate_id"] == "N9_01")
    a = t["affine"]
    extent = QgsRectangle(a[2], a[5] - 280 * 0.05, a[2] + 240 * 0.05, a[5])
    bookmark = QgsBookmark()
    bookmark.setName("N9_01 - FOCUSED TEXT CORRECTION - values/splits unconfirmed")
    bookmark.setGroup("Stage 9B four-window review")
    bookmark.setExtent(QgsReferencedRectangle(extent, crs))
    project.bookmarkManager().addBookmark(bookmark)
    project.viewSettings().setDefaultViewExtent(QgsReferencedRectangle(extent, crs))
    output = data / "stage9b_road_text_review.qgs"
    assert project.write(str(output))
    project.clear()
    assert project.read(str(output))
    assert all(layer.isValid() for layer in project.mapLayers().values())
    previews = data / "review_previews"
    previews.mkdir(exist_ok=True)
    for t in spec["tiles"]:
        settings = QgsMapSettings()
        settings.setLayers(
            [n.layer() for n in project.layerTreeRoot().findLayers() if n.isVisible()]
        )
        settings.setExtent(QgsRectangle(*t["bounds"]))
        settings.setOutputSize(QSize(1024, 1024))
        job = QgsMapRendererParallelJob(settings)
        job.start()
        job.waitForFinished()
        assert job.renderedImage().save(str(previews / f"{t['candidate_id']}.png"))
    result = {
        "project": str(output),
        "project_reopened": True,
        "all_layers_valid": True,
        "layer_count": len(project.mapLayers()),
        "bookmarks": len(project.bookmarkManager().bookmarks()),
        "geometry_or_character_approval_recorded": False,
        "existing_base_layers_read_only": all(
            layer.readOnly()
            for layer in project.mapLayers().values()
            if isinstance(layer, QgsVectorLayer) and not layer.name().startswith("EDIT:")
        ),
        "new_text_layers": "Character splits / group fragments / unknowns remain proposals; editable review layers are separate.",  # noqa: E501
        "T7_S8_included_for_review": False,
    }
    (out / "qgis_validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    project.clear()
    app.exitQgis()


if __name__ == "__main__":
    main()
