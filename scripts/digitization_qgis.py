"""Build and validate the QGIS review project for one digitization run (run with python-qgis).

Usage: python-qgis.bat scripts/digitization_qgis.py <run_dir>
All layers are added read-only except the review copy; nothing is approved automatically.
"""

import json
import sys
from pathlib import Path


def main(run):
    from qgis.core import (
        Qgis,
        QgsApplication,
        QgsBookmark,
        QgsColorRampShader,
        QgsCoordinateReferenceSystem,
        QgsFillSymbol,
        QgsLineSymbol,
        QgsMarkerSymbol,
        QgsProject,
        QgsRasterLayer,
        QgsRasterShader,
        QgsRectangle,
        QgsReferencedRectangle,
        QgsSingleBandPseudoColorRenderer,
        QgsVectorLayer,
    )
    from qgis.PyQt.QtGui import QColor

    run = Path(run)
    app = QgsApplication([], False)
    app.initQgis()
    project = QgsProject.instance()
    crs = QgsCoordinateReferenceSystem("EPSG:3763")
    project.setCrs(crs)
    project.setFilePathStorage(Qgis.FilePathType.Relative)
    project.setTitle(f"Road-marking digitization review - {run.name} - PENDING HUMAN REVIEW")
    tree = project.layerTreeRoot()
    g_final = tree.addGroup("Final 3D features (solid red, dash blue)")
    g_review = tree.addGroup("Review: low confidence / uncertain / flags")
    g_line = tree.addGroup("Centerline stages (raw skeleton, refined 2D)")
    g_ai = tree.addGroup("Accepted AI masks (solid / dash)")
    g_ref = tree.addGroup("Client reference lines (read-only)")
    g_ras = tree.addGroup("LiDAR intensity and observed support")
    eng = run / "geopackage/digitization_engineering.gpkg"
    added = {}

    def vec(path, layer, title, group, color, width=0.6, style="solid", readonly=True):
        lyr = QgsVectorLayer(f"{path}|layername={layer}", title, "ogr")
        if not lyr.isValid():
            return None
        lyr.setReadOnly(readonly)
        sym = (
            QgsLineSymbol.createSimple({"color": color, "width": str(width), "line_style": style})
            if lyr.geometryType() == Qgis.GeometryType.Line
            else QgsMarkerSymbol.createSimple({"color": color, "size": "1.2"})
        )
        if lyr.geometryType() == Qgis.GeometryType.Polygon:
            sym = QgsFillSymbol.createSimple({"style": "no", "outline_color": color})
        if lyr.renderer() is not None:  # empty layers have no renderer
            lyr.renderer().setSymbol(sym)
        project.addMapLayer(lyr, False)
        group.addLayer(lyr)
        added[title] = lyr
        return lyr

    def ras(path, title, group, colors=None):
        lyr = QgsRasterLayer(str(path), title)
        if not lyr.isValid():
            return None
        if colors:
            shader = QgsRasterShader()
            ramp = QgsColorRampShader()
            ramp.setColorRampType(QgsColorRampShader.Exact)
            ramp.setColorRampItemList(
                [QgsColorRampShader.ColorRampItem(v, QColor(c), lbl) for v, c, lbl in colors]
            )
            shader.setRasterShaderFunction(ramp)
            lyr.setRenderer(QgsSingleBandPseudoColorRenderer(lyr.dataProvider(), 1, shader))
        project.addMapLayer(lyr, False)
        group.addLayer(lyr)
        added[title] = lyr
        return lyr

    vec(eng, "other_markings_3d", "Other markings (3D)", g_final, "80,220,120")
    vec(eng, "other_markings_review_2d", "Other markings (2D)", g_review, "80,220,120")
    vec(eng, "solid_lines_3d", "Final 3D solid lines", g_final, "230,30,30", 0.7)
    vec(eng, "dash_lines_3d", "Final 3D dash lines", g_final, "30,110,255", 0.7)
    vec(
        eng,
        "review_required_2d",
        "Review required (flags / missing Z)",
        g_review,
        "255,140,0",
        1.0,
        "solid",
    )
    vec(
        eng,
        "continuity_uncertain",
        "Uncertain continuity across gaps (NOT asserted)",
        g_review,
        "255,0,255",
        0.5,
        "dot",
    )
    vec(
        eng,
        "inferred_occlusion_bridges",
        "INFERRED joins under vehicles (review)",
        g_review,
        "255,170,0",
        1.2,
        "dash",
    )
    vec(
        eng,
        "continuity_paint_evidence",
        "Joins supported by observed paint returns",
        g_review,
        "0,255,255",
        0.6,
        "dash",
    )
    vec(eng, "vertex_z_support", "Vertex Z support (attributes)", g_review, "255,255,0")
    vec(eng, "centerlines_refined_2d", "Refined 2D centerlines", g_line, "0,200,0", 0.4)
    vec(
        eng,
        "centerlines_raw_skeleton_2d",
        "Raw skeleton centerlines",
        g_line,
        "120,120,120",
        0.3,
        "dot",
    )
    # Client reference: the run's own clipped copy only (QGIS never opens data/raw sources).
    vec(
        eng,
        "client_reference_lines_READONLY_COPY",
        "Client reference lines (copy, read-only)",
        g_ref,
        "0,0,0",
        0.3,
        "dash",
    )
    ras(
        run / "rasters/ai_solid_mask.tif",
        "AI accepted solid mask",
        g_ai,
        [(0, "#00000000", "none"), (1, "#e61e1e", "solid")],
    )
    ras(
        run / "rasters/ai_dash_mask.tif",
        "AI accepted dash mask",
        g_ai,
        [(0, "#00000000", "none"), (1, "#1e6eff", "dash")],
    )
    ras(
        run / "rasters/observed_support.tif",
        "Observed support (green) / NoData",
        g_ras,
        [(0, "#00000000", "NoData"), (1, "#3000aa00", "observed")],
    )
    if (run / "rasters/intensity_bridge_deck.tif").exists():
        deck = ras(
            run / "rasters/intensity_bridge_deck.tif", "LiDAR intensity on bridge decks", g_ras
        )
        if deck is not None:  # zero = not a deck: transparent, the road below stays visible
            deck.dataProvider().setNoDataValue(1, 0)
    ras(run / "rasters/intensity.tif", "LiDAR intensity (production grayscale)", g_ras)
    for t in (
        "AI accepted solid mask",
        "AI accepted dash mask",
        "Observed support (green) / NoData",
    ):
        node = tree.findLayer(added[t].id()) if t in added else None
        if node:
            node.setItemVisibilityChecked(False)
    g_line.setItemVisibilityChecked(False)
    bm = json.loads((run / "qgis_review/bookmarks.json").read_text())
    for b in bm:
        x0, y0, x1, y1 = b["extent"]
        mark = QgsBookmark()
        mark.setName(b["name"])
        mark.setExtent(QgsReferencedRectangle(QgsRectangle(x0, y0, x1, y1), crs))
        project.bookmarkManager().addBookmark(mark)
    out = run / "qgis_review/digitization_review.qgs"
    ok = project.write(str(out))
    project.clear()
    ok2 = project.read(str(out))
    layers = {
        lyr.name(): dict(
            valid=lyr.isValid(),
            features=(lyr.featureCount() if hasattr(lyr, "featureCount") else None),
        )
        for lyr in project.mapLayers().values()
    }
    res = dict(
        project=str(out),
        written=bool(ok),
        reopened=bool(ok2),
        layers=layers,
        bookmarks=len(project.bookmarkManager().bookmarks()),
        all_valid=all(v["valid"] for v in layers.values()),
    )
    (run / "qgis_review/qgis_validation.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    project.clear()
    app.exitQgis()


if __name__ == "__main__":
    main(sys.argv[1])
