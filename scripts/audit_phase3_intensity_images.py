"""Independent persisted PNG/data checks; no image mutation or vector rendering."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

from roadmarking.analysis.corridor_pilot import read_json
from roadmarking.preprocessing.intensity import IntensityCalibration
from roadmarking.rasterization.pipeline import dump, sha256
from roadmarking.visualization.intensity_pipeline import NAMESPACE, write_report
from roadmarking.visualization.intensity_render import grayscale_intensity


def audit(root):
    directory = root / "outputs/reports" / NAMESPACE
    report = read_json(directory / "report.json")
    out = Path(report["output_root"])
    completion = read_json(out / "metadata/completion.json")
    if not all(sha256(p) == digest for p, digest in completion["artifacts"].items()):
        raise ValueError("Delivered image artifact hash mismatch")
    checked = []
    for tile in report["tiles"]:
        calibration = IntensityCalibration(**tile["normalization"])
        for product, paths, affine, images in (
            (
                "road",
                tile["rasters"],
                tile["affine"],
                [
                    (tile["images"]["preview"], False),
                    (tile["images"]["model_input"], False),
                    (tile["images"]["model_input_rgb"], True),
                ],
            ),
            (
                "context",
                tile["context_rasters"],
                tile["context_affine"],
                [(tile["images"]["context_preview"], False)],
            ),
        ):
            layers = {}
            for key, path in paths.items():
                with rasterio.open(path) as ds:
                    if list(ds.transform)[:6] != affine or ds.crs is not None:
                        raise ValueError("Unexpected georeferencing change")
                    layers[key] = ds.read(1)
            observed = layers["observed"] > 0
            if not np.array_equal(observed, layers["count"] > 0) or not np.array_equal(
                observed, layers["maximum"] >= 0
            ):
                raise ValueError("Observed/missingness disagreement")
            if not np.all(layers["normalized"][~observed] == -9999) or not np.all(
                layers["mean"][~observed] == -9999
            ):
                raise ValueError("Internal NoData was filled or overwritten")
            if not np.array_equal(
                layers["normalized"][observed], calibration.apply(layers["maximum"])[observed]
            ):
                raise ValueError("Normalization differs from fixed file calibration")
            expected = grayscale_intensity(
                layers["normalized"], observed, gamma=report["rendering"]["display_gamma"]
            )
            for path, rgb in images:
                with Image.open(path) as image:
                    pixels = np.asarray(image)
                    target = np.repeat(expected[:, :, None], 3, axis=2) if rgb else expected
                    if image.mode != ("RGB" if rgb else "L") or not np.array_equal(pixels, target):
                        raise ValueError("PNG is not the exact clean intensity transfer")
                checked.append(
                    {
                        "path": path,
                        "product": product,
                        "mode": "RGB" if rgb else "L",
                        "shape": list(pixels.shape),
                        "exact_raster_transfer": True,
                        "colored_pixels": 0,
                        "burned_annotations": False,
                    }
                )
    for mosaic in report["mosaics"]:
        with rasterio.open(mosaic["rasters"]["normalized"]) as ds:
            normalized = ds.read(1)
        with rasterio.open(mosaic["rasters"]["observed"]) as ds:
            observed = ds.read(1) > 0
        with Image.open(mosaic["image"]) as image:
            if image.mode != "L" or not np.array_equal(
                np.asarray(image),
                grayscale_intensity(
                    normalized, observed, gamma=report["rendering"]["display_gamma"]
                ),
            ):
                raise ValueError("Mosaic PNG contains a seam/transfer/overlay difference")
        checked.append(
            {
                "path": mosaic["image"],
                "mode": "L",
                "shape": mosaic["shape"],
                "exact_raster_transfer": True,
                "colored_pixels": 0,
                "burned_annotations": False,
            }
        )
    forbidden = [
        str(p)
        for p in out.rglob("*")
        if p.suffix.lower() in (".gpkg", ".shp", ".geojson", ".svg", ".dxf")
    ]
    if forbidden:
        raise ValueError("Debug vectors mixed into final intensity products")
    changed = [p for p, digest in report["protected_hashes"].items() if sha256(p) != digest]
    if changed:
        raise ValueError("Original source/previous protected output changed")
    suites = list(ET.parse(directory / "pytest.xml").getroot().iter("testsuite"))
    tests = {
        key: sum(int(s.attrib.get(key, 0)) for s in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    tests["passed"] = tests["tests"] - tests["failures"] - tests["errors"] - tests["skipped"]
    health = read_json(directory / "healthcheck.json")
    validation = {
        "pytest": tests,
        "healthcheck": {
            status: sum(c["status"] == status for c in health)
            for status in ("pass", "skip", "fail")
        },
        "ruff": (directory / "ruff.log").read_text().strip(),
        "black": (directory / "black.log").read_text().strip(),
    }
    visual_path = directory / "visual_qa.json"
    visual = read_json(visual_path) if visual_path.exists() else None
    report["validation"] = validation
    report["visual_qa"] = visual
    report["status"] = (
        "PASSED"
        if visual
        and visual.get("passed")
        and not tests["failures"]
        and not tests["errors"]
        and not validation["healthcheck"]["fail"]
        and validation["ruff"] == "All checks passed!"
        and "would be left unchanged" in validation["black"]
        and "would reformat" not in validation["black"]
        else "PROVISIONAL"
    )
    report["acceptance_scope"] = (
        "Clean image export only; no claim of training-dataset or road-geometry readiness"
    )
    report["image_audit"] = {
        "pngs_checked": len(checked),
        "all_exact_raster_transfer": True,
        "colored_pixels": 0,
        "vector_products_in_primary_folders": forbidden,
        "protected_hash_mismatches": changed,
    }
    dump(directory / "image_audit.json", {"images": checked, **report["image_audit"]})
    dump(directory / "validation.json", validation)
    dump(directory / "report.json", report)
    completion["report"] = report
    dump(out / "metadata/completion.json", completion)
    write_report(root, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "counts": report["counts"],
                "validation": validation,
                "image_audit": report["image_audit"],
            }
        )
    )


if __name__ == "__main__":
    audit(Path(__file__).resolve().parents[1])
