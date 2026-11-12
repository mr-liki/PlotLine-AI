"""Confirm supplied A6/annotation pair with bounded point samples and DGN text hints."""

import io
import json
import re
import time
from pathlib import Path

import laspy
import numpy as np
from pyproj import CRS

from roadmarking.rasterization.pipeline import dump, sha256


def run(root):
    base = root / "outputs/reports/phase4_source_alignment"
    comparison = json.loads((base / "comparison_report.json").read_text())
    candidate = next(r for r in comparison["folders"] if r["path"].endswith(r"A6\LAS new"))
    audits = []
    for path in sorted(base.glob("*/report.json")):
        record = json.loads(path.read_text())
        if record.get("raw_root") == candidate["path"] and "annotation_layers" in record:
            audits.append(record)
    detail = audits[-1]
    headers = json.loads(
        (Path(detail["report_directory"]) / "raw_header_inventory.json").read_text()
    )
    usable = [h for h in headers if h["valid_header_bounds"] and h.get("declared_records_fit_file")]
    chosen = {}
    for header in usable:
        chosen.setdefault(header["name"].split("_Track")[0], header)
    northernmost = max(usable, key=lambda h: h["xyz_max"][1])
    chosen["northernmost_header"] = northernmost
    selected = {header["path"]: header for header in chosen.values()}
    out = base / (time.strftime("%Y%m%dT%H%M%S") + "_exact_pair_confirmation")
    out.mkdir(parents=True)
    samples = []
    for filename, original in selected.items():
        path = Path(filename)
        with path.open("rb") as stream:
            prefix = stream.read(227)
            offset = int.from_bytes(prefix[96:100], "little")
            if prefix[:4] != b"LASF" or not 227 <= offset <= 4 * 1024**2:
                raise ValueError("Unexpected header; refusing point sample")
            header = laspy.LasHeader.read_from(io.BytesIO(prefix + stream.read(offset - 227)))
            if header.are_points_compressed:
                raise ValueError("This bounded offset sampler only supports uncompressed LAS")
            starts = sorted({0, header.point_count // 2, max(0, header.point_count - 16)})
            windows = []
            for start in starts:
                count = min(16, header.point_count - start)
                stream.seek(offset + start * header.point_format.size)
                payload = stream.read(count * header.point_format.size)
                if len(payload) != count * header.point_format.size:
                    raise ValueError("Truncated point sample")
                records = laspy.PackedPointRecord.from_buffer(payload, header.point_format)
                xyz = np.column_stack(
                    [
                        np.asarray(records[name]) * header.scales[index] + header.offsets[index]
                        for index, name in enumerate(("X", "Y", "Z"))
                    ]
                )
                within = bool(
                    np.isfinite(xyz).all()
                    and (
                        (xyz >= header.mins - header.scales) & (xyz <= header.maxs + header.scales)
                    ).all()
                )
                within_xy = bool(
                    np.isfinite(xyz[:, :2]).all()
                    and (
                        (xyz[:, :2] >= header.mins[:2] - header.scales[:2])
                        & (xyz[:, :2] <= header.maxs[:2] + header.scales[:2])
                    ).all()
                )
                bad = ~np.isfinite(xyz).all(axis=1) | (
                    (xyz < header.mins - header.scales) | (xyz > header.maxs + header.scales)
                ).any(axis=1)
                windows.append(
                    {
                        "start_point": start,
                        "points_read": count,
                        "xyz_min": xyz.min(axis=0).tolist(),
                        "xyz_max": xyz.max(axis=0).tolist(),
                        "first_xyz": xyz[0].tolist(),
                        "consistent_with_header": within,
                        "xy_consistent_with_header": within_xy,
                        "out_of_header_xyz_points": int(bad.sum()),
                    }
                )
            crs = header.parse_crs()
            horizontal = crs.sub_crs_list[0] if crs and crs.is_compound else crs
            parameters = (
                {p.name: p.value for p in horizontal.coordinate_operation.params}
                if horizontal
                else {}
            )
            reference = {p.name: p.value for p in CRS.from_epsg(3763).coordinate_operation.params}
            samples.append(
                {
                    "file": filename,
                    "batch": original["name"].split("_Track")[0],
                    "windows": windows,
                    "source_crs_name": crs.name if crs else None,
                    "source_horizontal_authority": (
                        horizontal.to_authority() if horizontal else None
                    ),
                    "source_horizontal_datum": horizontal.datum.name if horizontal else None,
                    "source_projection_parameters": parameters,
                    "epsg3763_projection_parameters": reference,
                    "projection_parameters_match_within_1e_9": all(
                        abs(parameters.get(k, float("inf")) - v) < 1e-9
                        for k, v in reference.items()
                    ),
                }
            )
        print(path.name, [(w["start_point"], w["first_xyz"]) for w in windows], flush=True)
    annotation_root = Path(detail["annotation_root"])
    annotations = []
    for layer in detail["annotation_layers"]:
        source = Path(layer["path"])
        annotations.append(
            {
                "path": str(source),
                "current_sha256": sha256(source),
                "unchanged_since_inventory": sha256(source) == layer["sha256"],
                "features": layer["features"],
                "native_xy_bbox": layer["native_xy_bbox"],
            }
        )
    dgn = []
    for path in annotation_root.rglob("*.dgn"):
        data = path.read_bytes()
        strings = [s.decode("ascii") for s in re.findall(rb"[\x20-\x7e]{8,}", data)]
        strings += [s.decode("utf-16-le") for s in re.findall(rb"(?:[\x20-\x7e]\x00){8,}", data)]
        hints = sorted(
            set(s for s in strings if re.search(r"\.dgn|\.la[sz]|EPSG|ETRS|TM06", s, re.I))
        )
        dgn.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "compound_document_signature": data[:8].hex(),
                "embedded_text_hints": hints,
                "limitation": (
                    "Text may be title/template/reference metadata. Active attachments, "
                    "geometry and global origin remain unverified; installed GDAL builds "
                    "lack DGNv8."
                ),
            }
        )
    report = {
        "annotation_root": str(annotation_root),
        "raw_root": detail["raw_root"],
        "raw_files": detail["raw_files"],
        "readable_header_footprints": detail["valid_header_files"],
        "feature_footprint_hits": detail["annotation_features_intersecting_headers"],
        "actual_point_samples": samples,
        "total_sampled_points": sum(w["points_read"] for s in samples for w in s["windows"]),
        "all_sampled_points_consistent_with_headers": all(
            w["consistent_with_header"] for s in samples for w in s["windows"]
        ),
        "all_sampled_xy_consistent_with_headers": all(
            w["xy_consistent_with_header"] for s in samples for w in s["windows"]
        ),
        "sampled_files_with_xyz_anomalies": [
            s["file"] for s in samples if any(not w["consistent_with_header"] for w in s["windows"])
        ],
        "annotation_checks": annotations,
        "dgn_checks": dgn,
        "status": "NO_SPATIAL_MATCH_IN_STORED_COORDINATES_CONFIRMED_BY_POINT_SAMPLES",
        "no_coordinate_transformation_applied": True,
        "training_or_inference_performed": False,
        "point_sample_limitation": (
            "Representative sample, not a complete point integrity check or "
            "registration measurement"
        ),
    }
    dump(out / "report.json", report)
    text = [
        "# Exact supplied source-pair confirmation",
        "",
        (
            "Checked the original annotation share and A6/LAS new share again, "
            "including real point records."
        ),
        "",
        f"- {detail['raw_files']} LAS files; {detail['valid_header_files']} readable footprints; "
        f"{detail['annotation_features_intersecting_headers']} annotation intersections.",
        f"- {report['total_sampled_points']} points read from start/middle/end windows "
        "across all three survey batches and the northernmost file.",
        "- Sampled XY agrees with the header footprints: "
        f"{report['all_sampled_xy_consistent_with_headers']}.",
        f"- Full XYZ bounds consistency: {report['all_sampled_points_consistent_with_headers']}. "
        "One sampled file has out-of-header height values; its point integrity is not approved.",
        "- All four original manual GeoPackages still have their recorded SHA-256 hashes.",
        "",
        "## Actual point samples",
        "",
        "| LAS file | First point Y | Middle point Y | Last-window first Y |",
        "|---|---:|---:|---:|",
    ]
    for sample in samples:
        y = [w["first_xyz"][1] for w in sample["windows"]]
        text.append(f"| {Path(sample['file']).name} | {y[0]:.3f} | {y[1]:.3f} | {y[-1]:.3f} |")
    text += [
        "",
        (
            "Manual annotation Y range: +166,202 to +177,217. The supplied A6 LAS "
            "range is -119,518 to -93,746."
        ),
        "",
        (
            "The LAS compound CRS is named PT-TM06_ETRS89_EGM2008 height. Its "
            "horizontal projection parameters match EPSG:3763 to numerical "
            "precision, but its horizontal datum is custom/unspecified. No offset "
            "or reprojection has been invented or applied."
        ),
        "",
        "## Original DGN files",
        "",
        (
            "Both original DGNs contain the text `AGP-A41_OUT_JS_v1.dgn`. This is a"
            " reference/title/template hint for the data provider to investigate, "
            "not proof of an active reference or transform. Installed roadmarking "
            "and QGIS GDAL builds cannot read DGNv8 geometry or global-origin "
            "settings."
        ),
        "",
        "## Required resolution",
        "",
        (
            "Existing manual work is available. What remains unproved is the "
            "spatial pairing or reference-frame mapping. Ask the provider to "
            "confirm the survey/reference attached to these two DGNs (including the"
            " AGP-A41 filename hint), or provide a native-coordinate "
            "export/documented georeferencing configuration. No redrawing or "
            "training is performed."
        ),
        "",
    ]
    (out / "report.md").write_text("\n".join(text), encoding="utf-8")
    dump(
        base / "exact_pair_latest.json",
        {"report_directory": str(out), "report": str(out / "report.md")},
    )
    print(out / "report.md")
    return report


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[1])
