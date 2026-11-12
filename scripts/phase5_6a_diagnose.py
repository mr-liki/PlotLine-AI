"""Controlled source-copy diagnostics; failed chunks never become admitted imagery."""

import json
import shutil
from pathlib import Path

import laspy
import numpy as np
from shapely.geometry import box

from roadmarking.io.pointcloud import ChunkReadReport, iter_laz_chunks, point_validity
from roadmarking.io.source_inventory import fingerprint, safe_header
from roadmarking.rasterization.pipeline import dump, sha256

root = Path(__file__).resolve().parents[1]
context = json.loads((root / "outputs/reports/phase5_6a/latest.json").read_text())
report = Path(context["report_dir"])
reserved = json.loads(
    (
        root / "outputs/reports/phase5_4/20261105T173740792650Z/"
        "future_test_region_reservations.json"
    ).read_text()
)["regions"]
inventory = json.loads(
    (root / "outputs/reports/phase5/20261105T172517_source_audit/" "raw_integrity.json").read_text()
)
results = []
for region in reserved:
    source = Path(region["raw_candidate"])
    expected = next(r for r in inventory if r["path"] == str(source))
    before = fingerprint(source)
    header = safe_header(source)
    row = {
        "region": region["reservation_id"],
        "source": str(source),
        "size_bytes": source.stat().st_size,
        "header_count": int(header.point_count),
        "expected_fingerprint": expected["fingerprint"],
        "observed_fingerprint": before,
    }
    if region is reserved[-1]:
        cache = root / "data/cache/phase5_6a_diagnostic" / context["run_id"] / source.name
        cache.parent.mkdir(parents=True, exist_ok=False)
        remote_hash = sha256(source)
        shutil.copyfile(source, cache)
        local_hash = sha256(cache)
        row.update(
            remote_sha256=remote_hash,
            local_sha256=local_hash,
            transport_copy_identical=remote_hash == local_hash,
            diagnostic_copy=str(cache),
        )
        if remote_hash != local_hash or before != fingerprint(source):
            raise ValueError("Source copy/version inconsistent")
        progress = ChunkReadReport(cache, 0, 0)
        for _ in iter_laz_chunks(cache, progress):
            pass
        row.update(
            decoded_admissible_points=progress.points_read,
            discarded_points=progress.dropped_points,
            bad_chunks=progress.bad_chunks,
            invalid_chunks=progress.corrupt_chunks,
            chunk_size=progress.chunk_size,
            whole_file_valid=not progress.dropped_points,
        )
        first_chunk, offset = progress.corrupt_chunks[0]
        first_record = first_chunk * progress.chunk_size + offset
        row["first_invalid_zero_based_record"] = first_record
        row["backend_availability"] = [str(b) for b in laspy.LazBackend.detect_available()]
        with laspy.open(cache, laz_backend=laspy.LazBackend.Lazrs) as reader:
            reader.seek(first_record)
            points = reader.read_points(1)
            row["first_invalid_values"] = {
                key: float(np.asarray(getattr(points, key))[0])
                for key in ("x", "y", "z", "return_number", "number_of_returns", "gps_time")
            }
            row["invalid_reproduced_on_local_copy"] = not bool(
                point_validity(
                    points,
                    header.mins,
                    header.maxs,
                    gps_week_time=not bool(header.global_encoding.gps_time_type),
                )[0]
            )
        row["cause"] = (
            "Reproducible local point-validity failure; not copy transport loss. "
            "Independent decoder confirmation pending."
        )
    else:
        row.update(
            decoded_admissible_points=int(header.point_count),
            discarded_points=0,
            whole_file_valid=True,
            cause="Passed prior and repeated full-point preparation",
        )
    results.append(row)
    dump(report / "file_integrity_findings.json", results)
alternatives = []
for region in reserved:
    footprint = box(*region["bounds"])
    candidates = []
    for item in inventory:
        if "mins" not in item or "maxs" not in item:
            continue
        source_box = box(item["mins"][0], item["mins"][1], item["maxs"][0], item["maxs"][1])
        if footprint.intersects(source_box):
            candidates.append(
                {
                    "source": item["path"],
                    "bbox_fraction": footprint.intersection(source_box).area / footprint.area,
                    "actual_coverage_validated": item["path"] == region["raw_candidate"]
                    and region != reserved[-1],
                }
            )
    alternatives.append(
        {
            "region": region["reservation_id"],
            "candidates": candidates,
            "bbox_overlap_is_not_proof_of_valid_observations": True,
        }
    )
dump(report / "alternative_inventory_candidates.json", alternatives)
print(json.dumps(results, indent=2))
