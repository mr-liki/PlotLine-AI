"""Phase 4 orchestration with hard gates.

Sequence: inventory -> layer inspection -> feature QC/taxonomy -> frame comparison and
displacement measurement -> road-candidate review -> status assignment -> (only if the
alignment gate passes and VERIFIED features exist) label generation, splits, QA.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import psutil
import rasterio
import shapely

from roadmarking.dataset import alignment as al
from roadmarking.dataset import qa, road_review, synthetic
from roadmarking.dataset.annotations import (
    QC_STATUSES,
    _num,
    assign_status,
    inspect_file,
    load_review_records,
    normalize_features,
)
from roadmarking.dataset.geometry import tile_affine, world_to_pixel
from roadmarking.dataset.labels import LabelFeature, build_tile_labels
from roadmarking.dataset.splits import spatial_split
from roadmarking.dataset.taxonomy import Taxonomy
from roadmarking.dataset.writer import read_tile_inputs, write_sample
from roadmarking.training.preflight import digest, inventory


def world_to_pixel_cr(affine, x, y):
    row, col = world_to_pixel(affine, x, y)
    return float(col), float(row)


IST = timezone(timedelta(hours=5, minutes=30))
DEFAULTS = {
    "width_range_m": (0.05, 1.0),
    "alignment_tolerance": 0.05,  # native units (5 cm if metres): 1 pixel
    "min_alignment_samples": 200,
    "max_alignment_tiles": 40,
    "split_ratios": (0.70, 0.15, 0.15),
    "split_block_tiles": 6,
    "min_instances_per_split": 50,
    "long_instance_extent": 25.0,  # native units; longer features span split blocks
}


class Timer:
    def __init__(self):
        self.times = {}

    def __call__(self, name):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.t = time.perf_counter()

            def __exit__(self, *exc):
                timer.times[name] = timer.times.get(name, 0) + time.perf_counter() - self.t

        return _Ctx()


def _sha_files(paths) -> dict[str, str]:
    return {str(p): digest(Path(p)) for p in paths}


def _code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()


def measure_alignment(features, tiles_by_id, feature_tiles, opts) -> dict:
    """Independent paint-position check on longitudinal centrelines inside rasters."""
    lines = features[features["kind"] == "line"]
    per_tile = defaultdict(list)
    for idx in lines.index:
        for tid in feature_tiles.get(int(idx), []):
            per_tile[tid].append(idx)
    results, all_offsets, misaligned = [], [], set()
    for tid in sorted(per_tile)[: opts["max_alignment_tiles"]]:
        tile = tiles_by_id[tid]
        with rasterio.open(tile["layers"]["normalized"]) as src:
            img = src.read(1).astype(float)
        img[img < -1] = np.nan
        for idx in per_tile[tid]:
            geom = features.at[idx, "geometry"]
            values = np.concatenate(
                [al.perpendicular_offsets(p, img, tile_affine(tile)) for p in al.line_parts(geom)]
                or [np.empty(0)]
            )
            if len(values) >= 3:
                med = float(np.median(np.abs(values)))
                if med > opts["alignment_tolerance"]:
                    misaligned.add(features.at[idx, "global_id"])
                all_offsets.append(values)
        results.append({"tile_id": tid, "lines": len(per_tile[tid])})
    if not all_offsets:
        return {"samples": 0, "tiles": results, "misaligned": misaligned}
    v = np.concatenate(all_offsets)
    return {
        "samples": len(v),
        "median_signed": float(np.median(v)),
        "median_abs": float(np.median(np.abs(v))),
        "p90_abs": float(np.percentile(np.abs(v), 90)),
        "tiles": results,
        "misaligned": misaligned,
    }


def converter_selftest(out_dir: Path) -> dict:
    """Run the converter on SYNTHETIC features and write QA overlays + integrity checks."""
    feats = synthetic.features()
    left = synthetic.make_tile("SYNTHETIC_left", 0, 0)
    right = synthetic.make_tile("SYNTHETIC_right", 0, 192)  # 64-pixel overlap
    results = {}
    shared = {}
    for tile in (left, right):
        image, observed = synthetic.paint_raster(tile, feats)
        labels = build_tile_labels(
            tile_affine(tile), (tile["height"], tile["width"]), observed, feats
        )
        qa.tile_overlay(
            out_dir / f"{tile['tile_id']}.png",
            tile,
            image,
            [f.geometry for f in feats],
            labels,
            "SYNTHETIC converter self-test (not client data): narrow lines, dashes, "
            "curve, arrow, hole, tile crossing",
        )
        shared[tile["tile_id"]] = labels
        results[tile["tile_id"]] = {
            "instances": [
                {k: r[k] for k in ("local_id", "instance_uid", "visual_id", "pixels", "status")}
                for r in labels.instances
            ],
        }
    # Overlap consistency: identical global instances on identical world pixels.
    lab_l, lab_r = shared["SYNTHETIC_left"], shared["SYNTHETIC_right"]
    uid_l = {r["local_id"]: r["instance_uid"] for r in lab_l.instances if r["local_id"]}
    uid_r = {r["local_id"]: r["instance_uid"] for r in lab_r.instances if r["local_id"]}
    a = np.vectorize(lambda x: uid_l.get(x, 0))(lab_l.instance[:, 192:])
    b = np.vectorize(lambda x: uid_r.get(x, 0))(lab_r.instance[:, :64])
    hole_feat = next(f for f in feats if f.kind == "polygon" and f.geometry.interiors)
    hx, hy = hole_feat.geometry.interiors[0].centroid.coords[0]
    col, row = world_to_pixel_cr(tile_affine(left), hx, hy)
    results["checks"] = {
        "overlap_global_ids_identical": bool(np.array_equal(a, b)),
        "hole_centre_not_instance": int(lab_l.instance[int(row), int(col)]) == 0,
        "every_feature_nonempty_somewhere": all(
            any(
                r["pixels"] > 0
                for lab in shared.values()
                for r in lab.instances
                if r["instance_uid"] == f.instance_uid
            )
            for f in feats
        ),
    }
    results["passed"] = all(results["checks"].values())
    return results


def run_phase4(
    cfg, project_root: Path, review_csv: Path | None = None, options: dict | None = None
) -> dict:
    opts = {**DEFAULTS, **(options or {})}
    timer = Timer()
    t_start = time.perf_counter()
    started = datetime.now(IST).isoformat()
    report_dir = cfg.path("outputs.reports") / "phase4"
    manifest_path = cfg.path("data.processed") / "phase3/metadata/tile_manifest.json"
    taxonomy_path = project_root / "configs/phase4_taxonomy.yaml"
    taxonomy = Taxonomy.load(taxonomy_path)

    with timer("inventory_and_inspection"):
        inv = {
            "annotations": inventory(cfg.path("data.raw_annotations")),
            "reference_gis": inventory(cfg.path("data.reference_gis")),
        }
        files = [
            Path(d["path"])
            for record in inv.values()
            for d in record["datasets"]
            if d["status"] == "UNINSPECTED"
        ]
        source_hashes_before = _sha_files(
            files + [Path(p) for r in inv.values() for p in r["other_files"]]
        )
        inspected = [inspect_file(p, taxonomy) for p in files]
        other = []
        for r in inv.values():
            for p in r["other_files"]:
                rec = inspect_file(Path(p), taxonomy)
                other.append({"path": p, "status": rec["status"], "error": rec.get("error")})

    tiles = json.loads(manifest_path.read_text(encoding="utf-8"))
    tiles_by_id = {t["tile_id"]: t for t in tiles}
    protected = [manifest_path, cfg.path("outputs.reports") / "phase3/phase3_report.json"]
    protected_before = _sha_files([p for p in protected if p.is_file()])

    identity = {
        "annotation_sha256": source_hashes_before,
        "manifest_sha256": digest(manifest_path),
        "taxonomy_sha256": digest(taxonomy_path),
        "code_sha256": _code_hash(),
        "review_csv_sha256": digest(review_csv) if review_csv and review_csv.is_file() else None,
        "options": {k: list(v) if isinstance(v, tuple) else v for k, v in opts.items()},
    }
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    out_root = cfg.path("data.processed") / "phase4" / f"run_{run_id}"
    meta_dir = out_root / "metadata"
    qc_dir = out_root / "quality_control"
    for d in (meta_dir, qc_dir, out_root / "splits"):
        d.mkdir(parents=True, exist_ok=True)

    with timer("feature_normalization_and_qc"):
        features = normalize_features(files, taxonomy, opts["width_range_m"])

    with timer("alignment"):
        lidar = al.lidar_header_extents(cfg.path("outputs.reports") / "lidar_cache")
        comparison = al.compare_frames(features, tiles, lidar)
        feature_tiles = comparison.pop("_feature_tiles")
        overlap_n = comparison["feature_overlap"]["features_intersecting_tiles"]
        if overlap_n == 0:
            measured = {"samples": 0, "misaligned": set()}
            align_status = "BLOCKED_NO_SPATIAL_OVERLAP"
        else:
            measured = measure_alignment(features, tiles_by_id, feature_tiles, opts)
            if measured["samples"] < opts["min_alignment_samples"]:
                align_status = "BLOCKED_INSUFFICIENT_ALIGNMENT_EVIDENCE"
            elif measured["median_abs"] <= opts["alignment_tolerance"]:
                align_status = "PASSED_SHARED_NATIVE_FRAME"
            else:
                align_status = "FAILED_DISPLACEMENT_ABOVE_TOLERANCE"
        alignment_passed = align_status == "PASSED_SHARED_NATIVE_FRAME"
        misaligned = measured.pop("misaligned")
        qa.extent_overview(
            report_dir / "alignment_extent_overview.png",
            comparison,
            {n: shapely.force_2d(g.geometry.values) for n, g in features.groupby("source_file")},
        )

    reviews = load_review_records(review_csv)
    assign_status(features, alignment_passed, reviews, misaligned)

    with timer("road_review"):
        tile_stats = {t["tile_id"]: road_review.tile_statistics(t) for t in tiles}
        source_cmp = {}
        for source in sorted({Path(t["source_file"]).stem for t in tiles}):
            run_dir = Path(
                next(t for t in tiles if Path(t["source_file"]).stem == source)["layers"][
                    "normalized"
                ]
            ).parents[1]
            surface = (
                cfg.path("data.processed")
                / "phase3/road_masks"
                / source
                / run_dir.name
                / "surface_state_1m.tif"
            )
            comp = road_review.source_comparison(
                source, cfg.path("outputs.reports") / "lidar_cache", surface
            )
            if comp and comp.get("comparable"):
                road_review.overview_figure(
                    report_dir / "road_review" / f"{source}_overview.png", source, comp
                )
                comp.pop("_arrays")
            source_cmp[source] = comp
        flag_counts = Counter(f for s in tile_stats.values() for f in s["flags"])
        flagged = sorted(
            tile_stats.values(),
            key=lambda s: (-len(s["flags"]), -(s["missing_in_candidate_pct"] or 0), s["tile_id"]),
        )
        by_source = defaultdict(list)
        for s in flagged:
            by_source[s["source"]].append(s)
        for source, stats_list in by_source.items():
            picks = road_review.pick_review_tiles(stats_list)
            if not picks:
                continue
            road_review.tile_side_by_side(
                report_dir / "road_review" / f"{source}_tiles.png",
                [tiles_by_id[i] for i in picks],
                tile_stats,
            )

    with timer("converter_selftest"):
        selftest = converter_selftest(report_dir / "converter_selftest")

    dataset = {
        "generated": False,
        "images": 0,
        "instances": 0,
        "polygon_instances": 0,
        "polyline_instances": 0,
    }
    split_result = None
    verified = features[features["qc_status"].isin(taxonomy.policy["allowed_qc_statuses"])]
    if alignment_passed and len(verified):
        with timer("label_generation"):
            excluded = {t for t, s in tile_stats.items() if "near_empty_tile" in s["flags"]}
            dataset, split_result = _generate_dataset(
                out_root, verified, tiles_by_id, feature_tiles, taxonomy, opts, qc_dir, excluded
            )

    with timer("metadata_writing"):
        _write_feature_tables(meta_dir, features)
        (meta_dir / "class_mapping.json").write_text(
            json.dumps(taxonomy.mapping_table(), indent=1, ensure_ascii=False), encoding="utf-8"
        )
        alignment_record = {
            "status": align_status,
            "passed": alignment_passed,
            "comparison": comparison,
            "displacement": measured,
            "tolerance_native_units": opts["alignment_tolerance"],
            "transformation_applied": False,
            "crs_assigned": False,
        }
        (meta_dir / "alignment.json").write_text(
            json.dumps(alignment_record, indent=1), encoding="utf-8"
        )
        (meta_dir / "road_review.json").write_text(
            json.dumps(
                {"tiles": tile_stats, "sources": source_cmp, "flag_counts": flag_counts}, indent=1
            ),
            encoding="utf-8",
        )
        (meta_dir / "annotation_inspection.json").write_text(
            json.dumps(
                {"inventory": inv, "files": inspected, "other_files": other},
                indent=1,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # Integrity: originals and Phase 3 metadata untouched.
    source_unchanged = _sha_files(source_hashes_before.keys()) == source_hashes_before
    phase3_unchanged = _sha_files(protected_before.keys()) == protected_before
    if not (source_unchanged and phase3_unchanged):
        raise RuntimeError("Client annotations or Phase 3 metadata changed during Phase 4")
    mem = psutil.Process().memory_info()
    status_counts = Counter(features["qc_status"])
    report = {
        "date": started,
        "stage": "Phase 4: annotation alignment and dataset preparation",
        "run_id": run_id,
        "output_root": str(out_root),
        "status": (
            "DATASET_GENERATED"
            if dataset["generated"]
            else (
                f"STOPPED_AT_ALIGNMENT_GATE: {align_status}"
                if not alignment_passed
                else "STOPPED_NO_VERIFIED_FEATURES"
            )
        ),
        "inputs": {"inventory": inv, "files": inspected, "other_files": other},
        "alignment": alignment_record,
        "annotation_quality": {
            "status_counts": {s: int(status_counts.get(s, 0)) for s in QC_STATUSES},
            "reason_counts": dict(Counter(r for rs in features["qc_reasons"] for r in rs)),
            "visual_class_distribution": {
                str(k): int(v)
                for k, v in Counter(features["visual_class"].fillna("UNMAPPED")).items()
            },
            "client_code_distribution": {
                kind: {
                    str(k): int(v) for k, v in Counter(g["client_code"].fillna("UNMAPPED")).items()
                }
                for kind, g in features.groupby("kind")
            },
            "by_file": {
                name: {
                    "features": len(g),
                    "status_counts": {k: int(v) for k, v in Counter(g["qc_status"]).items()},
                    "duplicates_flagged": int(
                        sum(
                            "duplicate_in_other_file" in r or "duplicate_in_same_file" in r
                            for r in g["qc_reasons"]
                        )
                    ),
                }
                for name, g in features.groupby("source_file")
            },
            "z_minus_elevation_attr": _percentiles(features["z_minus_elevation"]),
            "z_mismatch_gt_1m": int(
                sum(x is not None and abs(x) > 1 for x in features["z_minus_elevation"])
            ),
            "line_length_m": _percentiles(features.loc[features["kind"] == "line", "length_m"]),
            "polygon_area_m2": _percentiles(features.loc[features["kind"] == "polygon", "area_m2"]),
        },
        "road_review": {
            "flag_counts": dict(flag_counts),
            "tiles_reviewed": len(tile_stats),
            "sources": source_cmp,
        },
        "taxonomy": {"path": str(taxonomy_path), "version": taxonomy.version, "provisional": True},
        "converter_selftest": selftest,
        "dataset": dataset,
        "splits": split_result,
        "integrity": {
            "client_annotation_files_unchanged": source_unchanged,
            "phase3_metadata_unchanged": phase3_unchanged,
            "phase3_hashes": protected_before,
        },
        "performance": {
            "timings_seconds": timer.times,
            "total_seconds": time.perf_counter() - t_start,
            "working_set_bytes": mem.rss,
            "peak_working_set_bytes": getattr(mem, "peak_wset", None),
            "gpu_used": False,
            "laz_reread": False,
        },
        "phase4_complete": bool(dataset["generated"]),
        "phase5_ready": False,
    }
    report["performance"]["output_bytes"] = sum(
        p.stat().st_size for p in out_root.rglob("*") if p.is_file()
    )
    (meta_dir / "run_identity.json").write_text(json.dumps(identity, indent=1), encoding="utf-8")
    (meta_dir / "phase4_run.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return report


def _percentiles(series) -> dict | None:
    v = np.asarray([x for x in series if x is not None and np.isfinite(x)], dtype=float)
    if not len(v):
        return None
    q = np.percentile(v, [0, 5, 50, 95, 100])
    return {"n": len(v), "min": q[0], "p5": q[1], "median": q[2], "p95": q[3], "max": q[4]}


def _write_feature_tables(meta_dir: Path, features) -> None:
    cols = [
        "instance_uid",
        "global_id",
        "source_file",
        "layer",
        "fid",
        "id_sig",
        "kind",
        "visual_class",
        "visual_id",
        "client_code",
        "client_name",
        "width_m",
        "dash_m",
        "gap_m",
        "group_id",
        "parts",
        "holes",
        "length_m",
        "area_m2",
        "z_minus_elevation",
        "duplicate_of",
        "qc_status",
    ]
    with open(meta_dir / "annotation_validation.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([*cols, "qc_reasons"])
        for row in features[[*cols, "qc_reasons"]].itertuples(index=False):
            values = list(row)
            writer.writerow([*("" if v is None else v for v in values[:-1]), ";".join(values[-1])])
    with open(meta_dir / "instance_id_mapping.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "instance_uid",
                "global_id",
                "source_file",
                "layer",
                "fid",
                "id_sig",
                "group_id",
                "qc_status",
            ]
        )
        for row in features[
            [
                "instance_uid",
                "global_id",
                "source_file",
                "layer",
                "fid",
                "id_sig",
                "group_id",
                "qc_status",
            ]
        ].itertuples(index=False):
            writer.writerow(["" if v is None else v for v in row])


def _generate_dataset(
    out_root, verified, tiles_by_id, feature_tiles, taxonomy, opts, qc_dir, excluded=frozenset()
):
    """Only reachable after the alignment gate passes. Incremental per-tile writing.

    ``excluded`` tiles (explicit road-review exclusion policy) receive no labels.
    """
    policy = taxonomy.policy
    per_tile = defaultdict(list)
    long_instances = set()
    for idx, row in verified.iterrows():
        width = _num(row["width_m"])
        if _num(row["visual_id"]) is None or (row["kind"] == "line" and width is None):
            continue
        if isinstance(row["duplicate_of"], str):
            continue  # the earlier copy (sorted global_id) is the single training instance
        minx, miny, maxx, maxy = row["geometry"].bounds
        if max(maxx - minx, maxy - miny) > opts["long_instance_extent"]:
            long_instances.add(int(row["instance_uid"]))
        feat = LabelFeature(
            int(row["instance_uid"]),
            row["global_id"],
            int(row["visual_id"]),
            row["kind"],
            row["geometry"],
            width,
            row["group_id"] if isinstance(row["group_id"], str) else None,
        )
        for tid in feature_tiles.get(int(idx), []):
            if tid not in excluded:
                per_tile[tid].append(feat)
    records, tile_instances, tile_classes = [], {}, {}
    for n, tid in enumerate(sorted(per_tile)):
        tile = tiles_by_id[tid]
        inputs = read_tile_inputs(tile)
        labels = build_tile_labels(
            tile_affine(tile),
            (tile["height"], tile["width"]),
            inputs["observed"],
            per_tile[tid],
            completeness=None,
            supersample=policy["supersample"],
            coverage_threshold=policy["coverage_threshold"],
            boundary_ignore=policy["boundary_band_ignore"],
            ignore_id=taxonomy.ignore_id,
        )
        rec = write_sample(out_root, tile, inputs, labels, {"taxonomy_version": taxonomy.version})
        records.append(rec)
        kept = [r for r in labels.instances if r["pixels"]]
        tile_instances[tid] = {r["instance_uid"] for r in kept}
        tile_classes[tid] = Counter(r["visual_id"] for r in kept)
        if n < 12:
            qa.tile_overlay(
                qc_dir / f"{tid}.png",
                tile,
                inputs["normalized"],
                [f.geometry for f in per_tile[tid]],
                labels,
                tid,
            )
    split = spatial_split(
        list(tiles_by_id.values()),
        tile_instances,
        tile_classes,
        opts["split_ratios"],
        opts["split_block_tiles"],
        0.0,
        opts["min_instances_per_split"],
        long_instances,
    )
    (out_root / "splits" / "split_manifest.json").write_text(
        json.dumps(split, indent=1), encoding="utf-8"
    )
    manifest = [
        {
            **{k: r[k] for k in ("tile_id", "files", "instance_id_to_semantic_id")},
            "split": split["assignment"].get(r["tile_id"], "guard_excluded"),
        }
        for r in records
    ]
    (out_root / "metadata" / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )
    kinds = Counter(r["kind"] for rec in records for r in rec["instances"] if r["pixels"])
    return {
        "generated": True,
        "images": len(records),
        "instances": len(set().union(*tile_instances.values())) if tile_instances else 0,
        "polygon_instances": kinds["polygon"],
        "polyline_instances": kinds["line"],
    }, split
