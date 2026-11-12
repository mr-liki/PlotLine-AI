"""Phase 5.11 Stage 1: audit the existing approved real training data (unique identities)."""

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import shapely

from phase5_11_common import EDGE, MERGE, NATIVE, ROOT, client_features, load, reserved_geography
from roadmarking.dataset.coverage import cluster_sections, paint_contrast, solid_shape
from roadmarking.rasterization.pipeline import dump

BIG = 10**8


def main():
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = ROOT / "outputs/reports/phase5_11" / run_id
    out.mkdir(parents=True, exist_ok=False)
    features, layers, _ = client_features()
    by_id = {f["global_id"]: f for f in features}
    regions, _, manifest = reserved_geography()
    train_regions = [r for r in regions if r["role"] == "train"]
    crops = [c for c in manifest["crops"] if c["split"] == "train"]

    # Region -> source LAZ (from the review specs that rendered each region).
    sources = {}
    spec54 = load(
        ROOT / "data/processed/phase5_4_candidates/20261105T173740792650Z_attempt2/"
        "review_spec.json"
    )
    sources.update({t["candidate_id"]: t["source"] for t in spec54["tiles"]})
    p54 = load(
        ROOT / "data/processed/phase5_training/phase5_4_20261105T173740792650Z/metadata/"
        "dataset_manifest.json"
    )
    sources.update({t["tile_id"]: t["source_sample"]["source_file"] for t in p54["tiles"]})
    sources.update({r["id"]: r["source"] for r in regions if r.get("source")})

    pixels = defaultdict(set)
    appearances = Counter()
    contrast = defaultdict(list)
    labels = {}
    tiles_of = defaultdict(set)
    background_only = 0
    paint_values, bg_bright = [], {}
    bg_samples = []
    for crop in crops:
        with np.load(
            crop["file"]
            if Path(crop["file"]).is_absolute()
            else Path(manifest["parent"]) / crop["file"]
        ) as d:
            masks, cls, v, rgb = d["masks"].astype(bool), d["labels"], d["validity"], d["rgb"]
        gray = rgb[..., 0].astype(float)
        a = crop["affine"]
        r0, c0 = round(-a[5] / NATIVE), round(a[2] / NATIVE)
        if not len(masks):
            background_only += 1
        paint = masks.any(0) if len(masks) else np.zeros(v.shape, bool)
        for mask, label, uid in zip(masks, cls, crop["global_instance_ids"], strict=True):
            rr, cc = np.nonzero(mask & v)
            pixels[uid].update(((rr + r0) * BIG + cc + c0).tolist())
            appearances[uid] += 1
            labels[uid] = int(label)
            tiles_of[uid].add(crop["tile_id"])
            value = paint_contrast(gray, mask & v, v, exclude=paint & ~mask)
            if value is not None:
                contrast[uid].append(value)
            paint_values.append(gray[mask & v])
        bg = v & ~paint
        rr, cc = np.nonzero(bg)
        bg_samples.append((crop["tile_id"], (rr + r0) * BIG + cc + c0, gray[bg]))
    paint_all = np.concatenate(paint_values)
    bright = float(np.percentile(paint_all, 25))
    for tile, idx, values in bg_samples:
        bg_bright.setdefault(tile, set()).update(idx[values >= bright].tolist())

    tile_box = {r["id"]: r["box"] for r in train_regions}
    rows = []
    for uid, label in labels.items():
        f = by_id[uid]
        footprint = shapely.union_all([tile_box[t] for t in tiles_of[uid] if t in tile_box])
        shape, length, rate = ("n/a", 0.0, 0.0)
        if label == 0:
            shape, length, rate = solid_shape(f["centerline"].intersection(footprint))
        values = contrast[uid]
        rows.append(
            {
                "global_id": uid,
                "class": "solid" if label == 0 else "dash",
                "designation": f["designation"],
                "client_layer": f["layer"],
                "width": f["width"],
                "edge_line": f["designation"] in EDGE,
                "merge_line": f["designation"] in MERGE,
                "solid_shape": shape,
                "clipped_solid_length": length,
                "turning_deg_per_50": rate,
                "unique_observed_paint_pixels": len(pixels[uid]),
                "crop_appearances": appearances[uid],
                "regions": sorted(tiles_of[uid]),
                "median_paint_contrast": float(np.median(values)) if values else None,
            }
        )
    solids = [r for r in rows if r["class"] == "solid"]
    dashes = [r for r in rows if r["class"] == "dash"]
    contrasts = np.array(
        [r["median_paint_contrast"] for r in rows if r["median_paint_contrast"] is not None]
    )
    low = float(np.percentile(contrasts, 25))
    labels_box = [r["box"] for r in train_regions]
    section = cluster_sections(labels_box, 500.0)
    sections = defaultdict(list)
    for r, s in zip(train_regions, section, strict=True):
        sections[s].append(r["id"])
    acquisitions = Counter()
    for r in train_regions:
        src = sources.get(r["id"], "UNKNOWN")
        acquisitions[src.split("\\")[-1].split("_Track")[0] + " | " + src.split("\\")[-1]] += 1
    summary = {
        "unique_training_identities": len(rows),
        "unique_solid_identities": len(solids),
        "unique_dash_identities": len(dashes),
        "crop_appearances": len(crops),
        "background_only_training_crops": background_only,
        "unique_observed_paint_pixels": int(sum(len(p) for p in pixels.values())),
        "unique_observed_paint_pixels_by_class": {
            "solid": int(sum(len(pixels[r["global_id"]]) for r in solids)),
            "dash": int(sum(len(pixels[r["global_id"]]) for r in dashes)),
        },
        "solid_shape_counts": dict(Counter(r["solid_shape"] for r in solids)),
        "solid_edge_lines": sum(r["edge_line"] for r in solids),
        "merge_or_interchange_lines": {
            "merge_designations": sum(r["merge_line"] for r in rows),
            "from_interchange_layer": sum(r["client_layer"] == "merged" for r in rows),
        },
        "designation_counts": dict(Counter(r["designation"] for r in rows)),
        "width_counts": dict(Counter(str(r["width"]) for r in rows)),
        "narrow_or_faded_proxy": {
            "definition": "median paint-minus-ring contrast below the training 25th percentile "
            f"({low:.1f} grey levels, established gamma .7 rendering); a proxy, not a "
            "visual fading verdict",
            "count": int((contrasts < low).sum()),
            "identities_without_measurable_contrast": len(rows) - len(contrasts),
        },
        "training_regions": len(train_regions),
        "geographic_sections_500_native_guard": {str(k): v for k, v in sections.items()},
        "section_count": len(sections),
        "source_acquisitions": dict(acquisitions),
        "distinct_source_files": len({sources.get(r["id"]) for r in train_regions}),
        "bright_background_threshold": bright,
        "unique_bright_observed_background_pixels_by_region": {
            k: len(v) for k, v in bg_bright.items()
        },
        "verified_hard_negative_scope": "Observed non-target pixels inside approved complete "
        "regions only (Phase 5.4/5.5/5.7 approvals); no dedicated roadside hard-negative "
        "regions exist",
    }
    catalogue = Counter(
        (f["designation"], f["visual_id"]) for f in features if f["centerline"] is not None
    )
    summary["client_catalogue_longitudinal"] = {
        f"{k[0]} | class {k[1]}": v for k, v in catalogue.items()
    }
    weaknesses = [
        f"Only {len(solids)} unique solid identities vs {len(dashes)} dashes; solids appear in "
        f"{sum(r['crop_appearances'] for r in solids)} crop appearances, so long-line "
        "exposure comes from repeated views of very few physical lines.",
        f"Curved solids: {summary['solid_shape_counts'].get('curved', 0)}; the Phase 5.9 "
        "region failure is dominated by long curved edge lines.",
        f"{len(sections)} geographic sections from {summary['distinct_source_files']} source "
        "files and essentially one acquisition job; roadside context (vegetation, kerbs) is "
        "narrow.",
        "No dedicated verified roadside hard-negative regions; the background comes only from "
        "inside road-centred approved windows.",
        f"Merge/diverge lines: {summary['merge_or_interchange_lines']['merge_designations']}.",
    ]
    priority = [
        "1. Long curved solid lines (edge and continuous lines), new sections, observed support.",
        "2. Straight solid edge lines from additional acquisitions (Job 1218 / Job 20250924).",
        "3. Merge/diverge (acceleration/deceleration) areas with their adjoining solids.",
        "4. Diverse broken lines, including warning dashes and interchange ramps.",
        "5. Low-contrast/narrow paint.",
        "6. Windows with bright roadside returns (vegetation, kerbs) to supply verified "
        "observed background.",
    ]
    dump(
        out / "training_coverage_audit.json",
        {
            "summary": summary,
            "weaknesses": weaknesses,
            "acquisition_priority": priority,
            "per_identity": rows,
            "client_layers": layers,
            "note": "Counts are unique physical source identities; crop appearances overlap.",
        },
    )
    dump(
        ROOT / "outputs/reports/phase5_11/latest.json",
        {"run_id": run_id, "report_dir": str(out), "status": "AUDIT_COMPLETE"},
    )
    print(
        json.dumps(
            {
                "report_dir": str(out),
                **{
                    k: summary[k]
                    for k in (
                        "unique_training_identities",
                        "unique_solid_identities",
                        "unique_dash_identities",
                        "solid_shape_counts",
                        "solid_edge_lines",
                        "merge_or_interchange_lines",
                        "section_count",
                        "source_acquisitions",
                        "narrow_or_faded_proxy",
                        "background_only_training_crops",
                        "unique_observed_paint_pixels_by_class",
                    )
                },
            },
            indent=1,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
