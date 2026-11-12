"""Resolve mixed-acquisition solid support with resumable, fingerprint-bound scans."""

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from phase5_11_common import ROOT, client_features, load
from roadmarking.io.source_inventory import fingerprint
from roadmarking.rasterization.pipeline import dump
from stage8_solid_discovery import scan_file, support

OUT = ROOT / "outputs/reports/stage8b/acquisition_support"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = load(ROOT / "outputs/reports/stage8/8_3_solid_discovery/solid_inventory.json")
    features, _, audit = client_features(check_network=False)
    context = {
        "inventory_sha256": hashlib.sha256(
            (ROOT / "outputs/reports/stage8/8_3_solid_discovery/solid_inventory.json").read_bytes()
        ).hexdigest(),
        "annotation_snapshot_sha256": sorted(r["sha256"] for r in audit["annotation_layers"]),
        "pixel_rule": "Stage8 scan_file / support, native 0.05; per-acquisition pixel union; v1",
    }
    context_path = OUT / "cache_context.json"
    if context_path.exists() and load(context_path) != context:
        raise ValueError("Acquisition cache belongs to different inventory/annotation inputs")
    dump(context_path, context)
    features = {f["global_id"]: f for f in features}
    raw = {r["path"]: r for r in load(Path(audit["report_directory"]) / "raw_integrity.json")}
    passed = {s["source"] for s in inventory["scans"] if s["full_source_integrity_passed"]}
    items, known = defaultdict(list), []
    for row in inventory["solids"]:
        if "observed_paint_pixels" not in row:
            continue
        sources = [p for p in row["header_files"] if p in passed]
        jobs = {Path(p).name.split("_Track")[0] for p in sources}
        if len(jobs) <= 1:
            known.append(
                {
                    "global_id": row["global_id"],
                    "acquisition": next(iter(jobs)) if jobs else None,
                    "observed_paint_pixels": row["observed_paint_pixels"],
                    "supported_length": row["supported_length"],
                    "method": "Stage-8 support union has zero or one integrity-passed acquisition; unchanged fingerprints checked in this audit.",  # noqa: E501
                }
            )
            continue
        f = features[row["global_id"]]
        item = {
            "global_id": f["global_id"],
            "geometry": f["geometry"],
            "centerline": f["centerline"],
            "bounds": f["geometry"].bounds,
        }
        for path in sources:
            items[path].append(item)
    # Check all reused source fingerprints, including singleton-attribution sources.
    changed = [p for p in passed if fingerprint(Path(p)) != raw[p]["fingerprint"]]
    if changed:
        dump(OUT / "source_changes.json", changed)
        raise ValueError("Source fingerprints changed; cannot reuse Stage-8 integrity")
    per_identity, scans = defaultdict(list), []
    for k, (path, group) in enumerate(sorted(items.items()), 1):
        token = hashlib.sha256(path.encode()).hexdigest()[:20]
        meta, arrays = OUT / f"{token}.json", OUT / f"{token}.npz"
        if (
            meta.exists()
            and arrays.exists()
            and load(meta)["fingerprint"] == fingerprint(Path(path))
        ):
            scan = load(meta)["scan"]
            with np.load(arrays) as cache:
                keys = {uid: cache[uid] for uid in cache.files}
        else:
            keys, scan = scan_file(path, raw[path], group)
            np.savez_compressed(arrays, **keys)
            dump(meta, {"fingerprint": fingerprint(Path(path)), "scan": scan})
        scans.append(scan)
        job = Path(path).name.split("_Track")[0]
        if scan["full_source_integrity_passed"]:
            for uid, values in keys.items():
                per_identity[(uid, job)].append(values)
        print(
            f"[{k}/{len(items)}] {Path(path).name}: integrity={scan['full_source_integrity_passed']}",  # noqa: E501
            flush=True,
        )
        dump(OUT / "progress.json", {"completed": k, "total": len(items), "scans": scans})
    measured = []
    pairs = {
        (item["global_id"], Path(p).name.split("_Track")[0])
        for p, group in items.items()
        for item in group
    }
    for uid, job in sorted(pairs):
        chunks = per_identity[(uid, job)]
        keys = np.unique(np.concatenate(chunks)) if chunks else None
        f = features[uid]
        pixels, length = support(
            {"bounds": f["geometry"].bounds, "centerline": f["centerline"]}, keys
        )
        measured.append(
            {
                "global_id": uid,
                "acquisition": job,
                "observed_paint_pixels": pixels,
                "supported_length": length,
                "method": "Fully decoded, integrity-passed sources; pixel union within acquisition.",  # noqa: E501
            }
        )
    rows = known + measured
    original = {r["global_id"]: r for r in inventory["solids"]}
    reconciliation = []
    for uid in sorted({uid for uid, _ in pairs}):
        chunks = [
            chunk for (key, _job), values in per_identity.items() if key == uid for chunk in values
        ]
        keys = np.unique(np.concatenate(chunks)) if chunks else None
        f = features[uid]
        pixels, length = support(
            {"bounds": f["geometry"].bounds, "centerline": f["centerline"]}, keys
        )
        prior = original[uid]
        reconciliation.append(
            {
                "global_id": uid,
                "prior_pixels": prior["observed_paint_pixels"],
                "recomputed_union_pixels": pixels,
                "prior_supported_length": prior["supported_length"],
                "recomputed_supported_length": length,
                "matches": pixels == prior["observed_paint_pixels"]
                and length == prior["supported_length"],
            }
        )
    summary = {}
    for job in sorted({r["acquisition"] for r in rows if r["acquisition"]}):
        group = [r for r in rows if r["acquisition"] == job]
        summary[job] = {
            "solid_ids_with_any_observed_paint": sum(r["observed_paint_pixels"] > 0 for r in group),
            "solid_ids_supported_length_ge_10": sum(r["supported_length"] >= 10 for r in group),
            "observed_paint_pixels": sum(r["observed_paint_pixels"] for r in group),
            "supported_length": sum(r["supported_length"] for r in group),
        }
    dump(
        OUT / "resolved_inventory.json",
        {
            "per_id_acquisition": rows,
            "summary": summary,
            "scans": scans,
            "stage8_integrity_failed_sources_excluded": [
                s["source"] for s in inventory["scans"] if not s["full_source_integrity_passed"]
            ],
            "new_integrity_failures": [s for s in scans if not s["full_source_integrity_passed"]],
            "all_reused_fingerprints_unchanged": True,
            "unresolved_acquisition_attribution": 0,
            "mixed_identity_union_reconciliation": reconciliation,
            "mixed_identity_union_mismatches": sum(not r["matches"] for r in reconciliation),
            "caveat": "Counts are acquisition-specific appearances, not globally distinct physical markings. Paint observations from separate acquisitions are not added together as independent identities.",  # noqa: E501
        },
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
