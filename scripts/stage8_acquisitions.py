"""Attribute measured real paint to acquisitions using source manifests only."""

from collections import defaultdict
from pathlib import Path

import numpy as np

from phase5_11_common import ROOT, client_features, load
from roadmarking.rasterization.pipeline import dump
from stage8_paint_statistics import measure


def acquisition(source):
    return Path(source).name.split("_Track")[0] if source else "UNRESOLVED"


def main():
    mapping = {}
    p55 = load(ROOT / "outputs/reports/phase5_5/latest.json")
    parent = load(Path(p55["dataset"]) / "metadata/dataset_manifest.json")["parent_dataset"]
    for t in load(Path(parent) / "metadata/dataset_manifest.json")["tiles"]:
        mapping[t["tile_id"]] = acquisition(t["source_sample"]["source_file"])
    for phase in (
        "phase5_4",
        "phase5_7_new_review",
        "phase5_8_actual_review",
        "phase5_11",
        "phase5_12",
        "stage6_review",
        "stage7_new_test",
    ):
        latest = ROOT / f"outputs/reports/{phase}/latest.json"
        if latest.exists():
            folder = Path(load(latest)["report_dir"])
        else:
            folders = sorted(
                (ROOT / f"outputs/reports/{phase}").glob("*/new_region_candidates.json")
            )
            if not folders:
                continue
            folder = folders[-1].parent
        for t in load(folder / "new_region_candidates.json")["tiles"]:
            mapping[t["candidate_id"]] = acquisition(t["source"])
    paint = load(ROOT / "outputs/reports/stage8/8_4_paint_statistics.json")
    result = []
    unresolved = []
    remeasured = []
    features = {f["global_id"]: f for f in client_features(check_network=False)[0]}
    for phase in ("phase5_12", "stage6"):
        appearances = defaultdict(set)
        crops = load(paint[phase]["manifest"])["crops"]
        for c in crops:
            if c["split"] not in ("train", "val"):
                continue
            for uid in c["global_instance_ids"]:
                appearances[(c["split"], uid)].add(mapping.get(c["tile_id"], "UNRESOLVED"))
        for r in paint[phase]["per_id"]:
            jobs = appearances[(r["split"], r["global_id"])]
            job = next(iter(jobs)) if len(jobs) == 1 else "MULTIPLE_OR_UNRESOLVED"
            if "UNRESOLVED" in job:
                if "UNRESOLVED" not in jobs:
                    remeasured.append(
                        {"phase": phase, "global_id": r["global_id"], "acquisitions": sorted(jobs)}
                    )
                    for actual_job in sorted(jobs):
                        selected = [
                            c
                            for c in crops
                            if c["split"] == r["split"]
                            and r["global_id"] in c["global_instance_ids"]
                            and mapping.get(c["tile_id"]) == actual_job
                        ]
                        for actual in measure(selected, features):
                            if actual["global_id"] == r["global_id"]:
                                result.append(
                                    {"dataset": phase, "acquisition": actual_job, **actual}
                                )
                    continue
                unresolved.append({"phase": phase, **r, "acquisitions": sorted(jobs)})
            result.append({"dataset": phase, "acquisition": job, **r})
    dev = load(ROOT / "outputs/reports/stage8/8_4_dev_paint.json")
    legacy_jobs = {a for tile, a in mapping.items() if tile.startswith(("RIGHT_", "INTERCHANGE_"))}
    dev_jobs = {
        "S6_01": mapping.get("S6_01", "UNRESOLVED"),
        "NEW_POSITIVE_DEV": mapping.get("PHASE58_POSITIVE_DEV_REVIEW", "UNRESOLVED"),
        "SPARSE_NEGATIVE_CONTROL": mapping.get("NEW_DEV_REVIEW", "UNRESOLVED"),
        "LEGACY_POSITIVE_DEV": next(iter(legacy_jobs)) if len(legacy_jobs) == 1 else "UNRESOLVED",
    }
    for r in dev["per_id"]:
        result.append({**r, "dataset": r["region"], "acquisition": dev_jobs[r["region"]]})
    summaries = []
    groups = defaultdict(list)
    for name, job in dev_jobs.items():
        for cls in ("solid", "dash"):
            groups[(name, "development", job, cls)] = []
    for r in result:
        groups[(r["dataset"], r["split"], r["acquisition"], r["class"])].append(r)
    for (dataset, split, job, cls), rows in groups.items():
        contrasts = [
            r.get("contrast_rgb_mean_over_crops", r.get("contrast_grayscale")) for r in rows
        ]
        contrasts = [v for v in contrasts if v is not None]
        summaries.append(
            {
                "dataset": dataset,
                "split": split,
                "acquisition": job,
                "class": cls,
                "ids": len({r["global_id"] for r in rows}),
                "observed_paint_pixels": sum(r["observed_paint_pixels"] for r in rows),
                "supported_length_native": sum(r["supported_length_native"] or 0 for r in rows),
                "contrast_mean_over_ids": float(np.mean(contrasts)) if contrasts else None,
            }
        )
    dump(
        ROOT / "outputs/reports/stage8/8_4_paint_by_acquisition.json",
        {
            "summary": summaries,
            "unresolved": unresolved,
            "multi_acquisition_ids_remeasured": remeasured,
            "tile_acquisition_mapping": mapping,
            "development_acquisition_mapping": dev_jobs,
            "t7": (
                "Metadata only in 8_4_paint_metadata.json; supported length was not "
                "measured on protected labels."
            ),
        },
    )
    print(
        f"{len(summaries)} class/split/acquisition summaries; {len(unresolved)} unresolved training IDs"  # noqa: E501
    )


if __name__ == "__main__":
    main()
