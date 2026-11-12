"""Metadata-only T7 eligibility and observed-footprint audit; no inference."""

import subprocess
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import rasterio
from rasterio.features import shapes
from shapely import coverage_union_all
from shapely.geometry import shape

from phase5_11_common import ROOT, load
from roadmarking.rasterization.pipeline import dump

OUT = ROOT / "outputs/reports/stage8/8_2_eligibility"
ROLE = (
    "DASH-ONLY HELD-OUT CANDIDATE — pending first human geometry and "
    "complete-visible-marking review; NOT an independent solid-and-dash final test"
)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spec = load(
        ROOT / "outputs/reports/stage7_new_test/20261107T051050944913Z/new_region_candidates.json"
    )
    rows, records, ids = [], [], defaultdict(list)
    stats = []
    for phase in ("phase5_11", "phase5_12", "stage6_review", "stage7_new_test"):
        latest = load(ROOT / f"outputs/reports/{phase}/latest.json")
        source = Path(latest["report_dir"]) / "new_region_candidates.json"
        for t in load(source)["tiles"]:
            for cls in ("solid", "dash"):
                obs = [r for r in t.get("observed_ids", []) if r["class"] == cls]
                contrasts = [
                    r["contrast_normalized"]
                    for r in obs
                    if r.get("contrast_normalized") is not None
                ]
                stats.append(
                    {
                        "phase": phase,
                        "region": t["candidate_id"],
                        "role": (
                            "protected_candidate_metadata_only"
                            if phase == "stage7_new_test"
                            else (
                                "development"
                                if t["candidate_id"] == "S6_01"
                                else "training_candidate_metadata"
                            )
                        ),
                        "acquisition": t.get(
                            "acquisition", Path(t["source"]).name.split("_Track")[0]
                        ),
                        "class": cls,
                        "ids": [r["global_id"] for r in obs],
                        "observed_paint_pixels": sum(r["observed_pixels"] for r in obs),
                        "contrast_mean_over_ids": float(np.mean(contrasts)) if contrasts else None,
                        "supported_length": None,
                        "supported_length_status": "not present in source metadata; not inferred from polygon length",  # noqa: E501
                        "source_manifest": str(source),
                    }
                )
    for t in spec["tiles"]:
        name = t["candidate_id"]
        with rasterio.open(t["layers"]["observed"]) as ds:
            observed = ds.read(1).astype(bool)
            polys = [
                shape(g)
                for g, v in shapes(observed.astype("uint8"), mask=observed, transform=ds.transform)
                if v
            ]
        geom = coverage_union_all(polys)
        print(f"{name}: {int(observed.sum())} observed pixels polygonized", flush=True)
        rows.append(
            {
                "region": name,
                "role": ROLE,
                "observed_pixels": int(observed.sum()),
                "observed_area_native2": geom.area,
                "geometry": geom,
            }
        )
        for r in t["observed_ids"]:
            ids[r["global_id"]].append({"region": name, "class": r["class"]})
        records.append(
            {
                "region": name,
                "role": ROLE,
                "acquisition": t["acquisition"],
                "source": t["source"],
                "observed_fraction": float(observed.mean()),
                "shared_training_solid_ids": [
                    r
                    for r in t["observed_ids"]
                    if r["class"] == "solid"
                    and r["global_id"] in t["shares_ids_with_existing_training"]
                ],
                "per_id": t["observed_ids"],
                "observed_footprint_area_native2": geom.area,
                "min_distance_to_training_native": t["min_distance_to_training_native"],
                "min_distance_to_reserved_native": t["min_distance_to_any_reserved_native"],
                "approved": False,
                "inference_allowed": False,
            }
        )
    pyogrio.write_dataframe(
        gpd.GeoDataFrame(rows, geometry="geometry", crs=None),
        OUT / "observed_footprints.gpkg",
        layer="observed_footprints",
        driver="GPKG",
        promote_to_multi=True,
    )
    # Search existing artifacts, excluding this Stage-8 audit's own references.
    search = subprocess.run(
        [
            "rg",
            "-l",
            "--text",
            "T7_0[1-8]",
            "outputs",
            "-g",
            "!stage8/**",
            "-g",
            "*.json",
            "-g",
            "*.log",
            "-g",
            "*.md",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    matches = [p for p in search.stdout.splitlines() if "stage8" not in p]
    suspicious = [
        p for p in matches if any(w in p.lower() for w in ("inference", "prediction", "evaluation"))
    ]
    dump(
        OUT / "eligibility.json",
        {
            "role": ROLE,
            "regions": records,
            "shared_between_t7": {k: v for k, v in ids.items() if len(v) > 1},
            "inference_artifact_search": {
                "command": search.args,
                "exit_code": search.returncode,
                "matching_files": matches,
                "inference_named_matches": suspicious,
                "limitation": "Artifact search is evidence, not proof that no unlogged execution occurred.",  # noqa: E501
            },
        },
    )
    dump(
        OUT.parent / "8_4_paint_metadata.json",
        {
            "rows": stats,
            "limitations": (
                "Metadata only for T7. Region ID counts overlap and must not be summed "
                "as distinct identities. Supported length unavailable in these "
                "manifests; legacy development and crop-deduplicated dataset "
                "statistics still required."
            ),
        },
    )
    print(
        f"Wrote {len(records)} protected eligibility records; {len(suspicious)} inference-named matches"  # noqa: E501
    )


if __name__ == "__main__":
    main()
