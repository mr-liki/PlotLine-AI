"""Stage 9E-D: real road-text training-source inventory (metadata only, no rescan).

Establishes whether any LiDAR source is new since the Stage 9C.2 search and whether any
authoritative label source contains road text. No point cloud is decoded, because no
new reason to rescan identical sources exists unless a newer acquisition is found.
"""

import glob
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from stage9d_common import ROOT, dump, read
from stage9e_lib import new_run

SHARE = r"\\192.168.10.82\nagendra\NAGENDRA\PROJECTS\NT_895\GIS\Asset project\2026\INPUTS"
LOCAL = [ROOT / "data/raw", ROOT / "data/cache"]


def listing(root):
    rows = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".las", ".laz")):
                p = os.path.join(dirpath, f)
                st = os.stat(p)
                rows.append(
                    dict(
                        path=p,
                        bytes=st.st_size,
                        modified_utc=datetime.fromtimestamp(st.st_mtime, UTC).isoformat(),
                    )
                )
    return rows


def main():
    out = new_run("text_sources")
    s9c2 = Path(read(ROOT / "outputs/reports/stage9c_2/latest.json")["run"])
    inventory = read(s9c2 / "source_inventory.json")
    search_time = datetime.strptime(s9c2.name[:15], "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
    share = listing(SHARE) if os.path.isdir(SHARE) else []
    local = [r for root in LOCAL if root.exists() for r in listing(root)]
    newer = [r for r in share + local if datetime.fromisoformat(r["modified_utc"]) > search_time]
    approved_text = {}
    for f in glob.glob(str(ROOT / "data/processed/**/approved_geometry.geojson"), recursive=True):
        d = json.loads(Path(f).read_text(encoding="utf-8-sig"))
        approved_text[f] = sum(1 for x in d["features"] if x["properties"].get("visual_id") == 9)
    s9b = Path(read(ROOT / "outputs/reports/stage9b/latest.json")["report_dir"])
    prior_search = (
        read(s9b / "text_training_search.json")
        if (s9b / "text_training_search.json").exists()
        else {}
    )
    result = dict(
        stage9c2_search_utc=search_time.isoformat(),
        share_reachable=os.path.isdir(SHARE),
        share_lidar_files=len(share),
        share_newest_utc=max((r["modified_utc"] for r in share), default=None),
        local_lidar_files=len(local),
        local_newest_utc=max((r["modified_utc"] for r in local), default=None),
        sources_newer_than_last_search=newer,
        stage9c2_verified_source_records=len(inventory["verified_source_paths"]),
        stage9c2_metadata_windows=inventory["metadata_candidate_windows"],
        stage9c2_reserved_footprints_checked=inventory["reserved_footprints_checked"],
        client_reference_text_features=prior_search.get("client_text_features_available", 0),
        approved_geometry_text_features=sum(approved_text.values()),
        approved_geometry_files_checked=len(approved_text),
        future_mixed_500_test_inventory_supplied=inventory[
            "future_mixed_500_test_inventory_supplied"
        ],
        rescan_performed=False,
        rescan_reason="none: no source newer than the Stage 9C.2 search and no new label source",
        real_text_training_candidates=[],
        qgis_text_review_package=None,
        n9_01_role="development only (never training)",
        decision="REAL_TEXT_TRAINING_DATA_UNAVAILABLE",
        new_methods_possible_with_user_approval=[
            "Model-assisted screening: rasterize unscreened LAZ corridors and use a text detector "
            "only to propose windows for human QGIS review. The Stage 9D detector is unsuitable "
            "as-is: 75% of N9_01 arrow pixels were called text (V1), so proposals would repeat "
            "the arrow-guided windows already inspected.",
            "Client-supplied road sections known to contain painted words/numbers (fastest).",
            "Commissioned annotation of text in new acquisitions, geographically separate from "
            "N9_01, T7, S8 and the future mixed 500-image test (whose footprint inventory must be "
            "supplied to verify separation).",
        ],
    )
    dump(out / "real_text_source_inventory.json", result)
    print(out, len(share), len(local), len(newer))


if __name__ == "__main__":
    main()
