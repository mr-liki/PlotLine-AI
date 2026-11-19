"""Re-check every place the user reported in review against fresh outputs.

Usage: python scripts/check_reported_cases.py <output.gpkg> [<output.gpkg> ...]

Each output is matched to its cases by the source file recorded in its metadata table
(plotline_run_metadata). Prints PASS/FAIL per case and exits non-zero on any failure.
"""

import json
import sys
from pathlib import Path

import pyogrio
import shapely

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "tests/reference/reported_cases.json"


def source_stems(gpkg):
    meta = pyogrio.read_dataframe(gpkg, layer="plotline_run_metadata")
    row = meta[meta.key == "source_files"]
    if row.empty:
        return set()
    return {Path(f["path"].replace("\\", "/")).stem for f in json.loads(row.value.iloc[0])}


_CLIENT = {}


def client_iou(p, polys, r):
    """IoU between our polygon at ``p`` and the client's mark at ``p`` (0 if either missing)."""
    if "marks" not in _CLIENT:
        df = pyogrio.read_dataframe(ROOT / "data/raw/annotations/Highway/Other Marks.gpkg")
        _CLIENT["marks"] = [shapely.force_2d(g) for g in df.geometry]
    ours = [g for g in polys if g.distance(p) <= r]
    theirs = [g for g in _CLIENT["marks"] if g.distance(p) <= r]
    if not ours or not theirs:
        return 0.0
    return max(a.intersection(b).area / a.union(b).area for a in ours for b in theirs)


def check(case, lines, polys):
    p = shapely.Point(case["x"], case["y"])
    r = case["r"]
    if case["kind"] == "client_iou_at":
        return client_iou(p, polys, r) >= case["min"]
    if case["kind"] in ("line_at", "no_line_at"):
        hit = any(g.distance(p) <= r for g in lines)
        return hit if case["kind"] == "line_at" else not hit
    hit = any(g.distance(p) <= r for g in polys)
    return hit if case["kind"] == "polygon_at" else not hit


def main(paths):
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    failed = checked = 0
    for gpkg in paths:
        stems = source_stems(gpkg)
        mine = [c for c in cases if c["file"] in stems]
        if not mine:
            print(f"{gpkg}: no reported cases for {sorted(stems)}")
            continue
        lines = [
            shapely.force_2d(g) for g in pyogrio.read_dataframe(gpkg, layer="Highway").geometry
        ]
        polys = [
            shapely.force_2d(g) for g in pyogrio.read_dataframe(gpkg, layer="Other Marks").geometry
        ]
        for c in mine:
            ok = check(c, lines, polys)
            checked += 1
            failed += not ok
            mark = "PASS" if ok else "FAIL"
            where = f"{c['x']:.1f},{c['y']:.1f}"
            print(f"{mark}  {c['file']}  {c['kind']:13s} {where}  {c['what']}")
    print(f"{checked - failed}/{checked} reported cases pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
