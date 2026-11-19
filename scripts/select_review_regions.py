"""Pick diverse digitization review regions with client reference lines, away from all
protected/acceptance/test footprints (+100 m), the acceptance source files and earlier pilots.

Writes outputs/digitization/review_regions_<date>.json for ``digitize.py --regions``.
"""

import json
import sys
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import pyogrio
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from digitize import DASH, SOLID, protected_footprints  # noqa: E402

ANN = ROOT / "data/raw/annotations"
SIZE = 102.4
ACCEPTANCE_FILES = {
    p.name for p in (ROOT / "data/raw/pointclouds/acceptance_v1_source").glob("*.laz")
}
USED_FILES = ACCEPTANCE_FILES | {"Job_20250708_0834_Track01_11.laz"}


def main(out_name):
    protected, _ = protected_footprints()
    keep_out = protected.buffer(100)
    cov = json.loads(
        (ROOT / "outputs/reports/realistic_v1_training/las_free_coverage.json").read_text()
    )
    lines = pd.concat(
        [
            pyogrio.read_dataframe(ANN / "Highway/Longitudinal Marks .gpkg"),
            pyogrio.read_dataframe(ANN / "Interchange/Interchange-Longitudinal Marks.gpkg"),
        ]
    )
    lines = lines[lines.geometry.notna() & lines["DESIGNACAO"].isin(SOLID | DASH)]
    other = pd.concat(
        [
            pyogrio.read_dataframe(ANN / "Highway/Other Marks.gpkg"),
            pyogrio.read_dataframe(ANN / "Interchange/interchange_Other Marks.gpkg"),
        ]
    )
    other = other[other.geometry.notna()]
    sidx, oidx = lines.sindex, other.sindex
    cands = []
    for r in cov:
        name = PureWindowsPath(r["file"]).name
        group = PureWindowsPath(r["file"]).parent.name
        if name in USED_FILES or group == "backup":
            continue
        fb = box(*r["bounds"]).buffer(-SIZE / 2 - 5)
        if fb.is_empty:
            continue
        sub = lines.iloc[sidx.query(fb, predicate="intersects")]
        for g in sub.geometry[::7]:
            for part in g.geoms:
                for d in np.arange(0, part.length, 50.0):
                    p = part.interpolate(d)
                    if not fb.contains(p):
                        continue
                    b = box(p.x - SIZE / 2, p.y - SIZE / 2, p.x + SIZE / 2, p.y + SIZE / 2)
                    if b.intersects(keep_out):
                        continue
                    inb = lines.iloc[sidx.query(b, predicate="intersects")]
                    if len(inb) < 5:
                        continue
                    clipped = inb.geometry.intersection(b)
                    solid_len = float(clipped[inb["DESIGNACAO"].isin(SOLID).values].length.sum())
                    dash_n = int(inb["DESIGNACAO"].isin(DASH).sum())
                    straight = []
                    for gg in clipped:
                        for pp in getattr(gg, "geoms", [gg]):
                            if pp.length > 20:
                                c = np.asarray(pp.coords)[:, :2]
                                straight.append(np.linalg.norm(c[-1] - c[0]) / pp.length)
                    curvature = float(1 - min(straight)) if straight else 0.0
                    others = len(oidx.query(b, predicate="intersects"))
                    cands.append(
                        dict(
                            file=name,
                            group=group,
                            center=[p.x, p.y],
                            bbox=list(b.bounds),
                            solid_m=solid_len,
                            dashes=dash_n,
                            curvature=curvature,
                            other_marks=others,
                        )
                    )
    df = pd.DataFrame(cands)
    df = df.loc[~df["center"].map(tuple).duplicated()]
    picks = []

    def take(sel, why):
        for _, row in sel.iterrows():
            b = box(*row.bbox)
            if row.file in {p["file"] for p in picks}:
                continue
            if any(b.distance(box(*p["bbox"])) < 300 for p in picks):
                continue
            picks.append(dict(row.to_dict(), why=why))
            return

    take(
        df[df.group == "LEFT"].sort_values(["solid_m", "dashes"], ascending=False),
        "LEFT carriageway, long solid + dashes",
    )
    take(
        df[df.group == "INTERCHANGE"].sort_values("curvature", ascending=False), "Interchange curve"
    )
    take(
        df.sort_values("other_marks", ascending=False),
        "Junction with arrows/other markings (confusion test)",
    )
    take(
        df[df.group == "RIGHT"].sort_values(["dashes", "solid_m"], ascending=False),
        "RIGHT carriageway, different stretch",
    )
    take(
        df[df.group == "LEFT"].sort_values("curvature", ascending=False),
        "LEFT carriageway, curved section",
    )
    jobs = []
    for k, p in enumerate(picks, 1):
        p["name"] = f"review{k}_{p['group'].lower()}_{p['why'].split(',')[0].split()[0].lower()}"
        jobs.append(
            dict(
                las=str(ROOT / "data/raw/pointclouds/digitization_review_sources" / p["file"]),
                bbox=p["bbox"],
                name=p["name"],
                why=p["why"],
            )
        )
    out = ROOT / "outputs/digitization" / out_name
    out.write_text(json.dumps(jobs, indent=1))
    print(
        pd.DataFrame(picks)[
            ["name", "file", "solid_m", "dashes", "curvature", "other_marks", "why"]
        ].to_string()
    )
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "review_regions_20261113.json")
