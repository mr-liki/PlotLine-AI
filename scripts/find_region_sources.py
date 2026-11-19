"""Find every source LAS/LAZ file that contributes points to each digitization region.

The client specification requires all tracks covering a road part to be used (both sides of
the road). A region is usually crossed by several acquisitions: the two driving directions,
repeated passes and interchange ramps. This tool reads file headers to shortlist candidates,
then streams each candidate ONCE (read-only) and stores the points of every region it touches
in the extract cache, so ``digitize.py`` can fuse the sources without reading the files again.

Usage:
  python scripts/find_region_sources.py <regions.json> --search DIR [DIR ...]
         [--out <regions_multi.json>] [--min-points 20000] [--workers 3]

``--search`` directories are tried in order; a file name found in an earlier directory wins
(put local copies first, network shares last). Names containing "backup" are skipped.
"""

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import laspy
import numpy as np
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from roadmarking.digitization.raster_tiles import extract_regions, make_grid  # noqa: E402

CACHE = ROOT / "outputs/digitization/_cache"


def header_index(dirs):
    """{file name: (path, [minx, miny, maxx, maxy])} for every LAS/LAZ below ``dirs``."""
    index = {}
    for d in map(Path, dirs):
        store = CACHE / f"las_index_{hashlib.sha1(str(d).lower().encode()).hexdigest()[:10]}.json"
        listing = sorted(
            p for p in d.rglob("*.la[sz]") if "backup" not in str(p.relative_to(d)).lower()
        )
        known = json.loads(store.read_text()) if store.exists() else {}
        for p in listing:
            key = f"{p}|{p.stat().st_size}"
            if key not in known:
                with laspy.open(p) as f:
                    h = f.header
                    known[key] = [
                        float(h.mins[0]),
                        float(h.mins[1]),
                        float(h.maxs[0]),
                        float(h.maxs[1]),
                    ]
            index.setdefault(p.name, (str(p), known[key]))
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text(json.dumps(known))
    return index


def scan_file(path, bounds_list):
    extracts = extract_regions(path, bounds_list, cache_dir=CACHE / "extracts")
    out = []
    for e, (minx, miny, maxx, maxy) in zip(extracts, bounds_list, strict=True):
        inside = (e["x"] >= minx) & (e["x"] < maxx) & (e["y"] > miny) & (e["y"] <= maxy)
        cells = 0.0
        if inside.any():
            gx = ((e["x"][inside] - minx) // 0.5).astype(np.int64)
            gy = ((e["y"][inside] - miny) // 0.5).astype(np.int64)
            n = int(np.ceil((maxx - minx) / 0.5)) + 1
            cells = len(np.unique(gy * n + gx)) / (((maxx - minx) / 0.5) * ((maxy - miny) / 0.5))
        out.append(dict(points=int(inside.sum()), cells_05m=float(cells)))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("regions")
    ap.add_argument("--search", nargs="+", required=True)
    ap.add_argument("--out")
    ap.add_argument("--min-points", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    regions = json.loads(Path(a.regions).read_text())
    index = header_index(a.search)
    print(f"{len(index)} source files indexed", flush=True)
    bounds = [make_grid(r["bbox"]).bounds for r in regions]
    todo = {}
    for name, (path, hb) in index.items():
        hit = [k for k, b in enumerate(bounds) if box(*hb).intersects(box(*b))]
        if hit:
            todo[name] = (path, hit)
    print(f"{len(todo)} candidate files by header bounds", flush=True)
    found = {k: [] for k in range(len(regions))}
    with ProcessPoolExecutor(a.workers) as pool:
        jobs = {
            pool.submit(scan_file, path, [bounds[k] for k in hit]): (name, path, hit)
            for name, (path, hit) in todo.items()
        }
        for job in as_completed(jobs):
            name, path, hit = jobs[job]
            for k, res in zip(hit, job.result(), strict=True):
                found[k].append(dict(name=name, path=path, **res))
                print(
                    f"{regions[k]['name']:32s} {name:40s} {res['points']:>10d} pts"
                    f"  {100 * res['cells_05m']:5.1f} % cells",
                    flush=True,
                )
    out = []
    for k, r in enumerate(regions):
        primary = [r["las"]] if isinstance(r["las"], str) else list(r["las"])
        names = {Path(p).name for p in primary}
        extra = sorted(
            (f for f in found[k] if f["name"] not in names and f["points"] >= a.min_points),
            key=lambda f: -f["points"],
        )
        out.append(
            dict(
                r,
                las=primary + [f["path"] for f in extra],
                sources_scan=sorted(found[k], key=lambda f: -f["points"]),
            )
        )
        print(r["name"], "->", [Path(p).name for p in out[-1]["las"]])
    dst = Path(a.out) if a.out else Path(a.regions).with_name(Path(a.regions).stem + "_multi.json")
    dst.write_text(json.dumps(out, indent=1))
    print(dst)


if __name__ == "__main__":
    main()
