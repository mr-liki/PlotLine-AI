"""Development loop: cached rasterization + recognition, then tracing and scoring in seconds.

Usage:
  python scripts/dev_trace_eval.py <regions.json> [--names a b ...] [--surface low|all|top]
         [--single] [--stage recognition|trace] [--save DIR]

Scores are measured against the client's lines for the WHOLE region (not only the part one
file happened to scan), so missing coverage counts as a miss. Development/review areas only.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
import shapely
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree
from shapely.geometry import LineString, box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from roadmarking.digitization.raster_tiles import (  # noqa: E402
    make_grid,
    rasterize_region,
    xy_to_pixel,
)
from roadmarking.digitization.stages import (  # noqa: E402
    build_levels,
    level_pieces,
    piece_features,
    recognize_cached,
    trace_features,
)
from roadmarking.digitization.surface import Surface  # noqa: E402

CACHE = ROOT / "outputs/digitization/_cache"
ANN = ROOT / "data/raw/annotations"
SOLID = {"Guia direita", "Guia esquerda", "Linha contínua", "Linha contínua na sequência de LBTg"}
DASH = {
    "Linha descontínua",
    "Linha de aceleração",
    "Linha de abrandamento",
    "Linha descontínua de aviso",
}
_REFS = None


def reference_lines(bounds):
    """Client longitudinal lines clipped to ``bounds``: list of (LineString 2D, class, name)."""
    global _REFS
    if _REFS is None:
        _REFS = pd.concat(
            [
                pyogrio.read_dataframe(ANN / "Highway/Longitudinal Marks .gpkg"),
                pyogrio.read_dataframe(ANN / "Interchange/Interchange-Longitudinal Marks.gpkg"),
            ]
        )
        _REFS = _REFS[_REFS.geometry.notna()]
    b = box(*bounds)
    out = []
    sub = _REFS[_REFS.intersects(b)]
    for name, g in zip(sub["DESIGNACAO"], sub.geometry, strict=True):
        cls = 0 if name in SOLID else (1 if name in DASH else -1)
        for part in g.geoms:
            clipped = LineString(np.asarray(part.coords)[:, :2]).intersection(b)
            for p in getattr(clipped, "geoms", [clipped]):
                if p.geom_type == "LineString" and p.length > 0.05:
                    out.append((p, cls, name))
    return out


def sample(refs, step=0.1):
    pts, cls = [], []
    for g, c, _ in refs:
        d = np.arange(step / 2, g.length, step)
        pts += [g.interpolate(v).coords[0] for v in d]
        cls += [c] * len(d)
    return np.asarray(pts).reshape(-1, 2), np.asarray(cls)


def at(mask, grid, xy):
    r, c = xy_to_pixel(grid, xy[:, 0], xy[:, 1])
    r, c = np.round(r).astype(int), np.round(c).astype(int)
    ok = (r >= 0) & (c >= 0) & (r < grid.rows) & (c < grid.cols)
    out = np.zeros(len(xy), bool)
    out[ok] = mask[r[ok], c[ok]]
    return out


def load_region(region, surface="low", single=False):
    grid = make_grid(region["bbox"])
    las = region["las"] if isinstance(region["las"], list) else [region["las"]]
    ras = rasterize_region(
        las[:1] if single else las, grid, cache_dir=CACHE / "extracts", surface=surface
    )
    return grid, ras


def class_masks(shape, instances):
    solid, dash = np.zeros(shape, bool), np.zeros(shape, bool)
    for inst in instances:
        (solid if inst["class"] == 0 else dash)[...] |= inst["mask"]
    return solid, dash


def recognition_coverage(grid, ras, instances):
    """Share of client line length covered by accepted model paint (any class / right class)."""
    solid, dash = class_masks(ras["gray"].shape, instances)
    refs = reference_lines(grid.bounds)
    pts, cls = sample(refs)

    def near(mask):
        return at(binary_dilation(mask, iterations=3), grid, pts)

    seen, any_, s_, d_ = near(ras["observed"]), near(solid | dash), near(solid), near(dash)
    res = dict(ref_m=len(pts) / 10, seen=float(seen.mean()), model_any=float(any_.mean()))
    for name, c, m in (("solid", 0, s_), ("dash", 1, d_)):
        sel = cls == c
        res[name] = dict(
            ref_m=float(sel.sum() / 10),
            any=float(any_[sel].mean()) if sel.any() else None,
            right=float(m[sel].mean()) if sel.any() else None,
        )
    return res


def feature_points(features, step=0.1):
    pts, cls, fid = [], [], []
    for k, f in enumerate(features):
        g = f["geometry_2d"]
        d = np.arange(step / 2, g.length, step) if g.length > step else np.array([g.length / 2])
        pts += [g.interpolate(v).coords[0][:2] for v in d]
        cls += [f["class_id"]] * len(d)
        fid += [k] * len(d)
    return np.asarray(pts).reshape(-1, 2), np.asarray(cls), np.asarray(fid)


def score(features, refs, tol=0.30):
    """Line metrics against ALL client lines of the region (lengths in metres)."""
    rp, rc = sample(refs)
    op, oc, _ = feature_points(features)
    res = dict(ref_m=len(rp) / 10, our_m=len(op) / 10, features=len(features))
    if not len(op) or not len(rp):
        return res
    d_any = cKDTree(op).query(rp)[0]
    res["found_any"] = float((d_any <= tol).mean())
    for name, c in (("solid", 0), ("dash", 1)):
        sel = rc == c
        part = dict(ref_m=float(sel.sum() / 10), found_any=None, found=None, lateral_cm=None)
        part["precision"] = None
        ours = op[oc == c]
        if sel.any():
            part["found_any"] = float((d_any[sel] <= tol).mean())
            part["found"] = 0.0
            if len(ours):
                d = cKDTree(ours).query(rp[sel])[0]
                part["found"] = float((d <= tol).mean())
                if (d <= tol).any():
                    part["lateral_cm"] = float(100 * np.median(d[d <= tol]))
                part["precision"] = float((cKDTree(rp[sel]).query(ours)[0] <= tol).mean())
        res[name] = part
    res["on_client_line"] = float((cKDTree(rp).query(op)[0] <= tol).mean())
    lines = [f["geometry_2d"] for f in features]
    tree = shapely.STRtree(lines)
    overlaps = conflicts = 0
    for i, g in enumerate(lines):
        buf = g.buffer(0.12)
        for j in tree.query(buf, predicate="intersects"):
            if j > i and lines[j].intersection(buf).length > 0.3:
                overlaps += 1
                conflicts += features[i]["class_id"] != features[j]["class_id"]
    res.update(overlaps=overlaps, conflicts=conflicts)
    ref_d = [g for g, c, _ in refs if c == 1]
    our_d = [f["geometry_2d"] for f in features if f["class_id"] == 1]
    matched, ends = 0, []
    if ref_d and our_d:
        tree = shapely.STRtree(our_d)
        used = set()
        for r in ref_d:
            best = None
            zone = r.buffer(tol)
            for j in tree.query(zone, predicate="intersects"):
                if j in used:
                    continue
                inter = our_d[j].intersection(zone).length
                if inter >= 0.5 * min(r.length, our_d[j].length) and (
                    best is None or inter > best[0]
                ):
                    best = (inter, j)
            if best:
                used.add(best[1])
                matched += 1
                o = our_d[best[1]]
                a0, a1 = np.asarray(r.coords[0]), np.asarray(r.coords[-1])
                b0, b1 = np.asarray(o.coords[0])[:2], np.asarray(o.coords[-1])[:2]
                ends.append(
                    min(
                        np.linalg.norm(a0 - b0) + np.linalg.norm(a1 - b1),
                        np.linalg.norm(a0 - b1) + np.linalg.norm(a1 - b0),
                    )
                    / 2
                )
    res["dash"] = dict(
        res.get("dash", {}),
        ref_n=len(ref_d),
        our_n=len(our_d),
        matched_n=matched,
        end_error_cm=float(100 * np.median(ends)) if ends else None,
    )
    solids = [f for f in features if f["class_id"] == 0 and f["length_m"] >= 5]
    res["jitter_rms_mm"] = (
        float(np.median([f["jitter_rms_mm"] for f in solids])) if solids else None
    )
    res["jitter_max_mm"] = (
        float(np.max([f["jitter_max_mm"] for f in features])) if features else None
    )
    res["inferred_m"] = float(sum(f["inferred_length_m"] for f in features))
    res["paint_traced_m"] = float(sum(f["paint_traced_m"] for f in features))
    return res


def trace_region(grid, ras, instances):
    solid, dash = class_masks(ras["gray"].shape, instances)
    pieces = piece_features(instances, ras["gray"], ras["observed"], grid)
    level = dict(
        gray=ras["gray"],
        observed=ras["observed"],
        solid=solid,
        dash=dash,
        height=ras.get("height"),
        surface=Surface(ras["points"]),
        name=ras.get("surface", "low"),
    )
    features, audit = trace_features(pieces["features"], [level], grid)
    return features, audit, pieces


def render(path, grid, gray, refs, features, window=None, size=1600, misses=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x0, y0, x1, y1 = grid.bounds
    fig, ax = plt.subplots(figsize=(size / 100, size / 100), dpi=100)
    ax.imshow(gray, cmap="gray", extent=[x0, x1, y0, y1], vmin=0, vmax=255, interpolation="nearest")
    for g, c, _ in refs:
        xy = np.asarray(g.coords)
        color = "yellow" if c >= 0 else "lime"
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=5, alpha=0.3, solid_capstyle="butt")
    for f in features:
        xy = np.asarray(f["geometry_2d"].coords)
        color = "red" if f["class_id"] == 0 else "cyan"
        if "paint_only_detection" in f["flags"]:
            color = "orange"
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.2)
        ax.plot(xy[[0, -1], 0], xy[[0, -1], 1], ".", color=color, ms=4)
        for sg in f["inferred_segments"]:
            c = np.asarray(sg.coords)
            ax.plot(c[:, 0], c[:, 1], color="magenta", lw=2.5, ls=":")
    if misses is not None and len(misses):
        ax.plot(misses[:, 0], misses[:, 1], "x", color="deepskyblue", ms=3, alpha=0.7)
    w = window or (x0, y0, x1, y1)
    ax.set_xlim(w[0], w[2])
    ax.set_ylim(w[1], w[3])
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path)
    plt.close(fig)


def fmt(v, scale=100, width=5, digits=1):
    return " " * (width - 1) + "-" if v is None else f"{scale * v:{width}.{digits}f}"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("regions")
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--surface", default="low")
    ap.add_argument("--single", action="store_true", help="first source file only (old behaviour)")
    ap.add_argument("--stage", default="recognition", choices=["recognition", "trace"])
    ap.add_argument("--save", help="folder for overlay images and traced features")
    a = ap.parse_args()
    regions = json.loads(Path(a.regions).read_text())
    for region in regions:
        if a.names and region["name"] not in a.names:
            continue
        t = time.perf_counter()
        grid, ras = load_region(region, a.surface, a.single)
        instances, _, _, cached = recognize_cached(None, ras["gray"], ras["observed"], grid, CACHE)
        cov = recognition_coverage(grid, ras, instances)
        head = (
            f"{region['name']:26s} {a.surface:4s} {'single' if a.single else 'fused':6s}"
            f" files {len(ras['sources'])}"
        )
        if a.stage == "recognition":
            print(
                f"{head} seen {fmt(cov['seen'])} model-any {fmt(cov['model_any'])}"
                f" | solid any {fmt(cov['solid']['any'])} right {fmt(cov['solid']['right'])}"
                f" | dash any {fmt(cov['dash']['any'])} right {fmt(cov['dash']['right'])}"
                f" | inst {len(instances)} {'cache' if cached else 'gpu'}"
                f" {time.perf_counter() - t:.0f}s",
                flush=True,
            )
            continue
        t1 = time.perf_counter()
        las = region["las"] if isinstance(region["las"], list) else [region["las"]]
        levels, _, _ = build_levels(las[:1] if a.single else las, grid, CACHE)
        seeds = level_pieces(levels, grid)
        features, audit = trace_features(seeds, levels, grid)
        pieces = dict(features=seeds)
        gray_all = np.maximum.reduce([lv["gray"] for lv in levels])
        ras = dict(ras, gray=gray_all)
        head += f" lv {len(levels)}"
        refs = reference_lines(grid.bounds)
        sc = score(features, refs)
        so, da = sc.get("solid", {}), sc.get("dash", {})
        print(
            f"{head} | model s/d {fmt(cov['solid']['right'])}/{fmt(cov['dash']['right'])}"
            f" | SOLID found {fmt(so.get('found'))} (any {fmt(so.get('found_any'))})"
            f" prec {fmt(so.get('precision'))} off {fmt(so.get('lateral_cm'), 1, 4)}cm"
            f" | DASH found {fmt(da.get('found'))} (any {fmt(da.get('found_any'))})"
            f" prec {fmt(da.get('precision'))} off {fmt(da.get('lateral_cm'), 1, 4)}cm"
            f" n {da.get('matched_n')}/{da.get('ref_n')} ours {da.get('our_n')}"
            f" end {fmt(da.get('end_error_cm'), 1, 4)}cm"
            f" | on-client {fmt(sc.get('on_client_line'))}"
            f" | overlap {sc.get('overlaps')} conflict {sc.get('conflicts')}"
            f" | jitter {fmt(sc.get('jitter_rms_mm'), 1, 4, 2)}"
            f"/{fmt(sc.get('jitter_max_mm'), 1, 5)}mm"
            f" | feats {len(features)} (pieces {audit['seed_pieces']}+{audit['paint_only_seeds']}"
            f" seeds, held {audit['held_untraced']})"
            f" inferred {sc.get('inferred_m', 0):.1f}m traced {sc.get('paint_traced_m', 0):.0f}m"
            f" | {time.perf_counter() - t1:.0f}s",
            flush=True,
        )
        if a.save:
            out = Path(a.save)
            out.mkdir(parents=True, exist_ok=True)
            rp, _ = sample(refs)
            op, _, _ = feature_points(features)
            misses = rp[cKDTree(op).query(rp)[0] > 0.30] if len(op) else rp
            name = region["name"]
            render(out / f"{name}_overview.png", grid, ras["gray"], refs, features, misses=misses)
            x0, y0, x1, y1 = grid.bounds
            n = 4
            dx, dy = (x1 - x0) / n, (y1 - y0) / n
            for i in range(n):
                for j in range(n):
                    w = (x0 + i * dx, y0 + j * dy, x0 + (i + 1) * dx, y0 + (j + 1) * dy)
                    render(
                        out / f"{name}_x{i}y{j}.png",
                        grid,
                        ras["gray"],
                        refs,
                        features,
                        window=w,
                        size=1300,
                        misses=misses,
                    )
            with (out / f"{name}_features.pkl").open("wb") as fh:
                pickle.dump(
                    dict(features=features, audit=audit, score=sc, pieces=pieces["features"]), fh
                )


if __name__ == "__main__":
    main()
