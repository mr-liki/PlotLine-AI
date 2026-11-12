"""Realistic training data: new solid/dash markings painted into REAL labelled LiDAR tiles.

Background, pavement texture, point dropout, NoData and existing (reviewed) paint are real.
Only added markings are generated, at measured real width, contrast and dash rhythm. Labels:

* existing reviewed instances keep their reviewed labels; reviewed ignore regions stay ignored;
* an added stroke pixel is a target only if >= 50 % of it is covered (4x supersampling);
  partially covered pixels are ignored, never background;
* unobserved pixels (zero intensity, no LiDAR return) are never painted and never labelled
  (validity False), exactly like the reviewed real data;
* strokes are clipped to the observed road corridor and never touch existing paint or ignored
  regions (a conflicting stroke is discarded, not merged);
* optional unsupported markings (arrows, crosswalk/text, bike) come from real CC0 templates and
  are written to the semantic map but kept OUT of solid/dash supervision (validity False).
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, uniform_filter
from skimage.morphology import remove_small_holes

SIZE = 512
CELL_M = 0.05
SEM = dict(background=0, solid=1, dash=2, arrow=3, crosswalk_text=4, bike=5, ignore=9, nodata=255)
OREGON_TO_SEM = {2: "crosswalk_text", 3: "bike", 4: "arrow", 5: "arrow", 6: "arrow"}


@dataclass
class Tile:
    tile_id: str
    gray: np.ndarray  # uint8, 0 = unobserved
    valid: np.ndarray  # reviewed supervision support
    instances: list  # (mask, label)
    paint_pool: np.ndarray  # real paint intensities from this tile (may be empty)


@dataclass(frozen=True)
class GenConfig:
    markings: tuple = (1, 3)
    p_dash: float = 0.55
    width_m: tuple = (0.09, 0.13)
    lane_width_m: tuple = (3.0, 3.75)
    dash_patterns: tuple = (  # (dash range m, gap range m, weight) measured on real crops
        ((3.0, 3.5), (2.5, 4.0), 0.3),
        ((4.0, 4.5), (2.5, 4.0), 0.4),
        ((4.0, 4.5), (8.0, 10.0), 0.3),
    )
    p_curve: float = 0.25
    min_radius_m: float = 150.0
    p_fade: float = 0.25
    fade: tuple = (0.55, 0.9)
    dropout: tuple = (0.0, 0.015)
    visible_fringe_px: tuple = (0.8, 1.6)
    p_template: float = 0.35
    road_density: float = 0.35
    clearance_px: int = 6
    min_target_px: dict = field(default_factory=lambda: {0: 16, 1: 16})
    edge_margin_px: int = 20


def assemble_tile(tile_id, crops, load):
    """Rebuild one reviewed tile from its approved 512 crops. ``load(crop)`` -> npz dict."""
    h = max(c["window"][0] + c["window"][2] for c in crops)
    w = max(c["window"][1] + c["window"][3] for c in crops)
    gray = np.zeros((h, w), np.uint8)
    valid_and = np.ones((h, w), bool)
    seen = np.zeros((h, w), bool)
    inst, labels = {}, {}
    for c in crops:
        r, col, hh, ww = c["window"]
        d = load(c)
        sl = np.s_[r : r + hh, col : col + ww]
        g = d["rgb"][..., 0]
        assert not (seen[sl] & (gray[sl] != g)).any(), "inconsistent overlapping crops"
        gray[sl] = g
        valid_and[sl] &= d["validity"]
        seen[sl] = True
        assert len(d["labels"]) == len(c["global_instance_ids"])
        for m, lab, gid in zip(
            d["masks"].astype(bool), d["labels"], c["global_instance_ids"], strict=True
        ):
            if gid not in inst:
                inst[gid] = np.zeros((h, w), bool)
                labels[gid] = int(lab)
            assert labels[gid] == int(lab)
            inst[gid][sl] |= m
    valid = valid_and & seen
    instances = [(inst[g] & valid, labels[g]) for g in inst]
    paint = np.zeros((h, w), bool)
    for m, _ in instances:
        paint |= m
    pool = gray[paint & (gray > 0)]
    return Tile(tile_id, gray, valid, instances, pool)


def _road_surface(gray, observed, paint, density):
    """Paved surface: dense returns whose local texture matches the pavement beside reviewed
    lines (verges/vegetation are far more speckled). Empty when no reviewed line is present."""
    obs = observed.astype(np.float32)
    dens = uniform_filter(obs, 31)
    g = gray.astype(np.float32) * obs
    n = uniform_filter(obs, 15)
    mean = uniform_filter(g, 15) / np.maximum(n, 1e-3)
    sq = uniform_filter(g * g, 15) / np.maximum(n, 1e-3)
    std = np.sqrt(np.maximum(sq - mean**2, 0))
    ref = binary_dilation(paint, iterations=25) & ~binary_dilation(paint, iterations=4) & observed
    if ref.sum() < 200:
        return np.zeros_like(observed)
    std_hi = float(np.percentile(std[ref], 70)) * 1.15
    lo, hi = np.percentile(mean[ref], [5, 95])
    return (dens >= density) & (std <= std_hi) & (mean >= lo) & (mean <= hi)


def _direction(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) < 20:
        return None
    p = np.stack([xs, ys], 1).astype(float)
    p -= p.mean(0)
    _ev, evec = np.linalg.eigh(np.cov(p.T))
    return evec[:, 1] / np.linalg.norm(evec[:, 1])


def _centerline(anchor, u, curvature, length_px):
    n = np.array([-u[1], u[0]])
    s = np.arange(-length_px, length_px + 1, 1.0)
    pts = anchor[None] + s[:, None] * u[None] + (0.5 * curvature * s**2)[:, None] * n[None]
    return s, pts


def _draw(segments, width_px, shape, ss=4):
    """Coverage fraction per pixel for each segment (list of (N,2) xy polylines)."""
    out = []
    thick = max(1, round(width_px * ss))
    pad = int(np.ceil(width_px)) + 2
    for pts in segments:
        x0 = max(0, int(np.floor(pts[:, 0].min())) - pad)
        y0 = max(0, int(np.floor(pts[:, 1].min())) - pad)
        x1 = min(shape[1], int(np.ceil(pts[:, 0].max())) + pad + 1)
        y1 = min(shape[0], int(np.ceil(pts[:, 1].max())) + pad + 1)
        cov = np.zeros(shape, np.float32)
        if x1 <= x0 or y1 <= y0:
            out.append(cov)
            continue
        canvas = np.zeros(((y1 - y0) * ss, (x1 - x0) * ss), np.uint8)
        local = pts - np.array([x0, y0])
        poly = np.round((local + 0.5) * ss - 0.5).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [poly], False, 255, thickness=thick, lineType=cv2.LINE_8)
        cov[y0:y1, x0:x1] = cv2.resize(
            canvas.astype(np.float32) / 255, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA
        )
        out.append(cov)
    return out


def _paint(gray, cov, observed, base, rng, fade, dropout_rate, texture_sd=18.0):
    """Max-return raster model: a cell shows paint when any of its ~2-4 LiDAR returns hits the
    (visible) stroke, probability 1 - (1 - coverage)^n. Its value is the marking's real paint
    level (one draw from the tile's reviewed paint per marking) with mild per-cell texture,
    reduced only by wear/fading. Unhit cells keep the real background value."""
    sel = (cov > 0) & observed
    n = int(sel.sum())
    if not n:
        return gray
    paint = np.minimum(base + rng.normal(0, texture_sd, n), 255.0)
    returns = rng.integers(2, 5, n)
    hit = rng.random(n) < 1 - (1 - cov[sel]) ** returns
    keep = hit & (rng.random(n) >= dropout_rate)
    bg = gray[sel].astype(float)
    val = np.where(keep, bg + fade[sel] * (paint - bg), bg)
    out = gray.copy()
    out[sel] = np.clip(np.round(np.maximum(val, bg)), 1, 255).astype(np.uint8)
    return out


def _fade_field(shape, rng, lo, hi):
    noise = cv2.resize(
        rng.random((8, 8)).astype(np.float32), shape[::-1], interpolation=cv2.INTER_CUBIC
    )
    noise = (noise - noise.min()) / max(float(np.ptp(noise)), 1e-6)
    return lo + (hi - lo) * noise


def generate(tile, rng, global_pool, templates=(), config=None, window=None):
    """One 512x512 training sample.

    Returns dict(gray, masks, labels, validity, semantic, record).
    """
    cfg = config or GenConfig()
    h, w = tile.gray.shape
    oy = int(rng.integers(0, h - SIZE + 1)) if window is None else window[0]
    ox = int(rng.integers(0, w - SIZE + 1)) if window is None else window[1]
    sl = np.s_[oy : oy + SIZE, ox : ox + SIZE]
    k, flip = int(rng.integers(4)), bool(rng.random() < 0.5)

    def geo(a):
        a = np.rot90(a, k, axes=(-2, -1))
        return np.flip(a, -1) if flip else a

    gray = geo(tile.gray[sl]).copy()
    valid = geo(tile.valid[sl]).copy()
    observed = gray > 0
    masks, labels = [], []
    for m, lab in tile.instances:
        mm = geo(m[sl])
        if mm.sum() >= cfg.min_target_px[lab]:
            masks.append(mm.copy())
            labels.append(lab)
        else:
            valid &= ~mm
    existing = np.zeros_like(valid)
    for m in masks:
        existing |= m
    ignore = observed & ~valid
    semantic = np.full(gray.shape, SEM["background"], np.uint8)
    for m, lab in zip(masks, labels, strict=True):
        semantic[m] = SEM["solid"] if lab == 0 else SEM["dash"]
    semantic[ignore] = SEM["ignore"]
    semantic[~observed] = SEM["nodata"]
    road = _road_surface(gray, observed, existing, cfg.road_density)
    # Markings are placed inside the carriageway, never on the pavement/shoulder edge: a stroke
    # along an edge would teach "linear edge = paint", which reviewed data labels background.
    solid_road = remove_small_holes(road, max_size=4000)
    inner = binary_erosion(solid_road, iterations=cfg.edge_margin_px, border_value=1)
    blocked = binary_dilation(existing | ignore, iterations=cfg.clearance_px)
    free = road & valid & ~blocked & observed
    pool = tile.paint_pool if len(tile.paint_pool) >= 200 else global_pool
    u0 = None
    longest = max(((m.sum(), m) for m in masks), key=lambda t: t[0], default=(0, None))[1]
    if longest is not None:
        u0 = _direction(longest)
    if u0 is None:
        u0 = _direction(road) if road.sum() > 100 else np.array([1.0, 0.0])
    record = dict(tile=tile.tile_id, window=[oy, ox], rot90=k, flip=flip, added=[], rejected=0)
    n_new = int(rng.integers(cfg.markings[0], cfg.markings[1] + 1))
    attempts = 0
    lines = [m for m, _ in zip(masks, labels, strict=True) if m.sum() >= 20]
    while len(record["added"]) < n_new and attempts < 40 and free.any() and lines:
        attempts += 1
        # New lines sit at a measured lane offset from a reviewed real line, parallel to it.
        ref = lines[int(rng.integers(len(lines)))]
        u = _direction(ref)
        if u is None:
            break
        ang = np.deg2rad(rng.uniform(-1.5, 1.5))
        u = np.array(
            [u[0] * np.cos(ang) - u[1] * np.sin(ang), u[0] * np.sin(ang) + u[1] * np.cos(ang)]
        )
        nvec = np.array([-u[1], u[0]])
        ys, xs = np.nonzero(ref)
        i = int(rng.integers(len(ys)))
        lane = rng.uniform(*cfg.lane_width_m) / CELL_M
        anchor = np.array([xs[i], ys[i]], float) + nvec * lane * rng.choice([-1, 1])
        ai, aj = round(anchor[1]), round(anchor[0])
        if not (0 <= ai < SIZE and 0 <= aj < SIZE and free[ai, aj] and inner[ai, aj]):
            record["rejected"] += 1
            continue
        curv = 0.0
        if rng.random() < cfg.p_curve:
            curv = rng.choice([-1, 1]) / (
                rng.uniform(cfg.min_radius_m, 4 * cfg.min_radius_m) / CELL_M
            )
        s, pts = _centerline(anchor, u, curv, SIZE)
        inside = (
            (pts[:, 0] >= -2) & (pts[:, 0] < SIZE + 2) & (pts[:, 1] >= -2) & (pts[:, 1] < SIZE + 2)
        )
        s, pts = s[inside], pts[inside]
        kind = "dash" if rng.random() < cfg.p_dash else "solid"
        if kind == "dash":
            pats = cfg.dash_patterns
            j = rng.choice(len(pats), p=np.array([p[2] for p in pats]) / sum(p[2] for p in pats))
            dl = rng.uniform(*pats[j][0]) / CELL_M
            gl = rng.uniform(*pats[j][1]) / CELL_M
            phase = rng.uniform(0, dl + gl)
            pos = np.mod(s - s.min() + phase, dl + gl)
            on = pos < dl
            starts = np.flatnonzero(on & ~np.r_[False, on[:-1]])
            segs = []
            for st in starts:
                en = st
                while en + 1 < len(on) and on[en + 1]:
                    en += 1
                if en - st >= 2:
                    segs.append(pts[st : en + 1])
        else:
            segs = [pts]
        width_px = rng.uniform(*cfg.width_m) / CELL_M
        covs = _draw(segs, width_px, gray.shape)
        # Measured on reviewed data: paint is visible ~1 px beyond its annotation (bright fringe).
        fringe_px = rng.uniform(*cfg.visible_fringe_px)
        vis = _draw(segs, width_px + fringe_px, gray.shape)
        union = np.zeros(gray.shape, np.float32)
        for cv in covs + vis:
            union = np.maximum(union, cv)
        touched = union > 0
        # A stroke may never meet existing paint (no crossing or merging lines): discard it.
        if (touched & binary_dilation(existing, iterations=2)).any():
            record["rejected"] += 1
            continue
        on_obs = touched & observed
        if on_obs.sum() == 0 or (on_obs & inner).sum() < 0.9 * on_obs.sum():
            record["rejected"] += 1  # mostly off the paved surface: not a plausible marking
            continue
        # Clip to the observed road corridor and keep clear of reviewed ignore regions.
        corridor = road & ~binary_dilation(ignore, iterations=2)
        covs = [cv * corridor for cv in covs]
        vis = [cv * corridor for cv in vis]
        base = float(np.percentile(pool, rng.uniform(35, 90)))
        fade = (
            _fade_field(gray.shape, rng, *cfg.fade)
            if rng.random() < cfg.p_fade
            else np.ones(gray.shape, np.float32)
        )
        drop = float(rng.uniform(*cfg.dropout))
        added = []
        for cv, vv in zip(covs, vis, strict=True):
            target = (cv >= 0.5) & observed & valid
            if target.sum() < cfg.min_target_px[0 if kind == "solid" else 1]:
                continue
            gray = _paint(gray, vv, observed, base, rng, fade, drop)
            # Partially covered and visible-fringe cells are ignored, never background.
            partial = (((cv > 0) & (cv < 0.5)) | ((vv > 0) & (cv < 0.5))) & observed
            valid &= ~partial
            target &= valid
            added.append(target)
        if not added:
            record["rejected"] += 1
            continue
        lab = 0 if kind == "solid" else 1
        if kind == "solid":
            m = np.zeros_like(valid)
            for t in added:
                m |= t
            added = [m]
        for t in added:
            masks.append(t)
            labels.append(lab)
            semantic[t] = SEM["solid"] if lab == 0 else SEM["dash"]
            existing |= t
        semantic[(union > 0) & observed & ~valid & (semantic == SEM["background"])] = SEM["ignore"]
        blocked = binary_dilation(existing | ignore, iterations=cfg.clearance_px)
        free &= ~blocked
        record["added"].append(
            dict(
                kind=kind,
                width_px=float(width_px),
                curvature=float(curv),
                pieces=len(added),
                faded=bool(fade.min() < 1),
                dropout=drop,
            )
        )
    if templates and rng.random() < cfg.p_template and free.any():
        gray, valid, semantic, info = _place_template(
            gray, valid, semantic, free, observed, u0, pool, templates, rng
        )
        if info:
            record["template"] = info
    masks = [m & valid for m in masks]
    return dict(
        gray=gray,
        masks=np.asarray(masks, bool).reshape(len(masks), SIZE, SIZE),
        labels=np.asarray(labels, np.int64),
        validity=valid,
        semantic=semantic,
        record=record,
    )


def _place_template(gray, valid, semantic, free, observed, u, pool, templates, rng):
    """Paste one real unsupported marking (arrow/crosswalk/bike), oriented to the road."""
    t = templates[int(rng.integers(len(templates)))]
    mask, inten = t["mask"], t["intensity"]
    ang = np.degrees(np.arctan2(u[1], u[0])) - t["angle_deg"] + rng.choice([0, 180])
    hh, ww = mask.shape
    m = cv2.getRotationMatrix2D((ww / 2, hh / 2), -ang, 1.0)
    side = int(np.ceil(np.hypot(hh, ww)))
    m[0, 2] += side / 2 - ww / 2
    m[1, 2] += side / 2 - hh / 2
    rm = cv2.warpAffine(mask.astype(np.uint8), m, (side, side), flags=cv2.INTER_NEAREST) > 0
    ri = cv2.warpAffine(inten.astype(np.float32), m, (side, side), flags=cv2.INTER_NEAREST)
    if side >= SIZE:
        return gray, valid, semantic, None
    ys, xs = np.nonzero(free)
    for _ in range(10):
        i = int(rng.integers(len(ys)))
        y0, x0 = ys[i] - side // 2, xs[i] - side // 2
        if y0 < 0 or x0 < 0 or y0 + side > SIZE or x0 + side > SIZE:
            continue
        win = np.s_[y0 : y0 + side, x0 : x0 + side]
        if not free[win][rm].all():
            continue
        q = np.argsort(np.argsort(ri[rm])) / max(rm.sum() - 1, 1)
        vals = np.quantile(pool, np.clip(q, 0, 1))
        sel = rm & observed[win]
        g = gray[win].astype(float)
        g[rm] = np.where(observed[win][rm], np.maximum(vals, g[rm]), 0)
        gray[win] = np.clip(np.round(g), 0, 255).astype(np.uint8)
        gray[win][~observed[win]] = 0
        semantic[win][sel] = SEM[t["semantic"]]
        valid[win] &= ~binary_dilation(rm, iterations=2)
        return (
            gray,
            valid,
            semantic,
            dict(semantic=t["semantic"], source=t["source"], angle=float(ang)),
        )
    return gray, valid, semantic, None


__all__ = ["SEM", "GenConfig", "Tile", "assemble_tile", "generate"]
