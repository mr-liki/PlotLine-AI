"""Geometry-only extraction with explicit candidate uncertainty, never lowest-ground."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from roadmarking.analysis.lidar_stats import GridSpec


@dataclass(frozen=True)
class RoadCriteria:
    cell_m: float = 1.0
    min_points: int = 8
    max_rms_m: float = 0.045
    max_slope: float = 0.20
    max_neighbor_step_m: float = 0.20
    point_residual_m: float = 0.10
    min_component_m2: float = 100.0
    max_grid_cells: int = 8_000_000


class SurfaceAccumulator:
    """Exact local moments, with local XYZ origins to avoid cancellation.

    OLS fits use all points, rejecting mixed/rough cells conservatively. This is
    deliberately not a ground classifier. Tiny planar roofs are removed by area;
    other smooth surfaces remain uncertain without a user-supplied road ROI.
    """

    def __init__(self, header, criteria=None, bounds=None):
        self.criteria = criteria or RoadCriteria()
        lo, hi = header.mins.copy(), header.maxs.copy()
        if bounds is not None:
            lo[:2], hi[:2] = bounds[:2], bounds[2:]
        self.spec = GridSpec.covering(lo, hi, self.criteria.cell_m)
        self.shape = (self.spec.ny, self.spec.nx)
        if np.prod(self.shape) > self.criteria.max_grid_cells:
            raise ValueError("Coarse grid exceeds memory budget; supply --bounds/ROI")
        self.z_origin = float(header.offsets[2])
        self.moments = np.zeros((10, np.prod(self.shape)), np.float64)

    def add(self, xyz):
        s = self.spec
        inside = s.inside(xyz[:, 0], xyz[:, 1])
        xyz = xyz[inside]
        r, c = s.index(xyz[:, 0], xyz[:, 1])
        x0, y0 = s.cell_center(r, c)
        x, y, z = xyz[:, 0] - x0, xyz[:, 1] - y0, xyz[:, 2] - self.z_origin
        ids, inv = np.unique(r * s.nx + c, return_inverse=True)
        for dest, val in zip(
            self.moments,
            (np.ones(len(z)), x, y, z, x * x, x * y, x * z, y * y, y * z, z * z),
            strict=True,
        ):
            dest[ids] += np.bincount(inv, weights=val, minlength=len(ids))

    def finish(self, roi=None):
        n, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz = self.moments
        nn = np.maximum(n, 1)
        mx, my, mz = sx / nn, sy / nn, sz / nn
        xx, xy, yy = sxx / nn - mx * mx, sxy / nn - mx * my, syy / nn - my * my
        xz, yz = sxz / nn - mx * mz, syz / nn - my * mz
        det = xx * yy - xy * xy
        safe = np.where(det > 1e-10, det, 1)
        a, b = (xz * yy - yz * xy) / safe, (yz * xx - xz * xy) / safe
        intercept = mz - a * mx - b * my + self.z_origin
        rms = np.sqrt(np.maximum(0, szz / nn - mz * mz - a * xz - b * yz))
        slope = np.hypot(a, b)
        cr = self.criteria
        good = (
            (n >= cr.min_points) & (det > 1e-10) & (rms <= cr.max_rms_m) & (slope <= cr.max_slope)
        ).reshape(self.shape)
        z = intercept.reshape(self.shape)
        # At a discontinuity, do not join a vehicle/roof to a road. Missing/rough
        # neighbors do not penalize real road edges; no wraparound at array borders.
        for axis in (0, 1):
            u = [slice(None), slice(None)]
            v = [slice(None), slice(None)]
            u[axis], v[axis] = slice(1, None), slice(None, -1)
            u, v = tuple(u), tuple(v)
            jump = good[u] & good[v] & (np.abs(z[u] - z[v]) > cr.max_neighbor_step_m)
            gu, gv = good[u].copy(), good[v].copy()
            good[u], good[v] = gu & ~jump, gv & ~jump
        lab, count = ndimage.label(good)
        sizes = np.bincount(lab.ravel(), minlength=count + 1)
        keep = sizes * cr.cell_m**2 >= cr.min_component_m2
        keep[0] = False
        candidate = keep[lab]
        state = candidate.astype(np.uint8)  # 0 rejected, 1 uncertain, 2 ROI-confirmed
        if roi is not None:
            import shapely

            rr, cc = np.indices(self.shape)
            x, y = self.spec.cell_center(rr, cc)
            in_roi = shapely.contains_xy(roi, x, y)
            state[candidate & in_roi] = 2
            state[~in_roi] = 0
        return SurfaceModel(
            self.spec,
            cr,
            state,
            a.reshape(self.shape),
            b.reshape(self.shape),
            z,
            rms.reshape(self.shape),
            n.reshape(self.shape).astype(np.uint32),
            roi,
        )


@dataclass
class SurfaceModel:
    spec: GridSpec
    criteria: RoadCriteria
    state: np.ndarray
    a: np.ndarray
    b: np.ndarray
    z: np.ndarray
    rms: np.ndarray
    count: np.ndarray
    roi: object = None

    def classify(self, xyz):
        s = self.spec
        inside = s.inside(xyz[:, 0], xyz[:, 1])
        r, c = s.index(xyz[:, 0], xyz[:, 1])
        x, y = s.cell_center(r, c)
        predicted = self.z[r, c] + self.a[r, c] * (xyz[:, 0] - x) + self.b[r, c] * (xyz[:, 1] - y)
        ok = inside & (np.abs(xyz[:, 2] - predicted) <= self.criteria.point_residual_m)
        if self.roi is not None:
            import shapely

            ok &= shapely.contains_xy(self.roi, xyz[:, 0], xyz[:, 1])
        return np.where(ok, self.state[r, c], 0).astype(np.uint8)

    def state_at(self, x, y):
        r, c = self.spec.index(x, y)
        inside = self.spec.inside(x, y)
        if self.roi is not None:
            import shapely

            inside &= shapely.contains_xy(self.roi, x, y)
        return np.where(inside, self.state[r, c], 0)


# --------------------------------------------------------------------------------------
# Phase 3.1 corridor refinement ("v2"). Geometry only; intensity is never consulted.
# --------------------------------------------------------------------------------------

# Per-cell evidence categories (descriptions of algorithm evidence, not ground truth).
CONF_NO_OBSERVATION = 0
CONF_REJECTED = 1
CONF_INSUFFICIENT = 2
CONF_UNCERTAIN = 3
CONF_HIGH = 4
CONF_ROI_CONFIRMED = 5
CONFIDENCE_NAMES = {
    CONF_NO_OBSERVATION: "no_observation",
    CONF_REJECTED: "rejected_non_road_evidence",
    CONF_INSUFFICIENT: "insufficient_observations",
    CONF_UNCERTAIN: "uncertain_requires_review",
    CONF_HIGH: "high_confidence_geometric_candidate",
    CONF_ROI_CONFIRMED: "inside_approved_roi",
}

# Evidence bit flags stored per cell (uint16).
EV_CORE = 1  # Phase 3 (v1) candidate in a large corridor component
EV_SMALL_COMPONENT = 2  # v1 candidate in a small disconnected component
EV_EDGE_REFINED = 4  # robust inlier fit against the extrapolated corridor plane passed
EV_ENCLOSED = 8  # surrounded by corridor cells (scan gap / object inside the road)
EV_FEW_INLIERS = 16
EV_INLIER_RMS = 32
EV_SLOPE_CHANGE = 64
EV_OFFSET = 128
EV_SMOOTH_NEAR_CORRIDOR = 256  # strict cell fit passes but disconnected from corridor
EV_OUTSIDE_ROI = 512
EV_ROI = 1024
EV_PROMOTED = 2048  # coplanar v1 component linked to the corridor via verified cells
EV_DISCONTINUITY = 4096  # v1 cell with a plane step/slope break against a neighbour
EVIDENCE_NAMES = {
    EV_CORE: "v1_corridor_core",
    EV_SMALL_COMPONENT: "v1_small_component",
    EV_EDGE_REFINED: "edge_inlier_fit_passed",
    EV_ENCLOSED: "enclosed_by_corridor",
    EV_FEW_INLIERS: "too_few_inliers",
    EV_INLIER_RMS: "inlier_rms_too_high",
    EV_SLOPE_CHANGE: "slope_change_too_high",
    EV_OFFSET: "offset_from_corridor_plane",
    EV_SMOOTH_NEAR_CORRIDOR: "smooth_surface_near_corridor",
    EV_OUTSIDE_ROI: "outside_approved_roi",
    EV_ROI: "inside_approved_roi",
    EV_PROMOTED: "coplanar_component_promoted",
    EV_DISCONTINUITY: "plane_discontinuity_with_neighbour",
}


@dataclass(frozen=True)
class RefinementCriteria:
    """Phase 3.1 corridor refinement parameters (native units; provisionally metres).

    * ``core_min_area_m2``: v1 components at least this large form the corridor core;
      smaller v1 components are kept as UNCERTAIN instead of automatic road.
    * ``edge_buffer_cells``: rejected/sparse cells within this many cells of the core are
      re-tested point-by-point against the core plane extrapolated from the nearest core
      cell. This recovers mixed cells (road + barrier/kerb/vehicle) without relaxing any
      threshold for the cell as a whole.
    * ``inlier_band_m``: |z - corridor plane| for a point to be a surface inlier.
    * ``edge_min_inliers`` / ``enclosed_min_inliers``: inlier count needed (lower inside
      the corridor, where sparse scan lines occur), never zero.
    * ``edge_max_rms_m``, ``max_slope_change``, ``max_offset_m``: the inliers' own plane
      must be smooth, parallel and continuous with the corridor plane (allows banking and
      grade changes up to these limits).
    * ``parallel_search_cells``: strictly smooth cells (v1 cell tests) outside the
      corridor but within this distance are reported as UNCERTAIN parallel surfaces
      (e.g. the opposite carriageway behind a barrier, or a footpath/roof). Never auto.
    """

    core_min_area_m2: float = 500.0
    corridor_min_elongation: float = 3.0
    corridor_max_mean_width_m: float = 40.0
    edge_buffer_cells: int = 2
    inlier_band_m: float = 0.06
    edge_min_inliers: int = 8
    enclosed_min_inliers: int = 3
    enclosed_min_core_neighbors: int = 5
    edge_max_rms_m: float = 0.03
    max_slope_change: float = 0.06
    max_offset_m: float = 0.04
    continuity_step_m: float = 0.05
    parallel_search_cells: int = 30
    include_uncertain: bool = False


def surface_discontinuities(model, cr):
    """v1 candidate cells whose fitted plane is discontinuous with a candidate neighbour.

    For each 4-neighbour pair both planes are evaluated at the shared cell edge. A gap
    larger than ``continuity_step_m`` (a kerb) or a slope change larger than
    ``max_slope_change`` (a plane absorbing a step) marks both cells. The v1 rule
    compared centre heights with a 0.20 m limit, which lets 10-15 cm kerbs join
    footpaths to the road. Marked cells are not discarded: they are re-tested
    point-by-point by the edge inlier test, so smooth roads are not eroded.
    """
    cand = model.state > 0
    half = model.criteria.cell_m / 2
    z, a, b = model.z, model.a, model.b
    marks = np.zeros(cand.shape, bool)
    # columns: +x; rows: -y
    pairs = (
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None)), a * half, -a * half),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None)), -b * half, b * half),
    )
    for u, v, du, dv in pairs:
        both = cand[u] & cand[v]
        gap = np.abs((z[u] + du[u]) - (z[v] + dv[v]))
        dslope = np.hypot(a[u] - a[v], b[u] - b[v])
        bad = both & ((gap > cr.continuity_step_m) | (dslope > cr.max_slope_change))
        marks[u] |= bad
        marks[v] |= bad
    return marks


def corridor_shape(rows, cols, cell):
    """(length, mean width) of a cell component: length along its principal axis,
    mean width = area / length. Curved corridors stay elongated; broad fields do not."""
    pts = np.column_stack((cols, rows)).astype(np.float64) * cell
    if len(pts) < 3:
        return cell, cell
    centred = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    proj = centred @ vt[0]
    length = float(proj.max() - proj.min()) + cell
    return length, len(pts) * cell**2 / length


class CorridorRefiner:
    """Second streaming pass: robust inlier moments against the extrapolated corridor plane.

    Only cells in the edge zone accumulate. Residuals are taken against the plane of the
    nearest core cell, so a vehicle, kerb or barrier inside a 1 m cell becomes outliers
    instead of making the whole cell fail. Coordinates are local to cell centres.
    """

    def __init__(self, model: SurfaceModel, criteria: RefinementCriteria | None = None):
        self.base = model
        self.cr = criteria or RefinementCriteria()
        self.spec = model.spec
        shape = model.state.shape
        self.v1_candidate = model.state > 0
        self.breaks = surface_discontinuities(model, self.cr)
        candidate = self.v1_candidate & ~self.breaks
        lab, count = ndimage.label(candidate, structure=np.ones((3, 3)))
        cell = model.criteria.cell_m
        sizes = np.bincount(lab.ravel(), minlength=count + 1) * cell**2
        big = sizes >= self.cr.core_min_area_m2
        big[0] = False
        self.components = []
        for k in np.flatnonzero(big):
            rows, cols = np.nonzero(lab == k)
            length, width = corridor_shape(rows, cols, cell)
            corridor = (
                width <= self.cr.corridor_max_mean_width_m
                and length / max(width, 1e-9) >= self.cr.corridor_min_elongation
            )
            big[k] = corridor
            self.components.append(
                dict(
                    area=float(sizes[k]),
                    length=float(length),
                    mean_width=float(width),
                    corridor=bool(corridor),
                )
            )
        self.core = big[lab]
        # v1 cells outside the corridor core (including discontinuity cells) must earn
        # inclusion through the edge inlier test or coplanar promotion.
        self.small = candidate & ~self.core
        if self.core.any():
            dist, (ri, ci) = ndimage.distance_transform_edt(~self.core, return_indices=True)
        else:
            dist = np.full(shape, np.inf)
            ri, ci = np.indices(shape)
        self.dist = dist
        self.ref_r, self.ref_c = ri.astype(np.int32), ci.astype(np.int32)
        self.edge_zone = (~self.core) & (dist <= self.cr.edge_buffer_cells) & (model.count > 0)
        self.n = np.zeros(shape, np.float64)  # inlier moments of residual plane fit
        self.m = np.zeros((9, *shape), np.float64)
        self.above = np.zeros(shape, np.uint32)
        self.below = np.zeros(shape, np.uint32)

    def reference_residual(self, xyz, r, c):
        """z minus the plane of the nearest core cell, evaluated at the point."""
        b = self.base
        rr, cc = self.ref_r[r, c], self.ref_c[r, c]
        xr, yr = self.spec.cell_center(rr, cc)
        pred = b.z[rr, cc] + b.a[rr, cc] * (xyz[:, 0] - xr) + b.b[rr, cc] * (xyz[:, 1] - yr)
        return xyz[:, 2] - pred

    def _reference_at_centres(self, r, c):
        b = self.base
        rr, cc = self.ref_r[r, c], self.ref_c[r, c]
        dr, dc = (r - rr) * self.spec.cell, (c - cc) * self.spec.cell
        # +col is +x, +row is -y
        return b.z[rr, cc] + b.a[rr, cc] * dc - b.b[rr, cc] * dr

    def add(self, xyz):
        s = self.spec
        inside = s.inside(xyz[:, 0], xyz[:, 1])
        xyz = xyz[inside]
        r, c = s.index(xyz[:, 0], xyz[:, 1])
        zone = self.edge_zone[r, c]
        if not zone.any():
            return
        xyz, r, c = xyz[zone], r[zone], c[zone]
        res = self.reference_residual(xyz, r, c)
        band = self.cr.inlier_band_m
        np.add.at(self.above, (r[res > band], c[res > band]), 1)
        np.add.at(self.below, (r[res < -band], c[res < -band]), 1)
        keep = np.abs(res) <= band
        xyz, r, c, res = xyz[keep], r[keep], c[keep], res[keep]
        x0, y0 = s.cell_center(r, c)
        x, y = xyz[:, 0] - x0, xyz[:, 1] - y0
        flat = r * s.nx + c
        ids, inv = np.unique(flat, return_inverse=True)
        n = self.n.reshape(-1)
        m = self.m.reshape(9, -1)
        n[ids] += np.bincount(inv, minlength=len(ids))
        for k, val in enumerate((x, y, res, x * x, x * y, x * res, y * y, y * res, res * res)):
            m[k, ids] += np.bincount(inv, weights=val, minlength=len(ids))

    def finish(self, roi=None) -> RefinedSurfaceModel:
        cr, base = self.cr, self.base
        shape = base.state.shape
        n = self.n
        nn = np.maximum(n, 1)
        sx, sy, sr, sxx, sxy, sxr, syy, syr, srr = self.m
        mx, my, mr = sx / nn, sy / nn, sr / nn
        xx, xy, yy = sxx / nn - mx * mx, sxy / nn - mx * my, syy / nn - my * my
        xr, yr = sxr / nn - mx * mr, syr / nn - my * mr
        det = xx * yy - xy * xy
        ok_det = det > 1e-6
        safe = np.where(ok_det, det, 1)
        ga = np.where(ok_det, (xr * yy - yr * xy) / safe, 0.0)
        gb = np.where(ok_det, (yr * xx - xr * xy) / safe, 0.0)
        offset = mr - ga * mx - gb * my  # residual plane at the cell centre
        rms = np.sqrt(np.maximum(0, srr / nn - mr * mr - ga * xr - gb * yr))
        slope_change = np.hypot(ga, gb)
        core_neighbors = ndimage.convolve(
            self.core.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant"
        ) - self.core.astype(np.uint8)
        # Enclosed = topological holes in the corridor (scan gaps, vehicles) or cells
        # mostly surrounded by it. Empty cells inside holes still produce no pixels.
        holes = ndimage.binary_fill_holes(self.core) & ~self.core
        enclosed = self.edge_zone & ((core_neighbors >= cr.enclosed_min_core_neighbors) | holes)
        need = np.where(enclosed, cr.enclosed_min_inliers, cr.edge_min_inliers)
        few = n < need
        # Slope change is only meaningful with a well-conditioned inlier fit.
        slope_bad = ok_det & (n >= cr.edge_min_inliers) & (slope_change > cr.max_slope_change)
        rms_bad = rms > cr.edge_max_rms_m
        off_bad = np.abs(offset) > cr.max_offset_m
        edge_ok = self.edge_zone & ~few & ~rms_bad & ~slope_bad & ~off_bad

        # Promotion: a smaller v1 component joins the corridor only if (a) it touches the
        # corridor through verified edge cells and (b) its own cell planes are coplanar
        # with the extrapolated corridor plane (median |offset| <= max_offset_m). A raised
        # footpath or a lower parallel road therefore stays UNCERTAIN.
        promoted = np.zeros(shape, bool)
        if self.small.any():
            linked = self.core | edge_ok
            lab, k = ndimage.label(self.small, structure=np.ones((3, 3)))
            touch = ndimage.binary_dilation(linked, structure=np.ones((3, 3)))
            rr, cc = np.indices(shape)
            ref_off = base.z - self._reference_at_centres(rr, cc)
            for comp in range(1, k + 1):
                cells = lab == comp
                if not (cells & touch).any():
                    continue
                if np.median(np.abs(ref_off[cells])) <= cr.max_offset_m:
                    promoted |= cells
        self.promoted = promoted

        evidence = np.zeros(shape, np.uint16)
        evidence[self.core] |= EV_CORE
        evidence[self.small] |= EV_SMALL_COMPONENT
        evidence[promoted] |= EV_PROMOTED
        evidence[self.breaks] |= EV_DISCONTINUITY
        evidence[edge_ok] |= EV_EDGE_REFINED
        evidence[enclosed] |= EV_ENCLOSED
        z = self.edge_zone
        evidence[z & few] |= EV_FEW_INLIERS
        evidence[z & ~few & rms_bad] |= EV_INLIER_RMS
        evidence[z & ~few & slope_bad] |= EV_SLOPE_CHANGE
        evidence[z & ~few & off_bad] |= EV_OFFSET

        bc = base.criteria
        strict = (
            (base.count >= bc.min_points)
            & (base.rms <= bc.max_rms_m)
            & (np.hypot(base.a, base.b) <= bc.max_slope)
        )
        near = self.dist <= cr.parallel_search_cells
        parallel = strict & near & ~self.core & ~edge_ok & ~promoted
        evidence[parallel] |= EV_SMOOTH_NEAR_CORRIDOR

        conf = np.full(shape, CONF_NO_OBSERVATION, np.uint8)
        observed = base.count > 0
        conf[observed] = CONF_REJECTED
        # Too few points to judge. Cells with many points but few inliers (barrier,
        # vegetation, vehicle) stay REJECTED: that is evidence, not missing data.
        conf[observed & (base.count < bc.min_points)] = CONF_INSUFFICIENT
        conf[parallel | (self.small & ~promoted)] = CONF_UNCERTAIN
        conf[self.core | edge_ok | promoted] = CONF_HIGH
        selected = conf == CONF_HIGH
        if cr.include_uncertain:
            selected |= conf == CONF_UNCERTAIN
        state = selected.astype(np.uint8)  # 1 automatic candidate, 2 ROI-confirmed
        if roi is not None:
            import shapely

            rr, cc = np.indices(shape)
            x, y = self.spec.cell_center(rr, cc)
            in_roi = shapely.contains_xy(roi, x, y)
            supported = in_roi & np.isin(conf, (CONF_HIGH, CONF_UNCERTAIN))
            evidence[in_roi] |= EV_ROI
            evidence[~in_roi & (conf >= CONF_UNCERTAIN)] |= EV_OUTSIDE_ROI
            conf[supported] = CONF_ROI_CONFIRMED
            state[:] = 0
            state[supported] = 2
        return RefinedSurfaceModel(
            base=base,
            criteria=cr,
            state=state,
            confidence=conf,
            evidence=evidence,
            core=self.core | promoted,
            edge_ok=edge_ok,
            ref_r=self.ref_r,
            ref_c=self.ref_c,
            inliers=n.astype(np.uint32),
            inlier_rms=rms.astype(np.float32),
            inlier_offset=offset.astype(np.float32),
            slope_change=slope_change.astype(np.float32),
            above=self.above,
            below=self.below,
            roi=roi,
        )


@dataclass
class RefinedSurfaceModel:
    base: SurfaceModel
    criteria: RefinementCriteria
    state: np.ndarray  # 0 none, 1 automatic candidate, 2 ROI-confirmed (v1 semantics)
    confidence: np.ndarray  # CONF_* per cell
    evidence: np.ndarray  # EV_* bit flags per cell
    core: np.ndarray
    edge_ok: np.ndarray
    ref_r: np.ndarray
    ref_c: np.ndarray
    inliers: np.ndarray
    inlier_rms: np.ndarray
    inlier_offset: np.ndarray
    slope_change: np.ndarray
    above: np.ndarray
    below: np.ndarray
    roi: object = None

    @property
    def spec(self):
        return self.base.spec

    def classify(self, xyz):
        """Point state (0/1/2). Core cells: the v1 per-cell plane test (unchanged).
        Edge cells: |z - corridor plane| <= inlier band (outliers such as barrier
        points stay excluded even inside an accepted cell)."""
        s = self.spec
        inside = s.inside(xyz[:, 0], xyz[:, 1])
        r, c = s.index(xyz[:, 0], xyz[:, 1])
        x, y = s.cell_center(r, c)
        b = self.base
        own = b.z[r, c] + b.a[r, c] * (xyz[:, 0] - x) + b.b[r, c] * (xyz[:, 1] - y)
        ok_own = np.abs(xyz[:, 2] - own) <= b.criteria.point_residual_m
        rr, cc = self.ref_r[r, c], self.ref_c[r, c]
        xr, yr = s.cell_center(rr, cc)
        ref = b.z[rr, cc] + b.a[rr, cc] * (xyz[:, 0] - xr) + b.b[rr, cc] * (xyz[:, 1] - yr)
        ok_ref = np.abs(xyz[:, 2] - ref) <= self.criteria.inlier_band_m
        edge = self.edge_ok[r, c] & ~self.core[r, c]
        ok = inside & np.where(edge, ok_ref, ok_own)
        if self.roi is not None:
            import shapely

            ok &= shapely.contains_xy(self.roi, xyz[:, 0], xyz[:, 1])
        return np.where(ok, self.state[r, c], 0).astype(np.uint8)

    def _at(self, grid, x, y):
        r, c = self.spec.index(x, y)
        inside = self.spec.inside(x, y)
        return np.where(inside, grid[r, c], 0)

    def state_at(self, x, y):
        out = self._at(self.state, x, y)
        if self.roi is not None:
            import shapely

            out = np.where(shapely.contains_xy(self.roi, x, y), out, 0)
        return out

    def confidence_at(self, x, y):
        return self._at(self.confidence, x, y)

    def arrays(self) -> dict:
        return dict(
            state=self.state,
            confidence=self.confidence,
            evidence=self.evidence,
            core=self.core,
            edge_ok=self.edge_ok,
            inliers=self.inliers,
            inlier_rms=self.inlier_rms,
            inlier_offset=self.inlier_offset,
            slope_change=self.slope_change,
            above=self.above,
            below=self.below,
        )


def extract_road_surface(cloud, ground_classes=None, *, criteria=None, roi=None):
    """Return geometry candidates; inspect model states for semantic confidence."""
    acc = SurfaceAccumulator(cloud.header, criteria)
    acc.add(cloud.xyz)
    return acc.finish(roi).classify(cloud.xyz) > 0


def remove_noise(cloud, mean_k=8, std_multiplier=2.5):
    from scipy.spatial import cKDTree

    if len(cloud) < 2:
        return np.ones(len(cloud), bool)
    distances, _ = cKDTree(cloud.xyz).query(cloud.xyz, k=min(mean_k + 1, len(cloud)))
    mean = distances[:, 1:].mean(axis=1)
    return mean <= mean.mean() + std_multiplier * mean.std()
