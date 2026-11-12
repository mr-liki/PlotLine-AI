"""Per-tile Mask2Former labels: instance-id map, semantic map, label-state/ignore masks.

Hugging Face ``Mask2FormerImageProcessor`` consumes a per-pixel instance map plus an
``instance_id_to_semantic_id`` dict. Processor ``ignore_index`` excludes object IDs,
not pixels from stock mask losses; see ``verified_masks`` for the training gate.
Here: instance map 0 = no instance, 1..N = local instances; the semantic map holds
visual-class ids, 0 = verified background, 255 = ignore. Instance ids and class ids
are never the same array.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import shapely
from rasterio import features as rio_features
from shapely.geometry.base import BaseGeometry

from roadmarking.dataset.geometry import coverage_fraction, painted_area_from_centerline

# label_state codes (uint8)
NO_DATA = 0  # unobserved pixel, no instance -> ignore
UNKNOWN = 1  # observed, outside reviewed-complete regions -> ignore (not background)
BACKGROUND = 2  # observed, inside reviewed-complete region, no instance -> background
INSTANCE = 3  # confirmed marking instance pixel
BOUNDARY = 4  # partially covered instance boundary -> ignore
CONFLICT = 5  # two instances claim the pixel -> ignore
STATE_NAMES = {
    NO_DATA: "no_data_ignore",
    UNKNOWN: "unknown_ignore",
    BACKGROUND: "verified_background",
    INSTANCE: "instance",
    BOUNDARY: "boundary_ignore",
    CONFLICT: "overlap_conflict_ignore",
}


@dataclass
class LabelFeature:
    instance_uid: int  # stable global integer id
    global_id: str  # source file | layer | fid
    visual_id: int
    kind: str  # "line" | "polygon"
    geometry: BaseGeometry  # native XY, unmodified
    width: float | None = None  # physical painted width for lines (native units)
    group_id: str | None = None

    def painted_area(self) -> BaseGeometry:
        if self.kind == "line":
            if self.width is None:
                raise ValueError(f"{self.global_id}: centreline without verified width")
            return painted_area_from_centerline(self.geometry, self.width)
        return shapely.force_2d(self.geometry)


@dataclass
class TileLabels:
    instance: np.ndarray  # uint32, 0 none, 1..N local id
    semantic: np.ndarray  # uint8, 0 background, 255 ignore, else visual id
    state: np.ndarray  # uint8 label_state codes
    instances: list[dict] = field(default_factory=list)

    @property
    def ignore(self) -> np.ndarray:
        return (self.semantic == 255).astype(np.uint8)


def build_tile_labels(
    affine,
    shape: tuple[int, int],
    observed: np.ndarray,
    features: list[LabelFeature],
    completeness: BaseGeometry | None = None,
    supersample: int = 4,
    coverage_threshold: float = 0.5,
    boundary_ignore: bool = True,
    ignore_id: int = 255,
) -> TileLabels:
    """Rasterize verified features into one tile, deterministically ordered by uid."""
    observed = observed.astype(bool)
    claim = np.zeros(shape, np.uint32)  # local id of the instance owning a pixel
    claims = np.zeros(shape, np.uint8)  # number of instances claiming a pixel (>=0.5)
    partial = np.zeros(shape, bool)  # pixel touched by any instance below threshold
    records = []
    local = 0
    for feat in sorted(features, key=lambda f: f.instance_uid):
        area = feat.painted_area()
        result = coverage_fraction(area, affine, shape, supersample)
        if result is None:
            continue
        cov, (r0, r1, c0, c1) = result
        # Strictly more than the threshold: exactly half-covered pixels go to boundary-ignore.
        core = cov > coverage_threshold
        if not core.any():
            # Too thin to own any pixel centre at this resolution: ignore, never drop silently.
            partial[r0:r1, c0:c1] |= cov > 0
            records.append(_record(feat, 0, 0, cov, "below_coverage_threshold"))
            continue
        local += 1
        sub_claim = claim[r0:r1, c0:c1]
        sub_claim[core & (claims[r0:r1, c0:c1] == 0)] = local
        claims[r0:r1, c0:c1] += core.astype(np.uint8)
        partial[r0:r1, c0:c1] |= (cov > 0) & ~core
        records.append(_record(feat, local, int(core.sum()), cov, "ok"))

    conflict = claims > 1
    instance = np.where(conflict, 0, claim).astype(np.uint32)
    state = np.full(shape, UNKNOWN, np.uint8)
    if completeness is not None and not completeness.is_empty:
        inside = rio_features.rasterize(
            [(shapely.force_2d(completeness), 1)],
            out_shape=shape,
            transform=affine,
            fill=0,
            all_touched=False,
            dtype="uint8",
        ).astype(bool)
        state[inside] = BACKGROUND
    state[~observed] = NO_DATA
    if boundary_ignore:
        state[partial & (instance == 0)] = BOUNDARY
    state[instance > 0] = INSTANCE
    state[conflict] = CONFLICT

    semantic = np.full(shape, ignore_id, np.uint8)
    semantic[state == BACKGROUND] = 0
    lut = np.zeros(local + 1, np.uint8)
    for rec in records:
        if rec["local_id"]:
            lut[rec["local_id"]] = rec["visual_id"]
    semantic[state == INSTANCE] = lut[instance[state == INSTANCE]]
    # Final pixel counts after conflict removal.
    counts = np.bincount(instance.ravel(), minlength=local + 1)
    observed_counts = np.bincount(instance[observed].ravel(), minlength=local + 1)
    kept = []
    for rec in records:
        lid = rec["local_id"]
        if lid:
            rec["pixels"] = int(counts[lid])
            rec["observed_pixels"] = int(observed_counts[lid])
            if rec["pixels"] == 0:
                rec["status"] = "lost_to_overlap_conflict"
        kept.append(rec)
    return TileLabels(instance, semantic, state, kept)


def _record(feat: LabelFeature, local_id: int, pixels: int, cov: np.ndarray, status: str) -> dict:
    return {
        "local_id": local_id,
        "instance_uid": int(feat.instance_uid),
        "global_id": feat.global_id,
        "visual_id": int(feat.visual_id),
        "kind": feat.kind,
        "width": feat.width,
        "group_id": feat.group_id,
        "pixels": pixels,
        "coverage_area_px": float(cov.sum()),
        "status": status,
    }


def instance_id_to_semantic_id(labels: TileLabels) -> dict[int, int]:
    return {
        r["local_id"]: r["visual_id"] for r in labels.instances if r["local_id"] and r["pixels"]
    }
