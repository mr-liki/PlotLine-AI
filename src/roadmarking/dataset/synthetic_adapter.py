"""Conservative procedural-mask adaptation with no fabricated geospatial scale."""

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label


def ambiguous_components(parts, minimum_gap):
    bounds = []
    for _, part in parts:
        rows, cols = np.where(part)
        bounds.append((rows.min(), cols.min(), rows.max(), cols.max()))
    for i, (_, a) in enumerate(parts):
        distances = None
        for j, (_, b) in enumerate(parts[:i]):
            ay0, ax0, ay1, ax1 = bounds[i]
            by0, bx0, by1, bx1 = bounds[j]
            dx, dy = max(0, ax0 - bx1, bx0 - ax1), max(0, ay0 - by1, by0 - ay1)
            if dx * dx + dy * dy >= minimum_gap * minimum_gap:
                continue
            if distances is None:
                distances = distance_transform_edt(~a)
            if distances[b].min() < minimum_gap:
                return True
    return False


def adapt(image, instance, semantic, objects, minimum_area=12, minimum_span=6, minimum_gap=4):
    image, instance, semantic = map(np.asarray, (image, instance, semantic))
    if image.ndim != 2 or image.shape != instance.shape or image.shape != semantic.shape:
        raise ValueError("Aligned grayscale, instance and semantic arrays required")
    if not np.isin(semantic, np.arange(17)).all():
        raise ValueError("Unknown semantic label")
    if not np.array_equal(instance == 0, semantic == 0):
        raise ValueError("Foreground/background instance-semantic disagreement")
    valid = image > 0  # Generator explicitly voids raster dropout to zero; never interpolate it.
    valid &= ~binary_dilation(semantic >= 3, iterations=2)
    masks, classes, metadata = [], [], []
    declared = {int(obj["instance_id"]): obj for obj in objects}
    if set(np.unique(instance)) - {0} != set(declared):
        raise ValueError("Instance IDs and metadata disagree")
    for uid, obj in declared.items():
        raw = instance == uid
        cls = int(obj["class_id"])
        if not np.all(semantic[raw] == cls + 1) or raw.sum() != obj["pixel_count"]:
            raise ValueError("Metadata/class/area mismatch")
        if cls not in (0, 1):
            continue
        paint = raw & valid
        components, count = label(paint, structure=np.ones((3, 3), int))
        retained = []
        for component in range(1, count + 1):
            part = components == component
            rows, cols = np.where(part)
            if part.sum() < minimum_area or max(np.ptp(rows), np.ptp(cols)) < minimum_span:
                valid[part] = False
            else:
                retained.append((component, part))
        # Close observed fragments may be dropout pieces of one dash. Ignore the original
        # group instead of inventing a physical split or connecting through missing pixels.
        if cls == 0 and ambiguous_components(retained, minimum_gap):
            valid[raw] = False
            continue
        # A solid source identity survives observation holes.
        parts = [(0, paint & valid)] if cls == 1 else retained
        for component, part in parts:
            if part.sum() < minimum_area:
                valid[part] = False
                continue
            masks.append(part)
            classes.append(1 if cls == 0 else 0)
            metadata.append(
                {
                    "original_group_id": uid,
                    "component": component,
                    "synthetic_class": cls,
                    "model_class": classes[-1],
                }
            )
    targets = np.stack(masks) if masks else np.empty((0, *image.shape), bool)
    targets &= valid
    keep = targets.sum((1, 2)) >= minimum_area
    if (~keep).any():
        valid[targets[~keep].any(0)] = False
    return (
        targets[keep],
        np.asarray(classes, dtype=np.int64)[keep],
        valid,
        [m for m, k in zip(metadata, keep, strict=True) if k],
    )
