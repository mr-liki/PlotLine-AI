"""Shape-preserving diagnostic registration; a fit is not surveyed alignment."""

import numpy as np
import shapely
from scipy.spatial import cKDTree


def rigid_solution(source, target):
    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2 or len(source) < 3:
        raise ValueError("At least three corresponding XY points required")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Finite XY coordinates required")
    a, b = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - a).T @ (target - b))
    if np.linalg.det(vt.T @ u.T) < 0:
        vt[-1] *= -1
    rotation = vt.T @ u.T
    return rotation, b - rotation @ a


def apply_xy(points, rotation, translation):
    return np.asarray(points) @ np.asarray(rotation).T + np.asarray(translation)


def transform_geometry_xy(geometry, rotation, translation):
    """Avoid 0*NaN Z leaking into XY; preserve source dimensionality and Z."""
    if geometry is None:
        return None

    def transform_coordinates(coordinates):
        result = coordinates.copy()
        result[:, :2] = apply_xy(coordinates[:, :2], rotation, translation)
        return result

    include_z = int(shapely.get_coordinate_dimension(geometry)) >= 3
    return shapely.transform(geometry, transform_coordinates, include_z=include_z)


def grid_sample(points, cell_size):
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points) or cell_size <= 0:
        raise ValueError("Nonempty XY array and positive grid size required")
    _, indices = np.unique(np.floor(points / cell_size).astype(np.int64), axis=0, return_index=True)
    return points[np.sort(indices)]


def distance_stats(points, reference):
    distances = cKDTree(reference).query(points)[0]
    return {
        "points": len(distances),
        "rms": float(np.sqrt(np.mean(distances**2))),
        **{f"p{p}": float(np.percentile(distances, p)) for p in (50, 75, 90, 95, 99)},
        "maximum": float(distances.max()),
        **{
            f"within_{threshold:g}_pct": float(100 * np.mean(distances <= threshold))
            for threshold in (0.05, 0.25, 1, 5, 10, 25, 50, 100)
        },
    }


def fit_rigid_multistart(source, target, *, trim=0.9, iterations=80):
    """Deterministic trimmed ICP across PCA-angle/translation seeds.

    Nearest samples are geometric proxies, not verified common landmarks.
    Only proper rigid rotations are allowed: no reflection, scale or warping.
    """
    source, target = np.asarray(source, float), np.asarray(target, float)
    if len(source) < 10 or len(target) < 10 or not 0.5 <= trim <= 1:
        raise ValueError("Enough points and a valid trim fraction required")
    ac, bc = source.mean(axis=0), target.mean(axis=0)
    _, _, av = np.linalg.svd(source - ac, full_matrices=False)
    _, _, bv = np.linalg.svd(target - bc, full_matrices=False)
    axis_a, axis_b = av[0], bv[0]
    angle = np.arctan2(axis_b[1], axis_b[0]) - np.arctan2(axis_a[1], axis_a[0])
    tree = cKDTree(target)
    outcomes = []
    for flip in (0, np.pi):
        for delta in np.deg2rad([-30, -15, 0, 15, 30]):
            theta = angle + flip + delta
            initial = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            for shift in (-5000, 0, 5000):
                rotation = initial.copy()
                translation = bc - rotation @ ac + shift * axis_b
                previous = float("inf")
                for _step in range(iterations):
                    distances, indices = tree.query(apply_xy(source, rotation, translation))
                    keep = distances <= np.quantile(distances, trim)
                    rotation, translation = rigid_solution(source[keep], target[indices[keep]])
                    objective = float(np.mean(np.square(distances[keep])))
                    if abs(previous - objective) < 1e-6:
                        break
                    previous = objective
                distances = tree.query(apply_xy(source, rotation, translation))[0]
                score = float(
                    np.mean(np.square(np.sort(distances)[: max(3, int(len(distances) * trim))]))
                )
                outcomes.append(
                    {
                        "rotation": rotation,
                        "translation": translation,
                        "score": score,
                        "iterations": _step + 1,
                    }
                )
    best = min(outcomes, key=lambda r: r["score"])
    best["candidates"] = len(outcomes)
    best["all_seed_scores"] = sorted(float(r["score"]) for r in outcomes)
    return best
