"""Data-derived local domains for probing learned latent charts.

The domain is a union of local balls around encoded physical observations.  It
does not assert that a chart is convex, simply connected, or centred at zero:
annular and disconnected-looking latent supports are therefore represented
without filling their holes.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _neighbour_count(points: np.ndarray, neighbours: int | None) -> int:
    """Local rank grows with latent dimension instead of becoming directional."""
    if len(points) < 2:
        return 0
    requested = max(8, 4 * points.shape[1]) if neighbours is None else int(neighbours)
    return min(max(1, requested), len(points) - 1)


def local_support_radii(points: np.ndarray, *, neighbours: int | None = None) -> np.ndarray:
    """Return one data-driven local support radius for every latent point."""
    points = np.asarray(points, float)
    if len(points) == 0:
        return np.empty(0)
    if len(points) == 1:
        return np.asarray([0.0])
    k = _neighbour_count(points, neighbours)
    return cKDTree(points).query(points, k=k + 1)[0][:, -1].clip(1e-12)


def in_local_support(points: np.ndarray, candidates: np.ndarray, *, neighbours: int | None = None):
    """Classify candidates against the union of data-derived local balls."""
    points, candidates = np.asarray(points, float), np.asarray(candidates, float)
    if len(points) == 0:
        return np.zeros(len(candidates), bool)
    radii = local_support_radii(points, neighbours=neighbours)
    distance, nearest = cKDTree(points).query(candidates, k=1)
    return distance <= radii[nearest]


def locally_expanded(points: np.ndarray, multiplier: float, *, neighbours: int | None = None) -> np.ndarray:
    """Dilate each point from its own local neighbourhood, never an origin."""
    points = np.asarray(points, float)
    if len(points) < 2:
        return points.copy()
    k = _neighbour_count(points, neighbours)
    ids = cKDTree(points).query(points, k=k + 1)[1][:, 1:]
    local_centre = points[ids].mean(1)
    return local_centre + float(multiplier) * (points - local_centre)
