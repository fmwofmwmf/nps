"""A disk-like annulus with one radial chunk removed: a U-shaped region."""
from __future__ import annotations

import numpy as np

from manifold_charts.atlas_rom.cases.generic._generic_surface_common import observed_pose3, state_from_observation3, surface_case
from atlas_rom.simulate.sdf_oracle import SDFSystem

_RADIUS, _HALF_WIDTH, _OPENING = 1.5, .42, .62


def _u_sdf(q):
    x, y = np.asarray(q, float)
    angle = float(np.arctan2(y, x))
    # A full annulus with its positive-x arc removed, rounded at both cuts.
    # The nearest point on the retained circular centreline gives a capsule-like
    # U with smooth end caps instead of artificial corner singularities.
    if abs(angle) >= _OPENING:
        nearest_angle = angle
    else:
        nearest_angle = _OPENING if y >= 0.0 else -_OPENING
    nearest = _RADIUS * np.array([np.cos(nearest_angle), np.sin(nearest_angle)])
    return float(np.linalg.norm(np.array([x, y]) - nearest) - _HALF_WIDTH)


SDF_SYSTEM = SDFSystem(lambda q: float(np.asarray(q, float)[2]), seeds=((-1.5, 0.0, 0.0),),
                       dimension=3, observe=observed_pose3, state_from_observation=state_from_observation3,
                       surface=True, surface_accept=lambda q: _u_sdf(np.asarray(q, float)[:2]) <= 0.0,
                       fingerprint="generic-u-region-surface-sdf-v1", step=.25,
                       max_samples=900, max_seconds=900.)
CASE = surface_case("generic_u_region", (-1.5, 0.0, 0.0))
