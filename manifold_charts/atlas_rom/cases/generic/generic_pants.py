"""Pair of pants: a planar disk with two disjoint physical holes."""
from __future__ import annotations

import numpy as np

from manifold_charts.atlas_rom.cases.generic._generic_surface_common import observed_pose3, state_from_observation3, surface_case
from atlas_rom.simulate.sdf_oracle import SDFSystem

_OUTER, _HOLE = 2.5, .62
_HOLES = np.asarray(((-.82, 0.0), (.82, 0.0)))


def _pants_sdf(q):
    q = np.asarray(q, float)
    outer = float(np.linalg.norm(q) - _OUTER)
    holes = max(float(_HOLE - np.linalg.norm(q - center)) for center in _HOLES)
    return max(outer, holes)


SDF_SYSTEM = SDFSystem(lambda q: float(np.asarray(q, float)[2]), seeds=((0.0, 1.4, 0.0),),
                       dimension=3, observe=observed_pose3, state_from_observation=state_from_observation3,
                       surface=True, surface_accept=lambda q: _pants_sdf(np.asarray(q, float)[:2]) <= 0.0,
                       fingerprint="generic-pants-surface-sdf-v1", step=.25,
                       max_samples=1000, max_seconds=900.)
CASE = surface_case("generic_pants", (0.0, 1.4, 0.0))
