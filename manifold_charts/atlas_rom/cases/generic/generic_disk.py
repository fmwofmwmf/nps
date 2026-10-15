"""Planar disk: the contractible two-dimensional control."""
from __future__ import annotations

import numpy as np

from manifold_charts.atlas_rom.cases.generic._generic_surface_common import observed_pose3, state_from_observation3, surface_case
from atlas_rom.simulate.sdf_oracle import SDFSystem

_RADIUS = 2.0


def _accepted_surface(q):
    return float(np.linalg.norm(np.asarray(q, float)[:2])) <= _RADIUS


# An open piece of a smooth plane.  Treating the disk as a 2-D volume makes
# boundary finite differences lose a rank; this surface-plus-physical-mask form
# preserves the two tangent directions and rejects only out-of-disk frames.
SDF_SYSTEM = SDFSystem(lambda q: float(np.asarray(q, float)[2]), seeds=((0.0, 0.0, 0.0),),
                       dimension=3, observe=observed_pose3, state_from_observation=state_from_observation3,
                       surface=True, surface_accept=_accepted_surface,
                       fingerprint="generic-disk-surface-sdf-v1", step=.28,
                       max_samples=700, max_seconds=900.)
CASE = surface_case("generic_disk", (0.0, 0.0, 0.0))
