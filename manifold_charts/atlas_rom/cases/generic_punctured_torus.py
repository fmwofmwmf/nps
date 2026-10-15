"""A torus with one physical surface disk removed, producing a boundary circle."""
from __future__ import annotations

import numpy as np

from atlas_rom.cases._generic_surface_common import observed_pose3, state_from_observation3, surface_case
from atlas_rom.simulate.sdf_oracle import SDFSystem

_RING, _TUBE = 2.0, .8
_PUNCTURE = np.asarray((_RING + _TUBE, 0.0, 0.0))
_PUNCTURE_RADIUS = .54


def _torus_sdf(q):
    q = np.asarray(q, float)
    return float(np.hypot(np.hypot(q[0], q[1]) - _RING, q[2]) - _TUBE)


def _accepted_surface(q):
    return float(np.linalg.norm(np.asarray(q, float) - _PUNCTURE)) >= _PUNCTURE_RADIUS


SDF_SYSTEM = SDFSystem(_torus_sdf, seeds=((-_RING - _TUBE, 0.0, 0.0),), dimension=3,
                       observe=observed_pose3, state_from_observation=state_from_observation3,
                       surface=True, surface_accept=_accepted_surface,
                       fingerprint="generic-punctured-torus-sdf-v1", step=.26,
                       max_samples=1100, max_seconds=900.)
CASE = surface_case("generic_punctured_torus", (-_RING - _TUBE, 0.0, 0.0))
