"""Closed genus-2 surface: one smooth union of two ordinary tori."""
from __future__ import annotations

from atlas_rom.cases._generic_surface_common import (observed_pose3,
                                                     smooth_min_sdf, torus_sdf,
                                                     state_from_observation3, surface_case)
from atlas_rom.simulate.sdf_oracle import SDFSystem

_CENTERS = ((-.90, 0.0, 0.0), (.90, 0.0, 0.0))
_MAJOR, _TUBE = .85, .42


def _genus2_sdf(q):
    return smooth_min_sdf([torus_sdf(q, center, major_radius=_MAJOR, tube_radius=_TUBE)
                           for center in _CENTERS], blend=.08)


SDF_SYSTEM = SDFSystem(_genus2_sdf, seeds=((-_MAJOR - .90, 0.0, _TUBE),),
                       dimension=3, observe=observed_pose3,
                       state_from_observation=state_from_observation3, surface=True,
                       fingerprint="generic-genus2-surface-sdf-v1", step=.155, tangent_eps=.01,
                       max_samples=None, max_seconds=None)
CASE = surface_case("generic_genus2", (-_MAJOR - .90, 0.0, _TUBE))
