"""SDF zero-level 3-sphere S^3 in R^4 with physical pose observations.

The first intrinsic-dimension-3 case in the generic family, and the easiest
one: S^3 is parallelizable, so the tangent bundle is trivial and the analytic
tangent is exact everywhere, and its Lusternik--Schnirelmann category is 2 --
the same chart count as S^2.  A chart count that climbs with ``d`` here is a
dimension artifact, not topology.

R^4 does not fit in one rigid-body translation, so the observation carries the
state in two probe bodies: ``probe`` holds (q0, q1, q2) and ``probe_w`` holds
q3 in its x slot.  The map is a coordinate permutation, hence injective, and
the chart builder still measures its rank from pose secants alone.
"""
from __future__ import annotations

import numpy as np

from atlas_rom.spec.case_spec import AtlasCaseSpec, BodyDef, BuildOptions, CoordinateDef, Expectations, MechanismSpec, MeshSpec, SamplingOptions
from atlas_rom.simulate.sdf_oracle import SDFSystem

_RADIUS = 2.0

# Evaluation ground truth.  Deliberately not part of CASE, SDF_SYSTEM, or any
# artifact: the fitter must derive both numbers from measured evidence.
GROUND_TRUTH = {"topology": "S^3", "intrinsic_dimension": 3,
                "min_charts_ls_category": 2}


def _sphere3_sdf(q):
    return float(np.linalg.norm(q) - _RADIUS)


def _observed_pose(q):
    """Physical pose emitted by the surface-response simulator."""
    q = np.asarray(q, float)
    return np.stack((np.column_stack((np.eye(3), q[:3])),
                     np.column_stack((np.eye(3), np.array([q[3], 0.0, 0.0])))))


def _state_from_observation(pose):
    T = np.asarray(pose, float).reshape(2, 3, 4)
    return np.array([T[0, 0, 3], T[0, 1, 3], T[0, 2, 3], T[1, 0, 3]])


SDF_SYSTEM = SDFSystem(_sphere3_sdf, seeds=((_RADIUS, 0.0, 0.0, 0.0),), dimension=4,
                       observe=_observed_pose, surface=True,
                       state_from_observation=_state_from_observation,
                       fingerprint="generic-sphere3-surface-sdf-v1", step=.55,
                       max_samples=1200, max_seconds=1800.)


CASE = AtlasCaseSpec(
    name="generic_sphere3",
    mechanism=MechanismSpec(bodies=[
        BodyDef(name="probe", mesh=MeshSpec("sphere", {"radius": .14, "subdiv": 2}),
                initial_transform=[[1, 0, 0, _RADIUS], [0, 1, 0, 0],
                                   [0, 0, 1, 0], [0, 0, 0, 1]],
                color=(.25, .55, .90)),
        BodyDef(name="probe_w", mesh=MeshSpec("sphere", {"radius": .14, "subdiv": 2}),
                initial_transform=[[1, 0, 0, 0], [0, 1, 0, 0],
                                   [0, 0, 1, 0], [0, 0, 0, 1]],
                color=(.90, .55, .25)),
    ]),
    coordinates=[CoordinateDef("obs_x", source="body_translation", args={"body": "probe", "axis": "x"}),
                 CoordinateDef("obs_y", source="body_translation", args={"body": "probe", "axis": "y"}),
                 CoordinateDef("obs_z", source="body_translation", args={"body": "probe", "axis": "z"}),
                 CoordinateDef("obs_w", source="body_translation", args={"body": "probe_w", "axis": "x"})],
    contact_model="none",
    build=BuildOptions(sampling=SamplingOptions(n_samples=None, step=.28),
                       expect=Expectations(n_bodies=2)),
)
