"""SDF zero-level 3-torus T^3 in R^4 with physical pose observations.

The top rung of the generic ladder: intrinsic dimension 3 and Lusternik--
Schnirelmann category 4, the largest minimum chart count available here.  A
method that quietly builds a near-global chart has to fold in three
independent directions to cover it.

The SDF is the iterated tube -- wrap a tube of radius ``_R2`` around the circle
of radius ``_R1`` to get T^2 in the w = 0 hyperplane, then wrap a tube of
radius ``_R3`` around that in R^4.  Each step is an exact distance function, and
``_R1 > _R2 > _R3`` keeps every tube embedded (``_R3`` stays below the reach of
the T^2 core, ``min(_R2, _R1 - _R2)``).

Observation packing follows ``generic_sphere3``: ``probe`` carries (q0, q1, q2)
and ``probe_w`` carries q3 in its x slot.

Sampling budget warning: the iterated tube forces a scale hierarchy -- ``_R3``
must stay under the reach of the T^2 core, so the smallest feature is several
times smaller than the largest.  The advancing front needs a step below roughly
half ``_R3`` to stay local, which puts a fully covered T^3 at order 3e4 samples.
That is past what the current O(n^2) chart fitter can consume; this case is
correct and ready, but it is gated on the sampling/fitter scale work.
"""
from __future__ import annotations

import numpy as np

from atlas_rom.spec.case_spec import AtlasCaseSpec, BodyDef, BuildOptions, CoordinateDef, Expectations, MechanismSpec, MeshSpec, SamplingOptions
from atlas_rom.simulate.sdf_oracle import SDFSystem

_R1 = 2.0
_R2 = 0.9
_R3 = 0.35
_SEED_X = _R1 + _R2 + _R3

# Evaluation ground truth.  Deliberately not part of CASE, SDF_SYSTEM, or any
# artifact: the fitter must derive both numbers from measured evidence.
GROUND_TRUTH = {"topology": "T^3", "intrinsic_dimension": 3,
                "min_charts_ls_category": 4}


def _torus3_sdf(q):
    q = np.asarray(q, float)
    ring = np.hypot(q[0], q[1]) - _R1
    tube = np.hypot(ring, q[2]) - _R2
    return float(np.hypot(tube, q[3]) - _R3)


def _observed_pose(q):
    """Physical pose emitted by the surface-response simulator."""
    q = np.asarray(q, float)
    return np.stack((np.column_stack((np.eye(3), q[:3])),
                     np.column_stack((np.eye(3), np.array([q[3], 0.0, 0.0])))))


def _state_from_observation(pose):
    T = np.asarray(pose, float).reshape(2, 3, 4)
    return np.array([T[0, 0, 3], T[0, 1, 3], T[0, 2, 3], T[1, 0, 3]])


SDF_SYSTEM = SDFSystem(_torus3_sdf, seeds=((_SEED_X, 0.0, 0.0, 0.0),), dimension=4,
                       observe=_observed_pose, surface=True,
                       state_from_observation=_state_from_observation,
                       fingerprint="generic-torus3-surface-sdf-v1", step=.16,
                       max_samples=4000, max_seconds=1800.)


CASE = AtlasCaseSpec(
    name="generic_torus3",
    mechanism=MechanismSpec(bodies=[
        BodyDef(name="probe", mesh=MeshSpec("sphere", {"radius": .12, "subdiv": 2}),
                initial_transform=[[1, 0, 0, _SEED_X], [0, 1, 0, 0],
                                   [0, 0, 1, 0], [0, 0, 0, 1]],
                color=(.25, .55, .90)),
        BodyDef(name="probe_w", mesh=MeshSpec("sphere", {"radius": .12, "subdiv": 2}),
                initial_transform=[[1, 0, 0, 0], [0, 1, 0, 0],
                                   [0, 0, 1, 0], [0, 0, 0, 1]],
                color=(.90, .55, .25)),
    ]),
    coordinates=[CoordinateDef("obs_x", source="body_translation", args={"body": "probe", "axis": "x"}),
                 CoordinateDef("obs_y", source="body_translation", args={"body": "probe", "axis": "y"}),
                 CoordinateDef("obs_z", source="body_translation", args={"body": "probe", "axis": "z"}),
                 CoordinateDef("obs_w", source="body_translation", args={"body": "probe_w", "axis": "x"})],
    contact_model="none",
    build=BuildOptions(sampling=SamplingOptions(n_samples=None, step=.24),
                       expect=Expectations(n_bodies=2)),
)
