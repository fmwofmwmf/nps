"""SDF zero-level sphere with physical pose observations for chart learning."""
from __future__ import annotations

import numpy as np

from atlas_rom.spec.case_spec import AtlasCaseSpec, BodyDef, BuildOptions, CoordinateDef, Expectations, MechanismSpec, MeshSpec, SamplingOptions
from atlas_rom.simulate.sdf_oracle import SDFSystem

_RADIUS = 2.0


def _sphere_sdf(q):
    return float(np.linalg.norm(q) - _RADIUS)


def _observed_pose(q):
    """Physical pose emitted by the surface-response simulator."""
    return np.column_stack((np.eye(3), np.asarray(q, float)))[None]


def _state_from_observation(pose):
    return np.asarray(pose, float).reshape(1, 3, 4)[0, :, 3]


SDF_SYSTEM = SDFSystem(_sphere_sdf, seeds=((0.0, 0.0, _RADIUS),), dimension=3,
                       observe=_observed_pose, surface=True,
                       state_from_observation=_state_from_observation,
                       fingerprint="generic-sphere-surface-sdf-v1", step=.48,
                       max_samples=500, max_seconds=None)

# Read exclusively by ``generic_manifold_diagnostics`` after training.  This
# display parametrization is deliberately not part of CASE or any artifact.
VISUAL_UVS = {"kind": "sphere_equirectangular", "body": "probe"}


CASE = AtlasCaseSpec(
    name="generic_sphere",
    mechanism=MechanismSpec(bodies=[
        BodyDef(name="probe", mesh=MeshSpec("sphere", {"radius": .16, "subdiv": 2}),
                initial_transform=[[1, 0, 0, 0], [0, 1, 0, 0],
                                   [0, 0, 1, _RADIUS], [0, 0, 0, 1]],
                color=(.25, .55, .90)),
    ]),
    # System-observation declarations only; SDF sampling writes no ``u`` and
    # the chart builder derives its rank from measured pose secants.
    coordinates=[CoordinateDef("obs_x", source="body_translation", args={"body": "probe", "axis": "x"}),
                 CoordinateDef("obs_y", source="body_translation", args={"body": "probe", "axis": "y"}),
                 CoordinateDef("obs_z", source="body_translation", args={"body": "probe", "axis": "z"})],
    contact_model="none",
    build=BuildOptions(sampling=SamplingOptions(n_samples=None, step=.24),
                       expect=Expectations(n_bodies=1)),
)
