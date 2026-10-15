"""Particle-native spherical pendulum with a fixed-length physical cable.

The simulator state is the bob position, constrained by the cable's fixed
length.  It intentionally has no orientation block: spinning a spherical bob
around the cable is not a configuration degree of freedom and is not fabricated
as one for representation convenience.
"""
from __future__ import annotations

import numpy as np

from atlas_rom.spec.case_spec import AtlasCaseSpec, BodyDef, BuildOptions, CanonicalOptions, Expectations, MechanismSpec, MeshSpec, SamplingOptions
from atlas_rom.simulate.sdf_oracle import SDFSystem


CABLE_LENGTH = 1.35


def _cable_constraint(position):
    """Signed fixed-length constraint measured from the physical pivot."""
    return float(np.linalg.norm(np.asarray(position, float)) - CABLE_LENGTH)


def _observe_particle(position):
    """Emit a translation-only physical pose with an identity visual frame."""
    return np.column_stack((np.eye(3), np.asarray(position, float)))[None]


def _particle_position(pose):
    return np.asarray(pose, float).reshape(1, 3, 4)[0, :, 3]


SDF_SYSTEM = SDFSystem(
    _cable_constraint, seeds=((0.0, 0.0, -CABLE_LENGTH),), dimension=3,
    observe=_observe_particle, state_from_observation=_particle_position,
    surface=True, fingerprint="spherical-pendulum-fixed-cable-v1", step=.22,
    tangent_eps=.01, max_samples=1400, max_seconds=600.,
)

# The learner never receives this assertion; it is checked independently from
# dense physical position samples by the evaluator/test harness.
EVALUATOR_REFERENCE = {"configuration_space": "S^2", "intrinsic_dimension": 2,
                       "representation": "particle position on fixed cable"}


CASE = AtlasCaseSpec(
    name="spherical_pendulum",
    mechanism=MechanismSpec(bodies=[
        BodyDef("bob", MeshSpec("sphere", {"radius": .12, "subdiv": 2}),
                initial_transform=[[1, 0, 0, 0], [0, 1, 0, 0],
                                   [0, 0, 1, -CABLE_LENGTH], [0, 0, 0, 1]],
                color=(.25, .55, .90)),
        BodyDef("__pivot", MeshSpec("sphere", {"radius": .05, "subdiv": 1}), fixed=True,
                color=(.42, .44, .48)),
    ], up_axis="z", gravity=9.8),
    contact_model="none",
    coordinates=[],
    build=BuildOptions(
        sampling=SamplingOptions(sampler="analytic", oracle_backend="sdf", n_samples=None,
                                 step=.22, tangent_eps=.025, rank_gap_ratio=50., max_seconds=600.),
        canonical=CanonicalOptions(enabled=True, residual_discovery=True,
                                   autoencoder_charts=True),
        expect=Expectations(n_bodies=1),
    ),
)
