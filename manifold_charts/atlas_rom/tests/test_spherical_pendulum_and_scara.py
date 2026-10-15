"""Smoke and evaluator-isolation tests for the pendulum and compact SCARA cases."""
from __future__ import annotations

import numpy as np

from atlas_rom.simulate import ideal_oracle
from atlas_rom.cases import load_sdf_system, load_spec
from atlas_rom.cases.continuous_rotation_scara import EVALUATOR_REFERENCE as SCARA_REFERENCE
from atlas_rom.cases.spherical_pendulum import CABLE_LENGTH, EVALUATOR_REFERENCE as PENDULUM_REFERENCE
from atlas_rom.simulate.sdf_oracle import SDFDirectionalOracle


def test_spherical_pendulum_is_particle_native_and_has_rank_two():
    spec = load_spec("spherical_pendulum")
    system = load_sdf_system("spherical_pendulum")
    assert spec.coordinates == []
    assert spec.build.expect.dimension is None
    state = SDFDirectionalOracle(system).project(np.array((0., 0., -CABLE_LENGTH)))
    frame, _singular, rank, _diagnostics = SDFDirectionalOracle(system).tangent_measurement(state, eps=.01)
    assert frame.shape == (3, 2) and rank == 2
    assert abs(np.linalg.norm(state) - CABLE_LENGTH) < 1e-10
    assert PENDULUM_REFERENCE["configuration_space"] == "S^2"


def test_scara_has_three_unlimited_bearings_one_physical_slide_and_rank_four():
    spec = load_spec("continuous_rotation_scara")
    assert spec.coordinates == []
    assert spec.build.expect.dimension is None
    bearings = spec.mechanism.joints[:3]
    slide = spec.mechanism.joints[3]
    assert all(j.kind == "revolute" and not j.limited for j in bearings)
    assert slide.kind == "prismatic" and (slide.limit_lo, slide.limit_hi) == (-.32, .32)
    oracle = ideal_oracle.build_ideal_oracle(spec, log=lambda *_: None)
    q = oracle.seed()
    assert np.linalg.norm(oracle.residual(q), np.inf) < 1e-10
    _, _, rank = oracle.tangent_frame(q)
    assert rank == 4
    assert SCARA_REFERENCE["configuration_space"] == "T^3 x I"
