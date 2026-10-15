"""Analytic-constraint and evaluator-only smoke tests for the open differential."""
from __future__ import annotations

import numpy as np

from atlas_rom.simulate import ideal_oracle
from atlas_rom.cases import load_spec
from atlas_rom.cases.open_bevel_differential import EVALUATOR_REFERENCE, _RATIO


def _rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def _rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


def _differential_pose(oracle, carrier, left):
    """Evaluator helper: parameterize a known physical configuration only."""
    right = 2.0 * carrier - left
    spin = -_RATIO * (left - carrier)                 # spider roll, carrier-relative
    q = oracle.seed().reshape(oracle.n_bodies, 3, 4)
    for name, R in (("carrier", _rot_z(carrier)),
                    ("left_side_gear", _rot_z(left)),
                    ("right_side_gear", _rot_z(right)),
                    ("spider", _rot_z(carrier) @ _rot_x(spin))):
        q[oracle.order.index(name), :, :3] = R
    return q.reshape(-1)


def test_open_bevel_differential_is_prior_free_and_analytic():
    spec = load_spec("open_bevel_differential")
    assert spec.coordinates == []
    assert spec.build.expect.dimension is None
    assert spec.build.canonical.residual_discovery
    assert ideal_oracle.supports(spec)


def test_open_bevel_differential_has_a_regular_two_dimensional_component():
    oracle = ideal_oracle.build_ideal_oracle(load_spec("open_bevel_differential"), log=lambda *_: None)
    # Representatives kept inside a single branch of every wrapped angle, so the
    # closed-form pose lands exactly on the manifold with no re-projection.
    for carrier, left in ((0.0, 0.0), (.35, -.62), (.9, .4), (-.8, .7)):
        q = _differential_pose(oracle, carrier, left)
        assert np.linalg.norm(oracle.residual(q), np.inf) < 1e-9
        _, _, dimension = oracle.tangent_frame(q)
        assert dimension == 2


def test_bevel_mesh_constraint_has_a_consistent_jacobian():
    oracle = ideal_oracle.build_ideal_oracle(load_spec("open_bevel_differential"), log=lambda *_: None)
    q = _differential_pose(oracle, .35, -.62)
    analytic = oracle.jacobian(q)
    step = 1e-6
    numeric = np.empty_like(analytic)
    for column in range(q.size):
        delta = np.zeros_like(q)
        delta[column] = step
        numeric[:, column] = (oracle.residual(q + delta) - oracle.residual(q - delta)) / (2.0 * step)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-5)


def test_open_bevel_differential_reference_is_evaluator_only():
    spec = load_spec("open_bevel_differential")
    assert EVALUATOR_REFERENCE["configuration_space"] == "T^2"
    assert EVALUATOR_REFERENCE["intrinsic_dimension"] == 2
    assert "EVALUATOR_REFERENCE" not in spec.to_dict()
