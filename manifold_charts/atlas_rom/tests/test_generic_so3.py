"""generic_so3: the ideal-joint SO(3) = RP^3 fixture is a clean 3-manifold."""
from __future__ import annotations

import numpy as np

from atlas_rom.simulate import ideal_oracle
from atlas_rom.cases import load_spec
from manifold_charts.atlas_rom.cases.generic.generic_so3 import VIEW_EMBED, _rotvec


def _rot(axis, theta):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def test_case_routes_to_the_analytic_oracle():
    spec = load_spec("generic_so3")
    assert spec.name == "generic_so3"
    assert spec.contact_model == "none"
    assert spec.build.sampling.sampler == "analytic"
    assert ideal_oracle.supports(spec)


def test_manifold_is_rank_three_and_exactly_rotations_everywhere():
    oracle = ideal_oracle.build_ideal_oracle(load_spec("generic_so3"))
    for theta in (0.0, 0.5, 1.5, 2.8):
        transform = np.zeros((3, 4))
        transform[:, :3] = _rot((1.0, 2.0, 3.0), theta)
        projected = oracle.project(transform.ravel())
        assert np.linalg.norm(oracle.residual(projected)) < 1e-9
        rotation = projected.reshape(3, 4)[:, :3]
        assert np.linalg.norm(rotation.T @ rotation - np.eye(3)) < 1e-9
        assert oracle.tangent_frame(projected)[2] == 3


def test_view_embed_is_the_axis_angle_ball_of_radius_pi():
    identity = np.eye(3, 4).ravel()
    assert np.allclose(VIEW_EMBED(identity[None]), 0.0)

    quarter_turn = np.zeros((3, 4))
    quarter_turn[:, :3] = _rot((0, 0, 1), np.pi / 2)
    assert np.allclose(VIEW_EMBED(quarter_turn.ravel()[None]), (0, 0, np.pi / 2), atol=1e-9)

    half_turn = np.zeros((3, 4))
    half_turn[:, :3] = np.diag((1.0, -1.0, -1.0))  # pi about x
    assert abs(np.linalg.norm(VIEW_EMBED(half_turn.ravel()[None])) - np.pi) < 1e-6


def test_rotvec_handles_the_theta_equals_pi_seam():
    for axis in ((1, 0, 0), (0, 1, 0), (1, 1, 1), (1, -2, 3)):
        v = _rotvec(_rot(axis, np.pi))
        assert abs(np.linalg.norm(v) - np.pi) < 1e-6
        unit = np.asarray(axis, float) / np.linalg.norm(axis)
        assert np.allclose(np.abs(v / np.pi), np.abs(unit), atol=1e-6)  # sign is free on RP^3
