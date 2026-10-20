"""Evaluator-only topology smoke tests for the colliding twin pendulums."""
from __future__ import annotations

import numpy as np

from atlas_rom.cases import load_spec
from atlas_rom.cases.twin_pendulum_contact import (
    BALL_RADIUS, EVALUATOR_REFERENCE, ball_positions, characterize_collision_complement,
)


def test_case_has_full_rotation_pivots_and_explicit_physical_contact():
    spec = load_spec("twin_pendulum_contact")
    assert spec.coordinates == []
    assert spec.build.expect.dimension is None
    assert spec.contact_model == "exact_clearance"
    assert spec.mechanism.contact_pairs == [("pendulum_1", "pendulum_2")]
    assert all(j.kind == "revolute" and not j.limited for j in spec.mechanism.joints)


def test_hanging_pose_is_clear_and_facing_pose_collides():
    a, b = ball_positions(0.0, 0.0)
    assert np.linalg.norm(a - b) > 2.0 * BALL_RADIUS
    # Both pendulums horizontal and pointing at each other: ball centres meet.
    a, b = ball_positions(np.pi / 2, -np.pi / 2)
    assert np.linalg.norm(a - b) < 2.0 * BALL_RADIUS


def test_dense_periodic_evaluator_characterizes_collision_complement():
    report = characterize_collision_complement(resolution=240)
    assert len(report["collision_component_sizes"]) >= 1
    assert len(report["admissible_component_sizes"]) == 1
    assert report["collision_component_sizes"][0] > 100
    assert report["admissible_component_sizes"][0] > report["collision_component_sizes"][0]
    assert EVALUATOR_REFERENCE["unconstrained_joint_space"] == "T^2"
