"""Evaluator-only topology smoke tests for the obstacle-constrained 2R arm."""
from __future__ import annotations

import numpy as np

from atlas_rom.cases import load_spec
from atlas_rom.cases.obstacle_constrained_2r import (
    EVALUATOR_REFERENCE, OBSTACLE_CENTER, OBSTACLE_RADIUS, TOOL_RADIUS,
    characterize_collision_complement, tool_position,
)


def test_case_has_full_rotation_bearings_and_explicit_physical_contact():
    spec = load_spec("obstacle_constrained_2r")
    assert spec.coordinates == []
    assert spec.build.expect.dimension is None
    assert spec.contact_model == "exact_clearance"
    assert spec.mechanism.contact_pairs == [("link_2", "obstacle")]
    assert all(j.kind == "revolute" and not j.limited for j in spec.mechanism.joints)


def test_obstacle_is_reachable_and_the_straight_pose_is_collision_free():
    assert np.linalg.norm(tool_position(0.0, 0.0) - OBSTACLE_CENTER) > TOOL_RADIUS + OBSTACLE_RADIUS
    # A numerical inverse-kinematic witness places the physical tool inside the
    # obstacle clearance disk; it belongs to evaluator evidence only.
    assert np.linalg.norm(tool_position(-.468, 1.447) - OBSTACLE_CENTER) < TOOL_RADIUS + OBSTACLE_RADIUS


def test_dense_periodic_evaluator_characterizes_collision_complement():
    report = characterize_collision_complement(resolution=240)
    # Do not assume a once-punctured torus: this checks the geometry's observed
    # collision regions and connected admissible component before topology is frozen.
    assert len(report["collision_component_sizes"]) >= 1
    assert len(report["admissible_component_sizes"]) == 1
    assert report["collision_component_sizes"][0] > 100
    assert report["admissible_component_sizes"][0] > report["collision_component_sizes"][0]
    assert EVALUATOR_REFERENCE["unconstrained_joint_space"] == "T^2"
