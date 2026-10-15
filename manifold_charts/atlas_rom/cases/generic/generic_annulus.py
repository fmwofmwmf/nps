"""Production annulus test case: a free puck around a fixed central peg.

The feasible configuration set is determined solely by the standard contact
oracle.  This module declares scene data; it supplies neither samples nor a
projection implementation.
"""
from __future__ import annotations

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BoundaryOptions, BuildOptions, ChartOptions, CoordinateDef, Expectations, JointDef, MechanismSpec, MeshSpec, SamplingOptions, Tolerances
from atlas_rom.simulate.sdf_oracle import SDFSystem

_PUCK_RADIUS = 0.3
_PEG_RADIUS = 1.0
_ARENA_HALF = 3.0
_WALL_THICK = 0.3
_Z_HALF = 1.5
_A = _ARENA_HALF + _WALL_THICK
_LONG = _ARENA_HALF + 2.0 * _WALL_THICK


def _annulus_sdf(q):
    """The generic test system: valid exactly when this SDF is non-positive."""
    r = float(__import__("numpy").linalg.norm(q))
    return max(1.3 - r, r - 3.0)


def _observed_pose(q):
    """Simulator observation for the locked carriage/puck scene, not a chart map."""
    import numpy as np
    x, y = np.asarray(q, float)
    carriage = np.column_stack((np.eye(3), [x, 0.0, -0.75]))
    puck = np.column_stack((np.eye(3), [x, y, 0.0]))
    return np.stack((carriage, puck))


def _state_from_observation(pose):
    import numpy as np
    pose = np.asarray(pose, float).reshape(2, 3, 4)
    return pose[1, :, 3][:2]


SDF_SYSTEM = SDFSystem(_annulus_sdf, seeds=((2.0, 0.0),), dimension=2,
                       observe=_observed_pose,
                       state_from_observation=_state_from_observation,
                       fingerprint="generic-annulus-sdf-v1", step=.32,
                       max_samples=1000, max_seconds=None)


def _wall(name, center, half):
    return BodyDef(name=name, mesh=MeshSpec("box", {"center": center, "half": half}),
                   fixed=True, color=(0.60, 0.62, 0.66))


CASE = AtlasCaseSpec(
    name="generic_annulus",
    mechanism=MechanismSpec(
        bodies=[
            # A physical Cartesian stage: the two serial prismatic joints are
            # the only free modes.  This is deliberately not a semantic
            # ``planar_rigid`` label, which does not constrain the affine-body
            # solver by itself.
            BodyDef(name="__stage_ground",
                    mesh=MeshSpec("box", {"center": [0.0, 0.0, -2.0],
                                            "half": [0.2, 0.2, 0.2]}),
                    fixed=True, color=(0.45, 0.47, 0.50)),
            BodyDef(name="x_carriage",
                    mesh=MeshSpec("box", {"center": [0.0, 0.0, -0.75],
                                            "half": [0.25, 0.25, 0.10]}),
                    initial_transform=[[1, 0, 0, 2.0], [0, 1, 0, 0.0],
                                       [0, 0, 1, 0.0], [0, 0, 0, 1]],
                    color=(0.45, 0.55, 0.70)),
            BodyDef(name="puck",
                    mesh=MeshSpec("sphere", {"radius": _PUCK_RADIUS, "subdiv": 2}),
                    initial_transform=[[1, 0, 0, 2.0], [0, 1, 0, 0.0],
                                       [0, 0, 1, 0.0], [0, 0, 0, 1]],
                    color=(0.90, 0.35, 0.20)),
            BodyDef(name="__peg",
                    mesh=MeshSpec("sphere", {"radius": _PEG_RADIUS, "subdiv": 2}),
                    fixed=True, color=(0.55, 0.30, 0.30)),
            _wall("__wall_px", [_A, 0.0, 0.0], [_WALL_THICK, _LONG, _Z_HALF]),
            _wall("__wall_nx", [-_A, 0.0, 0.0], [_WALL_THICK, _LONG, _Z_HALF]),
            _wall("__wall_py", [0.0, _A, 0.0], [_LONG, _WALL_THICK, _Z_HALF]),
            _wall("__wall_ny", [0.0, -_A, 0.0], [_LONG, _WALL_THICK, _Z_HALF]),
        ],
        joints=[
            JointDef("stage_x", "prismatic", "__stage_ground", "x_carriage",
                     pivot=[0.0, 0.0, 0.0], axis=[1.0, 0.0, 0.0]),
            JointDef("stage_y", "prismatic", "x_carriage", "puck",
                     pivot=[2.0, 0.0, 0.0], axis=[0.0, 1.0, 0.0]),
        ],
        contact_pairs=[("puck", "__peg"), ("puck", "__wall_px"),
                       ("puck", "__wall_nx"), ("puck", "__wall_py"),
                       ("puck", "__wall_ny")],
        up_axis="y",
        render_up_axis="z",
        gravity=9.8,
    ),
    seeds=[],
    contact_model="exact_clearance",
    coordinates=[
        CoordinateDef(name="x", topology="euclidean", source="body_translation",
                      args={"body": "puck", "axis": "x"}),
        CoordinateDef(name="y", topology="euclidean", source="body_translation",
                      args={"body": "puck", "axis": "y"}),
    ],
    actuators=[
        ActuatorDef(name="push x", kind="coordinate_force", target="x", hi=200.0),
        ActuatorDef(name="push y", kind="coordinate_force", target="y", hi=200.0),
    ],
    build=BuildOptions(
        # No reuse: sampling.prepare invokes the standard live contact explorer.
        sampling=SamplingOptions(n_samples=None, step=0.08, max_seconds=1800.0),
        charts=ChartOptions(mode="single"),
        boundary=BoundaryOptions(kind="sdf", k=1e8),
        expect=Expectations(dimension=2, n_charts=1, n_bodies=2),
        tolerances=Tolerances(decoder_translation_rmse=0.05,
                              decoder_rotation_deg=1.0,
                              free_energy_drift=3e-1),
    ),
    scenarios={"default": {"dt": 2e-3, "initial_u": [0.0, 2.0]}},
)
