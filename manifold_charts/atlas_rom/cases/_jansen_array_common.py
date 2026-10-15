"""Black-box, shared-shaft arrays of ideal Jansen linkages.

The only kinematic input is the one-leg joint declaration.  Offset assemblies
are made by local continuation with the generic ideal-joint projector, not by
loading the historical analytic trajectory.  Each leg has its own ten pins;
``phase_lock`` couples crank increments while retaining its declared rest phase.
"""
from __future__ import annotations

import functools
import math

import numpy as np

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions, ChartOptions, CoordinateDef, DecoderOptions, DynamicsOptions, Expectations, JointDef, MechanismSpec, SamplingOptions, Tolerances
from atlas_rom.cases.jansen import CASE as ONE_LEG, COLORS, JOINTS
from atlas_rom.simulate.ideal_oracle import IdealJointOracle

STATION_DX = 0.90
PLANE_Z0 = 0.06
PLANE_DZ = 0.05
TORQUE_HI = 50.0
_Z = np.array([0.0, 0.0, 1.0])


def make_placements(n_legs: int):
    if n_legs % 2:
        raise ValueError(f"n_legs must be even for the mirrored layout, got {n_legs}")
    per_side = n_legs // 2
    return [(side + str(k), (k - 0.5 * (per_side - 1)) * STATION_DX,
             sign * (PLANE_Z0 + PLANE_DZ * k),
             2.0 * math.pi * k / per_side + (math.pi if side == "L" else 0.0),
             side == "L")
            for k in range(per_side) for side, sign in (("R", 1.0), ("L", -1.0))]


def _rotation_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


@functools.lru_cache(maxsize=None)
def _offset_pose(phase):
    """Solve a one-leg rest-to-phase continuation using ideal joints only."""
    oracle = IdealJointOracle(ONE_LEG, log=lambda _message: None)
    q = oracle.seed()
    crank = oracle.order.index("crank")
    pivot = np.asarray(next(j.pivot for j in JOINTS if j.name == "O"), float)
    n_steps = max(1, int(math.ceil(abs(phase) / 0.04)))
    for _ in range(n_steps):
        turn = _rotation_z(phase / n_steps)
        blocks = q.reshape(oracle.n_bodies, 3, 4).copy()
        blocks[crank, :, :3] = turn @ blocks[crank, :, :3]
        blocks[crank, :, 3] = turn @ (blocks[crank, :, 3] - pivot) + pivot
        q = oracle.project(blocks.reshape(-1), hold=("crank",))
    if np.linalg.norm(oracle.residual(q), np.inf) > 1e-9:
        raise RuntimeError(f"Jansen phase seed {phase:.6g} did not close")
    return q.reshape(oracle.n_bodies, 3, 4)


def _placed_transform(base, x, z, mirrored):
    T = np.eye(4)
    T[:3, :4] = base
    if mirrored:
        reflect = np.diag((1.0, 1.0, -1.0, 1.0))
        T = reflect @ T @ reflect
    T[:3, 3] += (x, 0.0, z)
    return T


_ONE_LEG_ORDER = [b.name for b in ONE_LEG.mechanism.bodies]


def _leg(leg, x, z, phase, mirrored):
    poses = _offset_pose(float(phase))
    placed = {name: _placed_transform(poses[i], x, z, mirrored)
              for i, name in enumerate(_ONE_LEG_ORDER)}
    bodies = []
    for body in ONE_LEG.mechanism.bodies:
        red, green, blue = COLORS[body.name]
        shade = 0.62 if mirrored else 1.0
        bodies.append(BodyDef(name=f"{body.name}@{leg}", mesh=body.mesh,
                              color=(red * shade, green * shade, blue * shade),
                              configuration=body.configuration,
                              initial_transform=placed[body.name]))
    # Each ONE_LEG joint pivot is a world point at ONE_LEG's *rest* pose.  This
    # leg is assembled at a nonzero crank phase and then placed/mirrored, so the
    # pivot has to ride body_a through exactly the same transform its mesh does:
    # recover the pin in body_a's local frame from ONE_LEG's rest transform, then
    # re-place it by body_a's phased-and-placed transform.  Translating the rest
    # pivot by (x, 0, z) alone left every phased leg's hinges ~0.5-0.9 off the
    # real pins, so the linkage visibly tore apart away from the rest pose.
    rest = {b.name: np.asarray(b.initial_transform, float)
            for b in ONE_LEG.mechanism.bodies}
    joints = []
    for joint in JOINTS:
        if joint.pivot is None:
            pivot = None
        else:
            local_pin = np.linalg.solve(
                rest[joint.body_a],
                np.append(np.asarray(joint.pivot, float), 1.0))
            pivot = (placed[joint.body_a] @ local_pin)[:3]
        joints.append(JointDef(joint.name + "@" + leg, joint.kind,
                               joint.body_a + "@" + leg,
                               None if joint.body_b is None else joint.body_b + "@" + leg,
                               pivot=pivot, axis=joint.axis,
                               exclude_collision=joint.exclude_collision,
                               limit_lo=joint.limit_lo, limit_hi=joint.limit_hi))
    return bodies, joints


def make_case(name: str, n_legs: int, placements=None) -> AtlasCaseSpec:
    placements = make_placements(n_legs) if placements is None else list(placements)
    if len(placements) != n_legs:
        raise ValueError(f"{name}: expected {n_legs} placements, got {len(placements)}")
    bodies, joints = [], []
    for leg, x, z, phase, mirrored in placements:
        leg_bodies, leg_joints = _leg(leg, x, z, phase, mirrored)
        bodies.extend(leg_bodies)
        joints.extend(leg_joints)
    master = placements[0][0]
    for leg, *_rest in placements[1:]:
        joints.append(JointDef(f"shaft:{master}:{leg}", "phase_lock",
                               f"crank@{master}", f"crank@{leg}", axis=_Z))
    return AtlasCaseSpec(
        name=name,
        mechanism=MechanismSpec(bodies=bodies, joints=joints, up_axis="y", gravity=9.8),
        seeds=[], contact_model="none",
        coordinates=[CoordinateDef(name="phi", topology="periodic", period=2.0 * math.pi,
                                   source="sample_field",
                                   args={"key": f"recorded_angle:crank@{master}:z"})],
        actuators=[ActuatorDef("crank torque", kind="joint_torque",
                               target=f"O@{master}", hi=TORQUE_HI)],
        build=BuildOptions(
            sampling=SamplingOptions(sampler="analytic"), charts=ChartOptions(mode="single"),
            decoder=DecoderOptions(epochs=6000, n_harmonics=6, smooth_lambda=1e-4),
            dynamics=DynamicsOptions(distill=True, epochs=3000, n_teacher=721),
            expect=Expectations(dimension=1, n_charts=1, n_bodies=7 * n_legs),
            tolerances=Tolerances(decoder_translation_rmse=0.005, decoder_rotation_deg=1.0)),
        scenarios={"default": {"dt": 1e-3, "initial_u": [0.0]}},
    )
